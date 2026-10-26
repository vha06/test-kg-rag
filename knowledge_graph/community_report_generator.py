from __future__ import annotations

import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from core.schema import (
    CanonicalEntity,
    CanonicalRelationship,
    CanonicalCommunity,
    CanonicalCommunityReport,
    CanonicalChunk,
)
from core.json_repair import robust_json_loads

logger = logging.getLogger(__name__)


class CommunityReportGenerator:
    """
    Sinh Báo cáo Cộng đồng (Community Report Generator) theo phong cách Microsoft GraphRAG:
    - Xây dựng ngữ cảnh phân tích từ các thực thể và mối quan hệ thành viên trong cụm.
    - Sinh tiêu đề, tóm tắt điều hành, báo cáo phân tích chi tiết và các phát hiện chính (Key Findings).
    - Bộ kiểm chứng tham chiếu nghiêm ngặt (Strict Reference Validator): loại bỏ các ID thực thể
      hoặc quan hệ bịa đặt (hallucinated) không thuộc cộng đồng.
    - Kế thừa chuỗi nguồn gốc provenance source_chunk_ids và source_document_ids.
    """

    PROMPT_VERSION = "2.0.0"

    SYSTEM_PROMPT = """Bạn là chuyên gia phân tích Đồ thị tri thức (GraphRAG Community Analyst).
Nhiệm vụ của bạn là đọc các thực thể (Entities) và mối quan hệ (Relationships) của một cụm cộng đồng và viết Báo cáo Phân tích Cộng đồng (Community Report).

Yêu cầu định dạng JSON chính xác:
{
  "title": "Tiêu đề mang tính tổng hợp cao về chủ đề chính của cộng đồng",
  "summary": "Tóm tắt điều hành từ 2-4 câu tổng quan các phát hiện chính",
  "full_content": "Báo cáo phân tích chi tiết cấu trúc, các thực thể chính, luồng tương tác và ý nghĩa nghiệp vụ",
  "key_findings": [
    "Phát hiện quan trọng 1",
    "Phát hiện quan trọng 2",
    "Phát hiện quan trọng 3"
  ],
  "supporting_entity_ids": ["ID thực thể có trong danh sách thành viên"],
  "supporting_relationship_ids": ["ID quan hệ có trong danh sách thành viên"],
  "rating": 8.5
}

QUY TẮC BẮT BUỘC:
1. Chỉ tham chiếu đúng các Entity ID và Relationship ID được cung cấp trong ngữ cảnh. Tuyệt đối không bịa đặt ID mới.
2. Trả về duy nhất khối JSON thuần, không thêm văn bản bên ngoài."""

    def __init__(self, llm_config: Optional[Any] = None):
        self.llm_config = llm_config

    def generate_report(
        self,
        community: CanonicalCommunity,
        entities_map: Dict[str, CanonicalEntity],
        relationships_map: Dict[str, CanonicalRelationship],
        chunks_map: Optional[Dict[str, CanonicalChunk]] = None,
    ) -> CanonicalCommunityReport:
        comm_entities = [entities_map[eid] for eid in community.entity_ids if eid in entities_map]
        comm_relationships = [relationships_map[rid] for rid in community.relationship_ids if rid in relationships_map]

        is_mock = (
            self.llm_config is None
            or getattr(self.llm_config, "provider", "mock") == "mock"
            or not getattr(self.llm_config, "api_key", None) and getattr(self.llm_config, "provider", "") == "gemini"
        )

        if is_mock or not comm_entities:
            return self._generate_deterministic_report(community, comm_entities, comm_relationships)

        from generation.llm_client import MultiProviderLLMClient

        client = MultiProviderLLMClient(self.llm_config)
        model_name = getattr(self.llm_config, "model_name", "unknown_model")

        # Xây dựng ngữ cảnh đầu vào cho LLM
        lines = [f"=== CỘNG ĐỒNG CẤP ĐỘ {community.level}: {community.title} ==="]
        lines.append("\n[DANH SÁCH THỰC THỂ THÀNH VIÊN]:")
        for ent in comm_entities[:25]:
            lines.append(f"- ID: {ent.entity_id} | Tên: {ent.canonical_name} ({ent.entity_type}) | Mô tả: {ent.description or ''}")

        lines.append("\n[DANH SÁCH MỐI QUAN HỆ NỘI BỘ]:")
        for rel in comm_relationships[:35]:
            s_name = entities_map.get(rel.subject_entity_id, ent).canonical_name
            t_name = entities_map.get(rel.object_entity_id, ent).canonical_name
            lines.append(f"- ID: {rel.relationship_id} | {s_name} --[{rel.predicate}]--> {t_name} | Trọng số: {rel.weight} | Mô tả: {rel.description}")

        user_prompt = "\n".join(lines)

        try:
            resp = client.generate(system_prompt=self.SYSTEM_PROMPT, user_prompt=user_prompt)
            data = robust_json_loads(resp)
        except Exception as ex:
            logger.warning(f"Community report LLM generation failed ({ex}), falling back to deterministic report.")
            return self._generate_deterministic_report(community, comm_entities, comm_relationships)

        # 4. Strict Reference Validation: Loại bỏ ID bịa đặt
        valid_ent_set = set(community.entity_ids)
        valid_rel_set = set(community.relationship_ids)

        raw_supp_ents = data.get("supporting_entity_ids", [])
        validated_supp_ents = [eid for eid in raw_supp_ents if eid in valid_ent_set]
        if not validated_supp_ents:
            validated_supp_ents = community.entity_ids[:10]

        raw_supp_rels = data.get("supporting_relationship_ids", [])
        validated_supp_rels = [rid for rid in raw_supp_rels if rid in valid_rel_set]
        if not validated_supp_rels:
            validated_supp_rels = community.relationship_ids[:10]

        # Ánh xạ provenance
        chunk_ids: List[str] = list(community.source_chunk_ids)
        doc_ids: List[str] = list(community.source_document_ids)

        rep_id = f"REP_{community.community_id}"
        return CanonicalCommunityReport(
            report_id=rep_id,
            community_id=community.community_id,
            namespace=community.namespace,
            level=community.level,
            title=data.get("title") or community.title,
            summary=data.get("summary") or f"Tóm tắt phân tích cho cụm {community.title}",
            full_content=data.get("full_content") or data.get("summary", ""),
            key_findings=data.get("key_findings", []),
            supporting_entity_ids=validated_supp_ents,
            supporting_relationship_ids=validated_supp_rels,
            source_chunk_ids=chunk_ids,
            source_document_ids=doc_ids,
            rating=float(data.get("rating", 7.5)),
            generation_model=model_name,
            prompt_version=self.PROMPT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_deterministic_report(
        self,
        community: CanonicalCommunity,
        entities: List[CanonicalEntity],
        relationships: List[CanonicalRelationship],
    ) -> CanonicalCommunityReport:
        ent_names = [e.canonical_name for e in entities]
        top_names = ent_names[:5]

        findings: List[str] = []
        for r in relationships[:5]:
            findings.append(f"Mối quan hệ {r.predicate}: {r.description or 'có liên kết mật thiết trong dữ liệu'}")
        if not findings and top_names:
            findings.append(f"Cụm tập trung vào các thực thể trọng tâm: {', '.join(top_names)}")

        summary = (
            f"Cộng đồng cấp độ {community.level} ({community.title}) bao gồm {len(entities)} thực thể "
            f"và {len(relationships)} liên kết tương tác. Các thực thể nòng cốt bao gồm: {', '.join(top_names)}."
        )

        content_sections = [
            f"# {community.title} - Báo Cáo Phân Tích",
            f"## Tóm Tắt Tổng Quan\n{summary}",
            "## Cấu Trúc Thực Thể Trọng Tâm\n" + "\n".join([f"- **{e.canonical_name}** ({e.entity_type}): {e.description or 'Không có mô tả chi tiết'}" for e in entities[:10]]),
            "## Các Mối Quan Hệ Chính\n" + "\n".join([f"- {r.description or (r.subject_entity_id + ' -> ' + r.predicate + ' -> ' + r.object_entity_id)}" for r in relationships[:10]]),
            "## Phát Hiện Chính (Key Findings)\n" + "\n".join([f"- {f}" for f in findings]),
        ]
        full_content = "\n\n".join(content_sections)

        rep_id = f"REP_{community.community_id}"
        return CanonicalCommunityReport(
            report_id=rep_id,
            community_id=community.community_id,
            namespace=community.namespace,
            level=community.level,
            title=community.title,
            summary=summary,
            full_content=full_content,
            key_findings=findings,
            supporting_entity_ids=community.entity_ids[:10],
            supporting_relationship_ids=community.relationship_ids[:10],
            source_chunk_ids=list(community.source_chunk_ids),
            source_document_ids=list(community.source_document_ids),
            rating=8.0,
            generation_model="deterministic_synthesizer",
            prompt_version=self.PROMPT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
