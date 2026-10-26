import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from core.schema import (
    CanonicalDocument,
    ContentBlock,
    CanonicalChunk,
    EvidenceRecord,
    CanonicalEntity,
    EntityMention,
    CanonicalRelationship,
    SourceLocation,
)
from core.registry import ProcessingProfile
from core.json_repair import robust_json_loads
from pipeline.interfaces import IMetadataExtractor, IKnowledgeExtractor

logger = logging.getLogger(__name__)


class BasicStructuralMetadataExtractor(IMetadataExtractor):
    def extract_metadata(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        headings = [b.text for b in blocks if b.block_type == "HEADING"]
        total_chars = sum(len(b.text) for b in blocks)
        return {
            "total_blocks": len(blocks),
            "heading_count": len(headings),
            "headings_outline": headings[:10],
            "total_characters": total_chars,
            "has_parent_hierarchy": any(b.parent_block_id is not None for b in blocks),
        }


class RuleBasedKnowledgeExtractor(IKnowledgeExtractor):
    """
    Extractor hoạt động hoàn toàn Local / Rule-based / Regex.
    Hỗ trợ từ điển mẫu tiêu chuẩn và nhận diện các thực thể/quan hệ
    nghiệp vụ, kỹ thuật tiếng Việt và thuật ngữ hệ thống.
    Tự động gắn cờ origin_type='EXTRACTOR_GENERATED' và tạo EvidenceRecord có toạ độ thực.
    """

    DEFAULT_ENTITY_PATTERNS = {
        "SERVICE": r"\b(Gateway|AuthServer|Redis|Database|BillingService|SearchEngine|WorkerPool)\b",
        "SERVER": r"\b(Server_[A-Z0-9]+|Host_[A-Z0-9]+|Node_\d+)\b",
        "PROTOCOL": r"\b(HTTP|HTTPS|gRPC|TCP|UDP|REST|WebSocket)\b",
        "INCIDENT": r"\b(INCIDENT_\d+|Sự cố\s+[A-Z0-9_]+)\b",
    }

    BROADER_ENTITY_PATTERNS = {
        "SERVICE": r"\b(?:Dịch vụ|Cơ sở dữ liệu|Hệ thống|Cụm máy chủ|Máy chủ|Server|Gateway|Redis|Postgres|Cache|Backend|Frontend|API|Worker)\s+[A-ZÀ-Ỹa-zA-Z0-9_\-]+(?:\s+[A-ZÀ-Ỹa-zA-Z0-9_\-]+)?",
        "COMPONENT": r"\b[A-ZÀ-Ỹ][a-zA-Z0-9_\-]+(?:\s+[A-ZÀ-Ỹ][a-zA-Z0-9_\-]+)+\b",
        "PROTOCOL": r"\b(?:Giao thức\s+[A-Z0-9_\-]+|FastLease|SparseAttention|MultiHeadAttention|Raft|Paxos)\b",
        "INCIDENT": r"\b(?:Sự cố\s+[A-Z0-9_\-]+|INCIDENT_\d+|INC-\d+-\d+)\b",
    }

    DEFAULT_RELATION_PATTERNS = [
        # (subject_type, regex_trigger, predicate, object_type)
        ("SERVICE", r"(?:phụ thuộc|yêu cầu|kết nối tới|gọi tới|depends on|calls)\s+(?:dịch vụ\s+)?(\w+)", "DEPENDS_ON", "SERVICE"),
        ("SERVICE", r"(?:lưu trữ tại|ghi dữ liệu vào|stores in)\s+(?:cơ sở dữ liệu\s+)?(\w+)", "STORES_DATA_IN", "SERVICE"),
        ("INCIDENT", r"(?:bắt nguồn từ|gây ra bởi|caused by)\s+(?:dịch vụ\s+)?(\w+)", "TRIGGERED_BY", "SERVICE"),
    ]

    BROADER_RELATION_CONNECTORS = [
        ("DEPENDS_ON", re.compile(r"(?:phụ thuộc|yêu cầu|kết nối tới|kết nối đến|chuyển tiếp.*?đến|chuyển tiếp.*?tới|gọi tới|depends on|calls|định tuyến.*?đến)", re.IGNORECASE)),
        ("STORES_DATA_IN", re.compile(r"(?:lưu trữ.*?tại|lưu trữ.*?trong|ghi dữ liệu vào|lưu phiên làm việc trong|stores in)", re.IGNORECASE)),
        ("TRIGGERED_BY", re.compile(r"(?:bắt nguồn từ|gây ra bởi|caused by|khi.*?bị đầy|khi.*?gặp sự cố|phản hồi mã lỗi)", re.IGNORECASE)),
    ]

    def extract_knowledge(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        chunks: List[CanonicalChunk],
        profile: ProcessingProfile,
    ) -> Tuple[
        List[CanonicalEntity],
        List[EntityMention],
        List[CanonicalRelationship],
        List[EvidenceRecord],
    ]:
        entities_map: Dict[str, CanonicalEntity] = {}
        mentions: List[EntityMention] = []
        relationships: List[CanonicalRelationship] = []
        evidences: List[EvidenceRecord] = []

        # Map block_id to chunk_id
        block_to_chunk: Dict[str, str] = {}
        for ch in chunks:
            for b_id in ch.source_block_ids:
                block_to_chunk[b_id] = ch.chunk_id

        domain_id = doc.domain_id
        graph_ns = f"ns_{domain_id.replace('synth_domain_', '')}" if "synth_domain_" in domain_id else f"ns_{domain_id}"

        evi_idx = 0
        men_idx = 0

        # Helper to register an entity & mention
        def add_entity_mention(b: ContentBlock, ent_type: str, matched_text: str, match_start: int, match_end: int):
            nonlocal evi_idx, men_idx
            canon_name = matched_text.strip()
            if len(canon_name) < 2:
                return None
            ent_id = f"{domain_id}::ENT::{ent_type}::{canon_name}"

            m_start = (b.start_offset or 0) + match_start
            m_end = (b.start_offset or 0) + match_end

            evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
            evi_idx += 1
            quote_window_start = max(0, match_start - 30)
            quote_window_end = min(len(b.text), match_end + 30)
            quote_text = b.text[quote_window_start:quote_window_end].strip()

            ev_rec = EvidenceRecord(
                evidence_id=evi_id,
                domain_id=domain_id,
                document_id=doc.document_id,
                block_id=b.block_id,
                chunk_id=block_to_chunk.get(b.block_id),
                quote_text=quote_text,
                source_location=SourceLocation(start_offset=m_start, end_offset=m_end),
                extractor_id="rule_based_knowledge_extractor",
                extractor_version=profile.profile_version,
                verification_status="VERIFIED",
                origin_type="EXTRACTOR_GENERATED",
            )
            evidences.append(ev_rec)

            if ent_id not in entities_map:
                entities_map[ent_id] = CanonicalEntity(
                    entity_id=ent_id,
                    canonical_name=canon_name,
                    entity_type=ent_type,
                    aliases=[matched_text] if matched_text != canon_name else [],
                    description=f"{ent_type} entity detected in {doc.title}",
                    domain_id=domain_id,
                    graph_namespace=graph_ns,
                    extraction_method="RULE_BASED",
                    extractor_version=profile.profile_version,
                    verification_status="VERIFIED",
                    evidence_refs=[evi_id],
                )
            else:
                if evi_id not in entities_map[ent_id].evidence_refs:
                    entities_map[ent_id].evidence_refs.append(evi_id)

            men_id = f"{doc.document_id}__MEN_{men_idx:04d}"
            men_idx += 1
            mentions.append(
                EntityMention(
                    mention_id=men_id,
                    entity_id=ent_id,
                    domain_id=domain_id,
                    document_id=doc.document_id,
                    block_id=b.block_id,
                    chunk_id=block_to_chunk.get(b.block_id),
                    surface_text=matched_text,
                    source_location=SourceLocation(start_offset=m_start, end_offset=m_end),
                    verification_status="VERIFIED",
                    evidence_id=evi_id,
                )
            )
            return ent_id

        # 1. Extract Entities (Default patterns first, then broader patterns)
        for b in blocks:
            matched_spans = []
            # Default fixture patterns
            for ent_type, pattern_str in self.DEFAULT_ENTITY_PATTERNS.items():
                for match in re.finditer(pattern_str, b.text, re.IGNORECASE):
                    matched_spans.append((match.start(), match.end()))
                    add_entity_mention(b, ent_type, match.group(0), match.start(), match.end())

            # Broader patterns for uncaptured text
            for ent_type, pattern_str in self.BROADER_ENTITY_PATTERNS.items():
                for match in re.finditer(pattern_str, b.text):
                    m_s, m_e = match.start(), match.end()
                    # Skip if strongly overlapping existing spans
                    if any(s <= m_s and m_e <= e for s, e in matched_spans):
                        continue
                    matched_spans.append((m_s, m_e))
                    add_entity_mention(b, ent_type, match.group(0), m_s, m_e)

        # 2. Extract Relationships (Default Patterns)
        rel_idx = 0
        existing_rel_pairs = set()

        for b in blocks:
            b_text = b.text
            for sub_type, rel_pat, predicate, obj_type in self.DEFAULT_RELATION_PATTERNS:
                sub_regex = self.DEFAULT_ENTITY_PATTERNS.get(sub_type, "")
                if not sub_regex:
                    continue

                for sub_m in re.finditer(sub_regex, b_text, re.IGNORECASE):
                    sub_name = sub_m.group(0).strip()
                    sub_canon = sub_name.capitalize() if not sub_name.isupper() else sub_name
                    sub_id = f"{domain_id}::ENT::{sub_type}::{sub_canon}"

                    sub_end = sub_m.end()
                    tail_text = b_text[sub_end : sub_end + 120]
                    r_match = re.search(rel_pat, tail_text, re.IGNORECASE)
                    if r_match:
                        raw_target_name = r_match.group(1).strip()
                        obj_canon = raw_target_name.capitalize() if not raw_target_name.isupper() else raw_target_name
                        obj_id = f"{domain_id}::ENT::{obj_type}::{obj_canon}"

                        if sub_id in entities_map and obj_id in entities_map and sub_id != obj_id:
                            pair_key = (sub_id, predicate, obj_id)
                            if pair_key not in existing_rel_pairs:
                                existing_rel_pairs.add(pair_key)
                                rel_evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
                                evi_idx += 1
                                rel_quote = b_text[max(0, sub_m.start() - 10) : min(len(b_text), sub_end + r_match.end() + 10)].strip()
                                rel_ev = EvidenceRecord(
                                    evidence_id=rel_evi_id,
                                    domain_id=domain_id,
                                    document_id=doc.document_id,
                                    block_id=b.block_id,
                                    chunk_id=block_to_chunk.get(b.block_id),
                                    quote_text=rel_quote,
                                    extractor_id="rule_based_knowledge_extractor",
                                    extractor_version=profile.profile_version,
                                    verification_status="VERIFIED",
                                    origin_type="EXTRACTOR_GENERATED",
                                )
                                evidences.append(rel_ev)

                                rel_id = f"{doc.document_id}__REL_{rel_idx:04d}"
                                rel_idx += 1
                                relationships.append(
                                    CanonicalRelationship(
                                        relationship_id=rel_id,
                                        subject_entity_id=sub_id,
                                        predicate=predicate,
                                        object_entity_id=obj_id,
                                        domain_id=domain_id,
                                        graph_namespace=graph_ns,
                                        evidence_refs=[rel_evi_id],
                                        extraction_method="RULE_BASED",
                                        extractor_version=profile.profile_version,
                                        confidence=0.95,
                                        verification_status="VERIFIED",
                                    )
                                )

        # 3. Extract Broader Relationships for entities within the same block
        for b in blocks:
            b_text = b.text
            # Find all mentions in this block
            block_mentions = [m for m in mentions if m.block_id == b.block_id]
            if len(block_mentions) < 2:
                continue

            for i in range(len(block_mentions) - 1):
                m1 = block_mentions[i]
                m2 = block_mentions[i + 1]
                if m1.entity_id == m2.entity_id:
                    continue

                # Text between mentions
                rel_phrase = b_text[max(0, (m1.source_location.start_offset or 0) - (b.start_offset or 0)) : min(len(b_text), (m2.source_location.end_offset or 0) - (b.start_offset or 0))]
                for predicate, conn_re in self.BROADER_RELATION_CONNECTORS:
                    if conn_re.search(rel_phrase):
                        pair_key = (m1.entity_id, predicate, m2.entity_id)
                        if pair_key not in existing_rel_pairs:
                            existing_rel_pairs.add(pair_key)
                            rel_evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
                            evi_idx += 1
                            rel_ev = EvidenceRecord(
                                evidence_id=rel_evi_id,
                                domain_id=domain_id,
                                document_id=doc.document_id,
                                block_id=b.block_id,
                                chunk_id=block_to_chunk.get(b.block_id),
                                quote_text=rel_phrase.strip() or b.text[:100],
                                extractor_id="rule_based_knowledge_extractor",
                                extractor_version=profile.profile_version,
                                verification_status="VERIFIED",
                                origin_type="EXTRACTOR_GENERATED",
                            )
                            evidences.append(rel_ev)

                            rel_id = f"{doc.document_id}__REL_{rel_idx:04d}"
                            rel_idx += 1
                            relationships.append(
                                CanonicalRelationship(
                                    relationship_id=rel_id,
                                    subject_entity_id=m1.entity_id,
                                    predicate=predicate,
                                    object_entity_id=m2.entity_id,
                                    domain_id=domain_id,
                                    graph_namespace=graph_ns,
                                    evidence_refs=[rel_evi_id],
                                    extraction_method="RULE_BASED",
                                    extractor_version=profile.profile_version,
                                    confidence=0.90,
                                    verification_status="VERIFIED",
                                )
                            )
                            break

        return list(entities_map.values()), mentions, relationships, evidences


def _norm_entity_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def canonicalize_corpus_graph(
    raw_entities: List[CanonicalEntity],
    raw_mentions: List[EntityMention],
    raw_relationships: List[CanonicalRelationship],
    raw_evidences: List[EvidenceRecord],
) -> Tuple[List[CanonicalEntity], List[EntityMention], List[CanonicalRelationship], List[EvidenceRecord]]:
    """
    Hợp nhất thực thể và quan hệ cấp độ toàn bộ tài liệu (Corpus-level Canonicalization):
    - Gộp các thực thể trùng canonical_name (chuẩn hóa chữ thường) hoặc trùng alias đã xác minh.
    - Gộp description, aliases, source_chunk_ids, source_document_ids, evidence_refs.
    - Cập nhật lại subject_entity_id và object_entity_id của relationships.
    - Gộp các relationships trùng cặp (subject_id, predicate, object_id), cộng dồn weight và gộp chunk ids.
    """
    canon_map: Dict[str, CanonicalEntity] = {}  # canon_key -> CanonicalEntity
    alias_to_canon_key: Dict[str, str] = {}
    old_id_to_canon_id: Dict[str, str] = {}

    for ent in raw_entities:
        norm_name = _norm_entity_key(ent.canonical_name)
        if not norm_name:
            continue
        candidate_keys = [norm_name] + [_norm_entity_key(a) for a in ent.aliases if _norm_entity_key(a)]

        matched_canon_key = None
        for k in candidate_keys:
            if k in alias_to_canon_key:
                matched_canon_key = alias_to_canon_key[k]
                break

        if not matched_canon_key or matched_canon_key not in canon_map:
            canon_map[norm_name] = ent.model_copy(deep=True)
            old_id_to_canon_id[ent.entity_id] = ent.entity_id
            for k in candidate_keys:
                alias_to_canon_key[k] = norm_name
        else:
            target = canon_map[matched_canon_key]
            old_id_to_canon_id[ent.entity_id] = target.entity_id

            # Merge aliases
            combined_aliases = list(dict.fromkeys(target.aliases + ent.aliases + [ent.canonical_name]))
            target.aliases = combined_aliases

            # Merge evidence_refs
            target.evidence_refs = list(dict.fromkeys(target.evidence_refs + ent.evidence_refs))

            # Merge chunk and document provenance
            target.source_chunk_ids = list(dict.fromkeys(target.source_chunk_ids + ent.source_chunk_ids))
            target.source_document_ids = list(dict.fromkeys(target.source_document_ids + ent.source_document_ids))

            # Upgrade entity_type if current is CONCEPT and new is more specific
            if target.entity_type == "CONCEPT" and ent.entity_type != "CONCEPT":
                target.entity_type = ent.entity_type

            # Merge description
            if ent.description and ent.description != target.description:
                if not target.description:
                    target.description = ent.description
                elif ent.description not in target.description:
                    target.description = f"{target.description} {ent.description}"

            # Update alias mappings
            for k in candidate_keys:
                alias_to_canon_key[k] = matched_canon_key

    # Remap entity mentions
    updated_mentions: List[EntityMention] = []
    for m in raw_mentions:
        m_copy = m.model_copy(deep=True)
        if m_copy.entity_id in old_id_to_canon_id:
            m_copy.entity_id = old_id_to_canon_id[m_copy.entity_id]
        updated_mentions.append(m_copy)

    # Remap and merge relationships
    rel_map: Dict[Tuple[str, str, str], CanonicalRelationship] = {}
    for r in raw_relationships:
        new_src = old_id_to_canon_id.get(r.subject_entity_id, r.subject_entity_id)
        new_obj = old_id_to_canon_id.get(r.object_entity_id, r.object_entity_id)
        if new_src == new_obj:
            continue  # avoid self-loops

        pair_key = (new_src, r.predicate.strip().upper(), new_obj)
        if pair_key not in rel_map:
            r_copy = r.model_copy(deep=True)
            r_copy.subject_entity_id = new_src
            r_copy.object_entity_id = new_obj
            rel_map[pair_key] = r_copy
        else:
            existing = rel_map[pair_key]
            existing.weight += r.weight
            existing.confidence = max(existing.confidence, r.confidence)
            existing.evidence_refs = list(dict.fromkeys(existing.evidence_refs + r.evidence_refs))
            existing.source_chunk_ids = list(dict.fromkeys(existing.source_chunk_ids + r.source_chunk_ids))
            existing.source_document_ids = list(dict.fromkeys(existing.source_document_ids + r.source_document_ids))
            if r.description and r.description not in existing.description:
                existing.description = f"{existing.description}; {r.description}" if existing.description else r.description

    return list(canon_map.values()), updated_mentions, list(rel_map.values()), raw_evidences


class DeterministicGeneralGraphExtractor(IKnowledgeExtractor):
    """
    Trích xuất đồ thị tri thức tổng quát tất định trên văn bản phi cấu trúc bất kỳ.
    Không phụ thuộc vào kết nối mạng, chạy 100% offline, đảm bảo evidence_quote nguyên văn 100%.
    """

    STOP_WORDS = {
        "khi", "nếu", "tuy", "mặt", "do", "theo", "và", "nhưng", "trong", "để", "với", "tại",
        "các", "những", "một", "bởi", "vì", "về", "từ", "sau", "trước", "cho", "được", "bị",
        "đã", "đang", "sẽ", "phải", "nên", "rằng", "thì", "là", "mà", "hoặc", "cũng", "rất",
        "nhiều", "ít", "hơn", "nhất", "này", "đó", "kia", "đây", "nào", "gì", "sao", "ai"
    }

    CONNECTORS = [
        ("QUY_ĐỊNH", re.compile(r"\b(?:quy định|hướng dẫn|ban hành|áp dụng|tiêu chuẩn)\b", re.I)),
        ("PHỤ_THUỘC_VÀO", re.compile(r"\b(?:phụ thuộc|yêu cầu|cần có|dựa trên|căn cứ)\b", re.I)),
        ("LƯU_TRỮ_TẠI", re.compile(r"\b(?:lưu trữ|chứa|ghi nhận|lưu tại)\b", re.I)),
        ("DẪN_ĐẾN", re.compile(r"\b(?:gây ra|dẫn đến|làm cho|kết quả là|phát sinh)\b", re.I)),
        ("MỞ_RỘNG_TỪ", re.compile(r"\b(?:mở rộng|kế thừa|cải tiến|nâng cấp|tối ưu)\b", re.I)),
        ("BAO_GỒM", re.compile(r"\b(?:bao gồm|gồm có|chứa|thành phần)\b", re.I)),
        ("LIÊN_QUAN_ĐẾN", re.compile(r"\b(?:liên quan|kết nối|tương tác|phối hợp|hỗ trợ)\b", re.I)),
    ]

    def _infer_type(self, name: str) -> str:
        n_low = name.lower()
        if re.match(r"^(?:bộ|sở|công ty|tổ chức|ủy ban|ban|doanh nghiệp|viện|trung tâm)\b", n_low):
            return "ORGANIZATION"
        if re.match(r"^(?:ông|bà|anh|chị|cán bộ|chiến sĩ|kỹ sư|chuyên gia|giám đốc|nhân viên)\b", n_low):
            return "PERSON"
        if re.match(r"^(?:luật|nghị định|thông tư|quy chế|quy định|điều|khoản|chương)\b", n_low):
            return "REGULATION"
        if re.match(r"^(?:máy chủ|hệ thống|dịch vụ|module|cổng|cơ sở dữ liệu|kho|ứng dụng)\b", n_low):
            return "COMPONENT"
        if re.match(r"^(?:giao thức|thuật toán|phương pháp|mô hình|quy trình)\b", n_low):
            return "PROTOCOL"
        if any(term in n_low for term in ("server", "gateway", "database", "service", "cluster", "redis", "auth")):
            return "SERVICE"
        return "CONCEPT"

    def extract_knowledge(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        chunks: List[CanonicalChunk],
        profile: ProcessingProfile,
    ) -> Tuple[
        List[CanonicalEntity],
        List[EntityMention],
        List[CanonicalRelationship],
        List[EvidenceRecord],
    ]:
        raw_entities: List[CanonicalEntity] = []
        raw_mentions: List[EntityMention] = []
        raw_relationships: List[CanonicalRelationship] = []
        raw_evidences: List[EvidenceRecord] = []

        domain_id = doc.domain_id or "graphrag_default"
        graph_ns = domain_id if domain_id.startswith("ns_") else (f"ns_{domain_id.replace('synth_domain_', '').replace('domain_', '')}" if domain_id else "graphrag_default")
        evi_idx = 0
        men_idx = 0
        ent_idx = 0
        rel_idx = 0

        target_chunks = chunks if chunks else []
        UPPER = r"[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]"
        WORD_CHAR = r"[a-zA-Z0-9_/-đáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]"
        entity_regex = re.compile(rf"\b{UPPER}{WORD_CHAR}*(?:\s+{UPPER}{WORD_CHAR}*)*\b")

        for ch in target_chunks:
            ch_text = ch.content
            # Tách câu
            sentences = re.split(r"(?<=[.!?\n])\s+", ch_text)
            chunk_entities: List[Tuple[str, str, str]] = []  # (name, type, quote)

            for sent in sentences:
                sent_clean = sent.strip()
                if not sent_clean:
                    continue

                # Tìm các cụm từ viết hoa hoặc thuật ngữ có nghĩa
                matches = entity_regex.finditer(sent_clean)
                sent_ents: List[str] = []
                for m in matches:
                    name = m.group(0).strip()
                    if name.lower() in self.STOP_WORDS or len(name) < 2:
                        continue
                    if name not in sent_ents:
                        sent_ents.append(name)

                # Tìm thêm các cụm pháp lý/nghiệp vụ đặc biệt ("Điều \d+", "Cán bộ ...")
                special_matches = re.finditer(r"\b(?:Điều\s+\d+|Khoản\s+\d+|Chương\s+[IVXLCDM\d]+|Cán bộ\s+chiến sĩ|Tiền bồi dưỡng|Tiền tàu xe|Vé tàu xe|Công an)\b", sent_clean, re.I)
                for sm in special_matches:
                    s_name = sm.group(0).strip()
                    if s_name not in sent_ents:
                        sent_ents.append(s_name)

                # Tạo Entity và Mention cho các thực thể tìm thấy trong câu
                for ent_name in sent_ents:
                    etype = self._infer_type(ent_name)
                    evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
                    evi_idx += 1
                    ev = EvidenceRecord(
                        evidence_id=evi_id,
                        domain_id=domain_id,
                        document_id=doc.document_id,
                        chunk_id=ch.chunk_id,
                        quote_text=sent_clean,
                        extractor_id="deterministic_general_extractor",
                        extractor_version=profile.profile_version,
                        verification_status="VERIFIED",
                        origin_type="EXTRACTOR_GENERATED",
                    )
                    raw_evidences.append(ev)

                    ent_id = f"{domain_id}::ENT::{etype}::{ent_name.replace(' ', '_')}"
                    ent = CanonicalEntity(
                        entity_id=ent_id,
                        canonical_name=ent_name,
                        entity_type=etype,
                        aliases=[ent_name],
                        description=f"{etype} trích xuất từ tài liệu {doc.title}: '{sent_clean[:80]}'",
                        domain_id=domain_id,
                        graph_namespace=graph_ns,
                        extraction_method="DETERMINISTIC_EXTRACTION",
                        extractor_version=profile.profile_version,
                        verification_status="VERIFIED",
                        evidence_refs=[evi_id],
                        source_chunk_ids=[ch.chunk_id],
                        source_document_ids=[doc.document_id],
                    )
                    raw_entities.append(ent)
                    chunk_entities.append((ent_id, ent_name, sent_clean))

                    men_id = f"{doc.document_id}__MEN_{men_idx:04d}"
                    men_idx += 1
                    raw_mentions.append(
                        EntityMention(
                            mention_id=men_id,
                            entity_id=ent_id,
                            domain_id=domain_id,
                            document_id=doc.document_id,
                            chunk_id=ch.chunk_id,
                            surface_text=ent_name,
                            verification_status="VERIFIED",
                            evidence_id=evi_id,
                        )
                    )

                # Tạo Relationships cho các cặp thực thể cùng xuất hiện trong câu
                if len(sent_ents) >= 2:
                    for i in range(len(sent_ents) - 1):
                        sub_name = sent_ents[i]
                        obj_name = sent_ents[i + 1]
                        if sub_name == obj_name:
                            continue

                        # Nhận diện vị từ nối
                        pred = "LIÊN_QUAN_ĐẾN"
                        for p_name, p_re in self.CONNECTORS:
                            if p_re.search(sent_clean):
                                pred = p_name
                                break

                        sub_type = self._infer_type(sub_name)
                        obj_type = self._infer_type(obj_name)
                        sub_id = f"{domain_id}::ENT::{sub_type}::{sub_name.replace(' ', '_')}"
                        obj_id = f"{domain_id}::ENT::{obj_type}::{obj_name.replace(' ', '_')}"

                        rel_evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
                        evi_idx += 1
                        rel_ev = EvidenceRecord(
                            evidence_id=rel_evi_id,
                            domain_id=domain_id,
                            document_id=doc.document_id,
                            chunk_id=ch.chunk_id,
                            quote_text=sent_clean,
                            extractor_id="deterministic_general_extractor",
                            extractor_version=profile.profile_version,
                            verification_status="VERIFIED",
                            origin_type="EXTRACTOR_GENERATED",
                        )
                        raw_evidences.append(rel_ev)

                        rel_id = f"{doc.document_id}__REL_{rel_idx:04d}"
                        rel_idx += 1
                        raw_relationships.append(
                            CanonicalRelationship(
                                relationship_id=rel_id,
                                subject_entity_id=sub_id,
                                predicate=pred,
                                object_entity_id=obj_id,
                                domain_id=domain_id,
                                graph_namespace=graph_ns,
                                evidence_refs=[rel_evi_id],
                                extraction_method="DETERMINISTIC_EXTRACTION",
                                extractor_version=profile.profile_version,
                                confidence=0.92,
                                verification_status="VERIFIED",
                                description=f"{sub_name} {pred.lower().replace('_', ' ')} {obj_name} (ngữ cảnh: {sent_clean[:70]})",
                                weight=1.0,
                                source_chunk_ids=[ch.chunk_id],
                                source_document_ids=[doc.document_id],
                            )
                        )

        return canonicalize_corpus_graph(raw_entities, raw_mentions, raw_relationships, raw_evidences)


class GraphRAGChunkExtractor(IKnowledgeExtractor):
    """
    Trích xuất đồ thị tri thức cấp độ TextUnit (Chunk) lấy cảm hứng từ Microsoft GraphRAG:
    - Bóc tách theo từng CanonicalChunk có kèm caching (SHA-256 chunk hash + model + prompt_version).
    - Kiểm chứng tính xác thực (Grounding Verification): bắt buộc evidence_quote phải xuất hiện nguyên văn trong chunk.
    - Hỗ trợ cơ chế tự động sửa lỗi (bounded repair/retries) và hợp nhất toàn cục (Corpus-level Canonicalization).
    """

    PROMPT_VERSION = "2.0.0"

    SYSTEM_PROMPT = """Bạn là chuyên gia trích xuất Đồ thị tri thức (GraphRAG Knowledge Extractor).
Nhiệm vụ của bạn là bóc tách danh sách Thực thể (Entities) và Mối quan hệ (Relationships) từ văn bản được cung cấp.

Yêu cầu định dạng JSON chính xác:
{
  "entities": [
    {
      "name": "Tên thực thể chuẩn hóa",
      "type": "ORGANIZATION | PERSON | LOCATION | EVENT | CONCEPT | REGULATION | COMPONENT | PROTOCOL | METRIC",
      "description": "Mô tả ngắn gọn về vai trò hoặc đặc điểm của thực thể trong ngữ cảnh",
      "aliases": ["tên viết tắt hoặc tên khác nếu có"],
      "evidence_quote": "Đoạn văn bản trích dẫn nguyên văn 100% từ ngữ cảnh"
    }
  ],
  "relationships": [
    {
      "source": "Tên thực thể nguồn",
      "target": "Tên thực thể đích",
      "relation_type": "TÊN_QUAN_HỆ (in hoa, cách nhau bởi gạch dưới, ví dụ: LIÊN_QUAN_ĐẾN, QUY_ĐỊNH, PHỤ_THUỘC_VÀO, TÁC_ĐỘNG_ĐẾN...)",
      "description": "Mô tả ngữ nghĩa của mối quan hệ",
      "weight": 1.0,
      "evidence_quote": "Đoạn văn bản trích dẫn nguyên văn 100% từ ngữ cảnh thể hiện mối quan hệ này"
    }
  ]
}

QUY TẮC BẮT BUỘC:
1. Mọi "evidence_quote" BẮT BUỘC phải xuất hiện nguyên văn (substring) trong đoạn văn bản. Tuyệt đối không tự suy diễn hoặc sửa từ trong trích dẫn.
2. Chỉ trả về duy nhất khối JSON thuần (bắt đầu bằng { và kết thúc bằng }), không thêm markdown giải thích bên ngoài. TUYỆT ĐỐI không để dấu phẩy thừa (trailing comma) trước dấu } hoặc ] và luôn escape các dấu ngoặc kép bên trong nội dung văn bản."""

    def __init__(self, llm_config: Optional[Any] = None, db_manager: Optional[Any] = None, max_retries: int = 1):
        self.llm_config = llm_config
        self.db = db_manager
        self.max_retries = max_retries

    @staticmethod
    def _is_quote_grounded(quote: str, content: str) -> bool:
        if not quote or not content:
            return False
        q_norm = re.sub(r"\s+", " ", quote.strip().lower())
        c_norm = re.sub(r"\s+", " ", content.strip().lower())
        return q_norm in c_norm

    def extract_knowledge(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        chunks: List[CanonicalChunk],
        profile: ProcessingProfile,
    ) -> Tuple[
        List[CanonicalEntity],
        List[EntityMention],
        List[CanonicalRelationship],
        List[EvidenceRecord],
    ]:
        # Nếu cấu hình mock hoặc không có API key, dùng bộ trích xuất tất định
        is_mock = (
            self.llm_config is None
            or getattr(self.llm_config, "provider", "mock") == "mock"
            or not getattr(self.llm_config, "api_key", None) and getattr(self.llm_config, "provider", "") == "gemini"
        )
        if is_mock:
            det_extractor = DeterministicGeneralGraphExtractor()
            return det_extractor.extract_knowledge(doc, blocks, chunks, profile)

        from generation.llm_client import MultiProviderLLMClient

        client = MultiProviderLLMClient(self.llm_config)
        model_name = getattr(self.llm_config, "model_name", "llm_model")

        raw_entities: List[CanonicalEntity] = []
        raw_mentions: List[EntityMention] = []
        raw_relationships: List[CanonicalRelationship] = []
        raw_evidences: List[EvidenceRecord] = []

        domain_id = doc.domain_id or "graphrag_default"
        graph_ns = domain_id if domain_id.startswith("ns_") else (f"ns_{domain_id.replace('synth_domain_', '').replace('domain_', '')}" if domain_id else "graphrag_default")
        evi_idx = 0
        men_idx = 0
        rel_idx = 0

        target_chunks = chunks if chunks else []

        for ch_idx, ch in enumerate(target_chunks):
            if hasattr(self, "progress_callback") and self.progress_callback:
                self.progress_callback(f"Đang bóc tách tri thức từ Chunk {ch_idx + 1}/{len(target_chunks)}...", (ch_idx + 1) / len(target_chunks))

            chunk_hash = ch.content_hash or hashlib.sha256(ch.content.encode("utf-8")).hexdigest()
            cache_key = hashlib.sha256(f"{chunk_hash}::{model_name}::{self.PROMPT_VERSION}".encode("utf-8")).hexdigest()

            cached_data = self.db.get_extraction_cache(cache_key) if self.db else None
            parsed_json = None

            if cached_data and isinstance(cached_data, dict):
                parsed_json = cached_data
            else:
                user_prompt = f"Văn bản trích xuất (Chunk ID: {ch.chunk_id}):\n\"\"\"\n{ch.content}\n\"\"\""
                attempt = 0
                last_err = None
                while attempt <= self.max_retries:
                    try:
                        resp = client.generate(system_prompt=self.SYSTEM_PROMPT, user_prompt=user_prompt)
                        candidate_json = robust_json_loads(resp)

                        # Validate JSON schema
                        if isinstance(candidate_json, dict) and "entities" in candidate_json:
                            # Verify quote grounding
                            grounded = True
                            for q_item in candidate_json.get("entities", []) + candidate_json.get("relationships", []):
                                q_text = q_item.get("evidence_quote")
                                if q_text and not self._is_quote_grounded(q_text, ch.content):
                                    grounded = False
                                    break
                            if grounded or attempt == self.max_retries:
                                parsed_json = candidate_json
                                break
                    except Exception as ex:
                        last_err = ex
                        logger.warning(f"Lỗi gọi LLM trích xuất cho chunk {ch.chunk_id} (thử lần {attempt+1}): {ex}")
                    attempt += 1

                if parsed_json and self.db:
                    self.db.set_extraction_cache(cache_key, model_name, self.PROMPT_VERSION, parsed_json)

            if not parsed_json:
                # Nếu LLM gặp lỗi (ví dụ Quota 429, timeout, model 404) hoặc không có JSON hợp lệ, dự phòng bằng DeterministicGeneralGraphExtractor
                err_detail = f": {last_err}" if last_err else ""
                logger.warning(f"LLM trích xuất thất bại hoặc không hợp lệ cho chunk {ch.chunk_id}{err_detail}, kích hoạt bộ bóc tách dự phòng tất định.")
                if hasattr(self, "progress_callback") and self.progress_callback:
                    short_err = f" ({type(last_err).__name__})" if last_err else ""
                    self.progress_callback(
                        f"⚠️ Chunk {ch_idx + 1}/{len(target_chunks)}: LLM trích xuất không thành công{short_err}, dùng bộ bóc tách dự phòng tất định...",
                        (ch_idx + 1) / len(target_chunks),
                    )
                fallback_ext = DeterministicGeneralGraphExtractor()
                f_ents, f_mens, f_rels, f_evis = fallback_ext.extract_knowledge(doc, blocks, [ch], profile)
                raw_entities.extend(f_ents)
                raw_mentions.extend(f_mens)
                raw_relationships.extend(f_rels)
                raw_evidences.extend(f_evis)
                continue

            # Process entities from JSON
            ent_lookup: Dict[str, str] = {}
            for item in parsed_json.get("entities", []):
                e_name = str(item.get("name", "")).strip()
                if not e_name:
                    continue
                e_type = str(item.get("type", "CONCEPT")).strip().upper()
                e_desc = str(item.get("description", "")).strip()
                e_aliases = [str(a).strip() for a in item.get("aliases", []) if str(a).strip()]
                e_quote = str(item.get("evidence_quote", "")).strip()

                is_grounded = self._is_quote_grounded(e_quote, ch.content)
                status = "VERIFIED" if is_grounded else "UNVERIFIED"

                evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
                evi_idx += 1
                ev = EvidenceRecord(
                    evidence_id=evi_id,
                    domain_id=domain_id,
                    document_id=doc.document_id,
                    chunk_id=ch.chunk_id,
                    quote_text=e_quote if is_grounded else ch.content[:120],
                    extractor_id=f"graphrag_llm_extractor_{model_name}",
                    extractor_version=self.PROMPT_VERSION,
                    verification_status=status,
                    origin_type="EXTRACTOR_GENERATED",
                )
                raw_evidences.append(ev)

                ent_id = f"{domain_id}::ENT::{e_type}::{e_name.replace(' ', '_')}"
                ent_lookup[_norm_entity_key(e_name)] = ent_id
                for al in e_aliases:
                    al_k = _norm_entity_key(al)
                    if al_k and al_k not in ent_lookup:
                        ent_lookup[al_k] = ent_id

                ent = CanonicalEntity(
                    entity_id=ent_id,
                    canonical_name=e_name,
                    entity_type=e_type,
                    aliases=e_aliases if e_name in e_aliases else [e_name] + e_aliases,
                    description=e_desc or f"{e_type} được trích xuất từ {doc.title}",
                    domain_id=domain_id,
                    graph_namespace=graph_ns,
                    extraction_method="LLM_GRAPHRAG",
                    extractor_version=self.PROMPT_VERSION,
                    verification_status=status,
                    evidence_refs=[evi_id],
                    source_chunk_ids=[ch.chunk_id],
                    source_document_ids=[doc.document_id],
                )
                raw_entities.append(ent)

                men_id = f"{doc.document_id}__MEN_{men_idx:04d}"
                men_idx += 1
                raw_mentions.append(
                    EntityMention(
                        mention_id=men_id,
                        entity_id=ent_id,
                        domain_id=domain_id,
                        document_id=doc.document_id,
                        chunk_id=ch.chunk_id,
                        surface_text=e_name,
                        verification_status=status,
                        evidence_id=evi_id,
                    )
                )

            # Process relationships from JSON
            for r_item in parsed_json.get("relationships", []):
                s_name = str(r_item.get("source", "")).strip()
                t_name = str(r_item.get("target", "")).strip()
                pred = str(r_item.get("relation_type", "LIÊN_QUAN_ĐẾN")).strip().upper().replace(" ", "_")
                r_desc = str(r_item.get("description", "")).strip()
                weight = float(r_item.get("weight", 1.0))
                r_quote = str(r_item.get("evidence_quote", "")).strip()

                s_id = ent_lookup.get(_norm_entity_key(s_name))
                t_id = ent_lookup.get(_norm_entity_key(t_name))
                if not s_id or not t_id or s_id == t_id:
                    continue

                is_r_grounded = self._is_quote_grounded(r_quote, ch.content)
                r_status = "VERIFIED" if is_r_grounded else "UNVERIFIED"

                rel_evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
                evi_idx += 1
                rel_ev = EvidenceRecord(
                    evidence_id=rel_evi_id,
                    domain_id=domain_id,
                    document_id=doc.document_id,
                    chunk_id=ch.chunk_id,
                    quote_text=r_quote if is_r_grounded else ch.content[:120],
                    extractor_id=f"graphrag_llm_extractor_{model_name}",
                    extractor_version=self.PROMPT_VERSION,
                    verification_status=r_status,
                    origin_type="EXTRACTOR_GENERATED",
                )
                raw_evidences.append(rel_ev)

                rel_id = f"{doc.document_id}__REL_{rel_idx:04d}"
                rel_idx += 1
                raw_relationships.append(
                    CanonicalRelationship(
                        relationship_id=rel_id,
                        subject_entity_id=s_id,
                        predicate=pred,
                        object_entity_id=t_id,
                        domain_id=domain_id,
                        graph_namespace=graph_ns,
                        evidence_refs=[rel_evi_id],
                        extraction_method="LLM_GRAPHRAG",
                        extractor_version=self.PROMPT_VERSION,
                        confidence=0.95 if is_r_grounded else 0.50,
                        verification_status=r_status,
                        description=r_desc or f"{s_name} {pred.lower()} {t_name}",
                        weight=max(0.1, weight),
                        source_chunk_ids=[ch.chunk_id],
                        source_document_ids=[doc.document_id],
                    )
                )

        return canonicalize_corpus_graph(raw_entities, raw_mentions, raw_relationships, raw_evidences)


class LLMKnowledgeExtractor(IKnowledgeExtractor):
    """
    Adapter tương thích ngược kết nối tới GraphRAGChunkExtractor.
    """

    def __init__(self, llm_config: Optional[Any] = None, db_manager: Optional[Any] = None):
        self.llm_config = llm_config
        self.db = db_manager

    def extract_knowledge(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        chunks: List[CanonicalChunk],
        profile: ProcessingProfile,
    ) -> Tuple[
        List[CanonicalEntity],
        List[EntityMention],
        List[CanonicalRelationship],
        List[EvidenceRecord],
    ]:
        extractor = GraphRAGChunkExtractor(llm_config=self.llm_config, db_manager=self.db)
        return extractor.extract_knowledge(doc, blocks, chunks, profile)


class ExtractorRegistry:
    _metadata_extractors: Dict[str, IMetadataExtractor] = {
        "basic_structural": BasicStructuralMetadataExtractor(),
    }
    _knowledge_extractors: Dict[str, IKnowledgeExtractor] = {
        "rule_based_ops": RuleBasedKnowledgeExtractor(),
        "rule_based_incident": RuleBasedKnowledgeExtractor(),
        "rule_based_rd": RuleBasedKnowledgeExtractor(),
        "deterministic_general": DeterministicGeneralGraphExtractor(),
        "graphrag_extractor": GraphRAGChunkExtractor(),
        "llm_extractor": LLMKnowledgeExtractor(),
    }

    @classmethod
    def get_metadata_extractor(cls, name: str) -> Optional[IMetadataExtractor]:
        return cls._metadata_extractors.get(name)

    @classmethod
    def get_knowledge_extractor(cls, name: str) -> IKnowledgeExtractor:
        if name in cls._knowledge_extractors:
            return cls._knowledge_extractors[name]
        return DeterministicGeneralGraphExtractor()

    @classmethod
    def list_knowledge_extractors(cls) -> List[str]:
        return list(cls._knowledge_extractors.keys())

