import re
import json
from typing import List, Dict, Any, Tuple, Optional
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
from pipeline.interfaces import IMetadataExtractor, IKnowledgeExtractor


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


class LLMKnowledgeExtractor(IKnowledgeExtractor):
    """
    Extractor sử dụng LLM Client (Gemini, OpenAI, Ollama) để trích xuất thực thể
    và mối quan hệ linh hoạt trên mọi thể loại văn bản.
    Tự động fallback về RuleBasedKnowledgeExtractor nếu không có API key hoặc lỗi.
    """

    def __init__(self, llm_config: Optional[Any] = None):
        self.llm_config = llm_config

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
        fallback_extractor = RuleBasedKnowledgeExtractor()
        if not self.llm_config or getattr(self.llm_config, "provider", "mock") == "mock":
            return fallback_extractor.extract_knowledge(doc, blocks, chunks, profile)

        try:
            from generation.llm_client import MultiProviderLLMClient
            client = MultiProviderLLMClient(self.llm_config)

            combined_text = "\n\n".join([f"[{b.block_id}]: {b.text}" for b in blocks[:12]])
            system_prompt = (
                "Bạn là chuyên gia bóc tách Đồ thị tri thức (Knowledge Graph) từ văn bản tiếng Việt. "
                "Hãy trích xuất danh sách thực thể (Entities) và mối quan hệ (Relationships) từ văn bản sau. "
                "Chỉ trả về duy nhất định dạng JSON thuần:\n"
                "{\n"
                '  "entities": [{"name": "Tên thực thể", "type": "SERVICE|SERVER|DATABASE|PROTOCOL|INCIDENT|COMPONENT"}],\n'
                '  "relationships": [{"subject": "Tên thực thể nguồn", "predicate": "DEPENDS_ON|STORES_DATA_IN|TRIGGERED_BY", "object": "Tên thực thể đích", "quote": "Đoạn văn bản trích dẫn ngắn"}]\n'
                "}"
            )
            resp = client.generate(system_prompt=system_prompt, user_prompt=combined_text)
            cleaned = re.sub(r"^```(?:json)?\s*", "", resp.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            raw_entities = data.get("entities", [])
            raw_relations = data.get("relationships", [])
            if not raw_entities:
                return fallback_extractor.extract_knowledge(doc, blocks, chunks, profile)

            entities_map: Dict[str, CanonicalEntity] = {}
            mentions: List[EntityMention] = []
            relationships: List[CanonicalRelationship] = []
            evidences: List[EvidenceRecord] = []

            block_to_chunk = {}
            for ch in chunks:
                for b_id in ch.source_block_ids:
                    block_to_chunk[b_id] = ch.chunk_id

            domain_id = doc.domain_id
            graph_ns = f"ns_{domain_id.replace('synth_domain_', '')}" if "synth_domain_" in domain_id else f"ns_{domain_id}"
            evi_idx = 0
            men_idx = 0

            # Register entities
            for item in raw_entities:
                name = str(item.get("name", "")).strip()
                etype = str(item.get("type", "COMPONENT")).strip().upper()
                if not name:
                    continue
                ent_id = f"{domain_id}::ENT::{etype}::{name}"

                # Find block containing name
                target_block = next((b for b in blocks if name.lower() in b.text.lower()), blocks[0] if blocks else None)
                if not target_block:
                    continue

                m_pos = target_block.text.lower().find(name.lower())
                m_start = (target_block.start_offset or 0) + (m_pos if m_pos != -1 else 0)
                m_end = m_start + len(name)

                evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
                evi_idx += 1
                ev_rec = EvidenceRecord(
                    evidence_id=evi_id,
                    domain_id=domain_id,
                    document_id=doc.document_id,
                    block_id=target_block.block_id,
                    chunk_id=block_to_chunk.get(target_block.block_id),
                    quote_text=name,
                    source_location=SourceLocation(start_offset=m_start, end_offset=m_end),
                    extractor_id="llm_knowledge_extractor",
                    extractor_version=profile.profile_version,
                    verification_status="VERIFIED",
                    origin_type="EXTRACTOR_GENERATED",
                )
                evidences.append(ev_rec)

                entities_map[ent_id] = CanonicalEntity(
                    entity_id=ent_id,
                    canonical_name=name,
                    entity_type=etype,
                    aliases=[name],
                    description=f"{etype} entity extracted by LLM from {doc.title}",
                    domain_id=domain_id,
                    graph_namespace=graph_ns,
                    extraction_method="LLM_ASSISTED",
                    extractor_version=profile.profile_version,
                    verification_status="VERIFIED",
                    evidence_refs=[evi_id],
                )

                men_id = f"{doc.document_id}__MEN_{men_idx:04d}"
                men_idx += 1
                mentions.append(
                    EntityMention(
                        mention_id=men_id,
                        entity_id=ent_id,
                        domain_id=domain_id,
                        document_id=doc.document_id,
                        block_id=target_block.block_id,
                        chunk_id=block_to_chunk.get(target_block.block_id),
                        surface_text=name,
                        source_location=SourceLocation(start_offset=m_start, end_offset=m_end),
                        verification_status="VERIFIED",
                        evidence_id=evi_id,
                    )
                )

            # Register relations
            rel_idx = 0
            for r in raw_relations:
                sub_name = str(r.get("subject", "")).strip()
                pred = str(r.get("predicate", "DEPENDS_ON")).strip().upper()
                obj_name = str(r.get("object", "")).strip()
                quote = str(r.get("quote", f"{sub_name} {pred} {obj_name}")).strip()

                sub_id = next((eid for eid, ent in entities_map.items() if ent.canonical_name.lower() == sub_name.lower()), None)
                obj_id = next((eid for eid, ent in entities_map.items() if ent.canonical_name.lower() == obj_name.lower()), None)

                if sub_id and obj_id and sub_id != obj_id:
                    rel_evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
                    evi_idx += 1
                    target_b = blocks[0]
                    rel_ev = EvidenceRecord(
                        evidence_id=rel_evi_id,
                        domain_id=domain_id,
                        document_id=doc.document_id,
                        block_id=target_b.block_id,
                        chunk_id=block_to_chunk.get(target_b.block_id),
                        quote_text=quote,
                        extractor_id="llm_knowledge_extractor",
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
                            predicate=pred,
                            object_entity_id=obj_id,
                            domain_id=domain_id,
                            graph_namespace=graph_ns,
                            evidence_refs=[rel_evi_id],
                            extraction_method="LLM_ASSISTED",
                            extractor_version=profile.profile_version,
                            confidence=0.90,
                            verification_status="VERIFIED",
                        )
                    )

            if not entities_map:
                return fallback_extractor.extract_knowledge(doc, blocks, chunks, profile)

            return list(entities_map.values()), mentions, relationships, evidences

        except Exception:
            return fallback_extractor.extract_knowledge(doc, blocks, chunks, profile)


class ExtractorRegistry:
    _metadata_extractors: Dict[str, IMetadataExtractor] = {
        "basic_structural": BasicStructuralMetadataExtractor(),
    }
    _knowledge_extractors: Dict[str, IKnowledgeExtractor] = {
        "rule_based_ops": RuleBasedKnowledgeExtractor(),
        "rule_based_incident": RuleBasedKnowledgeExtractor(),
        "rule_based_rd": RuleBasedKnowledgeExtractor(),
        "llm_extractor": LLMKnowledgeExtractor(),
    }

    @classmethod
    def get_metadata_extractor(cls, name: str) -> Optional[IMetadataExtractor]:
        return cls._metadata_extractors.get(name)

    @classmethod
    def get_knowledge_extractor(cls, name: str) -> IKnowledgeExtractor:
        if name in cls._knowledge_extractors:
            return cls._knowledge_extractors[name]
        return RuleBasedKnowledgeExtractor()

    @classmethod
    def list_knowledge_extractors(cls) -> List[str]:
        return list(cls._knowledge_extractors.keys())

