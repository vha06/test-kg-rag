import re
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
    Có thể cấu hình mẫu từ điển hoặc biểu thức trong profile.
    Tự động gắn cờ origin_type='EXTRACTOR_GENERATED' và tạo EvidenceRecord có toạ độ thực.
    """

    DEFAULT_ENTITY_PATTERNS = {
        "SERVICE": r"\b(Gateway|AuthServer|Redis|Database|BillingService|SearchEngine|WorkerPool)\b",
        "SERVER": r"\b(Server_[A-Z0-9]+|Host_[A-Z0-9]+|Node_\d+)\b",
        "PROTOCOL": r"\b(HTTP|HTTPS|gRPC|TCP|UDP|REST|WebSocket)\b",
        "INCIDENT": r"\b(INCIDENT_\d+|Sự cố\s+[A-Z0-9_]+)\b",
    }

    DEFAULT_RELATION_PATTERNS = [
        # (subject_type, regex_trigger, predicate, object_type)
        ("SERVICE", r"(?:phụ thuộc|yêu cầu|kết nối tới|gọi tới|depends on|calls)\s+(?:dịch vụ\s+)?(\w+)", "DEPENDS_ON", "SERVICE"),
        ("SERVICE", r"(?:lưu trữ tại|ghi dữ liệu vào|stores in)\s+(?:cơ sở dữ liệu\s+)?(\w+)", "STORES_DATA_IN", "SERVICE"),
        ("INCIDENT", r"(?:bắt nguồn từ|gây ra bởi|caused by)\s+(?:dịch vụ\s+)?(\w+)", "TRIGGERED_BY", "SERVICE"),
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

        # 1. Extract Entity Mentions
        evi_idx = 0
        men_idx = 0

        for b in blocks:
            for ent_type, pattern_str in self.DEFAULT_ENTITY_PATTERNS.items():
                for match in re.finditer(pattern_str, b.text, re.IGNORECASE):
                    matched_text = match.group(0).strip()
                    canon_name = matched_text.capitalize() if not matched_text.isupper() else matched_text
                    ent_id = f"{domain_id}::ENT::{ent_type}::{canon_name}"

                    # Compute offsets
                    m_start = (b.start_offset or 0) + match.start()
                    m_end = (b.start_offset or 0) + match.end()

                    # Create Evidence
                    evi_id = f"{doc.document_id}__EVI_{evi_idx:04d}"
                    evi_idx += 1
                    quote_window_start = max(0, match.start() - 30)
                    quote_window_end = min(len(b.text), match.end() + 30)
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

                    # Create or update Entity
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
                        if matched_text not in entities_map[ent_id].aliases and matched_text != canon_name:
                            entities_map[ent_id].aliases.append(matched_text)

                    # Create Mention
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

        # 2. Extract Relationships
        rel_idx = 0
        for b in blocks:
            b_text = b.text
            for sub_type, rel_pat, predicate, obj_type in self.DEFAULT_RELATION_PATTERNS:
                # Find matching subjects in this block
                sub_regex = self.DEFAULT_ENTITY_PATTERNS.get(sub_type, "")
                if not sub_regex:
                    continue

                for sub_m in re.finditer(sub_regex, b_text, re.IGNORECASE):
                    sub_name = sub_m.group(0).strip()
                    sub_canon = sub_name.capitalize() if not sub_name.isupper() else sub_name
                    sub_id = f"{domain_id}::ENT::{sub_type}::{sub_canon}"

                    # Look for relationship phrase right after or near subject
                    sub_end = sub_m.end()
                    tail_text = b_text[sub_end : sub_end + 120]
                    r_match = re.search(rel_pat, tail_text, re.IGNORECASE)
                    if r_match:
                        raw_target_name = r_match.group(1).strip()
                        obj_canon = raw_target_name.capitalize() if not raw_target_name.isupper() else raw_target_name
                        obj_id = f"{domain_id}::ENT::{obj_type}::{obj_canon}"

                        # Ensure both entities exist in our entity map
                        if sub_id in entities_map and obj_id in entities_map and sub_id != obj_id:
                            # Create an evidence for the relationship
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

        return list(entities_map.values()), mentions, relationships, evidences


class ExtractorRegistry:
    _metadata_extractors: Dict[str, IMetadataExtractor] = {
        "basic_structural": BasicStructuralMetadataExtractor(),
    }
    _knowledge_extractors: Dict[str, IKnowledgeExtractor] = {
        "rule_based_ops": RuleBasedKnowledgeExtractor(),
        "rule_based_incident": RuleBasedKnowledgeExtractor(),
        "rule_based_rd": RuleBasedKnowledgeExtractor(),
    }

    @classmethod
    def get_metadata_extractor(cls, name: str) -> Optional[IMetadataExtractor]:
        return cls._metadata_extractors.get(name)

    @classmethod
    def get_knowledge_extractor(cls, name: str) -> IKnowledgeExtractor:
        if name in cls._knowledge_extractors:
            return cls._knowledge_extractors[name]
        return RuleBasedKnowledgeExtractor()
