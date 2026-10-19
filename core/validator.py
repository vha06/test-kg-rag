from typing import List, Dict, Any, Set, Tuple, Optional
from core.schema import (
    CanonicalDocument,
    ContentBlock,
    CanonicalChunk,
    EvidenceRecord,
    CanonicalEntity,
    EntityMention,
    CanonicalRelationship,
)


class ValidationIssue:
    def __init__(self, level: str, entity_type: str, item_id: str, message: str):
        self.level = level  # ERROR, WARNING
        self.entity_type = entity_type
        self.item_id = item_id
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        return {
            "level": self.level,
            "entity_type": self.entity_type,
            "item_id": self.item_id,
            "message": self.message,
        }

    def __repr__(self) -> str:
        return f"[{self.level}] {self.entity_type}:{self.item_id} -> {self.message}"


class CanonicalSchemaValidator:
    """
    Validator kiểm tra tính toàn vẹn khoá ngoại, ranh giới namespace
    và tính hợp lệ của bằng chứng (provenance) trước khi ghi vào storage hoặc graph.
    """

    @classmethod
    def validate_document_bundle(
        cls,
        document: CanonicalDocument,
        blocks: List[ContentBlock],
        chunks: List[CanonicalChunk],
        entities: Optional[List[CanonicalEntity]] = None,
        mentions: Optional[List[EntityMention]] = None,
        relationships: Optional[List[CanonicalRelationship]] = None,
        evidences: Optional[List[EvidenceRecord]] = None,
        enforce_verified_evidence: bool = True,
    ) -> Tuple[bool, List[ValidationIssue]]:
        issues: List[ValidationIssue] = []

        doc_id = document.document_id
        domain_id = document.domain_id

        # 1. Validate ContentBlocks
        block_ids: Set[str] = set()
        for b in blocks:
            if b.document_id != doc_id:
                issues.append(ValidationIssue(
                    "ERROR", "ContentBlock", b.block_id,
                    f"Block document_id '{b.document_id}' does not match Document '{doc_id}'."
                ))
            if b.domain_id != domain_id:
                issues.append(ValidationIssue(
                    "ERROR", "ContentBlock", b.block_id,
                    f"Block domain_id '{b.domain_id}' does not match Document domain '{domain_id}'."
                ))
            if b.block_id in block_ids:
                issues.append(ValidationIssue(
                    "ERROR", "ContentBlock", b.block_id,
                    f"Duplicate block_id '{b.block_id}' in Document '{doc_id}'."
                ))
            block_ids.add(b.block_id)

        # Check parent_block_id validity
        for b in blocks:
            if b.parent_block_id and b.parent_block_id not in block_ids:
                issues.append(ValidationIssue(
                    "WARNING", "ContentBlock", b.block_id,
                    f"parent_block_id '{b.parent_block_id}' does not exist in document's blocks."
                ))

        # 2. Validate Chunks
        chunk_ids: Set[str] = set()
        for ch in chunks:
            if ch.document_id != doc_id:
                issues.append(ValidationIssue(
                    "ERROR", "CanonicalChunk", ch.chunk_id,
                    f"Chunk document_id '{ch.document_id}' does not match Document '{doc_id}'."
                ))
            if ch.domain_id != domain_id:
                issues.append(ValidationIssue(
                    "ERROR", "CanonicalChunk", ch.chunk_id,
                    f"Chunk domain_id '{ch.domain_id}' does not match Document domain '{domain_id}'."
                ))
            if ch.chunk_id in chunk_ids:
                issues.append(ValidationIssue(
                    "ERROR", "CanonicalChunk", ch.chunk_id,
                    f"Duplicate chunk_id '{ch.chunk_id}'."
                ))
            chunk_ids.add(ch.chunk_id)

            # Check that all source_block_ids exist
            for src_b_id in ch.source_block_ids:
                if src_b_id not in block_ids:
                    issues.append(ValidationIssue(
                        "ERROR", "CanonicalChunk", ch.chunk_id,
                        f"Chunk references non-existent source_block_id '{src_b_id}'."
                    ))

        # 3. Validate Evidences
        evidence_ids: Set[str] = set()
        if evidences:
            for ev in evidences:
                if ev.document_id != doc_id:
                    issues.append(ValidationIssue(
                        "ERROR", "EvidenceRecord", ev.evidence_id,
                        f"Evidence document_id '{ev.document_id}' does not match Document '{doc_id}'."
                    ))
                if ev.domain_id != domain_id:
                    issues.append(ValidationIssue(
                        "ERROR", "EvidenceRecord", ev.evidence_id,
                        f"Evidence domain_id '{ev.domain_id}' does not match Document domain '{domain_id}'."
                    ))
                if ev.block_id and ev.block_id not in block_ids:
                    issues.append(ValidationIssue(
                        "WARNING", "EvidenceRecord", ev.evidence_id,
                        f"Evidence references unknown block_id '{ev.block_id}'."
                    ))
                if ev.chunk_id and ev.chunk_id not in chunk_ids:
                    issues.append(ValidationIssue(
                        "WARNING", "EvidenceRecord", ev.evidence_id,
                        f"Evidence references unknown chunk_id '{ev.chunk_id}'."
                    ))
                evidence_ids.add(ev.evidence_id)

        # 4. Validate Entities
        entity_ids: Set[str] = set()
        if entities:
            for ent in entities:
                if ent.domain_id != domain_id:
                    issues.append(ValidationIssue(
                        "ERROR", "CanonicalEntity", ent.entity_id,
                        f"Entity domain_id '{ent.domain_id}' does not match Document domain '{domain_id}'."
                    ))
                if ent.entity_id in entity_ids:
                    issues.append(ValidationIssue(
                        "ERROR", "CanonicalEntity", ent.entity_id,
                        f"Duplicate entity_id '{ent.entity_id}'."
                    ))
                entity_ids.add(ent.entity_id)

        # 5. Validate EntityMentions
        if mentions:
            for m in mentions:
                if m.document_id != doc_id:
                    issues.append(ValidationIssue(
                        "ERROR", "EntityMention", m.mention_id,
                        f"Mention document_id '{m.document_id}' does not match Document '{doc_id}'."
                    ))
                if m.domain_id != domain_id:
                    issues.append(ValidationIssue(
                        "ERROR", "EntityMention", m.mention_id,
                        f"Mention domain_id '{m.domain_id}' does not match Document domain '{domain_id}'."
                    ))
                if entity_ids and m.entity_id not in entity_ids:
                    issues.append(ValidationIssue(
                        "ERROR", "EntityMention", m.mention_id,
                        f"Mention references unknown entity_id '{m.entity_id}'."
                    ))
                if m.block_id and m.block_id not in block_ids:
                    issues.append(ValidationIssue(
                        "ERROR", "EntityMention", m.mention_id,
                        f"Mention references unknown block_id '{m.block_id}'."
                    ))

        # 6. Validate Relationships
        if relationships:
            for rel in relationships:
                if rel.domain_id != domain_id:
                    issues.append(ValidationIssue(
                        "ERROR", "CanonicalRelationship", rel.relationship_id,
                        f"Relationship domain_id '{rel.domain_id}' does not match Document domain '{domain_id}'."
                    ))
                if entity_ids:
                    if rel.subject_entity_id not in entity_ids:
                        issues.append(ValidationIssue(
                            "ERROR", "CanonicalRelationship", rel.relationship_id,
                            f"Relationship subject '{rel.subject_entity_id}' does not exist in entities."
                        ))
                    if rel.object_entity_id not in entity_ids:
                        issues.append(ValidationIssue(
                            "ERROR", "CanonicalRelationship", rel.relationship_id,
                            f"Relationship object '{rel.object_entity_id}' does not exist in entities."
                        ))

                # Check evidence references
                if enforce_verified_evidence and rel.verification_status == "VERIFIED":
                    if not rel.evidence_refs:
                        issues.append(ValidationIssue(
                            "ERROR", "CanonicalRelationship", rel.relationship_id,
                            "VERIFIED relationship must have at least one valid evidence reference in evidence_refs."
                        ))
                    elif evidence_ids:
                        for ev_ref in rel.evidence_refs:
                            if ev_ref not in evidence_ids:
                                issues.append(ValidationIssue(
                                    "ERROR", "CanonicalRelationship", rel.relationship_id,
                                    f"Relationship references unknown evidence_id '{ev_ref}'."
                                ))

        has_errors = any(i.level == "ERROR" for i in issues)
        return (not has_errors), issues
