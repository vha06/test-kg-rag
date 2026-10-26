from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class SourceLocation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    page_number: Optional[int] = None
    section_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_offsets(self) -> SourceLocation:
        if self.start_offset is not None and self.end_offset is not None:
            if self.start_offset < 0 or self.end_offset < 0:
                raise ValueError("Offsets must be non-negative integers.")
            if self.start_offset > self.end_offset:
                raise ValueError(
                    f"start_offset ({self.start_offset}) cannot be greater than end_offset ({self.end_offset})."
                )
        return self


class ACLPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    owner: str = "system"
    allowed_roles: List[str] = Field(default_factory=lambda: ["public"])
    allowed_users: List[str] = Field(default_factory=list)
    classification: str = "INTERNAL"  # PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED

    @field_validator("classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        valid_cls = {"PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"}
        v_upper = v.upper()
        if v_upper not in valid_cls:
            raise ValueError(f"Invalid classification '{v}'. Allowed: {valid_cls}")
        return v_upper


class CanonicalDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str
    schema_version: str = "1.0.0"
    title: str
    source_id: str
    domain_id: str = "unstructured"
    document_type: str = "general"
    profile_id: str = "graphrag_default"
    profile_version: str = "1.0.0"
    checksum: str
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_status: str = "PENDING"  # PENDING, PARSED, CHUNKED, EXTRACTED, INDEXED, FAILED
    security_classification: str = "INTERNAL"
    source_uri: Optional[str] = None
    mime_type: str = "text/plain"
    language: str = "vi"
    acl: ACLPolicy = Field(default_factory=ACLPolicy)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("document_id", "domain_id", "document_type", "profile_id")
    @classmethod
    def check_non_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"Field '{info.field_name}' must be non-empty.")
        return v.strip()


class ContentBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    block_id: str
    document_id: str
    domain_id: str
    block_index: int
    block_type: str = "PARAGRAPH"  # HEADING, PARAGRAPH, LIST_ITEM, TABLE, CODE, GENERIC
    text: str = ""
    parent_block_id: Optional[str] = None
    source_location: SourceLocation = Field(default_factory=SourceLocation)
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("block_type")
    @classmethod
    def validate_block_type(cls, v: str) -> str:
        valid_types = {"HEADING", "PARAGRAPH", "LIST_ITEM", "TABLE", "CODE", "GENERIC"}
        v_upper = v.upper()
        if v_upper not in valid_types:
            raise ValueError(f"Invalid block_type '{v}'. Allowed: {valid_types}")
        return v_upper


class CanonicalChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    document_id: str
    domain_id: str = "unstructured"
    source_block_ids: List[str] = Field(default_factory=list)
    content: str
    chunk_index: int = 0
    chunking_strategy: str = "generic"
    profile_id: str = "graphrag_default"
    profile_version: str = "1.0.0"
    token_count: int = 0
    source_location: SourceLocation = Field(default_factory=SourceLocation)
    security_classification: str = "INTERNAL"
    acl: ACLPolicy = Field(default_factory=ACLPolicy)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""
    start_char: int = 0
    end_char: int = 0
    section_path: List[str] = Field(default_factory=list)
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    @field_validator("source_block_ids")
    @classmethod
    def check_source_blocks(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("CanonicalChunk must have at least one source_block_id.")
        return v


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    evidence_id: str
    domain_id: str
    document_id: str
    block_id: Optional[str] = None
    chunk_id: Optional[str] = None
    quote_text: str
    source_location: SourceLocation = Field(default_factory=SourceLocation)
    extractor_id: str = "system"
    extractor_version: str = "1.0.0"
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    verification_status: str = "UNVERIFIED"  # UNVERIFIED, VERIFIED, REJECTED
    origin_type: str = "EXTRACTOR_GENERATED"  # FIXTURE_PROVIDED, EXTRACTOR_GENERATED, RULE_BASED

    @field_validator("verification_status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"UNVERIFIED", "VERIFIED", "REJECTED"}
        v_up = v.upper()
        if v_up not in valid:
            raise ValueError(f"Invalid verification_status '{v}'. Allowed: {valid}")
        return v_up

    @field_validator("origin_type")
    @classmethod
    def validate_origin(cls, v: str) -> str:
        valid = {"FIXTURE_PROVIDED", "EXTRACTOR_GENERATED", "RULE_BASED"}
        v_up = v.upper()
        if v_up not in valid:
            raise ValueError(f"Invalid origin_type '{v}'. Allowed: {valid}")
        return v_up


class CanonicalEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    domain_id: str = "unstructured"
    graph_namespace: str = "graphrag_default"
    extraction_method: str = "RULE_BASED"
    extractor_version: str = "1.0.0"
    verification_status: str = "UNVERIFIED"  # UNVERIFIED, VERIFIED, REJECTED
    schema_version: str = "1.0.0"
    attributes: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_document_ids: List[str] = Field(default_factory=list)

    @field_validator("verification_status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"UNVERIFIED", "VERIFIED", "REJECTED"}
        v_up = v.upper()
        if v_up not in valid:
            raise ValueError(f"Invalid verification_status '{v}'. Allowed: {valid}")
        return v_up


class EntityMention(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mention_id: str
    entity_id: str
    domain_id: str
    document_id: str
    block_id: Optional[str] = None
    chunk_id: Optional[str] = None
    surface_text: str
    source_location: SourceLocation = Field(default_factory=SourceLocation)
    verification_status: str = "UNVERIFIED"
    evidence_id: Optional[str] = None


class CanonicalRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relationship_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    domain_id: str = "unstructured"
    graph_namespace: str = "graphrag_default"
    evidence_refs: List[str] = Field(default_factory=list)
    extraction_method: str = "RULE_BASED"
    extractor_version: str = "1.0.0"
    confidence: float = 1.0
    verification_status: str = "UNVERIFIED"  # UNVERIFIED, VERIFIED, REJECTED
    recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1.0.0"
    attributes: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_document_ids: List[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("verification_status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"UNVERIFIED", "VERIFIED", "REJECTED"}
        v_up = v.upper()
        if v_up not in valid:
            raise ValueError(f"Invalid verification_status '{v}'. Allowed: {valid}")
        return v_up


class CanonicalClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_id: str
    domain_id: str
    subject: str
    predicate: str
    object_value: str
    polarity: str = "POSITIVE"  # POSITIVE, NEGATIVE
    qualifiers: Dict[str, Any] = Field(default_factory=dict)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)
    verification_status: str = "UNVERIFIED"


class CanonicalCommunity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    community_id: str
    namespace: str = "graphrag_default"
    level: int = 0
    parent_community_id: Optional[str] = None
    child_community_ids: List[str] = Field(default_factory=list)
    title: str = ""
    entity_ids: List[str] = Field(default_factory=list)
    relationship_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_document_ids: List[str] = Field(default_factory=list)
    period: Optional[str] = None
    size: int = 0
    attributes: Dict[str, Any] = Field(default_factory=dict)


class CanonicalCommunityReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report_id: str
    community_id: str
    namespace: str = "graphrag_default"
    level: int = 0
    title: str = ""
    summary: str = ""
    full_content: str = ""
    key_findings: List[str] = Field(default_factory=list)
    supporting_entity_ids: List[str] = Field(default_factory=list)
    supporting_relationship_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_document_ids: List[str] = Field(default_factory=list)
    rating: float = 0.0
    generation_model: str = "unknown"
    prompt_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GraphRAGIndexManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    namespace: str = "graphrag_default"
    corpus_fingerprint: str = ""
    chunking_strategy: str = "recursive_boundary"
    chunk_size: int = 300
    chunk_overlap: int = 50
    extractor_model: str = "mock"
    extraction_prompt_version: str = "1.0.0"
    embedding_model: str = "sentence-transformers"
    community_algorithm: str = "louvain_hierarchical"
    max_community_levels: int = 3
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    artifact_counts: Dict[str, int] = Field(default_factory=dict)
    status: str = "FRESH"  # FRESH, STALE, BUILDING, FAILED


class GraphRAGQueryTrace(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_id: str
    query_text: str
    query_mode: str  # VECTOR, LOCAL, GLOBAL, DRIFT
    selected_chunk_ids: List[str] = Field(default_factory=list)
    seed_entity_ids: List[str] = Field(default_factory=list)
    expanded_entity_ids: List[str] = Field(default_factory=list)
    selected_relationship_ids: List[str] = Field(default_factory=list)
    selected_community_ids: List[str] = Field(default_factory=list)
    map_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    drift_steps: List[Dict[str, Any]] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    llm_calls: int = 0
    latency_ms: float = 0.0
    refusal_reason: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class GraphRAGQueryResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_id: str
    query_text: str
    query_mode: str
    answer: str
    grounded_citations: List[Dict[str, Any]] = Field(default_factory=list)
    trace: GraphRAGQueryTrace
