from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ProvisionVersionModel(BaseModel):
    version_id: str
    provision_id: str
    version_num: int
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None
    status: str = "ACTIVE"
    amending_provision_id: Optional[str] = None
    content: str
    version_note: Optional[str] = None


class ProvisionModel(BaseModel):
    provision_id: str
    document_id: str
    parent_id: Optional[str] = None
    level: str  # ARTICLE, CLAUSE, POINT
    article_number: Optional[str] = None
    clause_number: Optional[str] = None
    point_letter: Optional[str] = None
    title: str = ""
    structural_path: str = ""
    content: str = ""
    mapc: Optional[str] = None
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None
    status: str = "CÓ_HIỆU_LỰC_THI_HÀNH"
    source_url: Optional[str] = None
    is_external_stub: int = 0
    versions: List[ProvisionVersionModel] = Field(default_factory=list)


class LegalDocumentModel(BaseModel):
    document_id: str
    doc_number: str
    title: str
    doc_type: str
    issuer: str
    issue_date: Optional[str] = None
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None
    status: str = "ACTIVE"
    source_url: Optional[str] = None
    content_hash: str = ""
    provisions: List[ProvisionModel] = Field(default_factory=list)


class ChunkModel(BaseModel):
    chunk_id: str
    strategy: str  # A1_article, A2_fixed, A3_structure
    document_id: str
    provision_id: str
    source_provision_ids: List[str] = Field(default_factory=list)
    structural_path: str = ""
    content: str = ""
    token_count: int = 0
    crosses_boundary: bool = False
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None
    status: str = "ACTIVE"
    source_url: Optional[str] = None
    content_hash: str = ""


class KGEdgeModel(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relation_type: str  # CONTAINS, REFERENCES, DETAILS, AMENDS, SUPPLEMENTS, REPEALS
    evidence_text: str = ""
    source_url: Optional[str] = None
    effective_date: Optional[str] = None
    detection_method: str = "OFFICIAL"
    verified: int = 1


class RetrievalResult(BaseModel):
    chunk_id: str
    provision_id: str
    document_id: str
    doc_number: str
    doc_title: str
    provision_title: str
    structural_path: str
    content: str
    score: float
    rank: int
    retrieval_method: str  # BM25, DENSE, HYBRID, KG_EXPANSION
    source_url: Optional[str] = None
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None
    status: str = "ACTIVE"
    expansion_path: Optional[str] = None
    evidence_text: Optional[str] = None
    validity_status: str = "VALID"  # VALID, INVALID_NOT_YET_EFFECTIVE, INVALID_EXPIRED, UNDETERMINED


class BenchmarkItem(BaseModel):
    question_id: str
    group_id: str  # direct_article, paraphrased_semantic, cross_reference_kg, multi_provision, temporal_amendment
    group_name: str
    question_text: str
    relevant_provision_ids: List[str]
    relevant_document_ids: List[str]
    acceptable_provision_ids: List[str] = Field(default_factory=list)
    reference_date: Optional[str] = None
    expected_relations: List[str] = Field(default_factory=list)
    expected_kg_targets: List[str] = Field(default_factory=list)
    rationale: str = ""
