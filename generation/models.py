from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class LLMConfig(BaseModel):
    provider: str = "gemini"  # "gemini" | "openai_compatible" | "ollama" | "mock"
    model_name: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 8192


class CitationItem(BaseModel):
    ref_index: int                  # 1, 2, 3...
    provision_id: str
    doc_number: str
    doc_title: str
    provision_title: str
    structural_path: str
    retrieval_method: str           # BM25, DENSE, HYBRID, KG_EXPANSION
    expansion_path: Optional[str] = None
    evidence_text: Optional[str] = None
    validity_status: str = "VALID"  # VALID, INVALID_EXPIRED, INVALID_NOT_YET_EFFECTIVE, VALID_PRIOR_TO_AMENDMENT
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None
    is_cited_in_answer: bool = False


class LegalAnswerResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    query: str
    reference_date: str
    answer_text: str
    thought_process: Optional[str] = None
    has_sufficient_evidence: bool = True
    temporal_warnings: List[str] = Field(default_factory=list)
    citations: List[Any] = Field(default_factory=list)
    formatted_context: str = ""
    system_prompt: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
    error_message: Optional[str] = None


class ABComparisonResult(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    query: str
    reference_date: str
    without_kg: Any
    with_kg: Any


class LegalCommunity(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    community_id: str
    title: str
    summary: str
    doc_ids: List[str] = Field(default_factory=list)
    doc_numbers: List[str] = Field(default_factory=list)
    doc_titles: List[str] = Field(default_factory=list)
    provision_count: int = 0
    relations_count: Dict[str, int] = Field(default_factory=dict)
    key_provisions: List[Dict[str, str]] = Field(default_factory=list)
    level: int = 0
    weight: float = 1.0


class CommunityInsight(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    community_id: str
    community_title: str
    relevance_score: float = 0.0  # Thang điểm 0 - 100
    intermediate_summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    doc_numbers: List[str] = Field(default_factory=list)


class GlobalSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    query: str
    reference_date: str
    answer_text: str
    thought_process: Optional[str] = None
    community_insights: List[CommunityInsight] = Field(default_factory=list)
    selected_communities: List[Dict[str, Any]] = Field(default_factory=list)
    map_reduce_stats: Dict[str, Any] = Field(default_factory=dict)
    model_used: str = ""
    latency_ms: float = 0.0
    error_message: Optional[str] = None


class DRIFTTrajectory(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    trajectory_id: str
    sub_query: str
    entry_node_or_comm: str
    strategy_type: str  # "MACRO_COMMUNITY", "MICRO_PROVISION", "RELATION_HOP"
    path_description: str
    evidence_found: str
    score: float = 0.0


class DRIFTSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    query: str
    reference_date: str
    answer_text: str
    thought_process: Optional[str] = None
    decomposed_queries: List[str] = Field(default_factory=list)
    trajectories: List[DRIFTTrajectory] = Field(default_factory=list)
    visited_communities: List[str] = Field(default_factory=list)
    citations: List[Any] = Field(default_factory=list)
    latency_ms: float = 0.0
    model_used: str = ""
    error_message: Optional[str] = None


class ThreeWayComparisonResult(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    query: str
    reference_date: str
    local_search: LegalAnswerResponse
    global_search: GlobalSearchResponse
    drift_search: DRIFTSearchResponse
    comparison_matrix: List[Dict[str, Any]] = Field(default_factory=list)

