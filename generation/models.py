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
