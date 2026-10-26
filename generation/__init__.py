from generation.config_loader import load_keys_config

# Tự động nạp cấu hình API và biến môi trường từ key.txt hoặc .env
load_keys_config()

from generation.models import LLMConfig, CitationItem, LegalAnswerResponse, ABComparisonResult
from generation.context_builder import LegalContextBuilder
from generation.prompts import LegalPrompts
from generation.llm_client import MultiProviderLLMClient
from generation.answer_generator import LegalAnswerGenerator

__all__ = [
    "load_keys_config",
    "LLMConfig",
    "CitationItem",
    "LegalAnswerResponse",
    "ABComparisonResult",
    "LegalContextBuilder",
    "LegalPrompts",
    "MultiProviderLLMClient",
    "LegalAnswerGenerator",
]
