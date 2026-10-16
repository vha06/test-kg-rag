from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import re
import numpy as np
from storage.models import LegalDocumentModel, ChunkModel


class BaseChunker(ABC):
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    @staticmethod
    def count_tokens(text: str) -> int:
        # Simple, fast and consistent word-based token count for Vietnamese text
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return len(tokens)

    @abstractmethod
    def chunk_documents(self, documents: List[LegalDocumentModel]) -> List[ChunkModel]:
        pass

    @classmethod
    def compute_stats(cls, chunks: List[ChunkModel]) -> Dict[str, Any]:
        if not chunks:
            return {
                "total_chunks": 0,
                "token_min": 0,
                "token_median": 0,
                "token_p90": 0,
                "token_max": 0,
                "too_short_count": 0,  # < 20 tokens
                "too_long_count": 0,   # > 512 tokens
                "crosses_boundary_count": 0,
                "crosses_boundary_ratio": 0.0,
            }

        tokens = [c.token_count for c in chunks]
        crosses = sum(1 for c in chunks if c.crosses_boundary)
        too_short = sum(1 for t in tokens if t < 20)
        too_long = sum(1 for t in tokens if t > 512)

        return {
            "total_chunks": len(chunks),
            "token_min": int(np.min(tokens)),
            "token_median": float(np.median(tokens)),
            "token_p90": float(np.percentile(tokens, 90)),
            "token_max": int(np.max(tokens)),
            "too_short_count": too_short,
            "too_long_count": too_long,
            "crosses_boundary_count": crosses,
            "crosses_boundary_ratio": round(crosses / len(chunks), 4),
        }
