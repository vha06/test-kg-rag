from typing import List, Optional, Set
from storage.models import RetrievalResult
from embedding.index_manager import FAISSIndexManager
from embedding.adapter import EmbeddingAdapter


class DenseRetriever:
    def __init__(
        self,
        strategy: str = "A1_article",
        indices_dir: str = "artifacts/indices",
        adapter: Optional[EmbeddingAdapter] = None,
    ):
        self.strategy = strategy
        self.index_manager = FAISSIndexManager(
            strategy=strategy,
            indices_dir=indices_dir,
            adapter=adapter,
        )

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        raw_matches = self.index_manager.search(query, top_k=top_k)

        seen_provisions: Set[str] = set()
        results: List[RetrievalResult] = []
        rank = 1

        for chunk, score in raw_matches:
            prov_id = chunk["provision_id"]
            if prov_id in seen_provisions:
                continue
            seen_provisions.add(prov_id)

            results.append(RetrievalResult(
                chunk_id=chunk["chunk_id"],
                provision_id=prov_id,
                document_id=chunk["document_id"],
                doc_number=chunk.get("doc_number", ""),
                doc_title=chunk.get("doc_title", ""),
                provision_title=chunk.get("provision_title", ""),
                structural_path=chunk.get("structural_path", ""),
                content=chunk["content"],
                score=round(score, 4),
                rank=rank,
                retrieval_method="DENSE",
                source_url=chunk.get("source_url"),
                effective_start=chunk.get("effective_start"),
                effective_end=chunk.get("effective_end"),
                status=chunk.get("status", "ACTIVE"),
            ))
            rank += 1
            if len(results) >= top_k:
                break

        return results
