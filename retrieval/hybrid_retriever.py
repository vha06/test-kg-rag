from typing import List, Dict, Any, Optional
from storage.models import RetrievalResult
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from embedding.adapter import EmbeddingAdapter


class HybridRetriever:
    def __init__(
        self,
        strategy: str = "A1_article",
        indices_dir: str = "artifacts/indices",
        adapter: Optional[EmbeddingAdapter] = None,
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        self.strategy = strategy
        self.bm25_retriever = BM25Retriever(strategy=strategy, indices_dir=indices_dir)
        self.dense_retriever = DenseRetriever(strategy=strategy, indices_dir=indices_dir, adapter=adapter)
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: Optional[float] = None,
        dense_weight: Optional[float] = None,
    ) -> List[RetrievalResult]:
        w_bm25 = bm25_weight if bm25_weight is not None else self.bm25_weight
        w_dense = dense_weight if dense_weight is not None else self.dense_weight

        # Retrieve candidates from both systems
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 3)
        dense_results = self.dense_retriever.search(query, top_k=top_k * 3)

        rrf_scores: Dict[str, float] = {}
        item_by_prov: Dict[str, RetrievalResult] = {}

        for r in bm25_results:
            rrf_scores[r.provision_id] = rrf_scores.get(r.provision_id, 0.0) + (w_bm25 / (self.rrf_k + r.rank))
            item_by_prov[r.provision_id] = r

        for r in dense_results:
            rrf_scores[r.provision_id] = rrf_scores.get(r.provision_id, 0.0) + (w_dense / (self.rrf_k + r.rank))
            if r.provision_id not in item_by_prov:
                item_by_prov[r.provision_id] = r

        sorted_provs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results: List[RetrievalResult] = []
        rank = 1
        for prov_id, rrf_score in sorted_provs[:top_k]:
            base = item_by_prov[prov_id]
            final_results.append(RetrievalResult(
                chunk_id=base.chunk_id,
                provision_id=base.provision_id,
                document_id=base.document_id,
                doc_number=base.doc_number,
                doc_title=base.doc_title,
                provision_title=base.provision_title,
                structural_path=base.structural_path,
                content=base.content,
                score=round(rrf_score, 6),
                rank=rank,
                retrieval_method="HYBRID",
                source_url=base.source_url,
                effective_start=base.effective_start,
                effective_end=base.effective_end,
                status=base.status,
            ))
            rank += 1

        return final_results
