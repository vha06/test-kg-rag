import os
import pickle
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from storage.models import ChunkModel, RetrievalResult
from retrieval.tokenizer import LegalTokenizer


class BM25Retriever:
    def __init__(self, strategy: str = "A1_article", indices_dir: str = "artifacts/indices"):
        self.strategy = strategy
        self.indices_dir = indices_dir
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        os.makedirs(indices_dir, exist_ok=True)

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = [
            LegalTokenizer.tokenize(c["content"]) for c in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.save_index()

    def save_index(self):
        cache_path = os.path.join(self.indices_dir, f"bm25_{self.strategy}.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "corpus_tokens": self.corpus_tokens,
                "bm25": self.bm25,
            }, f)

    def load_index(self) -> bool:
        cache_path = os.path.join(self.indices_dir, f"bm25_{self.strategy}.pkl")
        if not os.path.exists(cache_path):
            return False
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.corpus_tokens = data["corpus_tokens"]
            self.bm25 = data["bm25"]
        return True

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        if self.bm25 is None:
            if not self.load_index():
                raise RuntimeError(f"BM25 index for {self.strategy} is not built.")

        query_tokens = LegalTokenizer.tokenize(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = doc_scores.argsort()[::-1][: max(top_k * 3, 50)]

        # Group by provision_id to get best chunk per provision
        seen_provisions = set()
        results: List[RetrievalResult] = []

        rank = 1
        for idx in top_indices:
            score = float(doc_scores[idx])
            if score <= 0:
                continue

            chunk = self.chunks[idx]
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
                retrieval_method="BM25",
                source_url=chunk.get("source_url"),
                effective_start=chunk.get("effective_start"),
                effective_end=chunk.get("effective_end"),
                status=chunk.get("status", "ACTIVE"),
            ))
            rank += 1
            if len(results) >= top_k:
                break

        return results
