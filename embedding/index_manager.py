import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import faiss
import numpy as np
from storage.models import ChunkModel
from embedding.adapter import EmbeddingAdapter


class FAISSIndexManager:
    def __init__(
        self,
        strategy: str = "A1_article",
        indices_dir: str = "artifacts/indices",
        adapter: Optional[EmbeddingAdapter] = None,
    ):
        self.strategy = strategy
        self.indices_dir = indices_dir
        self.adapter = adapter
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        os.makedirs(indices_dir, exist_ok=True)

    def _index_path(self) -> str:
        return os.path.join(self.indices_dir, f"faiss_{self.strategy}.index")

    def _meta_path(self) -> str:
        return os.path.join(self.indices_dir, f"meta_{self.strategy}.json")

    def build_index(self, chunks: List[Dict[str, Any]]) -> float:
        if self.adapter is None:
            self.adapter = EmbeddingAdapter()

        self.chunk_metadata = chunks
        texts = [c["content"] for c in chunks]

        print(f"Generating embeddings for {len(texts)} chunks ({self.strategy})...")
        t0 = time.time()
        embeddings = self.adapter.encode(texts)
        build_time = time.time() - t0

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Save to disk
        faiss.write_index(self.index, self._index_path())
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(self.chunk_metadata, f, ensure_ascii=False)

        print(f"Saved FAISS index to {self._index_path()} (build time: {build_time:.2f}s)")
        return build_time

    def load_index(self) -> bool:
        idx_path = self._index_path()
        meta_path = self._meta_path()
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            return False

        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunk_metadata = json.load(f)
        return True

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None:
            if not self.load_index():
                raise RuntimeError(f"FAISS index for {self.strategy} is not built.")

        if self.adapter is None:
            self.adapter = EmbeddingAdapter()

        q_vec = self.adapter.encode([query])
        scores, indices = self.index.search(q_vec, max(top_k * 3, 50))

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1 or idx >= len(self.chunk_metadata):
                continue
            results.append((self.chunk_metadata[idx], float(score)))

        return results
