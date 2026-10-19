import os
import hashlib
from typing import List, Union, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from pipeline.interfaces import ILocalModelAdapter


class EmbeddingAdapter(ILocalModelAdapter):
    def __init__(
        self,
        model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
        device: str = "cuda",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Auto-detect CUDA if requested
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available on this machine. Falling back to CPU for embeddings.")
            self.device = "cpu"
        else:
            self.device = device

        print(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            fallback = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            print(f"Could not load '{self.model_name}' ({e}). Attempting fallback to '{fallback}'...")
            self.model = SentenceTransformer(fallback, device=self.device)
            self.model_name = fallback

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "adapter_type": "SentenceTransformer",
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.get_dimension(),
        }


class DeterministicHashEmbeddingAdapter(ILocalModelAdapter):
    """
    Lightweight, deterministic embedding adapter using n-gram/word hashing.
    Runs 100% offline with zero GPU and zero dependencies on model weights.
    Ideal for unit tests, CI pipelines, and resource-constrained environments.
    """

    def __init__(self, dimension: int = 64, salt: str = "platform_hash"):
        self.dimension = dimension
        self.salt = salt

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = []
        for t in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = t.lower().split()
            if not words:
                vectors.append(vec)
                continue
            for w in words:
                h = int(hashlib.md5((self.salt + w).encode("utf-8")).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "adapter_type": "DeterministicHashEmbeddingAdapter",
            "dimension": self.dimension,
            "salt": self.salt,
        }

