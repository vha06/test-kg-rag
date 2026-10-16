import os
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingAdapter:
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
