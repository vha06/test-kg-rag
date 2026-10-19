from __future__ import annotations

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field

from storage.db import DatabaseManager
from pipeline.interfaces import ILocalModelAdapter

logger = logging.getLogger(__name__)


class CanonicalRetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    domain_id: str
    content: str
    score: float = 0.0
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    retrieval_method: str = "HYBRID"
    source_block_ids: List[str] = Field(default_factory=list)
    source_location: Dict[str, Any] = Field(default_factory=dict)
    security_classification: str = "INTERNAL"
    acl: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    profile_id: Optional[str] = None


class MultiDomainIndexManager:
    """
    Manages multi-domain BM25 lexical indices and FAISS dense vector indices,
    partitioned by domain_id and profile_id.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        indices_dir: str = "artifacts/indices/platform",
    ):
        self.db = db_manager or DatabaseManager()
        self.indices_dir = indices_dir
        os.makedirs(indices_dir, exist_ok=True)

        self._bm25_indices: Dict[str, BM25Okapi] = {}
        self._faiss_indices: Dict[str, faiss.IndexFlatIP] = {}
        self._chunks_map: Dict[str, List[Dict[str, Any]]] = {}
        self._embeddings_map: Dict[str, np.ndarray] = {}

    @staticmethod
    def _make_key(domain_id: str, profile_id: Optional[str] = None) -> str:
        return f"{domain_id}::{profile_id}" if profile_id else domain_id

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def build_index(
        self,
        domain_id: str,
        profile_id: Optional[str] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        embedding_adapter: Optional[ILocalModelAdapter] = None,
    ) -> bool:
        start_time = time.time()
        key = self._make_key(domain_id, profile_id)

        if chunks is None:
            chunks = self.db.get_platform_chunks(domain_id=domain_id, profile_id=profile_id)

        if not chunks:
            logger.warning(f"No chunks found to index for key: {key}")
            self._chunks_map[key] = []
            return False

        self._chunks_map[key] = chunks

        # 1. Build BM25 Index
        corpus_tokens = [self.tokenize(c["content"]) for c in chunks]
        self._bm25_indices[key] = BM25Okapi(corpus_tokens)

        # 2. Build FAISS Index
        if embedding_adapter is not None:
            texts = [c["content"] for c in chunks]
            vectors = embedding_adapter.encode(texts)
            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)
            self._faiss_indices[key] = index
            self._embeddings_map[key] = vectors

        elapsed = time.time() - start_time
        logger.info(f"Built multi-domain index '{key}' with {len(chunks)} chunks in {elapsed:.3f}s")
        return True

    def get_index_data(
        self,
        domain_id: str,
        profile_id: Optional[str] = None,
    ) -> Tuple[Optional[BM25Okapi], Optional[faiss.IndexFlatIP], List[Dict[str, Any]]]:
        key = self._make_key(domain_id, profile_id)
        return (
            self._bm25_indices.get(key),
            self._faiss_indices.get(key),
            self._chunks_map.get(key, []),
        )

    def has_index(self, domain_id: str, profile_id: Optional[str] = None) -> bool:
        key = self._make_key(domain_id, profile_id)
        return key in self._chunks_map and len(self._chunks_map[key]) > 0


class MultiDomainHybridRetriever:
    """
    Performs hybrid BM25 + Dense vector retrieval over specified domain partitions
    with strict enforcement of security_classification and ACL permissions.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        index_manager: MultiDomainIndexManager,
        embedding_adapter: ILocalModelAdapter,
    ):
        self.db = db_manager
        self.index_manager = index_manager
        self.embedding_adapter = embedding_adapter

    def retrieve(
        self,
        query: str,
        domain_ids: List[str],
        top_k: int = 5,
        dense_weight: float = 0.5,
        allowed_classifications: Optional[List[str]] = None,
        user_roles: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[CanonicalRetrievalResult]:
        if not domain_ids:
            return []

        all_candidates: List[CanonicalRetrievalResult] = []
        normalized_allowed_cls: Optional[Set[str]] = (
            {c.upper() for c in allowed_classifications}
            if allowed_classifications is not None
            else None
        )
        user_roles_set: Set[str] = {r.lower() for r in (user_roles or ["public"])}

        query_tokens = MultiDomainIndexManager.tokenize(query)
        query_vector = self.embedding_adapter.encode([query])

        for dom in domain_ids:
            # Ensure index is loaded
            if not self.index_manager.has_index(dom):
                self.index_manager.build_index(
                    domain_id=dom,
                    embedding_adapter=self.embedding_adapter,
                )

            bm25, faiss_idx, chunks = self.index_manager.get_index_data(dom)
            if not chunks:
                continue

            num_chunks = len(chunks)

            # BM25 scores
            if bm25 and query_tokens:
                raw_bm25 = bm25.get_scores(query_tokens)
                max_b = raw_bm25.max() if len(raw_bm25) > 0 and raw_bm25.max() > 0 else 1.0
                bm25_scores = raw_bm25 / max_b
            else:
                bm25_scores = np.zeros(num_chunks, dtype=np.float32)

            # Dense scores
            if faiss_idx is not None:
                d_scores, d_indices = faiss_idx.search(query_vector, num_chunks)
                dense_scores = np.zeros(num_chunks, dtype=np.float32)
                for rank, idx in enumerate(d_indices[0]):
                    if 0 <= idx < num_chunks:
                        dense_scores[idx] = max(0.0, float(d_scores[0][rank]))
            else:
                dense_scores = np.zeros(num_chunks, dtype=np.float32)

            # Compute combined hybrid score and apply ACL / classification filters
            for i, c in enumerate(chunks):
                # 1. Security Classification Check
                chunk_cls = c.get("security_classification", "INTERNAL").upper()
                if normalized_allowed_cls is not None and chunk_cls not in normalized_allowed_cls:
                    continue

                # 2. ACL Check
                chunk_acl = c.get("acl", {})
                allowed_roles = [r.lower() for r in chunk_acl.get("allowed_roles", ["public"])]
                allowed_users = [u.lower() for u in chunk_acl.get("allowed_users", [])]

                role_permitted = bool(user_roles_set.intersection(set(allowed_roles)))
                user_permitted = (user_id is not None) and (user_id.lower() in allowed_users)
                is_public = "public" in allowed_roles

                if not (role_permitted or user_permitted or is_public):
                    continue

                b_score = float(bm25_scores[i])
                d_score = float(dense_scores[i])
                hybrid_score = (1.0 - dense_weight) * b_score + dense_weight * d_score

                candidate = CanonicalRetrievalResult(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    domain_id=c["domain_id"],
                    content=c["content"],
                    score=hybrid_score,
                    dense_score=d_score,
                    bm25_score=b_score,
                    retrieval_method="HYBRID",
                    source_block_ids=c.get("source_block_ids", []),
                    source_location=c.get("source_location", {}),
                    security_classification=chunk_cls,
                    acl=chunk_acl,
                    metadata=c.get("metadata", {}),
                    profile_id=c.get("profile_id"),
                )
                all_candidates.append(candidate)

        all_candidates.sort(key=lambda x: x.score, reverse=True)
        return all_candidates[:top_k]
