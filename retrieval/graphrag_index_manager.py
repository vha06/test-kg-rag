from __future__ import annotations

import re
import os
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from storage.db import DatabaseManager
from pipeline.interfaces import ILocalModelAdapter
from core.schema import (
    CanonicalChunk,
    CanonicalEntity,
    CanonicalCommunityReport,
    GraphRAGIndexManifest,
)

logger = logging.getLogger(__name__)


class SingleCollectionIndex:
    """
    Quản lý 1 cặp chỉ mục (Dense FAISS IndexFlatIP + Lexical BM25Okapi)
    cho một tập dữ liệu cụ thể (TextUnits, Entities, hoặc CommunityReports).
    """

    def __init__(self, name: str):
        self.name = name
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.records: List[Dict[str, Any]] = []
        self.dimension: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def build(
        self,
        records: List[Dict[str, Any]],
        texts: List[str],
        embedding_adapter: ILocalModelAdapter,
    ) -> bool:
        if not records or not texts:
            self.records = []
            self.bm25 = None
            self.faiss_index = None
            return False

        self.records = records

        # 1. Lexical BM25
        corpus_tokens = [self.tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(corpus_tokens)

        # 2. Dense FAISS
        vectors = embedding_adapter.encode(texts)
        if len(vectors) > 0:
            self.dimension = vectors.shape[1]
            # Normalize for cosine similarity via Inner Product
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(self.dimension)
            index.add(vectors)
            self.faiss_index = index
        return True

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        if not self.records:
            return []

        num_records = len(self.records)
        q_tokens = self.tokenize(query)

        # BM25 scores
        if self.bm25 and q_tokens:
            raw_bm25 = self.bm25.get_scores(q_tokens)
            max_b = raw_bm25.max() if len(raw_bm25) > 0 and raw_bm25.max() > 0 else 1.0
            bm25_scores = raw_bm25 / max_b
        else:
            bm25_scores = np.zeros(num_records, dtype=np.float32)

        # Dense scores
        if self.faiss_index is not None and query_vector is not None:
            q_norm = query_vector.copy()
            faiss.normalize_L2(q_norm)
            d_scores, d_indices = self.faiss_index.search(q_norm, num_records)
            dense_scores = np.zeros(num_records, dtype=np.float32)
            for rank, idx in enumerate(d_indices[0]):
                if 0 <= idx < num_records:
                    dense_scores[idx] = max(0.0, float(d_scores[0][rank]))
        else:
            dense_scores = np.zeros(num_records, dtype=np.float32)

        results = []
        for i, rec in enumerate(self.records):
            b_s = float(bm25_scores[i])
            d_s = float(dense_scores[i])
            hybrid_score = (1.0 - dense_weight) * b_s + dense_weight * d_s
            item = dict(rec)
            item["score"] = hybrid_score
            item["dense_score"] = d_s
            item["bm25_score"] = b_s
            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class GraphRAGIndexManager:
    """
    Quản lý 3 không gian vector độc lập theo thiết kế Microsoft GraphRAG:
    1. text_units: Lập chỉ mục trên nội dung CanonicalChunk
    2. entities: Lập chỉ mục trên tên và mô tả ngữ nghĩa của CanonicalEntity
    3. community_reports: Lập chỉ mục trên tiêu đề, tóm tắt và toàn văn báo cáo CanonicalCommunityReport
    Đồng thời quản lý Index Manifest để phát hiện dữ liệu cũ (stale index invalidation).
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embedding_adapter: Optional[ILocalModelAdapter] = None,
    ):
        self.db = db_manager or DatabaseManager()
        self.embedding_adapter = embedding_adapter

        # 3 chỉ mục vector độc lập theo từng namespace
        self._indices: Dict[str, Dict[str, SingleCollectionIndex]] = {}

    def _get_namespace_indices(self, namespace: str) -> Dict[str, SingleCollectionIndex]:
        if namespace not in self._indices:
            self._indices[namespace] = {
                "text_units": SingleCollectionIndex(f"{namespace}::text_units"),
                "entities": SingleCollectionIndex(f"{namespace}::entities"),
                "community_reports": SingleCollectionIndex(f"{namespace}::community_reports"),
            }
        return self._indices[namespace]

    @staticmethod
    def compute_corpus_fingerprint(doc_checksums: List[str]) -> str:
        sorted_chk = sorted([c for c in doc_checksums if c])
        return hashlib.sha256("::".join(sorted_chk).encode("utf-8")).hexdigest()

    def build_all_indices(
        self,
        namespace: str,
        chunks: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        reports: List[Dict[str, Any]],
        embedding_adapter: Optional[ILocalModelAdapter] = None,
        manifest_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        adapter = embedding_adapter or self.embedding_adapter
        if adapter is None:
            from embedding.adapter import DeterministicHashEmbeddingAdapter
            adapter = DeterministicHashEmbeddingAdapter(dimension=64)
            self.embedding_adapter = adapter

        ns_indices = self._get_namespace_indices(namespace)

        # 1. Index 1: TextUnits (Chunks)
        chunk_texts = [c.get("content", "") for c in chunks]
        ns_indices["text_units"].build(chunks, chunk_texts, adapter)

        # 2. Index 2: Entities (Name, Type, Description, Aliases)
        entity_texts = [
            f"{e.get('canonical_name', '')} ({e.get('entity_type', 'ENTITY')}): {e.get('description', '')}. Bí danh: {', '.join(e.get('aliases', []))}"
            for e in entities
        ]
        ns_indices["entities"].build(entities, entity_texts, adapter)

        # 3. Index 3: Community Reports (Title, Summary, Content)
        report_texts = [
            f"{r.get('title', '')}\n{r.get('summary', '')}\n{r.get('full_content', '')}"
            for r in reports
        ]
        ns_indices["community_reports"].build(reports, report_texts, adapter)

        # 4. Save Index Manifest
        if manifest_config:
            doc_checksums = manifest_config.get("doc_checksums", [])
            fingerprint = self.compute_corpus_fingerprint(doc_checksums)
            run_id = f"RUN_{namespace}_{int(time.time())}"
            manifest = GraphRAGIndexManifest(
                run_id=run_id,
                namespace=namespace,
                corpus_fingerprint=fingerprint,
                chunking_strategy=manifest_config.get("chunking_strategy", "recursive_boundary"),
                chunk_size=manifest_config.get("chunk_size", 300),
                chunk_overlap=manifest_config.get("chunk_overlap", 50),
                extractor_model=manifest_config.get("extractor_model", "mock"),
                extraction_prompt_version=manifest_config.get("extraction_prompt_version", "1.0.0"),
                embedding_model=manifest_config.get("embedding_model", "sentence-transformers"),
                community_algorithm=manifest_config.get("community_algorithm", "louvain_hierarchical"),
                max_community_levels=manifest_config.get("max_community_levels", 3),
                artifact_counts={
                    "chunks": len(chunks),
                    "entities": len(entities),
                    "reports": len(reports),
                },
                status="FRESH",
            )
            self.db.save_graphrag_manifest(manifest)

        return True

    def is_namespace_loaded(self, namespace: str) -> bool:
        if namespace not in self._indices:
            return False
        ns_idx = self._indices[namespace]
        return bool(
            (ns_idx.get("text_units") and len(ns_idx["text_units"].records) > 0)
            or (ns_idx.get("entities") and len(ns_idx["entities"].records) > 0)
            or (ns_idx.get("community_reports") and len(ns_idx["community_reports"].records) > 0)
        )

    def ensure_indices_loaded(self, namespace: str = "graphrag_default") -> bool:
        """
        Tự động nạp chunks, entities và community reports từ SQLite và khởi tạo
        các chỉ mục vector FAISS và BM25 trong bộ nhớ RAM nếu namespace chưa được nạp.
        """
        if self.is_namespace_loaded(namespace):
            return True

        if not self.db:
            return False

        # 1. Chunks
        chunks = self.db.get_platform_chunks(domain_id=namespace)
        if not chunks:
            all_chunks = self.db.get_platform_chunks()
            chunks = [c for c in all_chunks if c.get("domain_id") == namespace]
            if not chunks and all_chunks:
                chunks = all_chunks

        # 2. Entities
        entities = self.db.get_platform_entities(namespace=namespace)
        if not entities:
            entities = self.db.get_platform_entities(graph_namespace=namespace)
        if not entities:
            entities = self.db.get_platform_entities(graph_namespace=f"ns_{namespace}")
        if not entities:
            entities = self.db.get_platform_entities(domain_id=namespace)
        if not entities:
            all_ents = self.db.get_platform_entities()
            if all_ents:
                entities = all_ents

        # 3. Community Reports
        reports = self.db.get_graphrag_community_reports(namespace=namespace)
        if not reports and namespace != "graphrag_default":
            reports = self.db.get_graphrag_community_reports(namespace="graphrag_default")

        if not chunks and not entities and not reports:
            return False

        return self.build_all_indices(
            namespace=namespace,
            chunks=chunks,
            entities=entities,
            reports=reports,
            embedding_adapter=self.embedding_adapter,
        )

    def check_freshness(
        self,
        namespace: str,
        current_checksums: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        latest = self.db.get_latest_graphrag_manifest(namespace)
        if not latest:
            return "STALE"
        current_fp = self.compute_corpus_fingerprint(current_checksums)
        if latest.get("corpus_fingerprint") != current_fp:
            return "STALE"
        if config:
            if config.get("chunking_strategy") and latest.get("chunking_strategy") != config.get("chunking_strategy"):
                return "STALE"
            if config.get("chunk_size") and latest.get("chunk_size") != config.get("chunk_size"):
                return "STALE"
        return "FRESH"

    def search_text_units(
        self,
        query: str,
        namespace: str = "graphrag_default",
        top_k: int = 5,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        self.ensure_indices_loaded(namespace)
        ns_indices = self._get_namespace_indices(namespace)
        q_vec = self.embedding_adapter.encode([query]) if self.embedding_adapter else None
        return ns_indices["text_units"].search(query, q_vec, top_k=top_k, dense_weight=dense_weight)

    def search_entities(
        self,
        query: str,
        namespace: str = "graphrag_default",
        top_k: int = 5,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        self.ensure_indices_loaded(namespace)
        ns_indices = self._get_namespace_indices(namespace)
        q_vec = self.embedding_adapter.encode([query]) if self.embedding_adapter else None
        return ns_indices["entities"].search(query, q_vec, top_k=top_k, dense_weight=dense_weight)

    def search_community_reports(
        self,
        query: str,
        namespace: str = "graphrag_default",
        level: Optional[int] = None,
        top_k: int = 5,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        self.ensure_indices_loaded(namespace)
        ns_indices = self._get_namespace_indices(namespace)
        q_vec = self.embedding_adapter.encode([query]) if self.embedding_adapter else None
        results = ns_indices["community_reports"].search(query, q_vec, top_k=top_k * 2, dense_weight=dense_weight)
        if level is not None:
            results = [r for r in results if r.get("level") == level]
        return results[:top_k]
