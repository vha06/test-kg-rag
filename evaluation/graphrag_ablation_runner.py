from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional

from storage.db import DatabaseManager
from pipeline.orchestrator import PipelineOrchestrator
from retrieval.graphrag_query_engine import GraphRAGQueryEngine
from retrieval.graphrag_index_manager import GraphRAGIndexManager
from embedding.adapter import DeterministicHashEmbeddingAdapter, EmbeddingAdapter

logger = logging.getLogger(__name__)


class GraphRAGAblationRunner:
    """
    Bộ chạy thực nghiệm đối chứng (Ablation & Benchmark Runner):
    - Đánh giá 4 chế độ truy vấn trên 4 nhóm câu hỏi:
      + Nhóm A: Cục bộ (Single-hop / Factoid)
      + Nhóm B: Đa bước (Multi-hop Reasoning)
      + Nhóm C: Toàn cục (Global Aggregation / Sensemaking)
      + Nhóm D: Không có bằng chứng (Negative Control / Refusal Check)
    - Thí nghiệm E: Đối chứng 2 chiến lược Chunking (Token Baseline vs Recursive Boundary-Aware)
    """

    BENCHMARK_QUESTIONS = [
        # Nhóm A: Cục bộ
        {"group": "A_LOCAL", "query": "Giao thức FastLease mở rộng từ thuật toán nào và cải thiện điều gì?"},
        {"group": "A_LOCAL", "query": "Cán bộ chiến sĩ cần nộp những giấy tờ gì để thanh toán tiền bồi dưỡng nghỉ phép?"},
        # Nhóm B: Đa bước
        {"group": "B_MULTI_HOP", "query": "Cổng Ingress Gateway phụ thuộc vào dịch vụ nào và dữ liệu phiên làm việc lưu ở đâu?"},
        {"group": "B_MULTI_HOP", "query": "Cơ quan nào duyệt chi trả tiền bồi dưỡng và căn cứ vào chứng từ nào?"},
        # Nhóm C: Toàn cục
        {"group": "C_GLOBAL", "query": "Tổng quan toàn bộ quy định về chế độ nghỉ phép và đi lại của cán bộ chiến sĩ."},
        {"group": "C_GLOBAL", "query": "Tóm tắt cấu trúc và các thành phần cốt lõi của nền tảng phân tán FastLease."},
        # Nhóm D: Không có bằng chứng
        {"group": "D_REFUSAL", "query": "Quy trình xin cấp thị thực điện tử cho công dân Hoa Kỳ được quy định thế nào?"},
        {"group": "D_REFUSAL", "query": "Thông số kỹ thuật của bộ xử lý lượng tử Sycamore do Google phát triển là gì?"},
    ]

    def __init__(self, db_manager: DatabaseManager, llm_config: Optional[Any] = None):
        self.db = db_manager
        self.llm_config = llm_config

    def run_benchmark_4_modes(
        self,
        namespace: str = "graphrag_default",
        embedding_adapter: Optional[Any] = None,
        questions: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        adapter = embedding_adapter or DeterministicHashEmbeddingAdapter(dimension=64)
        idx_mgr = GraphRAGIndexManager(db_manager=self.db, embedding_adapter=adapter)
        engine = GraphRAGQueryEngine(db_manager=self.db, index_manager=idx_mgr, llm_config=self.llm_config)

        test_questions = questions or self.BENCHMARK_QUESTIONS
        records: List[Dict[str, Any]] = []

        for q_item in test_questions:
            q_text = q_item["query"]
            q_group = q_item.get("group", "GENERAL")

            for mode in ["VECTOR", "LOCAL", "GLOBAL", "DRIFT"]:
                start_t = time.perf_counter()
                if mode == "VECTOR":
                    res = engine.query_vector_rag(q_text, namespace=namespace)
                elif mode == "LOCAL":
                    res = engine.query_local_search(q_text, namespace=namespace)
                elif mode == "GLOBAL":
                    res = engine.query_global_search(q_text, namespace=namespace)
                else:
                    res = engine.query_drift_search(q_text, namespace=namespace)

                latency_ms = (time.perf_counter() - start_t) * 1000
                is_refusal = (res.answer == GraphRAGQueryEngine.REFUSAL_MESSAGE)

                # Chấm điểm độ tin cậy trích dẫn
                citation_count = len(res.grounded_citations)
                has_citations = citation_count > 0 or is_refusal

                records.append({
                    "group": q_group,
                    "query": q_text,
                    "mode": mode,
                    "latency_ms": round(latency_ms, 2),
                    "llm_calls": res.trace.llm_calls,
                    "citation_count": citation_count,
                    "is_refusal": is_refusal,
                    "stop_reason": res.trace.stop_reason,
                    "answer_preview": res.answer[:180],
                })

        return records

    def run_chunking_ablation(
        self,
        corpus: List[Dict[str, Any]],
        namespace_prefix: str = "ablation",
        embedding_adapter: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Thí nghiệm E: So sánh Chiến lược A (fixed_token) và Chiến lược B (recursive_boundary)
        trên cùng một tập tài liệu và các câu hỏi mẫu.
        """
        adapter = embedding_adapter or DeterministicHashEmbeddingAdapter(dimension=64)
        orchestrator = PipelineOrchestrator(db_manager=self.db)

        # 1. Chạy Strategy A (Fixed Token)
        ns_a = f"{namespace_prefix}_strat_a_token"
        res_a = orchestrator.build_full_graphrag_pipeline(
            documents=corpus,
            namespace=ns_a,
            chunking_strategy="fixed_token",
            chunk_size=250,
            chunk_overlap=30,
            extractor_name="deterministic_general",
            embedding_adapter=adapter,
        )

        # 2. Chạy Strategy B (Recursive Boundary Aware)
        ns_b = f"{namespace_prefix}_strat_b_boundary"
        res_b = orchestrator.build_full_graphrag_pipeline(
            documents=corpus,
            namespace=ns_b,
            chunking_strategy="recursive_boundary",
            chunk_size=250,
            chunk_overlap=30,
            extractor_name="deterministic_general",
            embedding_adapter=adapter,
        )

        # Đánh giá truy vấn trên cả 2 chiến lược
        sample_q = self.BENCHMARK_QUESTIONS[:4]
        results_a = self.run_benchmark_4_modes(namespace=ns_a, embedding_adapter=adapter, questions=sample_q)
        results_b = self.run_benchmark_4_modes(namespace=ns_b, embedding_adapter=adapter, questions=sample_q)

        avg_lat_a = sum(r["latency_ms"] for r in results_a) / len(results_a) if results_a else 0
        avg_lat_b = sum(r["latency_ms"] for r in results_b) / len(results_b) if results_b else 0

        avg_cit_a = sum(r["citation_count"] for r in results_a) / len(results_a) if results_a else 0
        avg_cit_b = sum(r["citation_count"] for r in results_b) / len(results_b) if results_b else 0

        return {
            "strategy_a_token": {
                "chunk_count": res_a.get("chunk_count", 0),
                "entity_count": res_a.get("entity_count", 0),
                "community_count": res_a.get("community_count", 0),
                "avg_latency_ms": round(avg_lat_a, 2),
                "avg_citations": round(avg_cit_a, 2),
            },
            "strategy_b_boundary": {
                "chunk_count": res_b.get("chunk_count", 0),
                "entity_count": res_b.get("entity_count", 0),
                "community_count": res_b.get("community_count", 0),
                "avg_latency_ms": round(avg_lat_b, 2),
                "avg_citations": round(avg_cit_b, 2),
            },
            "benchmark_details_a": results_a,
            "benchmark_details_b": results_b,
        }
