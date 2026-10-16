import os
import time
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from storage.models import BenchmarkItem, RetrievalResult
from storage.db import DatabaseManager
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from knowledge_graph.expander import GraphExpander
from temporal.validity_filter import TemporalValidityFilter
from evaluation.metrics import RetrievalMetrics
from evaluation.benchmark_builder import BenchmarkBuilder


class BenchmarkRunner:
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        indices_dir: str = "artifacts/indices",
        reports_dir: str = "artifacts/reports",
    ):
        self.db = db_manager or DatabaseManager()
        self.indices_dir = indices_dir
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self.expander = GraphExpander()
        self.temporal_filter = TemporalValidityFilter(self.db)

    def load_benchmark(self, benchmark_path: str = "data/processed/benchmark_qa.jsonl") -> List[BenchmarkItem]:
        if not os.path.exists(benchmark_path):
            builder = BenchmarkBuilder(self.db)
            return builder.build_default_benchmark(benchmark_path)

        items = []
        with open(benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(BenchmarkItem.model_validate_json(line.strip()))
        return items

    def run_matrix(
        self,
        benchmark_items: Optional[List[BenchmarkItem]] = None,
        strategies: List[str] = ["A1_article", "A2_fixed", "A3_structure"],
        retrieval_modes: List[str] = ["BM25", "Dense", "Hybrid"],
        kg_modes: List[bool] = [False, True],
        top_k: int = 10,
        sample_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        if benchmark_items is None:
            benchmark_items = self.load_benchmark()

        if sample_limit:
            benchmark_items = benchmark_items[:sample_limit]

        summary_rows = []
        group_rows = []
        full_details = []

        total_configs = len(strategies) * len(retrieval_modes) * len(kg_modes)
        config_idx = 0

        for strat in strategies:
            # Instantiate retrievers for this strategy
            bm25 = BM25Retriever(strategy=strat, indices_dir=self.indices_dir)
            dense = DenseRetriever(strategy=strat, indices_dir=self.indices_dir)
            hybrid = HybridRetriever(strategy=strat, indices_dir=self.indices_dir)

            for mode in retrieval_modes:
                for kg_on in kg_modes:
                    config_idx += 1
                    config_name = f"{strat}__{mode}__KG_{'On' if kg_on else 'Off'}"
                    print(f"[{config_idx}/{total_configs}] Running config: {config_name} on {len(benchmark_items)} queries...")

                    all_query_metrics: List[Dict[str, float]] = []
                    group_metrics_map: Dict[str, List[Dict[str, float]]] = {}
                    t0 = time.time()

                    for item in benchmark_items:
                        # 1. Retrieval
                        if mode == "BM25":
                            seeds = bm25.search(item.question_text, top_k=top_k)
                        elif mode == "Dense":
                            seeds = dense.search(item.question_text, top_k=top_k)
                        else:
                            seeds = hybrid.search(item.question_text, top_k=top_k)

                        # 2. KG Expansion
                        if kg_on:
                            res = self.expander.expand(
                                query=item.question_text,
                                seed_results=seeds,
                                max_hops=1,
                                max_expanded_nodes=5,
                            )
                        else:
                            res = seeds

                        # 3. Temporal Validity
                        res = self.temporal_filter.filter_and_annotate(
                            results=res,
                            reference_date=item.reference_date,
                            filter_out_invalid=False,
                        )

                        # Limit to top_k
                        res = res[:top_k]

                        q_metric = RetrievalMetrics.evaluate_query(res, item)
                        all_query_metrics.append(q_metric)
                        group_metrics_map.setdefault(item.group_id, []).append(q_metric)

                        full_details.append({
                            "config": config_name,
                            "strategy": strat,
                            "mode": mode,
                            "kg_enabled": kg_on,
                            "question_id": item.question_id,
                            "group_id": item.group_id,
                            "metrics": q_metric,
                        })

                    elapsed = time.time() - t0
                    avg_latency_ms = (elapsed / len(benchmark_items)) * 1000.0

                    # Compute overall averages
                    def avg(key: str, m_list: List[Dict[str, float]]) -> float:
                        return float(sum(m[key] for m in m_list) / len(m_list)) if m_list else 0.0

                    summary_rows.append({
                        "Configuration": config_name,
                        "Chunking": strat,
                        "Retrieval": mode,
                        "KG": "On" if kg_on else "Off",
                        "Recall@1": round(avg("recall@1", all_query_metrics), 4),
                        "Recall@3": round(avg("recall@3", all_query_metrics), 4),
                        "Recall@5": round(avg("recall@5", all_query_metrics), 4),
                        "Recall@10": round(avg("recall@10", all_query_metrics), 4),
                        "MRR@10": round(avg("mrr@10", all_query_metrics), 4),
                        "nDCG@10": round(avg("ndcg@10", all_query_metrics), 4),
                        "HitRate@5": round(avg("hit_rate@5", all_query_metrics), 4),
                        "KG_Precision": round(avg("kg_precision", all_query_metrics), 4),
                        "Avg_Tokens": round(avg("total_tokens", all_query_metrics), 1),
                        "Latency_ms": round(avg_latency_ms, 2),
                    })

                    for gid, g_metrics in group_metrics_map.items():
                        group_rows.append({
                            "Configuration": config_name,
                            "Group": gid,
                            "Recall@5": round(avg("recall@5", g_metrics), 4),
                            "MRR@10": round(avg("mrr@10", g_metrics), 4),
                            "HitRate@5": round(avg("hit_rate@5", g_metrics), 4),
                            "KG_Precision": round(avg("kg_precision", g_metrics), 4),
                        })

        summary_df = pd.DataFrame(summary_rows)
        group_df = pd.DataFrame(group_rows)

        summary_csv = os.path.join(self.reports_dir, "benchmark_summary.csv")
        group_csv = os.path.join(self.reports_dir, "benchmark_by_group.csv")
        full_json = os.path.join(self.reports_dir, "benchmark_full.json")

        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        group_df.to_csv(group_csv, index=False, encoding="utf-8-sig")
        with open(full_json, "w", encoding="utf-8") as f:
            json.dump(full_details, f, ensure_ascii=False, indent=2)

        print(f"Benchmark completed! Reports saved to {self.reports_dir}/")
        return {
            "summary_df": summary_df,
            "group_df": group_df,
            "summary_csv": summary_csv,
            "group_csv": group_csv,
        }
