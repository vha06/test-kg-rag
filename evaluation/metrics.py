import math
from typing import List, Set, Dict, Any
from storage.models import RetrievalResult, BenchmarkItem


class RetrievalMetrics:
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        if not ground_truth_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        hits = len(top_k.intersection(ground_truth_ids))
        return hits / len(ground_truth_ids)

    @staticmethod
    def mrr_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        if not ground_truth_ids:
            return 0.0
        for rank, item_id in enumerate(retrieved_ids[:k], start=1):
            if item_id in ground_truth_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        if not ground_truth_ids:
            return 0.0

        dcg = 0.0
        for rank, item_id in enumerate(retrieved_ids[:k], start=1):
            if item_id in ground_truth_ids:
                dcg += 1.0 / math.log2(rank + 1)

        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(ground_truth_ids), k) + 1))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def provision_hit_rate(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
        if not ground_truth_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        return 1.0 if len(top_k.intersection(ground_truth_ids)) > 0 else 0.0

    @staticmethod
    def kg_precision(results: List[RetrievalResult], ground_truth_ids: Set[str]) -> float:
        kg_items = [r for r in results if r.retrieval_method == "KG_EXPANSION"]
        if not kg_items:
            return 0.0
        hits = sum(1 for r in kg_items if r.provision_id in ground_truth_ids)
        return hits / len(kg_items)

    @classmethod
    def evaluate_query(
        cls,
        results: List[RetrievalResult],
        item: BenchmarkItem,
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, float]:
        gt_ids = set(item.relevant_provision_ids)
        if item.acceptable_provision_ids:
            gt_ids.update(item.acceptable_provision_ids)

        retrieved_ids = [r.provision_id for r in results]

        metrics = {}
        for k in k_values:
            metrics[f"recall@{k}"] = cls.recall_at_k(retrieved_ids, gt_ids, k)

        metrics["mrr@10"] = cls.mrr_at_k(retrieved_ids, gt_ids, 10)
        metrics["ndcg@10"] = cls.ndcg_at_k(retrieved_ids, gt_ids, 10)
        metrics["hit_rate@5"] = cls.provision_hit_rate(retrieved_ids, gt_ids, 5)
        metrics["kg_precision"] = cls.kg_precision(results, gt_ids)
        metrics["total_retrieved"] = float(len(results))
        metrics["total_tokens"] = float(sum(len(r.content.split()) for r in results[:10]))

        return metrics
