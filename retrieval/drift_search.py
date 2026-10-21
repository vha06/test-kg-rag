import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set

from storage.models import RetrievalResult
from generation.models import (
    LLMConfig,
    CitationItem,
    DRIFTTrajectory,
    DRIFTSearchResponse,
)
from generation.context_builder import LegalContextBuilder
from generation.prompts import LegalPrompts
from generation.llm_client import MultiProviderLLMClient
from retrieval.hybrid_retriever import HybridRetriever
from knowledge_graph.expander import GraphExpander
from knowledge_graph.community import LegalCommunityDetector
from temporal.validity_filter import TemporalValidityFilter


class DRIFTSearchEngine:
    """
    Cơ chế DRIFT Search (Dynamic Reasoning and Inference with Flexible Traversal)
    theo chuẩn Microsoft GraphRAG:
    - Thu hẹp khoảng cách giữa Local Search (quá hẹp) và Global Search (quá tốn kém, mất chi tiết).
    - Pha 1 (Dynamic Reasoning): LLM tự động phân rã câu hỏi gốc thành 2-3 câu hỏi phụ (sub-queries).
    - Pha 2 (Flexible Traversal): Duyệt cây phân tầng thích ứng (Search Tree / Trajectories):
      + Quỹ đạo Vĩ mô: Ánh xạ câu hỏi vào các Cụm cộng đồng (Macro Communities).
      + Quỹ đạo Vi mô: Với từng câu hỏi phụ, kích hoạt Local Search + k-Hop Graph Expansion để tìm điều khoản chính xác.
      + Cắt tỉa (Pruning) và gom tụ các căn cứ chứng minh.
    - Pha 3 (Adaptive Synthesis): Tổng hợp câu trả lời đa tầng vừa có cấu trúc phân tích theo khía cạnh,
      vừa có trích dẫn nội tuyến [#X] chính xác đến từng điều khoản pháp luật.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        expander: Optional[GraphExpander] = None,
        temporal_filter: Optional[TemporalValidityFilter] = None,
        community_detector: Optional[LegalCommunityDetector] = None,
        default_config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        self.retriever = retriever or HybridRetriever(strategy="A3_structure")
        self.expander = expander or GraphExpander()
        self.temporal_filter = temporal_filter or TemporalValidityFilter()
        self.community_detector = community_detector or LegalCommunityDetector()
        self.default_config = default_config or LLMConfig()

    def search(
        self,
        query: str,
        reference_date: str = "2024-01-01",
        top_communities: int = 3,
        min_relevance_score: float = 40.0,
        max_hops: int = 1,
        top_k_per_subquery: int = 2,
        filter_invalid: bool = False,
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ) -> DRIFTSearchResponse:
        cfg = llm_config or self.default_config
        client = MultiProviderLLMClient(cfg)
        start_time = time.perf_counter()

        # ---------------------------------------------------------------------
        # 1. DYNAMIC REASONING: PHÂN RÃ CÂU HỎI (QUERY DECOMPOSITION)
        # ---------------------------------------------------------------------
        decompose_prompt = LegalPrompts.build_drift_decompose_prompt(query)
        try:
            raw_decomp, decomp_thought = client.generate_with_thought(
                system_prompt=LegalPrompts.DRIFT_DECOMPOSE_SYSTEM_PROMPT,
                user_prompt=decompose_prompt,
            )
        except Exception:
            raw_decomp = ""
            decomp_thought = None

        decomposed_queries = []
        for line in raw_decomp.split("\n"):
            line_clean = line.strip().lstrip("-*123456789. ")
            if line_clean and len(line_clean) > 8 and "?" in line_clean:
                decomposed_queries.append(line_clean)

        # Fallback nếu mô hình không trả về danh sách phân rã
        if not decomposed_queries:
            decomposed_queries = [
                f"Quy định về phạm vi, tiêu chuẩn và điều kiện liên quan đến: {query}",
                f"Các quy định sửa đổi, bổ sung và hướng dẫn thi hành liên quan đến: {query}",
            ]

        # Giới hạn tối đa 3 câu hỏi phụ để tối ưu latency
        decomposed_queries = decomposed_queries[:3]

        # ---------------------------------------------------------------------
        # 2. FLEXIBLE TRAVERSAL: DUYỆT CÂY ĐỒ THỊ ĐA TẦNG (SEARCH TREE & TRAJECTORIES)
        # ---------------------------------------------------------------------
        trajectories: List[DRIFTTrajectory] = []
        visited_communities: List[str] = []
        collected_provisions: Dict[str, RetrievalResult] = {}

        # 2.1. Trajectory Vĩ mô (Top-Down Macro Communities - Kế thừa từ Global Search)
        matched_comms = self.community_detector.find_relevant_communities(query=query, top_k=top_communities)
        # Lọc theo ngưỡng điểm liên quan tối thiểu (min_relevance_score)
        filtered_comms = [(c, sc) for c, sc in matched_comms if (sc * 100) >= min_relevance_score]
        if not filtered_comms and matched_comms:
            filtered_comms = [matched_comms[0]]  # Luôn giữ lại ít nhất 1 cụm liên quan nhất

        for c_idx, (comm, sc) in enumerate(filtered_comms, 1):
            visited_communities.append(comm.title)
            trajectories.append(
                DRIFTTrajectory(
                    trajectory_id=f"TRAJ_MACRO_{c_idx}",
                    sub_query=query,
                    entry_node_or_comm=comm.community_id,
                    strategy_type="MACRO_COMMUNITY",
                    path_description=f"Ánh xạ vào {comm.title} (Độ khớp: {sc:.2f})",
                    evidence_found=f"Bao gồm các VB: {', '.join(comm.doc_numbers[:3])}. Tóm tắt: {comm.summary[:150]}...",
                    score=sc,
                )
            )

        # 2.2. Trajectory Vi mô (Bottom-Up theo từng Sub-Query - Kế thừa từ Local Search)
        for q_idx, sub_q in enumerate(decomposed_queries, 1):
            raw_seeds = self.retriever.search(sub_q, top_k=top_k_per_subquery)
            valid_seeds = self.temporal_filter.filter_and_annotate(
                results=raw_seeds,
                reference_date=reference_date,
                filter_out_invalid=filter_invalid,
            )

            for s in valid_seeds:
                if s.provision_id not in collected_provisions:
                    collected_provisions[s.provision_id] = s
                trajectories.append(
                    DRIFTTrajectory(
                        trajectory_id=f"TRAJ_MICRO_{q_idx}_{s.provision_id}",
                        sub_query=sub_q,
                        entry_node_or_comm=s.provision_id,
                        strategy_type="MICRO_PROVISION",
                        path_description=f"Truy xuất hạt nhân: {s.provision_title} ({s.doc_number})",
                        evidence_found=s.content[:150] + "...",
                        score=s.score,
                    )
                )

            # Mở rộng đồ thị từ các seeds tìm được (Graph Expansion)
            if valid_seeds:
                raw_expanded = self.expander.expand(
                    query=sub_q,
                    seed_results=valid_seeds,
                    max_hops=max_hops,
                    max_expanded_nodes=2,
                )
                valid_expanded = self.temporal_filter.filter_and_annotate(
                    results=raw_expanded,
                    reference_date=reference_date,
                    filter_out_invalid=filter_invalid,
                )
                for exp_item in valid_expanded:
                    if exp_item.retrieval_method == "KG_EXPANSION" and exp_item.provision_id not in collected_provisions:
                        collected_provisions[exp_item.provision_id] = exp_item
                        trajectories.append(
                            DRIFTTrajectory(
                                trajectory_id=f"TRAJ_EXP_{q_idx}_{exp_item.provision_id}",
                                sub_query=sub_q,
                                entry_node_or_comm=exp_item.provision_id,
                                strategy_type="RELATION_HOP",
                                path_description=f"Mắt xích liên kết: {exp_item.expansion_path}",
                                evidence_found=exp_item.evidence_text or exp_item.content[:150],
                                score=exp_item.score,
                            )
                        )

        # ---------------------------------------------------------------------
        # 3. ADAPTIVE CONTEXT & SYNTHESIS
        # ---------------------------------------------------------------------
        active_results = list(collected_provisions.values())

        if not active_results:
            # Fallback nếu không có kết quả từ sub-queries: thử câu hỏi gốc
            fallback_seeds = self.retriever.search(query, top_k=3)
            active_results = self.temporal_filter.filter_and_annotate(
                results=fallback_seeds,
                reference_date=reference_date,
                filter_out_invalid=filter_invalid,
            )

        formatted_context, citations, temporal_warnings = LegalContextBuilder.build_context(
            results=active_results,
            reference_date=reference_date,
        )

        # Tóm tắt các quỹ đạo duyệt thành bản đồ điều hướng cho LLM
        traj_lines = []
        for t in trajectories[:6]:
            traj_lines.append(f"- [{t.strategy_type}] {t.path_description} --> {t.evidence_found[:100]}...")
        trajectory_summary = "\n".join(traj_lines)

        synthesis_prompt = LegalPrompts.build_drift_synthesis_prompt(
            query=query,
            decomposed_queries=decomposed_queries,
            trajectory_summary=trajectory_summary,
            formatted_context=formatted_context,
            reference_date=reference_date,
        )

        try:
            answer_text, synth_thought = client.generate_with_thought(
                system_prompt=LegalPrompts.DRIFT_SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=synthesis_prompt,
            )
        except Exception as e:
            answer_text = f"⚠️ Lỗi khi sinh câu trả lời DRIFT Synthesis: {str(e)}"
            synth_thought = None

        latency = (time.perf_counter() - start_time) * 1000

        # Kiểm chứng trích dẫn nội tuyến
        cited_indices = set()
        matches = re.findall(r"\[#(\d+)(?:[^\]]*)\]", answer_text)
        for m in matches:
            try:
                cited_indices.add(int(m))
            except ValueError:
                pass

        for c in citations:
            if c.ref_index in cited_indices:
                c.is_cited_in_answer = True

        # Ghép suy luận nếu có
        thought_combined = None
        t_parts = []
        if decomp_thought:
            t_parts.append("--- [PHÂN RÃ CÂU HỎI / DECOMPOSITION] ---\n" + decomp_thought)
        if synth_thought:
            t_parts.append("--- [DRIFT SYNTHESIS REASONING] ---\n" + synth_thought)
        if t_parts:
            thought_combined = "\n\n".join(t_parts)

        return DRIFTSearchResponse(
            query=query,
            reference_date=reference_date,
            answer_text=answer_text,
            thought_process=thought_combined,
            decomposed_queries=decomposed_queries,
            trajectories=trajectories,
            visited_communities=visited_communities,
            citations=citations,
            latency_ms=latency,
            model_used=f"{cfg.provider}:{cfg.model_name}",
        )
