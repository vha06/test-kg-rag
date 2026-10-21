import re
import time
from typing import List, Optional, Dict, Any
from storage.models import RetrievalResult
from generation.models import (
    LLMConfig,
    LegalAnswerResponse,
    ABComparisonResult,
    ThreeWayComparisonResult,
    CitationItem,
)
from generation.context_builder import LegalContextBuilder
from generation.prompts import LegalPrompts
from generation.llm_client import MultiProviderLLMClient


class LegalAnswerGenerator:
    """
    Bộ điều phối sinh câu trả lời pháp lý từ LLM, hậu kiểm trích dẫn nội tuyến,
    xử lý từ chối khi thiếu căn cứ và điều phối chạy đối chiếu A/B hoặc so sánh 3 cơ chế.
    """

    def __init__(self, default_config: Optional[LLMConfig] = None):
        self.default_config = default_config or LLMConfig()

    def generate_answer(
        self,
        query: str,
        results: List[RetrievalResult],
        reference_date: str = "2024-01-01",
        llm_config: Optional[LLMConfig] = None,
    ) -> LegalAnswerResponse:
        cfg = llm_config or self.default_config
        start_time = time.perf_counter()

        # 1. Trường hợp rỗng (Zero-Retrieval Refusal)
        if not results:
            latency = (time.perf_counter() - start_time) * 1000
            return LegalAnswerResponse(
                query=query,
                reference_date=reference_date,
                answer_text="⚠️ **[KHÔNG_ĐỦ_CĂN_CỨ]** Hệ thống không tìm thấy điều khoản pháp lý nào phù hợp trong kho dữ liệu để trả lời câu hỏi này.",
                has_sufficient_evidence=False,
                temporal_warnings=[],
                citations=[],
                formatted_context="",
                system_prompt="",
                model_used=f"{cfg.provider}:{cfg.model_name}",
                latency_ms=latency,
            )

        # 2. Xây dựng Context Bundle & Cảnh báo hiệu lực
        formatted_context, citations, temporal_warnings = LegalContextBuilder.build_context(
            results=results,
            reference_date=reference_date,
        )

        system_prompt = LegalPrompts.SYSTEM_PROMPT
        user_prompt = LegalPrompts.build_user_prompt(
            query=query,
            formatted_context=formatted_context,
            reference_date=reference_date,
        )

        # 3. Gọi LLM
        client = MultiProviderLLMClient(cfg)
        error_msg: Optional[str] = None
        thought_process: Optional[str] = None
        try:
            raw_answer, thought_process = client.generate_with_thought(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            raw_answer = f"⚠️ Lỗi khi gọi mô hình LLM ({cfg.provider} - {cfg.model_name}): {str(e)}"
            error_msg = str(e)

        latency = (time.perf_counter() - start_time) * 1000

        # 4. Hậu kiểm Trích dẫn (Citation Verification)
        cited_indices = set()
        matches = re.findall(r"\[#(\d+)(?:[^\]]*)\]", raw_answer)
        for m in matches:
            try:
                c_idx = int(m)
                cited_indices.add(c_idx)
            except ValueError:
                pass

        for c in citations:
            if c.ref_index in cited_indices:
                c.is_cited_in_answer = True

        # 5. Kiểm tra cờ từ chối
        # Từ chối nếu câu trả lời bắt đầu bằng thẻ [KHÔNG_ĐỦ_CĂN_CỨ] hoặc chứa thẻ từ chối mà không có trích dẫn hợp lệ
        has_sufficient = True
        stripped_ans = raw_answer.strip()
        refusal_tags = ["[KHÔNG_ĐỦ_CĂN_CỨ]", "[KHONG_DU_CAN_CU]"]
        has_tag = any(tag in raw_answer for tag in refusal_tags)
        starts_with_tag = any(stripped_ans.startswith(tag) for tag in refusal_tags)

        if starts_with_tag or (has_tag and len(cited_indices) == 0):
            has_sufficient = False

        return LegalAnswerResponse(
            query=query,
            reference_date=reference_date,
            answer_text=raw_answer,
            thought_process=thought_process,
            has_sufficient_evidence=has_sufficient,
            temporal_warnings=temporal_warnings,
            citations=citations,
            formatted_context=formatted_context,
            system_prompt=system_prompt,
            model_used=f"{cfg.provider}:{cfg.model_name}",
            latency_ms=latency,
            error_message=error_msg,
        )

    def generate_ab_comparison(
        self,
        query: str,
        seed_results: List[RetrievalResult],
        expanded_results: List[RetrievalResult],
        reference_date: str = "2024-01-01",
        llm_config: Optional[LLMConfig] = None,
    ) -> ABComparisonResult:
        """
        Chạy song song/tuần tự 2 luồng:
        - Luồng A: Không có Knowledge Graph (Chỉ đưa Top-K Seed Chunks vào LLM)
        - Luồng B: Có Knowledge Graph (Đưa đầy đủ Seed + KG Expanded Nodes vào LLM)
        """
        cfg = llm_config or self.default_config

        ans_without_kg = self.generate_answer(
            query=query,
            results=seed_results,
            reference_date=reference_date,
            llm_config=cfg,
        )

        ans_with_kg = self.generate_answer(
            query=query,
            results=expanded_results,
            reference_date=reference_date,
            llm_config=cfg,
        )

        return ABComparisonResult(
            query=query,
            reference_date=reference_date,
            without_kg=ans_without_kg,
            with_kg=ans_with_kg,
        )

    def generate_three_way_comparison(
        self,
        query: str,
        local_results: List[RetrievalResult],
        reference_date: str = "2024-01-01",
        top_communities: int = 3,
        min_relevance_score: float = 40.0,
        global_engine: Optional[Any] = None,
        drift_engine: Optional[Any] = None,
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ) -> ThreeWayComparisonResult:
        """
        Thực thi đối chiếu song song cả 3 cơ chế tìm kiếm GraphRAG:
        1. Local Search: Hạt nhân Seed Retrieval + KG 1-2 hops expansion
        2. Global Search: Top-down Map-Reduce qua các Community Summaries
        3. DRIFT Search: Dynamic reasoning + Flexible traversal đa tầng
        """
        cfg = llm_config or self.default_config

        # 1. Chạy Local Search
        local_res = self.generate_answer(
            query=query,
            results=local_results,
            reference_date=reference_date,
            llm_config=cfg,
        )

        # 2. Chạy Global Search
        if global_engine is None:
            from retrieval.global_search import GlobalSearchEngine
            global_engine = GlobalSearchEngine(default_config=cfg)

        global_res = global_engine.search(
            query=query,
            reference_date=reference_date,
            top_communities=top_communities,
            min_relevance_score=min_relevance_score,
            llm_config=cfg,
        )

        # 3. Chạy DRIFT Search
        if drift_engine is None:
            from retrieval.drift_search import DRIFTSearchEngine
            drift_engine = DRIFTSearchEngine(default_config=cfg)

        drift_res = drift_engine.search(
            query=query,
            reference_date=reference_date,
            top_communities=top_communities,
            min_relevance_score=min_relevance_score,
            llm_config=cfg,
        )

        # 4. Xây dựng ma trận so sánh định tính và định lượng
        local_cited = sum(1 for c in local_res.citations if c.is_cited_in_answer)
        drift_cited = sum(1 for c in drift_res.citations if c.is_cited_in_answer)

        comparison_matrix = [
            {
                "Tiêu chí": "1. Phạm vi & Góc nhìn (Scope & Perspective)",
                "Local Search (Cục bộ)": "Vi mô (Micro) - Tập trung vào thực thể và các điều khoản cụ thể",
                "Global Search (Toàn cục)": "Vĩ mô (Macro) - Sensemaking toàn cảnh qua các Cụm cộng đồng pháp lý",
                "DRIFT Search (Lai thích ứng)": "Đa tầng (Multi-tier) - Phân rã câu hỏi, kết nối từ Vĩ mô xuống Vi mô",
            },
            {
                "Tiêu chí": "2. Cơ chế duyệt đồ thị (Graph Traversal)",
                "Local Search (Cục bộ)": "Duyệt BFS 1-2 bước (k-hop) từ các Seed điều khoản lân cận",
                "Global Search (Toàn cục)": "Map-Reduce qua Hierarchical Community Summaries (Louvain)",
                "DRIFT Search (Lai thích ứng)": "Cây tìm kiếm linh hoạt (Search Tree): Macro Communities + Micro Hops",
            },
            {
                "Tiêu chí": "3. Thời gian phản hồi (Latency)",
                "Local Search (Cục bộ)": f"{local_res.latency_ms:.0f} ms",
                "Global Search (Toàn cục)": f"{global_res.latency_ms:.0f} ms",
                "DRIFT Search (Lai thích ứng)": f"{drift_res.latency_ms:.0f} ms",
            },
            {
                "Tiêu chí": "4. Số lượng nguồn / Cụm căn cứ",
                "Local Search (Cục bộ)": f"{len(local_res.citations)} điều khoản",
                "Global Search (Toàn cục)": f"{len(global_res.community_insights)} cụm cộng đồng pháp lý",
                "DRIFT Search (Lai thích ứng)": f"{len(drift_res.citations)} điều khoản + {len(drift_res.visited_communities)} cụm",
            },
            {
                "Tiêu chí": "5. Số lượng trích dẫn nội tuyến thực tế",
                "Local Search (Cục bộ)": f"{local_cited}/{len(local_res.citations)} trích dẫn",
                "Global Search (Toàn cục)": f"{len(global_res.community_insights)} cụm được phân tích tổng hợp",
                "DRIFT Search (Lai thích ứng)": f"{drift_cited}/{len(drift_res.citations)} trích dẫn nội tuyến",
            },
            {
                "Tiêu chí": "6. Trường hợp sử dụng tối ưu (Best For)",
                "Local Search (Cục bộ)": "Tra cứu điều khoản cụ thể, mức phạt, đối tượng, lịch sử sửa đổi 1 văn bản",
                "Global Search (Toàn cục)": "Câu hỏi bức tranh lớn: tổng quan chính sách, xu hướng điều chỉnh của hệ sinh thái",
                "DRIFT Search (Lai thích ứng)": "Câu hỏi phức tạp nhiều khía cạnh, cần cả góc nhìn bao quát lẫn điều khoản căn cứ",
            },
        ]

        return ThreeWayComparisonResult(
            query=query,
            reference_date=reference_date,
            local_search=local_res,
            global_search=global_res,
            drift_search=drift_res,
            comparison_matrix=comparison_matrix,
        )

