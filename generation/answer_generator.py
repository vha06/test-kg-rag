import re
import time
from typing import List, Optional
from storage.models import RetrievalResult
from generation.models import LLMConfig, LegalAnswerResponse, ABComparisonResult, CitationItem
from generation.context_builder import LegalContextBuilder
from generation.prompts import LegalPrompts
from generation.llm_client import MultiProviderLLMClient


class LegalAnswerGenerator:
    """
    Bộ điều phối sinh câu trả lời pháp lý từ LLM, hậu kiểm trích dẫn nội tuyến,
    xử lý từ chối khi thiếu căn cứ và điều phối chạy đối chiếu A/B.
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

        # 4. Kiểm tra cờ từ chối
        has_sufficient = True
        if "[KHÔNG_ĐỦ_CĂN_CỨ]" in raw_answer or "[KHONG_DU_CAN_CU]" in raw_answer:
            has_sufficient = False

        # 5. Hậu kiểm Trích dẫn (Citation Verification)
        # Bắt các mẫu: [#1], [#2], [#1 - Điều 8...], [#1: Điều...]
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

        # 1. Luồng A: Without KG
        ans_without_kg = self.generate_answer(
            query=query,
            results=seed_results,
            reference_date=reference_date,
            llm_config=cfg,
        )

        # 2. Luồng B: With KG
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
