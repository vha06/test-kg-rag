import re
import time
from typing import List, Dict, Any, Optional, Tuple

from generation.models import (
    LLMConfig,
    LegalCommunity,
    CommunityInsight,
    GlobalSearchResponse,
)
from generation.prompts import LegalPrompts
from generation.llm_client import MultiProviderLLMClient
from knowledge_graph.community import LegalCommunityDetector


class GlobalSearchEngine:
    """
    Cơ chế Global Search theo kiến trúc GraphRAG của Microsoft:
    - Tìm kiếm Top-down / Macro Sensemaking trên toàn bộ kho văn bản.
    - Pha 1 (Map Phase): Đánh giá câu hỏi trên các Cụm cộng đồng tri thức pháp lý (Community Summaries)
      để sinh ra các điểm phát hiện trung gian (Intermediate Findings) kèm điểm liên quan.
    - Pha 2 (Reduce Phase): Lọc và tổng hợp các câu trả lời trung gian thành Báo cáo Toàn cảnh Vĩ mô
      (Executive Legal Sensemaking Report) có cấu trúc.
    """

    def __init__(
        self,
        community_detector: Optional[LegalCommunityDetector] = None,
        default_config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        self.community_detector = community_detector or LegalCommunityDetector()
        self.default_config = default_config or LLMConfig()

    def search(
        self,
        query: str,
        reference_date: str = "2024-01-01",
        top_communities: int = 4,
        min_relevance_score: float = 40.0,
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ) -> GlobalSearchResponse:
        cfg = llm_config or self.default_config
        client = MultiProviderLLMClient(cfg)
        start_time = time.perf_counter()

        # 1. Tìm các Cụm cộng đồng tri thức liên quan nhất
        ranked_comms: List[Tuple[LegalCommunity, float]] = self.community_detector.find_relevant_communities(
            query=query,
            top_k=top_communities,
        )

        if not ranked_comms:
            latency = (time.perf_counter() - start_time) * 1000
            return GlobalSearchResponse(
                query=query,
                reference_date=reference_date,
                answer_text="⚠️ Không phát hiện cụm cộng đồng tri thức pháp lý nào phù hợp để phân tích toàn cục.",
                community_insights=[],
                selected_communities=[],
                map_reduce_stats={"communities_evaluated": 0, "communities_reduced": 0},
                model_used=f"{cfg.provider}:{cfg.model_name}",
                latency_ms=latency,
            )

        # 2. MAP PHASE: Đánh giá từng cụm cộng đồng độc lập
        insights: List[CommunityInsight] = []
        raw_map_thoughts: List[str] = []

        for comm, heuristic_score in ranked_comms:
            map_prompt = LegalPrompts.build_global_map_prompt(
                query=query,
                community_title=comm.title,
                community_summary=comm.summary,
                reference_date=reference_date,
            )

            try:
                map_resp, map_thought = client.generate_with_thought(
                    system_prompt=LegalPrompts.GLOBAL_MAP_SYSTEM_PROMPT,
                    user_prompt=map_prompt,
                )
                if map_thought:
                    raw_map_thoughts.append(f"[{comm.community_id}]: {map_thought}")
            except Exception as e:
                map_resp = f"ĐIỂM_LIÊN_QUAN: 50\nCÁC_ĐIỂM_CHÍNH:\n- Lỗi phân tích: {str(e)}\nTÓM_TẮT_ĐÓNG_GÓP: Lỗi gọi mô hình."

            # Phân tích kết quả từ Map Phase
            rel_score = 70.0
            score_match = re.search(r"ĐIỂM_LIÊN_QUAN:\s*(\d+)", map_resp, flags=re.I)
            if score_match:
                try:
                    rel_score = float(score_match.group(1))
                except ValueError:
                    pass
            else:
                # Nếu không tìm thấy thẻ số, kết hợp với điểm heuristic
                rel_score = round(heuristic_score * 100, 1)

            # Trích xuất các gạch đầu dòng
            findings = []
            for line in map_resp.split("\n"):
                line_str = line.strip()
                if line_str.startswith("- ") or line_str.startswith("* "):
                    findings.append(line_str.lstrip("-* ").strip())

            # Trích xuất tóm tắt đóng góp
            summary_match = re.search(r"TÓM_TẮT_ĐÓNG_GÓP:\s*(.+)", map_resp, flags=re.I | re.DOTALL)
            inter_summary = summary_match.group(1).strip() if summary_match else map_resp[:300]

            insights.append(
                CommunityInsight(
                    community_id=comm.community_id,
                    community_title=comm.title,
                    relevance_score=rel_score,
                    intermediate_summary=inter_summary,
                    key_findings=findings,
                    doc_numbers=comm.doc_numbers,
                )
            )

        # 3. REDUCE PHASE: Lọc các cụm điểm cao và tổng hợp vĩ mô
        # Sắp xếp theo điểm liên quan giảm dần
        insights.sort(key=lambda x: x.relevance_score, reverse=True)

        filtered_insights = [ins for ins in insights if ins.relevance_score >= min_relevance_score]
        if not filtered_insights:
            filtered_insights = insights[:2]  # Giữ ít nhất 2 cụm nếu tất cả đều dưới ngưỡng

        # Định dạng văn bản kết quả trung gian để gửi vào Reduce prompt
        insights_text_blocks = []
        for idx, ins in enumerate(filtered_insights, start=1):
            findings_bullets = "\n".join(f"  * {f}" for f in ins.key_findings) if ins.key_findings else "  * (Không có điểm tách biệt)"
            doc_str = ", ".join(ins.doc_numbers[:4]) if ins.doc_numbers else "Văn bản QPPL liên quan"
            block = (
                f"### [CỤM #{idx}]: {ins.community_title} (Độ liên quan: {ins.relevance_score:.0f}/100)\n"
                f"- **Văn bản trọng tâm:** {doc_str}\n"
                f"- **Các điểm quy định chính:**\n{findings_bullets}\n"
                f"- **Nhận định từ cụm:** {ins.intermediate_summary}"
            )
            insights_text_blocks.append(block)

        combined_insights_text = "\n\n".join(insights_text_blocks)

        reduce_prompt = LegalPrompts.build_global_reduce_prompt(
            query=query,
            community_insights_text=combined_insights_text,
            reference_date=reference_date,
        )

        reduce_thought: Optional[str] = None
        try:
            reduce_answer, reduce_thought = client.generate_with_thought(
                system_prompt=LegalPrompts.GLOBAL_REDUCE_SYSTEM_PROMPT,
                user_prompt=reduce_prompt,
            )
        except Exception as e:
            reduce_answer = f"⚠️ Lỗi khi tổng hợp báo cáo vĩ mô (Reduce Phase): {str(e)}"

        latency = (time.perf_counter() - start_time) * 1000

        # Ghép thought process nếu có
        final_thought = None
        all_thoughts = []
        if raw_map_thoughts:
            all_thoughts.append("--- [MAP PHASE THOUGHTS] ---\n" + "\n\n".join(raw_map_thoughts[:2]))
        if reduce_thought:
            all_thoughts.append("--- [REDUCE PHASE SYNTHESIS] ---\n" + reduce_thought)
        if all_thoughts:
            final_thought = "\n\n".join(all_thoughts)

        selected_comms_meta = [
            {
                "community_id": c.community_id,
                "title": c.title,
                "doc_count": len(c.doc_ids),
                "provision_count": c.provision_count,
                "heuristic_score": sc,
            }
            for c, sc in ranked_comms
        ]

        return GlobalSearchResponse(
            query=query,
            reference_date=reference_date,
            answer_text=reduce_answer,
            thought_process=final_thought,
            community_insights=insights,
            selected_communities=selected_comms_meta,
            map_reduce_stats={
                "communities_evaluated": len(ranked_comms),
                "communities_reduced": len(filtered_insights),
                "avg_relevance_score": sum(i.relevance_score for i in filtered_insights) / max(1, len(filtered_insights)),
            },
            model_used=f"{cfg.provider}:{cfg.model_name}",
            latency_ms=latency,
        )
