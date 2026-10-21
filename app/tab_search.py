import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from typing import Optional, List, Dict, Any

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from knowledge_graph.expander import GraphExpander
from knowledge_graph.community import LegalCommunityDetector
from knowledge_graph.visualizer import build_pyvis_html
from temporal.validity_filter import TemporalValidityFilter
from storage.db import DatabaseManager
from storage.models import RetrievalResult
from generation.models import (
    LLMConfig,
    LegalAnswerResponse,
    ABComparisonResult,
    GlobalSearchResponse,
    DRIFTSearchResponse,
    ThreeWayComparisonResult,
)
import importlib
import generation.answer_generator
import retrieval.drift_search
import retrieval.global_search

# Tự động nạp lại module nếu backend code có cập nhật mà Streamlit server chưa restart
importlib.reload(generation.answer_generator)
importlib.reload(retrieval.drift_search)
importlib.reload(retrieval.global_search)

from generation.answer_generator import LegalAnswerGenerator
from retrieval.drift_search import DRIFTSearchEngine
from retrieval.global_search import GlobalSearchEngine

PRESET_QUERIES = [
    "Quy định về phạm vi điều chỉnh của Luật An ninh quốc gia số 32/2004/QH11?",
    "Người dân tự giác tham gia bảo vệ an ninh mà bị thiệt hại tài sản hoặc thương tật thì được trợ cấp, đền bù ra sao?",
    "Các quy định về điều kiện, tiêu chuẩn và trang phục của lực lượng bảo vệ cơ quan, doanh nghiệp?",
    "Quy chế điều hành và kiểm soát phương tiện, người xuất cảnh nhập cảnh tại các cửa khẩu biên giới đất liền?",
    "Quy định tại Nghị định số 112/2014/NĐ-CP được sửa đổi bổ sung bởi Nghị định số 34/2023/NĐ-CP vào ngày 31/07/2023 như thế nào?",
]

SEARCH_MODES = [
    "🎯 1. Local Search (Cục bộ: Seed Retrieval + k-Hop KG Expansion)",
    "🌐 2. Global Search (Toàn cục: Map-Reduce qua Community Summaries)",
    "🧭 3. DRIFT Search (Lai thích ứng: Dynamic Reasoning & Flexible Traversal)",
    "⚖️ 4. So sánh đối chiếu 3 cơ chế (Local vs Global vs DRIFT)",
    "🔬 5. Chỉ truy xuất (Local Retrieval Only - Không LLM)",
    "🆚 6. So sánh đối chiếu A/B (Không KG vs Có KG)",
]


def render_search_tab(
    db: DatabaseManager,
    expander: GraphExpander,
    temporal_filter: TemporalValidityFilter,
    community_detector: Optional[LegalCommunityDetector] = None,
    llm_config: Optional[LLMConfig] = None,
):
    st.header("🔍 Tab 1: GraphRAG Legal Search & Comparative Laboratory")
    st.markdown(
        "Trải nghiệm và so sánh đối chiếu **3 cơ chế tìm kiếm GraphRAG** "
        "(**Local Search, Global Search, DRIFT Search**) trên đồ thị tri thức quy phạm pháp luật Việt Nam."
    )

    detector = community_detector or LegalCommunityDetector(db_manager=db)

    # 1. Lựa chọn Chế độ thực thi
    exec_mode = st.radio(
        "Chọn cơ chế tìm kiếm GraphRAG:",
        SEARCH_MODES,
        horizontal=False,
        index=0,
    )

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_preset = st.selectbox(
            "Chọn câu hỏi mẫu thử nghiệm:",
            ["-- Nhập câu hỏi tự do --"] + PRESET_QUERIES,
            help="Các câu hỏi mẫu bao quát từ tra cứu điều khoản cụ thể đến tổng hợp đa văn bản.",
        )
        if selected_preset != "-- Nhập câu hỏi tự do --":
            query = st.text_area("Câu hỏi tra cứu:", value=selected_preset, height=85)
        else:
            query = st.text_area("Câu hỏi tra cứu:", value="", placeholder="Nhập câu hỏi pháp lý tiếng Việt...", height=85)

    with col2:
        if "2. Global Search" in exec_mode:
            st.markdown("##### 🌐 Cấu hình Global Search (Toàn cục)")
            top_comms = st.slider("Số Cụm cộng đồng đánh giá (Map Phase):", min_value=2, max_value=6, value=4)
            min_score = st.slider("Ngưỡng điểm liên quan tối thiểu (0-100):", min_value=20, max_value=80, value=40, step=5)
            strategy = "A3_structure"
            mode = "Hybrid"
            top_k = 5
        elif "3. DRIFT Search" in exec_mode or "4. So sánh đối chiếu 3 cơ chế" in exec_mode:
            st.markdown("##### 🧭 Cấu hình Đa Tầng (Macro + Micro)")
            tab_macro_cfg, tab_micro_cfg = st.tabs(["🌐 Tầng Vĩ mô (Cụm)", "🎯 Tầng Vi mô (Hạt nhân)"])
            with tab_macro_cfg:
                top_comms = st.slider("Số Cụm cộng đồng (Macro Communities):", min_value=1, max_value=6, value=3, help="Số lượng cụm chủ đề pháp lý lớn được duyệt ở tầng trên")
                min_score = st.slider("Ngưỡng điểm liên quan cụm (0-100):", min_value=10, max_value=80, value=40, step=5, help="Chỉ mở rộng các cụm có độ tương đồng vượt ngưỡng")
            with tab_micro_cfg:
                strategy = st.selectbox(
                    "Chiến lược Chunking:",
                    ["A3_structure", "A1_article", "A2_fixed"],
                    help="A1: Cấp Điều | A2: Cửa sổ Token cố định | A3: Cấu trúc pháp lý Điều/Khoản/Điểm",
                )
                mode = st.selectbox("Phương thức Retrieval:", ["Hybrid", "BM25", "Dense"])
                top_k = st.slider("Hạt nhân Top-K (cho mỗi sub-query):", min_value=1, max_value=10, value=5)
        else:
            strategy = st.selectbox(
                "Chiến lược Chunking:",
                ["A3_structure", "A1_article", "A2_fixed"],
                help="A1: Cấp Điều | A2: Cửa sổ Token cố định | A3: Cấu trúc pháp lý Điều/Khoản/Điểm",
            )
            mode = st.selectbox("Phương thức Retrieval:", ["Hybrid", "BM25", "Dense"])
            top_k = st.slider("Số lượng kết quả hạt nhân (Top-K):", min_value=1, max_value=20, value=5)
            top_comms = 3
            min_score = 40.0

    # Cấu hình phụ trợ
    with st.expander("⚙️ Cấu hình Knowledge Graph & Hiệu lực Thời gian", expanded=False):
        c_kg1, c_kg2, c_kg3 = st.columns(3)
        with c_kg1:
            kg_enabled = st.checkbox("Bật Knowledge Graph Expansion", value=True)
            max_hops = st.selectbox("Độ sâu Graph (Hops):", [1, 2], index=0)
        with c_kg2:
            allowed_rels = st.multiselect(
                "Loại quan hệ mở rộng:",
                ["AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "REFERENCES", "CONTAINS"],
                default=["AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "REFERENCES", "CONTAINS"],
                help="Bao gồm CONTAINS để trực quan hóa Văn bản QPPL cha và các Khoản thuộc Điều.",
            )
        with c_kg3:
            ref_date_input = st.text_input("Ngày tham chiếu hiệu lực (YYYY-MM-DD):", value="2024-01-01")
            filter_invalid = st.checkbox("Ẩn các điều khoản chưa/hết hiệu lực", value=False)

    if st.button("🚀 Khởi Chạy Tìm Kiếm", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Vui lòng nhập câu hỏi tra cứu.")
            return

        generator = LegalAnswerGenerator(llm_config)

        # Hàm helper lấy seed + expanded
        def get_local_active_results():
            if mode == "BM25":
                retriever = BM25Retriever(strategy=strategy)
            elif mode == "Dense":
                retriever = DenseRetriever(strategy=strategy)
            else:
                retriever = HybridRetriever(strategy=strategy)

            raw_seeds = retriever.search(query, top_k=top_k)
            seeds = temporal_filter.filter_and_annotate(
                results=raw_seeds,
                reference_date=ref_date_input,
                filter_out_invalid=filter_invalid,
            )

            if kg_enabled:
                raw_expanded = expander.expand(
                    query=query,
                    seed_results=seeds,
                    max_hops=max_hops,
                    seed_limit=top_k,
                    max_expanded_nodes=5,
                    allowed_relations=allowed_rels,
                )
                expanded = temporal_filter.filter_and_annotate(
                    results=raw_expanded,
                    reference_date=ref_date_input,
                    filter_out_invalid=filter_invalid,
                )
            else:
                expanded = seeds
            return seeds, expanded

        # =====================================================================
        # 1. CHẾ ĐỘ: LOCAL SEARCH (CỤC BỘ)
        # =====================================================================
        if "1. Local Search" in exec_mode:
            with st.spinner("Đang chạy Local Search (Truy xuất hạt nhân + KG Expansion 1-2 bước)..."):
                seeds, expanded = get_local_active_results()
                active_results = expanded if kg_enabled else seeds
                answer_resp = generator.generate_answer(
                    query=query,
                    results=active_results,
                    reference_date=ref_date_input,
                    llm_config=llm_config,
                )

            st.divider()
            st.subheader("🎯 Kết Quả Local Search (Entity-Centric / Cục bộ)")
            st.caption("Cơ chế tập trung vào thực thể và các điều khoản cụ thể, mở rộng lân cận để tìm kiếm bằng chứng trực tiếp.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mô hình", answer_resp.model_used)
            m2.metric("Thời gian phản hồi", f"{answer_resp.latency_ms:.0f} ms")
            suff_str = "✅ Đủ căn cứ" if answer_resp.has_sufficient_evidence else "⚠️ Thiếu căn cứ"
            m3.metric("Độ tin cậy căn cứ", suff_str)
            m4.metric("Số nguồn trích dẫn", f"{sum(1 for c in answer_resp.citations if c.is_cited_in_answer)}/{len(answer_resp.citations)}")

            if answer_resp.temporal_warnings:
                st.warning("⚠️ **Lưu ý về Hiệu lực Thời gian & Sửa đổi:**\n\n" + "\n\n".join(answer_resp.temporal_warnings))

            _render_thought_process(answer_resp)

            with st.container(border=True):
                st.markdown(answer_resp.answer_text)

            _render_citations_table(answer_resp.citations)
            _render_pyvis_and_provisions(expander, active_results, kg_enabled, max_hops, allowed_rels)
            _render_debug_bundle(answer_resp)

        # =====================================================================
        # 2. CHẾ ĐỘ: GLOBAL SEARCH (TOÀN CỤC)
        # =====================================================================
        elif "2. Global Search" in exec_mode:
            with st.spinner("Đang chạy Global Search (Map-Reduce qua các Cụm cộng đồng tri thức)..."):
                global_eng = GlobalSearchEngine(community_detector=detector, default_config=llm_config)
                global_resp = global_eng.search(
                    query=query,
                    reference_date=ref_date_input,
                    top_communities=top_comms,
                    min_relevance_score=float(min_score),
                    llm_config=llm_config,
                )

            st.divider()
            st.subheader("🌐 Kết Quả Global Search (Macro Sensemaking / Toàn cục)")
            st.caption("Cơ chế giải quyết bài toán vĩ mô qua thuật toán Map-Reduce trên Hierarchical Community Summaries.")

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Mô hình", global_resp.model_used)
            g2.metric("Thời gian phản hồi", f"{global_resp.latency_ms:.0f} ms")
            g3.metric("Cụm đánh giá (Map)", f"{global_resp.map_reduce_stats.get('communities_evaluated', 0)} cụm")
            g4.metric("Cụm tổng hợp (Reduce)", f"{global_resp.map_reduce_stats.get('communities_reduced', 0)} cụm")

            _render_thought_process(global_resp)

            with st.container(border=True):
                st.markdown(global_resp.answer_text)

            # Chi tiết các Cụm cộng đồng được đánh giá ở Map Phase
            if global_resp.community_insights:
                with st.expander(f"🗺️ Chi tiết phân tích Map Phase qua {len(global_resp.community_insights)} Cụm cộng đồng pháp lý", expanded=True):
                    for idx, ins in enumerate(global_resp.community_insights, start=1):
                        with st.container():
                            badge_color = "green" if ins.relevance_score >= 60 else "orange" if ins.relevance_score >= 40 else "gray"
                            st.markdown(f"#### Cụm #{idx}: {ins.community_title} `:{badge_color}[Điểm liên quan: {ins.relevance_score:.0f}/100]`")
                            if ins.doc_numbers:
                                st.caption(f"**Văn bản nòng cốt:** {', '.join(ins.doc_numbers[:4])}")
                            st.markdown(f"**Nhận định trung gian:** {ins.intermediate_summary}")
                            if ins.key_findings:
                                st.markdown("**Các điểm phát hiện chính:**")
                                for f in ins.key_findings:
                                    st.markdown(f"- {f}")
                            st.divider()

        # =====================================================================
        # 3. CHẾ ĐỘ: DRIFT SEARCH (LAI THÍCH ỨNG)
        # =====================================================================
        elif "3. DRIFT Search" in exec_mode:
            with st.spinner("Đang chạy DRIFT Search (Dynamic reasoning + Flexible hierarchical traversal)..."):
                if mode == "BM25":
                    drift_retriever = BM25Retriever(strategy=strategy)
                elif mode == "Dense":
                    drift_retriever = DenseRetriever(strategy=strategy)
                else:
                    drift_retriever = HybridRetriever(strategy=strategy)

                drift_eng = DRIFTSearchEngine(
                    retriever=drift_retriever,
                    expander=expander,
                    temporal_filter=temporal_filter,
                    community_detector=detector,
                    default_config=llm_config,
                )
                drift_resp = drift_eng.search(
                    query=query,
                    reference_date=ref_date_input,
                    top_communities=top_comms,
                    min_relevance_score=float(min_score),
                    max_hops=max_hops,
                    top_k_per_subquery=top_k,
                    filter_invalid=filter_invalid,
                    llm_config=llm_config,
                )

            st.divider()
            st.subheader("🧭 Kết Quả DRIFT Search (Dynamic Reasoning & Flexible Traversal)")
            st.caption("Cơ chế lai thế hệ mới: Tự động phân rã câu hỏi, duyệt cây đồ thị từ Vĩ mô (Cụm) xuống Vi mô (Điều khoản).")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Mô hình", drift_resp.model_used)
            d2.metric("Thời gian phản hồi", f"{drift_resp.latency_ms:.0f} ms")
            d3.metric("Câu hỏi phụ phân rã", f"{len(drift_resp.decomposed_queries)} sub-queries")
            d4.metric("Quỹ đạo duyệt (Trajectories)", f"{len(drift_resp.trajectories)} nhánh")

            _render_thought_process(drift_resp)

            with st.container(border=True):
                st.markdown(drift_resp.answer_text)

            # Bản đồ phân rã câu hỏi & Quỹ đạo duyệt đồ thị
            with st.expander("🌳 Bản đồ Suy luận Phân tầng & Quỹ đạo Duyệt (DRIFT Search Tree)", expanded=True):
                st.markdown("**1. Các khía cạnh câu hỏi phụ được phân rã tự động:**")
                for q_idx, sq in enumerate(drift_resp.decomposed_queries, 1):
                    st.markdown(f"- **Sub-Query {q_idx}:** {sq}")

                st.markdown("**2. Các Quỹ đạo Duyệt đồ thị (Trajectories):**")
                traj_rows = []
                for t in drift_resp.trajectories:
                    traj_rows.append({
                        "Mã quỹ đạo": t.trajectory_id,
                        "Chiến lược": t.strategy_type,
                        "Đường dẫn duyệt": t.path_description,
                        "Bằng chứng phát hiện": t.evidence_found[:120] + "...",
                        "Điểm số": f"{t.score:.3f}",
                    })
                st.dataframe(pd.DataFrame(traj_rows), use_container_width=True)

            _render_citations_table(drift_resp.citations)

        # =====================================================================
        # 4. CHẾ ĐỘ: SO SÁNH ĐỐI CHIẾU 3 CƠ CHẾ (LOCAL VS GLOBAL VS DRIFT)
        # =====================================================================
        elif "4. So sánh đối chiếu 3 cơ chế" in exec_mode:
            with st.spinner("Đang kích hoạt song song cả 3 cơ chế: Local Search, Global Search và DRIFT Search..."):
                seeds, expanded = get_local_active_results()
                local_active = expanded if kg_enabled else seeds

                if mode == "BM25":
                    active_retriever = BM25Retriever(strategy=strategy)
                elif mode == "Dense":
                    active_retriever = DenseRetriever(strategy=strategy)
                else:
                    active_retriever = HybridRetriever(strategy=strategy)

                global_eng = GlobalSearchEngine(community_detector=detector, default_config=llm_config)
                drift_eng = DRIFTSearchEngine(
                    retriever=active_retriever,
                    expander=expander,
                    temporal_filter=temporal_filter,
                    community_detector=detector,
                    default_config=llm_config,
                )

                cmp_3way = generator.generate_three_way_comparison(
                    query=query,
                    local_results=local_active,
                    reference_date=ref_date_input,
                    top_communities=top_comms,
                    min_relevance_score=float(min_score),
                    global_engine=global_eng,
                    drift_engine=drift_eng,
                    llm_config=llm_config,
                )

            st.divider()
            st.subheader("⚖️ Bảng Đối Chiếu 3 Cơ Chế Tìm Kiếm GraphRAG")
            st.caption("Đối chiếu trực quan câu trả lời, độ sâu, độ rộng và hiệu năng của **Local, Global và DRIFT Search**.")

            # Hiển thị 3 cột kết quả song song
            c_loc, c_glo, c_drf = st.columns(3)

            with c_loc:
                st.markdown("### 🎯 1. Local Search")
                st.caption(f"**Vi mô (Micro)** | `{cmp_3way.local_search.latency_ms:.0f} ms` | {len(cmp_3way.local_search.citations)} nguồn")
                thought_loc = getattr(cmp_3way.local_search, "thought_process", None)
                if thought_loc:
                    with st.expander("💭 Suy luận (Local)", expanded=False):
                        st.markdown(thought_loc)
                with st.container(border=True, height=450):
                    st.markdown(cmp_3way.local_search.answer_text)
                st.caption(f"📊 Trích dẫn thực tế: {sum(1 for c in cmp_3way.local_search.citations if c.is_cited_in_answer)}/{len(cmp_3way.local_search.citations)}")

            with c_glo:
                st.markdown("### 🌐 2. Global Search")
                st.caption(f"**Vĩ mô (Macro)** | `{cmp_3way.global_search.latency_ms:.0f} ms` | {len(cmp_3way.global_search.community_insights)} cụm")
                thought_glo = getattr(cmp_3way.global_search, "thought_process", None)
                if thought_glo:
                    with st.expander("💭 Suy luận (Global)", expanded=False):
                        st.markdown(thought_glo)
                with st.container(border=True, height=450):
                    st.markdown(cmp_3way.global_search.answer_text)
                st.caption(f"📊 Cụm cộng đồng: {len(cmp_3way.global_search.community_insights)} cụm tham gia Map-Reduce")

            with c_drf:
                st.markdown("### 🧭 3. DRIFT Search")
                st.caption(f"**Lai thích ứng (Hybrid)** | `{cmp_3way.drift_search.latency_ms:.0f} ms` | {len(cmp_3way.drift_search.trajectories)} quỹ đạo")
                thought_drf = getattr(cmp_3way.drift_search, "thought_process", None)
                if thought_drf:
                    with st.expander("💭 Suy luận (DRIFT)", expanded=False):
                        st.markdown(thought_drf)
                with st.container(border=True, height=450):
                    st.markdown(cmp_3way.drift_search.answer_text)
                st.caption(f"📊 Trích dẫn thực tế: {sum(1 for c in cmp_3way.drift_search.citations if c.is_cited_in_answer)}/{len(cmp_3way.drift_search.citations)}")

            # Bảng Benchmark Matrix đa chiều
            st.divider()
            st.markdown("### 📊 Ma Trận So Sánh Toàn Diện (GraphRAG Paradigm Matrix)")
            st.dataframe(pd.DataFrame(cmp_3way.comparison_matrix), use_container_width=True)

            # Deep-dive tabs
            st.markdown("### 🔬 Xem chi tiết từng cơ chế:")
            tab_l, tab_g, tab_d = st.tabs(["🎯 Chi tiết Local Search", "🌐 Chi tiết Global Search", "🧭 Chi tiết DRIFT Search"])

            with tab_l:
                _render_citations_table(cmp_3way.local_search.citations)
                _render_pyvis_and_provisions(expander, local_active, kg_enabled, max_hops, allowed_rels)

            with tab_g:
                for idx, ins in enumerate(cmp_3way.global_search.community_insights, 1):
                    st.markdown(f"**Cụm #{idx}: {ins.community_title}** (Điểm liên quan: `{ins.relevance_score:.0f}/100`)")
                    st.write(ins.intermediate_summary)

            with tab_d:
                st.markdown("**Các câu hỏi phụ đã phân rã:**")
                for sq in cmp_3way.drift_search.decomposed_queries:
                    st.markdown(f"- {sq}")
                _render_citations_table(cmp_3way.drift_search.citations)

        # =====================================================================
        # 5. CHẾ ĐỘ: CHỈ TRUY XUẤT (RETRIEVAL ONLY)
        # =====================================================================
        elif "5. Chỉ truy xuất" in exec_mode:
            with st.spinner("Đang truy xuất hạt nhân và mở rộng đồ thị (không gọi LLM)..."):
                seeds, expanded = get_local_active_results()
                active_results = expanded if kg_enabled else seeds

            st.divider()
            st.subheader(f"🔬 Kết quả truy xuất thô ({len(active_results)} điều khoản)")
            _render_pyvis_and_provisions(expander, active_results, kg_enabled, max_hops, allowed_rels)

        # =====================================================================
        # 6. CHẾ ĐỘ: SO SÁNH A/B (KHÔNG KG VS CÓ KG)
        # =====================================================================
        elif "6. So sánh đối chiếu A/B" in exec_mode:
            with st.spinner("Đang chạy đối chiếu song song: Không KG vs. Có Knowledge Graph..."):
                seeds, expanded = get_local_active_results()
                ab_resp = generator.generate_ab_comparison(
                    query=query,
                    seed_results=seeds,
                    expanded_results=expanded,
                    reference_date=ref_date_input,
                    llm_config=llm_config,
                )

            st.divider()
            st.subheader("🆚 Bảng Đối Chiếu A/B: Hiệu Quả Của Knowledge Graph Trong Sinh Câu Trả Lời")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### ❌ Cấu hình A: KHÔNG có Knowledge Graph")
                st.caption(f"Chỉ dựa vào {len(seeds)} Chunks | `{ab_resp.without_kg.latency_ms:.0f} ms`")
                st.info(ab_resp.without_kg.answer_text)
            with col_b:
                st.markdown("### ✅ Cấu hình B: CÓ Knowledge Graph Expansion")
                st.caption(f"Bổ sung {len(expanded) - len(seeds)} điều khoản qua KG | `{ab_resp.with_kg.latency_ms:.0f} ms`")
                st.success(ab_resp.with_kg.answer_text)


# =============================================================================
# HELPER RENDERING FUNCTIONS
# =============================================================================

def _render_thought_process(resp: Any):
    thought = getattr(resp, "thought_process", None)
    if thought:
        with st.expander("💭 Quá trình suy luận nội bộ của mô hình (Thinking Process)", expanded=False):
            st.markdown(thought)


def _render_citations_table(citations: List[Any]):
    if not citations:
        return
    with st.expander(f"📋 Bảng kiểm chứng nguồn trích dẫn nội tuyến ({len(citations)} căn cứ)", expanded=True):
        rows = []
        for c in citations:
            status_str = "✅ Đã trích dẫn" if getattr(c, "is_cited_in_answer", False) else "⚪ Cung cấp thêm"
            origin_str = getattr(c, "retrieval_method", "SEED")
            if getattr(c, "expansion_path", None) and origin_str != "KG_EXPANSION":
                origin_str = "Seed + KG Liên kết"
            elif origin_str == "KG_EXPANSION":
                origin_str = "KG Expansion"

            rows.append({
                "Mã nguồn": f"[#{getattr(c, 'ref_index', 0)}]",
                "Văn bản QPPL": f"{getattr(c, 'doc_title', '')} ({getattr(c, 'doc_number', '')})",
                "Điều khoản": getattr(c, "provision_title", ""),
                "Nguồn gốc": origin_str,
                "Mối quan hệ KG": getattr(c, "expansion_path", "—") or "—",
                "Trạng thái trong câu trả lời": status_str,
                "Hiệu lực": getattr(c, "validity_status", "VALID"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_pyvis_and_provisions(
    expander: GraphExpander,
    active_results: List[RetrievalResult],
    kg_enabled: bool,
    max_hops: int,
    allowed_rels: List[str],
):
    if not active_results:
        st.info("Không tìm thấy điều khoản phù hợp.")
        return

    if kg_enabled and len(active_results) > 1:
        with st.expander("🕸️ Đồ thị mạng lưới điều khoản liên quan (Interactive Subgraph)", expanded=True):
            top_seed_id = active_results[0].provision_id
            all_prov_ids = [r.provision_id for r in active_results]
            try:
                graph_html = build_pyvis_html(
                    graph=expander.graph,
                    center_node_id=top_seed_id,
                    highlight_node_ids=all_prov_ids,
                    max_hops=max_hops,
                    allowed_relations=allowed_rels,
                    height="420px",
                )
            except Exception:
                graph_html = build_pyvis_html(
                    graph=expander.graph,
                    center_node_id=top_seed_id,
                    max_hops=max_hops,
                    allowed_relations=allowed_rels,
                    height="420px",
                )
            components.html(graph_html, height=430, scrolling=False)

    with st.expander(f"📑 Danh sách chi tiết {len(active_results)} điều khoản trích xuất", expanded=False):
        for idx, res in enumerate(active_results, start=1):
            with st.container():
                st.markdown(f"**[#{idx}] {res.provision_title}** ({res.doc_number})")
                st.caption(f"**Cấu trúc:** {res.structural_path} | **Điểm số:** `{res.score}` | **Phương thức:** `{res.retrieval_method}`")
                if res.expansion_path:
                    st.info(f"**Liên kết KG:** {res.expansion_path}\n\n*Chứng cứ:* {res.evidence_text}")
                st.markdown(res.content)
                if res.source_url:
                    st.markdown(f"🔗 [Tra cứu nguồn chính thức]({res.source_url})")
                st.divider()


def _render_debug_bundle(answer_resp: LegalAnswerResponse):
    with st.expander("🔍 Xem Context Bundle & System Prompt thực tế gửi cho LLM (Debug View)", expanded=False):
        st.markdown("#### System Prompt")
        st.code(answer_resp.system_prompt, language="text")
        st.markdown("#### Context Bundle gửi vào Prompt")
        st.code(answer_resp.formatted_context, language="text")
