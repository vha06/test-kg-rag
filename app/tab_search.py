import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from typing import Optional, List
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from knowledge_graph.expander import GraphExpander
from knowledge_graph.visualizer import build_pyvis_html
from temporal.validity_filter import TemporalValidityFilter
from storage.db import DatabaseManager
from storage.models import RetrievalResult
from generation.models import LLMConfig, LegalAnswerResponse, ABComparisonResult
from generation.answer_generator import LegalAnswerGenerator

PRESET_QUERIES = [
    "Quy định về phạm vi điều chỉnh của Luật An ninh quốc gia số 32/2004/QH11?",
    "Người dân tự giác tham gia bảo vệ an ninh mà bị thiệt hại tài sản hoặc thương tật thì được trợ cấp, đền bù ra sao?",
    "Các quy định về điều kiện, tiêu chuẩn và trang phục của lực lượng bảo vệ cơ quan, doanh nghiệp?",
    "Quy chế điều hành và kiểm soát phương tiện, người xuất cảnh nhập cảnh tại các cửa khẩu biên giới đất liền?",
    "Quy định tại Nghị định số 112/2014/NĐ-CP được sửa đổi bổ sung bởi Nghị định số 34/2023/NĐ-CP vào ngày 31/07/2023 như thế nào?",
]


def render_search_tab(
    db: DatabaseManager,
    expander: GraphExpander,
    temporal_filter: TemporalValidityFilter,
    llm_config: Optional[LLMConfig] = None,
):
    st.header("🔍 Tab 1: Interactive Search & Legal Provenance")
    st.markdown("Truy xuất điều khoản quy phạm pháp luật, đối chiếu hiệu quả của **Chunking, Embedding và Knowledge Graph** kết hợp **LLM Answer Generation**.")

    # 1. Chế độ thực thi
    exec_mode = st.radio(
        "Chế độ thực thi:",
        [
            "🔍 Chỉ truy xuất (Retrieval + KG Only)",
            "🤖 Truy xuất + Sinh câu trả lời LLM (Single Mode)",
            "⚖️ So sánh đối chiếu A/B (Không KG vs. Có KG)",
        ],
        horizontal=True,
        index=1,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_preset = st.selectbox("Chọn câu hỏi mẫu thử nghiệm:", ["-- Nhập câu hỏi tự do --"] + PRESET_QUERIES)
        if selected_preset != "-- Nhập câu hỏi tự do --":
            query = st.text_area("Câu hỏi tra cứu:", value=selected_preset, height=80)
        else:
            query = st.text_area("Câu hỏi tra cứu:", value="", placeholder="Nhập câu hỏi pháp lý tiếng Việt...", height=80)

    with col2:
        strategy = st.selectbox(
            "Chiến lược Chunking:",
            ["A3_structure", "A1_article", "A2_fixed"],
            help="A1: Cấp Điều | A2: Cửa sổ Token cố định | A3: Cấu trúc pháp lý Điều/Khoản/Điểm",
        )
        mode = st.selectbox("Phương thức Retrieval:", ["Hybrid", "BM25", "Dense"])
        top_k = st.slider("Số lượng kết quả (Top-K):", min_value=1, max_value=20, value=5)

    with st.expander("⚙️ Cấu hình Knowledge Graph & Temporal Filtering", expanded=False):
        c_kg1, c_kg2, c_kg3 = st.columns(3)
        with c_kg1:
            kg_enabled = st.checkbox("Bật Knowledge Graph Expansion", value=True)
            max_hops = st.selectbox("Độ sâu Graph (Hops):", [1, 2], index=0)
        with c_kg2:
            allowed_rels = st.multiselect(
                "Loại quan hệ mở rộng:",
                ["AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "REFERENCES", "CONTAINS"],
                default=["AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "REFERENCES"],
                help="Chọn thêm CONTAINS nếu muốn trực quan hóa thêm Văn bản cha và các Khoản thuộc Điều.",
            )
        with c_kg3:
            ref_date_input = st.text_input("Ngày tham chiếu hiệu lực (YYYY-MM-DD):", value="2024-01-01")
            filter_invalid = st.checkbox("Ẩn các điều khoản chưa/hết hiệu lực", value=False)

    if st.button("🚀 Chạy Truy Vấn", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Vui lòng nhập câu hỏi.")
            return

        with st.spinner("Đang tìm kiếm và duyệt đồ thị tri thức..."):
            # 1. Retrieval
            if mode == "BM25":
                retriever = BM25Retriever(strategy=strategy)
                raw_seeds = retriever.search(query, top_k=top_k)
            elif mode == "Dense":
                retriever = DenseRetriever(strategy=strategy)
                raw_seeds = retriever.search(query, top_k=top_k)
            else:
                retriever = HybridRetriever(strategy=strategy)
                raw_seeds = retriever.search(query, top_k=top_k)

            # Lọc hiệu lực cho Seeds
            seeds = temporal_filter.filter_and_annotate(
                results=raw_seeds,
                reference_date=ref_date_input,
                filter_out_invalid=filter_invalid,
            )

            # 2. KG Expansion
            if kg_enabled:
                raw_expanded = expander.expand(
                    query=query,
                    seed_results=seeds,
                    max_hops=max_hops,
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

        active_results: List[RetrievalResult] = expanded if kg_enabled else seeds

        generator = LegalAnswerGenerator(llm_config)

        # ------------------------------------------------------------------
        # PHẦN 1: SINH CÂU TRẢ LỜI LLM (NẾU ĐƯỢC CHỌN)
        # ------------------------------------------------------------------
        if exec_mode == "🤖 Truy xuất + Sinh câu trả lời LLM (Single Mode)":
            with st.spinner("Đang tổng hợp căn cứ và sinh câu trả lời pháp lý từ LLM..."):
                answer_resp = generator.generate_answer(
                    query=query,
                    results=active_results,
                    reference_date=ref_date_input,
                    llm_config=llm_config,
                )

            st.divider()
            st.subheader("🤖 Câu trả lời Pháp lý từ LLM")

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Mô hình sử dụng", answer_resp.model_used)
            m_col2.metric("Thời gian phản hồi", f"{answer_resp.latency_ms:.0f} ms")
            suff_label = "✅ Đủ căn cứ pháp lý" if answer_resp.has_sufficient_evidence else "⚠️ Thiếu căn cứ / Từ chối"
            m_col3.metric("Độ tin cậy căn cứ", suff_label)

            # Hiển thị cảnh báo hiệu lực thời gian
            if answer_resp.temporal_warnings:
                with st.container():
                    st.warning("⚠️ **Lưu ý về Hiệu lực Thời gian & Văn bản Sửa đổi:**\n\n" + "\n\n".join(answer_resp.temporal_warnings))

            # Nội dung câu trả lời
            st.markdown(
                f"""
                <div style="background-color: rgba(28, 131, 225, 0.08); padding: 18px; border-radius: 8px; border-left: 5px solid #1C83E1; margin: 12px 0;">
                {answer_resp.answer_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Bảng trích dẫn nguồn (Citation Verification Table)
            if answer_resp.citations:
                with st.expander("📋 Bảng kiểm chứng nguồn trích dẫn nội tuyến (Citation Verification)", expanded=True):
                    table_rows = []
                    for c in answer_resp.citations:
                        status_str = "✅ Đã trích dẫn" if c.is_cited_in_answer else "⚪ Cung cấp thêm"
                        if c.expansion_path and c.retrieval_method != "KG_EXPANSION":
                            origin_str = "Seed + KG Liên kết"
                        elif c.retrieval_method == "KG_EXPANSION":
                            origin_str = "KG Expansion"
                        else:
                            origin_str = "Seed Retrieval"

                        table_rows.append({
                            "Mã nguồn": f"[#{c.ref_index}]",
                            "Văn bản QPPL": f"{c.doc_title} ({c.doc_number})",
                            "Điều khoản": c.provision_title,
                            "Nguồn gốc": origin_str,
                            "Mối quan hệ KG": c.expansion_path or "—",
                            "Trạng thái trong câu trả lời": status_str,
                            "Hiệu lực": c.validity_status,
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # Debug View
            with st.expander("🔍 Xem Context Bundle & System Prompt thực tế gửi cho LLM (Debug View)", expanded=False):
                st.markdown("#### System Prompt")
                st.code(answer_resp.system_prompt, language="text")
                st.markdown("#### Context Bundle gửi vào Prompt")
                st.code(answer_resp.formatted_context, language="text")

        elif exec_mode == "⚖️ So sánh đối chiếu A/B (Không KG vs. Có KG)":
            with st.spinner("Đang chạy đối chiếu song song: Cấu hình Không KG vs. CÓ Knowledge Graph..."):
                ab_resp = generator.generate_ab_comparison(
                    query=query,
                    seed_results=seeds,
                    expanded_results=expanded,
                    reference_date=ref_date_input,
                    llm_config=llm_config,
                )

            st.divider()
            st.subheader("⚖️ Bảng Đối Chiếu A/B: Hiệu Quả Của Knowledge Graph Trong Sinh Câu Trả Lời")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### ❌ Cấu hình A: KHÔNG có Knowledge Graph")
                st.caption(f"Chỉ dựa vào {len(seeds)} Chunks điểm cao nhất từ Retrieval | Model: `{ab_resp.without_kg.model_used}` | `{ab_resp.without_kg.latency_ms:.0f} ms`")
                if ab_resp.without_kg.temporal_warnings:
                    st.warning("\n\n".join(ab_resp.without_kg.temporal_warnings))
                st.info(ab_resp.without_kg.answer_text)
                st.caption(f"📊 Nguồn trích dẫn: {sum(1 for c in ab_resp.without_kg.citations if c.is_cited_in_answer)}/{len(ab_resp.without_kg.citations)} nguồn được dùng.")

            with col_b:
                st.markdown("### ✅ Cấu hình B: CÓ Knowledge Graph Expansion")
                st.caption(f"Bổ sung {len(expanded) - len(seeds)} điều khoản sửa đổi/dẫn chiếu | Model: `{ab_resp.with_kg.model_used}` | `{ab_resp.with_kg.latency_ms:.0f} ms`")
                if ab_resp.with_kg.temporal_warnings:
                    st.warning("\n\n".join(ab_resp.with_kg.temporal_warnings))
                st.success(ab_resp.with_kg.answer_text)
                st.caption(f"📊 Nguồn trích dẫn: {sum(1 for c in ab_resp.with_kg.citations if c.is_cited_in_answer)}/{len(ab_resp.with_kg.citations)} nguồn được dùng.")

            # Bảng tóm tắt chỉ số so sánh
            st.markdown("#### 📊 Thống kê đối chiếu định lượng:")
            stats_df = pd.DataFrame([
                {
                    "Chỉ số": "Số lượng điều khoản cung cấp cho LLM",
                    "A: Không có KG": len(ab_resp.without_kg.citations),
                    "B: Có KG Expansion": len(ab_resp.with_kg.citations),
                    "Chênh lệch": f"+{len(ab_resp.with_kg.citations) - len(ab_resp.without_kg.citations)} điều khoản",
                },
                {
                    "Chỉ số": "Số nguồn LLM thực sự trích dẫn",
                    "A: Không có KG": sum(1 for c in ab_resp.without_kg.citations if c.is_cited_in_answer),
                    "B: Có KG Expansion": sum(1 for c in ab_resp.with_kg.citations if c.is_cited_in_answer),
                    "Chênh lệch": f"+{sum(1 for c in ab_resp.with_kg.citations if c.is_cited_in_answer) - sum(1 for c in ab_resp.without_kg.citations if c.is_cited_in_answer)} trích dẫn",
                },
                {
                    "Chỉ số": "Số quan hệ sửa đổi / dẫn chiếu phát hiện",
                    "A: Không có KG": sum(1 for c in ab_resp.without_kg.citations if c.retrieval_method == "KG_EXPANSION"),
                    "B: Có KG Expansion": sum(1 for c in ab_resp.with_kg.citations if c.retrieval_method == "KG_EXPANSION"),
                    "Chênh lệch": f"+{sum(1 for c in ab_resp.with_kg.citations if c.retrieval_method == 'KG_EXPANSION')} liên kết pháp lý",
                },
                {
                    "Chỉ số": "Thời gian xử lý LLM (ms)",
                    "A: Không có KG": f"{ab_resp.without_kg.latency_ms:.0f} ms",
                    "B: Có KG Expansion": f"{ab_resp.with_kg.latency_ms:.0f} ms",
                    "Chênh lệch": f"{(ab_resp.with_kg.latency_ms - ab_resp.without_kg.latency_ms):+.0f} ms",
                },
            ])
            st.dataframe(stats_df, use_container_width=True)

        # ------------------------------------------------------------------
        # PHẦN 2: ĐỒ THỊ PYVIS VÀ DANH SÁCH CHI TIẾT ĐIỀU KHOẢN
        # ------------------------------------------------------------------
        st.divider()
        st.subheader(f"Danh sách căn cứ điều khoản ({len(active_results)} điều khoản)")

        if not active_results:
            st.info("Không tìm thấy điều khoản phù hợp.")
            return

        # Interactive Graph visualization for search results
        if kg_enabled and len(active_results) > 1:
            with st.expander("🕸️ Đồ thị động tương tác mạng lưới kết quả truy xuất (Seed ↔ KG Expansion)", expanded=True):
                st.caption("Các node bao gồm Seed được truy xuất trực tiếp và các node được mở rộng tự động qua Knowledge Graph:")
                top_seed_id = active_results[0].provision_id
                try:
                    graph_html = build_pyvis_html(
                        graph=expander.graph,
                        center_node_id=top_seed_id,
                        highlight_node_ids=all_prov_ids,
                        max_hops=max_hops,
                        allowed_relations=allowed_rels,
                        height="420px",
                    )
                except TypeError:
                    # Fallback phòng trường hợp tiến trình Streamlit cũ chưa tải lại module visualizer
                    graph_html = build_pyvis_html(
                        graph=expander.graph,
                        center_node_id=top_seed_id,
                        max_hops=max_hops,
                        allowed_relations=allowed_rels,
                        height="420px",
                    )
                components.html(graph_html, height=430, scrolling=False)

        for idx, res in enumerate(active_results, start=1):
            with st.container():
                if res.expansion_path and res.retrieval_method != "KG_EXPANSION":
                    kg_badge = ":blue-background[SEED + KG LIÊN KẾT]"
                elif res.retrieval_method == "KG_EXPANSION":
                    kg_badge = ":purple-background[KG_EXPANSION]"
                else:
                    kg_badge = f":blue-background[{res.retrieval_method}]"

                val_badge = ""
                if res.validity_status == "VALID":
                    val_badge = ":green-background[Đang có hiệu lực]"
                elif res.validity_status == "VALID_PRIOR_TO_AMENDMENT":
                    val_badge = ":orange-background[Hiệu lực nguyên bản (trước sửa đổi)]"
                elif res.validity_status == "INVALID_NOT_YET_EFFECTIVE":
                    val_badge = ":red-background[Chưa có hiệu lực tại ngày tham chiếu]"
                elif res.validity_status == "INVALID_EXPIRED":
                    val_badge = ":red-background[Đã hết hiệu lực]"

                st.markdown(f"### [#{idx}] {res.provision_title} | {kg_badge} {val_badge}")
                st.caption(f"**Văn bản:** {res.doc_title} ({res.doc_number}) | **Cấu trúc:** {res.structural_path} | **Điểm số:** `{res.score}`")

                if res.expansion_path:
                    st.info(f"**Lý do bổ sung qua KG:** {res.expansion_path}\n\n*Căn cứ chứng minh:* {res.evidence_text}")

                with st.expander("Xem toàn văn điều khoản trích xuất", expanded=(idx <= 2)):
                    st.markdown(res.content)
                    if res.source_url:
                        st.markdown(f"🔗 [Tra cứu nguồn chính thức trên VBPL/Pháp điển]({res.source_url})")

                st.divider()
