import streamlit as st
from typing import Optional
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from knowledge_graph.expander import GraphExpander
from temporal.validity_filter import TemporalValidityFilter
from storage.db import DatabaseManager

PRESET_QUERIES = [
    "Quy định về phạm vi điều chỉnh của Luật An ninh quốc gia số 32/2004/QH11?",
    "Người dân tự giác tham gia bảo vệ an ninh mà bị thiệt hại tài sản hoặc thương tật thì được trợ cấp, đền bù ra sao?",
    "Các quy định về điều kiện, tiêu chuẩn và trang phục của lực lượng bảo vệ cơ quan, doanh nghiệp?",
    "Quy chế điều hành và kiểm soát phương tiện, người xuất cảnh nhập cảnh tại các cửa khẩu biên giới đất liền?",
    "Quy định tại Nghị định số 112/2014/NĐ-CP được sửa đổi bổ sung bởi Nghị định số 34/2023/NĐ-CP vào ngày 31/07/2023 như thế nào?",
]


def render_search_tab(db: DatabaseManager, expander: GraphExpander, temporal_filter: TemporalValidityFilter):
    st.header("🔍 Tab 1: Interactive Search & Legal Provenance")
    st.markdown("Truy xuất điều khoản quy phạm pháp luật, đối chiếu hiệu quả của **Chunking, Embedding và Knowledge Graph**.")

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
            ["A1_article", "A2_fixed", "A3_structure"],
            help="A1: Cấp Điều | A2: Cửa sổ Token cố định | A3: Cấu trúc pháp lý Điều/Khoản/Điểm",
        )
        mode = st.selectbox("Phương thức Retrieval:", ["Hybrid", "BM25", "Dense"])
        top_k = st.slider("Số lượng kết quả (Top-K):", min_value=1, max_value=20, value=5)

    with st.expander("⚙️ Cấu hình Knowledge Graph & Temporal Filtering", expanded=True):
        c_kg1, c_kg2, c_kg3 = st.columns(3)
        with c_kg1:
            kg_enabled = st.checkbox("Bật Knowledge Graph Expansion", value=True)
            max_hops = st.selectbox("Độ sâu Graph (Hops):", [1, 2], index=0)
        with c_kg2:
            allowed_rels = st.multiselect(
                "Loại quan hệ mở rộng:",
                ["AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "REFERENCES"],
                default=["AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "REFERENCES"],
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
                seeds = retriever.search(query, top_k=top_k)
            elif mode == "Dense":
                retriever = DenseRetriever(strategy=strategy)
                seeds = retriever.search(query, top_k=top_k)
            else:
                retriever = HybridRetriever(strategy=strategy)
                seeds = retriever.search(query, top_k=top_k)

            # 2. KG Expansion
            if kg_enabled:
                results = expander.expand(
                    query=query,
                    seed_results=seeds,
                    max_hops=max_hops,
                    max_expanded_nodes=5,
                    allowed_relations=allowed_rels,
                )
            else:
                results = seeds

            # 3. Temporal Validity
            results = temporal_filter.filter_and_annotate(
                results=results,
                reference_date=ref_date_input,
                filter_out_invalid=filter_invalid,
            )

        st.subheader(f"Kết quả truy xuất ({len(results)} điều khoản)")

        if not results:
            st.info("Không tìm thấy điều khoản phù hợp.")
            return

        for idx, res in enumerate(results, start=1):
            with st.container():
                # Badges
                badge_color = "blue" if res.retrieval_method != "KG_EXPANSION" else "purple"
                kg_badge = f":{badge_color}-background[{res.retrieval_method}]"

                val_badge = ""
                if res.validity_status == "VALID":
                    val_badge = ":green-background[Đang có hiệu lực]"
                elif res.validity_status == "VALID_PRIOR_TO_AMENDMENT":
                    val_badge = ":orange-background[Hiệu lực nguyên bản (trước sửa đổi)]"
                elif res.validity_status == "INVALID_NOT_YET_EFFECTIVE":
                    val_badge = ":red-background[Chưa có hiệu lực tại ngày tham chiếu]"
                elif res.validity_status == "INVALID_EXPIRED":
                    val_badge = ":red-background[Đã hết hiệu lực]"

                st.markdown(f"### #{idx}. {res.provision_title} | {kg_badge} {val_badge}")
                st.caption(f"**Văn bản:** {res.doc_title} ({res.doc_number}) | **Cấu trúc:** {res.structural_path} | **Điểm số:** `{res.score}`")

                if res.expansion_path:
                    st.info(f"**Lý do bổ sung qua KG:** {res.expansion_path}\n\n*Căn cứ chứng minh:* {res.evidence_text}")

                with st.expander("Xem toàn văn điều khoản trích xuất", expanded=(idx <= 2)):
                    st.markdown(res.content)
                    if res.source_url:
                        st.markdown(f"🔗 [Tra cứu nguồn chính thức trên VBPL/Pháp điển]({res.source_url})")

                st.divider()
