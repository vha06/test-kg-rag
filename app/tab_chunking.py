import streamlit as st
import pandas as pd
from typing import Dict, Any
from storage.db import DatabaseManager
from chunking.base import BaseChunker
from retrieval.bm25_retriever import BM25Retriever


def render_chunking_tab(db: DatabaseManager):
    st.header("🔬 Tab 2: Chunking Lab — So sánh 3 chiến lược phân đoạn")
    st.markdown("""
    Đánh giá chi tiết cách chia văn bản quy phạm pháp luật ảnh hưởng đến việc bảo toàn ngữ cảnh:
    - **A1 (Article-level):** Đơn vị là Điều, giữ trọn vẹn ngữ cảnh nhưng các Điều dài có thể vượt context window.
    - **A2 (Fixed-size):** Cắt theo số token cố định (256 tokens, overlap 32), dễ cắt ngang qua Khoản/Điểm và làm mất tiêu đề Điều.
    - **A3 (Structure-aware):** Chia theo ranh giới cấu trúc Khoản/Điểm nhưng **tiêm ngữ cảnh cha (Context Injection)** vào từng chunk.
    """)

    # 1. Overall Statistics Table
    st.subheader("1. Thống kê phân bố Chunks trên toàn bộ kho dữ liệu")
    stats_data = []
    for strat in ["A1_article", "A2_fixed", "A3_structure"]:
        chunks = db.get_chunks_by_strategy(strat)
        # Mock ChunkModel objects to compute stats
        class SimpleChunk:
            def __init__(self, c):
                self.token_count = c["token_count"]
                self.crosses_boundary = bool(c["crosses_boundary"])
        s_chunks = [SimpleChunk(c) for c in chunks]
        stats = BaseChunker.compute_stats(s_chunks)
        stats["strategy"] = strat
        stats_data.append(stats)

    df_stats = pd.DataFrame(stats_data)[[
        "strategy", "total_chunks", "token_min", "token_median", "token_p90", "token_max",
        "too_short_count", "too_long_count", "crosses_boundary_count", "crosses_boundary_ratio"
    ]]
    st.dataframe(df_stats, use_container_width=True)

    st.divider()

    # 2. Document & Article Visual Inspector
    st.subheader("2. So sánh trực quan trên một Văn bản cụ thể")
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT document_id, doc_number, title FROM documents WHERE doc_type IN ('Luật', 'Nghị định') LIMIT 20")
        docs = cursor.fetchall()

    doc_options = {f"{r['doc_number']} - {r['title']}": r["document_id"] for r in docs}
    selected_doc_label = st.selectbox("Chọn văn bản để kiểm tra phân đoạn:", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_doc_label]

    # Fetch provisions of this doc
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT provision_id, title, content
            FROM provisions
            WHERE document_id = ? AND level = 'ARTICLE'
            LIMIT 15
        """, (selected_doc_id,))
        art_rows = cursor.fetchall()

    art_options = {r["title"]: r for r in art_rows}
    if not art_options:
        st.info("Văn bản này chưa có dữ liệu điều khoản.")
        return

    selected_art_title = st.selectbox("Chọn Điều cụ thể:", list(art_options.keys()))
    target_art = art_options[selected_art_title]

    st.markdown(f"#### Nội dung gốc của {selected_art_title}")
    st.info(target_art["content"])

    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.markdown("### A1 (Article-level)")
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE strategy = 'A1_article' AND provision_id = ?", (target_art["provision_id"],))
            a1_chunks = cursor.fetchall()
        st.caption(f"Số lượng chunk: `{len(a1_chunks)}`")
        for c in a1_chunks:
            st.text_area(f"Chunk ID: {c['chunk_id']} ({c['token_count']} tokens)", value=c["content"], height=220)

    with col_a2:
        st.markdown("### A2 (Fixed-size)")
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE strategy = 'A2_fixed' AND provision_id = ?", (target_art["provision_id"],))
            a2_chunks = cursor.fetchall()
        st.caption(f"Số lượng chunk: `{len(a2_chunks)}`")
        for c in a2_chunks:
            flag = "⚠️ CẮT QUA RANH GIỚI" if c["crosses_boundary"] else "Chuẩn"
            st.text_area(f"Chunk ID: {c['chunk_id']} ({c['token_count']} tokens) | {flag}", value=c["content"], height=220)

    with col_a3:
        st.markdown("### A3 (Structure-aware)")
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chunks
                WHERE strategy = 'A3_structure'
                  AND (provision_id = ? OR provision_id LIKE ?)
            """, (target_art["provision_id"], f"{target_art['provision_id']}__%"))
            a3_chunks = cursor.fetchall()
        st.caption(f"Số lượng chunk: `{len(a3_chunks)}` (Tách theo Khoản + Tiêm ngữ cảnh)")
        for c in a3_chunks:
            st.text_area(f"Chunk ID: {c['chunk_id']} ({c['token_count']} tokens)", value=c["content"], height=220)

    st.divider()

    # 3. Retrieval Comparison across A1 vs A2 vs A3 on the same query
    st.subheader("3. So sánh khả năng tìm kiếm của A1 vs A2 vs A3 trên cùng một câu hỏi")
    test_q = st.text_input("Nhập câu hỏi để so sánh trực tiếp kết quả retrieval:", value="Nhiệm vụ và quyền hạn của lực lượng bảo vệ cơ quan?")
    if st.button("So sánh Kết Quả 3 Chiến Lược"):
        c1, c2, c3 = st.columns(3)
        for col, s_name in zip([c1, c2, c3], ["A1_article", "A2_fixed", "A3_structure"]):
            with col:
                st.markdown(f"#### Kết quả {s_name}")
                bm25 = BM25Retriever(strategy=s_name)
                res = bm25.search(test_q, top_k=3)
                for r in res:
                    st.markdown(f"**#{r.rank}. {r.provision_title}** (`Score: {r.score}`)")
                    st.caption(r.structural_path)
                    st.text(r.content[:150] + "...")
