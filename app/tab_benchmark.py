import os
import streamlit as st
import pandas as pd
from storage.db import DatabaseManager
from evaluation.runner import BenchmarkRunner


def render_benchmark_tab(db: DatabaseManager):
    st.header("📊 Tab 4: Benchmark & Evaluation — Ma trận 18 Cấu hình")
    st.markdown("""
    Đánh giá thực nghiệm khách quan trên **18 cấu hình cơ bản** kết hợp giữa:
    - **3 Chiến lược Chunking:** `A1_article` vs `A2_fixed` vs `A3_structure`
    - **3 Chế độ Retrieval:** `BM25` vs `Dense` vs `Hybrid (RRF)`
    - **2 Trạng thái Knowledge Graph:** `KG Off` vs `KG On`
    
    Phân rã trên **5 nhóm câu hỏi pháp lý** để chứng minh rõ khi nào mỗi kỹ thuật phát huy hiệu quả!
    """)

    runner = BenchmarkRunner(db_manager=db)
    summary_csv = "artifacts/reports/benchmark_summary.csv"
    group_csv = "artifacts/reports/benchmark_by_group.csv"

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run_quick = st.button("⚡ Chạy Benchmark nhanh (10 câu hỏi kiểm thử)", type="secondary")
    with col_btn2:
        run_full = st.button("🚀 Chạy Toàn bộ Benchmark (60 câu hỏi)", type="primary")

    if run_quick or run_full:
        sample_limit = 10 if run_quick else None
        with st.spinner("Đang chạy benchmark qua các cấu hình..."):
            runner.run_matrix(sample_limit=sample_limit)
        st.success("Đã hoàn thành đánh giá benchmark và lưu kết quả!")

    if os.path.exists(summary_csv):
        df_summary = pd.read_csv(summary_csv)
        st.subheader("1. Bảng so sánh tổng thể 18 Cấu hình")
        st.dataframe(df_summary, use_container_width=True)

        # Highlight best configurations
        best_recall = df_summary.loc[df_summary["Recall@5"].idxmax()]
        best_mrr = df_summary.loc[df_summary["MRR@10"].idxmax()]

        c1, c2 = st.columns(2)
        c1.success(f"🏆 **Recall@5 cao nhất:** `{best_recall['Configuration']}` ({best_recall['Recall@5']})")
        c2.info(f"🎯 **MRR@10 cao nhất:** `{best_mrr['Configuration']}` ({best_mrr['MRR@10']})")

        # Visual Chart: Recall@5 across configurations
        st.subheader("2. Biểu đồ so sánh Recall@5 và MRR@10")
        chart_data = df_summary.set_index("Configuration")[["Recall@5", "MRR@10"]]
        st.bar_chart(chart_data)

        # Group Breakdown
        if os.path.exists(group_csv):
            st.subheader("3. Phân rã hiệu quả theo từng Nhóm câu hỏi (Group 1 → 5)")
            df_group = pd.read_csv(group_csv)
            st.dataframe(df_group, use_container_width=True)

        # Download buttons
        st.subheader("4. Tải xuống báo cáo kết quả")
        cdl1, cdl2 = st.columns(2)
        with cdl1:
            with open(summary_csv, "rb") as f:
                st.download_button("📥 Tải Benchmark Summary (CSV)", data=f, file_name="benchmark_summary.csv", mime="text/csv")
        with cdl2:
            full_json = "artifacts/reports/benchmark_full.json"
            if os.path.exists(full_json):
                with open(full_json, "rb") as f:
                    st.download_button("📥 Tải Chi tiết từng truy vấn (JSON)", data=f, file_name="benchmark_full.json", mime="application/json")
    else:
        st.info("Chưa có kết quả benchmark. Bấm nút phía trên để bắt đầu chạy kiểm thử.")
