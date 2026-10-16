import os
import sys
import streamlit as st

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import DatabaseManager
from knowledge_graph.builder import KnowledgeGraphBuilder
from knowledge_graph.expander import GraphExpander
from temporal.validity_filter import TemporalValidityFilter
from app.tab_search import render_search_tab
from app.tab_chunking import render_chunking_tab
from app.tab_kg import render_kg_tab
from app.tab_benchmark import render_benchmark_tab

st.set_page_config(
    page_title="Legal Retrieval Lab",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_system_components():
    db = DatabaseManager()
    builder = KnowledgeGraphBuilder(db)
    builder.build_graph(verified_only=True)
    expander = GraphExpander(builder)
    temporal_filter = TemporalValidityFilter(db)
    return db, builder, expander, temporal_filter


def main():
    st.title("⚖️ LEGAL RETRIEVAL LAB")
    st.caption("Demo chứng minh hiệu quả của **Chunking, Embedding và Knowledge Graph** trong truy xuất văn bản quy phạm pháp luật Việt Nam")

    db, builder, expander, temporal_filter = get_system_components()

    # Sidebar overview
    counts = db.count_summary()
    st.sidebar.title("📚 Thông tin Kho dữ liệu")
    st.sidebar.markdown(f"""
    - **Văn bản QPPL:** `{counts.get('documents', 0)}`
    - **Điều / Khoản / Điểm:** `{counts.get('provisions', 0)}`
    - **Chunks đã lập chỉ mục:** `{counts.get('chunks', 0)}`
    - **Quan hệ Knowledge Graph:** `{counts.get('kg_edges', 0)}`
    - **Phiên bản lịch sử:** `{counts.get('provision_versions', 0)}`
    """)
    st.sidebar.divider()
    st.sidebar.info("💡 Hệ thống hỗ trợ 18 cấu hình thí nghiệm (3 Chunking × 3 Retrieval × 2 KG) và kiểm soát hiệu lực theo thời gian.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 1. Interactive Search",
        "🔬 2. Chunking Lab",
        "🕸️ 3. Knowledge Graph Explorer",
        "📊 4. Benchmark & Evaluation",
    ])

    with tab1:
        render_search_tab(db, expander, temporal_filter)

    with tab2:
        render_chunking_tab(db)

    with tab3:
        render_kg_tab(db, builder)

    with tab4:
        render_benchmark_tab(db)


if __name__ == "__main__":
    main()
