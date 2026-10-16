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
    st.sidebar.title("🤖 Cấu hình LLM Generator")
    provider_choice = st.sidebar.selectbox(
        "Nhà cung cấp (Provider):",
        ["Google Gemini API", "OpenAI / Compatible API", "Local Ollama", "Chế độ mô phỏng (Mock Test)"],
        index=0,
    )

    provider_map = {
        "Google Gemini API": "gemini",
        "OpenAI / Compatible API": "openai_compatible",
        "Local Ollama": "ollama",
        "Chế độ mô phỏng (Mock Test)": "mock",
    }
    provider = provider_map[provider_choice]

    api_key_input = None
    base_url_input = None

    if provider == "gemini":
        model_name = st.sidebar.text_input("Tên Model Gemini:", value="gemini-2.5-flash")
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if env_key:
            st.sidebar.caption("✅ Đã nhận diện `GEMINI_API_KEY` từ môi trường.")
        api_key_input = st.sidebar.text_input(
            "Gemini API Key:",
            value=env_key,
            type="password",
            placeholder="AIzaSy...",
            help="Để trống sẽ dùng biến môi trường GEMINI_API_KEY. Nếu chưa có key, hệ thống sẽ dùng chế độ Mock để xem trước định dạng.",
        )
    elif provider == "openai_compatible":
        model_name = st.sidebar.text_input("Tên Model:", value="gpt-4o-mini")
        env_key = os.environ.get("OPENAI_API_KEY", "")
        api_key_input = st.sidebar.text_input("API Key:", value=env_key, type="password")
        base_url_input = st.sidebar.text_input("Base URL:", value="https://api.openai.com/v1", help="Có thể thay bằng Groq, OpenRouter, vLLM...")
    elif provider == "ollama":
        model_name = st.sidebar.text_input("Tên Model Ollama:", value="qwen2.5:7b")
        base_url_input = st.sidebar.text_input("Ollama URL:", value="http://localhost:11434")
    else:
        model_name = "mock-evaluator"

    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    from generation.models import LLMConfig
    llm_config = LLMConfig(
        provider=provider,
        model_name=model_name,
        api_key=api_key_input,
        base_url=base_url_input,
        temperature=temperature,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 1. Interactive Search",
        "🔬 2. Chunking Lab",
        "🕸️ 3. Knowledge Graph Explorer",
        "📊 4. Benchmark & Evaluation",
    ])

    with tab1:
        render_search_tab(db, expander, temporal_filter, llm_config=llm_config)

    with tab2:
        render_chunking_tab(db)

    with tab3:
        render_kg_tab(db, builder)

    with tab4:
        render_benchmark_tab(db)


if __name__ == "__main__":
    main()
