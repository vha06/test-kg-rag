import os
import sys
import streamlit as st

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.config_loader import load_keys_config

# Tự động nạp cấu hình từ key.txt hoặc .env
load_keys_config()

from storage.db import DatabaseManager
from app.tab_platform_overview import render_platform_overview_tab
from app.tab_multi_domain_lab import render_multi_domain_lab_tab
from app.tab_data_repository import render_data_repository_tab

st.set_page_config(
    page_title="Multi-Domain GraphRAG Platform",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_database_manager():
    return DatabaseManager()


def main():
    st.title("🌐 Multi-Domain GraphRAG Platform")
    st.caption("Nền tảng Tiếp nhận dữ liệu, Xây dựng Đồ thị tri thức phân cấp và Truy xuất thông tin (GraphRAG)")

    db = get_database_manager()

    # Sidebar overview
    counts = db.count_summary()
    st.sidebar.title("📚 Kho Dữ Liệu Tri Thức")
    st.sidebar.markdown(f"""
    - **Tài liệu đã nạp (Documents):** `{counts.get('platform_documents', 0)}`
    - **Khối văn bản (TextUnits / Chunks):** `{counts.get('platform_chunks', 0)}`
    - **Thực thể đồ thị (Entities):** `{counts.get('platform_entities', 0)}`
    - **Mối quan hệ (Relationships):** `{counts.get('platform_relationships', 0)}`
    - **Cụm cộng đồng (Communities):** `{counts.get('graphrag_communities', 0)}`
    - **Báo cáo tóm tắt (Reports):** `{counts.get('graphrag_community_reports', 0)}`
    """)
    st.sidebar.divider()
    st.sidebar.title("🤖 Cấu hình LLM Generator")

    has_qwen_config = bool(os.environ.get("QWEN3_BASE_URL"))
    provider_options = [
        "Mô hình Local Qwen (VPN / OpenAI-compatible)",
        "Google Gemini API",
        "OpenAI / Compatible API (Tùy chỉnh)",
        "Local Ollama",
        "Chế độ mô phỏng (Mock Test)",
    ]
    provider_choice = st.sidebar.selectbox(
        "Nhà cung cấp (Provider):",
        provider_options,
        index=0 if has_qwen_config else 1,
    )

    api_key_input = None
    base_url_input = None

    if provider_choice == "Mô hình Local Qwen (VPN / OpenAI-compatible)":
        provider = "openai_compatible"
        qwen_default_model = os.environ.get("QWEN3_MODEL", "qwen3.6-35b-a3b")
        qwen_default_url = os.environ.get("QWEN3_BASE_URL", "http://222.252.99.27:8035/v1")
        qwen_default_key = os.environ.get("QWEN3_API_KEY", "")

        model_name = st.sidebar.text_input("Tên Model Local:", value=qwen_default_model)
        base_url_input = st.sidebar.text_input("Local Base URL:", value=qwen_default_url)
        api_key_input = st.sidebar.text_input("API Key:", value=qwen_default_key, type="password")

        st.sidebar.success("🟢 Đã thiết lập sẵn cấu hình Local Qwen từ `key.txt`!")
        st.sidebar.caption("🌐 **Lưu ý:** Hãy bật kết nối VPN để truy cập được máy chủ nội bộ `222.252.99.27`.")

    elif provider_choice == "Google Gemini API":
        provider = "gemini"
        model_name = st.sidebar.text_input("Tên Model Gemini:", value="gemini-1.5-flash")
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if env_key:
            st.sidebar.caption("✅ Đã nhận diện `GEMINI_API_KEY` từ môi trường/key.txt.")
        api_key_input = st.sidebar.text_input(
            "Gemini API Key:",
            value=env_key,
            type="password",
            placeholder="AIzaSy...",
            help="Để trống sẽ dùng biến môi trường GEMINI_API_KEY. Nếu chưa có key, hệ thống sẽ dùng chế độ Mock để xem trước định dạng.",
        )
    elif provider_choice == "OpenAI / Compatible API (Tùy chỉnh)":
        provider = "openai_compatible"
        model_name = st.sidebar.text_input("Tên Model:", value="gpt-4o-mini")
        env_key = os.environ.get("OPENAI_API_KEY", "")
        api_key_input = st.sidebar.text_input("API Key:", value=env_key, type="password")
        base_url_input = st.sidebar.text_input("Base URL:", value="https://api.openai.com/v1", help="Có thể thay bằng Groq, OpenRouter, vLLM...")
    elif provider_choice == "Local Ollama":
        provider = "ollama"
        model_name = st.sidebar.text_input("Tên Model Ollama:", value="qwen2.5:7b")
        base_url_input = st.sidebar.text_input("Ollama URL:", value="http://localhost:11434")
    else:
        provider = "mock"
        model_name = "mock-evaluator"

    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    from generation.models import LLMConfig
    llm_config = LLMConfig(
        provider=provider,
        model_name=model_name,
        api_key=api_key_input,
        base_url=base_url_input,
        temperature=temperature,
        timeout=120,
    )

    tab_ingest, tab_qa, tab_repo = st.tabs([
        "📥 1. Nạp Dữ Liệu & Dựng Đồ Thị Tri Thức",
        "💬 2. Truy Xuất Tri Thức & Hỏi Đáp GraphRAG",
        "🗄️ 3. Quản Lý Kho Dữ Liệu & Tri Thức",
    ])

    with tab_ingest:
        render_platform_overview_tab(db, llm_config=llm_config)

    with tab_qa:
        render_multi_domain_lab_tab(db, llm_config=llm_config)

    with tab_repo:
        render_data_repository_tab(db)


if __name__ == "__main__":
    main()
