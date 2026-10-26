from __future__ import annotations

import json
import time
from typing import List, Dict, Any, Optional
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis.network import Network

from storage.db import DatabaseManager
from embedding.adapter import DeterministicHashEmbeddingAdapter, EmbeddingAdapter
from retrieval.graphrag_index_manager import GraphRAGIndexManager
from retrieval.graphrag_query_engine import GraphRAGQueryEngine
from core.schema import (
    GraphRAGQueryResult,
    CanonicalEntity,
    CanonicalRelationship,
)


@st.cache_resource
def get_cached_embedding_adapter(model_type: str):
    """
    Tối ưu nạp và cache Embedding Model trên Streamlit.
    """
    if "hash" in model_type.lower() or "offline" in model_type.lower():
        return DeterministicHashEmbeddingAdapter(dimension=64)
    elif "vietnamese" in model_type.lower():
        try:
            return EmbeddingAdapter(model_name="bkai-foundation-models/vietnamese-bi-encoder")
        except Exception as e:
            st.warning(f"Không thể tải model bkai-foundation-models/vietnamese-bi-encoder ({e}), dùng Deterministic Hash Adapter.")
            return DeterministicHashEmbeddingAdapter(dimension=64)
    elif "paraphrase" in model_type.lower() or "multilingual" in model_type.lower():
        try:
            return EmbeddingAdapter(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            st.warning(f"Không thể tải model multilingual MiniLM ({e}), dùng Deterministic Hash Adapter.")
            return DeterministicHashEmbeddingAdapter(dimension=64)
    return DeterministicHashEmbeddingAdapter(dimension=64)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_query_subgraph_pyvis_html(
    entities: List[Any],
    relationships: List[Any],
    seed_ids: List[str] = None,
    expanded_ids: List[str] = None,
    height: str = "520px",
) -> str:
    seed_set = set(seed_ids or [])
    exp_set = set(expanded_ids or [])
    net = Network(height=height, width="100%", bgcolor="#0e1117", font_color="#ffffff", directed=True)

    for e in entities:
        eid = str(_safe_get(e, "entity_id", ""))
        if not eid:
            continue
        label = str(_safe_get(e, "canonical_name", eid))
        etype = str(_safe_get(e, "entity_type", "ENTITY"))
        desc = str(_safe_get(e, "description", ""))

        if eid in seed_set:
            color = "#ef4444"  # Red for seed
            size = 28
            node_type = "SEED ENTITY (Hạt giống)"
        elif eid in exp_set:
            color = "#10b981"  # Green for expanded
            size = 22
            node_type = "EXPANDED ENTITY (Mở rộng)"
        else:
            color = "#3b82f6"  # Blue
            size = 18
            node_type = "NEIGHBOR ENTITY"

        title_text = f"[{node_type}]\nName: {label}\nType: {etype}\nID: {eid}"
        if desc:
            title_text += f"\nDescription: {desc[:120]}"

        net.add_node(eid, label=label, color=color, title=title_text, size=size)

    for r in relationships:
        src = str(_safe_get(r, "subject_entity_id", ""))
        tgt = str(_safe_get(r, "object_entity_id", ""))
        if not src or not tgt:
            continue
        pred = str(_safe_get(r, "predicate", "RELATED_TO"))
        w = float(_safe_get(r, "weight", 1.0) or 1.0)
        desc = str(_safe_get(r, "description", ""))
        net.add_edge(src, tgt, label=pred, color="#cbd5e1", title=f"Predicate: {pred}\nWeight: {w}\n{desc}")

    net.set_options("""
    var options = {
      "nodes": {"font": {"size": 13, "color": "#ffffff"}},
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
        "color": {"inherit": false},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "barnesHut": {"gravitationalConstant": -2500, "springLength": 110}
      }
    }
    """)
    return net.generate_html()


def render_multi_domain_lab_tab(db: DatabaseManager, llm_config: Optional[Any] = None):
    st.header("💬 Truy Xuất Tri Thức & Hỏi Đáp GraphRAG")
    st.caption(
        "Hệ thống hỏi đáp tăng cường đồ thị tri thức (GraphRAG): Tìm kiếm đa tầng, duyệt đồ thị thực thể con, "
        "đối chiếu bằng chứng nguồn và sinh câu trả lời chính xác kèm trích dẫn."
    )

    # Determine available namespaces in DB
    db_domains = db.get_platform_domains()
    avail_namespaces = ["graphrag_default"]
    for d in db_domains:
        ns = d.get("graph_namespace") or d.get("domain_id")
        if ns and ns not in avail_namespaces:
            avail_namespaces.append(ns)

    q_col1, q_col2, q_col3 = st.columns([1.2, 1.4, 1.4])
    with q_col1:
        selected_namespace = st.selectbox(
            "Không gian Tri thức (Namespace):",
            avail_namespaces,
            index=0,
            help="Chọn namespace chứa chỉ mục TextUnits, Entities và Community Reports.",
        )

    with q_col2:
        query_mode_choice = st.selectbox(
            "Chế độ Truy vấn (Query Mode):",
            [
                "📍 Local Search (Khuyên dùng - Duyệt đồ thị thực thể & ngữ cảnh)",
                "🌐 Global Search (Tổng hợp bao quát qua các cụm cộng đồng)",
                "🌀 DRIFT Search (Duyệt đồ thị đệ quy có kiểm soát)",
                "📌 Vector RAG (Chỉ tìm kiếm theo độ tương đồng văn bản thuần túy)",
            ],
            index=0,
        )

    with q_col3:
        emb_choice = st.selectbox(
            "Bộ tạo Vector (Embedding Model):",
            [
                "bkai-foundation-models/vietnamese-bi-encoder (Khuyên dùng)",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "Deterministic Hash Adapter (Tốc độ cao / Offline)",
            ],
            index=0,
            key="query_emb_choice",
        )

    st.markdown("---")

    # Preset Questions tailored to real document
    preset_questions = [
        "Tổng sản phẩm trong nước (GDP) năm 2015 tăng trưởng bao nhiêu phần trăm?",
        "Tình hình đăng ký doanh nghiệp thành lập mới và giải thể trong năm 2015 diễn biến ra sao?",
        "Tổng quan toàn bộ bức tranh kinh tế - xã hội Việt Nam năm 2015 gồm những điểm nổi bật nào?",
        "Kim ngạch xuất khẩu, nhập khẩu và cán cân thương mại năm 2015 như thế nào?",
        "Cơ quan nào là cơ quan chuyên trách bảo vệ an ninh quốc gia và có quyền hạn gì?",
        "Nhập câu hỏi tự do...",
    ]

    preset_sel = st.selectbox("Chọn câu hỏi gợi ý hoặc nhập tự do:", preset_questions, index=0)
    if preset_sel != "Nhập câu hỏi tự do...":
        active_query = st.text_input("Nội dung câu hỏi:", value=preset_sel)
    else:
        active_query = st.text_input("Nội dung câu hỏi:", value="Tổng sản phẩm trong nước (GDP) năm 2015 tăng trưởng bao nhiêu phần trăm?")

    # Advanced retrieval hyperparameters
    with st.expander("⚙️ Tham số Nâng cao cho Truy vấn GraphRAG", expanded=False):
        hp_c1, hp_c2, hp_c3, hp_c4 = st.columns(4)
        with hp_c1:
            top_k_chunks = st.slider("Top K Chunks (Vector / Local):", min_value=1, max_value=10, value=5)
        with hp_c2:
            top_k_ents = st.slider("Top K Seed Entities (Local):", min_value=1, max_value=10, value=5)
        with hp_c3:
            max_hops_local = st.selectbox("Max Hops Subgraph (Local):", [1, 2], index=0)
        with hp_c4:
            drift_max_depth = st.slider("DRIFT Max Depth:", min_value=1, max_value=3, value=2)

    exec_button = st.button("🚀 Gửi Câu Hỏi & Sinh Câu Trả Lời", type="primary")

    if exec_button:
        cache_key = f"idx_mgr_{selected_namespace}_{emb_choice}"
        if cache_key not in st.session_state:
            adapter = get_cached_embedding_adapter(emb_choice)
            mgr = GraphRAGIndexManager(db_manager=db, embedding_adapter=adapter)
            mgr.ensure_indices_loaded(selected_namespace)
            st.session_state[cache_key] = mgr
        idx_mgr = st.session_state[cache_key]
        engine = GraphRAGQueryEngine(db_manager=db, index_manager=idx_mgr, llm_config=llm_config)

        with st.spinner("Đang truy xuất đồ thị tri thức và sinh câu trả lời..."):
            if "Local Search" in query_mode_choice:
                result = engine.query_local_search(active_query, namespace=selected_namespace, top_k_entities=top_k_ents, max_hops=max_hops_local, top_k_chunks=top_k_chunks)
            elif "Global Search" in query_mode_choice:
                result = engine.query_global_search(active_query, namespace=selected_namespace, top_k_communities=top_k_chunks)
            elif "DRIFT Search" in query_mode_choice:
                result = engine.query_drift_search(active_query, namespace=selected_namespace, max_depth=drift_max_depth)
            else:
                result = engine.query_vector_rag(active_query, namespace=selected_namespace, top_k=top_k_chunks)

            st.session_state["graphrag_last_result"] = result
            st.session_state["graphrag_last_query"] = active_query

    # Display Result if available in session state
    if "graphrag_last_result" in st.session_state:
        result: GraphRAGQueryResult = st.session_state["graphrag_last_result"]

        st.markdown("---")
        st.subheader("💡 Câu Trả Lời Được Sinh Ra:")
        if result.answer == GraphRAGQueryEngine.REFUSAL_MESSAGE:
            st.warning(f"⚠️ **{result.answer}**")
            if result.trace.refusal_reason:
                st.caption(f"Lý do từ chối: `{result.trace.refusal_reason}`")
        else:
            st.success(result.answer)

        # Thought / Reasoning process if present (e.g. from Local Qwen 3.5 / DeepSeek)
        thinking_text = result.trace.details.get("thinking")
        if thinking_text:
            with st.expander("🧠 Quá trình suy luận của mô hình (Thinking / Chain-of-Thought)", expanded=False):
                st.markdown(thinking_text)

        # Metrics row
        met_c1, met_c2, met_c3, met_c4 = st.columns(4)
        met_c1.metric("Thời gian phản hồi", f"{result.trace.latency_ms:.1f}ms")
        met_c2.metric("Số lệnh gọi LLM", result.trace.llm_calls)
        met_c3.metric("Số lượng trích dẫn", len(result.grounded_citations))
        met_c4.metric("Chế độ truy vấn", result.query_mode)

        # Subtabs for inspection
        v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
            f"🕸️ 1. Đồ Thị Con Đã Duyệt",
            f"📑 2. Bằng Chứng Trích Dẫn ({len(result.grounded_citations)})",
            f"👥 3. Báo Cáo Cộng Đồng Liên Quan",
            f"🔍 4. Vết Thực Thi (Query Trace)",
        ])

        with v_tab1:
            st.subheader("Đồ Thị Tri Thức Con Đã Duyệt & Mở Rộng")
            entities_db = db.get_platform_entities(domain_id=selected_namespace, namespace=selected_namespace)
            rels_db = db.get_platform_relationships(domain_id=selected_namespace, namespace=selected_namespace, include_unverified=True)
            if entities_db:
                html_net = build_query_subgraph_pyvis_html(
                    entities=entities_db,
                    relationships=rels_db,
                    seed_ids=result.trace.seed_entity_ids,
                    expanded_ids=result.trace.expanded_entity_ids,
                    height="520px",
                )
                components.html(html_net, height=540)
                st.caption("🔴 Đỏ: Thực thể Hạt giống (Seed) | 🟢 Xanh lá: Thực thể Mở rộng (Expanded) | 🔵 Xanh dương: Thực thể lân cận")
            else:
                st.info("Chưa có thực thể trong namespace này.")

        with v_tab2:
            st.subheader("Bằng Chứng Nguồn Đã Xác Minh (Grounded Citations)")
            if not result.grounded_citations:
                st.info("Không có trích dẫn nào được chọn.")
            else:
                for cit in result.grounded_citations:
                    c_id = _safe_get(cit, "citation_id")
                    c_type = _safe_get(cit, "source_type", "TEXT_UNIT")
                    c_score = float(_safe_get(cit, "score", 0.0) or 0.0)
                    c_doc = _safe_get(cit, "document_id", "")
                    c_snip = _safe_get(cit, "content_snippet", "")
                    with st.container():
                        st.markdown(f"**[{c_id}] Loại nguồn: `{c_type}` | Score: `{c_score:.3f}` | Doc: `{c_doc}`**")
                        st.info(c_snip)
                        st.divider()

        with v_tab3:
            st.subheader("Báo Cáo Cộng Đồng Được Kích Hoạt")
            if not result.trace.selected_community_ids:
                st.info("Chế độ truy vấn này tập trung vào thực thể và TextUnits, không sử dụng trực tiếp báo cáo cộng đồng.")
            else:
                all_reps = db.get_graphrag_community_reports(namespace=selected_namespace)
                sel_comm_set = set(result.trace.selected_community_ids)
                matched_reps = [r for r in all_reps if r.get("community_id") in sel_comm_set]
                for rep in matched_reps:
                    with st.expander(f"📑 [Level {rep.get('level', 0)}] {rep.get('title', 'Community')}", expanded=True):
                        st.markdown(f"**Tóm tắt:** {rep.get('summary')}")
                        st.markdown(f"**Nội dung chi tiết:** {rep.get('full_content')}")

        with v_tab4:
            st.subheader("Chi Tiết Vết Truy Vết Thực Thi (GraphRAGQueryTrace)")
            st.json(result.trace.model_dump())
