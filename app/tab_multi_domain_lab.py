from __future__ import annotations

import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from typing import List, Dict, Any

from storage.db import DatabaseManager
from embedding.adapter import DeterministicHashEmbeddingAdapter
from retrieval.multi_domain_retriever import (
    MultiDomainIndexManager,
    MultiDomainHybridRetriever,
    CanonicalRetrievalResult,
)
from knowledge_graph.multi_graph_builder import MultiDomainGraphBuilder
from knowledge_graph.namespace_expander import NamespaceGraphExpander
from pyvis.network import Network


def build_platform_pyvis_html(
    graph,
    seed_entity_ids: List[str] = None,
    expanded_entity_ids: List[str] = None,
    height: str = "520px",
) -> str:
    seed_set = set(seed_entity_ids or [])
    exp_set = set(expanded_entity_ids or [])

    net = Network(
        height=height,
        width="100%",
        bgcolor="#0e1117",
        font_color="#ffffff",
        directed=True,
    )

    for node_id, data in graph.nodes(data=True):
        label = data.get("canonical_name", str(node_id))
        etype = data.get("entity_type", "ENTITY")
        ns = data.get("graph_namespace", "")
        v_status = data.get("verification_status", "UNVERIFIED")

        # Color coding
        if node_id in seed_set:
            color = "#ef4444"  # Red for seed
            title = f"[SEED ENTITY]\nName: {label}\nType: {etype}\nNamespace: {ns}"
            size = 28
        elif node_id in exp_set:
            color = "#10b981"  # Green for expanded
            title = f"[EXPANDED ENTITY]\nName: {label}\nType: {etype}\nNamespace: {ns}\nStatus: {v_status}"
            size = 22
        elif ns == "ns_ops":
            color = "#3b82f6"  # Blue for Ops
            title = f"Name: {label}\nType: {etype}\nNamespace: {ns}"
            size = 18
        elif ns == "ns_rd":
            color = "#8b5cf6"  # Purple for R&D
            title = f"Name: {label}\nType: {etype}\nNamespace: {ns}"
            size = 18
        else:
            color = "#64748b"
            title = f"Name: {label}\nType: {etype}"
            size = 16

        net.add_node(
            str(node_id),
            label=label,
            color=color,
            title=title,
            size=size,
        )

    for src, tgt, key, data in graph.edges(keys=True, data=True):
        pred = data.get("predicate", "RELATED_TO")
        is_cross = data.get("is_cross_domain", False)
        edge_color = "#f59e0b" if is_cross else "#94a3b8"
        edge_title = f"Predicate: {pred}\nCross-domain: {is_cross}\nConf: {data.get('confidence', 1.0)}"
        net.add_edge(
            str(src),
            str(tgt),
            label=pred,
            color=edge_color,
            title=edge_title,
            dashes=is_cross,
        )

    net.set_options("""
    var options = {
      "nodes": {
        "font": {"size": 13, "color": "#ffffff"}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
        "color": {"inherit": false},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      }
    }
    """)
    return net.generate_html()


def render_multi_domain_lab_tab(db: DatabaseManager):
    st.header("🧭 Multi-Domain GraphRAG Lab")
    st.caption(
        "Trải nghiệm truy xuất và mở rộng đồ thị tri thức trên các miền dữ liệu tổng hợp (Synthetic Ops & R&D), "
        "với cơ chế cách ly Graph Namespace, lọc quyền truy cập ACL và kiểm tra nguồn gốc Provenance."
    )

    # 1. Configuration & Security Context
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 1.3])
    with ctrl_col1:
        domain_choice = st.selectbox(
            "Phạm vi Miền dữ liệu (Target Domain):",
            [
                "synth_domain_ops (Hạ tầng & Vận hành)",
                "synth_domain_rd (Nghiên cứu & Phát triển)",
                "🌐 Multi-Domain (Cả hai miền)",
            ],
            index=0,
        )
        if "ops" in domain_choice:
            target_domain_ids = ["synth_domain_ops"]
            target_namespaces = ["ns_ops"]
        elif "rd" in domain_choice:
            target_domain_ids = ["synth_domain_rd"]
            target_namespaces = ["ns_rd"]
        else:
            target_domain_ids = ["synth_domain_ops", "synth_domain_rd"]
            target_namespaces = ["ns_ops", "ns_rd"]

    with ctrl_col2:
        user_role = st.selectbox(
            "Vai trò Người dùng (Security Role):",
            ["ops", "developer", "researcher", "sec_admin", "public"],
            index=0,
            help="Kiểm tra cơ chế bảo mật ACL Policy ở cấp độ chunk.",
        )

    with ctrl_col3:
        security_cls = st.multiselect(
            "Cấp độ Bảo mật được phép (Classifications):",
            ["PUBLIC", "INTERNAL", "CONFIDENTIAL"],
            default=["PUBLIC", "INTERNAL"],
        )

    st.divider()

    # 2. Preset Benchmark Questions & Query Input
    preset_questions = [
        "Cổng Ingress Gateway định tuyến lưu lượng vào hệ thống và phụ thuộc vào thành phần nào?",
        "Nguyên nhân sự cố INC-2024-001 là gì và liên quan đến dịch vụ nào?",
        "Giao thức FastLease mở rộng từ thuật toán nào và cải thiện điều gì?",
        "Bộ đo benchmark nào được sử dụng để đánh giá module SparseAttention?",
        "Nhập câu hỏi tự do...",
    ]
    selected_preset = st.selectbox("Chọn câu hỏi mẫu benchmark kỹ thuật:", preset_questions, index=0)
    if selected_preset != "Nhập câu hỏi tự do...":
        query_text = st.text_input("Câu hỏi truy vấn:", value=selected_preset)
    else:
        query_text = st.text_input("Câu hỏi truy vấn:", value="Cổng Ingress Gateway kết nối dịch vụ nào?")

    # Advanced Toggles
    with st.expander("⚙️ Tùy chọn Nâng cao cho Retrieval & Graph Expansion", expanded=False):
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            enable_kg = st.checkbox("Bật KG Expansion", value=True)
        with t_col2:
            include_candidates = st.checkbox("Bao gồm Candidate Edges", value=False)
        with t_col3:
            allow_cross_domain = st.checkbox("Cho phép liên kết Cross-Domain", value=True)
        with t_col4:
            max_hops = st.selectbox("Số bước duyệt (Max Hops):", [1, 2], index=0)

    # 3. Execution
    if st.button("🔎 Thực Thi Truy Vấn Đa Miền (Search & Expand)", type="primary"):
        with st.spinner("Đang truy xuất và duyệt đồ thị theo namespace..."):
            adapter = DeterministicHashEmbeddingAdapter(dimension=64)
            idx_mgr = MultiDomainIndexManager(db)
            retriever = MultiDomainHybridRetriever(db, idx_mgr, adapter)
            graph_builder = MultiDomainGraphBuilder(db)
            expander = NamespaceGraphExpander(db, graph_builder)

            # Retrieval
            seed_results = retriever.retrieve(
                query=query_text,
                domain_ids=target_domain_ids,
                top_k=3,
                allowed_classifications=security_cls,
                user_roles=[user_role],
            )

            # Expansion
            expansion = {}
            if enable_kg and seed_results:
                expansion = expander.expand(
                    seed_results=seed_results,
                    max_hops=max_hops,
                    allowed_namespaces=target_namespaces,
                    allow_cross_domain=allow_cross_domain,
                    include_candidates=include_candidates,
                )

        if not seed_results:
            st.warning("⚠️ Không tìm thấy kết quả nào phù hợp với câu hỏi và chính sách bảo mật đã chọn.")
            return

        # Display Summary
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        sum_col1.metric("Số Chunks Truy xuất (Seeds)", len(seed_results))
        sum_col2.metric("Số Thực thể Mở rộng (Expanded)", len(expansion.get("expanded_entities", [])))
        sum_col3.metric("Số Đường dẫn Đồ thị (Paths)", len(expansion.get("expansion_paths", [])))

        tab_res1, tab_res2, tab_res3 = st.tabs([
            "📑 1. Chunks Truy Xuất (Seeds & Chunks)",
            "🔬 2. Thực Thể Mở Rộng & Bằng Chứng (Provenance Inspector)",
            "🕸️ 3. Namespace Graph Explorer",
        ])

        with tab_res1:
            st.subheader("Danh Sách Canonical Chunks Phù Hợp")
            for idx, chunk in enumerate(seed_results):
                with st.container():
                    st.markdown(f"**[{idx + 1}] Chunk `{chunk.chunk_id}` | Domain: `{chunk.domain_id}` | Score: `{chunk.score:.3f}` | Bảo mật: `{chunk.security_classification}`**")
                    st.info(chunk.content)
                    st.caption(f"Trích xuất từ Document: `{chunk.document_id}` | Source Blocks: `{chunk.source_block_ids}`")
                    st.divider()

        with tab_res2:
            st.subheader("Provenance Inspector: Bằng Chứng Bóc Tách & Chuỗi Dẫn Xuất")
            expanded_ents = expansion.get("expanded_entities", [])
            if not expanded_ents:
                st.info("Không có thực thể nào được mở rộng trong bán kính hops đã chọn.")
            else:
                for ent in expanded_ents:
                    with st.expander(f"🏷️ Thực thể: **{ent['canonical_name']}** ({ent['entity_type']}) - Namespace: `{ent['graph_namespace']}`", expanded=True):
                        st.markdown(f"- **Entity ID:** `{ent['entity_id']}`")
                        st.markdown(f"- **Domain:** `{ent['domain_id']}` | **Trạng thái:** `{ent['verification_status']}`")

                        # Provenance details
                        prov_list = ent.get("provenance", [])
                        if prov_list:
                            st.markdown("**Bằng chứng gốc đã xác minh (EvidenceRecord):**")
                            for ev in prov_list:
                                st.success(f"❝ {ev.get('quote_text', '')} ❞")
                                st.caption(f"Evidence ID: `{ev.get('evidence_id')}` | Extractor: `{ev.get('extractor_id')}` | Origin: `{ev.get('origin_type')}`")
                        else:
                            st.caption("Chưa có bản ghi EvidenceRecord trực tiếp.")

        with tab_res3:
            st.subheader("Đồ Thị Tri Thức Trong Giới Hạn Namespace (Interactive PyVis Graph)")
            # Build visualization graph
            g_vis = graph_builder.build_multi_namespace_graph(
                namespaces=target_namespaces,
                include_cross_domain_links=allow_cross_domain,
                include_candidates=include_candidates,
            )
            seed_ents = expansion.get("seed_entity_ids", [])
            exp_ents = [e["entity_id"] for e in expansion.get("expanded_entities", [])]

            html_code = build_platform_pyvis_html(
                g_vis,
                seed_entity_ids=seed_ents,
                expanded_entity_ids=exp_ents,
                height="540px",
            )
            components.html(html_code, height=560)
            st.caption("🔴 Đỏ: Seed Entities từ Chunks | 🟢 Xanh lá: Expanded Entities qua KG | 🔵 Xanh dương: Ops Nodes | 🟣 Tím: R&D Nodes | 🟠 Nét đứt: Liên kết Cross-Domain")
