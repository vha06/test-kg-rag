import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from typing import Dict, Any, List
from storage.db import DatabaseManager
from knowledge_graph.builder import KnowledgeGraphBuilder
from knowledge_graph.visualizer import build_pyvis_html, RELATION_COLORS


def render_kg_tab(db: DatabaseManager, builder: KnowledgeGraphBuilder):
    st.header("🕸️ Tab 3: Knowledge Graph Explorer")
    st.markdown("""
    Khám phá đồ thị quan hệ pháp lý có căn cứ được xây dựng từ:
    - **Cấu trúc (`CONTAINS`):** Văn bản → Điều → Khoản → Điểm.
    - **Dẫn chiếu (`REFERENCES`):** 551 liên kết Chỉ dẫn chính thức từ Bộ Pháp điển + Dẫn chiếu trong văn bản.
    - **Sửa đổi / Bổ sung / Bãi bỏ (`AMENDS`, `SUPPLEMENTS`, `REPEALS`):** Trích từ lịch sử sửa đổi chính thức.
    - **Quy định chi tiết (`DETAILS`):** Nghị định/Thông tư hướng dẫn thi hành Luật/Nghị định gốc.
    """)

    graph = builder.graph
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số Nodes", total_nodes)
    col2.metric("Tổng số Edges", total_edges)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM provision_versions")
        v_count = cursor.fetchone()[0]
    col3.metric("Số bản ghi Lịch sử phiên bản", v_count)

    st.divider()

    # 1. Edge Filter and Explorer Table
    st.subheader("1. Danh sách quan hệ pháp lý (Edges)")
    rel_filter = st.multiselect(
        "Lọc theo loại quan hệ:",
        ["REFERENCES", "AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "CONTAINS"],
        default=["AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "REFERENCES"],
    )

    with db.get_connection() as conn:
        cursor = conn.cursor()
        if rel_filter:
            placeholders = ",".join("?" for _ in rel_filter)
            cursor.execute(f"""
                SELECT source_id, relation_type, target_id, evidence_text, effective_date, detection_method
                FROM kg_edges
                WHERE relation_type IN ({placeholders}) AND verified = 1
                LIMIT 50
            """, rel_filter)
        else:
            cursor.execute("SELECT source_id, relation_type, target_id, evidence_text, effective_date, detection_method FROM kg_edges WHERE verified = 1 LIMIT 50")
        edge_rows = cursor.fetchall()

    if edge_rows:
        df_edges = pd.DataFrame([dict(r) for r in edge_rows])
        st.dataframe(df_edges, use_container_width=True)
    else:
        st.info("Không có quan hệ nào phù hợp bộ lọc.")

    st.divider()

    # 2. Node Inspector & Interactive Dynamic Graph
    st.subheader("2. Đồ thị mạng lưới quan hệ trực quan (Interactive Dynamic Graph)")

    c_srch1, c_srch2 = st.columns([2, 1])
    with c_srch1:
        search_kw = st.text_input(
            "🔍 Tìm kiếm nhanh theo Số hiệu văn bản / Tên điều khoản:",
            value="",
            placeholder="Nhập số hiệu hoặc từ khóa (VD: 34/2023, 112/2014, An ninh...)...",
        )
    with c_srch2:
        only_related = st.checkbox("Chỉ hiện điều khoản có liên kết KG (Khuyên dùng)", value=True)

    with db.get_connection() as conn:
        cursor = conn.cursor()
        if search_kw.strip():
            kw_param = f"%{search_kw.strip()}%"
            if only_related:
                cursor.execute("""
                    SELECT DISTINCT p.provision_id, p.title, d.doc_number
                    FROM provisions p
                    JOIN documents d ON p.document_id = d.document_id
                    WHERE (p.title LIKE ? OR d.doc_number LIKE ? OR d.title LIKE ?)
                      AND p.provision_id IN (
                          SELECT source_id FROM kg_edges WHERE relation_type != 'CONTAINS'
                          UNION
                          SELECT target_id FROM kg_edges WHERE relation_type != 'CONTAINS'
                      )
                    ORDER BY d.doc_number, p.title
                    LIMIT 200
                """, (kw_param, kw_param, kw_param))
            else:
                cursor.execute("""
                    SELECT p.provision_id, p.title, d.doc_number
                    FROM provisions p
                    JOIN documents d ON p.document_id = d.document_id
                    WHERE (p.title LIKE ? OR d.doc_number LIKE ? OR d.title LIKE ?)
                    ORDER BY d.doc_number, p.title
                    LIMIT 200
                """, (kw_param, kw_param, kw_param))
        else:
            if only_related:
                cursor.execute("""
                    SELECT DISTINCT p.provision_id, p.title, d.doc_number
                    FROM provisions p
                    JOIN documents d ON p.document_id = d.document_id
                    WHERE p.provision_id IN (
                        SELECT source_id FROM kg_edges WHERE relation_type != 'CONTAINS'
                        UNION
                        SELECT target_id FROM kg_edges WHERE relation_type != 'CONTAINS'
                    )
                    ORDER BY d.doc_number, p.title
                """)
            else:
                cursor.execute("""
                    SELECT p.provision_id, p.title, d.doc_number
                    FROM provisions p
                    JOIN documents d ON p.document_id = d.document_id
                    WHERE p.level = 'ARTICLE'
                    LIMIT 200
                """)
        prov_rows = cursor.fetchall()

    if not prov_rows:
        st.warning("⚠️ Không tìm thấy điều khoản nào khớp với từ khóa tìm kiếm. Vui lòng thử từ khóa khác.")
        return

    prov_options = {f"{r['title']} ({r['doc_number']})": r["provision_id"] for r in prov_rows}

    # Đặt mặc định ưu tiên tìm thấy Điều 1 Nghị định 34/2023/NĐ-CP nếu có
    default_idx = 0
    keys_list = list(prov_options.keys())
    for i, k in enumerate(keys_list):
        if "34/2023" in k and "Điều 1" in k:
            default_idx = i
            break

    selected_label = st.selectbox(
        "Chọn Điều khoản để trực quan hóa mạng lưới quan hệ liên kết:",
        keys_list,
        index=default_idx,
    )
    selected_id = prov_options[selected_label]

    if graph.has_node(selected_id):
        node_data = graph.nodes[selected_id]

        # Interactive Graph Controls
        cg1, cg2, cg3 = st.columns([1, 2, 1])
        with cg1:
            graph_hops = st.selectbox("Độ sâu đồ thị:", [1, 2], index=0, format_func=lambda h: f"{h} hop (Láng giềng trực tiếp)" if h == 1 else f"{h} hops (Mở rộng 2 tầng)")
        with cg2:
            graph_rels = st.multiselect(
                "Lọc quan hệ vẽ trên đồ thị:",
                ["REFERENCES", "AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "CONTAINS"],
                default=["AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "REFERENCES", "CONTAINS"],
            )
        with cg3:
            enable_physics = st.checkbox("Mô phỏng vật lý động (Physics)", value=True)

        # Legend explanation
        st.markdown("""
        **Chú giải màu sắc:** 
        🟡 **Vàng**: Điều khoản đang chọn &nbsp;|&nbsp; 
        🔵 **Xanh dương**: Văn bản pháp luật cha &nbsp;|&nbsp; 
        🟢 **Xanh lá**: Điều khoản liên kết &nbsp;|&nbsp; 
        🔴 **Đỏ**: Điều khoản sửa đổi/bổ sung (`AMENDS`) &nbsp;|&nbsp; 
        🟣 **Tím**: Dẫn chiếu (`REFERENCES`) &nbsp;|&nbsp; 
        🌐 **Xanh lơ**: Quy định chi tiết (`DETAILS`)
        """)

        # Generate & Embed Interactive PyVis Network
        with st.spinner("Đang vẽ đồ thị tương tác..."):
            graph_html = build_pyvis_html(
                graph=graph,
                center_node_id=selected_id,
                max_hops=graph_hops,
                allowed_relations=graph_rels,
                height="540px",
                physics=enable_physics,
            )
            components.html(graph_html, height=550, scrolling=False)

        st.caption("💡 *Mẹo tương tác: Bạn có thể kéo thả các node, cuộn chuột để phóng to/thu nhỏ (Zoom), nhấp chuột vào node hoặc di chuột qua đường nối để xem chi tiết căn cứ và loại quan hệ.*")

        st.divider()

        st.markdown(f"### Chi tiết: {node_data.get('title')}")
        st.caption(f"**Đường dẫn:** {node_data.get('structural_path')} | **Hiệu lực từ:** `{node_data.get('effective_start')}` | **Trạng thái:** `{node_data.get('status')}`")
        st.info(node_data.get("content"))

        # In-edges and out-edges text inspection
        c_in, c_out = st.columns(2)
        with c_in:
            st.markdown("#### ⬅️ Quan hệ trỏ vào (Incoming Edges)")
            in_edges = graph.in_edges(selected_id, data=True)
            if not in_edges:
                st.caption("Không có quan hệ trỏ vào.")
            for src, _, data in in_edges:
                src_node = graph.nodes.get(src, {})
                rel = data.get("relation_type", "")
                st.markdown(f"- **{src_node.get('title', src)}** `--[{rel}]-->` `{selected_id}`")
                if data.get("evidence_text"):
                    st.caption(f"  *Căn cứ:* {data.get('evidence_text')}")

        with c_out:
            st.markdown("#### ➡️ Quan hệ trỏ ra (Outgoing Edges)")
            out_edges = graph.out_edges(selected_id, data=True)
            if not out_edges:
                st.caption("Không có quan hệ trỏ ra.")
            for _, tgt, data in out_edges:
                tgt_node = graph.nodes.get(tgt, {})
                rel = data.get("relation_type", "")
                st.markdown(f"- `{selected_id}` `--[{rel}]-->` **{tgt_node.get('title', tgt)}**")
                if data.get("evidence_text"):
                    st.caption(f"  *Căn cứ:* {data.get('evidence_text')}")

        # Check version history
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM provision_versions WHERE provision_id = ? ORDER BY version_num ASC", (selected_id,))
            versions = cursor.fetchall()

        if versions:
            st.markdown("#### ⏳ Chuỗi lịch sử phiên bản (Temporal Versioning)")
            for v in versions:
                v_badge = ":green-background[Phiên bản hiện hành]" if v["status"] == "ACTIVE_AMENDED" else ":orange-background[Phiên bản cũ trước sửa đổi]"
                st.markdown(f"**Phiên bản {v['version_num']}** | {v_badge}")
                st.caption(f"Áp dụng: từ `{v['effective_start']}` đến `{v['effective_end'] or 'Nay'}`")
                st.text(v["version_note"])
