from __future__ import annotations

import os
import json
import time
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd

from storage.db import DatabaseManager


def format_bytes(size: int) -> str:
    """Định dạng kích thước byte sang KB hoặc MB dễ đọc."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.2f} MB"


def render_data_repository_tab(db: DatabaseManager):
    st.header("🗄️ Quản Lý Kho Dữ Liệu & Tri Thức")
    st.caption(
        "Theo dõi, quản lý tập trung toàn bộ tài liệu đã nạp, kiểm tra các phân đoạn (chunks), "
        "thực thể, quan hệ, xuất dữ liệu đồ thị tri thức và dọn dẹp kho lưu trữ."
    )

    # =========================================================================
    # PHẦN 1: TỔNG QUAN KHO DỮ LIỆU & DUNG LƯỢNG VẬT LÝ
    # =========================================================================
    counts = db.count_summary()
    db_path = getattr(db, "db_path", "artifacts/legal_lab.db")

    db_size_str = "N/A"
    db_modified_str = "N/A"
    if os.path.exists(db_path):
        stat = os.stat(db_path)
        db_size_str = format_bytes(stat.st_size)
        db_modified_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

    st.subheader("📊 Tổng Quan Dung Lượng & Tài Nguyên Lưu Trữ")

    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("📄 Tài liệu đã nạp", counts.get("platform_documents", 0))
    m_col2.metric("🧱 Khối TextUnits", counts.get("platform_chunks", 0))
    m_col3.metric("🏷️ Thực thể (Entities)", counts.get("platform_entities", 0))
    m_col4.metric("🔗 Mối quan hệ", counts.get("platform_relationships", 0))
    m_col5.metric("👥 Cụm cộng đồng", counts.get("graphrag_communities", 0))
    m_col6.metric("📑 Báo cáo tóm tắt", counts.get("graphrag_community_reports", 0))

    st.caption(f"📁 **Vị trí Database SQLite:** `{db_path}` | **Dung lượng:** `{db_size_str}` | **Cập nhật gần nhất:** `{db_modified_str}`")

    st.markdown("---")

    # =========================================================================
    # PHẦN 2: QUẢN LÝ TÀI LIỆU ĐÃ NẠP (DOCUMENTS)
    # =========================================================================
    st.subheader("📄 Quản Lý Danh Sách Tài Liệu Đã Nạp")

    docs = db.get_platform_documents()

    if not docs:
        st.info("Chưa có tài liệu nào được nạp vào hệ thống. Bạn có thể sang **Tab '1. Nạp Dữ Liệu & Dựng Đồ Thị'** để bắt đầu nạp tài liệu.")
    else:
        doc_rows = []
        for d in docs:
            # Count chunks and blocks for this doc
            doc_id = d["document_id"]
            c_list = db.get_platform_chunks(document_id=doc_id)
            doc_rows.append({
                "Mã tài liệu (ID)": doc_id,
                "Tiêu đề": d.get("title", "Không có tiêu đề"),
                "Không gian (Domain)": d.get("domain_id", "graphrag_default"),
                "Loại văn bản": d.get("document_type", "general"),
                "Số Chunks": len(c_list),
                "Bảo mật": d.get("security_classification", "INTERNAL"),
                "Ngày nạp": str(d.get("ingested_at", ""))[:19],
            })

        st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)

        # Document actions
        doc_ids = [d["document_id"] for d in docs]
        doc_titles = {d["document_id"]: f"{d.get('title', 'Tài liệu')} ({d['document_id']})" for d in docs}

        st.markdown("##### 🔍 Kiểm Tra Chi Tiết & Thao Tác Trên Từng Tài Liệu:")
        sel_doc_col, action_doc_col = st.columns([1.5, 1])

        with sel_doc_col:
            selected_doc_id = st.selectbox(
                "Chọn tài liệu cần xem chi tiết:",
                doc_ids,
                format_func=lambda did: doc_titles.get(did, did),
            )

        with action_doc_col:
            st.write("")
            st.write("")
            confirm_del_doc = st.checkbox("Xác nhận xóa tài liệu này", key=f"chk_del_{selected_doc_id}")
            if st.button("🗑️ Xóa Tài Liệu Này Khỏi Hệ Thống", disabled=not confirm_del_doc, type="secondary"):
                deleted = db.delete_platform_document(selected_doc_id)
                if deleted:
                    st.success(f"Đã xóa thành công tài liệu `{selected_doc_id}` và toàn bộ chunks liên quan!")
                    time.sleep(1)
                    st.rerun()

        if selected_doc_id:
            sel_doc_meta = db.get_platform_document(selected_doc_id)
            sel_chunks = db.get_platform_chunks(document_id=selected_doc_id)
            sel_blocks = db.get_platform_blocks(selected_doc_id)

            with st.expander(f"📖 Chi tiết tài liệu: **{sel_doc_meta.get('title', selected_doc_id)}** ({len(sel_chunks)} Chunks, {len(sel_blocks)} Blocks)", expanded=False):
                d_tab1, d_tab2 = st.tabs([f"🧱 Danh sách Chunks ({len(sel_chunks)})", f"📑 Danh sách Blocks ({len(sel_blocks)})"])
                with d_tab1:
                    chunk_table = [
                        {
                            "Chunk ID": c["chunk_id"],
                            "Tokens": c["token_count"],
                            "Chiến lược": c.get("chunking_strategy", ""),
                            "Nội dung trích đoạn": c["content"][:160] + "...",
                        }
                        for c in sel_chunks
                    ]
                    st.dataframe(pd.DataFrame(chunk_table), use_container_width=True) if chunk_table else st.info("Không có chunk nào.")
                with d_tab2:
                    block_table = [
                        {
                            "Block ID": b["block_id"],
                            "Loại block": b["block_type"],
                            "Offsets": f"{b['start_offset']} -> {b['end_offset']}",
                            "Nội dung": b["text"][:140] + "...",
                        }
                        for b in sel_blocks
                    ]
                    st.dataframe(pd.DataFrame(block_table), use_container_width=True) if block_table else st.info("Không có block nào.")

    st.markdown("---")

    # =========================================================================
    # PHẦN 3: QUẢN LÝ KHÔNG GIAN TRI THỨC (NAMESPACE) & XUẤT DỮ LIỆU
    # =========================================================================
    st.subheader("🌐 Quản Lý Không Gian Tri Thức & Xuất Dữ Liệu Đồ Thị (Export)")

    db_domains = db.get_platform_domains()
    avail_namespaces = ["graphrag_default"]
    for d in db_domains:
        ns = d.get("graph_namespace") or d.get("domain_id")
        if ns and ns not in avail_namespaces:
            avail_namespaces.append(ns)

    target_ns = st.selectbox(
        "Chọn Không gian Tri thức (Namespace) để quản lý:",
        avail_namespaces,
        index=0,
        help="Mỗi namespace hoạt động độc lập và chứa một đồ thị tri thức riêng biệt.",
    )

    ns_entities = db.get_platform_entities(domain_id=target_ns, namespace=target_ns)
    ns_rels = db.get_platform_relationships(domain_id=target_ns, namespace=target_ns, include_unverified=True)
    ns_communities = db.get_graphrag_communities(namespace=target_ns)
    ns_reports = db.get_graphrag_community_reports(namespace=target_ns)
    ns_chunks = db.get_platform_chunks(domain_id=target_ns)

    st.markdown(f"""
    **Thống kê hiện tại của Namespace `{target_ns}`:**
    - Khối văn bản (Chunks): `{len(ns_chunks)}` | Thực thể (Entities): `{len(ns_entities)}` | Quan hệ (Relationships): `{len(ns_rels)}`
    - Cụm cộng đồng (Communities): `{len(ns_communities)}` | Báo cáo tóm tắt (Reports): `{len(ns_reports)}`
    """)

    exp_col1, exp_col2 = st.columns([1.2, 1.2])

    with exp_col1:
        st.markdown("##### 📥 Xuất Dữ Liệu Đồ Thị Tri Thức:")
        # Prepare export bundle JSON
        export_bundle = {
            "namespace": target_ns,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stats": {
                "entities_count": len(ns_entities),
                "relationships_count": len(ns_rels),
                "communities_count": len(ns_communities),
                "reports_count": len(ns_reports),
            },
            "entities": ns_entities,
            "relationships": ns_rels,
            "community_reports": ns_reports,
        }
        json_export_str = json.dumps(export_bundle, ensure_ascii=False, indent=2)

        st.download_button(
            label=f"💾 Tải Toàn Bộ Đồ Thị `{target_ns}` (.JSON)",
            data=json_export_str,
            file_name=f"knowledge_graph_{target_ns}.json",
            mime="application/json",
            help="Tải file JSON chứa toàn bộ thực thể, mối quan hệ và báo cáo cộng đồng.",
        )

        # CSV Exports
        if ns_entities:
            df_ent = pd.DataFrame([
                {
                    "entity_id": e["entity_id"],
                    "canonical_name": e["canonical_name"],
                    "entity_type": e["entity_type"],
                    "aliases": ", ".join(e.get("aliases", [])),
                    "description": e.get("description", ""),
                }
                for e in ns_entities
            ])
            st.download_button(
                label=f"📊 Tải Bảng Thực Thể (.CSV)",
                data=df_ent.to_csv(index=False).encode("utf-8"),
                file_name=f"entities_{target_ns}.csv",
                mime="text/csv",
            )

        if ns_rels:
            df_rel = pd.DataFrame([
                {
                    "relationship_id": r["relationship_id"],
                    "subject": r["subject_entity_id"],
                    "predicate": r["predicate"],
                    "object": r["object_entity_id"],
                    "weight": r.get("weight", 1.0),
                    "description": r.get("description", ""),
                }
                for r in ns_rels
            ])
            st.download_button(
                label=f"📊 Tải Bảng Mối Quan Hệ (.CSV)",
                data=df_rel.to_csv(index=False).encode("utf-8"),
                file_name=f"relationships_{target_ns}.csv",
                mime="text/csv",
            )

    with exp_col2:
        st.markdown("##### 🧹 Dọn Dẹp / Reset Không Gian Tri Thức:")
        st.caption("Thao tác này sẽ xóa sạch các Thực thể, Mối quan hệ, Cụm cộng đồng, Báo cáo và Chunks trong namespace này.")
        confirm_clear_ns = st.checkbox(f"Tôi xác nhận muốn xóa sạch tri thức namespace `{target_ns}`", key=f"chk_clear_{target_ns}")
        if st.button("🧹 Dọn Sạch Toàn Bộ Dữ Liệu Namespace", disabled=not confirm_clear_ns, type="primary"):
            db.clear_namespace_artifacts(target_ns)
            # Clear cached index in session_state
            for k in list(st.session_state.keys()):
                if target_ns in k:
                    del st.session_state[k]
            st.success(f"Đã dọn sạch toàn bộ dữ liệu trong namespace `{target_ns}`!")
            time.sleep(1)
            st.rerun()

    st.markdown("---")

    # =========================================================================
    # PHẦN 4: QUẢN LÝ THƯ MỤC FILE NGUỒN (data/user_documents/)
    # =========================================================================
    st.subheader("📁 Quản Lý Thư Mục File Đầu Vào (`data/user_documents/`)")
    user_doc_dir = os.path.join("data", "user_documents")
    os.makedirs(user_doc_dir, exist_ok=True)

    file_col1, file_col2 = st.columns([1.3, 1])

    with file_col1:
        st.markdown("**Danh sách các file hiện có trong thư mục `data/user_documents/`:**")
        local_files = [f for f in os.listdir(user_doc_dir) if os.path.isfile(os.path.join(user_doc_dir, f))]
        if not local_files:
            st.info("Thư mục `data/user_documents/` hiện đang trống.")
        else:
            file_records = []
            for fn in local_files:
                fpath = os.path.join(user_doc_dir, fn)
                st_info = os.stat(fpath)
                file_records.append({
                    "Tên file": fn,
                    "Kích thước": format_bytes(st_info.st_size),
                    "Lần sửa cuối": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st_info.st_mtime)),
                })
            st.dataframe(pd.DataFrame(file_records), use_container_width=True)

            del_file_sel = st.selectbox("Chọn file cần xóa khỏi thư mục:", local_files)
            confirm_del_f = st.checkbox("Xác nhận xóa file vật lý này khỏi ổ đĩa", key="chk_del_f")
            if st.button("🗑️ Xóa File Khỏi Thư Mục", disabled=not confirm_del_f):
                try:
                    os.remove(os.path.join(user_doc_dir, del_file_sel))
                    st.success(f"Đã xóa file `{del_file_sel}` thành công!")
                    time.sleep(1)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Không thể xóa file: {ex}")

    with file_col2:
        st.markdown("**Tải thêm file mới vào thư mục `data/user_documents/`:**")
        uploaded_new_file = st.file_uploader(
            "Chọn file từ máy tính để lưu vào kho:",
            type=["docx", "pdf", "txt", "md", "json"],
            key="uploader_repo",
        )
        if uploaded_new_file is not None:
            dest_path = os.path.join(user_doc_dir, uploaded_new_file.name)
            if st.button(f"📥 Lưu File `{uploaded_new_file.name}` Vào Thư Mục"):
                try:
                    with open(dest_path, "wb") as f_out:
                        f_out.write(uploaded_new_file.read())
                    st.success(f"Đã lưu thành công file vào `{dest_path}`!")
                    time.sleep(1)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Lỗi khi lưu file: {ex}")
