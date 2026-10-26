from __future__ import annotations

import os
import json
import hashlib
from typing import Optional, Any, List, Dict
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis.network import Network

from core.registry import DomainRegistry, ProfileRegistry, DomainConfig, ProcessingProfile
from core.schema import (
    CanonicalDocument,
    ContentBlock,
    CanonicalChunk,
    CanonicalEntity,
    EntityMention,
    CanonicalRelationship,
    EvidenceRecord,
    ACLPolicy,
)
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.chunkers import ChunkerRegistry
from pipeline.parsers import ParserRegistry, extract_text_from_file
from pipeline.extractors import ExtractorRegistry
from storage.db import DatabaseManager
from embedding.adapter import DeterministicHashEmbeddingAdapter, EmbeddingAdapter


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


def render_community_pyvis_html(entities: List[Any], relationships: List[Any], communities: List[Any] = None, height: str = "540px") -> str:
    """
    Vẽ đồ thị cộng đồng phân cấp L0/L1/L2, tô màu theo cụm cộng đồng.
    Hỗ trợ cả Pydantic schema model lẫn dict bản ghi từ database.
    """
    net = Network(height=height, width="100%", bgcolor="#0e1117", font_color="#ffffff", directed=True)
    palette = [
        "#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6",
        "#06b6d4", "#f97316", "#14b8a6", "#a855f7", "#e11d48",
    ]
    ent_comm_map: Dict[str, Any] = {}
    if communities:
        for idx, comm in enumerate(communities):
            c_color = palette[idx % len(palette)]
            cid = _safe_get(comm, "community_id", f"COMM_{idx}")
            c_title = _safe_get(comm, "title") or f"Community {cid}"
            c_level = _safe_get(comm, "level", 0)
            e_ids = _safe_get(comm, "entity_ids", [])
            if isinstance(e_ids, str):
                try:
                    e_ids = json.loads(e_ids)
                except Exception:
                    e_ids = []
            for e_id in (e_ids or []):
                ent_comm_map[str(e_id)] = (c_title, c_level, c_color)

    for e in entities:
        eid = str(_safe_get(e, "entity_id", ""))
        if not eid:
            continue
        label = str(_safe_get(e, "canonical_name", eid))
        etype = str(_safe_get(e, "entity_type", "ENTITY"))
        comm_meta = ent_comm_map.get(eid)
        if comm_meta:
            c_title, c_level, c_color = comm_meta
            title_text = f"Entity: {label}\nType: {etype}\nCommunity (L{c_level}): {c_title}\nID: {eid}"
            node_color = c_color
        else:
            title_text = f"Entity: {label}\nType: {etype}\nID: {eid}"
            node_color = "#94a3b8"

        net.add_node(
            eid,
            label=label,
            color=node_color,
            title=title_text,
            size=22,
        )

    for r in relationships:
        src = str(_safe_get(r, "subject_entity_id", ""))
        tgt = str(_safe_get(r, "object_entity_id", ""))
        if not src or not tgt:
            continue
        pred = str(_safe_get(r, "predicate", "RELATED_TO"))
        w = float(_safe_get(r, "weight", 1.0) or 1.0)
        net.add_edge(
            src,
            tgt,
            label=pred,
            color="#cbd5e1",
            title=f"Predicate: {pred}\nWeight: {w}",
        )

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


def render_platform_overview_tab(db: DatabaseManager, llm_config: Optional[Any] = None):
    st.header("📥 Tiếp Nhận Dữ Liệu & Xây Dựng Đồ Thị Tri Thức")
    st.caption(
        "Tải lên hoặc chọn tài liệu, tự động phân đoạn văn bản (Chunking), "
        "trích xuất Thực thể & Quan hệ và xây dựng Đồ thị tri thức phân cấp cụm cộng đồng (GraphRAG)."
    )

    domain_reg = DomainRegistry()
    yaml_path = "config/platform_profiles.yaml"
    try:
        domain_reg.load_from_yaml(yaml_path)
    except Exception:
        pass

    # Load custom domains from DB
    try:
        for d in db.get_platform_domains():
            if not domain_reg.get_domain(d["domain_id"]):
                domain_reg.register_domain(DomainConfig.model_validate(d))
    except Exception:
        pass

    profile_reg = ProfileRegistry(domain_reg)
    try:
        profile_reg.load_from_yaml(yaml_path)
    except Exception:
        pass

    # Load custom profiles from DB
    try:
        for p in db.get_platform_profiles():
            if not profile_reg.get_profile(p["profile_id"]):
                profile_reg.register_profile(ProcessingProfile.model_validate(p))
    except Exception:
        pass

    # -------------------------------------------------------------
    # BƯỚC 1: FILE VĂN BẢN VÀ METADATA ĐẦU VÀO
    # -------------------------------------------------------------
    st.markdown("#### 📥 Bước 1: Tiếp Nhận Dữ Liệu Đầu Vào & Khai Báo Metadata")

    in_col1, in_col2 = st.columns([1.1, 0.9])
    with in_col1:
        input_mode = st.radio(
            "Nguồn cung cấp nội dung:",
            [
                "📚 Chọn file từ data/user_documents/",
                "📁 Tải file lên (.txt, .md, .docx, .pdf, .json)",
                "✍️ Nhập / Dán văn bản trực tiếp",
            ],
            horizontal=True,
        )
        raw_file_text = ""
        default_title = "Thông Cáo Báo Chí 2015"
        file_meta = {}

        if "data/user_documents" in input_mode:
            user_doc_dir = os.path.join("data", "user_documents")
            os.makedirs(user_doc_dir, exist_ok=True)
            avail_files = [f for f in os.listdir(user_doc_dir) if os.path.isfile(os.path.join(user_doc_dir, f))]
            if avail_files:
                selected_file = st.selectbox("Chọn file từ thư mục data/user_documents/:", avail_files)
                if selected_file:
                    fpath = os.path.join(user_doc_dir, selected_file)
                    try:
                        with open(fpath, "rb") as f_obj:
                            content_bytes = f_obj.read()
                        if selected_file.endswith(".json"):
                            decoded_text = content_bytes.decode("utf-8", errors="replace")
                            parsed_json = json.loads(decoded_text)
                            if isinstance(parsed_json, dict) and "content" in parsed_json:
                                raw_file_text = parsed_json.get("content", "")
                                file_meta = parsed_json.get("metadata", {})
                            else:
                                raw_file_text = decoded_text
                        else:
                            raw_file_text = extract_text_from_file(content_bytes, selected_file)
                        default_title = selected_file.rsplit(".", 1)[0].replace("_", " ").title()
                        st.success(f"Đã nạp file: **{selected_file}** ({len(raw_file_text)} ký tự)")
                    except Exception as err:
                        st.error(f"Lỗi khi đọc file: {err}")
            else:
                st.info("Chưa có file nào trong thư mục `data/user_documents/`. Vui lòng đặt file vào thư mục này hoặc dùng tính năng Tải file.")

        elif "Tải file" in input_mode:
            uploaded_doc = st.file_uploader(
                "Tải lên file văn bản:",
                type=["txt", "md", "markdown", "json", "docx", "pdf"],
                help="Hỗ trợ file Text, Markdown, Docx, PDF hoặc JSON có trường content/metadata.",
            )
            if uploaded_doc is not None:
                try:
                    content_bytes = uploaded_doc.read()
                    if uploaded_doc.name.endswith(".json"):
                        decoded_text = content_bytes.decode("utf-8", errors="replace")
                        parsed_json = json.loads(decoded_text)
                        if isinstance(parsed_json, dict) and "content" in parsed_json:
                            raw_file_text = parsed_json.get("content", "")
                            file_meta = parsed_json.get("metadata", {})
                        else:
                            raw_file_text = decoded_text
                    else:
                        raw_file_text = extract_text_from_file(content_bytes, uploaded_doc.name)
                    default_title = uploaded_doc.name.rsplit(".", 1)[0].replace("_", " ").title()
                    st.success(f"Đã đọc file: **{uploaded_doc.name}** ({len(raw_file_text)} ký tự)")
                except Exception as err:
                    st.error(f"Lỗi khi đọc file: {err}")

        fallback_sample = (
            "Kinh tế - xã hội nước ta năm 2015 diễn ra trong bối cảnh kinh tế thế giới phục hồi chậm.\n"
            "Tổng sản phẩm trong nước (GDP) năm 2015 ước tính tăng 6,68% so với năm 2014, cao hơn mục tiêu 6,2% đề ra.\n"
            "Trong đó, khu vực công nghiệp và xây dựng tăng 9,64%, đóng góp lớn nhất vào mức tăng trưởng chung."
        )
        initial_text = raw_file_text if raw_file_text else fallback_sample
        input_raw_text = st.text_area(
            "Xem trước nội dung văn bản đầu vào (Raw text):",
            value=initial_text,
            height=140,
        )

    with in_col2:
        st.markdown("**Khai báo Metadata của Tài liệu:**")
        test_doc_title = st.text_input("Tiêu đề Tài liệu:", value=file_meta.get("title", default_title))

        existing_domains = [d.domain_id for d in domain_reg.list_domains(enabled_only=False)]
        default_doms = ["graphrag_default"] + [d for d in existing_domains if d != "graphrag_default"]
        domain_choice_list = default_doms + ["➕ Tạo Domain mới..."]
        selected_domain_raw = st.selectbox("Chọn Không gian Tri thức / Namespace:", domain_choice_list, index=0)

        if selected_domain_raw == "➕ Tạo Domain mới...":
            new_dom_col1, new_dom_col2 = st.columns(2)
            with new_dom_col1:
                new_domain_id = st.text_input("Mã Domain mới (ID):", value="domain_kinh_te")
            with new_dom_col2:
                new_domain_name = st.text_input("Tên Domain mới:", value="Lĩnh vực Kinh tế")
            selected_domain = new_domain_id
            if not domain_reg.get_domain(new_domain_id):
                new_dom_obj = DomainConfig(
                    domain_id=new_domain_id,
                    name=new_domain_name,
                    graph_namespace=f"ns_{new_domain_id.replace('synth_domain_', '').replace('domain_', '')}",
                )
                domain_reg.register_domain(new_dom_obj)
        else:
            selected_domain = selected_domain_raw

        meta_c1, meta_c2 = st.columns(2)
        with meta_c1:
            selected_doc_type = st.selectbox(
                "Loại văn bản (Document Type):",
                ["general", "press_release", "report", "legal_regulation", "technical_spec"],
                index=0,
            )
        with meta_c2:
            selected_sec_class = st.selectbox(
                "Cấp độ Bảo mật (Security):",
                ["PUBLIC", "INTERNAL", "CONFIDENTIAL"],
                index=0,
            )

    st.markdown("---")
    # -------------------------------------------------------------
    # BƯỚC 2: PHÂN ĐOẠN VĂN BẢN (CHUNKING)
    # -------------------------------------------------------------
    st.markdown("#### 🧩 Bước 2: Phân Đoạn Văn Bản (Chunking Strategy)")

    chunk_col1, chunk_col2, chunk_col3, chunk_col4 = st.columns([1.3, 1.2, 1, 1])
    with chunk_col1:
        chunker_options = {
            "recursive_boundary": "Recursive Boundary-Aware (Đệ quy: Tiêu đề -> Đoạn -> Câu -> Tokens)",
            "fixed_token": "Fixed Token (Cắt cố định theo token window)",
            "block_preserving": "Block Preserving (Bảo toàn trọn vẹn khối đoạn/mục)",
            "sliding_window": "Sliding Window (Cửa sổ trượt có overlap)",
            "hierarchical_heading": "Hierarchical Heading (Phân cấp theo tiêu đề mục)",
        }
        selected_chunker_key = st.selectbox(
            "Cơ chế Chunking:",
            list(chunker_options.keys()),
            format_func=lambda k: chunker_options[k],
            index=0,
        )

    with chunk_col2:
        parser_options = {
            "generic_paragraph": "Generic Paragraph (Tách theo đoạn văn & tiêu đề)",
            "structured_heading": "Structured Heading (Nhận diện tiêu đề #, Điều, Mục)",
        }
        selected_parser_key = st.selectbox(
            "Cơ chế Phân Tách Khối (Parser):",
            list(parser_options.keys()),
            format_func=lambda k: parser_options[k],
            index=0,
        )

    with chunk_col3:
        max_tokens_val = st.slider("Kích thước Chunk (Tokens):", min_value=64, max_value=1024, value=300, step=32)

    with chunk_col4:
        overlap_val = st.slider("Độ chồng lấp (Overlap):", min_value=0, max_value=128, value=50, step=8)

    st.markdown("---")
    # -------------------------------------------------------------
    # BƯỚC 3: CẤU HÌNH TRÍCH XUẤT THỰC THỂ & QUAN HỆ
    # -------------------------------------------------------------
    st.markdown("#### 🏷️ Bước 3: Cấu Hình Trích Xuất Tri Thức & Mô Hình Embedding")
    ext_col1, ext_col2, ext_col3 = st.columns([1.4, 1.3, 1.1])

    with ext_col1:
        extractor_labels = {
            "deterministic_general": "Deterministic General Extractor (Bóc tách chuẩn xác có Grounding, 100% Offline)",
            "graphrag_extractor": "GraphRAG Chunk-level LLM Extractor (Dùng LLM đã chọn ở Sidebar để bóc tách thông minh)",
            "rule_based_ops": "Rule-based & Regex Heuristic (Chạy nhanh, không cần LLM)",
        }
        selected_extractor_key = st.selectbox(
            "Phương thức trích xuất tri thức (Extractor):",
            list(extractor_labels.keys()),
            format_func=lambda k: extractor_labels[k],
            index=0,
        )

    with ext_col2:
        embedding_choice = st.selectbox(
            "Bộ tạo Vector (Embedding Model):",
            [
                "bkai-foundation-models/vietnamese-bi-encoder (Khuyên dùng)",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "Deterministic Hash Adapter (Tốc độ cao / Offline)",
            ],
            index=0,
        )

    with ext_col3:
        clear_existing_toggle = st.checkbox(
            "🧹 Xóa tri thức cũ cùng Namespace khi Build",
            value=True,
            help="Tránh trùng lặp khi xây dựng lại đồ thị cho cùng một namespace.",
        )

    st.markdown("---")
    # -------------------------------------------------------------
    # BƯỚC 4: THỰC THI XÂY DỰNG ĐỒ THỊ TRI THỨC
    # -------------------------------------------------------------
    st.markdown("#### 🚀 Bước 4: Thực Thi Xây Dựng Đồ Thị Tri Thức & Chỉ Mục GraphRAG")

    if st.button("🚀 Thực Thi Xây Dựng Đồ Thị Tri Thức (Build Knowledge Graph)", type="primary"):
        status_box = st.status("🚀 Đang tiến hành phân đoạn, trích xuất thực thể và dựng đồ thị...", expanded=True)
        prog_bar = st.progress(0.0)

        def update_progress(msg: str, pct: float):
            prog_bar.progress(min(1.0, max(0.0, pct)), text=msg)
            status_box.write(f"⏱️ {msg}")

        try:
            emb_adapter = get_cached_embedding_adapter(embedding_choice)

            import inspect
            orchestrator = PipelineOrchestrator(
                domain_registry=domain_reg,
                profile_registry=profile_reg,
                db_manager=db,
            )

            documents = [{
                "document_id": f"DOC_{hashlib.sha256(input_raw_text.encode('utf-8')).hexdigest()[:10]}",
                "title": test_doc_title,
                "text": input_raw_text,
            }]

            build_kwargs = {
                "documents": documents,
                "namespace": selected_domain,
                "chunking_strategy": selected_chunker_key,
                "chunk_size": max_tokens_val,
                "chunk_overlap": overlap_val,
                "extractor_name": selected_extractor_key,
                "llm_config": llm_config,
                "embedding_adapter": emb_adapter,
                "max_community_levels": 3,
                "clear_existing": clear_existing_toggle,
            }
            sig = inspect.signature(orchestrator.build_full_graphrag_pipeline)
            if "progress_callback" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                build_kwargs["progress_callback"] = update_progress

            build_result = orchestrator.build_full_graphrag_pipeline(**build_kwargs)
            status_box.update(label="✅ Xây dựng đồ thị tri thức hoàn tất!", state="complete", expanded=False)
            st.success(f"🎉 **Hoàn tất xây dựng Đồ thị tri thức trong {build_result.get('elapsed_seconds', 0.0)}s!**")
        except Exception as err:
            status_box.update(label=f"❌ Xây dựng thất bại: {err}", state="error", expanded=True)
            st.error(f"Chi tiết lỗi: {err}")

    # -------------------------------------------------------------
    # HIỂN THỊ ĐỒ THỊ TRI THỨC VÀ CÁC THÀNH PHẦN HIỆN CÓ
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"🕸️ Đồ Thị Tri Thức & Chỉ Mục Trong Namespace `{selected_domain}`")

    chunks_db = db.get_platform_chunks(domain_id=selected_domain)
    entities_db = db.get_platform_entities(domain_id=selected_domain, namespace=selected_domain)
    rels_db = db.get_platform_relationships(domain_id=selected_domain, namespace=selected_domain, include_unverified=True)
    communities_db = db.get_graphrag_communities(selected_domain)
    reports_db = db.get_graphrag_community_reports(selected_domain)

    if entities_db or chunks_db:
        # Metrics summary
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        m_c1.metric("Khối văn bản (Chunks)", len(chunks_db))
        m_c2.metric("Thực thể (Entities)", len(entities_db))
        m_c3.metric("Mối quan hệ (Relations)", len(rels_db))
        m_c4.metric("Cụm cộng đồng (Communities)", len(communities_db))
        m_c5.metric("Báo cáo tóm tắt (Reports)", len(reports_db))

        st.caption("💡 Chuyển sang **Tab '2. Truy Xuất Tri Thức & Hỏi Đáp GraphRAG'** để đặt câu hỏi trên dữ liệu này.")

        g_tab1, g_tab2, g_tab3, g_tab4 = st.tabs([
            f"🕸️ 1. Đồ Thị Tri Thức Trực Quan ({len(entities_db)} Nút, {len(rels_db)} Cạnh)",
            f"🏷️ 2. Thực Thể & Quan Hệ",
            f"👥 3. Cụm Cộng Đồng & Báo Cáo Tóm Tắt ({len(communities_db)} Cụm)",
            f"🧱 4. Danh Sách TextUnits ({len(chunks_db)} Chunks)",
        ])

        with g_tab1:
            st.subheader("Đồ Thị Tri Thức Phân Cụm Cộng Đồng (Interactive Community Knowledge Graph)")
            if entities_db:
                html_net = render_community_pyvis_html(
                    entities=[CanonicalEntity.model_validate(e) for e in entities_db],
                    relationships=[CanonicalRelationship.model_validate(r) for r in rels_db],
                    communities=communities_db,
                    height="540px",
                )
                components.html(html_net, height=560)
                st.caption("🎨 Các node cùng màu thuộc cùng một Cụm Cộng đồng (Community Cluster). Bạn có thể phóng to, thu nhỏ và rê chuột để xem chi tiết.")
            else:
                st.info("Chưa có thực thể nào để hiển thị đồ thị.")

        with g_tab2:
            st.subheader("Thực Thể & Mối Quan Hệ Đã Bóc Tách")
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.markdown("**Thực thể (Canonical Entities):**")
                ent_rows = [
                    {
                        "Tên thực thể": e["canonical_name"],
                        "Phân loại": e["entity_type"],
                        "Bí danh": ", ".join(e.get("aliases", [])),
                        "Mô tả": e.get("description", "")[:120],
                        "Số Chunks nguồn": len(e.get("source_chunk_ids", [])),
                    }
                    for e in entities_db
                ]
                st.dataframe(pd.DataFrame(ent_rows), use_container_width=True) if ent_rows else st.info("Chưa có thực thể.")

            with e_col2:
                st.markdown("**Mối quan hệ (Canonical Relationships):**")
                rel_rows = [
                    {
                        "Thực thể nguồn": r["subject_entity_id"],
                        "Vị từ (Quan hệ)": r["predicate"],
                        "Thực thể đích": r["object_entity_id"],
                        "Trọng số": r.get("weight", 1.0),
                        "Mô tả": r.get("description", "")[:100],
                    }
                    for r in rels_db
                ]
                st.dataframe(pd.DataFrame(rel_rows), use_container_width=True) if rels_db else st.info("Chưa có quan hệ.")

        with g_tab3:
            st.subheader("Cụm Cộng Đồng Phân Cấp & Báo Cáo Tóm Tắt (Community Reports)")
            if not reports_db:
                st.info("Chưa có báo cáo cộng đồng nào được tạo.")
            else:
                for rep in reports_db:
                    with st.expander(f"📑 [Cấp {rep.get('level', 0)}] **{rep.get('title', 'Community Report')}** (ID: `{rep.get('community_id')}`)", expanded=False):
                        st.markdown(f"**Tóm tắt:** {rep.get('summary', '')}")
                        st.markdown(f"**Phát hiện chính (Key Findings):**")
                        findings = rep.get("key_findings", [])
                        if isinstance(findings, list):
                            for f in findings:
                                st.markdown(f"- {f}")
                        st.caption(f"Thực thể hỗ trợ: `{len(rep.get('supporting_entity_ids', []))}` | Quan hệ hỗ trợ: `{len(rep.get('supporting_relationship_ids', []))}`")

        with g_tab4:
            st.subheader("Danh Sách Khối Văn Bản (TextUnits / CanonicalChunks)")
            c_rows = [
                {
                    "Chunk ID": c["chunk_id"],
                    "Tokens": c["token_count"],
                    "Hash": c.get("content_hash", "")[:12],
                    "Mục": " > ".join(c.get("section_path", [])) if isinstance(c.get("section_path"), list) else str(c.get("section_path", "")),
                    "Nội dung": c["content"],
                }
                for c in chunks_db
            ]
            st.dataframe(pd.DataFrame(c_rows), use_container_width=True) if c_rows else st.info("Không có chunk nào.")
    else:
        st.info(f"ℹ️ Không gian tri thức `{selected_domain}` chưa có dữ liệu. Hãy chọn hoặc tải file văn bản và nhấn nút 'Thực Thi Xây Dựng Đồ Thị Tri Thức' ở Bước 4.")
