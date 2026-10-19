from __future__ import annotations

import json
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
from core.validator import CanonicalSchemaValidator
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.chunkers import ChunkerRegistry
from pipeline.parsers import ParserRegistry
from pipeline.extractors import ExtractorRegistry
from storage.db import DatabaseManager


def render_bundle_pyvis_html(entities: List[Any], relationships: List[Any], height: str = "480px") -> str:
    net = Network(height=height, width="100%", bgcolor="#0e1117", font_color="#ffffff", directed=True)
    color_map = {
        "SERVICE": "#3b82f6",     # Blue
        "SERVER": "#06b6d4",      # Cyan
        "DATABASE": "#10b981",    # Emerald
        "PROTOCOL": "#8b5cf6",    # Purple
        "INCIDENT": "#ef4444",    # Red
        "COMPONENT": "#f59e0b",   # Amber
    }
    for e in entities:
        etype = getattr(e, "entity_type", "COMPONENT")
        c = color_map.get(etype, "#94a3b8")
        label = getattr(e, "canonical_name", str(getattr(e, "entity_id", "")))
        net.add_node(
            str(getattr(e, "entity_id", "")),
            label=label,
            color=c,
            title=f"Name: {label}\nType: {etype}\nDomain: {getattr(e, 'domain_id', '')}",
            size=22,
        )
    for r in relationships:
        pred = getattr(r, "predicate", "RELATED_TO")
        net.add_edge(
            str(getattr(r, "subject_entity_id", "")),
            str(getattr(r, "object_entity_id", "")),
            label=pred,
            color="#cbd5e1",
            title=f"Predicate: {pred}\nMethod: {getattr(r, 'extraction_method', 'RULE_BASED')}",
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
    st.header("🌐 Extensible Multi-Domain Platform Architecture")
    st.caption(
        "Khám phá kiến trúc nền tảng GraphRAG đa miền: Hệ thống Canonical Schema, "
        "Domain/Profile Registry, Pipeline xử lý độc lập domain và Công cụ thẩm định (Validator)."
    )

    domain_reg = DomainRegistry()
    yaml_path = "config/platform_profiles.yaml"
    try:
        domain_reg.load_from_yaml(yaml_path)
    except Exception as e:
        st.warning(f"Không thể tải domain YAML ({e}). Dùng cấu hình mặc định.")

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
    except Exception as e:
        st.warning(f"Không thể tải profile YAML ({e}). Dùng cấu hình bộ nhớ mặc định.")

    # Load custom profiles from DB
    try:
        for p in db.get_platform_profiles():
            if not profile_reg.get_profile(p["profile_id"]):
                profile_reg.register_profile(ProcessingProfile.model_validate(p))
    except Exception:
        pass

    subtab1, subtab2, subtab3 = st.tabs([
        "📋 1. Pipeline & Profile Registry",
        "📐 2. Canonical Schema & Validator",
        "🧪 3. Quy Trình Xử Lý & Dựng Đồ Thị Từ File",
    ])


    # =========================================================================
    # SUBTAB 1: PIPELINE & PROFILE REGISTRY
    # =========================================================================
    with subtab1:
        st.subheader("Luồng Xử Lý Dữ Liệu Nền Tảng (Pipeline Architecture)")
        st.markdown("""
        Kiến trúc **Extensible Multi-Domain GraphRAG** loại bỏ hoàn toàn các logic hard-code theo từng loại văn bản hay lĩnh vực.
        Toàn bộ quy trình từ dữ liệu thô tới đồ thị tri thức được tham số hoá qua **Processing Profiles**:
        """)

        flow_col1, flow_col2, flow_col3, flow_col4, flow_col5 = st.columns(5)
        with flow_col1:
            st.info("**1. Ingest & Profile**\n\nResolve Profile theo `(domain, doc_type)`.")
        with flow_col2:
            st.info("**2. Parse Blocks**\n\nTách `ContentBlock` chuẩn hoá vị trí và offset.")
        with flow_col3:
            st.info("**3. Chunking**\n\nCắt `CanonicalChunk` bảo toàn source_block_ids.")
        with flow_col4:
            st.info("**4. Extraction**\n\nBóc tách Entity, Relation kèm `EvidenceRecord`.")
        with flow_col5:
            st.success("**5. Storage & Graph**\n\nLập chỉ mục Namespace & Multi-Graph.")

        st.divider()
        st.subheader("Danh Sách Domain Được Đăng Ký (Domain Registry)")
        domains = domain_reg.list_domains(enabled_only=False)
        domain_rows = []
        for d in domains:
            domain_rows.append({
                "Domain ID": d.domain_id,
                "Tên Domain": d.name,
                "Graph Namespace": d.graph_namespace,
                "Default Profile": d.default_profile_id,
                "Test Fixture?": "✅" if d.is_test_fixture else "❌",
                "Trạng thái": "Kích hoạt" if d.enabled else "Tắt",
            })
        st.dataframe(pd.DataFrame(domain_rows), use_container_width=True)

        st.subheader("Danh Sách Processing Profiles Đang Hoạt Động (Profile Registry)")
        profiles = list(profile_reg._profiles.values())
        profile_rows = []
        for p in profiles:
            profile_rows.append({
                "Profile ID": p.profile_id,
                "Domain ID": p.domain_id,
                "Document Type": p.document_type,
                "Parser Strategy": p.parser_strategy,
                "Chunking Strategy": p.chunking_strategy,
                "Chunk Parameters": json.dumps(p.chunking_parameters),
                "Fallback Profile": p.fallback_profile_id or "—",
            })
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

    # =========================================================================
    # SUBTAB 2: CANONICAL SCHEMA & VALIDATOR
    # =========================================================================
    with subtab2:
        st.subheader("Mô Hình Dữ Liệu Chuẩn Hoá (Canonical Metadata Schema)")
        st.markdown(
            "Mọi loại tài liệu từ bất kỳ miền tri thức nào đều được chuẩn hoá về bộ "
            "7 Canonical Entities cốt lõi nhằm đảm bảo tính toàn vẹn và truy xuất nguồn gốc (provenance):"
        )

        model_choice = st.selectbox(
            "Chọn Canonical Model để xem cấu trúc và JSON Schema:",
            [
                "CanonicalDocument",
                "ContentBlock",
                "CanonicalChunk",
                "EvidenceRecord (Provenance)",
                "CanonicalEntity",
                "EntityMention",
                "CanonicalRelationship",
                "ACLPolicy",
            ],
        )

        model_map = {
            "CanonicalDocument": CanonicalDocument,
            "ContentBlock": ContentBlock,
            "CanonicalChunk": CanonicalChunk,
            "EvidenceRecord (Provenance)": EvidenceRecord,
            "CanonicalEntity": CanonicalEntity,
            "EntityMention": EntityMention,
            "CanonicalRelationship": CanonicalRelationship,
            "ACLPolicy": ACLPolicy,
        }

        selected_model = model_map[model_choice]
        col_sch1, col_sch2 = st.columns([1, 1])
        with col_sch1:
            st.markdown(f"**Danh sách các trường của `{selected_model.__name__}`:**")
            fields_data = []
            for name, field in selected_model.model_fields.items():
                fields_data.append({
                    "Trường (Field)": name,
                    "Kiểu dữ liệu": str(field.annotation),
                    "Bắt buộc": "✅" if field.is_required() else "Tùy chọn",
                })
            st.dataframe(pd.DataFrame(fields_data), use_container_width=True)

        with col_sch2:
            st.markdown(f"**JSON Schema Definition (`pydantic`):**")
            st.json(selected_model.model_json_schema())

        st.divider()
        st.subheader("Thẩm Định Tính Toàn Vẹn (Schema & Provenance Validator)")
        st.caption("Kiểm tra toàn vẹn quan hệ, ranh giới domain, và bắt buộc liên kết bằng chứng cho mọi liên kết đã xác minh (VERIFIED).")

        val_col1, val_col2 = st.columns(2)
        with val_col1:
            if st.button("Kiểm tra với Dữ Liệu Mẫu Hợp Lệ (Valid Bundle)"):
                try:
                    with open("data/examples/canonical_valid_examples.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    doc = CanonicalDocument.model_validate(data["document"])
                    blocks = [ContentBlock.model_validate(b) for b in data["content_blocks"]]
                    chunks = [CanonicalChunk.model_validate(c) for c in data["chunks"]]
                    evs = [EvidenceRecord.model_validate(e) for e in data["evidences"]]
                    ents = [CanonicalEntity.model_validate(en) for en in data["entities"]]
                    rels = [CanonicalRelationship.model_validate(r) for r in data["relationships"]]

                    validator = CanonicalSchemaValidator()
                    res = validator.validate_bundle(
                        documents=[doc],
                        content_blocks=blocks,
                        chunks=chunks,
                        entities=ents,
                        relationships=rels,
                        evidences=evs,
                    )
                    if res.is_valid:
                        st.success("✅ Thẩm định thành công: Toàn bộ khóa ngoại, provenance và domain namespace hợp lệ 100%!")
                    else:
                        st.error(f"Thẩm định thất bại: {res.errors}")
                except Exception as ex:
                    st.error(f"Lỗi: {ex}")

        with val_col2:
            if st.button("Kiểm tra với Dữ Liệu Mẫu Lỗi (Invalid Bundle - Bắt Lỗi Tự Động)"):
                try:
                    with open("data/examples/canonical_invalid_examples.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    doc = CanonicalDocument.model_validate(data["document"])
                    blocks = [ContentBlock.model_validate(b) for b in data["content_blocks"]]
                    chunks = [CanonicalChunk.model_validate(c) for c in data["chunks"]]
                    evs = [EvidenceRecord.model_validate(e) for e in data["evidences"]]
                    ents = [CanonicalEntity.model_validate(en) for en in data["entities"]]
                    rels = [CanonicalRelationship.model_validate(r) for r in data["relationships"]]

                    validator = CanonicalSchemaValidator()
                    res = validator.validate_bundle(
                        documents=[doc],
                        content_blocks=blocks,
                        chunks=chunks,
                        entities=ents,
                        relationships=rels,
                        evidences=evs,
                    )
                    st.warning(f"🛡️ Validator đã bắt được chính xác {len(res.errors)} vi phạm toàn vẹn dữ liệu:")
                    for err in res.errors:
                        st.markdown(f"- **[{err.field}]** `{err.error_type}`: {err.message}")
                except Exception as ex:
                    st.error(f"Lỗi: {ex}")

    # =========================================================================
    # SUBTAB 3: QUY TRÌNH XỬ LÝ & DỰNG ĐỒ THỊ TỪ FILE (4 BƯỚC)
    # =========================================================================
    with subtab3:
        st.subheader("🧪 Quy Trình Xử Lý Văn Bản, Tùy Biến Chunking & Dựng Đồ Thị Tri Thức")
        st.markdown(
            "Trải nghiệm quy trình toàn diện từ việc đưa vào một file văn bản bất kỳ (`.txt`, `.md`, `.json`), "
            "tự chọn cơ chế Chunking và Parser phù hợp, bóc tách Thực thể/Quan hệ và đưa trực tiếp vào Đồ thị tri thức."
        )

        st.markdown("---")
        # -------------------------------------------------------------
        # BƯỚC 1: FILE VĂN BẢN VÀ METADATA ĐẦU VÀO
        # -------------------------------------------------------------
        st.markdown("#### 📥 Bước 1: Đưa Dữ Liệu Đầu Vào (File Văn bản & Metadata)")
        
        in_col1, in_col2 = st.columns([1, 1])
        with in_col1:
            input_mode = st.radio(
                "Nguồn cung cấp nội dung:",
                ["📁 Tải file lên (.txt, .md, .json)", "✍️ Nhập / Dán văn bản trực tiếp"],
                horizontal=True,
            )
            raw_file_text = ""
            default_title = "Tài liệu Quy trình Vận hành Trạm Giám sát"
            file_meta = {}

            if "Tải file" in input_mode:
                uploaded_doc = st.file_uploader(
                    "Tải lên file văn bản:",
                    type=["txt", "md", "markdown", "json"],
                    help="Hỗ trợ file Text, Markdown hoặc JSON có trường content/metadata.",
                )
                if uploaded_doc is not None:
                    try:
                        content_bytes = uploaded_doc.read()
                        decoded_text = content_bytes.decode("utf-8", errors="replace")
                        if uploaded_doc.name.endswith(".json"):
                            parsed_json = json.loads(decoded_text)
                            if isinstance(parsed_json, dict) and "content" in parsed_json:
                                raw_file_text = parsed_json.get("content", "")
                                file_meta = parsed_json.get("metadata", {})
                            else:
                                raw_file_text = decoded_text
                        else:
                            raw_file_text = decoded_text
                        default_title = uploaded_doc.name.rsplit(".", 1)[0].replace("_", " ").title()
                        st.success(f"Đã đọc file: **{uploaded_doc.name}** ({len(raw_file_text)} ký tự)")
                    except Exception as err:
                        st.error(f"Lỗi khi đọc file: {err}")

            fallback_sample = (
                "# 1. Khởi động hệ thống\n\n"
                "Cụm máy chủ Gateway tiếp nhận kết nối từ Internet và chuyển tiếp yêu cầu đến Server Backend.\n\n"
                "# 2. Quản lý lưu trữ\n\n"
                "Server Backend lưu trữ phiên làm việc trong Redis Cache và dữ liệu dài hạn trong Cơ sở dữ liệu Postgres.\n\n"
                "Khi Redis Cache bị đầy, Gateway sẽ phản hồi mã lỗi 504 Timeout."
            )
            initial_text = raw_file_text if raw_file_text else fallback_sample
            input_raw_text = st.text_area(
                "Xem trước & chỉnh sửa nội dung văn bản đầu vào (Raw text):",
                value=initial_text,
                height=160,
            )

        with in_col2:
            st.markdown("**Khai báo Metadata của Tài liệu:**")
            test_doc_title = st.text_input("Tiêu đề Tài liệu:", value=file_meta.get("title", default_title))

            # Domain selection
            existing_domains = [d.domain_id for d in domain_reg.list_domains(enabled_only=False)]
            domain_choice_list = existing_domains + ["➕ Tạo Domain mới..."]
            selected_domain_raw = st.selectbox("Chọn Lĩnh vực / Miền dữ liệu (Domain):", domain_choice_list, index=0 if existing_domains else None)

            if selected_domain_raw == "➕ Tạo Domain mới...":
                new_dom_col1, new_dom_col2 = st.columns(2)
                with new_dom_col1:
                    new_domain_id = st.text_input("Mã Domain mới (ID):", value="domain_custom_ops")
                with new_dom_col2:
                    new_domain_name = st.text_input("Tên Domain mới:", value="Miền Nghiệp vụ Mới")
                selected_domain = new_domain_id
                # Register on the fly
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
                    ["operational_manual", "incident_report", "research_memo", "legal_regulation", "technical_spec", "general"],
                    index=0,
                )
            with meta_c2:
                selected_sec_class = st.selectbox(
                    "Cấp độ Bảo mật (Security):",
                    ["INTERNAL", "PUBLIC", "CONFIDENTIAL", "RESTRICTED"],
                    index=0,
                )

            selected_roles = st.multiselect(
                "Vai trò được phép truy cập (ACL Allowed Roles):",
                ["public", "ops", "developer", "researcher", "sec_admin"],
                default=["public", "ops", "developer"],
            )

        st.markdown("---")
        # -------------------------------------------------------------
        # BƯỚC 2: LỰA CHỌN CƠ CHẾ CHUNKING & PARSER
        # -------------------------------------------------------------
        st.markdown("#### 🧩 Bước 2: Lựa Chọn Cơ Chế Chunking & Tách Khối (Parser)")
        st.caption("Bạn có thể tự do chọn cơ chế Chunking và tinh chỉnh kích thước phân mảnh phù hợp nhất với cấu trúc văn bản.")

        chunk_col1, chunk_col2, chunk_col3, chunk_col4 = st.columns([1.3, 1.2, 1, 1])
        with chunk_col1:
            chunker_options = {
                "block_preserving": "Block Preserving (Bảo toàn trọn vẹn khối đoạn/mục)",
                "sliding_window": "Sliding Window (Cửa sổ trượt có overlap)",
                "hierarchical_heading": "Hierarchical Heading (Phân cấp theo tiêu đề mục)",
                "fixed_token": "Fixed Token (Cắt cố định theo số token)",
            }
            selected_chunker_key = st.selectbox(
                "Cơ chế Chunking (Strategy):",
                list(chunker_options.keys()),
                format_func=lambda k: chunker_options[k],
                index=0,
            )

        with chunk_col2:
            parser_options = {
                "structured_heading": "Structured Heading (Nhận diện tiêu đề #, Điều, Mục)",
                "generic_paragraph": "Generic Paragraph (Tách theo đoạn văn thông thường)",
            }
            selected_parser_key = st.selectbox(
                "Cơ chế Phân Tách Khối (Parser):",
                list(parser_options.keys()),
                format_func=lambda k: parser_options[k],
                index=0,
            )

        with chunk_col3:
            max_tokens_val = st.slider("Kích thước Chunk (Max Tokens):", min_value=64, max_value=1024, value=256, step=32)

        with chunk_col4:
            overlap_val = st.slider("Độ chồng lấp (Chunk Overlap):", min_value=0, max_value=128, value=32, step=8)

        st.markdown("---")
        # -------------------------------------------------------------
        # BƯỚC 3: CẤU HÌNH TRÍCH XUẤT THỰC THỂ & QUAN HỆ
        # -------------------------------------------------------------
        st.markdown("#### 🏷️ Bước 3: Cấu Hình Trích Xuất Thực Thể & Mối Quan Hệ")
        ext_col1, ext_col2 = st.columns([1.5, 1])

        with ext_col1:
            extractor_labels = {
                "rule_based_ops": "Rule-based & Regex Heuristic (Chạy Local - Nhanh, không cần API Key)",
                "llm_extractor": "LLM Assisted Extractor (Sử dụng Model đã chọn ở Sidebar)",
            }
            selected_extractor_key = st.selectbox(
                "Phương thức trích xuất tri thức (Knowledge Extractor):",
                list(extractor_labels.keys()),
                format_func=lambda k: extractor_labels[k],
                index=0,
            )

        with ext_col2:
            persist_toggle = st.checkbox(
                "💾 Tự động đưa vào Cơ sở dữ liệu & Đồ thị (Persist to DB & Graph)",
                value=True,
                help="Lưu toàn bộ Chunks, Thực thể và Quan hệ vào CSDL để có thể tìm kiếm và mở rộng đồ thị ngay tại Tab 1 (Multi-Domain Lab).",
            )

        st.markdown("---")
        # -------------------------------------------------------------
        # BƯỚC 4: THỰC THI & ĐƯA VÀO ĐỒ THỊ TRI THỨC
        # -------------------------------------------------------------
        st.markdown("#### 🚀 Bước 4: Thực Thi Quy Trình & Hiển Thị Đồ Thị")

        if st.button("🚀 Chạy Quy Trình Toàn Trình & Dựng Đồ Thị (Execute & Build Graph)", type="primary"):
            with st.spinner("Đang bóc tách blocks, cắt chunks, trích xuất tri thức và thẩm định schema..."):
                orchestrator = PipelineOrchestrator(
                    domain_registry=domain_reg,
                    profile_registry=profile_reg,
                    db_manager=db,
                )
                doc_meta = {
                    "source_id": "SANDBOX_DOC",
                    "domain_id": selected_domain,
                    "document_type": selected_doc_type,
                    "security_classification": selected_sec_class,
                    "acl": {
                        "allowed_roles": selected_roles,
                        "allowed_users": [],
                        "classification": selected_sec_class,
                    },
                }

                bundle, trace = orchestrator.process_raw_text(
                    raw_text=input_raw_text,
                    title=test_doc_title,
                    document_meta=doc_meta,
                    parser_strategy_override=selected_parser_key,
                    chunking_strategy_override=selected_chunker_key,
                    chunking_params_override={"max_tokens": max_tokens_val, "chunk_overlap": overlap_val},
                    extractor_override=selected_extractor_key,
                    llm_config=llm_config,
                )

                if persist_toggle:
                    try:
                        orchestrator.persist_bundle(bundle)
                        st.toast("✅ Đã lưu thành công dữ liệu vào kho tri thức!", icon="💾")
                    except Exception as p_err:
                        st.warning(f"Không thể lưu vào DB: {p_err}")

            st.success(
                f"✅ Xử lý hoàn tất trong **{trace.total_duration_sec * 1000:.1f}ms**! "
                f"Tạo được: **{len(bundle['blocks'])} Blocks**, **{len(bundle['chunks'])} Chunks**, "
                f"**{len(bundle['entities'])} Thực thể**, **{len(bundle['relationships'])} Quan hệ**."
            )

            if persist_toggle:
                st.info(
                    f"💡 **Dữ liệu đã được nạp vào Đồ thị tri thức & CSDL:** Bạn có thể chuyển sang "
                    f"**Tab 1 (Multi-Domain Lab)**, chọn Domain `{selected_domain}` để truy vấn Hybrid và mở rộng đồ thị (KG Expansion) ngay trên tài liệu này!"
                )

            # 5 View Tabs
            r_tab1, r_tab2, r_tab3, r_tab4, r_tab5 = st.tabs([
                f"🧱 1. ContentBlocks ({len(bundle['blocks'])})",
                f"🧩 2. Canonical Chunks ({len(bundle['chunks'])})",
                f"🏷️ 3. Thực Thể & Quan Hệ ({len(bundle['entities'])} Ent, {len(bundle['relationships'])} Rel)",
                f"🕸️ 4. Đồ Thị Tri Thức Trực Quan (Interactive KG)",
                f"⏱️ 5. Thẩm Định Schema & Trace",
            ])

            with r_tab1:
                st.subheader("Danh Sách Khối Nội Dung (ContentBlocks)")
                b_rows = [
                    {
                        "Block ID": b.block_id,
                        "Loại": b.block_type,
                        "Offsets": f"{b.start_offset} -> {b.end_offset}",
                        "Parent Block": b.parent_block_id or "Root",
                        "Nội dung": b.text[:140] + ("..." if len(b.text) > 140 else ""),
                    }
                    for b in bundle["blocks"]
                ]
                st.dataframe(pd.DataFrame(b_rows), use_container_width=True) if b_rows else st.info("Không có block nào.")

            with r_tab2:
                st.subheader(f"Danh Sách Chunks Đã Cắt (Chiến lược: `{selected_chunker_key}`)")
                c_rows = [
                    {
                        "Chunk ID": c.chunk_id,
                        "Strategy": c.chunking_strategy,
                        "Tokens": c.token_count,
                        "Source Block IDs": ", ".join(c.source_block_ids),
                        "Nội dung": c.content,
                    }
                    for c in bundle["chunks"]
                ]
                st.dataframe(pd.DataFrame(c_rows), use_container_width=True) if c_rows else st.info("Không có chunk nào.")

            with r_tab3:
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    st.markdown("**Thực thể bóc tách (Entities):**")
                    ent_rows = [
                        {
                            "Entity ID": e.entity_id,
                            "Tên chuẩn hóa": e.canonical_name,
                            "Loại": e.entity_type,
                            "Namespace": e.graph_namespace,
                            "Bằng chứng": ", ".join(e.evidence_refs),
                        }
                        for e in bundle["entities"]
                    ]
                    st.dataframe(pd.DataFrame(ent_rows), use_container_width=True) if ent_rows else st.info("Chưa phát hiện thực thể.")

                with e_col2:
                    st.markdown("**Quan hệ bóc tách (Relationships):**")
                    rel_rows = [
                        {
                            "Subject": r.subject_entity_id.split("::")[-1] if "::" in r.subject_entity_id else r.subject_entity_id,
                            "Predicate": r.predicate,
                            "Object": r.object_entity_id.split("::")[-1] if "::" in r.object_entity_id else r.object_entity_id,
                            "Bằng chứng": ", ".join(r.evidence_refs),
                        }
                        for r in bundle["relationships"]
                    ]
                    st.dataframe(pd.DataFrame(rel_rows), use_container_width=True) if rel_rows else st.info("Chưa phát hiện quan hệ.")

            with r_tab4:
                st.subheader("Đồ Thị Tri Thức Bóc Tách Từ Tài Liệu (Interactive Knowledge Graph)")
                if bundle["entities"]:
                    html_graph = render_bundle_pyvis_html(bundle["entities"], bundle["relationships"], height="500px")
                    components.html(html_graph, height=520)
                    st.caption("🔵 Xanh dương: Service | 🟢 Xanh lá: Database | 🟣 Tím: Protocol | 🔴 Đỏ: Incident | 🟠 Vàng: Component | ⚪ Xám: Liên kết có hướng")
                else:
                    st.info("Chưa có thực thể nào để hiển thị đồ thị mạng lưới.")

            with r_tab5:
                st.subheader("Kết Quả Thẩm Định Toàn Vẹn & Vết Thời Gian")
                st.markdown(
                    f"**Trạng thái Pipeline:** `{trace.status}` | **Hợp lệ Schema:** {'✅ ĐẠT' if trace.is_valid else '❌ LỖI'} | "
                    f"**Document ID:** `{trace.document_id}` | **Profile:** `{trace.profile_id}`"
                )
                if trace.validation_issues:
                    st.warning(f"Phát hiện {len(trace.validation_issues)} cảnh báo schema:")
                    st.json(trace.validation_issues)
                else:
                    st.success("Tất cả các điều kiện ràng buộc Schema, Offset và Provenance đều đạt chuẩn 100%!")
                st.dataframe(pd.DataFrame(trace.step_logs), use_container_width=True)

