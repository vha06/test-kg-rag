from __future__ import annotations

import json
import streamlit as st
import pandas as pd

from core.registry import DomainRegistry, ProfileRegistry
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
from storage.db import DatabaseManager


def render_platform_overview_tab(db: DatabaseManager):
    st.header("🌐 Extensible Multi-Domain Platform Architecture")
    st.caption(
        "Khám phá kiến trúc nền tảng GraphRAG đa miền: Hệ thống Canonical Schema, "
        "Domain/Profile Registry, Pipeline xử lý độc lập domain và Công cụ thẩm định (Validator)."
    )

    domain_reg = DomainRegistry()
    profile_reg = ProfileRegistry(domain_reg)
    yaml_path = "config/platform_profiles.yaml"
    try:
        profile_reg.load_from_yaml(yaml_path)
    except Exception as e:
        st.warning(f"Không thể tải profile YAML ({e}). Dùng cấu hình bộ nhớ mặc định.")

    subtab1, subtab2, subtab3 = st.tabs([
        "📋 1. Pipeline & Profile Registry",
        "📐 2. Canonical Schema & Validator",
        "🧪 3. Live Pipeline Sandbox",
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
    # SUBTAB 3: LIVE PIPELINE SANDBOX
    # =========================================================================
    with subtab3:
        st.subheader("Môi Trường Chạy Thử Nghiệm Trực Tiếp (Live Pipeline Sandbox)")
        st.markdown(
            "Nhập văn bản bất kỳ, chọn Domain và Profile xử lý để xem trực tiếp "
            "quá trình bóc tách `ContentBlock` -> `CanonicalChunk` -> `Entity/Relationship` hoàn toàn không phụ thuộc logic cũ."
        )

        sb_col1, sb_col2 = st.columns([1, 1])
        with sb_col1:
            domain_opts = [d.domain_id for d in domain_reg.list_domains(enabled_only=True)]
            selected_domain = st.selectbox("Chọn Domain:", domain_opts, index=0 if domain_opts else None)
            available_profiles = [p.profile_id for p in profile_reg.list_profiles_for_domain(selected_domain)]
            if not available_profiles:
                available_profiles = ["profile_generic_default"]
            selected_profile_id = st.selectbox("Chọn Profile Xử Lý:", available_profiles)
            test_doc_title = st.text_input("Tiêu đề Tài liệu:", value="Quy trình Vận hành Trạm Giám sát Hạ tầng")

        with sb_col2:
            default_text = (
                "# 1. Khởi động hệ thống\n\n"
                "Cụm máy chủ Gateway tiếp nhận kết nối từ Internet và chuyển tiếp yêu cầu đến Server Backend.\n\n"
                "# 2. Quản lý lưu trữ\n\n"
                "Server Backend lưu trữ phiên làm việc trong Redis Cache và dữ liệu dài hạn trong Cơ sở dữ liệu Postgres.\n\n"
                "Khi Redis Cache bị đầy, Gateway sẽ phản hồi mã lỗi 504 Timeout."
            )
            input_raw_text = st.text_area("Văn bản thô (Raw text):", value=default_text, height=180)

        if st.button("🚀 Chạy Thử Nghiệm Toàn Trình (Execute Pipeline)", type="primary"):
            orchestrator = PipelineOrchestrator(
                domain_registry=domain_reg,
                profile_registry=profile_reg,
                db_manager=db,
            )
            doc_meta = {
                "source_id": "SANDBOX_01",
                "domain_id": selected_domain,
                "profile_id": selected_profile_id,
                "document_type": "operational_manual",
                "security_classification": "INTERNAL",
            }
            bundle, trace = orchestrator.process_raw_text(
                raw_text=input_raw_text,
                title=test_doc_title,
                document_meta=doc_meta,
            )

            st.success(f"Xử lý thành công trong **{trace.total_duration_sec * 1000:.1f}ms**!")

            r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs([
                f"🧱 ContentBlocks ({len(bundle['blocks'])})",
                f"🧩 Chunks ({len(bundle['chunks'])})",
                f"🏷️ Extracted Knowledge ({len(bundle['entities'])} Ent, {len(bundle['relationships'])} Rel)",
                f"⏱️ Execution Trace",
            ])

            with r_tab1:
                b_rows = [
                    {
                        "Block ID": b.block_id,
                        "Type": b.block_type,
                        "Offsets": f"{b.start_offset} -> {b.end_offset}",
                        "Parent Block": b.parent_block_id or "Root",
                        "Nội dung": b.text[:120] + ("..." if len(b.text) > 120 else ""),
                    }
                    for b in bundle["blocks"]
                ]
                st.dataframe(pd.DataFrame(b_rows), use_container_width=True)

            with r_tab2:
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
                st.dataframe(pd.DataFrame(c_rows), use_container_width=True)

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
                            "Bằng chứng": e.evidence_refs,
                        }
                        for e in bundle["entities"]
                    ]
                    st.dataframe(pd.DataFrame(ent_rows), use_container_width=True) if ent_rows else st.info("Không có thực thể.")

                with e_col2:
                    st.markdown("**Quan hệ bóc tách (Relationships):**")
                    rel_rows = [
                        {
                            "Subject": r.subject_entity_id,
                            "Predicate": r.predicate,
                            "Object": r.object_entity_id,
                            "Bằng chứng": r.evidence_refs,
                        }
                        for r in bundle["relationships"]
                    ]
                    st.dataframe(pd.DataFrame(rel_rows), use_container_width=True) if rel_rows else st.info("Không có quan hệ.")

            with r_tab4:
                st.markdown(f"**Trạng thái Pipeline:** `{trace.status}` | **Document ID:** `{trace.document_id}`")
                st.dataframe(pd.DataFrame(trace.step_logs), use_container_width=True)
