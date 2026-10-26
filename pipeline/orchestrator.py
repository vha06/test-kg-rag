import time
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field

from core.schema import (
    CanonicalDocument,
    ContentBlock,
    CanonicalChunk,
    EvidenceRecord,
    CanonicalEntity,
    EntityMention,
    CanonicalRelationship,
    ACLPolicy,
)
from core.registry import DomainRegistry, ProfileRegistry, ProcessingProfile
from core.validator import CanonicalSchemaValidator, ValidationIssue
from pipeline.parsers import ParserRegistry
from pipeline.chunkers import ChunkerRegistry
from pipeline.extractors import ExtractorRegistry


class PipelineExecutionTrace(BaseModel):
    document_id: str
    domain_id: str
    profile_id: str
    elapsed_ms: float = 0.0
    block_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    mention_count: int = 0
    relationship_count: int = 0
    evidence_count: int = 0
    is_valid: bool = False
    validation_issues: List[Dict[str, str]] = Field(default_factory=list)
    step_timings: Dict[str, float] = Field(default_factory=dict)
    status: str = "SUCCESS"

    @property
    def total_duration_sec(self) -> float:
        return self.elapsed_ms / 1000.0

    @property
    def step_logs(self) -> List[Dict[str, Any]]:
        return [
            {"Step": step, "Duration (ms)": timing}
            for step, timing in self.step_timings.items()
        ]


class PipelineOrchestrator:
    """
    Bộ điều phối toàn diện:
    Raw Text -> Profile Resolution -> Parse Blocks -> Chunking ->
    Metadata & Knowledge Extraction -> Schema Validation -> Artifacts + Trace.
    """

    def __init__(
        self,
        domain_registry: Optional[DomainRegistry] = None,
        profile_registry: Optional[ProfileRegistry] = None,
        db_manager: Optional[Any] = None,
    ):
        self.domain_registry = domain_registry or DomainRegistry()
        self.profile_registry = profile_registry or ProfileRegistry(domain_registry=self.domain_registry)
        self.db = db_manager

    def process_document(
        self,
        raw_text: str,
        document_metadata: Dict[str, Any],
        profile_id: Optional[str] = None,
        parser_strategy_override: Optional[str] = None,
        chunking_strategy_override: Optional[str] = None,
        chunking_params_override: Optional[Dict[str, Any]] = None,
        extractor_override: Optional[str] = None,
        llm_config: Optional[Any] = None,
    ) -> Tuple[
        CanonicalDocument,
        List[ContentBlock],
        List[CanonicalChunk],
        List[CanonicalEntity],
        List[EntityMention],
        List[CanonicalRelationship],
        List[EvidenceRecord],
        PipelineExecutionTrace,
    ]:
        start_time = time.perf_counter()
        timings: Dict[str, float] = {}

        doc_id = document_metadata.get("document_id")
        if not doc_id:
            h = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12]
            doc_id = f"DOC_{h}"

        domain_id = document_metadata.get("domain_id", "default_domain")
        doc_type = document_metadata.get("document_type", "general")

        # 1. Resolve Profile
        t0 = time.perf_counter()
        profile = self.profile_registry.resolve_profile(
            domain_id=domain_id,
            document_type=doc_type,
            profile_id=profile_id,
        )

        # Apply overrides if requested
        eff_parser_strategy = parser_strategy_override or document_metadata.get("parser_strategy_override") or profile.parser_strategy
        eff_chunking_strategy = chunking_strategy_override or document_metadata.get("chunking_strategy_override") or profile.chunking_strategy
        eff_chunking_params = dict(profile.chunking_parameters or {})
        override_params = chunking_params_override or document_metadata.get("chunking_params_override")
        if override_params:
            eff_chunking_params.update(override_params)

        effective_profile = profile.model_copy(
            update={
                "parser_strategy": eff_parser_strategy,
                "chunking_strategy": eff_chunking_strategy,
                "chunking_parameters": eff_chunking_params,
            }
        )
        timings["resolve_profile_ms"] = (time.perf_counter() - t0) * 1000

        # Build initial CanonicalDocument
        chksum = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        acl_data = document_metadata.get("acl", {})
        acl = ACLPolicy.model_validate(acl_data) if acl_data else ACLPolicy()

        doc = CanonicalDocument(
            document_id=doc_id,
            schema_version="1.0.0",
            title=document_metadata.get("title", f"Document {doc_id}"),
            source_id=document_metadata.get("source_id", doc_id),
            domain_id=domain_id,
            document_type=doc_type,
            profile_id=effective_profile.profile_id,
            profile_version=effective_profile.profile_version,
            checksum=chksum,
            processing_status="PENDING",
            security_classification=document_metadata.get("security_classification", "INTERNAL"),
            source_uri=document_metadata.get("source_uri"),
            mime_type=document_metadata.get("mime_type", "text/plain"),
            language=document_metadata.get("language", "vi"),
            acl=acl,
            metadata=document_metadata.get("metadata", {}),
        )

        # 2. Parse into ContentBlocks
        t0 = time.perf_counter()
        parser = ParserRegistry.get(effective_profile.parser_strategy)
        blocks = parser.parse(raw_text=raw_text, doc=doc)
        timings["parse_blocks_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Chunking
        t0 = time.perf_counter()
        chunker = ChunkerRegistry.get(effective_profile.chunking_strategy)
        chunks = chunker.chunk(doc=doc, blocks=blocks, profile=effective_profile)
        timings["chunking_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Metadata Extraction
        t0 = time.perf_counter()
        for meta_name in effective_profile.metadata_extractors:
            m_ext = ExtractorRegistry.get_metadata_extractor(meta_name)
            if m_ext:
                extracted_meta = m_ext.extract_metadata(doc=doc, blocks=blocks)
                doc.metadata[meta_name] = extracted_meta
        timings["metadata_extraction_ms"] = (time.perf_counter() - t0) * 1000

        # 5. Knowledge Extraction (Entities, Mentions, Relationships, Evidences)
        t0 = time.perf_counter()
        eff_extractor_name = extractor_override or document_metadata.get("extractor_override")
        eff_llm_config = llm_config or document_metadata.get("llm_config")
        if eff_extractor_name:
            k_ext = ExtractorRegistry.get_knowledge_extractor(eff_extractor_name)
        else:
            k_ext_name = effective_profile.metadata_extractors[1] if len(effective_profile.metadata_extractors) > 1 else "rule_based_ops"
            k_ext = ExtractorRegistry.get_knowledge_extractor(k_ext_name)

        if hasattr(k_ext, "llm_config"):
            k_ext.llm_config = eff_llm_config
        if hasattr(k_ext, "db"):
            k_ext.db = self.db

        entities, mentions, relationships, evidences = k_ext.extract_knowledge(
            doc=doc, blocks=blocks, chunks=chunks, profile=effective_profile
        )
        timings["knowledge_extraction_ms"] = (time.perf_counter() - t0) * 1000

        # 6. Schema Validation
        t0 = time.perf_counter()
        is_valid, issues = CanonicalSchemaValidator.validate_document_bundle(
            document=doc,
            blocks=blocks,
            chunks=chunks,
            entities=entities,
            mentions=mentions,
            relationships=relationships,
            evidences=evidences,
        )
        timings["validation_ms"] = (time.perf_counter() - t0) * 1000

        doc.processing_status = "INDEXED" if is_valid else "FAILED"
        total_elapsed = (time.perf_counter() - start_time) * 1000

        trace = PipelineExecutionTrace(
            document_id=doc.document_id,
            domain_id=doc.domain_id,
            profile_id=effective_profile.profile_id,
            elapsed_ms=round(total_elapsed, 2),
            block_count=len(blocks),
            chunk_count=len(chunks),
            entity_count=len(entities),
            mention_count=len(mentions),
            relationship_count=len(relationships),
            evidence_count=len(evidences),
            is_valid=is_valid,
            validation_issues=[i.to_dict() for i in issues],
            step_timings={k: round(v, 2) for k, v in timings.items()},
        )

        return doc, blocks, chunks, entities, mentions, relationships, evidences, trace

    def process_raw_text(
        self,
        raw_text: str,
        title: str = "Untitled Document",
        document_meta: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        parser_strategy_override: Optional[str] = None,
        chunking_strategy_override: Optional[str] = None,
        chunking_params_override: Optional[Dict[str, Any]] = None,
        extractor_override: Optional[str] = None,
        llm_config: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], PipelineExecutionTrace]:
        meta = dict(document_meta or {})
        meta["title"] = title
        if document_id:
            meta["document_id"] = document_id
        doc, blocks, chunks, entities, mentions, relationships, evidences, trace = self.process_document(
            raw_text=raw_text,
            document_metadata=meta,
            profile_id=meta.get("profile_id"),
            parser_strategy_override=parser_strategy_override,
            chunking_strategy_override=chunking_strategy_override,
            chunking_params_override=chunking_params_override,
            extractor_override=extractor_override,
            llm_config=llm_config,
        )
        bundle = {
            "document": doc,
            "blocks": blocks,
            "chunks": chunks,
            "entities": entities,
            "mentions": mentions,
            "relationships": relationships,
            "evidences": evidences,
        }
        return bundle, trace

    def persist_bundle(self, bundle: Dict[str, Any]) -> None:
        """
        Lưu trữ toàn diện một bundle (Document, Blocks, Chunks, Entities, Mentions, Relationships, Evidences)
        vào cơ sở dữ liệu nền tảng và bảo đảm tính toàn vẹn của Domain/Profile.
        """
        if not self.db:
            raise RuntimeError("Cannot persist bundle: db_manager is not configured on PipelineOrchestrator.")

        doc = bundle["document"]
        # 1. Ensure domain is registered in DB platform_domains
        domain = self.domain_registry.get_domain(doc.domain_id)
        if domain:
            self.db.insert_platform_domains([domain])
        else:
            from core.registry import DomainConfig
            fallback_dom = DomainConfig(
                domain_id=doc.domain_id,
                name=f"Domain {doc.domain_id}",
                graph_namespace=f"ns_{doc.domain_id.replace('synth_domain_', '')}",
                default_profile_id=doc.profile_id,
            )
            self.db.insert_platform_domains([fallback_dom])

        # 2. Ensure profile is registered in DB platform_profiles
        profile = self.profile_registry.get_profile(doc.profile_id)
        if profile:
            self.db.insert_platform_profiles([profile])

        # 3. Insert complete bundle into platform tables
        self.db.insert_platform_bundle(
            doc=bundle["document"],
            blocks=bundle.get("blocks", []),
            chunks=bundle.get("chunks", []),
            entities=bundle.get("entities", []),
            mentions=bundle.get("mentions", []),
            relationships=bundle.get("relationships", []),
            evidences=bundle.get("evidences", []),
        )

    def build_full_graphrag_pipeline(
        self,
        documents: List[Dict[str, Any]],
        namespace: str = "graphrag_default",
        chunking_strategy: str = "recursive_boundary",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        extractor_name: str = "deterministic_general",
        llm_config: Optional[Any] = None,
        embedding_adapter: Optional[Any] = None,
        max_community_levels: int = 3,
        clear_existing: bool = True,
        progress_callback: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Thực thi toàn trình Indexing theo thiết kế Microsoft GraphRAG:
        1. Parse & Chunk toàn bộ tài liệu thành TextUnits (CanonicalChunk).
        2. Bóc tách Thực thể & Quan hệ ở cấp độ Chunk (Grounding Verification + Canonicalization).
        3. Phân cụm cộng đồng đa cấp L0/L1/L2 (Hierarchical Louvain).
        4. Sinh Báo cáo Cộng đồng (Community Reports) kèm kiểm chứng tham chiếu.
        5. Xây dựng 3 chỉ mục vector độc lập (TextUnits, Entities, Community Reports) & Index Manifest.
        """
        start_t = time.perf_counter()

        if progress_callback:
            progress_callback("Đang dọn dẹp và chuẩn bị không gian tri thức...", 0.05)

        if clear_existing and self.db:
            self.db.clear_namespace_artifacts(namespace)

        all_docs: List[CanonicalDocument] = []
        all_blocks: List[ContentBlock] = []
        all_chunks: List[CanonicalChunk] = []

        # 1. Parsing & Chunking
        if progress_callback:
            progress_callback("Đang phân tích cấu trúc và cắt TextUnits...", 0.1)

        chunker = ChunkerRegistry.get(chunking_strategy)
        parser = ParserRegistry.get("generic_paragraph")

        for idx, doc_item in enumerate(documents):
            raw_text = doc_item.get("text", "")
            if not raw_text.strip():
                continue
            doc_id = doc_item.get("document_id") or f"DOC_{hashlib.sha256(raw_text.encode('utf-8')).hexdigest()[:10]}"
            title = doc_item.get("title") or f"Tài liệu {idx + 1}"
            chksum = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

            doc = CanonicalDocument(
                document_id=doc_id,
                title=title,
                source_id=doc_id,
                domain_id=namespace,
                document_type="unstructured",
                profile_id="graphrag_default",
                checksum=chksum,
            )
            all_docs.append(doc)

            blocks = parser.parse(raw_text=raw_text, doc=doc)
            all_blocks.extend(blocks)

            dummy_profile = ProcessingProfile(
                profile_id="graphrag_default",
                domain_id=namespace,
                document_type="unstructured",
                chunking_strategy=chunking_strategy,
                chunking_parameters={"max_tokens": chunk_size, "chunk_overlap": chunk_overlap},
            )
            chunks = chunker.chunk(doc=doc, blocks=blocks, profile=dummy_profile)
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"status": "FAILED", "reason": "No text units generated from input documents."}

        # 2. Knowledge Extraction
        if progress_callback:
            progress_callback(f"Đang trích xuất thực thể & quan hệ từ {len(all_chunks)} chunks...", 0.25)

        k_ext = ExtractorRegistry.get_knowledge_extractor(extractor_name)
        if hasattr(k_ext, "llm_config"):
            k_ext.llm_config = llm_config
        if hasattr(k_ext, "db"):
            k_ext.db = self.db
        if hasattr(k_ext, "progress_callback"):
            k_ext.progress_callback = progress_callback

        lead_doc = all_docs[0] if all_docs else CanonicalDocument(document_id="DOC_MERGED", title="Merged", source_id="M1", checksum="chk")
        lead_profile = ProcessingProfile(profile_id="graphrag_default", domain_id=namespace, document_type="unstructured")

        entities, mentions, relationships, evidences = k_ext.extract_knowledge(
            doc=lead_doc,
            blocks=all_blocks,
            chunks=all_chunks,
            profile=lead_profile,
        )

        # 3. Persist core entities/chunks to DB
        if self.db:
            for doc in all_docs:
                self.db.insert_platform_documents([doc])
            self.db.insert_platform_content_blocks(all_blocks)
            self.db.insert_platform_chunks(all_chunks)
            self.db.insert_platform_entities(entities)
            self.db.insert_platform_entity_mentions(mentions)
            self.db.insert_platform_relationships(relationships)
            self.db.insert_platform_evidences(evidences)

        # 4. Hierarchical Community Detection
        if progress_callback:
            progress_callback(f"Đang phát hiện cộng đồng phân cấp L0/L1/L2 từ {len(entities)} thực thể...", 0.6)

        from knowledge_graph.community_detector import HierarchicalCommunityDetector
        detector = HierarchicalCommunityDetector(
            algorithm="louvain_hierarchical",
            max_levels=max_community_levels,
            min_split_size=6,
            seed=42,
        )
        communities = detector.detect_communities(entities, relationships, namespace=namespace)
        if self.db:
            self.db.insert_graphrag_communities(communities)

        # 5. Community Reports Generation
        if progress_callback:
            progress_callback(f"Đang sinh báo cáo phân tích cho {len(communities)} cụm cộng đồng...", 0.75)

        from knowledge_graph.community_report_generator import CommunityReportGenerator
        rep_gen = CommunityReportGenerator(llm_config=llm_config)
        ent_map = {e.entity_id: e for e in entities}
        rel_map = {r.relationship_id: r for r in relationships}
        chunk_map = {c.chunk_id: c for c in all_chunks}

        reports = []
        for comm in communities:
            rep = rep_gen.generate_report(comm, ent_map, rel_map, chunk_map)
            reports.append(rep)

        if self.db:
            self.db.insert_graphrag_community_reports(reports)

        # 6. Build 3 Vector Indices & Save Manifest
        if progress_callback:
            progress_callback("Đang xây dựng 3 chỉ mục Vector (TextUnits, Entities, Reports)...", 0.9)

        from retrieval.graphrag_index_manager import GraphRAGIndexManager
        idx_mgr = GraphRAGIndexManager(db_manager=self.db, embedding_adapter=embedding_adapter)
        doc_checksums = [d.checksum for d in all_docs]

        manifest_config = {
            "doc_checksums": doc_checksums,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "extractor_model": extractor_name,
            "extraction_prompt_version": "2.0.0",
            "embedding_model": "sentence-transformers" if embedding_adapter is None else getattr(embedding_adapter, "model_name", "hash"),
            "community_algorithm": "louvain_hierarchical",
            "max_community_levels": max_community_levels,
        }

        chunks_dict = [c.model_dump() for c in all_chunks]
        entities_dict = [e.model_dump() for e in entities]
        reports_dict = [r.model_dump() for r in reports]

        idx_mgr.build_all_indices(
            namespace=namespace,
            chunks=chunks_dict,
            entities=entities_dict,
            reports=reports_dict,
            embedding_adapter=embedding_adapter,
            manifest_config=manifest_config,
        )

        if progress_callback:
            progress_callback("Hoàn tất xây dựng toàn trình GraphRAG Index!", 1.0)

        total_sec = time.perf_counter() - start_t
        return {
            "status": "SUCCESS",
            "namespace": namespace,
            "document_count": len(all_docs),
            "chunk_count": len(all_chunks),
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "community_count": len(communities),
            "report_count": len(reports),
            "elapsed_seconds": round(total_sec, 2),
            "index_manager": idx_mgr,
        }


