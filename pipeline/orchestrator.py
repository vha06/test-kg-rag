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
        if eff_extractor_name == "llm_extractor":
            eff_llm_config = llm_config or document_metadata.get("llm_config")
            k_ext = ExtractorRegistry.get_knowledge_extractor("llm_extractor")
            if hasattr(k_ext, "llm_config"):
                k_ext.llm_config = eff_llm_config
        elif eff_extractor_name:
            k_ext = ExtractorRegistry.get_knowledge_extractor(eff_extractor_name)
        else:
            k_ext_name = effective_profile.metadata_extractors[1] if len(effective_profile.metadata_extractors) > 1 else "rule_based_ops"
            k_ext = ExtractorRegistry.get_knowledge_extractor(k_ext_name)

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


