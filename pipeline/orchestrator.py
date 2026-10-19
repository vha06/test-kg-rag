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
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
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
        parser = ParserRegistry.get(profile.parser_strategy)
        blocks = parser.parse(raw_text=raw_text, doc=doc)
        timings["parse_blocks_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Chunking
        t0 = time.perf_counter()
        chunker = ChunkerRegistry.get(profile.chunking_strategy)
        chunks = chunker.chunk(doc=doc, blocks=blocks, profile=profile)
        timings["chunking_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Metadata Extraction
        t0 = time.perf_counter()
        for meta_name in profile.metadata_extractors:
            m_ext = ExtractorRegistry.get_metadata_extractor(meta_name)
            if m_ext:
                extracted_meta = m_ext.extract_metadata(doc=doc, blocks=blocks)
                doc.metadata[meta_name] = extracted_meta
        timings["metadata_extraction_ms"] = (time.perf_counter() - t0) * 1000

        # 5. Knowledge Extraction (Entities, Mentions, Relationships, Evidences)
        t0 = time.perf_counter()
        k_ext_name = profile.metadata_extractors[1] if len(profile.metadata_extractors) > 1 else "rule_based_ops"
        k_ext = ExtractorRegistry.get_knowledge_extractor(k_ext_name)
        entities, mentions, relationships, evidences = k_ext.extract_knowledge(
            doc=doc, blocks=blocks, chunks=chunks, profile=profile
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
            profile_id=profile.profile_id,
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
    ) -> Tuple[Dict[str, Any], PipelineExecutionTrace]:
        meta = dict(document_meta or {})
        meta["title"] = title
        if document_id:
            meta["document_id"] = document_id
        doc, blocks, chunks, entities, mentions, relationships, evidences, trace = self.process_document(
            raw_text=raw_text,
            document_metadata=meta,
            profile_id=meta.get("profile_id"),
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

