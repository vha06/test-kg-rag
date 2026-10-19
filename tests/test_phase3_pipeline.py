import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schema import CanonicalDocument
from core.registry import DomainRegistry, ProfileRegistry
from pipeline.parsers import GenericParagraphParser, StructuredHeadingParser
from pipeline.chunkers import GenericSlidingWindowChunker, BlockPreservingChunker
from pipeline.extractors import BasicStructuralMetadataExtractor, RuleBasedKnowledgeExtractor
from pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def sample_raw_text():
    return """# Hướng dẫn Khởi động Hệ thống
Trước khi khởi động máy chủ Gateway, kỹ sư phải kiểm tra dịch vụ AuthServer và cơ sở dữ liệu Redis.

Dịch vụ Gateway phụ thuộc AuthServer để xác thực người dùng.
Hệ thống Gateway lưu trữ tại Redis để tăng tốc độ phản hồi.

## Xử lý sự cố
Nếu xuất hiện sự cố INCIDENT_01, lỗi này gây ra bởi AuthServer do quá tải."""


def test_01_parsers(sample_raw_text):
    doc = CanonicalDocument(
        document_id="DOC_P3_01",
        title="Test Doc",
        source_id="S1",
        domain_id="synth_domain_ops",
        document_type="operational_manual",
        profile_id="profile_ops_standard",
        checksum="chk",
    )

    # Generic paragraph parser
    p1 = GenericParagraphParser()
    blocks1 = p1.parse(sample_raw_text, doc)
    assert len(blocks1) >= 3
    assert blocks1[0].block_type == "HEADING"
    assert blocks1[1].block_type == "PARAGRAPH"

    # Structured heading parser
    p2 = StructuredHeadingParser()
    blocks2 = p2.parse(sample_raw_text, doc)
    assert any(b.block_type == "HEADING" for b in blocks2)
    # Check that child paragraph has parent_block_id set
    child_blocks = [b for b in blocks2 if b.parent_block_id is not None]
    assert len(child_blocks) > 0


def test_02_chunkers(sample_raw_text):
    doc = CanonicalDocument(
        document_id="DOC_P3_02",
        title="Test Doc",
        source_id="S1",
        domain_id="synth_domain_ops",
        document_type="operational_manual",
        profile_id="profile_ops_standard",
        checksum="chk",
    )
    blocks = GenericParagraphParser().parse(sample_raw_text, doc)

    # Sliding window chunker
    from core.registry import ProcessingProfile
    prof_sliding = ProcessingProfile(
        profile_id="p_slide",
        domain_id="synth_domain_ops",
        document_type="operational_manual",
        chunking_strategy="sliding_window",
        chunking_parameters={"max_tokens": 30, "chunk_overlap": 5},
    )
    ch1 = GenericSlidingWindowChunker().chunk(doc, blocks, prof_sliding)
    assert len(ch1) > 0
    assert all(len(c.source_block_ids) > 0 for c in ch1)

    # Block preserving chunker
    prof_preserving = ProcessingProfile(
        profile_id="p_pres",
        domain_id="synth_domain_ops",
        document_type="incident_report",
        chunking_strategy="block_preserving",
        chunking_parameters={"max_tokens": 100},
    )
    ch2 = BlockPreservingChunker().chunk(doc, blocks, prof_preserving)
    assert len(ch2) > 0
    assert all(len(c.source_block_ids) > 0 for c in ch2)


def test_03_extractors(sample_raw_text):
    doc = CanonicalDocument(
        document_id="DOC_P3_03",
        title="Test Doc",
        source_id="S1",
        domain_id="synth_domain_ops",
        document_type="operational_manual",
        profile_id="profile_ops_standard",
        checksum="chk",
    )
    from core.registry import ProcessingProfile
    prof = ProcessingProfile(
        profile_id="p_ops",
        domain_id="synth_domain_ops",
        document_type="operational_manual",
    )
    blocks = GenericParagraphParser().parse(sample_raw_text, doc)
    chunks = GenericSlidingWindowChunker().chunk(doc, blocks, prof)

    # Structural metadata extractor
    meta = BasicStructuralMetadataExtractor().extract_metadata(doc, blocks)
    assert meta["total_blocks"] == len(blocks)
    assert meta["heading_count"] > 0

    # Knowledge extractor
    entities, mentions, relationships, evidences = RuleBasedKnowledgeExtractor().extract_knowledge(
        doc, blocks, chunks, prof
    )
    assert len(entities) >= 3  # Gateway, AuthServer, Redis...
    assert len(mentions) >= 3
    assert len(relationships) >= 1  # Gateway DEPENDS_ON AuthServer
    assert len(evidences) >= len(relationships)
    # Check that relationship has evidence_refs
    for rel in relationships:
        assert len(rel.evidence_refs) > 0
        assert rel.predicate in ("DEPENDS_ON", "STORES_DATA_IN", "TRIGGERED_BY")


def test_04_orchestrator_end_to_end(sample_raw_text):
    yaml_path = "config/platform_profiles.yaml"
    d_reg = DomainRegistry()
    d_reg.load_from_yaml(yaml_path)
    p_reg = ProfileRegistry(domain_registry=d_reg)
    p_reg.load_from_yaml(yaml_path)

    orchestrator = PipelineOrchestrator(domain_registry=d_reg, profile_registry=p_reg)

    doc, blocks, chunks, entities, mentions, relationships, evidences, trace = orchestrator.process_document(
        raw_text=sample_raw_text,
        document_metadata={
            "document_id": "DOC_E2E_01",
            "title": "Sổ tay Vận hành Cụm",
            "domain_id": "synth_domain_ops",
            "document_type": "operational_manual",
            "security_classification": "INTERNAL",
        },
    )

    assert doc.document_id == "DOC_E2E_01"
    assert doc.processing_status == "INDEXED"
    assert trace.is_valid is True
    assert trace.block_count == len(blocks)
    assert trace.chunk_count == len(chunks)
    assert trace.entity_count == len(entities)
    assert trace.relationship_count == len(relationships)
    assert trace.elapsed_ms > 0
    assert "parse_blocks_ms" in trace.step_timings


def test_05_new_chunkers_and_overrides(sample_raw_text):
    from pipeline.chunkers import HierarchicalHeadingChunker, FixedTokenChunker, ChunkerRegistry
    from pipeline.parsers import ParserRegistry
    from pipeline.extractors import ExtractorRegistry
    from core.registry import ProcessingProfile

    assert "hierarchical_heading" in ChunkerRegistry.list_strategies()
    assert "fixed_token" in ChunkerRegistry.list_strategies()
    assert "structured_heading" in ParserRegistry.list_strategies()
    assert "llm_extractor" in ExtractorRegistry.list_knowledge_extractors()

    doc = CanonicalDocument(
        document_id="DOC_P3_05",
        title="Hierarchical Doc",
        source_id="S5",
        domain_id="synth_domain_ops",
        document_type="operational_manual",
        profile_id="profile_ops_standard",
        checksum="chk",
    )
    blocks = ParserRegistry.get("structured_heading").parse(sample_raw_text, doc)
    prof = ProcessingProfile(
        profile_id="p_custom",
        domain_id="synth_domain_ops",
        document_type="operational_manual",
        chunking_strategy="hierarchical_heading",
        chunking_parameters={"max_tokens": 100},
    )

    chunks_h = HierarchicalHeadingChunker().chunk(doc, blocks, prof)
    assert len(chunks_h) > 0
    assert all(c.chunking_strategy == "hierarchical_heading" for c in chunks_h)

    chunks_f = FixedTokenChunker().chunk(doc, blocks, prof)
    assert len(chunks_f) > 0
    assert all(c.chunking_strategy == "fixed_token" for c in chunks_f)


def test_06_orchestrator_overrides_and_persist(sample_raw_text, tmp_path):
    from storage.db import DatabaseManager

    test_db_path = str(tmp_path / "test_persist.db")
    db = DatabaseManager(db_path=test_db_path)

    yaml_path = "config/platform_profiles.yaml"
    d_reg = DomainRegistry()
    d_reg.load_from_yaml(yaml_path)
    p_reg = ProfileRegistry(domain_registry=d_reg)
    p_reg.load_from_yaml(yaml_path)

    orchestrator = PipelineOrchestrator(
        domain_registry=d_reg, profile_registry=p_reg, db_manager=db
    )

    bundle, trace = orchestrator.process_raw_text(
        raw_text=sample_raw_text,
        title="Override Document",
        document_meta={"domain_id": "synth_domain_ops", "document_type": "operational_manual"},
        chunking_strategy_override="hierarchical_heading",
        chunking_params_override={"max_tokens": 80},
    )

    assert trace.is_valid is True
    assert all(c.chunking_strategy == "hierarchical_heading" for c in bundle["chunks"])

    # Persist bundle
    orchestrator.persist_bundle(bundle)

    # Verify persisted in DB
    persisted_chunks = db.get_platform_chunks(domain_id="synth_domain_ops")
    assert len(persisted_chunks) == len(bundle["chunks"])

