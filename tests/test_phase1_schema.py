import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schema import (
    SourceLocation,
    ACLPolicy,
    CanonicalDocument,
    ContentBlock,
    CanonicalChunk,
    EvidenceRecord,
    CanonicalEntity,
    EntityMention,
    CanonicalRelationship,
)
from core.validator import CanonicalSchemaValidator


def test_01_canonical_schema_initialization():
    doc = CanonicalDocument(
        document_id="DOC_01",
        title="Test Document",
        source_id="SRC_01",
        domain_id="domain_test",
        document_type="report",
        profile_id="profile_default",
        checksum="hash123",
    )
    assert doc.document_id == "DOC_01"
    assert doc.security_classification == "INTERNAL"
    assert doc.schema_version == "1.0.0"

    block = ContentBlock(
        block_id="BLK_01",
        document_id="DOC_01",
        domain_id="domain_test",
        block_index=0,
        block_type="PARAGRAPH",
        text="Sample text content.",
    )
    assert block.block_type == "PARAGRAPH"

    chunk = CanonicalChunk(
        chunk_id="CHK_01",
        document_id="DOC_01",
        domain_id="domain_test",
        source_block_ids=["BLK_01"],
        content="Sample text content.",
        chunk_index=0,
        chunking_strategy="sliding_window",
        profile_id="profile_default",
    )
    assert chunk.source_block_ids == ["BLK_01"]


def test_02_source_location_offsets_validation():
    loc = SourceLocation(start_offset=10, end_offset=20)
    assert loc.start_offset == 10
    assert loc.end_offset == 20

    with pytest.raises(ValueError, match="cannot be greater than"):
        SourceLocation(start_offset=25, end_offset=10)

    with pytest.raises(ValueError, match="must be non-negative"):
        SourceLocation(start_offset=-1, end_offset=10)


def test_03_canonical_chunk_requires_source_blocks():
    with pytest.raises(ValueError, match="at least one source_block_id"):
        CanonicalChunk(
            chunk_id="CHK_EMPTY",
            document_id="DOC_01",
            domain_id="domain_test",
            source_block_ids=[],
            content="Empty source blocks",
            chunk_index=0,
            chunking_strategy="fixed",
            profile_id="profile_default",
        )


def test_04_validator_bundle_valid_example():
    with open("data/examples/canonical_valid_examples.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    doc = CanonicalDocument.model_validate(data["document"])
    blocks = [ContentBlock.model_validate(b) for b in data["blocks"]]
    chunks = [CanonicalChunk.model_validate(c) for c in data["chunks"]]
    evidences = [EvidenceRecord.model_validate(e) for e in data["evidences"]]
    entities = [CanonicalEntity.model_validate(ent) for ent in data["entities"]]
    mentions = [EntityMention.model_validate(m) for m in data["mentions"]]
    relationships = [CanonicalRelationship.model_validate(r) for r in data["relationships"]]

    is_valid, issues = CanonicalSchemaValidator.validate_document_bundle(
        document=doc,
        blocks=blocks,
        chunks=chunks,
        entities=entities,
        mentions=mentions,
        relationships=relationships,
        evidences=evidences,
    )
    assert is_valid is True
    assert len(issues) == 0


def test_05_validator_catches_integrity_and_provenance_errors():
    doc = CanonicalDocument(
        document_id="DOC_ERR_01",
        title="Err Doc",
        source_id="S1",
        domain_id="domain_A",
        document_type="type_x",
        profile_id="prof_1",
        checksum="ch1",
    )
    # Block belongs to another document
    block = ContentBlock(
        block_id="BLK_01",
        document_id="DOC_OTHER",
        domain_id="domain_A",
        block_index=0,
        text="text",
    )
    # Chunk references non-existent block
    chunk = CanonicalChunk(
        chunk_id="CHK_01",
        document_id="DOC_ERR_01",
        domain_id="domain_A",
        source_block_ids=["NON_EXISTENT_BLOCK"],
        content="chunk text",
        chunk_index=0,
        chunking_strategy="strat",
        profile_id="prof_1",
    )
    # Verified relationship missing evidence_refs
    rel = CanonicalRelationship(
        relationship_id="REL_01",
        subject_entity_id="ENT_1",
        predicate="CONNECTS",
        object_entity_id="ENT_2",
        domain_id="domain_A",
        graph_namespace="ns_A",
        evidence_refs=[],
        verification_status="VERIFIED",
    )

    is_valid, issues = CanonicalSchemaValidator.validate_document_bundle(
        document=doc,
        blocks=[block],
        chunks=[chunk],
        relationships=[rel],
        enforce_verified_evidence=True,
    )
    assert is_valid is False
    error_messages = [i.message for i in issues if i.level == "ERROR"]
    assert any("does not match Document" in m for m in error_messages)
    assert any("non-existent source_block_id" in m for m in error_messages)
    assert any("must have at least one valid evidence reference" in m for m in error_messages)
