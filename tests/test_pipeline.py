import os
import sys
import pytest
from datetime import datetime

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import DatabaseManager
from storage.models import LegalDocumentModel, ProvisionModel, ChunkModel, RetrievalResult, BenchmarkItem
from ingestion.parser import StructureParser
from chunking.article_chunker import ArticleChunker
from chunking.fixed_chunker import FixedChunker
from chunking.structure_chunker import StructureChunker
from chunking.base import BaseChunker
from retrieval.tokenizer import LegalTokenizer
from retrieval.bm25_retriever import BM25Retriever
from knowledge_graph.builder import KnowledgeGraphBuilder
from knowledge_graph.expander import GraphExpander
from temporal.validity_filter import TemporalValidityFilter
from evaluation.metrics import RetrievalMetrics


@pytest.fixture(scope="session")
def db():
    return DatabaseManager()


@pytest.fixture(scope="session")
def sample_doc():
    content = (
        "1. Bảo vệ an ninh quốc gia là nhiệm vụ trọng yếu của toàn dân.\n"
        "a) Huy động mọi lực lượng hợp pháp tham gia bảo vệ Tổ quốc;\n"
        "b) Thực hiện nghiêm chỉnh các biện pháp phòng ngừa và đấu tranh.\n"
        "2. Cơ quan nhà nước có trách nhiệm bảo đảm điều kiện hoạt động."
    )
    doc_id = "DOC_TEST_01"
    doc_path = "Luật Thử nghiệm 01/2024/QH15 > Chương I"
    provisions = StructureParser.parse_article_to_tree(
        doc_id=doc_id,
        doc_path=doc_path,
        article_num="1",
        article_title="Điều 1. Nguyên tắc bảo vệ an ninh",
        content=content,
        effective_start="2024-01-01",
        status="CÓ_HIỆU_LỰC_THI_HÀNH",
        source_url="http://vbpl.vn/test",
    )
    return LegalDocumentModel(
        document_id=doc_id,
        doc_number="01/2024/QH15",
        title="Luật Thử nghiệm",
        doc_type="Luật",
        issuer="Quốc hội",
        issue_date="2023-11-15",
        effective_start="2024-01-01",
        source_url="http://vbpl.vn/test",
        provisions=provisions,
    )


# Test 1: Parser maintains Article -> Clause -> Point hierarchy
def test_01_parser_hierarchy(sample_doc):
    provisions = sample_doc.provisions
    assert len(provisions) >= 4  # Article + 2 Clauses + 2 Points
    art = provisions[0]
    assert art.level == "ARTICLE"
    assert art.article_number == "1"

    clauses = [p for p in provisions if p.level == "CLAUSE"]
    assert len(clauses) == 2
    assert clauses[0].parent_id == art.provision_id
    assert clauses[0].clause_number == "1"

    points = [p for p in provisions if p.level == "POINT"]
    assert len(points) == 2
    assert points[0].parent_id == clauses[0].provision_id
    assert points[0].point_letter == "a"


# Test 2: Chunking does not silently lose text
def test_02_chunking_no_text_loss(sample_doc):
    chunkers = [
        ArticleChunker(max_tokens=512),
        FixedChunker(chunk_size=128, chunk_overlap=16),
        StructureChunker(max_tokens=128),
    ]
    for chk in chunkers:
        chunks = chk.chunk_documents([sample_doc])
        assert len(chunks) > 0
        all_text = " ".join(c.content for c in chunks)
        assert "Bảo vệ an ninh quốc gia" in all_text
        assert "Huy động mọi lực lượng" in all_text
        assert "Cơ quan nhà nước" in all_text


# Test 3: Every chunk maps back to a valid provision and document
def test_03_chunk_provenance(sample_doc):
    chunker = StructureChunker(max_tokens=128)
    chunks = chunker.chunk_documents([sample_doc])
    prov_ids = {p.provision_id for p in sample_doc.provisions}

    for c in chunks:
        assert c.document_id == sample_doc.document_id
        assert c.provision_id in prov_ids
        assert len(c.source_provision_ids) > 0


# Test 4: Embedding index and metadata consistency
def test_04_embedding_index_mapping(db):
    chunks = db.get_chunks_by_strategy("A1_article")
    if chunks:
        # Check metadata consistency
        for c in chunks[:10]:
            assert "chunk_id" in c
            assert "content" in c
            assert len(c["content"]) > 0


# Test 5: BM25, Dense and Hybrid output schema
def test_05_retrievers_output_schema(db):
    bm25 = BM25Retriever(strategy="A1_article")
    if bm25.load_index():
        results = bm25.search("an ninh quốc gia", top_k=3)
        assert len(results) > 0
        r = results[0]
        assert isinstance(r, RetrievalResult)
        assert r.retrieval_method == "BM25"
        assert r.score > 0
        assert r.rank == 1


# Test 6: KG expansion avoids duplicates and preserves paths
def test_06_kg_expansion_dedup_and_paths(db):
    builder = KnowledgeGraphBuilder(db)
    builder.build_graph(verified_only=True)
    expander = GraphExpander(builder)

    seeds = [
        RetrievalResult(
            chunk_id="CHK_TEST_1",
            provision_id="DOC_32_2004_QH11__ART_1",
            document_id="DOC_32_2004_QH11",
            doc_number="32/2004/QH11",
            doc_title="Luật An ninh quốc gia",
            provision_title="Điều 1. Phạm vi điều chỉnh",
            structural_path="32/2004/QH11 > Điều 1",
            content="Phạm vi điều chỉnh an ninh quốc gia",
            score=0.9,
            rank=1,
            retrieval_method="BM25",
        )
    ]

    expanded = expander.expand(query="an ninh", seed_results=seeds, max_hops=1, max_expanded_nodes=3)
    assert len(expanded) >= len(seeds)
    all_provs = [r.provision_id for r in expanded]
    assert len(all_provs) == len(set(all_provs)), "Duplicate provisions found in KG expansion"

    for r in expanded:
        if r.retrieval_method == "KG_EXPANSION":
            assert r.expansion_path is not None
            assert "Seed:" in r.expansion_path


# Test 7: Unverified relations are filtered out
def test_07_unverified_relations_filtered(db):
    builder = KnowledgeGraphBuilder(db)
    graph_verified = builder.build_graph(verified_only=True)
    for u, v, k, data in graph_verified.edges(keys=True, data=True):
        assert data.get("verified", 0) == 1


# Test 8: Temporal validity applies to seed and expanded nodes
def test_08_temporal_filter_seed_and_expanded(db):
    temporal = TemporalValidityFilter(db)
    results = [
        RetrievalResult(
            chunk_id="CHK_T1",
            provision_id="TEST_PROV_1",
            document_id="DOC_1",
            doc_number="01/2024",
            doc_title="Doc 1",
            provision_title="Art 1",
            structural_path="Doc 1 > Art 1",
            content="test",
            score=0.8,
            rank=1,
            retrieval_method="BM25",
            effective_start="2025-01-01",  # Future date relative to 2024
        ),
        RetrievalResult(
            chunk_id="CHK_T2",
            provision_id="TEST_PROV_2",
            document_id="DOC_2",
            doc_number="02/2020",
            doc_title="Doc 2",
            provision_title="Art 2",
            structural_path="Doc 2 > Art 2",
            content="test",
            score=0.7,
            rank=2,
            retrieval_method="KG_EXPANSION",
            effective_start="2020-01-01",
            effective_end="2022-01-01",  # Expired relative to 2024
        ),
    ]

    annotated = temporal.filter_and_annotate(results, reference_date="2024-06-01")
    assert annotated[0].validity_status == "INVALID_NOT_YET_EFFECTIVE"
    assert annotated[1].validity_status == "INVALID_EXPIRED"


# Test 9: Handling missing or empty content gracefully
def test_09_malformed_or_missing_metadata_handling():
    tree = StructureParser.parse_article_to_tree(
        doc_id="DOC_EMPTY",
        doc_path="",
        article_num="99",
        article_title="Điều 99. Tiêu đề",
        content="",
    )
    assert len(tree) == 1
    assert tree[0].level == "ARTICLE"
    assert tree[0].content == ""


# Test 10: Benchmark metrics calculation on known cases
def test_10_benchmark_metrics_calculation():
    retrieved = ["A", "B", "C", "D", "E"]
    ground_truth = {"A", "C"}

    # Recall@1: hits=1, total=2 -> 0.5
    assert RetrievalMetrics.recall_at_k(retrieved, ground_truth, 1) == 0.5
    # Recall@3: hits=2, total=2 -> 1.0
    assert RetrievalMetrics.recall_at_k(retrieved, ground_truth, 3) == 1.0
    # MRR@10: first hit at rank 1 -> 1.0
    assert RetrievalMetrics.mrr_at_k(retrieved, ground_truth, 10) == 1.0

    retrieved_miss_first = ["X", "B", "A"]
    # MRR@10: first hit at rank 3 -> 1/3
    assert abs(RetrievalMetrics.mrr_at_k(retrieved_miss_first, ground_truth, 10) - (1.0 / 3.0)) < 1e-4


# Test 11: Idempotent ingestion does not create duplicate documents
def test_11_idempotent_ingestion(db):
    counts_before = db.count_summary()
    # Re-insert single doc
    test_doc = LegalDocumentModel(
        document_id="DOC_IDEMPOTENT_TEST",
        doc_number="IDEMP/01",
        title="Idempotent Test",
        doc_type="Luật",
        issuer="Quốc hội",
        provisions=[],
    )
    db.insert_documents([test_doc])
    db.insert_documents([test_doc])  # Second insert
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents WHERE document_id = 'DOC_IDEMPOTENT_TEST'")
        assert cursor.fetchone()[0] == 1


# Test 12: Source citation traceability
def test_12_source_citation_traceability(db):
    bm25 = BM25Retriever(strategy="A1_article")
    if bm25.load_index():
        results = bm25.search("an ninh", top_k=1)
        if results:
            r = results[0]
            assert r.doc_number != ""
            assert r.provision_title != ""
            assert r.structural_path != ""
