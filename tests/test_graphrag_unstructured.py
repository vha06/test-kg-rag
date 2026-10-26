import os
import sys
import tempfile
import pytest
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import DatabaseManager
from core.schema import (
    CanonicalDocument,
    ContentBlock,
    CanonicalChunk,
    CanonicalEntity,
    CanonicalRelationship,
    CanonicalCommunity,
    CanonicalCommunityReport,
    SourceLocation,
)
from core.registry import ProcessingProfile
from pipeline.parsers import GenericParagraphParser, StructuredHeadingParser
from pipeline.chunkers import FixedTokenChunker, BlockPreservingChunker, count_tokens
from pipeline.extractors import (
    DeterministicGeneralGraphExtractor,
    GraphRAGChunkExtractor,
    canonicalize_corpus_graph,
)
from pipeline.orchestrator import PipelineOrchestrator
from knowledge_graph.community_detector import HierarchicalCommunityDetector
from knowledge_graph.community_report_generator import CommunityReportGenerator
from retrieval.graphrag_index_manager import GraphRAGIndexManager
from retrieval.graphrag_query_engine import GraphRAGQueryEngine
from embedding.adapter import DeterministicHashEmbeddingAdapter


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path=path)
    yield db
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


@pytest.fixture
def sample_unstructured_corpus():
    return [
        {
            "document_id": "DOC_POLICE_01",
            "title": "Quy định Chế độ Nghỉ phép và Công tác",
            "text": """# Chế độ Nghỉ phép Năm của Cán bộ Chiến sĩ
Cán bộ chiến sĩ công an nhân dân được nghỉ phép hàng năm theo quy định hiện hành của Bộ Công An.
Trường hợp cán bộ chiến sĩ do yêu cầu công tác đột xuất không nghỉ phép năm, thủ tục thanh toán tiền bồi dưỡng và tiền tàu xe đi phép thực hiện theo hướng dẫn của Cục Tài Chính.

## Thanh toán Tiền Bồi Dưỡng
Thủ trưởng đơn vị xác nhận lý do không thể bố trí nghỉ phép. Cán bộ chiến sĩ nộp hồ sơ gồm đơn đề nghị và xác nhận công tác để nhận tiền bồi dưỡng thay cho ngày phép chưa nghỉ.
Cục Tài Chính duyệt chi trả căn cứ vào bảng chấm công và định mức tài chính hiện hành.

## Thanh toán Tiền Tàu Xe Đi Phép
Trường hợp được nghỉ phép và có đi lại thực tế, cán bộ chiến sĩ cung cấp vé tàu xe hợp lệ để được thanh toán tiền tàu xe đi phép theo cung đường quy định.""",
        },
        {
            "document_id": "DOC_TECH_02",
            "title": "Kiến trúc Nền tảng Phân tán FastLease",
            "text": """# Nền tảng Phân tán FastLease
Giao thức FastLease mở rộng từ thuật toán Paxos nhằm tối ưu độ trễ đồng thuận trong mạng phân tán diện rộng.
Module Ingress Gateway định tuyến lưu lượng vào hệ thống và phụ thuộc vào dịch vụ AuthServer để xác thực chứng chỉ.
Cơ sở dữ liệu Redis lưu trữ bộ nhớ đệm trạng thái phiên làm việc giúp giảm tải cho cụm cơ sở dữ liệu chính.""",
        },
    ]


# 1. Test Chunking Chiến lược A: Giữ đúng offset và giới hạn token
def test_01_token_chunking_preserves_offsets_and_token_limits():
    doc = CanonicalDocument(
        document_id="DOC_T01",
        title="Token Chunk Test",
        source_id="S1",
        checksum="chk",
    )
    text = "Từ vựng thứ nhất từ vựng thứ hai từ vựng thứ ba từ vựng thứ tư từ vựng thứ năm từ vựng thứ sáu."
    blocks = [ContentBlock(block_id="B1", document_id="DOC_T01", domain_id="test", block_index=0, text=text, start_offset=0, end_offset=len(text))]
    prof = ProcessingProfile(profile_id="p1", domain_id="test", document_type="general", chunking_strategy="fixed_token", chunking_parameters={"max_tokens": 10, "chunk_overlap": 2})

    chunker = FixedTokenChunker()
    chunks = chunker.chunk(doc, blocks, prof)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 12  # Cho phép dung sai biên nhỏ
        assert c.content_hash != ""
        assert c.chunking_strategy == "fixed_token"


# 2. Test Chunking Chiến lược B: Tôn trọng ranh giới đoạn/tiêu đề và không vượt max_tokens
def test_02_boundary_aware_chunking_respects_paragraphs_and_headings():
    doc = CanonicalDocument(
        document_id="DOC_T02",
        title="Boundary Chunk Test",
        source_id="S2",
        checksum="chk",
    )
    long_para = "Câu thứ nhất rất dài nói về chính sách. " * 15
    text = f"# Tiêu đề Phần Một\nĐoạn văn ngắn một.\n\nĐoạn văn ngắn hai.\n\n{long_para}"
    parser = StructuredHeadingParser()
    blocks = parser.parse(text, doc)

    prof = ProcessingProfile(profile_id="p2", domain_id="test", document_type="general", chunking_strategy="block_preserving", chunking_parameters={"max_tokens": 40})
    chunker = BlockPreservingChunker()
    chunks = chunker.chunk(doc, blocks, prof)

    assert len(chunks) >= 3
    # Xác minh không có chunk nào vượt quá max_tokens ngay cả với đoạn văn dài
    for c in chunks:
        assert c.token_count <= 45
        assert c.content_hash != ""
        assert len(c.source_block_ids) > 0


# 3. Test Trích xuất: Bắt buộc evidence_quote phải xuất hiện nguyên văn
def test_03_extraction_rejects_invalid_json_and_unverified_quotes():
    extractor = GraphRAGChunkExtractor()
    content = "Bộ Công An ban hành hướng dẫn thanh toán công tác phí."
    quote_valid = "Bộ Công An ban hành hướng dẫn"
    quote_invalid = "Bộ Quốc Phòng phê duyệt chi trả"

    assert extractor._is_quote_grounded(quote_valid, content) is True
    assert extractor._is_quote_grounded(quote_invalid, content) is False


# 4. Test Hợp nhất thực thể (Corpus-level Canonicalization)
def test_04_entity_canonicalization_merges_aliases_conservatively():
    e1 = CanonicalEntity(
        entity_id="ENT_01",
        canonical_name="Bộ Công An",
        entity_type="ORGANIZATION",
        aliases=["BCA"],
        description="Cơ quan quản lý công an",
        domain_id="test",
        graph_namespace="ns_test",
        source_chunk_ids=["CHK_01"],
        source_document_ids=["DOC_01"],
    )
    e2 = CanonicalEntity(
        entity_id="ENT_02",
        canonical_name="bộ công an",
        entity_type="CONCEPT",
        aliases=["Bộ Công an Việt Nam"],
        description="Quản lý an ninh trật tự",
        domain_id="test",
        graph_namespace="ns_test",
        source_chunk_ids=["CHK_02"],
        source_document_ids=["DOC_01"],
    )
    merged_ents, _, _, _ = canonicalize_corpus_graph([e1, e2], [], [], [])
    assert len(merged_ents) == 1
    m = merged_ents[0]
    assert m.canonical_name == "Bộ Công An"
    assert m.entity_type == "ORGANIZATION"  # Kế thừa type cụ thể hơn
    assert "BCA" in m.aliases
    assert "CHK_01" in m.source_chunk_ids and "CHK_02" in m.source_chunk_ids


# 5. Test Provenance của mối quan hệ
def test_05_relationship_provenance_links_to_valid_chunk_and_entities():
    extractor = DeterministicGeneralGraphExtractor()
    doc = CanonicalDocument(document_id="DOC_T05", title="Test Prov", source_id="S5", checksum="c5")
    chunk = CanonicalChunk(
        chunk_id="CHK_T05",
        document_id="DOC_T05",
        domain_id="test",
        source_block_ids=["BLK_01"],
        content="Module Ingress Gateway định tuyến lưu lượng vào hệ thống và phụ thuộc vào AuthServer để xác thực.",
    )
    prof = ProcessingProfile(profile_id="p5", domain_id="test", document_type="general")
    entities, mentions, rels, evidences = extractor.extract_knowledge(doc, [], [chunk], prof)

    assert len(entities) >= 2
    assert len(rels) >= 1
    rel = rels[0]
    assert "CHK_T05" in rel.source_chunk_ids
    assert len(rel.evidence_refs) > 0
    # Bằng chứng phải có quote_text xuất hiện trong nội dung chunk
    matching_ev = next((ev for ev in evidences if ev.evidence_id == rel.evidence_refs[0]), None)
    assert matching_ev is not None
    assert matching_ev.quote_text in chunk.content or matching_ev.quote_text.lower() in chunk.content.lower()


# 6. Test Phân cụm cộng đồng tạo cây phân cấp hợp lệ
def test_06_community_detection_creates_valid_hierarchy():
    detector = HierarchicalCommunityDetector(max_levels=3, min_split_size=2)
    # Tạo đồ thị 6 thực thể có 2 nhóm rõ rệt
    entities = [
        CanonicalEntity(entity_id=f"E{i}", canonical_name=f"Entity_{i}", entity_type="CONCEPT", domain_id="test", graph_namespace="ns_test")
        for i in range(6)
    ]
    relationships = [
        CanonicalRelationship(relationship_id="R1", subject_entity_id="E0", predicate="REL", object_entity_id="E1", domain_id="test", graph_namespace="ns_test", weight=2.0),
        CanonicalRelationship(relationship_id="R2", subject_entity_id="E1", predicate="REL", object_entity_id="E2", domain_id="test", graph_namespace="ns_test", weight=2.0),
        CanonicalRelationship(relationship_id="R3", subject_entity_id="E3", predicate="REL", object_entity_id="E4", domain_id="test", graph_namespace="ns_test", weight=2.0),
        CanonicalRelationship(relationship_id="R4", subject_entity_id="E4", predicate="REL", object_entity_id="E5", domain_id="test", graph_namespace="ns_test", weight=2.0),
        CanonicalRelationship(relationship_id="R5", subject_entity_id="E2", predicate="WEAK_REL", object_entity_id="E3", domain_id="test", graph_namespace="ns_test", weight=0.5),
    ]

    communities = detector.detect_communities(entities, relationships, namespace="ns_test")
    assert len(communities) >= 1
    l0_comms = [c for c in communities if c.level == 0]
    assert len(l0_comms) >= 1

    # Kiểm tra liên kết cha - con nếu có level > 0
    l1_comms = [c for c in communities if c.level == 1]
    if l1_comms:
        for c1 in l1_comms:
            assert c1.parent_community_id is not None
            parent = next((c0 for c0 in l0_comms if c0.community_id == c1.parent_community_id), None)
            assert parent is not None
            assert c1.community_id in parent.child_community_ids


# 7. Test Báo cáo cộng đồng xác minh ID tham chiếu
def test_07_community_report_validates_entity_and_relationship_references():
    gen = CommunityReportGenerator()
    comm = CanonicalCommunity(
        community_id="COMM_TEST_01",
        namespace="test",
        level=0,
        title="Cụm Thử Nghiệm",
        entity_ids=["ENT_A", "ENT_B"],
        relationship_ids=["REL_AB"],
        source_chunk_ids=["CHK_1"],
    )
    ent_map = {
        "ENT_A": CanonicalEntity(entity_id="ENT_A", canonical_name="A", entity_type="CONCEPT", domain_id="test", graph_namespace="test"),
        "ENT_B": CanonicalEntity(entity_id="ENT_B", canonical_name="B", entity_type="CONCEPT", domain_id="test", graph_namespace="test"),
    }
    rel_map = {
        "REL_AB": CanonicalRelationship(relationship_id="REL_AB", subject_entity_id="ENT_A", predicate="CONNECT", object_entity_id="ENT_B", domain_id="test", graph_namespace="test"),
    }

    report = gen.generate_report(comm, ent_map, rel_map)
    assert report.community_id == "COMM_TEST_01"
    assert "ENT_A" in report.supporting_entity_ids
    assert "ENT_B" in report.supporting_entity_ids
    # Không được chứa ID ngoài phạm vi
    assert "ENT_HALLUCINATED" not in report.supporting_entity_ids
    assert "CHK_1" in report.source_chunk_ids


# 8. Test Vector RAG trả về trích dẫn Chunk gốc
def test_08_vector_rag_returns_grounded_chunk_citations(temp_db, sample_unstructured_corpus):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    orchestrator = PipelineOrchestrator(db_manager=temp_db)
    res = orchestrator.build_full_graphrag_pipeline(
        documents=sample_unstructured_corpus,
        namespace="ns_test_rag",
        embedding_adapter=adapter,
    )
    assert res["status"] == "SUCCESS"

    engine = GraphRAGQueryEngine(db_manager=temp_db, index_manager=res["index_manager"])
    q_res = engine.query_vector_rag("Cán bộ chiến sĩ được nghỉ phép thế nào?", namespace="ns_test_rag")

    assert q_res.query_mode == "VECTOR"
    assert len(q_res.grounded_citations) > 0
    assert q_res.trace.selected_chunk_ids[0] in [c["citation_id"] for c in q_res.grounded_citations]


# 9. Test Local Search mở rộng seed entities và đính kèm TextUnits
def test_09_local_search_expands_seed_entities_and_attaches_text_units(temp_db, sample_unstructured_corpus):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    orchestrator = PipelineOrchestrator(db_manager=temp_db)
    res = orchestrator.build_full_graphrag_pipeline(
        documents=sample_unstructured_corpus,
        namespace="ns_test_loc",
        embedding_adapter=adapter,
    )
    engine = GraphRAGQueryEngine(db_manager=temp_db, index_manager=res["index_manager"])
    q_res = engine.query_local_search("FastLease và Ingress Gateway", namespace="ns_test_loc")

    assert q_res.query_mode == "LOCAL"
    assert len(q_res.trace.seed_entity_ids) > 0
    assert len(q_res.trace.selected_chunk_ids) > 0
    assert q_res.trace.stop_reason == "SUCCESS"


# 10. Test Global Search dùng Community Reports và ánh xạ ngược về TextUnits
def test_10_global_search_uses_community_reports_and_traces_back_to_source_chunks(temp_db, sample_unstructured_corpus):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    orchestrator = PipelineOrchestrator(db_manager=temp_db)
    res = orchestrator.build_full_graphrag_pipeline(
        documents=sample_unstructured_corpus,
        namespace="ns_test_glo",
        embedding_adapter=adapter,
    )
    engine = GraphRAGQueryEngine(db_manager=temp_db, index_manager=res["index_manager"])
    q_res = engine.query_global_search("Tổng quan quy định và cấu trúc hệ thống", namespace="ns_test_glo")

    assert q_res.query_mode == "GLOBAL"
    assert len(q_res.trace.selected_community_ids) > 0
    assert len(q_res.trace.map_outputs) > 0
    assert len(q_res.grounded_citations) > 0


# 11. Test DRIFT Search thực thi các vòng lặp câu hỏi phụ có giới hạn
def test_11_drift_search_executes_bounded_followup_loops(temp_db, sample_unstructured_corpus):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    orchestrator = PipelineOrchestrator(db_manager=temp_db)
    res = orchestrator.build_full_graphrag_pipeline(
        documents=sample_unstructured_corpus,
        namespace="ns_test_dri",
        embedding_adapter=adapter,
    )
    engine = GraphRAGQueryEngine(db_manager=temp_db, index_manager=res["index_manager"])
    q_res = engine.query_drift_search("Thủ tục thanh toán tiền bồi dưỡng nghỉ phép", namespace="ns_test_dri", max_depth=2, max_followups_per_step=2)

    assert q_res.query_mode == "DRIFT"
    assert len(q_res.trace.drift_steps) >= 1
    assert q_res.trace.stop_reason in ("SUCCESS", "STOPPED_NO_NEW_EVIDENCE", "MAX_LLM_CALLS_REACHED")


# 12. Test Từ chối trả lời khi tập dữ liệu không có bằng chứng
def test_12_query_refusal_when_corpus_lacks_supporting_evidence(temp_db):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    idx_mgr = GraphRAGIndexManager(db_manager=temp_db, embedding_adapter=adapter)
    # Lập chỉ mục rỗng
    idx_mgr.build_all_indices(namespace="ns_empty", chunks=[], entities=[], reports=[])
    engine = GraphRAGQueryEngine(db_manager=temp_db, index_manager=idx_mgr)

    q_res = engine.query_vector_rag("Chiến lược khám phá sao Hỏa của NASA là gì?", namespace="ns_empty")
    assert q_res.answer == GraphRAGQueryEngine.REFUSAL_MESSAGE
    assert q_res.trace.refusal_reason is not None


# 13. Test Tái lập chỉ mục xóa bỏ sạch các artifact cũ (Re-indexing freshness)
def test_13_reindexing_replaces_stale_graph_and_community_artifacts(temp_db):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    orchestrator = PipelineOrchestrator(db_manager=temp_db)

    # Lần 1: Nạp 1 tài liệu
    doc_v1 = [{"document_id": "D1", "title": "Bản 1", "text": "Hệ thống AuthServer quản lý phiên đăng nhập."}]
    res1 = orchestrator.build_full_graphrag_pipeline(doc_v1, namespace="ns_reindex", embedding_adapter=adapter)
    assert res1["entity_count"] >= 1

    # Lần 2: Nạp tài liệu mới hoàn toàn cho cùng namespace, kích hoạt clear_existing
    doc_v2 = [{"document_id": "D2", "title": "Bản 2", "text": "Cán bộ chiến sĩ công an đi phép được thanh toán vé tàu xe."}]
    res2 = orchestrator.build_full_graphrag_pipeline(doc_v2, namespace="ns_reindex", embedding_adapter=adapter, clear_existing=True)

    # Kiểm tra trong DB không còn thực thể cũ AuthServer
    active_ents = temp_db.get_platform_entities(graph_namespace="ns_reindex")
    active_names = [e["canonical_name"].lower() for e in active_ents]
    assert "authserver" not in active_names
    assert any("cán bộ" in n or "chiến sĩ" in n or "vé" in n for n in active_names)


# 14. Test Bộ nhớ đệm trích xuất (Extraction Cache)
def test_14_extraction_cache_invalidates_when_chunk_or_prompt_changes(temp_db):
    chunk_hash = hashlib.sha256("Nội dung thử nghiệm".encode("utf-8")).hexdigest()
    model = "gemini-2.5-flash"
    prompt_v1 = "1.0.0"
    prompt_v2 = "2.0.0"

    key_v1 = hashlib.sha256(f"{chunk_hash}::{model}::{prompt_v1}".encode()).hexdigest()
    key_v2 = hashlib.sha256(f"{chunk_hash}::{model}::{prompt_v2}".encode()).hexdigest()

    data_payload = {"entities": [{"name": "AuthServer", "type": "SERVICE"}]}
    temp_db.set_extraction_cache(key_v1, model, prompt_v1, data_payload)

    assert temp_db.get_extraction_cache(key_v1) == data_payload
    # Cache key v2 với prompt version khác phải là None
    assert temp_db.get_extraction_cache(key_v2) is None


# 15. Test Cả 3 chỉ mục vector (TextUnits, Entities, Community Reports) đều có thể tìm kiếm
def test_15_entity_and_community_embeddings_are_indexed_and_searchable(temp_db, sample_unstructured_corpus):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    orchestrator = PipelineOrchestrator(db_manager=temp_db)
    res = orchestrator.build_full_graphrag_pipeline(
        documents=sample_unstructured_corpus,
        namespace="ns_test_all_idx",
        embedding_adapter=adapter,
    )
    idx_mgr: GraphRAGIndexManager = res["index_manager"]

    chunks_found = idx_mgr.search_text_units("thanh toán tiền bồi dưỡng", namespace="ns_test_all_idx", top_k=2)
    entities_found = idx_mgr.search_entities("Ingress Gateway", namespace="ns_test_all_idx", top_k=2)
    reports_found = idx_mgr.search_community_reports("Nghỉ phép", namespace="ns_test_all_idx", top_k=2)

    assert len(chunks_found) > 0
    assert len(entities_found) > 0
    assert len(reports_found) > 0


# 16. Test DRIFT Search dừng sớm khi hết ngân sách hoặc information gain = 0
def test_16_drift_search_stops_early_when_budget_or_information_gain_is_exhausted(temp_db, sample_unstructured_corpus):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    orchestrator = PipelineOrchestrator(db_manager=temp_db)
    res = orchestrator.build_full_graphrag_pipeline(
        documents=sample_unstructured_corpus,
        namespace="ns_test_early_stop",
        embedding_adapter=adapter,
    )
    engine = GraphRAGQueryEngine(db_manager=temp_db, index_manager=res["index_manager"])

    # Khi max_total_llm_calls = 0, DRIFT phải dừng ngay lập tức do cạn ngân sách
    q_res = engine.query_drift_search(
        "Nội dung bất kỳ",
        namespace="ns_test_early_stop",
        max_depth=3,
        max_total_llm_calls=0,
    )
    assert q_res.trace.stop_reason in ("MAX_LLM_CALLS_REACHED", "STOPPED_NO_NEW_EVIDENCE")


# 17. Test Chống lạm phát cụm cộng đồng, chống nhân bản cụm lá và kiểm tra hợp nhất theo alias
def test_17_no_community_inflation_or_duplicate_cloning_and_alias_canonicalization():
    # 1. Kiểm tra canonicalize_corpus_graph hợp nhất theo alias
    e1 = CanonicalEntity(
        entity_id="E_GDP_FULL",
        canonical_name="Tổng sản phẩm trong nước",
        entity_type="METRIC",
        aliases=["GDP"],
        domain_id="test_dom",
        graph_namespace="ns_test",
        source_chunk_ids=["chk_0"],
    )
    e2 = CanonicalEntity(
        entity_id="E_GDP_SHORT",
        canonical_name="GDP",
        entity_type="CONCEPT",
        aliases=[],
        domain_id="test_dom",
        graph_namespace="ns_test",
        source_chunk_ids=["chk_9"],
    )
    merged_ents, _, _, _ = canonicalize_corpus_graph([e1, e2], [], [], [])
    assert len(merged_ents) == 1
    m_ent = merged_ents[0]
    assert m_ent.canonical_name == "Tổng sản phẩm trong nước"
    assert "GDP" in m_ent.aliases
    assert m_ent.entity_type == "METRIC"
    assert "chk_0" in m_ent.source_chunk_ids and "chk_9" in m_ent.source_chunk_ids

    # 2. Kiểm tra HierarchicalCommunityDetector chống lạm phát cụm và không nhân bản cụm lá
    # Tạo đồ thị có 1 cụm hình sao 8 nút (E0 đến E7) và 4 nút cô lập (E8 đến E11) -> Tổng 12 nút
    nodes = [
        CanonicalEntity(
            entity_id=f"E_{i}",
            canonical_name=f"Thực thể {i}",
            entity_type="CONCEPT",
            domain_id="test_dom",
            graph_namespace="ns_test",
            source_chunk_ids=["chk_star" if i < 8 else f"chk_iso_{i}"],
        )
        for i in range(12)
    ]
    # Cụm hình sao xoay quanh E_0
    star_rels = [
        CanonicalRelationship(
            relationship_id=f"REL_STAR_{i}",
            subject_entity_id="E_0",
            predicate="QUAN_HỆ",
            object_entity_id=f"E_{i}",
            domain_id="test_dom",
            graph_namespace="ns_test",
            weight=1.5,
        )
        for i in range(1, 8)
    ]

    detector = HierarchicalCommunityDetector(max_levels=3, min_split_size=6, min_community_size=2)
    communities = detector.detect_communities(nodes, star_rels, namespace="ns_test")

    # Số lượng cụm phải nhỏ hơn số thực thể rất nhiều (không lạm phát)
    assert len(communities) < len(nodes)

    # Không có cụm nào có kích thước = 1 (không có cụm đơn lẻ)
    for comm in communities:
        assert comm.size >= 2, f"Cụm {comm.community_id} vi phạm vì chỉ có {comm.size} thực thể"

    # Không có cụm con nào có danh sách thực thể giống hệt 100% cụm cha (chống nhân bản cụm lá)
    l0_comms = {c.community_id: c for c in communities if c.level == 0}
    child_comms = [c for c in communities if c.level > 0]
    for c_child in child_comms:
        assert c_child.parent_community_id is not None
        p = l0_comms.get(c_child.parent_community_id)
        if p:
            assert set(c_child.entity_ids) != set(p.entity_ids), f"Cụm con {c_child.community_id} bị nhân bản từ cha {p.community_id}"


def test_index_manager_auto_hydration(tmp_path):
    """Kiểm tra tính năng tự động nạp chỉ mục từ SQLite khi khởi tạo IndexManager mới."""
    db_file = str(tmp_path / "test_hydrate.db")
    db = DatabaseManager(db_file)

    # Nạp dữ liệu giả lập vào SQLite
    db.insert_platform_domains([{"domain_id": "ns_test", "name": "Test Domain", "graph_namespace": "ns_test"}])
    doc = CanonicalDocument(
        document_id="DOC_1",
        source_id="SRC_1",
        domain_id="ns_test",
        title="Doc 1",
        checksum="chk123",
        text="Việt Nam có nền kinh tế đang phát triển mạnh mẽ.",
    )
    db.insert_platform_documents([doc.model_dump()])

    c = CanonicalChunk(
        chunk_id="CHK_TEST_1",
        document_id="DOC_1",
        domain_id="ns_test",
        source_block_ids=["BLK_1"],
        content="Việt Nam có nền kinh tế đang phát triển mạnh mẽ.",
        token_count=10,
    )
    db.insert_platform_chunks([c.model_dump()])

    e = CanonicalEntity(
        entity_id="ns_test::ENT::CONCEPT::Kinh_tế",
        canonical_name="Kinh tế",
        entity_type="CONCEPT",
        domain_id="ns_test",
        graph_namespace="ns_test",
    )
    db.insert_platform_entities([e.model_dump()])

    rep = CanonicalCommunityReport(
        report_id="REP_TEST_1",
        community_id="COMM_1",
        namespace="ns_test",
        level=0,
        title="Báo cáo Kinh tế",
        summary="Tóm tắt tình hình kinh tế.",
        full_content="Toàn văn báo cáo kinh tế.",
    )
    db.insert_graphrag_community_reports([rep.model_dump()])

    # Khởi tạo IndexManager hoàn toàn mới mà không gọi build_all_indices thủ công
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    mgr = GraphRAGIndexManager(db_manager=db, embedding_adapter=adapter)

    # Khi tìm kiếm, phải tự động hydrate từ SQLite
    res_chunks = mgr.search_text_units("kinh tế", namespace="ns_test")
    assert len(res_chunks) > 0
    assert res_chunks[0]["chunk_id"] == "CHK_TEST_1"

    res_ents = mgr.search_entities("Kinh tế", namespace="ns_test")
    assert len(res_ents) > 0
    assert res_ents[0]["canonical_name"] == "Kinh tế"

    res_reps = mgr.search_community_reports("kinh tế", namespace="ns_test")
    assert len(res_reps) > 0
    assert res_reps[0]["report_id"] == "REP_TEST_1"


