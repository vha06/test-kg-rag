import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.models import RetrievalResult
from generation.models import (
    LLMConfig,
    LegalCommunity,
    GlobalSearchResponse,
    DRIFTSearchResponse,
    ThreeWayComparisonResult,
)
from knowledge_graph.community import LegalCommunityDetector
from knowledge_graph.expander import GraphExpander
from retrieval.global_search import GlobalSearchEngine
from retrieval.drift_search import DRIFTSearchEngine
from generation.answer_generator import LegalAnswerGenerator


@pytest.fixture
def mock_cfg():
    return LLMConfig(provider="mock")


@pytest.fixture
def sample_seeds():
    return [
        RetrievalResult(
            chunk_id="CHK_BORDER_01",
            provision_id="PROV_112_ART_08",
            document_id="DOC_112_2014",
            doc_number="112/2014/NĐ-CP",
            doc_title="Nghị định Quản lý cửa khẩu biên giới đất liền",
            provision_title="Điều 8. Kiểm soát xuất cảnh, nhập cảnh",
            structural_path="112/2014/NĐ-CP > Điều 8",
            content="Người và phương tiện xuất nhập cảnh phải làm thủ tục tại cửa khẩu biên giới theo quy định.",
            score=0.015,
            rank=1,
            retrieval_method="HYBRID",
            effective_start="2015-01-15",
            effective_end=None,
            validity_status="VALID",
        ),
        RetrievalResult(
            chunk_id="CHK_BORDER_02",
            provision_id="PROV_34_ART_01",
            document_id="DOC_34_2023",
            doc_number="34/2023/NĐ-CP",
            doc_title="Nghị định sửa đổi Nghị định 112/2014/NĐ-CP",
            provision_title="Điều 1. Sửa đổi Điều 8",
            structural_path="34/2023/NĐ-CP > Điều 1",
            content="Sửa đổi, bổ sung quy chế kiểm soát phương tiện tự động tại cửa khẩu biên giới.",
            score=0.012,
            rank=2,
            retrieval_method="KG_EXPANSION",
            expansion_path="Seed: Điều 8 -->[AMENDS] Điều 1 (34/2023)",
            evidence_text="Điều 8 Nghị định 112/2014 được sửa đổi bởi Nghị định 34/2023",
            effective_start="2023-07-31",
            effective_end=None,
            validity_status="VALID",
        ),
    ]


def test_01_community_detector_initialization():
    detector = LegalCommunityDetector()
    comms = detector.get_communities()
    assert len(comms) > 0
    assert isinstance(comms[0], LegalCommunity)

    # Test community matching
    query = "Quy định quản lý cửa khẩu biên giới và người phương tiện xuất nhập cảnh"
    relevant = detector.find_relevant_communities(query=query, top_k=3)
    assert len(relevant) > 0
    assert relevant[0][1] > 0.0


def test_02_global_search_map_reduce(mock_cfg):
    detector = LegalCommunityDetector()
    engine = GlobalSearchEngine(community_detector=detector, default_config=mock_cfg)

    query = "Tổng quan các quy định về an ninh trật tự và bảo vệ biên giới?"
    resp = engine.search(
        query=query,
        reference_date="2024-01-01",
        top_communities=2,
        llm_config=mock_cfg,
    )

    assert isinstance(resp, GlobalSearchResponse)
    assert len(resp.community_insights) > 0
    assert resp.map_reduce_stats["communities_evaluated"] > 0
    assert "Macro Overview" in resp.answer_text or len(resp.answer_text) > 50
    assert resp.latency_ms >= 0


def test_03_drift_search_decomposition_and_traversal(mock_cfg, sample_seeds):
    detector = LegalCommunityDetector()
    engine = DRIFTSearchEngine(community_detector=detector, default_config=mock_cfg)

    query = "Quy chế điều hành và kiểm soát phương tiện tại cửa khẩu biên giới đất liền?"
    resp = engine.search(
        query=query,
        reference_date="2024-01-01",
        top_communities=2,
        min_relevance_score=35.0,
        max_hops=1,
        llm_config=mock_cfg,
        extra_test_arg="val",
    )

    assert isinstance(resp, DRIFTSearchResponse)
    assert len(resp.decomposed_queries) >= 2
    assert len(resp.trajectories) > 0
    assert len(resp.citations) > 0
    assert "DRIFT Trajectories" in resp.answer_text or len(resp.answer_text) > 50


def test_04_three_way_comparison_pipeline(mock_cfg, sample_seeds):
    generator = LegalAnswerGenerator(mock_cfg)
    detector = LegalCommunityDetector()
    global_eng = GlobalSearchEngine(community_detector=detector, default_config=mock_cfg)
    drift_eng = DRIFTSearchEngine(community_detector=detector, default_config=mock_cfg)

    query = "Quy chế kiểm soát người và phương tiện xuất nhập cảnh cửa khẩu biên giới?"
    res_3way = generator.generate_three_way_comparison(
        query=query,
        local_results=sample_seeds,
        reference_date="2024-01-01",
        top_communities=2,
        min_relevance_score=30.0,
        global_engine=global_eng,
        drift_engine=drift_eng,
        llm_config=mock_cfg,
        extra_arbitrary_param="safe_to_ignore",
    )

    assert isinstance(res_3way, ThreeWayComparisonResult)
    assert res_3way.local_search is not None
    assert res_3way.global_search is not None
    assert res_3way.drift_search is not None
    assert len(res_3way.comparison_matrix) >= 5
    assert res_3way.comparison_matrix[0]["Tiêu chí"] is not None


def test_05_expander_parent_article_resolution_and_no_child_flooding():
    expander = GraphExpander()

    # Seed là Khoản 4 của Điều 6
    sub_clause_seed = RetrievalResult(
        chunk_id="CHK_SUB_01",
        provision_id="DOC_33_2012_TT_BCA__ART_6__CL_4",
        document_id="DOC_33_2012_TT_BCA",
        doc_number="33/2012/TT-BCA",
        doc_title="Thông tư số 33/2012/TT-BCA",
        provision_title="Khoản 4",
        structural_path="33/2012/TT-BCA > Điều 6 > Khoản 4",
        content="Cán bộ chiến sĩ theo quy định...",
        score=0.015,
        rank=1,
        retrieval_method="HYBRID",
        effective_start="2012-07-27",
        effective_end=None,
        status="ACTIVE",
    )

    expanded = expander.expand(
        query="tiền tàu xe phương tiện đi lại khi nghỉ phép",
        seed_results=[sub_clause_seed],
        max_hops=1,
        max_expanded_nodes=5,
    )

    prov_ids = [r.provision_id for r in expanded]
    # Xác minh đã mở rộng ngược thành công lên Điều 6 cha
    assert "DOC_33_2012_TT_BCA__ART_6" in prov_ids

    # Xác minh nếu seed là một ARTICLE (Điều 6), expand() không đẻ thêm các Khoản con CL_1, CL_2...
    parent_article_seed = RetrievalResult(
        chunk_id="CHK_ART_06",
        provision_id="DOC_33_2012_TT_BCA__ART_6",
        document_id="DOC_33_2012_TT_BCA",
        doc_number="33/2012/TT-BCA",
        doc_title="Thông tư số 33/2012/TT-BCA",
        provision_title="Điều 6. Đối tượng thanh toán tiền tàu xe",
        structural_path="33/2012/TT-BCA > Điều 6",
        content="Nội dung toàn văn Điều 6 gồm các khoản...",
        score=0.020,
        rank=1,
        retrieval_method="HYBRID",
        effective_start="2012-07-27",
        effective_end=None,
        status="ACTIVE",
    )

    expanded_from_art = expander.expand(
        query="tiền tàu xe phương tiện đi lại khi nghỉ phép",
        seed_results=[parent_article_seed],
        max_hops=1,
        max_expanded_nodes=5,
    )

    expanded_prov_ids = [r.provision_id for r in expanded_from_art if r.retrieval_method == "KG_EXPANSION"]
    # Không được chứa các Khoản con của Điều 6
    assert "DOC_33_2012_TT_BCA__ART_6__CL_1" not in expanded_prov_ids
    assert "DOC_33_2012_TT_BCA__ART_6__CL_2" not in expanded_prov_ids

