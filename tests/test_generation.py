import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.models import RetrievalResult
from generation.models import LLMConfig, CitationItem, LegalAnswerResponse, ABComparisonResult
from generation.context_builder import LegalContextBuilder
from generation.prompts import LegalPrompts
from generation.llm_client import MultiProviderLLMClient
from generation.answer_generator import LegalAnswerGenerator


@pytest.fixture
def sample_retrieval_results():
    return [
        RetrievalResult(
            chunk_id="CHK_01",
            provision_id="PROV_34_ART_1",
            document_id="DOC_34_2023",
            doc_number="34/2023/NĐ-CP",
            doc_title="Nghị định sửa đổi",
            provision_title="Điều 1. Sửa đổi, bổ sung",
            structural_path="34/2023/NĐ-CP > Điều 1",
            content="Điều 1. Sửa đổi một số điều của Nghị định số 112/2014/NĐ-CP...",
            score=0.016,
            rank=1,
            retrieval_method="HYBRID",
            effective_start="2023-07-31",
            effective_end=None,
            validity_status="VALID",
        ),
        RetrievalResult(
            chunk_id="CHK_02",
            provision_id="PROV_112_ART_18",
            document_id="DOC_112_2014",
            doc_number="112/2014/NĐ-CP",
            doc_title="Nghị định Quản lý cửa khẩu",
            provision_title="Điều 18. Quy hoạch cửa khẩu",
            structural_path="112/2014/NĐ-CP > Điều 18",
            content="Quy hoạch cửa khẩu biên giới đất liền phải tuân thủ điều kiện an ninh...",
            score=0.012,
            rank=2,
            retrieval_method="KG_EXPANSION",
            expansion_path="Seed: Điều 1 (34/2023) -->[AMENDS] Điều 18 (112/2014)",
            evidence_text="Điều 18 Nghị định số 112/2014/NĐ-CP được sửa đổi bởi Điều 1 Nghị định số 34/2023/NĐ-CP",
            effective_start="2015-01-15",
            effective_end=None,
            validity_status="VALID",
        ),
    ]


def test_01_context_builder_formatting_and_citations(sample_retrieval_results):
    context, citations, warnings = LegalContextBuilder.build_context(
        results=sample_retrieval_results,
        reference_date="2024-01-01",
    )

    # Đảm bảo có đánh số thứ tự [NGUỒN #1] và [NGUỒN #2]
    assert "[NGUỒN #1]" in context
    assert "[NGUỒN #2]" in context
    assert "34/2023/NĐ-CP" in context
    assert "112/2014/NĐ-CP" in context

    # Đảm bảo đường dẫn KG và chứng cứ pháp lý được tiêm vào context
    assert "Seed: Điều 1 (34/2023) -->[AMENDS] Điều 18 (112/2014)" in context
    assert "được sửa đổi bởi Điều 1 Nghị định số 34/2023/NĐ-CP" in context

    # Kiểm tra citations list
    assert len(citations) == 2
    assert citations[0].ref_index == 1
    assert citations[1].ref_index == 2
    assert citations[1].retrieval_method == "KG_EXPANSION"

    # Kiểm tra trích xuất cảnh báo hiệu lực sửa đổi
    assert any("AMENDS" in w for w in warnings)


def test_02_refusal_on_empty_results():
    generator = LegalAnswerGenerator(LLMConfig(provider="mock"))
    resp = generator.generate_answer(
        query="Hành vi vi phạm nào bị phạt?",
        results=[],
        reference_date="2024-01-01",
    )
    assert resp.has_sufficient_evidence is False
    assert "[KHÔNG_ĐỦ_CĂN_CỨ]" in resp.answer_text
    assert len(resp.citations) == 0


def test_03_citation_verification(sample_retrieval_results):
    generator = LegalAnswerGenerator(LLMConfig(provider="mock"))
    resp = generator.generate_answer(
        query="Quy định tại Nghị định 112/2014 được sửa đổi ra sao?",
        results=sample_retrieval_results,
        reference_date="2024-01-01",
    )
    assert resp.has_sufficient_evidence is True
    assert len(resp.citations) == 2
    # Mock generator sinh câu trả lời có chứa [#1] và [#2]
    assert resp.citations[0].is_cited_in_answer is True
    assert resp.citations[1].is_cited_in_answer is True


def test_04_ab_comparison_pipeline(sample_retrieval_results):
    generator = LegalAnswerGenerator(LLMConfig(provider="mock"))
    seed_only = [sample_retrieval_results[0]]
    expanded = sample_retrieval_results

    ab_res = generator.generate_ab_comparison(
        query="Nghị định 34 sửa đổi Nghị định 112 ra sao?",
        seed_results=seed_only,
        expanded_results=expanded,
        reference_date="2024-01-01",
    )

    assert isinstance(ab_res, ABComparisonResult)
    assert len(ab_res.without_kg.citations) == 1
    assert len(ab_res.with_kg.citations) == 2


def test_05_multi_provider_llm_client_mock():
    client = MultiProviderLLMClient(LLMConfig(provider="mock"))
    ans = client.generate(
        system_prompt="Bạn là luật sư",
        user_prompt="Hỏi về an ninh",
    )
    assert len(ans) > 50
    assert "[#1]" in ans
