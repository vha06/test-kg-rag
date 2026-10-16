from typing import List, Tuple
from storage.models import RetrievalResult
from generation.models import CitationItem


class LegalContextBuilder:
    """
    Đóng gói danh sách RetrievalResult thành Context Bundle có cấu trúc chuẩn hóa,
    đánh số thứ tự nguồn [NGUỒN #1], [NGUỒN #2]... và trích xuất cảnh báo hiệu lực thời gian.
    """

    @staticmethod
    def build_context(
        results: List[RetrievalResult],
        reference_date: str = "2024-01-01",
    ) -> Tuple[str, List[CitationItem], List[str]]:
        if not results:
            return "", [], []

        citations: List[CitationItem] = []
        temporal_warnings: List[str] = []
        context_blocks: List[str] = []

        for idx, res in enumerate(results, start=1):
            # 1. Map to CitationItem
            citation = CitationItem(
                ref_index=idx,
                provision_id=res.provision_id,
                doc_number=res.doc_number,
                doc_title=res.doc_title,
                provision_title=res.provision_title,
                structural_path=res.structural_path,
                retrieval_method=res.retrieval_method,
                expansion_path=res.expansion_path,
                evidence_text=res.evidence_text,
                validity_status=res.validity_status,
                effective_start=res.effective_start,
                effective_end=res.effective_end,
                is_cited_in_answer=False,
            )
            citations.append(citation)

            # 2. Extract Temporal Warnings
            if res.validity_status == "INVALID_EXPIRED":
                temporal_warnings.append(
                    f"⚠️ [Nguồn #{idx}] {res.doc_number} - {res.provision_title} đã HẾT HIỆU LỰC tại ngày tham chiếu {reference_date} (kết thúc ngày {res.effective_end})."
                )
            elif res.validity_status == "INVALID_NOT_YET_EFFECTIVE":
                temporal_warnings.append(
                    f"⚠️ [Nguồn #{idx}] {res.doc_number} - {res.provision_title} CHƯA CÓ HIỆU LỰC tại ngày tham chiếu {reference_date} (bắt đầu ngày {res.effective_start})."
                )
            elif res.validity_status == "VALID_PRIOR_TO_AMENDMENT":
                temporal_warnings.append(
                    f"ℹ️ [Nguồn #{idx}] {res.doc_number} - {res.provision_title} áp dụng theo phiên bản gốc trước thời điểm có văn bản sửa đổi bổ sung."
                )

            if res.expansion_path and ("AMENDS" in res.expansion_path or "REPEALS" in res.expansion_path):
                temporal_warnings.append(
                    f"ℹ️ [Nguồn #{idx}] Có quan hệ sửa đổi/bãi bỏ: {res.expansion_path} (Căn cứ: {res.evidence_text or 'Không có'})"
                )

            # 3. Format Context Block
            if res.expansion_path and res.retrieval_method != "KG_EXPANSION":
                method_desc = f"Truy xuất trực tiếp (Seed) + Có liên kết Knowledge Graph"
            elif res.retrieval_method == "KG_EXPANSION":
                method_desc = "Mở rộng qua Knowledge Graph (KG Expansion)"
            else:
                method_desc = "Truy xuất trực tiếp (Seed Retrieval)"

            val_desc = "ĐANG CÓ HIỆU LỰC"
            if res.validity_status == "INVALID_EXPIRED":
                val_desc = f"ĐÃ HẾT HIỆU LỰC (kết thúc {res.effective_end})"
            elif res.validity_status == "INVALID_NOT_YET_EFFECTIVE":
                val_desc = f"CHƯA CÓ HIỆU LỰC (bắt đầu {res.effective_start})"
            elif res.validity_status == "VALID_PRIOR_TO_AMENDMENT":
                val_desc = "CÓ HIỆU LỰC THEO PHIÊN BẢN GỐC"

            block_lines = [
                f"====================================================================",
                f"[NGUỒN #{idx}]",
                f"- Văn bản: {res.doc_title} (Số hiệu: {res.doc_number})",
                f"- Điều khoản: {res.provision_title}",
                f"- Đường dẫn phân cấp: {res.structural_path}",
                f"- Nguồn gốc trích xuất: {method_desc}",
                f"- Hiệu lực tại ngày tham chiếu ({reference_date}): {val_desc} (Khoảng hiệu lực: {res.effective_start or 'N/A'} -> {res.effective_end or 'Hiện hành'})",
            ]

            if res.expansion_path:
                block_lines.append(f"- Đường dẫn liên kết KG: {res.expansion_path}")
            if res.evidence_text:
                block_lines.append(f"- Căn cứ pháp lý liên kết: {res.evidence_text}")

            block_lines.append("- Nội dung quy định:")
            block_lines.append(res.content.strip())

            context_blocks.append("\n".join(block_lines))

        formatted_context = "\n\n".join(context_blocks)
        # Remove duplicate temporal warnings if any
        unique_warnings = list(dict.fromkeys(temporal_warnings))
        return formatted_context, citations, unique_warnings
