import os
import json
from typing import List, Optional
from storage.models import BenchmarkItem
from storage.db import DatabaseManager


class BenchmarkBuilder:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def build_default_benchmark(self, output_path: str = "data/processed/benchmark_qa.jsonl") -> List[BenchmarkItem]:
        items: List[BenchmarkItem] = []

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Group 1: direct_article (12 questions)
            cursor.execute("""
                SELECT p.provision_id, p.document_id, p.title, p.content, d.doc_number, d.title as doc_title
                FROM provisions p
                JOIN documents d ON p.document_id = d.document_id
                WHERE p.level = 'ARTICLE' AND p.title LIKE 'Điều %' AND length(p.content) > 60
                LIMIT 12
            """)
            rows = cursor.fetchall()
            for idx, r in enumerate(rows, 1):
                clean_t = r["title"].split(".", 1)[-1].strip()
                q_text = f"Quy định về {clean_t} theo {r['doc_title']} ({r['doc_number']}) được nêu như thế nào?"
                items.append(BenchmarkItem(
                    question_id=f"G1_Q{idx:02d}",
                    group_id="direct_article",
                    group_name="1. Truy xuất trực tiếp Điều/Khoản",
                    question_text=q_text,
                    relevant_provision_ids=[r["provision_id"]],
                    relevant_document_ids=[r["document_id"]],
                    rationale=f"Hỏi trực tiếp về {r['title']} của {r['doc_number']}.",
                ))

            # Group 2: paraphrased_semantic (12 questions)
            cursor.execute("""
                SELECT p.provision_id, p.document_id, p.title, p.content, d.doc_number, d.title as doc_title
                FROM provisions p
                JOIN documents d ON p.document_id = d.document_id
                WHERE p.level = 'ARTICLE' AND length(p.content) > 100
                LIMIT 12 OFFSET 20
            """)
            rows = cursor.fetchall()
            for idx, r in enumerate(rows, 1):
                words = r["content"].split()
                snippet = " ".join(words[:15]).strip(" ,.;:")
                q_text = f"Theo quy định hiện hành thì việc {snippet.lower()} được pháp luật xử lý và quy định ra sao?"
                items.append(BenchmarkItem(
                    question_id=f"G2_Q{idx:02d}",
                    group_id="paraphrased_semantic",
                    group_name="2. Diễn đạt lại ngữ nghĩa",
                    question_text=q_text,
                    relevant_provision_ids=[r["provision_id"]],
                    relevant_document_ids=[r["document_id"]],
                    rationale="Câu hỏi diễn đạt bằng ngôn ngữ đời thường dựa trên nội dung điều khoản.",
                ))

            # Group 3: cross_reference_kg (12 questions)
            cursor.execute("""
                SELECT e.source_id, e.target_id, e.relation_type, e.evidence_text,
                       ps.title as src_title, pt.title as tgt_title,
                       ps.document_id as src_doc, pt.document_id as tgt_doc
                FROM kg_edges e
                JOIN provisions ps ON e.source_id = ps.provision_id
                JOIN provisions pt ON e.target_id = pt.provision_id
                WHERE e.relation_type IN ('REFERENCES', 'DETAILS') AND e.verified = 1
                LIMIT 12
            """)
            rows = cursor.fetchall()
            for idx, r in enumerate(rows, 1):
                q_text = f"Căn cứ pháp lý nào quy định và có mối quan hệ dẫn chiếu/hướng dẫn liên quan giữa {r['src_title']} và {r['tgt_title']}?"
                items.append(BenchmarkItem(
                    question_id=f"G3_Q{idx:02d}",
                    group_id="cross_reference_kg",
                    group_name="3. Theo dấu dẫn chiếu Knowledge Graph",
                    question_text=q_text,
                    relevant_provision_ids=[r["source_id"], r["target_id"]],
                    relevant_document_ids=[r["src_doc"], r["tgt_doc"]],
                    expected_relations=[r["relation_type"]],
                    expected_kg_targets=[r["target_id"]],
                    rationale=f"Dẫn chiếu {r['relation_type']}: {r['evidence_text']}",
                ))

            # Group 4: multi_provision (12 questions)
            cursor.execute("""
                SELECT d.document_id, d.doc_number, d.title as doc_title,
                       GROUP_CONCAT(p.provision_id, ',') as prov_ids,
                       COUNT(p.provision_id) as cnt
                FROM documents d
                JOIN provisions p ON d.document_id = p.document_id
                WHERE p.level = 'ARTICLE'
                GROUP BY d.document_id
                HAVING cnt >= 3
                LIMIT 12
            """)
            rows = cursor.fetchall()
            for idx, r in enumerate(rows, 1):
                p_list = r["prov_ids"].split(",")[:3]
                q_text = f"Tổng hợp các căn cứ pháp lý cơ bản về nguyên tắc, phạm vi và trách nhiệm trong {r['doc_title']} ({r['doc_number']})?"
                items.append(BenchmarkItem(
                    question_id=f"G4_Q{idx:02d}",
                    group_id="multi_provision",
                    group_name="4. Nhiều căn cứ pháp lý đồng thời",
                    question_text=q_text,
                    relevant_provision_ids=p_list,
                    relevant_document_ids=[r["document_id"]],
                    rationale="Câu hỏi yêu cầu tổng hợp 3 điều khoản căn cứ ban đầu của văn bản.",
                ))

            # Group 5: temporal_amendment (12 questions)
            cursor.execute("""
                SELECT v.provision_id, v.effective_start, v.effective_end, v.status,
                       v.amending_provision_id, p.title as prov_title, p.document_id
                FROM provision_versions v
                JOIN provisions p ON v.provision_id = p.provision_id
                WHERE v.version_num = 2 AND v.effective_start IS NOT NULL
                LIMIT 12
            """)
            rows = cursor.fetchall()
            for idx, r in enumerate(rows, 1):
                eff_date = r["effective_start"]
                q_text = f"Quy định tại {r['prov_title']} được áp dụng như thế nào tại ngày tham chiếu {eff_date} khi đã có văn bản sửa đổi bổ sung?"
                items.append(BenchmarkItem(
                    question_id=f"G5_Q{idx:02d}",
                    group_id="temporal_amendment",
                    group_name="5. Lịch sử sửa đổi & Hiệu lực theo thời gian",
                    question_text=q_text,
                    relevant_provision_ids=[r["provision_id"]],
                    relevant_document_ids=[r["document_id"]],
                    reference_date=eff_date,
                    expected_relations=["AMENDS", "SUPPLEMENTS"],
                    expected_kg_targets=[r["amending_provision_id"]] if r["amending_provision_id"] else [],
                    rationale=f"Đánh giá hiệu lực tại ngày tham chiếu {eff_date} của điều khoản có phiên bản sửa đổi.",
                ))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(it.model_dump_json() + "\n")

        print(f"Generated and saved {len(items)} benchmark Q&A pairs to {output_path}.")
        return items


if __name__ == "__main__":
    b = BenchmarkBuilder()
    b.build_default_benchmark()
