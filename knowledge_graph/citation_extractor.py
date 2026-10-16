import re
from typing import List, Dict, Tuple, Optional
from storage.models import KGEdgeModel, ProvisionModel


class CitationExtractor:
    IN_TEXT_REF_RE = re.compile(
        r"(?:theo quy định tại|quy định tại|tại)\s+(?:khoản\s+(\d+)\s+)?điều\s+(\d+[a-z]?)(?:\s+(?:của|thuộc)\s+(luật|bộ luật|nghị định|thông tư)\s+(?:này|số\s+(\S+)))?",
        re.IGNORECASE,
    )

    DETAILS_RE = re.compile(
        r"(?:quy định chi tiết|hướng dẫn thi hành|hướng dẫn thực hiện)(?:\s+thi hành)?\s+(?:một số điều của\s+)?(luật|pháp lệnh|nghị định)\s+số\s+(\S+)",
        re.IGNORECASE,
    )

    @classmethod
    def extract_details_relations(
        cls,
        doc_id: str,
        doc_number: str,
        first_article_text: str,
        ghichu_text: str,
        all_doc_numbers: Dict[str, str],  # doc_number -> doc_id
    ) -> List[KGEdgeModel]:
        edges: List[KGEdgeModel] = []
        combined_text = f"{first_article_text}\n{ghichu_text}"
        matches = cls.DETAILS_RE.findall(combined_text)

        for target_type, target_num in matches:
            target_num_clean = target_num.strip(" ,.;:")
            target_doc_id = all_doc_numbers.get(target_num_clean)

            if target_doc_id and target_doc_id != doc_id:
                edges.append(KGEdgeModel(
                    edge_id=f"EDGE_DETAILS_{doc_id}_{target_doc_id}",
                    source_id=doc_id,
                    target_id=target_doc_id,
                    relation_type="DETAILS",
                    evidence_text=f"Quy định chi tiết / hướng dẫn thi hành {target_type} số {target_num_clean}",
                    detection_method="REGEX_CITATION",
                    verified=1,
                ))

        return edges

    @classmethod
    def extract_in_text_references(
        cls,
        current_provision: ProvisionModel,
        article_lookup: Dict[str, str],  # article_number -> provision_id (within same document)
    ) -> List[KGEdgeModel]:
        edges: List[KGEdgeModel] = []
        matches = cls.IN_TEXT_REF_RE.findall(current_provision.content)

        for cl_num, art_num, ext_type, ext_num in matches:
            # Internal reference within same document
            if not ext_num:
                target_prov_id = article_lookup.get(art_num)
                if target_prov_id and target_prov_id != current_provision.provision_id:
                    edges.append(KGEdgeModel(
                        edge_id=f"EDGE_IN_TEXT_REF_{current_provision.provision_id}_{target_prov_id}",
                        source_id=current_provision.provision_id,
                        target_id=target_prov_id,
                        relation_type="REFERENCES",
                        evidence_text=f"Dẫn chiếu trong văn bản tới Điều {art_num}",
                        detection_method="REGEX_CITATION",
                        verified=1,
                    ))

        return edges
