import re
from typing import List, Dict, Any, Optional
from storage.models import ProvisionModel


class StructureParser:
    CLAUSE_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
    POINT_RE = re.compile(r"^\s*([a-zđ])\)\s+(.*)$", re.IGNORECASE)

    @classmethod
    def parse_article_to_tree(
        cls,
        doc_id: str,
        doc_path: str,
        article_num: str,
        article_title: str,
        content: str,
        mapc: Optional[str] = None,
        effective_start: Optional[str] = None,
        effective_end: Optional[str] = None,
        status: str = "CÓ_HIỆU_LỰC_THI_HÀNH",
        source_url: Optional[str] = None,
    ) -> List[ProvisionModel]:
        art_id = f"{doc_id}__ART_{article_num}"
        art_path = f"{doc_path} > {article_title}" if doc_path else article_title

        art_node = ProvisionModel(
            provision_id=art_id,
            document_id=doc_id,
            parent_id=None,
            level="ARTICLE",
            article_number=article_num,
            clause_number=None,
            point_letter=None,
            title=article_title,
            structural_path=art_path,
            content=content.strip(),
            mapc=mapc,
            effective_start=effective_start,
            effective_end=effective_end,
            status=status,
            source_url=source_url,
            is_external_stub=0,
            versions=[],
        )

        nodes = [art_node]
        lines = [line.rstrip() for line in content.split("\n")]
        
        current_clause: Optional[ProvisionModel] = None
        current_clause_lines: List[str] = []
        clause_points: List[ProvisionModel] = []
        current_point: Optional[ProvisionModel] = None
        current_point_lines: List[str] = []

        def flush_point():
            nonlocal current_point, current_point_lines
            if current_point:
                current_point.content = "\n".join(current_point_lines).strip()
                clause_points.append(current_point)
                current_point = None
                current_point_lines = []

        def flush_clause():
            nonlocal current_clause, current_clause_lines, clause_points
            flush_point()
            if current_clause:
                current_clause.content = "\n".join(current_clause_lines).strip()
                nodes.append(current_clause)
                nodes.extend(clause_points)
                current_clause = None
                current_clause_lines = []
                clause_points = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            clause_match = cls.CLAUSE_RE.match(stripped)
            if clause_match:
                flush_clause()
                cl_num = clause_match.group(1)
                cl_text = clause_match.group(2)
                cl_id = f"{art_id}__CL_{cl_num}"
                cl_path = f"{art_path} > Khoản {cl_num}"
                current_clause = ProvisionModel(
                    provision_id=cl_id,
                    document_id=doc_id,
                    parent_id=art_id,
                    level="CLAUSE",
                    article_number=article_num,
                    clause_number=cl_num,
                    point_letter=None,
                    title=f"Khoản {cl_num}",
                    structural_path=cl_path,
                    content=cl_text,
                    mapc=mapc,
                    effective_start=effective_start,
                    effective_end=effective_end,
                    status=status,
                    source_url=source_url,
                    is_external_stub=0,
                    versions=[],
                )
                current_clause_lines = [f"{cl_num}. {cl_text}"]
                continue

            point_match = cls.POINT_RE.match(stripped)
            if point_match and current_clause:
                flush_point()
                pt_letter = point_match.group(1).lower()
                pt_text = point_match.group(2)
                pt_id = f"{current_clause.provision_id}__PT_{pt_letter}"
                pt_path = f"{current_clause.structural_path} > Điểm {pt_letter}"
                current_point = ProvisionModel(
                    provision_id=pt_id,
                    document_id=doc_id,
                    parent_id=current_clause.provision_id,
                    level="POINT",
                    article_number=article_num,
                    clause_number=current_clause.clause_number,
                    point_letter=pt_letter,
                    title=f"Điểm {pt_letter}",
                    structural_path=pt_path,
                    content=pt_text,
                    mapc=mapc,
                    effective_start=effective_start,
                    effective_end=effective_end,
                    status=status,
                    source_url=source_url,
                    is_external_stub=0,
                    versions=[],
                )
                current_point_lines = [f"{pt_letter}) {pt_text}"]
                continue

            if current_point:
                current_point_lines.append(stripped)
            elif current_clause:
                current_clause_lines.append(stripped)

        flush_clause()
        return nodes
