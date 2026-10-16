import os
import sys
import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from storage.models import (
    LegalDocumentModel,
    ProvisionModel,
    ProvisionVersionModel,
    KGEdgeModel,
)
from ingestion.parser import StructureParser


class PhapdienConverter:
    META_TITLE_RE = re.compile(
        r"^Điều\s+(\d+[a-z]?)\s+(Luật|Bộ luật|Pháp lệnh|Nghị quyết|Nghị định|Quyết định|Thông tư(?:\s+liên\s+tịch)?)\s+số\s+(\S+)$",
        re.IGNORECASE,
    )

    FULL_GHICHU_RE = re.compile(
        r"Điều\s+\d+[a-z]?\s+(?:Luật|Bộ luật|Pháp lệnh|Nghị quyết|Nghị định|Quyết định|Thông tư(?:\s+liên\s+tịch)?)\s+số\s+\S+(?:\s+-\S+)?\s+(.+?)\s+ngày\s+(\d{2}/\d{2}/\d{4})\s+của\s+(.+?),\s*có hiệu lực thi hành kể từ ngày\s+(\d{2}/\d{2}/\d{4})",
        re.IGNORECASE | re.DOTALL,
    )

    LINK_PAT = re.compile(
        r"ItemID=(\d+)#(?:Chuong_([IVXLCDM\d]+)_)?(?:Muc_([IVXLCDM\d]+)_)?Dieu_(\d+[a-z]?)",
        re.IGNORECASE,
    )

    CLEAN_ARTICLE_TITLE_RE = re.compile(
        r"^Điều\s+[\d\.]+(?:LQ|PL|NĐ|QĐ|TT|TL|NQ|BL)[\d\.]*\s*(.*)$",
        re.IGNORECASE,
    )

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[str]:
        if not date_str:
            return None
        date_str = date_str.strip()
        try:
            return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        except Exception:
            return date_str

    @staticmethod
    def _doc_id_from_number(doc_number: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", doc_number).strip("_")
        return f"DOC_{clean}"

    def convert(
        self,
        raw_json_path: str = "mau_data.json",
        output_docs_path: str = "data/processed/vbqppl_documents.jsonl",
        output_edges_path: str = "data/processed/kg_relations.jsonl",
    ) -> Tuple[List[LegalDocumentModel], List[KGEdgeModel]]:
        print(f"Reading raw Phapdien data from {raw_json_path}...")
        with open(raw_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        docs_map: Dict[str, Dict[str, Any]] = {}
        all_raw_articles: List[Dict[str, Any]] = []

        for ch in raw_data:
            ch_demuc = ch.get("Tên đề mục", "")
            ch_title = ch.get("Tên chương", "")
            for art in ch.get("Các điều", []):
                art["_phapdien_demuc"] = ch_demuc
                art["_phapdien_chuong"] = ch_title
                all_raw_articles.append(art)

        print(f"Loaded {len(all_raw_articles)} raw articles across {len(raw_data)} chapters.")

        # First pass: identify all primary documents
        for art in all_raw_articles:
            m0 = art.get("Ghi chú", {}).get("metadata", [{}])[0]
            td = m0.get("Tên điều", "").strip()
            mm = self.META_TITLE_RE.match(td)
            if not mm:
                continue

            art_num, doc_type, doc_number = mm.groups()
            if doc_number == "11/2014/TTLT-BQP-BNV":
                doc_number = "11/2014/TTLT-BQP-BNV-BLĐTBXH-BTC"

            doc_id = self._doc_id_from_number(doc_number)
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "doc_number": doc_number,
                    "doc_type": doc_type,
                    "title": f"{doc_type} số {doc_number}",
                    "issuer": "Cơ quan nhà nước",
                    "issue_date": None,
                    "effective_start": None,
                    "effective_end": None,
                    "status": "ACTIVE",
                    "source_url": m0.get("Link"),
                    "articles": [],
                }

            gc_text = art.get("Ghi chú", {}).get("Ghi chú", "").strip()
            fm = self.FULL_GHICHU_RE.search(gc_text)
            if fm:
                title, issue_d, issuer, eff_d = fm.groups()
                d_entry = docs_map[doc_id]
                if d_entry["title"] == f"{doc_type} số {doc_number}":
                    d_entry["title"] = f"{doc_type} {title.strip()}"
                d_entry["issuer"] = issuer.strip()
                d_entry["issue_date"] = self._parse_date(issue_d)
                d_entry["effective_start"] = self._parse_date(eff_d)

            eff_from_m0 = self._parse_date(m0.get("Hiệu lực"))
            if eff_from_m0 and not docs_map[doc_id]["effective_start"]:
                docs_map[doc_id]["effective_start"] = eff_from_m0

            docs_map[doc_id]["articles"].append((art_num, art))

        print(f"Identified {len(docs_map)} primary legal documents.")

        # Second pass: Process articles and create Provision trees
        final_docs: List[LegalDocumentModel] = []
        kg_edges: List[KGEdgeModel] = []
        mapc_to_prov_id: Dict[str, str] = {}
        amending_docs_map: Dict[str, Dict[str, Any]] = {}

        for doc_id, d_data in docs_map.items():
            doc_provisions: List[ProvisionModel] = []

            for art_num, art in d_data["articles"]:
                raw_title = art.get("Tên điều", "").strip()
                clean_title_match = self.CLEAN_ARTICLE_TITLE_RE.match(raw_title)
                clean_suffix = clean_title_match.group(1).strip() if clean_title_match else raw_title
                
                content = art.get("Nội dung", "").strip()
                # Fix edge case of empty content (e.g. QĐ 45/2013/QĐ-TTg)
                if not content:
                    content = clean_suffix
                    clean_suffix = "Quy định"

                art_title = f"Điều {art_num}. {clean_suffix}".strip(". ")

                # Find Chapter from link
                m0 = art.get("Ghi chú", {}).get("metadata", [{}])[0]
                link = m0.get("Link", "")
                link_match = self.LINK_PAT.search(link)
                chuong_rom = link_match.group(2) if link_match and link_match.group(2) else None
                muc_rom = link_match.group(3) if link_match and link_match.group(3) else None

                doc_path_parts = [d_data["doc_number"]]
                if chuong_rom:
                    doc_path_parts.append(f"Chương {chuong_rom}")
                if muc_rom:
                    doc_path_parts.append(f"Mục {muc_rom}")
                doc_path = " > ".join(doc_path_parts)

                mapc = art.get("Điều MAPC")
                eff_start = self._parse_date(m0.get("Hiệu lực")) or d_data["effective_start"]
                eff_status = m0.get("Thi hành", "CÓ_HIỆU_LỰC_THI_HÀNH")

                # Parse into tree
                prov_tree = StructureParser.parse_article_to_tree(
                    doc_id=doc_id,
                    doc_path=doc_path,
                    article_num=art_num,
                    article_title=art_title,
                    content=content,
                    mapc=mapc,
                    effective_start=eff_start,
                    effective_end=None,
                    status=eff_status,
                    source_url=link,
                )

                art_node = prov_tree[0]
                if mapc:
                    mapc_to_prov_id[mapc] = art_node.provision_id

                # CONTAINS edge from doc to article
                kg_edges.append(KGEdgeModel(
                    edge_id=f"EDGE_CONTAINS_{doc_id}_{art_node.provision_id}",
                    source_id=doc_id,
                    target_id=art_node.provision_id,
                    relation_type="CONTAINS",
                    evidence_text=f"{d_data['title']} bao gồm {art_title}",
                    source_url=link,
                    verified=1,
                ))

                # CONTAINS edges down the tree
                for child_prov in prov_tree[1:]:
                    kg_edges.append(KGEdgeModel(
                        edge_id=f"EDGE_CONTAINS_{child_prov.parent_id}_{child_prov.provision_id}",
                        source_id=child_prov.parent_id,
                        target_id=child_prov.provision_id,
                        relation_type="CONTAINS",
                        evidence_text=f"{child_prov.title} thuộc {child_prov.structural_path}",
                        source_url=link,
                        verified=1,
                    ))

                # Handle amending metadata (35 articles)
                metas = art.get("Ghi chú", {}).get("metadata", [])
                if len(metas) > 1:
                    for idx, amend_meta in enumerate(metas[1:], start=1):
                        amend_status = amend_meta.get("Thi hành", "")
                        amend_title = amend_meta.get("Tên điều", "").strip()
                        amend_eff = self._parse_date(amend_meta.get("Hiệu lực"))
                        amend_link = amend_meta.get("Link", "")

                        # Extract amending document number
                        amend_match = self.META_TITLE_RE.match(amend_title)
                        if amend_match:
                            am_art_num, am_type, am_num = amend_match.groups()
                            am_doc_id = self._doc_id_from_number(am_num)
                            am_prov_id = f"{am_doc_id}__ART_{am_art_num}"

                            if am_doc_id not in amending_docs_map:
                                amending_docs_map[am_doc_id] = {
                                    "document_id": am_doc_id,
                                    "doc_number": am_num,
                                    "doc_type": am_type,
                                    "title": f"{am_type} số {am_num} (Văn bản sửa đổi, bổ sung)",
                                    "issuer": "Cơ quan có thẩm quyền",
                                    "issue_date": None,
                                    "effective_start": amend_eff,
                                    "status": "AMENDING_DOC",
                                    "source_url": amend_link,
                                    "provisions": {},
                                }

                            if am_prov_id not in amending_docs_map[am_doc_id]["provisions"]:
                                amending_docs_map[am_doc_id]["provisions"][am_prov_id] = ProvisionModel(
                                    provision_id=am_prov_id,
                                    document_id=am_doc_id,
                                    level="ARTICLE",
                                    article_number=am_art_num,
                                    title=f"Điều {am_art_num}. Sửa đổi, bổ sung theo {am_type} số {am_num}",
                                    structural_path=f"{am_num} > Điều {am_art_num}",
                                    content=f"Quy định sửa đổi, bổ sung đối với {art_title} ({d_data['title']}) có hiệu lực từ ngày {amend_eff}.",
                                    effective_start=amend_eff,
                                    status="ACTIVE_AMENDED",
                                    source_url=amend_link,
                                )

                            # Versions v1 and v2
                            end_of_v1 = None
                            if amend_eff:
                                try:
                                    dt = datetime.strptime(amend_eff, "%Y-%m-%d") - timedelta(days=1)
                                    end_of_v1 = dt.strftime("%Y-%m-%d")
                                except Exception:
                                    end_of_v1 = amend_eff

                            art_node.versions.append(ProvisionVersionModel(
                                version_id=f"{art_node.provision_id}__V1",
                                provision_id=art_node.provision_id,
                                version_num=1,
                                effective_start=eff_start,
                                effective_end=end_of_v1,
                                status="SUPERSEDED_BY_AMENDMENT",
                                amending_provision_id=am_prov_id,
                                content=content,
                                version_note=f"Nội dung nguyên bản áp dụng từ {eff_start} đến {end_of_v1} trước khi được sửa đổi bởi {amend_title}.",
                            ))

                            art_node.versions.append(ProvisionVersionModel(
                                version_id=f"{art_node.provision_id}__V2",
                                provision_id=art_node.provision_id,
                                version_num=2,
                                effective_start=amend_eff,
                                effective_end=None,
                                status="ACTIVE_AMENDED",
                                amending_provision_id=am_prov_id,
                                content=content,
                                version_note=f"Nội dung đã hợp nhất áp dụng từ ngày {amend_eff} theo {amend_title}.",
                            ))

                            rel_type = "AMENDS"
                            if "BỔ_SUNG" in amend_status:
                                rel_type = "SUPPLEMENTS"
                            elif "BÃI_BỎ" in amend_status:
                                rel_type = "REPEALS"

                            kg_edges.append(KGEdgeModel(
                                edge_id=f"EDGE_{rel_type}_{am_prov_id}_{art_node.provision_id}",
                                source_id=am_prov_id,
                                target_id=art_node.provision_id,
                                relation_type=rel_type,
                                evidence_text=art.get("Ghi chú", {}).get("Ghi chú", ""),
                                source_url=amend_link,
                                effective_date=amend_eff,
                                detection_method="OFFICIAL_METADATA",
                                verified=1,
                            ))

                doc_provisions.extend(prov_tree)

            doc_hash = hashlib.sha256(
                "".join(p.content for p in doc_provisions).encode("utf-8")
            ).hexdigest()

            doc_model = LegalDocumentModel(
                document_id=doc_id,
                doc_number=d_data["doc_number"],
                title=d_data["title"],
                doc_type=d_data["doc_type"],
                issuer=d_data["issuer"],
                issue_date=d_data["issue_date"],
                effective_start=d_data["effective_start"],
                effective_end=d_data["effective_end"],
                status=d_data["status"],
                source_url=d_data["source_url"],
                content_hash=doc_hash,
                provisions=doc_provisions,
            )
            final_docs.append(doc_model)

        # Append amending documents into final_docs
        for am_doc_id, am_info in amending_docs_map.items():
            am_provisions = list(am_info["provisions"].values())
            am_hash = hashlib.sha256("".join(p.content for p in am_provisions).encode("utf-8")).hexdigest()
            final_docs.append(LegalDocumentModel(
                document_id=am_doc_id,
                doc_number=am_info["doc_number"],
                title=am_info["title"],
                doc_type=am_info["doc_type"],
                issuer=am_info["issuer"],
                issue_date=am_info["issue_date"],
                effective_start=am_info["effective_start"],
                effective_end=None,
                status=am_info["status"],
                source_url=am_info["source_url"],
                content_hash=am_hash,
                provisions=am_provisions,
            ))

        print(f"Total documents prepared: {len(final_docs)} (including {len(amending_docs_map)} amending documents).")

        # Third pass: Process 1071 "Chỉ dẫn" links for REFERENCES edges
        ref_count = 0
        for art in all_raw_articles:
            src_mapc = art.get("Điều MAPC")
            if not src_mapc or src_mapc not in mapc_to_prov_id:
                continue
            src_prov_id = mapc_to_prov_id[src_mapc]

            for chidan in art.get("Chỉ dẫn", []):
                tgt_mapc = chidan.get("ID Điều liên quan")
                tgt_title = chidan.get("Điều liên quan", "")
                if tgt_mapc and tgt_mapc in mapc_to_prov_id:
                    tgt_prov_id = mapc_to_prov_id[tgt_mapc]
                    kg_edges.append(KGEdgeModel(
                        edge_id=f"EDGE_REF_{src_prov_id}_{tgt_prov_id}",
                        source_id=src_prov_id,
                        target_id=tgt_prov_id,
                        relation_type="REFERENCES",
                        evidence_text=f"Chỉ dẫn chính thức từ Pháp điển: {tgt_title}",
                        source_url=chidan.get("Link"),
                        detection_method="OFFICIAL_CHIDAN",
                        verified=1,
                    ))
                    ref_count += 1

        print(f"Created {ref_count} verified internal REFERENCES edges from Chỉ dẫn.")

        # Save to JSONL
        os.makedirs(os.path.dirname(os.path.abspath(output_docs_path)), exist_ok=True)
        print(f"Writing processed documents to {output_docs_path}...")
        with open(output_docs_path, "w", encoding="utf-8") as f:
            for d in final_docs:
                f.write(d.model_dump_json() + "\n")

        print(f"Writing {len(kg_edges)} KG edges to {output_edges_path}...")
        with open(output_edges_path, "w", encoding="utf-8") as f:
            for e in kg_edges:
                f.write(e.model_dump_json() + "\n")

        print("Conversion completed successfully!")
        return final_docs, kg_edges


if __name__ == "__main__":
    converter = PhapdienConverter()
    converter.convert()
