import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from networkx.algorithms.community import louvain_communities

from storage.db import DatabaseManager
from storage.models import RetrievalResult
from generation.models import LegalCommunity
from knowledge_graph.builder import KnowledgeGraphBuilder


class LegalCommunityDetector:
    """
    Phát hiện và quản lý các Cụm cộng đồng tri thức pháp lý (Hierarchical Communities)
    từ Knowledge Graph, phục vụ cơ chế Global Search (Map-Reduce) và DRIFT Search.
    """

    CACHE_FILE = "artifacts/legal_communities.json"

    def __init__(
        self,
        kg_builder: Optional[KnowledgeGraphBuilder] = None,
        db_manager: Optional[DatabaseManager] = None,
        cache_file: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        self.db = db_manager or DatabaseManager()
        self.kg_builder = kg_builder or KnowledgeGraphBuilder(self.db)
        self.cache_file = cache_file or self.CACHE_FILE
        self.communities: List[LegalCommunity] = []

        if not force_rebuild and os.path.exists(self.cache_file):
            try:
                self.load_cache()
            except Exception as e:
                self.build_communities()
        else:
            self.build_communities()

    def build_communities(self) -> List[LegalCommunity]:
        """Phân cụm đồ thị tri thức pháp lý dựa trên thuật toán Louvain và tạo tóm tắt cộng đồng."""
        graph = self.kg_builder.build_graph(verified_only=True)

        doc_nodes = {n: d for n, d in graph.nodes(data=True) if d.get("node_type") == "DOCUMENT"}
        prov_nodes = {n: d for n, d in graph.nodes(data=True) if d.get("node_type") == "PROVISION"}
        prov_to_doc = {n: d.get("document_id") for n, d in prov_nodes.items()}

        # Xây dựng đồ thị liên kết giữa các văn bản QPPL
        doc_g = nx.Graph()
        for doc_id, data in doc_nodes.items():
            doc_g.add_node(doc_id, **data)

        edge_counter: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for u, v, k, d in graph.edges(keys=True, data=True):
            rel = d.get("relation_type", "")
            if rel == "CONTAINS":
                continue

            doc_u = u if u in doc_nodes else prov_to_doc.get(u)
            doc_v = v if v in doc_nodes else prov_to_doc.get(v)

            if doc_u and doc_v and doc_u != doc_v and doc_u in doc_nodes and doc_v in doc_nodes:
                pair = tuple(sorted([doc_u, doc_v]))
                if pair not in edge_counter:
                    edge_counter[pair] = {"weight": 0, "rels": {}}
                edge_counter[pair]["weight"] += 1
                edge_counter[pair]["rels"][rel] = edge_counter[pair]["rels"].get(rel, 0) + 1

        for (u, v), info in edge_counter.items():
            doc_g.add_edge(u, v, weight=info["weight"], rels=info["rels"])

        # Chạy thuật toán phát hiện cộng đồng Louvain
        raw_communities = louvain_communities(doc_g, weight="weight", seed=42)

        community_objects: List[LegalCommunity] = []

        # Sắp xếp các cụm theo số lượng văn bản và điều khoản
        for idx, doc_set in enumerate(raw_communities, start=1):
            docs = [doc_nodes[d] for d in doc_set if d in doc_nodes]
            if not docs:
                continue

            # Xác định văn bản trọng tâm (Anchor Document)
            # Ưu tiên: Luật > Pháp lệnh > Nghị quyết > Nghị định > Quyết định > Thông tư
            type_priority = {
                "Luật": 1,
                "Pháp lệnh": 2,
                "Nghị quyết": 3,
                "Nghị định": 4,
                "Quyết định": 5,
                "Thông tư": 6,
            }
            sorted_docs = sorted(
                docs,
                key=lambda d: (
                    type_priority.get(d.get("doc_type", ""), 99),
                    d.get("issue_date") or "",
                ),
            )
            anchor_doc = sorted_docs[0]

            # Thu thập danh sách điều khoản của cụm
            cluster_provs = [
                d for n, d in prov_nodes.items()
                if d.get("document_id") in doc_set and not d.get("is_external_stub", False)
            ]

            # Thống kê quan hệ nội bộ cụm
            comm_rels: Dict[str, int] = {}
            for (u, v), info in edge_counter.items():
                if u in doc_set and v in doc_set:
                    for r, c in info["rels"].items():
                        comm_rels[r] = comm_rels.get(r, 0) + c

            # Đặt tiêu đề cụm pháp lý rõ ràng
            anchor_title = anchor_doc.get("full_title") or anchor_doc.get("title", "")
            # Rút gọn tên Luật / Nghị định để làm chủ đề
            clean_topic = re.sub(r"^(Luật|Nghị định|Thông tư|Quyết định)\s+(số\s+[^\s]+\s+)?", "", anchor_title, flags=re.I).strip()
            if not clean_topic:
                clean_topic = anchor_title

            doc_nums = [d.get("doc_number", "") for d in docs if d.get("doc_number")]
            top_nums_str = ", ".join(doc_nums[:3])
            if len(doc_nums) > 3:
                top_nums_str += f" (+{len(doc_nums)-3} VB)"

            comm_title = f"Cụm {clean_topic[:60]} ({top_nums_str})"

            # Chọn các điều khoản tiêu biểu (Phạm vi, nguyên tắc, hoặc sửa đổi)
            key_provisions: List[Dict[str, str]] = []
            for p in cluster_provs:
                t = p.get("title", "")
                if any(kw in t.lower() for kw in ["phạm vi", "đối tượng", "nguyên tắc", "sửa đổi", "bổ sung", "tiêu chuẩn", "thẩm quyền", "nhiệm vụ", "quy chế"]):
                    key_provisions.append({
                        "provision_id": p.get("provision_id", ""),
                        "title": t,
                        "structural_path": p.get("structural_path", ""),
                        "snippet": (p.get("content", "") or "")[:200],
                    })
                if len(key_provisions) >= 6:
                    break

            if not key_provisions and cluster_provs:
                for p in cluster_provs[:4]:
                    key_provisions.append({
                        "provision_id": p.get("provision_id", ""),
                        "title": p.get("title", ""),
                        "structural_path": p.get("structural_path", ""),
                        "snippet": (p.get("content", "") or "")[:200],
                    })

            # Tổng hợp báo cáo tóm tắt cộng đồng (Community Summary)
            summary_lines = [
                f"**Cụm chủ đề:** {clean_topic}.",
                f"**Văn bản nòng cốt:** {anchor_doc.get('doc_type')} số {anchor_doc.get('doc_number')} ({anchor_doc.get('full_title')}).",
                f"**Quy mô:** Bao gồm {len(docs)} văn bản QPPL liên quan với tổng cộng {len(cluster_provs)} điều khoản.",
            ]
            if comm_rels:
                rel_parts = [f"{r}: {c}" for r, c in comm_rels.items()]
                summary_lines.append(f"**Các liên kết pháp lý nội cụm:** {', '.join(rel_parts)}.")

            doc_details = []
            for d in docs[:5]:
                doc_details.append(f"- {d.get('doc_type')} {d.get('doc_number')}: {d.get('full_title', '')}")
            if doc_details:
                summary_lines.append("**Danh mục văn bản chính:**\n" + "\n".join(doc_details))

            comm_summary = "\n\n".join(summary_lines)

            comm_obj = LegalCommunity(
                community_id=f"COMM_{idx:02d}",
                title=comm_title,
                summary=comm_summary,
                doc_ids=list(doc_set),
                doc_numbers=doc_nums,
                doc_titles=[d.get("full_title", "") for d in docs],
                provision_count=len(cluster_provs),
                relations_count=comm_rels,
                key_provisions=key_provisions,
                level=0,
                weight=float(len(cluster_provs)),
            )
            community_objects.append(comm_obj)

        # Sắp xếp theo số lượng điều khoản giảm dần
        community_objects.sort(key=lambda c: c.provision_count, reverse=True)
        self.communities = community_objects

        self.save_cache()
        print(f"LegalCommunityDetector: built and saved {len(self.communities)} legal communities.")
        return self.communities

    def save_cache(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
        data = [c.model_dump() for c in self.communities]
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_cache(self):
        with open(self.cache_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.communities = [LegalCommunity(**item) for item in raw]
        print(f"LegalCommunityDetector: loaded {len(self.communities)} communities from cache.")

    def get_communities(self) -> List[LegalCommunity]:
        return self.communities

    def find_relevant_communities(self, query: str, top_k: int = 4) -> List[Tuple[LegalCommunity, float]]:
        """
        Tìm và xếp hạng các cụm cộng đồng liên quan nhất tới câu hỏi người dùng
        dựa trên độ tương đồng từ khóa pháp lý (Lexical overlap + bonus trọng số).
        """
        if not self.communities:
            return []

        q_terms = set(re.findall(r"\w+", query.lower()))
        if not q_terms:
            return [(c, 1.0) for c in self.communities[:top_k]]

        scored: List[Tuple[LegalCommunity, float]] = []

        # Các thuật ngữ chuyên ngành quan trọng để tăng trọng số
        key_weights = {
            "biên": 2.0, "giới": 2.0, "cửa": 2.0, "khẩu": 2.0,
            "bảo": 1.5, "vệ": 1.5, "doanh": 1.8, "nghiệp": 1.8, "cơ": 1.5, "quan": 1.5,
            "an": 1.5, "ninh": 1.8, "quốc": 1.5, "gia": 1.5,
            "sửa": 1.8, "đổi": 1.8, "bổ": 1.8, "sung": 1.8,
            "đền": 2.0, "bù": 2.0, "trợ": 1.8, "cấp": 1.8, "thiệt": 1.8, "hại": 1.8,
            "dân": 1.8, "phố": 2.0, "trang": 1.8, "phục": 2.0, "tiêu": 1.5, "chuẩn": 1.5,
        }

        for comm in self.communities:
            # Tạo corpus văn bản của cụm
            corpus = " ".join([
                comm.title,
                comm.summary,
                " ".join(comm.doc_numbers),
                " ".join(comm.doc_titles),
                " ".join(p.get("title", "") for p in comm.key_provisions),
            ]).lower()

            c_terms = set(re.findall(r"\w+", corpus))
            matched_terms = q_terms.intersection(c_terms)

            if not matched_terms:
                score = 0.0
            else:
                raw_overlap = sum(key_weights.get(t, 1.0) for t in matched_terms)
                total_q_weight = sum(key_weights.get(t, 1.0) for t in q_terms)
                score = min(1.0, raw_overlap / max(1.0, total_q_weight))

                # Kiểm tra số hiệu văn bản trực tiếp trong câu hỏi (ví dụ: "112/2014", "32/2004", "06/2013")
                for doc_num in comm.doc_numbers:
                    num_core = doc_num.split("/")[0] if "/" in doc_num else doc_num
                    if num_core in query:
                        score = min(1.0, score + 0.4)
                        break

            scored.append((comm, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Nếu top 1 có điểm > 0 thì trả về top_k kết quả có điểm
        positives = [item for item in scored if item[1] > 0]
        if positives:
            return positives[:top_k]

        # Fallback về các cụm lớn nhất
        return [(c, 0.1) for c in self.communities[:top_k]]
