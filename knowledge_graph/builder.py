from typing import Dict, Any, List, Optional
import networkx as nx
from storage.db import DatabaseManager


class KnowledgeGraphBuilder:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.graph = nx.MultiDiGraph()

    def build_graph(self, verified_only: bool = True) -> nx.MultiDiGraph:
        self.graph.clear()

        # Add document nodes
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents")
            for doc in cursor.fetchall():
                d = dict(doc)
                self.graph.add_node(
                    d["document_id"],
                    node_type="DOCUMENT",
                    title=f"{d['doc_type']} số {d['doc_number']}",
                    doc_number=d["doc_number"],
                    doc_type=d["doc_type"],
                    full_title=d["title"],
                    effective_start=d["effective_start"],
                    effective_end=d["effective_end"],
                    status=d["status"],
                    source_url=d["source_url"],
                )

            # Add provision nodes
            cursor.execute("SELECT * FROM provisions")
            for prov in cursor.fetchall():
                p = dict(prov)
                self.graph.add_node(
                    p["provision_id"],
                    node_type="PROVISION",
                    level=p["level"],
                    document_id=p["document_id"],
                    title=p["title"],
                    structural_path=p["structural_path"],
                    content=p["content"],
                    mapc=p["mapc"],
                    effective_start=p["effective_start"],
                    effective_end=p["effective_end"],
                    status=p["status"],
                    source_url=p["source_url"],
                    is_external_stub=bool(p["is_external_stub"]),
                )

            # Add edges
            edges = self.db.get_kg_edges(verified_only=verified_only)
            for e in edges:
                src = e["source_id"]
                tgt = e["target_id"]
                if self.graph.has_node(src) and self.graph.has_node(tgt):
                    self.graph.add_edge(
                        src,
                        tgt,
                        key=e["edge_id"],
                        relation_type=e["relation_type"],
                        evidence_text=e["evidence_text"],
                        effective_date=e["effective_date"],
                        detection_method=e["detection_method"],
                        verified=e["verified"],
                        source_url=e["source_url"],
                    )

        print(f"Knowledge Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return self.graph

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None
