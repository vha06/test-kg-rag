from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
import networkx as nx

from storage.db import DatabaseManager

logger = logging.getLogger(__name__)


class MultiDomainGraphBuilder:
    """
    MultiDomainGraphBuilder builds and inspects NetworkX MultiDiGraph instances
    partitioned strictly by graph_namespace, ensuring namespace isolation,
    candidate edge filtering, and complete provenance traceability.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def build_namespace_graph(
        self,
        graph_namespace: str,
        include_candidates: bool = False,
    ) -> nx.MultiDiGraph:
        """
        Build an in-memory MultiDiGraph strictly isolated to the specified graph_namespace.
        
        Args:
            graph_namespace: The graph partition namespace (e.g., 'ops_v1', 'rd_v1').
            include_candidates: If True, candidate (UNVERIFIED) entities and relationships
                                are included. Defaults to False (VERIFIED only).
        """
        graph = nx.MultiDiGraph(graph_namespace=graph_namespace)

        # 1. Fetch Entities for this namespace
        raw_entities = self.db.get_platform_entities(graph_namespace=graph_namespace)
        for ent in raw_entities:
            v_status = ent.get("verification_status", "UNVERIFIED")
            if not include_candidates and v_status != "VERIFIED":
                continue

            graph.add_node(
                ent["entity_id"],
                canonical_name=ent.get("canonical_name", ""),
                entity_type=ent.get("entity_type", "ENTITY"),
                aliases=ent.get("aliases", []),
                description=ent.get("description", ""),
                domain_id=ent.get("domain_id", ""),
                graph_namespace=ent.get("graph_namespace", graph_namespace),
                extraction_method=ent.get("extraction_method", "RULE_BASED"),
                extractor_version=ent.get("extractor_version", "1.0.0"),
                verification_status=v_status,
                attributes=ent.get("attributes", {}),
                evidence_refs=ent.get("evidence_refs", []),
            )

        # 2. Fetch Relationships for this namespace
        raw_relationships = self.db.get_platform_relationships(
            graph_namespace=graph_namespace,
            include_unverified=include_candidates,
        )

        for rel in raw_relationships:
            src = rel["subject_entity_id"]
            tgt = rel["object_entity_id"]
            # Enforce that endpoints exist in this graph partition
            if graph.has_node(src) and graph.has_node(tgt):
                graph.add_edge(
                    src,
                    tgt,
                    key=rel["relationship_id"],
                    relationship_id=rel["relationship_id"],
                    predicate=rel["predicate"],
                    domain_id=rel.get("domain_id", ""),
                    graph_namespace=rel.get("graph_namespace", graph_namespace),
                    evidence_refs=rel.get("evidence_refs", []),
                    confidence=float(rel.get("confidence", 1.0)),
                    verification_status=rel.get("verification_status", "UNVERIFIED"),
                    extraction_method=rel.get("extraction_method", "RULE_BASED"),
                    extractor_version=rel.get("extractor_version", "1.0.0"),
                    recorded_at=rel.get("recorded_at", ""),
                    attributes=rel.get("attributes", {}),
                )

        return graph

    def build_multi_namespace_graph(
        self,
        namespaces: List[str],
        include_cross_domain_links: bool = False,
        include_candidates: bool = False,
    ) -> nx.MultiDiGraph:
        """
        Build a unified graph combining multiple namespaces, optionally weaving
        cross-domain mappings from platform_entity_mappings.
        """
        combined = nx.MultiDiGraph(namespaces=namespaces)

        for ns in namespaces:
            ns_graph = self.build_namespace_graph(
                graph_namespace=ns,
                include_candidates=include_candidates,
            )
            combined.add_nodes_from(ns_graph.nodes(data=True))
            combined.add_edges_from(ns_graph.edges(keys=True, data=True))

        if include_cross_domain_links:
            # Query all cross-domain mappings between the selected namespaces
            mappings = self.db.get_platform_entity_mappings()
            for m in mappings:
                src_ns = m.get("source_namespace")
                tgt_ns = m.get("target_namespace")
                if src_ns in namespaces and tgt_ns in namespaces:
                    src_id = m.get("source_entity_id")
                    tgt_id = m.get("target_entity_id")
                    v_status = m.get("verification_status", "VERIFIED")
                    if not include_candidates and v_status != "VERIFIED":
                        continue
                    if combined.has_node(src_id) and combined.has_node(tgt_id):
                        combined.add_edge(
                            src_id,
                            tgt_id,
                            key=m.get("mapping_id", f"map_{src_id}_{tgt_id}"),
                            relationship_id=m.get("mapping_id"),
                            predicate=m.get("relation_type", "SAME_AS"),
                            confidence=float(m.get("confidence", 1.0)),
                            verification_status=v_status,
                            is_cross_domain=True,
                            attributes=m.get("metadata", {}),
                        )

        return combined

    def get_node_provenance(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all EvidenceRecord entries supporting a given CanonicalEntity.
        """
        entities = self.db.get_platform_entities()
        target = next((e for e in entities if e["entity_id"] == entity_id), None)
        if not target:
            return []
        evidence_ids = target.get("evidence_refs", [])
        if not evidence_ids:
            return []
        return self.db.get_platform_evidences(evidence_ids=evidence_ids)

    def get_edge_provenance(self, relationship_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all EvidenceRecord entries supporting a given CanonicalRelationship.
        """
        rels = self.db.get_platform_relationships(include_unverified=True)
        target = next((r for r in rels if r["relationship_id"] == relationship_id), None)
        if not target:
            return []
        evidence_ids = target.get("evidence_refs", [])
        if not evidence_ids:
            return []
        return self.db.get_platform_evidences(evidence_ids=evidence_ids)
