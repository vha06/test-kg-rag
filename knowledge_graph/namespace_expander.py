from __future__ import annotations

import logging
from collections import deque
from typing import List, Dict, Any, Optional, Set

from storage.db import DatabaseManager
from knowledge_graph.multi_graph_builder import MultiDomainGraphBuilder
from retrieval.multi_domain_retriever import CanonicalRetrievalResult

logger = logging.getLogger(__name__)


class NamespaceGraphExpander:
    """
    NamespaceGraphExpander performs graph expansion around retrieval seed chunks,
    strictly constrained within authorized graph namespaces and optionally
    weaving verified cross-domain links.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        graph_builder: Optional[MultiDomainGraphBuilder] = None,
    ):
        self.db = db_manager
        self.graph_builder = graph_builder or MultiDomainGraphBuilder(db_manager)

    def _resolve_seed_entities(
        self, seed_results: List[CanonicalRetrievalResult]
    ) -> Dict[str, Set[str]]:
        """
        Map chunk_id -> set of entity_ids mentioned in or associated with that chunk.
        """
        chunk_to_entities: Dict[str, Set[str]] = {s.chunk_id: set() for s in seed_results}
        chunk_ids = list(chunk_to_entities.keys())
        if not chunk_ids:
            return chunk_to_entities

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(chunk_ids))
            query = f"""
                SELECT entity_id, chunk_id 
                FROM platform_entity_mentions 
                WHERE chunk_id IN ({placeholders})
            """
            cursor.execute(query, tuple(chunk_ids))
            for row in cursor.fetchall():
                ent_id = row["entity_id"]
                c_id = row["chunk_id"]
                if c_id in chunk_to_entities:
                    chunk_to_entities[c_id].add(ent_id)

        # Fallback: surface text match against all entities in DB
        all_entities = self.db.get_platform_entities()
        for seed in seed_results:
            content_lower = seed.content.lower()
            for ent in all_entities:
                cname = ent.get("canonical_name", "").lower()
                if cname and len(cname) > 3 and cname in content_lower:
                    chunk_to_entities[seed.chunk_id].add(ent["entity_id"])

        return chunk_to_entities

    def expand(
        self,
        seed_results: List[CanonicalRetrievalResult],
        max_hops: int = 1,
        allowed_namespaces: Optional[List[str]] = None,
        allow_cross_domain: bool = False,
        include_candidates: bool = False,
        max_expanded_nodes: int = 5,
    ) -> Dict[str, Any]:
        """
        Expand knowledge graph from seed chunks.
        
        Args:
            seed_results: The initial retrieval chunks.
            max_hops: Maximum graph traversal hops (typically 1 or 2).
            allowed_namespaces: List of namespaces allowed for traversal.
            allow_cross_domain: Whether cross-domain mappings are traversed.
            include_candidates: Whether unverified edges/entities are included.
            max_expanded_nodes: Maximum number of new nodes to expand.
        """
        if not seed_results:
            return {
                "seed_chunk_ids": [],
                "seed_entity_ids": [],
                "expanded_entities": [],
                "expanded_chunks": [],
                "expansion_paths": [],
            }

        # 1. Resolve seed entities
        chunk_entity_map = self._resolve_seed_entities(seed_results)
        seed_entity_ids: Set[str] = set()
        for ents in chunk_entity_map.values():
            seed_entity_ids.update(ents)

        # 2. Determine namespaces
        if allowed_namespaces is None:
            # Derive from entities or domains
            all_ents = self.db.get_platform_entities()
            derived_ns = {
                e["graph_namespace"]
                for e in all_ents
                if e["entity_id"] in seed_entity_ids and e.get("graph_namespace")
            }
            allowed_namespaces = list(derived_ns) if derived_ns else ["default"]

        # 3. Build graph partition
        graph = self.graph_builder.build_multi_namespace_graph(
            namespaces=allowed_namespaces,
            include_cross_domain_links=allow_cross_domain,
            include_candidates=include_candidates,
        )

        # 4. BFS Traversal
        visited_nodes: Set[str] = set(seed_entity_ids)
        expanded_entities: List[Dict[str, Any]] = []
        expansion_paths: List[Dict[str, Any]] = []
        expanded_chunk_ids: Set[str] = set()
        seed_chunk_ids = [s.chunk_id for s in seed_results]

        queue: deque = deque([(ent_id, 0, [ent_id]) for ent_id in seed_entity_ids if graph.has_node(ent_id)])

        while queue and len(expanded_entities) < max_expanded_nodes:
            curr_id, hops, current_path = queue.popleft()
            if hops >= max_hops:
                continue

            # Check outgoing and incoming edges
            edges = []
            for successor in graph.successors(curr_id):
                for edge_key, edge_data in graph.get_edge_data(curr_id, successor).items():
                    edges.append((curr_id, successor, "OUT", edge_data))
            for predecessor in graph.predecessors(curr_id):
                for edge_key, edge_data in graph.get_edge_data(predecessor, curr_id).items():
                    edges.append((predecessor, curr_id, "IN", edge_data))

            for src, tgt, direction, edge_data in edges:
                neighbor = tgt if direction == "OUT" else src
                if neighbor in visited_nodes:
                    continue

                visited_nodes.add(neighbor)
                neighbor_data = graph.nodes[neighbor]
                node_provenance = self.graph_builder.get_node_provenance(neighbor)
                edge_provenance = self.graph_builder.get_edge_provenance(
                    edge_data.get("relationship_id", "")
                )

                step_info = {
                    "source": src,
                    "target": tgt,
                    "predicate": edge_data.get("predicate", "RELATED_TO"),
                    "confidence": edge_data.get("confidence", 1.0),
                    "is_cross_domain": edge_data.get("is_cross_domain", False),
                    "verification_status": edge_data.get("verification_status", "VERIFIED"),
                    "edge_provenance": edge_provenance,
                }
                new_path = list(current_path) + [step_info, neighbor]

                expanded_ent_entry = {
                    "entity_id": neighbor,
                    "canonical_name": neighbor_data.get("canonical_name", neighbor),
                    "entity_type": neighbor_data.get("entity_type", "ENTITY"),
                    "graph_namespace": neighbor_data.get("graph_namespace", ""),
                    "domain_id": neighbor_data.get("domain_id", ""),
                    "verification_status": neighbor_data.get("verification_status", "VERIFIED"),
                    "provenance": node_provenance,
                    "expansion_path": new_path,
                }
                expanded_entities.append(expanded_ent_entry)
                expansion_paths.append({
                    "target_entity_id": neighbor,
                    "path": new_path,
                    "edge_info": step_info,
                })

                # Find chunks mentioning this expanded entity
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT chunk_id FROM platform_entity_mentions WHERE entity_id = ?",
                        (neighbor,)
                    )
                    for r in cursor.fetchall():
                        cid = r["chunk_id"]
                        if cid and cid not in seed_chunk_ids:
                            expanded_chunk_ids.add(cid)

                queue.append((neighbor, hops + 1, new_path))
                if len(expanded_entities) >= max_expanded_nodes:
                    break

        # 5. Fetch expanded chunk objects
        expanded_chunks: List[CanonicalRetrievalResult] = []
        if expanded_chunk_ids:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                p_holders = ",".join(["?"] * len(expanded_chunk_ids))
                cursor.execute(f"SELECT * FROM platform_chunks WHERE chunk_id IN ({p_holders})", tuple(expanded_chunk_ids))
                for row in cursor.fetchall():
                    c_dict = dict(row)
                    expanded_chunks.append(
                        CanonicalRetrievalResult(
                            chunk_id=c_dict["chunk_id"],
                            document_id=c_dict["document_id"],
                            domain_id=c_dict["domain_id"],
                            content=c_dict["content"],
                            score=0.85,
                            retrieval_method="GRAPH_EXPANSION",
                            source_block_ids=[],
                            source_location={},
                            security_classification=c_dict.get("security_classification", "INTERNAL"),
                            profile_id=c_dict.get("profile_id"),
                        )
                    )

        return {
            "seed_chunk_ids": seed_chunk_ids,
            "seed_entity_ids": list(seed_entity_ids),
            "expanded_entities": expanded_entities,
            "expanded_chunks": expanded_chunks,
            "expansion_paths": expansion_paths,
        }
