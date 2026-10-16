from typing import List, Dict, Any, Set, Optional, Tuple
import networkx as nx
from storage.models import RetrievalResult
from knowledge_graph.builder import KnowledgeGraphBuilder


class GraphExpander:
    DEFAULT_WEIGHTS = {
        "AMENDS": 1.0,
        "SUPPLEMENTS": 1.0,
        "REPEALS": 1.0,
        "DETAILS": 0.9,
        "REFERENCES": 0.85,
        "CONTAINS": 0.5,
    }

    def __init__(
        self,
        kg_builder: Optional[KnowledgeGraphBuilder] = None,
        decay_factor: float = 0.7,
        relation_weights: Optional[Dict[str, float]] = None,
    ):
        self.kg_builder = kg_builder or KnowledgeGraphBuilder()
        self.graph = self.kg_builder.build_graph(verified_only=True)
        self.decay_factor = decay_factor
        self.relation_weights = relation_weights or self.DEFAULT_WEIGHTS

    def expand(
        self,
        query: str,
        seed_results: List[RetrievalResult],
        max_hops: int = 1,
        seed_limit: int = 3,
        max_expanded_nodes: int = 5,
        allowed_relations: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        if not seed_results:
            return []

        if allowed_relations is None:
            allowed_relations = ["AMENDS", "SUPPLEMENTS", "REPEALS", "DETAILS", "REFERENCES"]

        allowed_set = set(allowed_relations)
        seen_provisions: Set[str] = {r.provision_id for r in seed_results}
        seeds_to_expand = seed_results[:seed_limit]

        q_terms = set(query.lower().split())

        expanded_candidates: Dict[str, Tuple[RetrievalResult, float]] = {}

        for seed in seeds_to_expand:
            seed_id = seed.provision_id
            if not self.graph.has_node(seed_id):
                continue

            # Queue: (current_node, current_hop, path_str, current_score)
            queue = [(seed_id, 0, f"Seed: {seed.provision_title}", seed.score)]
            visited_in_path = {seed_id}

            while queue:
                curr_node, hop, path, curr_score = queue.pop(0)
                if hop >= max_hops:
                    continue

                # Inspect both outgoing and incoming edges
                neighbors: List[Tuple[str, str, str, str]] = []  # (next_node, rel, evidence, direction)
                
                # Outgoing
                for _, nxt, k, d in self.graph.out_edges(curr_node, keys=True, data=True):
                    neighbors.append((nxt, d.get("relation_type", ""), d.get("evidence_text", ""), "-->"))
                # Incoming
                for prev, _, k, d in self.graph.in_edges(curr_node, keys=True, data=True):
                    neighbors.append((prev, d.get("relation_type", ""), d.get("evidence_text", ""), "<--"))

                for nxt_node, rel, evidence, direction in neighbors:
                    if rel not in allowed_set or nxt_node in visited_in_path:
                        continue

                    node_data = self.graph.nodes.get(nxt_node, {})
                    if node_data.get("is_external_stub", False):
                        continue

                    # Only expand to provisions (or document level mapped to doc)
                    if node_data.get("node_type") != "PROVISION":
                        continue

                    rel_weight = self.relation_weights.get(rel, 0.5)
                    node_content = node_data.get("content", "")

                    # Lexical overlap bonus
                    overlap = 0.0
                    if q_terms and node_content:
                        c_terms = set(node_content.lower().split())
                        overlap = len(q_terms.intersection(c_terms)) / len(q_terms)

                    cand_score = curr_score * (self.decay_factor ** (hop + 1)) * rel_weight * (0.5 + 0.5 * overlap)
                    step_str = f" {direction}[{rel}] {node_data.get('title', nxt_node)}"
                    new_path = path + step_str

                    if nxt_node not in seen_provisions:
                        doc_id = node_data.get("document_id", "")
                        doc_node = self.graph.nodes.get(doc_id, {})
                        
                        cand_res = RetrievalResult(
                            chunk_id=f"KG_{nxt_node}",
                            provision_id=nxt_node,
                            document_id=doc_id,
                            doc_number=doc_node.get("doc_number", ""),
                            doc_title=doc_node.get("full_title", ""),
                            provision_title=node_data.get("title", ""),
                            structural_path=node_data.get("structural_path", ""),
                            content=node_content,
                            score=round(cand_score, 4),
                            rank=0,  # will assign after sort
                            retrieval_method="KG_EXPANSION",
                            source_url=node_data.get("source_url"),
                            effective_start=node_data.get("effective_start"),
                            effective_end=node_data.get("effective_end"),
                            status=node_data.get("status", "ACTIVE"),
                            expansion_path=new_path,
                            evidence_text=evidence,
                        )

                        if nxt_node not in expanded_candidates or cand_score > expanded_candidates[nxt_node][1]:
                            expanded_candidates[nxt_node] = (cand_res, cand_score)

                    visited_in_path.add(nxt_node)
                    queue.append((nxt_node, hop + 1, new_path, cand_score))

        # Sort expanded nodes by score
        sorted_expanded = sorted(expanded_candidates.values(), key=lambda x: x[1], reverse=True)
        top_expanded = [item[0] for item in sorted_expanded[:max_expanded_nodes]]

        # Combine seed and expanded
        combined = list(seed_results)
        current_rank = len(seed_results) + 1
        for item in top_expanded:
            item.rank = current_rank
            combined.append(item)
            current_rank += 1

        return combined
