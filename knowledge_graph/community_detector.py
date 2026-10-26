from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx

from core.schema import (
    CanonicalEntity,
    CanonicalRelationship,
    CanonicalCommunity,
)

logger = logging.getLogger(__name__)


class HierarchicalCommunityDetector:
    """
    Phân cụm đồ thị phân cấp đa tầng (Hierarchical Community Detection):
    - Chuyển đổi đồ thị sang đồ thị vô hướng có trọng số (weighted undirected graph).
    - Hỗ trợ liên kết ngữ cảnh mềm (Soft Chunk-Hub Bridging) cho các thực thể cô lập cùng chunk.
    - Phân cụm đa cấp Level 0 (Coarse/Macro) -> Level 1 (Intermediate) -> Level 2 (Fine/Micro)
      sử dụng Multi-resolution Louvain trên NetworkX.
    - Cơ chế Early Stopping: Không nhân bản cụm lá sang các tầng con khi không thể chia nhỏ.
    - Thiết lập liên kết phân cấp parent_community_id <-> child_community_ids.
    - Kế thừa toàn bộ provenance source_chunk_ids và source_document_ids từ các thực thể thành viên.
    """

    def __init__(
        self,
        algorithm: str = "louvain_hierarchical",
        max_levels: int = 3,
        min_split_size: int = 6,
        min_community_size: int = 2,
        seed: int = 42,
    ):
        self.algorithm = algorithm
        self.max_levels = max(1, min(max_levels, 3))
        self.min_split_size = max(2, min_split_size)
        self.min_community_size = max(1, min(min_community_size, self.min_split_size))
        self.seed = seed

    def _merge_small_clusters(
        self,
        subgraph: nx.Graph,
        raw_clusters: List[Any],
        min_size: int = 2,
    ) -> List[List[str]]:
        """
        Gộp các cụm đơn lẻ (size < min_size) vào cụm láng giềng liên kết mạnh nhất trong subgraph
        để loại bỏ hiện tượng cụm đơn lẻ 1 thực thể.
        """
        valid_clusters: List[List[str]] = [list(c) for c in raw_clusters if len(c) >= min_size]
        small_nodes: List[str] = []
        for c in raw_clusters:
            if len(c) < min_size:
                small_nodes.extend(list(c))

        if not valid_clusters:
            all_nodes = list(subgraph.nodes())
            return [all_nodes] if all_nodes else []

        for node in small_nodes:
            best_idx = 0
            best_score = -1.0
            for idx, cluster in enumerate(valid_clusters):
                conn_weight = sum(
                    subgraph[node][nbr].get("weight", 1.0)
                    for nbr in cluster
                    if subgraph.has_edge(node, nbr)
                )
                if conn_weight > best_score:
                    best_score = conn_weight
                    best_idx = idx

            if best_score <= 0.0:
                best_idx = min(range(len(valid_clusters)), key=lambda i: len(valid_clusters[i]))

            valid_clusters[best_idx].append(node)

        cleaned: List[List[str]] = []
        for cl in valid_clusters:
            uniq = list(dict.fromkeys(cl))
            if uniq:
                cleaned.append(uniq)
        return sorted(cleaned, key=len, reverse=True)

    def _try_split_subgraph(self, subgraph: nx.Graph, resolution: float) -> List[List[str]]:
        """
        Thử phân rã một subgraph thành các cụm con:
        - Chỉ phân rã nếu số đỉnh >= min_split_size và số cạnh >= 1.
        - Chỉ chấp nhận nếu tạo thành ít nhất 2 cụm con hợp lệ (Early Stopping).
        """
        if subgraph.number_of_nodes() < self.min_split_size or subgraph.number_of_edges() < 1:
            return []
        try:
            raw_sub = nx.community.louvain_communities(
                subgraph, weight="weight", resolution=resolution, seed=self.seed
            )
            merged = self._merge_small_clusters(subgraph, raw_sub, min_size=self.min_community_size)
            if len(merged) >= 2:
                return merged
            return []
        except Exception as ex:
            logger.debug(f"Sub-community split skipped: {ex}")
            return []

    def detect_communities(
        self,
        entities: List[CanonicalEntity],
        relationships: List[CanonicalRelationship],
        namespace: str = "graphrag_default",
    ) -> List[CanonicalCommunity]:
        if not entities:
            return []

        # 1. Xây dựng đồ thị vô hướng có trọng số
        G = nx.Graph()
        ent_by_id: Dict[str, CanonicalEntity] = {e.entity_id: e for e in entities}
        for e in entities:
            G.add_node(e.entity_id, canonical_name=e.canonical_name, entity_type=e.entity_type)

        rel_by_pair: Dict[Tuple[str, str], List[CanonicalRelationship]] = {}
        for r in relationships:
            src = r.subject_entity_id
            tgt = r.object_entity_id
            if src not in ent_by_id or tgt not in ent_by_id or src == tgt:
                continue
            pair = tuple(sorted([src, tgt]))
            rel_by_pair.setdefault(pair, []).append(r)

        for (u, v), rel_list in rel_by_pair.items():
            combined_weight = sum(r.weight for r in rel_list)
            rel_ids = [r.relationship_id for r in rel_list]
            G.add_edge(u, v, weight=combined_weight, relationship_ids=rel_ids, is_soft=False)

        # 2. Soft Chunk-Hub Bridging: Nối các thực thể trong cùng một chunk về hub node của chunk
        chunk_to_nodes: Dict[str, List[str]] = {}
        for e in entities:
            for cid in (e.source_chunk_ids or []):
                chunk_to_nodes.setdefault(cid, []).append(e.entity_id)

        for cid, nlist in chunk_to_nodes.items():
            uniq_nodes = list(dict.fromkeys(nlist))
            if len(uniq_nodes) > 1:
                hub = max(uniq_nodes, key=lambda n: G.degree(n, weight="weight"))
                for n in uniq_nodes:
                    if n != hub and not G.has_edge(n, hub):
                        G.add_edge(n, hub, weight=0.25, relationship_ids=[], is_soft=True)

        # Xử lý các node cô lập còn lại (nếu có)
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        if isolated:
            connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
            if connected_nodes:
                top_hub = max(connected_nodes, key=lambda n: G.degree(n, weight="weight"))
                for n in isolated:
                    G.add_edge(n, top_hub, weight=0.15, relationship_ids=[], is_soft=True)
            elif len(isolated) > 1:
                for i in range(len(isolated) - 1):
                    G.add_edge(isolated[i], isolated[i + 1], weight=0.15, relationship_ids=[], is_soft=True)

        # 3. Xử lý đồ thị quá nhỏ
        if G.number_of_edges() == 0 or G.number_of_nodes() <= self.min_community_size:
            return [self._build_single_community(entities, relationships, namespace, level=0, comm_idx=0)]

        # 4. Phân cụm đa cấp Level 0 -> Level 1 -> Level 2
        all_communities: List[CanonicalCommunity] = []
        comm_counter = 0

        # --- LEVEL 0: Coarse / Macro clustering ---
        try:
            raw_l0 = nx.community.louvain_communities(
                G, weight="weight", resolution=0.8, seed=self.seed
            )
        except Exception as ex:
            logger.warning(f"Louvain level 0 failed ({ex}), falling back to connected components")
            raw_l0 = list(nx.connected_components(G))

        l0_clusters = self._merge_small_clusters(G, raw_l0, min_size=self.min_community_size)
        if not l0_clusters:
            return [self._build_single_community(entities, relationships, namespace, level=0, comm_idx=0)]

        for l0_idx, l0_nodes in enumerate(l0_clusters):
            comm_counter += 1
            l0_id = f"{namespace}__COMM_L0_{comm_counter:04d}"
            l0_subgraph = G.subgraph(l0_nodes)
            l0_rel_ids = self._get_subgraph_relation_ids(l0_subgraph, rel_by_pair)

            l0_comm = CanonicalCommunity(
                community_id=l0_id,
                namespace=namespace,
                level=0,
                parent_community_id=None,
                child_community_ids=[],
                title=self._generate_community_title(l0_nodes, ent_by_id, level=0, idx=l0_idx),
                entity_ids=l0_nodes,
                relationship_ids=l0_rel_ids,
                source_chunk_ids=self._aggregate_chunk_ids(l0_nodes, l0_rel_ids, ent_by_id, relationships),
                source_document_ids=self._aggregate_doc_ids(l0_nodes, ent_by_id),
                size=len(l0_nodes),
            )
            all_communities.append(l0_comm)

            if self.max_levels <= 1:
                continue

            # --- LEVEL 1: Intermediate clustering ---
            l1_clusters = self._try_split_subgraph(l0_subgraph, resolution=1.5)
            # Early Stopping: Nếu không chia nhỏ thành >= 2 cụm con hợp lệ thì dừng lại, KHÔNG nhân bản cụm lá!
            if not l1_clusters:
                continue

            for l1_idx, l1_nodes in enumerate(l1_clusters):
                comm_counter += 1
                l1_id = f"{namespace}__COMM_L1_{comm_counter:04d}"
                l0_comm.child_community_ids.append(l1_id)

                l1_subgraph = G.subgraph(l1_nodes)
                l1_rel_ids = self._get_subgraph_relation_ids(l1_subgraph, rel_by_pair)

                l1_comm = CanonicalCommunity(
                    community_id=l1_id,
                    namespace=namespace,
                    level=1,
                    parent_community_id=l0_id,
                    child_community_ids=[],
                    title=self._generate_community_title(l1_nodes, ent_by_id, level=1, idx=l1_idx),
                    entity_ids=l1_nodes,
                    relationship_ids=l1_rel_ids,
                    source_chunk_ids=self._aggregate_chunk_ids(l1_nodes, l1_rel_ids, ent_by_id, relationships),
                    source_document_ids=self._aggregate_doc_ids(l1_nodes, ent_by_id),
                    size=len(l1_nodes),
                )
                all_communities.append(l1_comm)

                if self.max_levels <= 2:
                    continue

                # --- LEVEL 2: Fine / Micro clustering ---
                l2_clusters = self._try_split_subgraph(l1_subgraph, resolution=2.0)
                if not l2_clusters:
                    continue

                for l2_idx, l2_nodes in enumerate(l2_clusters):
                    comm_counter += 1
                    l2_id = f"{namespace}__COMM_L2_{comm_counter:04d}"
                    l1_comm.child_community_ids.append(l2_id)

                    l2_subgraph = G.subgraph(l2_nodes)
                    l2_rel_ids = self._get_subgraph_relation_ids(l2_subgraph, rel_by_pair)

                    l2_comm = CanonicalCommunity(
                        community_id=l2_id,
                        namespace=namespace,
                        level=2,
                        parent_community_id=l1_id,
                        child_community_ids=[],
                        title=self._generate_community_title(l2_nodes, ent_by_id, level=2, idx=l2_idx),
                        entity_ids=l2_nodes,
                        relationship_ids=l2_rel_ids,
                        source_chunk_ids=self._aggregate_chunk_ids(l2_nodes, l2_rel_ids, ent_by_id, relationships),
                        source_document_ids=self._aggregate_doc_ids(l2_nodes, ent_by_id),
                        size=len(l2_nodes),
                    )
                    all_communities.append(l2_comm)

        return all_communities

    def _build_single_community(
        self,
        entities: List[CanonicalEntity],
        relationships: List[CanonicalRelationship],
        namespace: str,
        level: int = 0,
        comm_idx: int = 0,
    ) -> CanonicalCommunity:
        ent_ids = [e.entity_id for e in entities]
        rel_ids = [r.relationship_id for r in relationships]
        chunk_ids: List[str] = []
        doc_ids: List[str] = []
        for e in entities:
            chunk_ids.extend(e.source_chunk_ids)
            doc_ids.extend(e.source_document_ids)

        names = [e.canonical_name for e in entities[:3]]
        title = f"Cộng đồng tổng thể ({', '.join(names)})" if names else f"Cộng đồng {namespace}"

        return CanonicalCommunity(
            community_id=f"{namespace}__COMM_L{level}_{comm_idx:04d}",
            namespace=namespace,
            level=level,
            parent_community_id=None,
            child_community_ids=[],
            title=title,
            entity_ids=ent_ids,
            relationship_ids=rel_ids,
            source_chunk_ids=list(dict.fromkeys(chunk_ids)),
            source_document_ids=list(dict.fromkeys(doc_ids)),
            size=len(ent_ids),
        )

    def _generate_community_title(
        self,
        node_ids: List[str],
        ent_by_id: Dict[str, CanonicalEntity],
        level: int,
        idx: int,
    ) -> str:
        names = [ent_by_id[nid].canonical_name for nid in node_ids if nid in ent_by_id]
        lead_names = names[:3]
        prefix = "Cụm Vĩ mô" if level == 0 else ("Cụm Trung cấp" if level == 1 else "Cụm Vi mô")
        if lead_names:
            return f"{prefix} #{idx + 1}: {', '.join(lead_names)}"
        return f"{prefix} #{idx + 1}"

    def _get_subgraph_relation_ids(
        self,
        subgraph: nx.Graph,
        rel_by_pair: Dict[Tuple[str, str], List[CanonicalRelationship]],
    ) -> List[str]:
        rel_ids: List[str] = []
        for u, v in subgraph.edges():
            pair = tuple(sorted([u, v]))
            if pair in rel_by_pair:
                for r in rel_by_pair[pair]:
                    rel_ids.append(r.relationship_id)
        return list(dict.fromkeys(rel_ids))

    def _aggregate_chunk_ids(
        self,
        node_ids: List[str],
        rel_ids: List[str],
        ent_by_id: Dict[str, CanonicalEntity],
        all_relationships: List[CanonicalRelationship],
    ) -> List[str]:
        chunk_ids: List[str] = []
        for nid in node_ids:
            if nid in ent_by_id:
                chunk_ids.extend(ent_by_id[nid].source_chunk_ids)

        rel_by_id = {r.relationship_id: r for r in all_relationships}
        for rid in rel_ids:
            if rid in rel_by_id:
                chunk_ids.extend(rel_by_id[rid].source_chunk_ids)

        return list(dict.fromkeys(chunk_ids))

    def _aggregate_doc_ids(
        self,
        node_ids: List[str],
        ent_by_id: Dict[str, CanonicalEntity],
    ) -> List[str]:
        doc_ids: List[str] = []
        for nid in node_ids:
            if nid in ent_by_id:
                doc_ids.extend(ent_by_id[nid].source_document_ids)
        return list(dict.fromkeys(doc_ids))
