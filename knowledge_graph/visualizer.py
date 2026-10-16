from typing import Optional, List, Set, Dict, Any
import networkx as nx
from pyvis.network import Network


RELATION_COLORS = {
    "AMENDS": "#ef4444",       # Red
    "SUPPLEMENTS": "#f97316",  # Orange
    "REPEALS": "#dc2626",      # Dark Red
    "DETAILS": "#38bdf8",      # Sky Blue
    "REFERENCES": "#a855f7",   # Purple
    "CONTAINS": "#64748b",     # Slate Gray
}


def build_pyvis_html(
    graph: nx.MultiDiGraph,
    center_node_id: Optional[str] = None,
    highlight_node_ids: Optional[List[str]] = None,
    max_hops: int = 1,
    allowed_relations: Optional[List[str]] = None,
    height: str = "520px",
    width: str = "100%",
    physics: bool = True,
) -> str:
    allowed_rels = set(allowed_relations) if allowed_relations else set(RELATION_COLORS.keys())

    # 1. Determine nodes to include
    sub_nodes: Set[str] = set()
    if highlight_node_ids:
        sub_nodes.update([nid for nid in highlight_node_ids if graph.has_node(nid)])
    if center_node_id and graph.has_node(center_node_id):
        sub_nodes.add(center_node_id)

    if sub_nodes:
        current_frontier = set(sub_nodes)
        for _ in range(max_hops):
            next_frontier = set()
            for n in current_frontier:
                # Outgoing
                for _, tgt, d in graph.out_edges(n, data=True):
                    if d.get("relation_type") in allowed_rels:
                        next_frontier.add(tgt)
                # Incoming
                for src, _, d in graph.in_edges(n, data=True):
                    if d.get("relation_type") in allowed_rels:
                        next_frontier.add(src)
            sub_nodes.update(next_frontier)
            current_frontier = next_frontier
    else:
        # Default top connected nodes if no center node provided
        sub_nodes = set(list(graph.nodes())[:40])

    # 2. Build pyvis network
    net = Network(
        height=height,
        width=width,
        bgcolor="#0e1117",
        font_color="#ffffff",
        directed=True,
        cdn_resources="remote",
    )

    if physics:
        net.barnes_hut(
            gravity=-3500,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.05,
            damping=0.9,
        )
    else:
        net.toggle_physics(False)

    # 3. Add nodes
    for node_id in sub_nodes:
        if not graph.has_node(node_id):
            continue
        ndata = graph.nodes[node_id]
        ntype = ndata.get("node_type", "PROVISION")
        raw_title = ndata.get("title") or node_id
        # Truncate label for neat display
        label = raw_title if len(raw_title) <= 28 else raw_title[:25] + "..."

        tooltip_lines = [
            f"<b>{raw_title}</b>",
            f"<b>Loại:</b> {ntype}",
        ]
        if ndata.get("structural_path"):
            tooltip_lines.append(f"<b>Đường dẫn:</b> {ndata['structural_path']}")
        if ndata.get("effective_start"):
            tooltip_lines.append(f"<b>Hiệu lực từ:</b> {ndata['effective_start']}")
        if ndata.get("status"):
            tooltip_lines.append(f"<b>Trạng thái:</b> {ndata['status']}")
        if ndata.get("content"):
            content_snippet = ndata['content'][:250].replace('\n', '<br>') + "..."
            tooltip_lines.append(f"<hr><b>Nội dung trích đoạn:</b><br>{content_snippet}")

        tooltip = "<br>".join(tooltip_lines)

        # Node styling based on role
        if node_id == center_node_id:
            color = "#f59e0b"  # Amber gold for central selected node
            size = 32
            shape = "dot"
            border_width = 4
        elif highlight_node_ids and node_id in highlight_node_ids:
            color = "#10b981"  # Emerald green for other active retrieved results
            size = 25
            shape = "dot"
            border_width = 3
        elif ntype == "DOCUMENT":
            color = "#3b82f6"  # Royal blue for legal document
            size = 26
            shape = "diamond"
            border_width = 2
        elif "AMEND" in ndata.get("status", "") or "sửa đổi" in raw_title.lower():
            color = "#ef4444"  # Red for amended / amending provisions
            size = 22
            shape = "triangle"
            border_width = 2
        else:
            color = "#a855f7"  # Purple for KG expanded neighbor provisions
            size = 20
            shape = "dot"
            border_width = 2

        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color={"background": color, "border": "#ffffff", "highlight": {"background": "#fbbf24", "border": "#ffffff"}},
            size=size,
            shape=shape,
            borderWidth=border_width,
            font={"size": 13, "color": "#ffffff", "face": "sans-serif"},
        )

    # 4. Add edges between included nodes
    edge_counts: Dict[tuple, int] = {}
    for src in sub_nodes:
        for _, tgt, d in graph.out_edges(src, data=True):
            if tgt not in sub_nodes:
                continue
            rel = d.get("relation_type", "")
            if rel not in allowed_rels:
                continue

            edge_key = (src, tgt)
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
            idx = edge_counts[edge_key]

            color = RELATION_COLORS.get(rel, "#94a3b8")
            evidence = d.get("evidence_text") or ""
            edge_tooltip = f"<b>Quan hệ:</b> {rel}<br><b>Căn cứ:</b> {evidence}"
            if d.get("effective_date"):
                edge_tooltip += f"<br><b>Ngày hiệu lực:</b> {d['effective_date']}"

            # Smooth curve so parallel edges don't overlap
            smooth_opt = {"type": "curvedCW", "roundness": 0.15 * idx}

            net.add_edge(
                src,
                tgt,
                label=rel,
                title=edge_tooltip,
                color={"color": color, "highlight": "#ffffff"},
                arrows="to",
                width=2,
                font={"size": 10, "color": color, "align": "middle", "strokeWidth": 2, "strokeColor": "#0e1117"},
                smooth=smooth_opt,
            )

    return net.generate_html()
