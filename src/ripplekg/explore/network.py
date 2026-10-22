"""Interactive network view of graph edges (pyvis), centred on a company if one is chosen."""

from __future__ import annotations

import re

import pandas as pd
from pyvis.network import Network

from ripplekg.ontology.schema import Ontology

KIND_COLORS = {
    "company": "#4C78A8",
    "facility": "#F58518",
    "material": "#54A24B",
    "component": "#72B7B2",
    "region": "#B279A2",
    "authority": "#E45756",
    "program": "#9D755D",
}


def neighbourhood(edges: pd.DataFrame, focus: str | None, hops: int) -> pd.DataFrame:
    """Edges within `hops` of `focus` (in either direction), or all edges if no focus."""
    if not focus:
        return edges
    frontier, seen = {focus}, {focus}
    for _ in range(hops):
        touching = edges[edges["source_ref"].isin(frontier) | edges["target_ref"].isin(frontier)]
        frontier = (set(touching["source_ref"]) | set(touching["target_ref"])) - seen
        seen |= frontier
    return edges[edges["source_ref"].isin(seen) & edges["target_ref"].isin(seen)]


def network_html(edges: pd.DataFrame, ont: Ontology, focus: str | None = None, height: int = 650) -> str:
    net = Network(
        height=f"{height}px", width="100%", directed=True, cdn_resources="in_line", bgcolor="#ffffff"
    )
    net.barnes_hut(gravity=-6000, spring_length=140)
    nodes = set(edges["source_ref"]) | set(edges["target_ref"])
    degree = pd.concat([edges["source_ref"], edges["target_ref"]]).value_counts()
    for node in nodes:
        n = ont.nodes.get(node)
        kind = n.kind if n else "company"
        label = node.split(":", 1)[-1]
        title = f"{n.name if n else node} ({kind}) · {int(degree.get(node, 0))} edges"
        net.add_node(
            node,
            label=label,
            title=title,
            color=KIND_COLORS.get(kind, "#999999"),
            size=12 + min(int(degree.get(node, 0)), 20),
            borderWidth=4 if node == focus else 1,
        )
    for r in edges.itertuples(index=False):
        net.add_edge(
            r.source_ref,
            r.target_ref,
            title=(
                f"{r.canonical_relation} · {r.triples} triple(s), {r.filings} filing(s), "
                f"{r.first_seen} to {r.last_seen}"
            ),
            label=r.canonical_relation,
            width=1 + min(r.triples, 8) / 2,
            arrows="to",
            font={"size": 9, "align": "middle"},
        )
    return _self_contained(net.generate_html(notebook=False))


_EXTERNAL_TAG = re.compile(
    r"<(?:link|script)[^>]*(?:href|src)=\"(?:\.\./node_modules/|https?://)[^\"]*\"[^>]*>(?:\s*</script>)?",
    re.I,
)


def _self_contained(html: str) -> str:
    """pyvis inlines vis.js but still links Bootstrap from a CDN and two dead local paths; the
    graph needs none of them (they only serve pyvis's optional menus), so drop those tags."""
    return _EXTERNAL_TAG.sub("", html)
