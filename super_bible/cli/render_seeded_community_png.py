#!/usr/bin/env python3
# coding: utf-8

"""
Render seeded local community detection results to PNGs (undirected).

It expects the JSON produced by `seeded_local_community.py` which includes:
  - ego_nodes: list[str]
  - community_nodes: list[str]

It then loads the verse graph (NetworkX DiGraph) from:
  - --graph (default: out/crossref_graph.gpickle), or
  - rebuilds from --csv if --graph is missing (opt-in).

Rendering:
  - Builds undirected induced subgraphs on ego/ community node sets.
  - Writes DOT and calls Graphviz `dot` to generate PNG.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from super_bible import paths


def _require_networkx():
    try:
        import networkx as nx  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("networkx is required. Install with `pip install -r requirements.txt`.") from e
    return nx


def _optional_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    return plt


def _escape_dot_label(s: str) -> str:
    # DOT: wrap labels in quotes; escape backslash + quotes.
    return s.replace("\\", "\\\\").replace('"', '\\"')


def induced_undirected_subgraph(g_di: "object", nodes: Iterable[str]) -> "object":
    nx = _require_networkx()
    nset = set(nodes)
    sub = g_di.subgraph(nset)

    # Build a simple undirected graph, preserving an `abs_votes` edge attribute
    # (from the directed edge with the maximum |Votes| within the induced set).
    h = nx.Graph()
    h.add_nodes_from(nset)

    # key = frozenset({u,v}); store max abs votes and signed votes for that edge.
    edge_best: dict[frozenset[str], tuple[float, float]] = {}
    for u, v, data in sub.edges(data=True):
        if u == v:
            continue
        w = data.get("weight", 0)
        try:
            w = float(w)
        except Exception:
            w = 0.0
        key = frozenset((u, v))
        abs_w = abs(w)
        prev = edge_best.get(key)
        if prev is None or abs_w > prev[0]:
            # store (abs_w, signed_w)
            edge_best[key] = (abs_w, w)

    for key, (abs_w, signed_w) in edge_best.items():
        u, v = tuple(key)
        h.add_edge(u, v, abs_votes=abs_w, votes=signed_w)

    return h


def render_with_matplotlib(
    g_und: "object",
    *,
    node_list: Sequence[str],
    out_png: Path,
    title: Optional[str] = None,
    node_color: str = "#1e90ff",
    edge_color: str = "#888888",
    node_size: int = 60,
    edge_alpha: float = 0.35,
    seed: int = 42,
    canvas_inches: float = 8.0,
    edge_width: float = 2.5,
    edge_width_min: float = 0.8,
    spring_k: Optional[float] = 0.75,
    spring_scale: float = 3.2,
    spring_iterations: int = 450,
) -> None:
    """
    Render undirected graph to PNG using matplotlib + networkx layouts.
    This avoids Graphviz "long and thin" aspect issues.
    """
    plt = _optional_import_matplotlib()
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib not available")

    nx = _require_networkx()

    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Force deterministic layout for stable visuals, and spread nodes out.
    pos = nx.spring_layout(
        g_und,
        seed=seed,
        k=spring_k,
        scale=spring_scale,
        iterations=spring_iterations,
    )

    fig = plt.figure(figsize=(canvas_inches, canvas_inches), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # Draw edges with curvature + vote-based styling.
    edges = list(g_und.edges())
    abs_vals: List[float] = []
    for u, v in edges:
        abs_votes = g_und.edges[u, v].get("abs_votes", 0.0)
        try:
            abs_votes = float(abs_votes)
        except Exception:
            abs_votes = 0.0
        abs_vals.append(abs_votes)

    # Map abs_votes -> t in [0,1] using log1p scaling.
    import math

    nonzero = [v for v in abs_vals if v > 0]
    if not nonzero:
        t_vals = [0.0 for _ in abs_vals]
    else:
        min_v = min(nonzero)
        max_v = max(nonzero)
        log_min = math.log1p(min_v)
        log_max = math.log1p(max_v)
        denom = (log_max - log_min) if log_max > log_min else 1.0
        t_vals = [(math.log1p(v) - log_min) / denom for v in abs_vals]

    # Black -> Red gradient.
    orange_rgb = (1.0, 0.0, 0.0)
    black_rgb = (0.0, 0.0, 0.0)
    edge_colors: List[str] = []
    for t in t_vals:
        t = max(0.0, min(1.0, float(t)))
        r = black_rgb[0] * (1.0 - t) + orange_rgb[0] * t
        g = black_rgb[1] * (1.0 - t) + orange_rgb[1] * t
        b = black_rgb[2] * (1.0 - t) + orange_rgb[2] * t
        edge_colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")

    # Curved edges: use FancyArrowPatch with arc3.
    from matplotlib.patches import FancyArrowPatch  # type: ignore

    rad = 0.15
    for (u, v), t, color in zip(edges, t_vals, edge_colors):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Thickness changes with color intensity (same log-scaled t).
        t_clamped = max(0.0, min(1.0, float(t)))
        w = edge_width_min + (edge_width - edge_width_min) * t_clamped

        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-",
            mutation_scale=10,
            linewidth=w,
            color=color,
            alpha=edge_alpha,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(patch)

    # Nodes: no labels. Bright solid blue.
    # Draw in node_list order so the seed/community nodes are stable in styling.
    node_set = set(node_list)
    ordered_nodes = [n for n in node_list if n in node_set and n in g_und]
    # If for some reason ordering excludes everything, fall back to all nodes.
    if not ordered_nodes:
        ordered_nodes = list(g_und.nodes())

    nx.draw_networkx_nodes(
        g_und,
        pos,
        nodelist=ordered_nodes,
        ax=ax,
        node_color=node_color,
        node_size=node_size,
        linewidths=0.0,
    )

    if title:
        ax.set_title(title)

    # Crop away surrounding whitespace.
    fig.savefig(out_png, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_dot_for_undirected_graph(
    g_und: "object",
    *,
    node_list: Sequence[str],
    show_labels: bool = False,
    node_fillcolor: str = "#1e90ff",
    canvas_size: str = "8,8",
) -> str:
    """
    Create a DOT graph string for an undirected graph.
    `node_list` controls which nodes appear (and in what order).
    """
    # We don't need to import networkx just to build DOT string,
    # but we rely on `.edges()` and `.degree()`.
    lines: List[str] = []
    lines.append("graph G {")
    lines.append(f'  size="{canvas_size}";')
    lines.append('  overlap=false;')
    lines.append('  splines=true;')
    lines.append('  ratio=compress;')
    lines.append(
        f'  node [shape=circle, style=filled, fillcolor="{node_fillcolor}", color="{node_fillcolor}", '
        'fontname="Helvetica", fontsize=10, fixedsize=true, width=0.25, height=0.25, fontcolor="white"];'
    )

    # Nodes
    node_set = set(node_list)
    for n in node_list:
        if n not in node_set:
            continue
        if show_labels:
            label = _escape_dot_label(n)
            lines.append(f'  "{n}" [label="{label}"];')
        else:
            lines.append(f'  "{n}" [label=""];')

    # Edges (undirected)
    for u, v in g_und.edges():
        if u == v:
            continue
        if u not in node_set or v not in node_set:
            continue
        lines.append(f'  "{u}" -- "{v}";')

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_png_from_dot(dot_text: str, out_png: Path) -> None:
    # Use force-directed layout to avoid "long and thin" bounding boxes.
    neato_bin = shutil.which("neato")
    if not neato_bin:
        raise RuntimeError("Graphviz `neato` not found on PATH. Install graphviz.")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_png.with_suffix(".dot")
    dot_path.write_text(dot_text, encoding="utf-8")

    # `-Goverlap=false` helps prevent node collisions.
    # `-Gsize=...` is not always honored, but `neato` generally gives better aspect.
    cmd = [neato_bin, "-Tpng", str(dot_path), "-o", str(out_png), "-Goverlap=false"]
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_graph_pickle(path: Path) -> "object":
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render seeded local community JSON to PNG via Graphviz.")
    parser.add_argument("--json", type=str, required=True, help="Path to community JSON.")
    parser.add_argument(
        "--graph",
        type=str,
        default="out/crossref_graph.gpickle",
        help="Path to pickled NetworkX DiGraph.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(paths.CROSS_REFERENCES_CSV),
        help="CSV path (used only when --graph missing and --rebuild-from-csv set). Default: data/cross_references.csv.",
    )
    parser.add_argument(
        "--rebuild-from-csv",
        action="store_true",
        help="If --graph is missing, rebuild from --csv (can be slow).",
    )
    parser.add_argument("--out-dir", type=str, default="out", help="Output directory for PNG files.")
    parser.add_argument(
        "--renderer",
        type=str,
        default="auto",
        choices=["auto", "matplotlib", "graphviz"],
        help="PNG renderer. auto=matplotlib if available, else graphviz.",
    )
    parser.add_argument(
        "--canvas",
        type=float,
        default=8.0,
        help="Square canvas size in inches for matplotlib rendering.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    json_path = Path(args.json).resolve()
    out_dir = Path(args.out_dir).resolve()
    graph_path = Path(args.graph).resolve()
    csv_path = Path(args.csv).resolve()

    data = load_json(json_path)
    ego_nodes = data.get("ego_nodes", [])
    community_nodes = data.get("community_nodes", [])

    if not ego_nodes and not community_nodes:
        raise ValueError("JSON does not contain ego_nodes or community_nodes.")

    if graph_path.exists():
        g_di = load_graph_pickle(graph_path)
    else:
        if not args.rebuild_from_csv:
            raise FileNotFoundError(f"Graph pickle not found: {graph_path}. Use --rebuild-from-csv to rebuild.")
        from super_bible.cli.esv_crossref_graph import build_digraph_from_csv, save_digraph_pickle

        g_di = build_digraph_from_csv(csv_path, expand_ranges=True, dedupe="max_abs_votes")
        # Persist so subsequent runs don’t need to rebuild.
        try:
            save_digraph_pickle(g_di, graph_path)
        except Exception:
            # Fail-soft: rendering can still proceed even if we can't save.
            pass

    renderer = args.renderer
    if renderer == "auto":
        renderer = "matplotlib" if _optional_import_matplotlib() is not None else "graphviz"

    # ego
    if ego_nodes:
        g_ego_und = induced_undirected_subgraph(g_di, ego_nodes)
        out_png = out_dir / f"{json_path.stem}_ego.png"
        if renderer == "matplotlib":
            render_with_matplotlib(
                g_ego_und,
                node_list=list(ego_nodes),
                out_png=out_png,
                canvas_inches=args.canvas,
            )
        else:
            dot_ego = build_dot_for_undirected_graph(g_ego_und, node_list=list(ego_nodes))
            write_png_from_dot(dot_ego, out_png)

    # community
    if community_nodes:
        g_comm_und = induced_undirected_subgraph(g_di, community_nodes)
        out_png = out_dir / f"{json_path.stem}_community.png"
        if renderer == "matplotlib":
            render_with_matplotlib(
                g_comm_und,
                node_list=list(community_nodes),
                out_png=out_png,
                canvas_inches=args.canvas,
            )
        else:
            dot_comm = build_dot_for_undirected_graph(g_comm_und, node_list=list(community_nodes))
            write_png_from_dot(dot_comm, out_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

