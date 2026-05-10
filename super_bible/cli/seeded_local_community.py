#!/usr/bin/env python3
# coding: utf-8

"""
Seeded local community detection on a verse cross-reference graph.

Inputs
  - A directed NetworkX DiGraph where nodes are verse ids (e.g. "Gen.1.1")
  - Optional: build the graph from `cross_references.csv` via
    `esv_crossref_graph.py`

Algorithm (k-hop=1 ego)
  1. Ego node set: seed + in-neighbors(seed) + out-neighbors(seed)
  2. Build ego subgraph(s):
       - EgoDi: directed induced subgraph on ego nodes
       - EgoUnd: undirected induced graph on ego nodes
  3. Densest subgraph containing the seed within EgoUnd:
       - Min-degree peeling while never removing the seed
       - Track max density subsets; tie-break by smaller |V|
  4. Compute ranks within the densest subset:
       - k-core (k-core number) on EgoUnd restricted to subset
       - k-truss membership: max k where node appears in nx.k_truss(subset,k)
       - local clustering coefficient + triangle_count on EgoUnd restricted to subset
       - personalized PageRank on EgoDi restricted to subset

Note:
  - Community metrics are computed on an undirected graph, because k-core/k-truss/
    clustering/triangles are defined there.
  - The input edges may be weighted; this implementation treats the graph as
    unweighted for community metrics. PageRank is also unweighted here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote

from super_bible import paths


def _require_networkx():
    try:
        import networkx as nx  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "networkx is required. Install with `pip install -r requirements.txt`."
        ) from e
    return nx


_BOOK_CHAPTER_RE = re.compile(
    r"^\s*(?P<book>[A-Za-z0-9]+)\s+"
    r"(?P<chapter>\d+)\s*:\s*"
    r"(?P<verses>\d+\s*(?:-\s*\d+)?)\s*$"
)

_VERSE_RANGE_RE = re.compile(r"^(?P<a>\d+)\s*-\s*(?P<b>\d+)$")


def _parse_verses_part(verses_part: str) -> List[int]:
    verses_part = verses_part.strip()
    m = _VERSE_RANGE_RE.match(verses_part)
    if m:
        a = int(m.group("a"))
        b = int(m.group("b"))
        if b < a:
            raise ValueError("invalid verse range: end < start")
        return list(range(a, b + 1))
    return [int(verses_part)]


def parse_seed_reference_to_nodes(seed_ref: str, *, graph_nodes: Iterable[str]) -> Set[str]:
    """
    Parse a seed reference string into verse-node ids that exist in `graph_nodes`.

    Supported examples:
      - "John 1:1-4"
      - "John 1:1-4, 14"   (comma parts without ':' reuse prior chapter+book)
      - "John 1:14"

    Strict matching:
      - book token must match the node prefix used in the graph, i.e. nodes like
        "<Book>.<chapter>.<verse>".
      - We only return nodes that exist in `graph_nodes`.
    """
    nodes_set = set(graph_nodes)
    seed_ref = seed_ref.strip()
    if not seed_ref:
        raise ValueError("seed reference is empty")

    parts = [p.strip() for p in seed_ref.split(",") if p.strip()]
    if not parts:
        raise ValueError("seed reference parsed to no parts")

    book: Optional[str] = None
    chapter: Optional[int] = None

    out: Set[str] = set()
    for part in parts:
        if ":" in part:
            m = _BOOK_CHAPTER_RE.match(part)
            if not m:
                raise ValueError(f"invalid seed part: '{part}'")
            book = m.group("book").strip()
            chapter = int(m.group("chapter"))
            verses_part = m.group("verses")
            verses = _parse_verses_part(verses_part)
        else:
            # Token like "14" (no book/chapter) -> reuse last book+chapter.
            if book is None or chapter is None:
                raise ValueError(
                    f"seed part '{part}' missing ':' and cannot reuse book/chapter yet"
                )
            verses = _parse_verses_part(part)

        assert book is not None
        assert chapter is not None

        for v in verses:
            node_id = f"{book}.{chapter}.{v}"
            if node_id in nodes_set:
                out.add(node_id)

    if not out:
        raise ValueError(
            f"seed reference '{seed_ref}' did not match any verse nodes in the graph"
        )
    return out


def ego_nodes_k1(g_di: "object", seed: str) -> Set[str]:
    """
    k-hop=1 ego node set including:
      - seed
      - predecessors(seed)
      - successors(seed)
    """
    nx = _require_networkx()
    if seed not in g_di:
        raise KeyError(f"seed '{seed}' not present in graph")
    ego: Set[str] = {seed}
    ego.update(g_di.predecessors(seed))
    ego.update(g_di.successors(seed))
    return ego


def ego_nodes_k1_multi(g_di: "object", seeds: Set[str]) -> Set[str]:
    """k-hop=1 ego node set for multiple seed nodes."""
    nx = _require_networkx()
    missing = [s for s in seeds if s not in g_di]
    if missing:
        raise KeyError(f"seed nodes not present in graph: {sorted(missing)[:10]}")
    ego: Set[str] = set(seeds)
    for s in seeds:
        ego.update(g_di.predecessors(s))
        ego.update(g_di.successors(s))
    return ego


def densest_subgraph_containing_any_seed(g_und: "object", seeds: Set[str]) -> Set[str]:
    """
    Densest subgraph containing at least one seed by min-degree peeling with tie-break.

    Density definition (undirected):
        density(S) = 2|E(S)| / |V(S)|
    """
    nx = _require_networkx()
    missing = [s for s in seeds if s not in g_und]
    if missing:
        raise KeyError(f"seed nodes not present in undirected graph: {sorted(missing)[:10]}")

    remaining: Set[str] = set(g_und.nodes())
    best_nodes: Set[str] = set(remaining)
    best_density: float = -1.0

    # Peeling while we can remove some non-seed node.
    while True:
        # Compute density for current remaining.
        sub = g_und.subgraph(remaining)
        v = sub.number_of_nodes()
        if v == 0:
            break
        e = sub.number_of_edges()
        density = (2.0 * e) / float(v)

        if density > best_density or (density == best_density and len(remaining) < len(best_nodes)):
            best_density = density
            best_nodes = set(remaining)

        seed_remaining = remaining & seeds
        if not seed_remaining:
            break

        # We can remove nodes as long as at least one seed remains.
        if len(seed_remaining) == 1:
            # Keep the single last seed; remove everyone else.
            removable = remaining - seed_remaining
        else:
            # Multiple seeds remain; removal can include some seeds.
            removable = set(remaining)

        if not removable:
            break

        # Remove a min-degree node among removable nodes.
        # Determinism: tie-break by node id string.
        degrees = {n: sub.degree(n) for n in removable}
        node_to_remove = min(sorted(removable), key=lambda n: degrees[n])
        remaining.remove(node_to_remove)

    return best_nodes


def _max_k_truss_membership(g_und: "object", nodes: Iterable[str]) -> Dict[str, int]:
    """
    For each node in `nodes`, compute max k such that node appears in nx.k_truss(g_und,k).
    """
    nx = _require_networkx()
    result: Dict[str, int] = {n: 0 for n in nodes}

    # Upper bound: k can't exceed number of nodes, but we stop earlier when no edges remain.
    max_k = max(2, g_und.number_of_nodes())
    last_nonempty_k = 0
    for k in range(2, max_k + 1):
        try:
            h = nx.k_truss(g_und, k)
        except nx.NetworkXError:
            break
        if h.number_of_edges() == 0:
            break
        last_nonempty_k = k
        for n in nodes:
            if n in h:
                result[n] = k

    # Nodes not present in any k-truss keep 0.
    return result


def compute_seeded_local_community(
    g_di: "object",
    seed: "str | Set[str]",
    *,
    alpha: float = 0.85,
    prune_degree_threshold: int = 2,
) -> Dict[str, object]:
    """
    Returns a JSON-serializable dict:
      - ego_nodes
      - community_nodes (densest subset containing seed)
      - ranks: per-node metrics
      - sorted_by: ordering lists per metric
    """
    nx = _require_networkx()

    seeds: Set[str] = {seed} if isinstance(seed, str) else set(seed)

    ego = ego_nodes_k1_multi(g_di, seeds)
    ego_di = g_di.subgraph(ego).copy()
    ego_und = nx.Graph(ego_di)  # ignore directions; keeps simple adjacency
    # NetworkX core/clustering/truss algorithms assume simple graphs and reject
    # self-loops for core/truss. Remove them deterministically.
    ego_und.remove_edges_from(nx.selfloop_edges(ego_und))

    # Prune low-interconnectivity nodes (undirected) from the ego graph.
    # Keep seed nodes even if they fall below the threshold.
    if prune_degree_threshold > 0 and ego_und.number_of_nodes() > 0:
        pruned_ego = {
            n for n in ego
            if n in seeds or ego_und.degree(n) >= prune_degree_threshold
        }
        ego = pruned_ego
        # Rebuild ego subgraphs after pruning to keep degrees/edges consistent.
        ego_di = g_di.subgraph(ego).copy()
        ego_und = nx.Graph(ego_di)
        ego_und.remove_edges_from(nx.selfloop_edges(ego_und))

    community = densest_subgraph_containing_any_seed(ego_und, seeds)
    comm_di = ego_di.subgraph(community).copy()
    comm_und = ego_und.subgraph(community).copy()
    comm_und.remove_edges_from(nx.selfloop_edges(comm_und))

    # k-core number
    if comm_und.number_of_nodes() > 0 and comm_und.number_of_edges() > 0:
        core_number = nx.core_number(comm_und)
    else:
        core_number = {n: 0 for n in community}

    # k-truss membership
    truss_number = _max_k_truss_membership(comm_und, community)

    # Clustering + triangles
    clustering = nx.clustering(comm_und) if comm_und.number_of_nodes() > 0 else {n: 0.0 for n in community}
    triangles = nx.triangles(comm_und) if comm_und.number_of_nodes() > 0 else {n: 0 for n in community}

    # Personalized PageRank (unweighted; directed)
    personalization = {n: 0.0 for n in comm_di.nodes()}
    valid_seeds = [s for s in seeds if s in comm_di]
    if not valid_seeds:
        # Fail-soft: leave personalization all zeros -> networkx will error.
        # But densest selection should guarantee at least one seed remains.
        valid_seeds = []

    if valid_seeds:
        w = 1.0 / float(len(valid_seeds))
        for s in valid_seeds:
            personalization[s] = w
    if len(comm_di) == 0:
        ppr = {}
    else:
        ppr = nx.pagerank(comm_di, alpha=alpha, personalization=personalization, weight=None)

    ranks: Dict[str, Dict[str, float]] = {}
    for n in community:
        ranks[n] = {
            "k_core": float(core_number.get(n, 0)),
            "k_truss": float(truss_number.get(n, 0)),
            "local_clustering": float(clustering.get(n, 0.0)),
            "triangle_count": float(triangles.get(n, 0)),
            "personalized_pagerank": float(ppr.get(n, 0.0)),
        }

    def _sort_nodes(metric: str, reverse: bool = True) -> List[str]:
        return sorted(community, key=lambda n: (ranks[n][metric], n), reverse=reverse)

    sorted_by = {
        "k_core": _sort_nodes("k_core"),
        "k_truss": _sort_nodes("k_truss"),
        "local_clustering": _sort_nodes("local_clustering"),
        "triangle_count": _sort_nodes("triangle_count"),
        "personalized_pagerank": _sort_nodes("personalized_pagerank"),
    }

    return {
        "seed_nodes": sorted(seeds),
        "ego_nodes": sorted(ego),
        "community_nodes": sorted(community),
        "ranks": ranks,
        "sorted_by": sorted_by,
        "params": {"alpha": alpha},
        "prune_degree_threshold": prune_degree_threshold,
    }


def top_k_metrics_rows(
    result: Dict[str, object],
    *,
    k: int = 15,
) -> List[Tuple[str, int, str, float]]:
    """
    For each metric in `sorted_by`, take top-k nodes and emit:
      (metric_name, rank_1_based, node_id, score)
    """
    sorted_by = result.get("sorted_by", {})  # type: ignore[assignment]
    ranks = result.get("ranks", {})  # type: ignore[assignment]
    if not isinstance(sorted_by, dict) or not isinstance(ranks, dict):
        return []
    if k <= 0:
        return []

    rows: List[Tuple[str, int, str, float]] = []
    for metric, nodes in sorted_by.items():
        if not isinstance(nodes, list) or not nodes:
            continue
        # `nodes` are already sorted by the metric score (descending).
        for idx, node in enumerate(nodes[:k], start=1):
            if not isinstance(node, str):
                continue
            node_rank = ranks.get(node, {})
            if not isinstance(node_rank, dict):
                continue
            score = node_rank.get(metric)
            if not isinstance(score, (int, float)):
                continue
            rows.append((metric, idx, node, float(score)))
    return rows


def _parse_node_id(node_id: str) -> Tuple[str, int, int]:
    """
    Node ids are like: <BookToken>.<chapter>.<verse> (e.g. John.1.14, 1Kgs.8.27).
    """
    parts = node_id.split(".")
    if len(parts) < 3:
        raise ValueError(f"invalid node_id: {node_id}")
    book_token = ".".join(parts[:-2])
    chapter = int(parts[-2])
    verse = int(parts[-1])
    return book_token, chapter, verse


_EN_BOOK_MAPPING_CACHE: Optional[Dict[str, str]] = None


def load_en_book_mapping() -> Dict[str, str]:
    """
    Map OsisID -> BookName from `.zraw_metadata/EN_book_index.txt`.
    Example: Gen -> Genesis, Ps -> Psalms, John -> John, 1Kgs -> 1 Kings.
    """
    global _EN_BOOK_MAPPING_CACHE
    if _EN_BOOK_MAPPING_CACHE is not None:
        return _EN_BOOK_MAPPING_CACHE

    idx_path = paths.ZRAW_METADATA_DIR / "EN_book_index.txt"
    mapping: Dict[str, str] = {}
    with idx_path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        osis_id = parts[1]
        book_name = parts[2]
        mapping[osis_id] = book_name

    _EN_BOOK_MAPPING_CACHE = mapping
    return mapping


def get_verse_text(
    db_path: Path,
    *,
    version: str,
    book_title: str,
    chapter: int,
    verse: int,
) -> str:
    # Reuse consistent parsing/SQL behavior from `esv_query.py`.
    from super_bible.cli.esv_query import query_bible

    rows = query_bible(
        db_path=db_path,
        version=version,
        book=book_title,
        chapter=chapter,
        start_verse=verse,
        end_verse=verse,
    )
    return str(rows[0][1]) if rows else ""


def write_top_metrics_csv(
    result: Dict[str, object],
    out_csv: Path,
    *,
    version: str,
    k: int = 15,
) -> None:
    rows = top_k_metrics_rows(result, k=k)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    db_path = paths.SUPER_BIBLE_DB_PATH

    # Lazy-load mapping and cache verse text.
    en_book_map = load_en_book_mapping()
    verse_cache: Dict[Tuple[str, int, int], str] = {}

    def node_id_to_book_title_and_chv(node_id: str) -> Tuple[str, int, int]:
        book_token, chapter, verse = _parse_node_id(node_id)
        book_title = en_book_map.get(book_token, book_token)
        return book_title, chapter, verse

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "rank", "score", "citation", "verse", "url"])

        for metric, rank, node_id, score in rows:
            book_title, chapter, verse = node_id_to_book_title_and_chv(node_id)
            citation = f"{book_title} {chapter}:{verse}"
            url = f"https://www.biblegateway.com/passage/?search={quote(citation)}&version={version}"

            cache_key = (book_title, chapter, verse, version)
            verse_text = verse_cache.get(cache_key)
            if verse_text is None:
                verse_text = get_verse_text(
                    db_path,
                    version=version,
                    book_title=book_title,
                    chapter=chapter,
                    verse=verse,
                )
                verse_cache[cache_key] = verse_text

            w.writerow([metric, rank, score, citation, verse_text, url])


def _load_graph_pickle(path: Path) -> "object":
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Seeded local community detection")
    parser.add_argument("--seed", type=str, required=True, help="Seed verse id (e.g. Gen.1.1)")
    parser.add_argument(
        "--graph",
        type=str,
        default=str(paths.CROSSREF_GRAPH_PICKLE_PATH),
        help="Pickled NetworkX DiGraph built from cross_references.csv",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(paths.CROSS_REFERENCES_CSV),
        help="Fallback CSV used to rebuild the graph if --graph is missing.",
    )
    parser.add_argument("--alpha", type=float, default=0.85, help="PageRank damping factor")
    parser.add_argument("--out", type=str, default="out/seed_community.json", help="Output JSON path")
    parser.add_argument("--version", type=str, default="ESV", help="Bible translation version (default: ESV).")
    parser.set_defaults(clear_out=True)
    parser.add_argument(
        "--no-clear-out",
        action="store_false",
        dest="clear_out",
        help="Do not clear generated files in out/ before running.",
    )
    parser.add_argument(
        "-T",
        "--T",
        type=int,
        default=2,
        help="Prune ego nodes with undirected degree < T (T=0 disables). Seeds are always kept.",
    )
    # Back-compat alias (if someone used the older flag name).
    parser.add_argument(
        "--prune-degree-threshold",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Disable ego pruning (equivalent to --prune-degree-threshold 0).",
    )
    parser.add_argument(
        "--no-render-png",
        action="store_true",
        help="Disable generating ego/community PNGs into out/ after writing the JSON.",
    )
    parser.add_argument(
        "--no-top-csv",
        action="store_true",
        help="Disable generating top-metric CSV into out/ after writing the JSON.",
    )
    parser.add_argument(
        "--no-nlp-theme",
        action="store_true",
        help="Disable NLP theme scoring (TF-IDF/LDA metrics) vs seeds.",
    )
    parser.add_argument(
        "--no-nlp-pareto",
        action="store_true",
        help="Disable recompute Pareto front using NLP aggregated across seeds.",
    )
    parser.add_argument(
        "--theme-workers",
        type=int,
        default=24,
        help="Parallel workers for theme scoring (default 24).",
    )
    parser.add_argument(
        "--theme-top-n-pareto",
        type=int,
        default=50,
        help="Top N Pareto-front candidates for theme scoring (default 50).",
    )
    parser.add_argument(
        "--expand-ranges",
        action="store_true",
        help="When rebuilding from CSV, expand same-book/chapter ranges (default in builder).",
    )
    parser.add_argument(
        "--no-expand-ranges",
        action="store_true",
        help="When rebuilding from CSV, do not expand same-book/chapter range tokens.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from super_bible.cli.esv_query import normalize_version

    version_n = normalize_version(args.version)

    graph_path = Path(args.graph).resolve()
    csv_path = Path(args.csv).resolve()
    out_path = Path(args.out).resolve()
    out_dir = out_path.parent

    # Standard run: clear previous outputs from `out/` so each run is clean.
    if args.clear_out:
        if out_dir.name != "out":
            raise ValueError(
                f"Refusing to clear non-standard out directory: {out_dir}. "
                "Use --no-clear-out if you need to preserve existing files."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in out_dir.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if graph_path.exists():
        g_di = _load_graph_pickle(graph_path)
    else:
        from super_bible.cli.esv_crossref_graph import build_digraph_from_csv
        from super_bible.cli.esv_crossref_graph import save_digraph_pickle

        expand_ranges = True
        if args.no_expand_ranges:
            expand_ranges = False
        if args.expand_ranges:
            expand_ranges = True

        g_di = build_digraph_from_csv(csv_path, expand_ranges=expand_ranges)
        # Persist so later renders/runs can reuse without rebuilding.
        try:
            save_digraph_pickle(g_di, graph_path)
        except Exception:
            # Fail-soft: computation already succeeded; rendering will still work.
            pass

    seed_nodes = parse_seed_reference_to_nodes(args.seed, graph_nodes=g_di.nodes())
    prune_degree_threshold = 0 if args.no_prune else args.T
    # If the suppressed alias is provided, let it override.
    if getattr(args, "prune_degree_threshold", None) is not None:
        prune_degree_threshold = args.prune_degree_threshold
    result = compute_seeded_local_community(
        g_di,
        seed_nodes,
        alpha=args.alpha,
        prune_degree_threshold=prune_degree_threshold,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}", flush=True)

    stem = out_path.stem

    # 1) PNG rendering (undirected)
    if not args.no_render_png:
        # Use same naming convention as render_seeded_community_png.py:
        #   <stem>_ego.png and <stem>_community.png
        try:
            # Temporarily render_seeded_community_png expects out_dir.name as stem,
            # so pass a folder with the desired stem name by writing into a
            # subdir and then moving outputs. To keep this simple and
            # deterministic, we instead render directly with the DOT builder.
            from render_seeded_community_png import (
                build_dot_for_undirected_graph,
                induced_undirected_subgraph,
                write_png_from_dot,
                render_with_matplotlib,
            )

            ego_nodes = result.get("ego_nodes", [])
            community_nodes = result.get("community_nodes", [])
            if isinstance(ego_nodes, list) and ego_nodes:
                g_ego_und = induced_undirected_subgraph(g_di, ego_nodes)
                out_png = out_dir / f"{stem}_ego.png"
                try:
                    render_with_matplotlib(g_ego_und, node_list=list(ego_nodes), out_png=out_png)
                except Exception:
                    dot_ego = build_dot_for_undirected_graph(g_ego_und, node_list=list(ego_nodes))
                    write_png_from_dot(dot_ego, out_png)
            if isinstance(community_nodes, list) and community_nodes:
                g_comm_und = induced_undirected_subgraph(g_di, community_nodes)
                out_png = out_dir / f"{stem}_community.png"
                try:
                    render_with_matplotlib(
                        g_comm_und,
                        node_list=list(community_nodes),
                        out_png=out_png,
                    )
                except Exception:
                    dot_comm = build_dot_for_undirected_graph(
                        g_comm_und, node_list=list(community_nodes)
                    )
                    write_png_from_dot(dot_comm, out_png)
        except Exception as e:  # pragma: no cover
            print(f"warning: png rendering skipped: {e}", flush=True)

    # 2) Top metrics CSV
    if not args.no_top_csv:
        out_csv = out_dir / f"{stem}_top_metrics.csv"
        write_top_metrics_csv(result, out_csv, version=version_n)
        print(f"wrote: {out_csv}", flush=True)

    # 3) Pareto ranking CSV
    try:
        from super_bible.cli.rank_verses_pareto import (
            DEFAULT_METRICS,
            compute_metric_ranks,
            pareto_non_dominated,
            write_pareto_csv,
        )

        eligible_nodes = result.get("community_nodes", [])
        ranks = result.get("ranks", {})
        if isinstance(eligible_nodes, list) and isinstance(ranks, dict):
            eligible_nodes = [n for n in eligible_nodes if isinstance(n, str)]
            metric_ranks = compute_metric_ranks(
                nodes=eligible_nodes, ranks=ranks, metrics=DEFAULT_METRICS
            )
            pareto_nodes = pareto_non_dominated(eligible_nodes, metric_ranks, DEFAULT_METRICS)
            out_pareto = out_dir / f"{stem}_pareto_ranking.csv"
            write_pareto_csv(
                out_csv=out_pareto,
                nodes=eligible_nodes,
                metric_ranks=metric_ranks,
                pareto_nodes=pareto_nodes,
                metrics=DEFAULT_METRICS,
            )
            print(f"wrote: {out_pareto}", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"warning: pareto ranking skipped: {e}", flush=True)

    # 4) NLP theme scoring vs seeds (optional)
    if not args.no_nlp_theme:
        try:
            import subprocess
            import sys

            pareto_csv = out_dir / f"{stem}_pareto_ranking.csv"
            if pareto_csv.exists():
                theme_script = Path(__file__).resolve().parent / "rank_verses_vs_seeds_theme_parallel.py"
                theme_csv_expected = out_dir / f"{stem}_theme_vs_seeds_ranked.csv"
                # Let the script choose its own default output name unless we want
                # to be explicit; we do it explicitly so subsequent step can find it.
                out_theme_csv = theme_csv_expected

                cmd = [
                    sys.executable,
                    str(theme_script),
                    "--community-json",
                    str(out_path),
                    "--pareto-csv",
                    str(pareto_csv),
                    "--version",
                    str(version_n),
                    "--workers",
                    str(args.theme_workers),
                    "--top-n-pareto",
                    str(args.theme_top_n_pareto),
                    "--out",
                    str(out_theme_csv),
                    "--reps-cache-dir",
                    str(out_dir / ".cache"),
                ]
                subprocess.run(cmd, check=True)
                print(f"wrote: {out_theme_csv}", flush=True)
            else:
                print(f"warning: skipped NLP theme scoring; pareto CSV missing: {pareto_csv}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"warning: NLP theme scoring skipped: {e}", flush=True)

    # 5) Pareto recompute using NLP (optional)
    if not args.no_nlp_pareto:
        try:
            import subprocess
            import sys

            theme_csv = out_dir / f"{stem}_theme_vs_seeds_ranked.csv"
            if theme_csv.exists():
                pareto_nlp_script = Path(__file__).resolve().parent / "rank_verses_pareto_with_nlp.py"
                out_pareto_nlp_csv = out_dir / f"{stem}_pareto_ranking_with_nlp.csv"

                cmd = [
                    sys.executable,
                    str(pareto_nlp_script),
                    "--community-json",
                    str(out_path),
                    "--theme-csv",
                    str(theme_csv),
                    "--version",
                    str(version_n),
                    "--out",
                    str(out_pareto_nlp_csv),
                ]
                subprocess.run(cmd, check=True)
                print(f"wrote: {out_pareto_nlp_csv}", flush=True)
            else:
                print(f"warning: skipped pareto-with-nlp; theme CSV missing: {theme_csv}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"warning: pareto-with-nlp skipped: {e}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

