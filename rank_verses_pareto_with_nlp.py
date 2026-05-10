#!/usr/bin/env python3
# coding: utf-8

"""
Recompute Pareto-front ranking using:
  - graph metrics from seeded_local_community JSON (`ranks` field)
  - NLP metrics aggregated across all seed verses using the theme CSV

NLP aggregation:
  - For each candidate node_id, compute avg(rank_score) over all rows/seeds.

Candidate universe:
  - `community_nodes` excluding `seed_nodes` (seed nodes are skipped)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rank_verses_pareto import (
    DEFAULT_METRICS as DEFAULT_GRAPH_METRICS,
    compute_metric_ranks,
    pareto_non_dominated,
)

NLP_METRIC_NAME = "nlp_avg_rank_score"

import csv as _csv  # keep module-level csv clean for this file
import re as _re
from urllib.parse import quote as _quote

_VERSE_NODE_RE = _re.compile(r"^(?P<book>[^.]+)\.(?P<chapter>\d+)\.(?P<verse>\d+)$")
_EN_BOOK_MAPPING_CACHE: Optional[Dict[str, str]] = None


def _load_en_book_mapping() -> Dict[str, str]:
    global _EN_BOOK_MAPPING_CACHE
    if _EN_BOOK_MAPPING_CACHE is not None:
        return _EN_BOOK_MAPPING_CACHE

    idx_path = Path(__file__).resolve().parent / ".zraw_metadata" / "EN_book_index.txt"
    mapping: Dict[str, str] = {}
    lines = idx_path.read_text(encoding="utf-8").splitlines()
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


def _node_id_to_url(node_id: str, *, version: str = "ESV") -> str:
    m = _VERSE_NODE_RE.match(node_id.strip())
    if not m:
        return ""
    book_token = m.group("book")
    chapter = int(m.group("chapter"))
    verse = int(m.group("verse"))
    book_title = _load_en_book_mapping().get(book_token, book_token)
    citation = f"{book_title} {chapter}:{verse}"
    return f"https://www.biblegateway.com/passage/?search={_quote(citation)}&version={version}"


def _write_pareto_csv_with_urls(
    *,
    out_csv: Path,
    nodes: Sequence[str],
    metric_ranks: Dict[str, Dict[str, int]],
    pareto_nodes: Sequence[str],
    metrics: Sequence[str],
    bible_version: str,
) -> None:
    pareto_set = set(pareto_nodes)
    n = len(nodes)

    rows: List[Tuple[bool, float, str, Dict[str, int]]] = []
    for node in nodes:
        # Reuse the exact tie-break logic from rank_verses_pareto.py:
        # normalized sum: (rank-1)/(n-1)
        pareto_member = node in pareto_set

        if n <= 1:
            tiebreak = 0.0
        else:
            denom = float(n - 1)
            s = 0.0
            for m in metrics:
                r = metric_ranks[m].get(node, n)
                s += float(r - 1) / denom
            tiebreak = s

        ranks_for_node = {m: metric_ranks[m].get(node, n) for m in metrics}
        rows.append((pareto_member, tiebreak, node, ranks_for_node))

    # Final ordering:
    #   - Pareto members first
    #   - Within those: smaller tiebreak_sum is better
    #   - Deterministic tie-break: node_id
    rows.sort(key=lambda t: (not t[0], t[1], t[2]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(
            ["node_id", "node_url", "pareto_member", "final_order", "tiebreak_sum_normalized_ranks"]
            + [f"{m}_rank" for m in metrics]
        )
        for order, (_, tiebreak, node, ranks_for_node) in enumerate(rows, start=1):
            w.writerow(
                [
                    node,
                    _node_id_to_url(node, version=bible_version),
                    str(node in pareto_set),
                    order,
                    tiebreak,
                ]
                + [ranks_for_node[m] for m in metrics]
            )


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_theme_csv_avg_rank_score(theme_csv_path: Path) -> Dict[str, float]:
    """
    Return: candidate_node_id -> avg(rank_score) across all seeds.
    """
    accum: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    with theme_csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cand = (row.get("candidate_node_id") or "").strip()
            if not cand:
                continue
            score_raw = (row.get("rank_score") or "").strip()
            try:
                score = float(score_raw)
            except Exception:
                score = 0.0
            accum[cand] = accum.get(cand, 0.0) + float(score)
            counts[cand] = counts.get(cand, 0) + 1

    out: Dict[str, float] = {}
    for cand, total in accum.items():
        c = counts.get(cand, 0) or 0
        out[cand] = (total / float(c)) if c > 0 else 0.0
    return out


def pick_eligible_nodes(community_result: Dict[str, object]) -> Tuple[List[str], List[str], List[str]]:
    seed_nodes_raw = community_result.get("seed_nodes", [])
    comm_nodes_raw = community_result.get("community_nodes", [])
    ranks = community_result.get("ranks", {})

    if not isinstance(seed_nodes_raw, list) or not isinstance(comm_nodes_raw, list) or not isinstance(ranks, dict):
        raise ValueError("community JSON missing expected fields: seed_nodes/community_nodes/ranks")

    seed_nodes = [n for n in seed_nodes_raw if isinstance(n, str)]
    community_nodes = [n for n in comm_nodes_raw if isinstance(n, str)]

    seed_set = set(seed_nodes)
    eligible = [n for n in community_nodes if n not in seed_set]
    # Keep only nodes that have ranks info
    eligible = [n for n in eligible if n in ranks]

    return seed_nodes, community_nodes, eligible


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute Pareto ranking with NLP aggregated across seeds.")
    parser.add_argument("--community-json", type=str, required=True, help="Path to seeded_local_community JSON.")
    parser.add_argument("--theme-csv", type=str, required=True, help="Path to theme_vs_seeds_ranked_clamped CSV.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Default: out/<stem>_pareto_ranking_with_nlp.csv",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_GRAPH_METRICS),
        help="Comma-separated graph metrics to use (must exist in community JSON ranks).",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default="community_nodes",
        choices=["community_nodes"],
        help="Eligibility set.",
    )
    parser.add_argument("--version", type=str, default="ESV", help="Bible translation version (default: ESV).")

    args = parser.parse_args(list(argv) if argv is not None else None)
    bible_version = str(args.version).strip().upper()

    community_path = Path(args.community_json).resolve()
    theme_csv_path = Path(args.theme_csv).resolve()
    if not community_path.exists():
        raise FileNotFoundError(f"community JSON not found: {community_path}")
    if not theme_csv_path.exists():
        raise FileNotFoundError(f"theme CSV not found: {theme_csv_path}")

    community_result = load_json(community_path)
    seed_nodes, _community_nodes, eligible_nodes = pick_eligible_nodes(community_result)
    if not eligible_nodes:
        raise ValueError("No eligible nodes after excluding seed nodes.")

    graph_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not graph_metrics:
        raise ValueError("No graph metrics provided.")

    nlp_avg_scores = load_theme_csv_avg_rank_score(theme_csv_path)

    # Build `ranks` in the same shape expected by rank_verses_pareto:
    # ranks[node][metric] -> float
    ranks_field = community_result.get("ranks", {})
    assert isinstance(ranks_field, dict)

    ranks_by_node: Dict[str, Dict[str, float]] = {}
    for node in eligible_nodes:
        node_rank_raw = ranks_field.get(node, {})
        if not isinstance(node_rank_raw, dict):
            continue
        node_rank: Dict[str, float] = {}
        for m in graph_metrics:
            v_raw = node_rank_raw.get(m, 0.0)
            try:
                node_rank[m] = float(v_raw)
            except Exception:
                node_rank[m] = 0.0
        node_rank[NLP_METRIC_NAME] = float(nlp_avg_scores.get(node, 0.0))
        ranks_by_node[node] = node_rank

    metrics_all = graph_metrics + [NLP_METRIC_NAME]
    metric_ranks = compute_metric_ranks(nodes=eligible_nodes, ranks=ranks_by_node, metrics=metrics_all)
    pareto_nodes = pareto_non_dominated(eligible_nodes, metric_ranks, metrics_all)

    if args.out:
        out_csv = Path(args.out).resolve()
    else:
        out_csv = community_path.with_name(community_path.stem + "_pareto_ranking_with_nlp.csv")

    _write_pareto_csv_with_urls(
        out_csv=out_csv,
        nodes=eligible_nodes,
        metric_ranks=metric_ranks,
        pareto_nodes=pareto_nodes,
        metrics=metrics_all,
        bible_version=bible_version,
    )
    print(f"wrote: {out_csv} (eligible={len(eligible_nodes)}, pareto={len(pareto_nodes)}, seeds={len(seed_nodes)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

