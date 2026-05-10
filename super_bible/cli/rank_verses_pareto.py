#!/usr/bin/env python3
# coding: utf-8

"""
Rank verses using Pareto non-dominance across multiple graph metrics,
with tie-break by sum of normalized ranks.

Input:
  - JSON from `seeded_local_community.py`

Eligibility:
  - `community_nodes` only

Output:
  - CSV written to `out/<stem>_pareto_ranking.csv`

Columns:
  - node_id
  - pareto_member (True/False)
  - final_order (1..N)
  - tiebreak_sum_normalized_ranks
  - <metric>_rank for each metric
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_METRICS = [
    "k_core",
    "k_truss",
    "local_clustering",
    "triangle_count",
    "personalized_pagerank",
]


def load_result_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_metric_ranks(
    *,
    nodes: Sequence[str],
    ranks: Dict[str, Dict[str, float]],
    metrics: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    """
    Higher metric score is better => rank 1 is best.
    Tie-break within metric: node_id lexicographic.
    """
    metric_to_rank: Dict[str, Dict[str, int]] = {m: {} for m in metrics}

    for m in metrics:
        scored: List[Tuple[float, str]] = []
        for n in nodes:
            node_rank = ranks.get(n, {})
            v = node_rank.get(m, 0.0)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            scored.append((v, n))

        # Sort best first: value desc, node_id asc.
        scored.sort(key=lambda t: (-t[0], t[1]))
        for idx, (_, n) in enumerate(scored, start=1):
            metric_to_rank[m][n] = idx
    return metric_to_rank


def pareto_non_dominated(nodes: Sequence[str], metric_ranks: Dict[str, Dict[str, int]], metrics: Sequence[str]) -> List[str]:
    """
    Node A dominates B if for all metrics: rankA <= rankB
    and for at least one metric: rankA < rankB.

    Lower rank is better.
    """
    nodes_set = set(nodes)
    for m in metrics:
        # Ensure metrics exist for each node; missing treated as worst (large).
        if m not in metric_ranks:
            raise ValueError(f"missing metric ranks for {m}")

    # Precompute rank dict per node for speed.
    node_to_rank_vec: Dict[str, List[int]] = {}
    for n in nodes:
        node_to_rank_vec[n] = [metric_ranks[m].get(n, 10**9) for m in metrics]

    pareto: List[str] = []
    for i, a in enumerate(nodes):
        a_vec = node_to_rank_vec[a]
        dominated = False
        for j, b in enumerate(nodes):
            if i == j:
                continue
            b_vec = node_to_rank_vec[b]
            # Check b dominates a (since smaller rank is better)
            b_better_or_equal = True
            b_strictly_better = False
            for k in range(len(metrics)):
                if b_vec[k] > a_vec[k]:
                    b_better_or_equal = False
                    break
                if b_vec[k] < a_vec[k]:
                    b_strictly_better = True
            if b_better_or_equal and b_strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    return pareto


def compute_normalized_rank_sum(
    *,
    node: str,
    metric_ranks: Dict[str, Dict[str, int]],
    metrics: Sequence[str],
    n_nodes: int,
) -> float:
    """
    Normalize each metric rank to [0,1] using:
      (rank-1)/(n_nodes-1)
    """
    if n_nodes <= 1:
        return 0.0
    denom = float(n_nodes - 1)
    s = 0.0
    for m in metrics:
        r = metric_ranks[m].get(node, n_nodes)
        s += (float(r - 1) / denom)
    return s


def write_pareto_csv(
    *,
    out_csv: Path,
    nodes: Sequence[str],
    metric_ranks: Dict[str, Dict[str, int]],
    pareto_nodes: Sequence[str],
    metrics: Sequence[str],
) -> None:
    pareto_set = set(pareto_nodes)
    n = len(nodes)

    rows: List[Tuple[bool, float, str, Dict[str, int]]] = []
    for node in nodes:
        pareto_member = node in pareto_set
        tiebreak = compute_normalized_rank_sum(
            node=node,
            metric_ranks=metric_ranks,
            metrics=metrics,
            n_nodes=n,
        )
        ranks_for_node = {m: metric_ranks[m].get(node, n) for m in metrics}
        rows.append((pareto_member, tiebreak, node, ranks_for_node))

    # Final ordering:
    #   - Pareto members first
    #   - Within those: smaller tiebreak_sum is better
    #   - Deterministic tie-break: node_id
    rows.sort(key=lambda t: (not t[0], t[1], t[2]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["node_id", "pareto_member", "final_order", "tiebreak_sum_normalized_ranks"]
            + [f"{m}_rank" for m in metrics]
        )
        for order, (_, tiebreak, node, ranks_for_node) in enumerate(rows, start=1):
            w.writerow(
                [node, str(node in pareto_set), order, tiebreak]
                + [ranks_for_node[m] for m in metrics]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pareto ranking for verse nodes.")
    parser.add_argument("--json", type=str, required=True, help="Path to seeded_local_community JSON.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Default: out/<stem>_pareto_ranking.csv",
    )
    parser.add_argument(
        "--k",
        type=str,
        default=None,
        help="(unused) kept for compatibility; no-op.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to use.",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default="community_nodes",
        choices=["community_nodes", "ego_nodes"],
        help="Node eligibility set.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = load_result_json(Path(args.json).resolve())
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    eligible = result.get(args.nodes, [])
    ranks = result.get("ranks", {})
    if not isinstance(eligible, list) or not isinstance(ranks, dict):
        raise ValueError("JSON does not contain expected fields.")

    eligible_nodes = [n for n in eligible if isinstance(n, str)]
    if not eligible_nodes:
        raise ValueError("No eligible nodes found.")

    metric_ranks = compute_metric_ranks(nodes=eligible_nodes, ranks=ranks, metrics=metrics)
    pareto_nodes = pareto_non_dominated(eligible_nodes, metric_ranks, metrics)

    out_csv = Path(args.out).resolve() if args.out else Path(args.json).resolve().parent / (
        Path(args.json).resolve().stem + "_pareto_ranking.csv"
    )
    write_pareto_csv(
        out_csv=out_csv,
        nodes=eligible_nodes,
        metric_ranks=metric_ranks,
        pareto_nodes=pareto_nodes,
        metrics=metrics,
    )
    print(f"wrote: {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

