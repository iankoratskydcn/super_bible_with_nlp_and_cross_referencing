#!/usr/bin/env python3
# coding: utf-8

"""
Build a directed node-edge graph of verse cross-references.

Input: `cross_references.csv` with columns:
  - From Verse
  - To Verse
  - Votes

Nodes are verse identifiers like `Gen.1.1`.
Edges are From -> To, with attribute `weight = Votes`.

Range tokens (e.g. `Ps.89.11-Ps.89.12`) can be expanded into individual verses
when they share the same book + chapter. Otherwise the range token is kept as a
single node.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from super_bible import paths


_RANGE_RE = re.compile(
    r"^(?P<book_a>[^.]+)\.(?P<chap_a>\d+)\.(?P<verse_a>\d+)"
    r"-(?P<book_b>[^.]+)\.(?P<chap_b>\d+)\.(?P<verse_b>\d+)$"
)


@dataclass(frozen=True)
class EdgeKey:
    src: str
    dst: str


def _expand_range_token(token: str) -> List[str]:
    """
    Expand `Book.C.V-Book.C.V2` into `[Book.C.V, ..., Book.C.V2]` if within same
    book + chapter. Otherwise keep `token` as-is (as one node).
    """
    token = token.strip()
    m = _RANGE_RE.match(token)
    if not m:
        return [token]

    book_a = m.group("book_a")
    chap_a = int(m.group("chap_a"))
    verse_a = int(m.group("verse_a"))

    book_b = m.group("book_b")
    chap_b = int(m.group("chap_b"))
    verse_b = int(m.group("verse_b"))

    if book_a != book_b or chap_a != chap_b:
        return [token]
    if verse_b < verse_a:
        # Invalid/inverted range: keep token as node (fail-soft).
        return [token]

    return [f"{book_a}.{chap_a}.{v}" for v in range(verse_a, verse_b + 1)]


def expand_verse_token(token: str, *, expand_ranges: bool) -> List[str]:
    token = token.strip()
    if not token:
        return []
    if not expand_ranges:
        return [token]
    return _expand_range_token(token)


def iter_crossref_rows(csv_path: Path) -> Iterator[Tuple[str, str, int]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"From Verse", "To Verse", "Votes"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            src_tok = (row.get("From Verse") or "").strip()
            dst_tok = (row.get("To Verse") or "").strip()
            if not src_tok or not dst_tok:
                continue
            votes_raw = (row.get("Votes") or "").strip()
            votes = int(votes_raw)
            yield src_tok, dst_tok, votes


def build_digraph_from_csv(
    csv_path: Path,
    *,
    expand_ranges: bool = True,
    dedupe: str = "max_abs_votes",
) -> "object":
    """
    Return a NetworkX DiGraph.

    Dedupe strategy for duplicate edges (same src+dst):
      - max_abs_votes: keep the weight with maximum abs(Votes)
    """
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "networkx is required. Install with `pip install -r requirements.txt`."
        ) from e

    if dedupe != "max_abs_votes":
        raise ValueError("Only dedupe='max_abs_votes' is currently supported.")

    # Accumulate in a dict first for speed and dedupe correctness.
    # After building weights, we create the DiGraph once.
    edge_weight: Dict[EdgeKey, int] = {}

    nodes: set[str] = set()
    for src_tok, dst_tok, votes in iter_crossref_rows(csv_path):
        src_nodes = expand_verse_token(src_tok, expand_ranges=expand_ranges)
        dst_nodes = expand_verse_token(dst_tok, expand_ranges=expand_ranges)
        if not src_nodes or not dst_nodes:
            continue

        for src in src_nodes:
            nodes.add(src)
            for dst in dst_nodes:
                nodes.add(dst)
                k = EdgeKey(src=src, dst=dst)
                prev = edge_weight.get(k)
                if prev is None or abs(votes) > abs(prev):
                    edge_weight[k] = votes

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for k, w in edge_weight.items():
        g.add_edge(k.src, k.dst, weight=w)
    return g


def save_digraph_pickle(g: object, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_digraph_pickle(in_path: Path) -> object:
    with in_path.open("rb") as f:
        return pickle.load(f)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build verse graph from cross_references.csv")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(paths.CROSS_REFERENCES_CSV),
        help="Input CSV path (default: data/cross_references.csv).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/crossref_graph.gpickle",
        help="Output graph pickle path.",
    )
    parser.add_argument(
        "--no-expand-ranges",
        action="store_true",
        help="Do not expand Book.C.V-Book.C.V2 tokens.",
    )
    parser.add_argument(
        "--dedupe",
        type=str,
        default="max_abs_votes",
        choices=["max_abs_votes"],
        help="Edge dedupe strategy for duplicate src->dst.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    csv_path = Path(args.csv).resolve()
    out_path = Path(args.out).resolve()

    g = build_digraph_from_csv(
        csv_path,
        expand_ranges=not args.no_expand_ranges,
        dedupe=args.dedupe,
    )
    save_digraph_pickle(g, out_path)
    print(f"nodes={g.number_of_nodes()} edges={g.number_of_edges()} out={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

