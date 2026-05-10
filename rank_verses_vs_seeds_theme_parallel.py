#!/usr/bin/env python3
# coding: utf-8

"""
Rank verses vs seed verses using theme-similarity metrics.

Inputs:
  - community JSON (from seeded_local_community.py) containing `seed_nodes`
  - Pareto ranking CSV (from rank_verses_pareto.py) containing `node_id`, `pareto_member`, `final_order`

Candidate selection:
  - default: only Pareto-front candidates (pareto_member=True), sorted by final_order asc, then take top N
  - --test-all-verses: use all node_ids present in the Pareto CSV (regardless of pareto_member),
    sorted by final_order asc, but always skip seed nodes

Theme metrics (computed memoized once in main process):
  1) Topic distributions:
       - shared LDA topic space
       - Jensen–Shannon divergence (JSD) and top-k topic overlap Jaccard
  2) Full-text cosine similarity:
       - TF-IDF cosine similarity
  3) Keyphrases + entities:
       - TF-IDF cosine similarity on TF-IDF-weighted keyphrase n-grams
       - TF-IDF cosine similarity on heuristic "entity" strings

Parallelism:
  - Use a ThreadPool with --workers threads to compute per-candidate scoring vs all seeds.
    (Representation matrices are shared in memory; workers do cheap metric math + aggregate rank_score.)

Output:
  - CSV in out/ with one row per (seed, candidate) with rank_order and rank_score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from urllib.parse import quote
import hashlib
import os

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


_VERSE_NODE_RE = re.compile(r"^(?P<book>[^.]+)\.(?P<chapter>\d+)\.(?P<verse>\d+)$")

_CAP_ENTITY_RE = re.compile(
    # Capitalized sequences like "God", "Lord", "Israel", "Holy Spirit"
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+[A-Z][a-z]+|[A-Z]{2,}){0,3}\b"
)

_EN_BOOK_MAPPING_CACHE: Optional[Dict[str, str]] = None


def load_en_book_mapping() -> Dict[str, str]:
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
        # columns: (book index, osisID, title, total_chapters, testament)
        osis_id = parts[1]
        book_name = parts[2]
        mapping[osis_id] = book_name
    _EN_BOOK_MAPPING_CACHE = mapping
    return mapping


def parse_node_id(node_id: str) -> Tuple[str, int, int]:
    m = _VERSE_NODE_RE.match(node_id.strip())
    if not m:
        raise ValueError(f"Invalid node_id: {node_id}")
    book_token = m.group("book")
    chapter = int(m.group("chapter"))
    verse = int(m.group("verse"))
    return book_token, chapter, verse


def node_id_to_biblegateway_url(node_id: str, *, db_language_version: str = "ESV") -> str:
    """
    Build a stable BibleGateway URL that matches the repo's existing citation format:
      https://www.biblegateway.com/passage/?search=<Book> <chapter>:<verse>&version=ESV
    """
    book_token, chapter, verse = parse_node_id(node_id)
    en_book_map = load_en_book_mapping()
    book_title = en_book_map.get(book_token, book_token)
    citation = f"{book_title} {chapter}:{verse}"
    return f"https://www.biblegateway.com/passage/?search={quote(citation)}&version={db_language_version}"


def get_esv_texts_for_node_ids(db_path: Path, node_ids: Sequence[str]) -> Dict[str, str]:
    """
    Batched DB fetch for ESV texts.

    Groups by (title, chapter) so we can query verse ranges with a single SQL
    statement per (title, chapter).
    """
    if not node_ids:
        return {}

    en_book_map = load_en_book_mapping()

    # Group by (book_title, chapter) -> list of (node_id, verse)
    groups: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    for node_id in node_ids:
        book_token, chapter, verse = parse_node_id(node_id)
        book_title = en_book_map.get(book_token, book_token)
        groups[(book_title, chapter)].append((node_id, verse))

    out: Dict[str, str] = {}

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        for (book_title, chapter), id_verse_list in groups.items():
            verses = [v for (_nid, v) in id_verse_list]
            # Build IN (...) placeholders
            placeholders = ",".join(["?"] * len(verses))
            sql = f"""
                SELECT verse, text
                FROM ESV
                WHERE title = ?
                  AND chapter = ?
                  AND verse IN ({placeholders})
            """
            params: List[object] = [book_title, chapter]
            params.extend(verses)
            cur.execute(sql, params)
            rows = cur.fetchall()
            verse_to_text: Dict[int, str] = {int(v): str(t) for (v, t) in rows}
            for node_id, verse in id_verse_list:
                text = verse_to_text.get(verse, "")
                out[node_id] = text
    finally:
        conn.close()

    return out


def load_community_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ParetoRow:
    node_id: str
    pareto_member: bool
    final_order: int


def load_pareto_csv(path: Path) -> List[ParetoRow]:
    rows: List[ParetoRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            node_id = (row.get("node_id") or "").strip()
            if not node_id:
                continue
            pm_raw = (row.get("pareto_member") or "").strip()
            pareto_member = pm_raw.lower() == "true"
            fo_raw = (row.get("final_order") or "").strip()
            try:
                final_order = int(float(fo_raw))
            except Exception:
                final_order = 10**9
            rows.append(ParetoRow(node_id=node_id, pareto_member=pareto_member, final_order=final_order))
    rows.sort(key=lambda rr: rr.final_order)
    return rows


def select_candidates(
    *,
    pareto_rows: Sequence[ParetoRow],
    seed_nodes: Sequence[str],
    top_n_pareto: int,
    test_all_verses: bool,
) -> List[str]:
    seed_set = set(seed_nodes)
    if test_all_verses:
        candidates = [r.node_id for r in pareto_rows if r.node_id not in seed_set]
        return candidates

    pareto_front = [r for r in pareto_rows if r.pareto_member]
    pareto_front_sorted = sorted(pareto_front, key=lambda rr: rr.final_order)
    limited = pareto_front_sorted[: max(0, int(top_n_pareto))]
    return [r.node_id for r in limited if r.node_id not in seed_set]


def _clamp_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p2 = np.asarray(p, dtype=float)
    p2 = np.clip(p2, eps, 1.0)
    s = float(p2.sum())
    if s <= 0:
        return np.full_like(p2, 1.0 / float(len(p2)))
    return p2 / s


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, *, log_base: float = 2.0) -> float:
    p_arr = _clamp_prob(np.asarray(p, dtype=float))
    q_arr = _clamp_prob(np.asarray(q, dtype=float))
    m = 0.5 * (p_arr + q_arr)

    logb = float(log_base)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = a / b
            return float(np.sum(a * np.log(ratio) / math.log(logb)))

    jsd = 0.5 * kl(p_arr, m) + 0.5 * kl(q_arr, m)
    # Numerical error can lead to tiny negative values.
    return max(0.0, float(jsd))


def top_k_topic_jaccard(p: np.ndarray, q: np.ndarray, *, k: int) -> float:
    if k <= 0:
        return 0.0
    k = int(min(k, len(p), len(q)))
    p_top = set(np.argsort(p)[::-1][:k].tolist())
    q_top = set(np.argsort(q)[::-1][:k].tolist())
    inter = len(p_top.intersection(q_top))
    union = len(p_top.union(q_top)) or 1
    return float(inter / union)


def extract_entity_candidates(text: str, *, top_entities: int, min_len: int = 2) -> List[str]:
    matches = _CAP_ENTITY_RE.findall(text)
    normed: List[str] = []
    for m in matches:
        m2 = re.sub(r"\s+", " ", m.strip())
        if len(m2) < min_len:
            continue
        normed.append(m2.lower())
    if not normed:
        return []

    freq: Dict[str, int] = {}
    for t in normed:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.keys(), key=lambda k: (freq[k], k), reverse=True)
    return ranked[:top_entities]


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def _mask_top_terms_per_row(X_dense: np.ndarray, top_terms: int) -> np.ndarray:
    if top_terms <= 0:
        return np.zeros_like(X_dense)
    out = np.zeros_like(X_dense)
    for i in range(X_dense.shape[0]):
        row = X_dense[i]
        if not np.any(row):
            continue
        # Take top-k indices by weight
        k = min(top_terms, row.shape[0])
        idx = np.argpartition(row, -k)[-k:]
        # Keep only positive weights
        for j in idx.tolist():
            if row[j] > 0:
                out[i, j] = row[j]
    return out


def compute_memoized_representations(
    docs_texts: Sequence[str],
    seed_indices: Sequence[int],
    cand_indices: Sequence[int],
    *,
    # topics
    lda_num_topics: int,
    lda_max_features: int,
    lda_min_df: int,
    lda_max_df: float,
    lda_top_k_topics: int,
    # fulltext cosine
    tfidf_full_max_features: int,
    # keyphrases cosine
    keyphrase_ngram_max: int,
    keyphrase_top_terms: int,
    keyphrase_tfidf_max_features: int,
    # entities cosine
    entity_top_entities: int,
    # stopwords/language knobs
    stop_words: str = "english",
) -> Dict[str, object]:
    # -------------------------
    # A) Topics (LDA)
    # -------------------------
    # Some CountVectorizer parameter combinations can become invalid with only 2 docs;
    # we always fit on the full doc set, but still clamp min/max defensively.
    n_docs = len(docs_texts)
    # abs max doc count implied by max_df
    abs_max_doc_count = int(math.floor(float(lda_max_df) * float(max(1, n_docs))))
    min_df = int(max(1, lda_min_df))
    if abs_max_doc_count < min_df:
        min_df = 1

    count_vec = CountVectorizer(
        lowercase=True,
        stop_words=stop_words,
        max_features=int(lda_max_features),
        min_df=min_df,
        max_df=float(lda_max_df),
        token_pattern=r"(?u)\b\w+\b",
    )
    counts = count_vec.fit_transform(list(docs_texts))
    vocab_size = int(counts.shape[1])

    if vocab_size <= 0:
        # Degenerate: create empty topic representations.
        theta = np.zeros((n_docs, int(lda_num_topics)), dtype=float)
        # Ensure each doc sums to 1
        theta[:] = 1.0 / float(max(1, int(lda_num_topics)))
    else:
        lda = LatentDirichletAllocation(
            n_components=int(lda_num_topics),
            random_state=42,
            learning_method="batch",
            max_iter=30,
        )
        lda.fit(counts)
        theta = lda.transform(counts)  # (n_docs, K)
    # float32 reduces memory and speeds up dot products.
    theta = theta.astype(np.float32, copy=False)

    # Precompute each doc's top-k topic indices for faster Jaccard.
    k = int(lda_top_k_topics)
    topk_indices = np.argsort(theta, axis=1)[:, ::-1][:, :k] if k > 0 else np.zeros((n_docs, 0), dtype=int)

    # -------------------------
    # B) Full-text TF-IDF cosine
    # -------------------------
    tfidf_full = TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        max_features=int(tfidf_full_max_features),
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b\w+\b",
    )
    X_full = tfidf_full.fit_transform(list(docs_texts))  # sparse

    # Convert just seeds + candidates to dense, for thread sharing / fast dot.
    seed_idx = np.asarray(list(seed_indices), dtype=int)
    cand_idx = np.asarray(list(cand_indices), dtype=int)

    X_full_seed = X_full[seed_idx].toarray()
    X_full_cand = X_full[cand_idx].toarray()
    # TfidfVectorizer defaults to L2-normalized vectors; cosine = dot product.
    # Still normalize defensively after conversion.
    X_full_seed = _l2_normalize_rows(X_full_seed)
    X_full_cand = _l2_normalize_rows(X_full_cand)
    X_full_seed = X_full_seed.astype(np.float32, copy=False)
    X_full_cand = X_full_cand.astype(np.float32, copy=False)

    # -------------------------
    # C) Keyphrases cosine
    # -------------------------
    tfidf_key = TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        max_features=int(keyphrase_tfidf_max_features),
        ngram_range=(1, int(keyphrase_ngram_max)),
        token_pattern=r"(?u)\b\w+\b",
    )
    X_key = tfidf_key.fit_transform(list(docs_texts)).toarray()  # dense (n_docs, V)
    X_key = _mask_top_terms_per_row(X_key, int(keyphrase_top_terms))
    X_key = _l2_normalize_rows(X_key)
    X_key = X_key.astype(np.float32, copy=False)

    X_key_seed = X_key[seed_idx]
    X_key_cand = X_key[cand_idx]

    # -------------------------
    # D) Entities cosine
    # -------------------------
    entity_docs: List[str] = []
    for t in docs_texts:
        ents = extract_entity_candidates(t, top_entities=int(entity_top_entities))
        entity_docs.append(" ".join(ents))

    tfidf_ent = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        max_features=20000,
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b\w+\b",
    )
    X_ent = tfidf_ent.fit_transform(entity_docs).toarray()
    X_ent = _l2_normalize_rows(X_ent)
    X_ent = X_ent.astype(np.float32, copy=False)
    X_ent_seed = X_ent[seed_idx]
    X_ent_cand = X_ent[cand_idx]

    # Return all memoized arrays used by workers.
    return {
        "theta": theta,  # (n_docs, K) float32
        "topk_indices": topk_indices.astype(np.int32, copy=False),  # (n_docs, k)
        "seed_idx_in_docs": seed_idx,
        "cand_idx_in_docs": cand_idx,
        "X_full_seed": X_full_seed,
        "X_full_cand": X_full_cand,
        "X_key_seed": X_key_seed,
        "X_key_cand": X_key_cand,
        "X_ent_seed": X_ent_seed,
        "X_ent_cand": X_ent_cand,
        "lda_top_k_topics": int(lda_top_k_topics),
    }


def make_reps_cache_key(
    *,
    docs_nodes: Sequence[str],
    seed_count: int,
    cand_count: int,
    params: Dict[str, object],
) -> str:
    """
    Deterministic cache key for memoized representations.

    Includes ordered `docs_nodes` so different seed/candidate slices don’t collide.
    """
    payload = {
        "algo": "memo_reps_v1",
        "docs_nodes": list(docs_nodes),
        "seed_count": int(seed_count),
        "cand_count": int(cand_count),
        "params": params,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_or_compute_memoized_representations_cached(
    *,
    docs_texts: Sequence[str],
    seed_indices: Sequence[int],
    cand_indices: Sequence[int],
    docs_nodes: Sequence[str],
    use_cache: bool,
    cache_dir: Path,
    cache_compress: bool,
    cache_key_params: Dict[str, object],
    # compute params
    lda_num_topics: int,
    lda_max_features: int,
    lda_min_df: int,
    lda_max_df: float,
    lda_top_k_topics: int,
    tfidf_full_max_features: int,
    keyphrase_ngram_max: int,
    keyphrase_top_terms: int,
    keyphrase_tfidf_max_features: int,
    entity_top_entities: int,
    stop_words: str = "english",
) -> Dict[str, object]:
    if not use_cache:
        return compute_memoized_representations(
            docs_texts=docs_texts,
            seed_indices=seed_indices,
            cand_indices=cand_indices,
            lda_num_topics=lda_num_topics,
            lda_max_features=lda_max_features,
            lda_min_df=lda_min_df,
            lda_max_df=lda_max_df,
            lda_top_k_topics=lda_top_k_topics,
            tfidf_full_max_features=tfidf_full_max_features,
            keyphrase_ngram_max=keyphrase_ngram_max,
            keyphrase_top_terms=keyphrase_top_terms,
            keyphrase_tfidf_max_features=keyphrase_tfidf_max_features,
            entity_top_entities=entity_top_entities,
            stop_words=stop_words,
        )

    seed_count = len(seed_indices)
    cand_count = len(cand_indices)
    cache_key = make_reps_cache_key(
        docs_nodes=docs_nodes,
        seed_count=seed_count,
        cand_count=cand_count,
        params=cache_key_params,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"reps_cache_{cache_key}.npz"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        loaded = np.load(str(cache_path))
        return {
            "theta": loaded["theta"],
            "topk_indices": loaded["topk_indices"],
            "seed_idx_in_docs": loaded["seed_idx_in_docs"],
            "cand_idx_in_docs": loaded["cand_idx_in_docs"],
            "X_full_seed": loaded["X_full_seed"],
            "X_full_cand": loaded["X_full_cand"],
            "X_key_seed": loaded["X_key_seed"],
            "X_key_cand": loaded["X_key_cand"],
            "X_ent_seed": loaded["X_ent_seed"],
            "X_ent_cand": loaded["X_ent_cand"],
            "lda_top_k_topics": int(loaded["lda_top_k_topics"][0]),
        }

    reps = compute_memoized_representations(
        docs_texts=docs_texts,
        seed_indices=seed_indices,
        cand_indices=cand_indices,
        lda_num_topics=lda_num_topics,
        lda_max_features=lda_max_features,
        lda_min_df=lda_min_df,
        lda_max_df=lda_max_df,
        lda_top_k_topics=lda_top_k_topics,
        tfidf_full_max_features=tfidf_full_max_features,
        keyphrase_ngram_max=keyphrase_ngram_max,
        keyphrase_top_terms=keyphrase_top_terms,
        keyphrase_tfidf_max_features=keyphrase_tfidf_max_features,
        entity_top_entities=entity_top_entities,
        stop_words=stop_words,
    )

    tmp_path = cache_dir / f".tmp_reps_cache_{os.getpid()}_{cache_key}.npz"
    if cache_compress:
        np.savez_compressed(
            str(tmp_path),
            theta=reps["theta"],
            topk_indices=reps["topk_indices"],
            seed_idx_in_docs=reps["seed_idx_in_docs"],
            cand_idx_in_docs=reps["cand_idx_in_docs"],
            X_full_seed=reps["X_full_seed"],
            X_full_cand=reps["X_full_cand"],
            X_key_seed=reps["X_key_seed"],
            X_key_cand=reps["X_key_cand"],
            X_ent_seed=reps["X_ent_seed"],
            X_ent_cand=reps["X_ent_cand"],
            lda_top_k_topics=np.asarray([int(reps["lda_top_k_topics"])], dtype=np.int32),
        )
    else:
        np.savez(
            str(tmp_path),
            theta=reps["theta"],
            topk_indices=reps["topk_indices"],
            seed_idx_in_docs=reps["seed_idx_in_docs"],
            cand_idx_in_docs=reps["cand_idx_in_docs"],
            X_full_seed=reps["X_full_seed"],
            X_full_cand=reps["X_full_cand"],
            X_key_seed=reps["X_key_seed"],
            X_key_cand=reps["X_key_cand"],
            X_ent_seed=reps["X_ent_seed"],
            X_ent_cand=reps["X_ent_cand"],
            lda_top_k_topics=np.asarray([int(reps["lda_top_k_topics"])], dtype=np.int32),
        )
    tmp_path.replace(cache_path)
    return reps


def compute_topk_jaccard_from_indices(p_topk: np.ndarray, q_topk: np.ndarray) -> float:
    if p_topk.size == 0 and q_topk.size == 0:
        return 0.0
    p_set = set(p_topk.tolist())
    q_set = set(q_topk.tolist())
    inter = len(p_set.intersection(q_set))
    union = len(p_set.union(q_set)) or 1
    return float(inter / union)


def aggregate_rank_score(
    *,
    cos_fulltext: float,
    cos_keyphrases: float,
    cos_entities: float,
    jsd_topics: float,
    topk_jaccard: float,
    w_cos_fulltext: float,
    w_cos_keyphrases: float,
    w_cos_entities: float,
    w_topics_jsd: float,
    w_topics_topk_jaccard: float,
) -> float:
    # Convert "distance-like" metrics to "similarity-like"
    sim_topics = 1.0 - float(jsd_topics)
    return (
        float(w_cos_fulltext) * float(cos_fulltext)
        + float(w_cos_keyphrases) * float(cos_keyphrases)
        + float(w_cos_entities) * float(cos_entities)
        + float(w_topics_jsd) * sim_topics
        + float(w_topics_topk_jaccard) * float(topk_jaccard)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank verses vs seeds using theme metrics (parallel).")
    parser.add_argument("--community-json", type=str, required=True, help="Path to seeded_local_community JSON.")
    parser.add_argument("--pareto-csv", type=str, required=True, help="Path to *_pareto_ranking.csv.")
    parser.add_argument("--db", type=str, default=None, help="Path to super_bible.db (defaults to SUPER_BIBLE/super_bible.db).")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path.")

    parser.add_argument("--workers", type=int, default=24, help="Thread workers for scoring.")
    parser.add_argument("--top-n-pareto", type=int, default=50, help="Top N Pareto-front candidates.")
    parser.add_argument("--test-all-verses", action="store_true", help="Score all Pareto CSV node_ids (skip seeds).")

    # memoized representations cache (speeds repeated runs)
    parser.add_argument(
        "--use-reps-cache",
        dest="use_reps_cache",
        action="store_true",
        help="Enable memoized representations cache (default).",
    )
    parser.add_argument(
        "--no-use-reps-cache",
        dest="use_reps_cache",
        action="store_false",
        help="Disable memoized representations cache.",
    )
    parser.set_defaults(use_reps_cache=True)
    parser.add_argument("--reps-cache-dir", type=str, default="out/.cache", help="Directory for reps cache.")
    parser.add_argument(
        "--reps-cache-compress",
        action="store_true",
        help="Compress cache (smaller files; slower writes).",
    )

    # topic model params
    parser.add_argument("--num-topics", type=int, default=20)
    parser.add_argument("--lda-max-features", type=int, default=5000)
    parser.add_argument("--lda-min-df", type=int, default=1)
    parser.add_argument("--lda-max-df", type=float, default=0.9)
    parser.add_argument("--lda-top-k-topics", type=int, default=5)

    # cosine / keyphrase / entity params
    parser.add_argument("--tfidf-full-max-features", type=int, default=5000)
    parser.add_argument("--keyphrase-ngram-max", type=int, default=2)
    parser.add_argument("--keyphrase-top-terms", type=int, default=30)
    parser.add_argument("--keyphrase-tfidf-max-features", type=int, default=5000)
    parser.add_argument("--entity-top-entities", type=int, default=50)

    # aggregate weights
    parser.add_argument("--w-cos-fulltext", type=float, default=1.0)
    parser.add_argument("--w-cos-keyphrases", type=float, default=1.0)
    parser.add_argument("--w-cos-entities", type=float, default=1.0)
    parser.add_argument("--w-topics-jsd", type=float, default=1.0)
    parser.add_argument("--w-topics-topk-jaccard", type=float, default=1.0)

    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parent
    db_path = Path(args.db).resolve() if args.db else (root / "SUPER_BIBLE" / "super_bible.db")
    community_path = Path(args.community_json).resolve()
    pareto_path = Path(args.pareto_csv).resolve()

    if not community_path.exists():
        raise FileNotFoundError(f"community JSON not found: {community_path}")
    if not pareto_path.exists():
        raise FileNotFoundError(f"pareto CSV not found: {pareto_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    community = load_community_json(community_path)
    seed_nodes_raw = community.get("seed_nodes", [])
    if not isinstance(seed_nodes_raw, list) or not seed_nodes_raw:
        raise ValueError("community JSON missing seed_nodes list or it's empty")
    seed_nodes = [n for n in seed_nodes_raw if isinstance(n, str)]
    if not seed_nodes:
        raise ValueError("community JSON seed_nodes list has no strings")

    pareto_rows = load_pareto_csv(pareto_path)
    candidate_nodes = select_candidates(
        pareto_rows=pareto_rows,
        seed_nodes=seed_nodes,
        top_n_pareto=args.top_n_pareto,
        test_all_verses=args.test_all_verses,
    )
    if not candidate_nodes:
        raise ValueError("No candidate nodes after skipping seeds.")

    # Docs list for shared representations
    docs_nodes = list(seed_nodes) + list(candidate_nodes)
    seed_count = len(seed_nodes)
    cand_count = len(candidate_nodes)
    seed_doc_indices = list(range(0, seed_count))
    cand_doc_indices = list(range(seed_count, seed_count + cand_count))

    # Fetch verse texts for all docs nodes once
    docs_texts_map = get_esv_texts_for_node_ids(db_path, docs_nodes)
    docs_texts: List[str] = [docs_texts_map.get(n, "") for n in docs_nodes]

    cache_dir = Path(args.reps_cache_dir).resolve()
    cache_key_params = {
        "lda_num_topics": int(args.num_topics),
        "lda_max_features": int(args.lda_max_features),
        "lda_min_df": int(args.lda_min_df),
        "lda_max_df": float(args.lda_max_df),
        "lda_top_k_topics": int(args.lda_top_k_topics),
        "tfidf_full_max_features": int(args.tfidf_full_max_features),
        "keyphrase_ngram_max": int(args.keyphrase_ngram_max),
        "keyphrase_top_terms": int(args.keyphrase_top_terms),
        "keyphrase_tfidf_max_features": int(args.keyphrase_tfidf_max_features),
        "entity_top_entities": int(args.entity_top_entities),
        "stop_words": "english",
        "tfidf_ent_max_features": 20000,
        "float32": True,
    }

    # Memoize all representations (with optional on-disk cache)
    reps = load_or_compute_memoized_representations_cached(
        docs_texts=docs_texts,
        seed_indices=seed_doc_indices,
        cand_indices=cand_doc_indices,
        docs_nodes=docs_nodes,
        use_cache=bool(args.use_reps_cache),
        cache_dir=cache_dir,
        cache_compress=bool(args.reps_cache_compress),
        cache_key_params=cache_key_params,
        lda_num_topics=int(args.num_topics),
        lda_max_features=int(args.lda_max_features),
        lda_min_df=int(args.lda_min_df),
        lda_max_df=float(args.lda_max_df),
        lda_top_k_topics=int(args.lda_top_k_topics),
        tfidf_full_max_features=int(args.tfidf_full_max_features),
        keyphrase_ngram_max=int(args.keyphrase_ngram_max),
        keyphrase_top_terms=int(args.keyphrase_top_terms),
        keyphrase_tfidf_max_features=int(args.keyphrase_tfidf_max_features),
        entity_top_entities=int(args.entity_top_entities),
    )

    theta: np.ndarray = reps["theta"]  # (n_docs, K)
    topk_indices: np.ndarray = reps["topk_indices"]  # (n_docs, k)

    # Candidate + seed indices *within their doc-aligned arrays*
    X_full_seed: np.ndarray = reps["X_full_seed"]
    X_full_cand: np.ndarray = reps["X_full_cand"]
    X_key_seed: np.ndarray = reps["X_key_seed"]
    X_key_cand: np.ndarray = reps["X_key_cand"]
    X_ent_seed: np.ndarray = reps["X_ent_seed"]
    X_ent_cand: np.ndarray = reps["X_ent_cand"]

    num_seeds = X_full_seed.shape[0]
    num_cands = X_full_cand.shape[0]
    if num_seeds != seed_count or num_cands != cand_count:
        raise RuntimeError("Internal shape mismatch for seed/candidate splits.")

    # Map local candidate index -> global node_id
    cand_node_ids_local = candidate_nodes  # in order
    seed_node_ids_local = seed_nodes
    cand_urls = {nid: node_id_to_biblegateway_url(nid) for nid in cand_node_ids_local}

    # Worker function: score a slice of candidate indices against all seeds.
    def score_candidate_slice(start: int, end: int) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        # Pre-bind arrays for speed
        Xfs_seed = X_full_seed
        Xfs_cand = X_full_cand
        Xks_seed = X_key_seed
        Xks_cand = X_key_cand
        Xes_seed = X_ent_seed
        Xes_cand = X_ent_cand

        lda_top_k = int(args.lda_top_k_topics)

        # docs indices in the original docs_texts array
        # seed doc indices = 0..seed_count-1
        # candidate doc indices = seed_count..seed_count+cand_count-1
        for cand_local_i in range(start, end):
            cand_node_id = cand_node_ids_local[cand_local_i]
            cand_doc_i = seed_count + cand_local_i
            theta_c = theta[cand_doc_i, :]
            topk_c = topk_indices[cand_doc_i, :] if lda_top_k > 0 else np.zeros((0,), dtype=int)
            # cosine metrics: since rows are L2 normalized, cosine = dot
            cos_full_vec = np.dot(Xfs_cand[cand_local_i, :], Xfs_seed.T)
            cos_key_vec = np.dot(Xks_cand[cand_local_i, :], Xks_seed.T)
            cos_ent_vec = np.dot(Xes_cand[cand_local_i, :], Xes_seed.T)

            for seed_local_j in range(num_seeds):
                seed_node_id = seed_node_ids_local[seed_local_j]
                seed_doc_i = seed_local_j
                theta_s = theta[seed_doc_i, :]
                topk_s = topk_indices[seed_doc_i, :] if lda_top_k > 0 else np.zeros((0,), dtype=int)

                jsd = jensen_shannon_divergence(theta_c, theta_s, log_base=2.0)
                topk_jacc = compute_topk_jaccard_from_indices(topk_c, topk_s) if lda_top_k > 0 else 0.0

                cos_full = float(cos_full_vec[seed_local_j])
                cos_key = float(cos_key_vec[seed_local_j])
                cos_ent = float(cos_ent_vec[seed_local_j])

                rank_score = aggregate_rank_score(
                    cos_fulltext=cos_full,
                    cos_keyphrases=cos_key,
                    cos_entities=cos_ent,
                    jsd_topics=jsd,
                    topk_jaccard=topk_jacc,
                    w_cos_fulltext=args.w_cos_fulltext,
                    w_cos_keyphrases=args.w_cos_keyphrases,
                    w_cos_entities=args.w_cos_entities,
                    w_topics_jsd=args.w_topics_jsd,
                    w_topics_topk_jaccard=args.w_topics_topk_jaccard,
                )

                rows.append(
                    {
                        "seed_node_id": seed_node_id,
                        "candidate_node_id": cand_node_id,
                        "rank_score": rank_score,
                        "cos_fulltext": cos_full,
                        "cos_keyphrases": cos_key,
                        "cos_entities": cos_ent,
                        "jsd_topics": jsd,
                        "topk_overlap_jaccard": topk_jacc,
                    }
                )
        return rows

    # Parallel scoring across candidate slices
    workers = max(1, int(args.workers))
    # Aim for ~4 slices per worker for load balance
    slices = max(workers * 4, 1)
    slice_size = int(math.ceil(num_cands / float(slices)))
    slice_ranges = [(i, min(num_cands, i + slice_size)) for i in range(0, num_cands, slice_size) if i < num_cands]
    slice_ranges = slice_ranges[: slices]  # cap

    all_scored_rows: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(score_candidate_slice, start, end) for (start, end) in slice_ranges]
        for fut in as_completed(futures):
            all_scored_rows.extend(fut.result())

    # Rank candidates per seed (rank_score desc, deterministic tie by candidate_node_id)
    per_seed: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in all_scored_rows:
        per_seed[str(row["seed_node_id"])].append(row)

    out_rows: List[Dict[str, object]] = []
    for seed_node_id in seed_node_ids_local:
        rows = per_seed.get(seed_node_id, [])
        # Sort: best rank_score desc, then candidate_node_id asc
        rows.sort(key=lambda r: (-float(r["rank_score"]), str(r["candidate_node_id"])))
        for idx, r in enumerate(rows, start=1):
            out_rows.append(
                {
                    "seed_node_id": seed_node_id,
                    "candidate_node_id": str(r["candidate_node_id"]),
                    "candidate_url": str(cand_urls.get(str(r["candidate_node_id"]), "")),
                    "rank_order": idx,
                    "rank_score": float(r["rank_score"]),
                    "cos_fulltext": float(r["cos_fulltext"]),
                    "cos_keyphrases": float(r["cos_keyphrases"]),
                    "cos_entities": float(r["cos_entities"]),
                    "jsd_topics": float(r["jsd_topics"]),
                    "topk_overlap_jaccard": float(r["topk_overlap_jaccard"]),
                }
            )

    out_csv = (
        Path(args.out).resolve()
        if args.out
        else pareto_path.with_name(pareto_path.stem.replace("_pareto_ranking", "") + "_theme_vs_seeds_ranked.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed_node_id",
        "candidate_node_id",
        "candidate_url",
        "rank_order",
        "rank_score",
        "cos_fulltext",
        "cos_keyphrases",
        "cos_entities",
        "jsd_topics",
        "topk_overlap_jaccard",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(
        f"wrote: {out_csv} (seeds={len(seed_node_ids_local)}, candidates={len(candidate_nodes)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

