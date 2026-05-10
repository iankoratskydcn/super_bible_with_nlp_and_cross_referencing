#!/usr/bin/env python3
# coding: utf-8

"""
Compute "theme similarity" metrics between two ESV passages (or any two texts).

Metrics family A (topics):
  - LDA topic distributions shared topic space
  - Jensen–Shannon divergence (JSD) between topic distributions
  - Top-k topic overlap (intersection + Jaccard)

Metrics family B (full-text cosine):
  - TF-IDF cosine similarity between two full texts

Metrics family C (keyphrases + entities):
  - Extract keyphrase candidates via TF-IDF top n-grams
  - Extract entity-like strings via heuristic capitalization patterns
  - For each (keyphrases, entities):
      TF-IDF cosine similarity over the extracted term vocab
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
except Exception as e:  # pragma: no cover
    raise ImportError("scikit-learn is required. Install with `pip install -r requirements.txt`.") from e


_CAP_ENTITY_RE = re.compile(
    # Capitalized sequences like "God", "Lord", "Israel", "Holy Spirit"
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+[A-Z][a-z]+|[A-Z]{2,}){0,3}\b"
)


def _clamp_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p2 = np.asarray(p, dtype=float)
    p2 = np.clip(p2, eps, 1.0)
    s = float(p2.sum())
    if s <= 0:
        return np.full_like(p2, 1.0 / float(len(p2)))
    return p2 / s


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float], *, log_base: float = 2.0) -> float:
    """
    Jensen–Shannon divergence between discrete probability distributions p and q.
    Uses log_base=2 so the output is in [0, 1] (when p,q are normalized).
    """
    p_arr = _clamp_prob(np.asarray(p, dtype=float))
    q_arr = _clamp_prob(np.asarray(q, dtype=float))
    m = 0.5 * (p_arr + q_arr)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        # KL(a||b) = sum a_i * log(a_i / b_i)
        # We clamp inside _clamp_prob so log is safe.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = a / b
            return float(np.sum(a * np.log(ratio) / np.log(log_base)))

    return 0.5 * kl(p_arr, m) + 0.5 * kl(q_arr, m)


def top_k_topic_overlap(p: Sequence[float], q: Sequence[float], *, k: int) -> Dict[str, float]:
    if k <= 0:
        return {"k": float(k), "intersection_size": 0.0, "jaccard": 0.0}
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    k = int(min(k, len(p_arr), len(q_arr)))
    p_top = set(np.argsort(p_arr)[::-1][:k].tolist())
    q_top = set(np.argsort(q_arr)[::-1][:k].tolist())
    inter = len(p_top.intersection(q_top))
    union = len(p_top.union(q_top)) or 1
    jaccard = inter / union
    return {"k": float(k), "intersection_size": float(inter), "jaccard": float(jaccard)}


def cosine_similarity_dense(a: np.ndarray, b: np.ndarray) -> float:
    a2 = np.asarray(a, dtype=float)
    b2 = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a2) * np.linalg.norm(b2))
    if denom <= 0:
        return 0.0
    return float(np.dot(a2, b2) / denom)


def tfidf_cosine_similarity(text_a: str, text_b: str, *, ngram_range: Tuple[int, int] = (1, 1)) -> float:
    vec = TfidfVectorizer(ngram_range=ngram_range, lowercase=True, stop_words="english")
    X = vec.fit_transform([text_a, text_b])
    # cosine similarity for tf-idf vectors: dot(u,v) / (||u||*||v||)
    a = X[0].toarray()[0]
    b = X[1].toarray()[0]
    return cosine_similarity_dense(a, b)


def _extract_keyphrase_candidates(
    text: str,
    *,
    ngram_max: int,
    top_terms: int,
) -> List[str]:
    # Keyphrase candidates = top TF-IDF n-grams from the text.
    # We keep them as "tokens" (including spaces for multi-word n-grams).
    vec = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        lowercase=True,
        stop_words="english",
    )
    X = vec.fit_transform([text])
    if X.shape[1] <= 0:
        return []
    # For a single doc, TF-IDF weight is monotonic to tf; still okay for candidate selection.
    weights = X.toarray()[0]
    vocab = vec.get_feature_names_out().tolist()
    if top_terms <= 0:
        return []
    top_idx = np.argsort(weights)[::-1][:top_terms]
    out = [vocab[i] for i in top_idx if weights[i] > 0]
    return out


def _extract_entity_candidates(text: str, *, top_entities: int, min_len: int = 2) -> List[str]:
    # Heuristic entity-like spans: capitalization patterns.
    # Normalize to lowercase so TF-IDF vocab is stable.
    matches = _CAP_ENTITY_RE.findall(text)
    normed = []
    for m in matches:
        m2 = re.sub(r"\s+", " ", m.strip())
        if len(m2) < min_len:
            continue
        normed.append(m2.lower())
    if not normed:
        return []

    # Term frequency ranking for "top entities"
    freq: Dict[str, int] = {}
    for t in normed:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.keys(), key=lambda k: (freq[k], k), reverse=True)
    return ranked[:top_entities]


def tfidf_cosine_over_term_vocab(
    doc_a_terms: Sequence[str],
    doc_b_terms: Sequence[str],
    *,
    vocab: Sequence[str],
) -> Dict[str, object]:
    """
    TF-IDF cosine similarity where the term space is exactly `vocab`.
    We compute a standard smoothed IDF across two docs:
      idf(t) = log((N + 1) / (df(t) + 1)) + 1, with N=2.
    """
    vocab_list = list(vocab)
    if not vocab_list:
        return {"cosine": 0.0, "vocab_size": 0}

    idx = {t: i for i, t in enumerate(vocab_list)}

    def term_tf(terms: Sequence[str]) -> np.ndarray:
        tf = np.zeros(len(vocab_list), dtype=float)
        for t in terms:
            i = idx.get(t)
            if i is not None:
                tf[i] += 1.0
        return tf

    tf_a = term_tf(doc_a_terms)
    tf_b = term_tf(doc_b_terms)
    # df across two docs (presence/absence)
    df = ((tf_a > 0).astype(float) + (tf_b > 0).astype(float))
    N = 2.0
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0

    tfidf_a = tf_a * idf
    tfidf_b = tf_b * idf
    return {"cosine": cosine_similarity_dense(tfidf_a, tfidf_b), "vocab_size": len(vocab_list)}


def load_text_from_esv_ref(db_path: Path, ref_str: str, *, version: str = "ESV") -> str:
    # Reuse parsing + SQL querying from esv_query.py to stay consistent.
    from esv_query import get_db_path, parse_reference, query_bible, normalize_version

    _ = get_db_path  # silence unused import warning for editors
    version_n = normalize_version(version)
    verse_ref = parse_reference(ref_str)
    # query_esv expects db_path, and book/chapter/verse params.
    # We join verses with a space so TF-IDF sees word continuity.
    verses = query_bible(
        db_path=db_path,
        version=version_n,
        book=verse_ref.book,
        chapter=verse_ref.chapter,
        start_verse=verse_ref.start_verse,
        end_verse=verse_ref.end_verse,
    )
    if not verses:
        return ""
    return " ".join(t for (_v, t) in verses)


def compute_theme_similarity_between_texts(
    text_a: str,
    text_b: str,
    *,
    num_topics: int = 20,
    top_k_topics: int = 5,
    lda_max_features: int = 5000,
    lda_min_df: int = 1,
    lda_max_df: float = 0.9,
    keyphrase_ngram_max: int = 2,
    keyphrase_top_terms: int = 30,
    entity_top_entities: int = 50,
) -> Dict[str, object]:
    text_a = text_a.strip()
    text_b = text_b.strip()

    if not text_a or not text_b:
        return {
            "ok": False,
            "reason": "empty_text_a_or_text_b",
            "tfidf_cosine_fulltext": 0.0,
            "topics": {"ok": False, "reason": "empty_text"},
            "keyphrases": {"cosine": 0.0, "vocab_size": 0, "ok": False},
            "entities": {"cosine": 0.0, "vocab_size": 0, "ok": False},
        }

    # -------------------------
    # A) LDA topics + JSD
    # -------------------------
    # We always fit on exactly 2 documents (text_a, text_b). Some `CountVectorizer`
    # parameter combinations like `min_df=2` and `max_df=0.9` become invalid
    # because max_df translates into an absolute max doc frequency of < min_df.
    n_docs = 2
    abs_max_doc_count = int(math.floor(float(lda_max_df) * float(n_docs)))
    if abs_max_doc_count < int(lda_min_df):
        lda_min_df = 1

    count_vec = CountVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=lda_max_features,
        min_df=lda_min_df,
        max_df=lda_max_df,
    )
    counts = count_vec.fit_transform([text_a, text_b])
    vocab_size = int(counts.shape[1])
    if vocab_size <= 0:
        topics_out: Dict[str, object] = {"ok": False, "reason": "lda_vocab_empty", "vocab_size": 0}
    else:
        lda = LatentDirichletAllocation(
            n_components=int(num_topics),
            random_state=42,
            learning_method="batch",
            max_iter=20,
        )
        lda.fit(counts)
        theta = lda.transform(counts)  # shape: (2, K), each row sums to 1
        p = theta[0, :]
        q = theta[1, :]
        jsd = jensen_shannon_divergence(p, q, log_base=2.0)
        overlap = top_k_topic_overlap(p, q, k=top_k_topics)
        topics_out = {
            "ok": True,
            "num_topics": float(num_topics),
            "vocab_size": float(vocab_size),
            "jsd_topic_distribution": float(jsd),
            "top_k_overlap": overlap,
        }

    # -------------------------
    # B) TF-IDF cosine full text
    # -------------------------
    tfidf_cos = tfidf_cosine_similarity(text_a, text_b, ngram_range=(1, 1))

    # -------------------------
    # C) TF-IDF cosine over extracted keyphrases/entities
    # -------------------------
    keyphrases_a = _extract_keyphrase_candidates(
        text_a,
        ngram_max=keyphrase_ngram_max,
        top_terms=keyphrase_top_terms,
    )
    keyphrases_b = _extract_keyphrase_candidates(
        text_b,
        ngram_max=keyphrase_ngram_max,
        top_terms=keyphrase_top_terms,
    )
    keyphrase_vocab = sorted(set(keyphrases_a).union(set(keyphrases_b)))

    keyphrase_terms_a = keyphrases_a[:]  # candidate list with implicit weights via TF
    keyphrase_terms_b = keyphrases_b[:]
    keyphrase_cos = tfidf_cosine_over_term_vocab(
        keyphrase_terms_a,
        keyphrase_terms_b,
        vocab=keyphrase_vocab,
    )

    entities_a = _extract_entity_candidates(text_a, top_entities=entity_top_entities)
    entities_b = _extract_entity_candidates(text_b, top_entities=entity_top_entities)
    entity_vocab = sorted(set(entities_a).union(set(entities_b)))
    entity_cos = tfidf_cosine_over_term_vocab(entities_a, entities_b, vocab=entity_vocab)

    return {
        "ok": True,
        "tfidf_cosine_fulltext": float(tfidf_cos),
        "topics": topics_out,
        "keyphrases": {
            "ok": True,
            "vocab_size": int(keyphrase_cos["vocab_size"]),  # type: ignore[arg-type]
            "cosine": float(keyphrase_cos["cosine"]),  # type: ignore[arg-type]
            "keyphrases_a_top": keyphrases_a[:10],
            "keyphrases_b_top": keyphrases_b[:10],
        },
        "entities": {
            "ok": True,
            "vocab_size": int(entity_cos["vocab_size"]),  # type: ignore[arg-type]
            "cosine": float(entity_cos["cosine"]),  # type: ignore[arg-type]
            "entities_a_top": entities_a[:10],
            "entities_b_top": entities_b[:10],
        },
        "params": {
            "num_topics": int(num_topics),
            "top_k_topics": int(top_k_topics),
            "lda_max_features": int(lda_max_features),
            "lda_min_df": int(lda_min_df),
            "lda_max_df": float(lda_max_df),
            "keyphrase_ngram_max": int(keyphrase_ngram_max),
            "keyphrase_top_terms": int(keyphrase_top_terms),
            "entity_top_entities": int(entity_top_entities),
        },
    }


def _sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute Bible passage theme similarity metrics.")
    parser.add_argument("--ref-a", type=str, required=True, help='Bible ref like "John 1:1-4".')
    parser.add_argument("--ref-b", type=str, required=True, help='Bible ref like "John 1:14".')
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(__file__).resolve().parent / "SUPER_BIBLE" / "super_bible.db"),
        help="Path to SUPER_BIBLE/super_bible.db.",
    )
    parser.add_argument("--out-json", type=str, default="out/theme_similarity.json", help="Output JSON path.")

    parser.add_argument("--num-topics", type=int, default=20, help="LDA topic count.")
    parser.add_argument("--top-k-topics", type=int, default=5, help="Top-k topic overlap.")
    parser.add_argument("--lda-max-features", type=int, default=5000)
    parser.add_argument("--lda-min-df", type=int, default=1)
    parser.add_argument("--lda-max-df", type=float, default=0.9)

    parser.add_argument("--keyphrase-ngram-max", type=int, default=2)
    parser.add_argument("--keyphrase-top-terms", type=int, default=30)
    parser.add_argument("--entity-top-entities", type=int, default=50)
    parser.add_argument("--version", type=str, default="ESV", help="Bible translation version (default: ESV).")

    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    from esv_query import normalize_version
    version_n = normalize_version(args.version)

    text_a = load_text_from_esv_ref(db_path, args.ref_a, version=version_n)
    text_b = load_text_from_esv_ref(db_path, args.ref_b, version=version_n)

    result = compute_theme_similarity_between_texts(
        text_a,
        text_b,
        num_topics=args.num_topics,
        top_k_topics=args.top_k_topics,
        lda_max_features=args.lda_max_features,
        lda_min_df=args.lda_min_df,
        lda_max_df=args.lda_max_df,
        keyphrase_ngram_max=args.keyphrase_ngram_max,
        keyphrase_top_terms=args.keyphrase_top_terms,
        entity_top_entities=args.entity_top_entities,
    )

    # Make output path deterministic-ish per input unless overridden explicitly.
    out_json = Path(args.out_json).resolve()
    if args.out_json == "out/theme_similarity.json":
        a = _sanitize_filename(args.ref_a)
        b = _sanitize_filename(args.ref_b)
        out_json = out_json.with_name(f"theme_similarity_{a}__vs__{b}_{version_n}.json")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "ref_a": args.ref_a,
                "ref_b": args.ref_b,
                "version": version_n,
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

