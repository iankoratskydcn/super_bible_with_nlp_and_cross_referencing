import tempfile
from pathlib import Path
import unittest

import numpy as np

from rank_verses_vs_seeds_theme_parallel import (
    compute_memoized_representations,
    load_or_compute_memoized_representations_cached,
    make_reps_cache_key,
)


class TestRepsCache(unittest.TestCase):
    def test_make_reps_cache_key_stable(self):
        docs_nodes = ["A", "B", "C"]
        params = {"x": 1, "y": "z"}
        k1 = make_reps_cache_key(docs_nodes=docs_nodes, seed_count=1, cand_count=2, params=params)
        k2 = make_reps_cache_key(docs_nodes=list(docs_nodes), seed_count=1, cand_count=2, params=dict(params))
        self.assertEqual(k1, k2)

    def test_compute_memoized_representations_dtypes(self):
        docs_texts = ["God is love.", "God loves the world.", "Love is patient."]
        seed_indices = [0]
        cand_indices = [1, 2]
        reps = compute_memoized_representations(
            docs_texts=docs_texts,
            seed_indices=seed_indices,
            cand_indices=cand_indices,
            lda_num_topics=2,
            lda_max_features=20,
            lda_min_df=1,
            lda_max_df=0.9,
            lda_top_k_topics=1,
            tfidf_full_max_features=20,
            keyphrase_ngram_max=1,
            keyphrase_top_terms=5,
            keyphrase_tfidf_max_features=20,
            entity_top_entities=5,
        )
        self.assertEqual(reps["theta"].dtype, np.float32)
        self.assertEqual(reps["topk_indices"].dtype, np.int32)
        self.assertEqual(reps["X_full_seed"].dtype, np.float32)

    def test_cache_load_returns_same_shapes(self):
        docs_texts = ["God is love.", "God loves the world.", "Love is patient."]
        docs_nodes = ["A", "B", "C"]
        seed_indices = [0]
        cand_indices = [1, 2]
        params = {
            "lda_num_topics": 2,
            "lda_max_features": 20,
            "lda_min_df": 1,
            "lda_max_df": 0.9,
            "lda_top_k_topics": 1,
            "tfidf_full_max_features": 20,
            "keyphrase_ngram_max": 1,
            "keyphrase_top_terms": 5,
            "keyphrase_tfidf_max_features": 20,
            "entity_top_entities": 5,
            "stop_words": "english",
            "tfidf_ent_max_features": 20000,
            "keyphrase_algo": "mask_top_terms_per_row",
            "float32": True,
        }
        with tempfile.TemporaryDirectory() as d:
            cache_dir = Path(d)
            reps1 = load_or_compute_memoized_representations_cached(
                docs_texts=docs_texts,
                seed_indices=seed_indices,
                cand_indices=cand_indices,
                docs_nodes=docs_nodes,
                use_cache=True,
                cache_dir=cache_dir,
                cache_compress=False,
                cache_key_params=params,
                lda_num_topics=2,
                lda_max_features=20,
                lda_min_df=1,
                lda_max_df=0.9,
                lda_top_k_topics=1,
                tfidf_full_max_features=20,
                keyphrase_ngram_max=1,
                keyphrase_top_terms=5,
                keyphrase_tfidf_max_features=20,
                entity_top_entities=5,
            )
            reps2 = load_or_compute_memoized_representations_cached(
                docs_texts=docs_texts,
                seed_indices=seed_indices,
                cand_indices=cand_indices,
                docs_nodes=docs_nodes,
                use_cache=True,
                cache_dir=cache_dir,
                cache_compress=False,
                cache_key_params=params,
                lda_num_topics=2,
                lda_max_features=20,
                lda_min_df=1,
                lda_max_df=0.9,
                lda_top_k_topics=1,
                tfidf_full_max_features=20,
                keyphrase_ngram_max=1,
                keyphrase_top_terms=5,
                keyphrase_tfidf_max_features=20,
                entity_top_entities=5,
            )
            self.assertEqual(reps1["theta"].shape, reps2["theta"].shape)
            self.assertEqual(reps1["X_full_seed"].shape, reps2["X_full_seed"].shape)


if __name__ == "__main__":
    unittest.main()

