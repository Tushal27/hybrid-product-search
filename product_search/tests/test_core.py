"""
Fast tests for the pieces that don't need the 890k-product index.
Run:  python -m unittest discover -s product_search/tests -v
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine import BM25Index, DenseIndex, SearchEngine, reciprocal_rank_fusion
from evaluate import mrr_at_k, ndcg_at_k, recall_at_k


class MetricTests(unittest.TestCase):
    def test_perfect_ranking_scores_one(self):
        j = {"a": 1.0, "b": 0.1}
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "x"], j), 1.0)

    def test_swapped_ranking_scores_less(self):
        j = {"a": 1.0, "b": 0.1}
        self.assertLess(ndcg_at_k(["b", "a"], j), 1.0)

    def test_ndcg_known_value(self):
        # only item at rank 2 is relevant: DCG = 1/log2(3); ideal DCG = 1/log2(2) = 1
        self.assertAlmostEqual(ndcg_at_k(["x", "a"], {"a": 1.0}), 1 / math.log2(3))

    def test_nothing_found_scores_zero(self):
        self.assertEqual(ndcg_at_k(["x", "y"], {"a": 1.0}), 0.0)

    def test_mrr_ignores_substitutes(self):
        self.assertEqual(mrr_at_k(["s", "a"], {"s": 0.1, "a": 1.0}), 0.5)

    def test_recall(self):
        self.assertEqual(recall_at_k(["a", "x"], {"a": 1.0, "b": 1.0}), 0.5)


class FusionTests(unittest.TestCase):
    def test_item_in_both_lists_wins(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [9, 3, 8]])
        self.assertEqual(fused[0][0], 3)

    def test_weights_shift_the_winner(self):
        fused = reciprocal_rank_fusion([[1, 2], [2, 1]], weights=[5.0, 1.0])
        self.assertEqual(fused[0][0], 1)


class DenseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(500, 16)).astype(np.float32)
        self.v = v / np.linalg.norm(v, axis=1, keepdims=True)

    def test_flat_matches_brute_force(self):
        q = self.v[7]
        ids, _ = DenseIndex(self.v).search(q, 10)
        expected = np.argsort(-(self.v @ q))[:10]
        self.assertEqual(list(ids), list(expected))

    def test_mask_excludes_disallowed(self):
        mask = np.zeros(500, dtype=bool)
        mask[100:200] = True
        ids, _ = DenseIndex(self.v).search(self.v[7], 20, mask)
        self.assertTrue(all(100 <= i < 200 for i in ids))
        self.assertEqual(len(ids), 20)

    def test_hnsw_finds_nearly_the_same_neighbours(self):
        exact, _ = DenseIndex(self.v).search(self.v[3], 10)
        approx, _ = DenseIndex(self.v, kind="hnsw").search(self.v[3], 10)
        self.assertGreaterEqual(len(set(exact) & set(approx)), 8)


class BM25Tests(unittest.TestCase):
    DOCS = ["red toy fire truck", "blue toy submarine", "lego star wars millennium falcon 75192", "plush teddy bear"]

    def test_exact_model_number_found(self):
        ids, _ = BM25Index(self.DOCS).search("75192", 3)
        self.assertEqual(ids[0], 2)

    def test_stemming_matches_plurals(self):
        ids, _ = BM25Index(self.DOCS).search("trucks", 3)
        self.assertEqual(ids[0], 0)

    def test_mask_applies(self):
        mask = np.array([False, True, True, True])
        ids, _ = BM25Index(self.DOCS).search("toy", 3, mask)
        self.assertNotIn(0, ids)

    def test_unknown_word_returns_nothing(self):
        ids, _ = BM25Index(self.DOCS).search("zzzzqq", 3)
        self.assertEqual(len(ids), 0)


class FilterTests(unittest.TestCase):
    def test_price_filter_excludes_unknown_prices(self):
        cat = pd.DataFrame({"pid": list("abc"), "title": list("xyz"), "brand": ["A", "B", None],
                            "price": [5.0, 50.0, np.nan], "rating": [4.0, 5.0, 3.0], "category": [""] * 3})
        eng = SearchEngine(cat, None, None, None)
        self.assertEqual(list(eng._mask(None, 10.0, None, None)), [True, False, False])
        self.assertEqual(list(eng._mask(None, None, 4.5, None)), [False, True, False])
        self.assertEqual(list(eng._mask(None, None, None, "a")), [True, False, False])
        self.assertIsNone(eng._mask(None, None, None, None))


if __name__ == "__main__":
    unittest.main()
