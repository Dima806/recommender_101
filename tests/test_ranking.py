"""NDCG from scratch must match an independent reference implementation to 1e-9."""

from __future__ import annotations

import math

import numpy as np

from src.evaluation.ranking import (
    average_precision_at_k,
    catalog_coverage,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def reference_ndcg(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """A plain-Python NDCG@k reference written independently of the library version."""
    relevant_set = set(int(x) for x in relevant)
    dcg = 0.0
    for position, item in enumerate(recommended[:k]):
        if int(item) in relevant_set:
            dcg += 1.0 / math.log2(position + 2)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def test_ndcg_matches_reference_on_random_cases():
    rng = np.random.default_rng(0)
    n_items, k = 50, 10
    for _ in range(200):
        recommended = rng.permutation(n_items)[:k]
        n_relevant = rng.integers(1, 12)
        relevant = rng.choice(n_items, size=n_relevant, replace=False)
        assert (
            abs(ndcg_at_k(recommended, relevant, k) - reference_ndcg(recommended, relevant, k))
            < 1e-9
        )


def test_ndcg_perfect_ranking_is_one():
    relevant = np.array([3, 7, 1])
    recommended = np.array([3, 7, 1, 9, 2])
    assert abs(ndcg_at_k(recommended, relevant, 5) - 1.0) < 1e-12


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k(np.array([0, 1, 2]), np.array([7, 8]), 3) == 0.0


def test_precision_and_recall_known_values():
    recommended = np.array([1, 2, 3, 4])
    relevant = np.array([2, 4, 9])
    assert precision_at_k(recommended, relevant, 4) == 0.5  # 2 of 4
    assert recall_at_k(recommended, relevant, 4) == 2 / 3  # 2 of 3 relevant


def test_average_precision_known_value():
    recommended = np.array([2, 5, 4, 1])  # relevant at ranks 1 and 3
    relevant = np.array([2, 4])
    # AP = (1/1 + 2/3) / 2
    assert abs(average_precision_at_k(recommended, relevant, 4) - (1.0 + 2 / 3) / 2) < 1e-12


def test_catalog_coverage():
    recommendations = np.array([[0, 1], [1, 2], [2, 3]])  # unique {0,1,2,3}
    assert catalog_coverage(recommendations, n_items=8) == 0.5
