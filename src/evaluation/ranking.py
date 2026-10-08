"""Ranking metrics for top-N recommendation.

Recommenders are judged by how they *rank* items, not by rating-prediction
accuracy (the metrics_101 lesson). Relevance is binary: an item is relevant to a
user if it is in that user's held-out set. All functions take a ranked array of
recommended item indices and an array of relevant item indices.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    if k == 0:
        return 0.0
    hits = np.isin(recommended[:k], relevant).sum()
    return float(hits) / k


def recall_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Fraction of the relevant items that appear in the top-k."""
    if len(relevant) == 0:
        return 0.0
    hits = np.isin(recommended[:k], relevant).sum()
    return float(hits) / len(relevant)


def ndcg_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Normalized discounted cumulative gain at k with binary relevance."""
    top = recommended[:k]
    gains = np.isin(top, relevant).astype(np.float64)
    if gains.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float((gains * discounts).sum())
    ideal_hits = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_hits + 2))).sum())
    return dcg / idcg


def average_precision_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Average precision at k — the per-user term that MAP averages over users."""
    if len(relevant) == 0:
        return 0.0
    relevant_set = set(relevant.tolist())
    hits = 0
    score = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if int(item) in relevant_set:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def catalog_coverage(recommendations: np.ndarray, n_items: int) -> float:
    """Fraction of the item catalog that ever appears in any recommendation list."""
    if n_items == 0:
        return 0.0
    return float(len(np.unique(recommendations))) / n_items


def evaluate_ranking(
    recommendations: np.ndarray, test_items: dict[int, np.ndarray], k: int
) -> dict[str, float]:
    """Mean ranking metrics over the evaluated users (those with a held-out set)."""
    precisions, recalls, ndcgs, aps = [], [], [], []
    for user, relevant in test_items.items():
        recommended = recommendations[user]
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        aps.append(average_precision_at_k(recommended, relevant, k))
    return {
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "map": float(np.mean(aps)) if aps else 0.0,
    }
