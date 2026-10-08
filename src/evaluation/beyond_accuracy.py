"""Beyond-accuracy metrics: novelty, diversity, and popularity bias.

Accuracy metrics hide a cost: a recommender that only surfaces popular items
scores respectably yet offers no discovery. These metrics make that visible.
They operate on the matrix of recommendations (n_evaluated_users, k).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.data.ratings import item_popularity


def novelty(recommendations: np.ndarray, train: csr_matrix) -> float:
    """Mean self-information of recommended items: rarer items are more novel."""
    popularity = item_popularity(train)
    prob = np.clip(popularity / max(train.shape[0], 1), 1e-12, 1.0)
    self_information = -np.log2(prob)
    return float(self_information[recommendations].mean())


def intra_list_diversity(recommendations: np.ndarray, item_features: csr_matrix) -> float:
    """Mean pairwise (1 - cosine) between items within each recommendation list."""
    features = normalize(csr_matrix(item_features).astype(np.float64), norm="l2", axis=1)
    k = recommendations.shape[1]
    if k < 2:
        return 0.0
    upper = np.triu_indices(k, k=1)
    diversities = []
    for row in recommendations:
        sims = (features[row] @ features[row].T).toarray()
        diversities.append(1.0 - float(sims[upper].mean()))
    return float(np.mean(diversities))


def popularity_bias(recommendations: np.ndarray, train: csr_matrix) -> float:
    """Mean popularity (rating support) of recommended items — higher means more biased."""
    popularity = item_popularity(train)
    return float(popularity[recommendations].mean())
