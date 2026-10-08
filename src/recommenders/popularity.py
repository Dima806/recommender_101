"""The popularity baseline: recommend the most-rated items to everyone.

It ignores the individual user entirely. This is the mandatory yardstick — report
it FIRST, because a surprisingly large fraction of ranking metrics reward getting
the obvious hits right, and the fancy methods may not clear this bar.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.recommenders.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    name = "popularity"
    item_scores_: np.ndarray

    def fit(self, train: csr_matrix) -> PopularityRecommender:
        super().fit(train)
        support = (self.train_ > 0).astype(np.float64)
        self.item_scores_ = np.asarray(support.sum(axis=0)).ravel()
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return np.tile(self.item_scores_, (len(user_indices), 1))
