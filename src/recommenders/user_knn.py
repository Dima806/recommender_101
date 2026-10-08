"""User-based collaborative filtering, from scratch.

"Find the users most similar to you and copy what they liked." Similarity is
cosine between users' rating rows; a user's predicted score for an item is the
similarity-weighted sum of their neighbours' ratings for that item. Its weakness
is visible by construction: a brand-new user has an all-zero row, so every
similarity is zero and the method falls back to noise — the cold-start problem.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.recommenders.base import BaseRecommender
from src.recommenders.similarity import topk_cosine


class UserKNNRecommender(BaseRecommender):
    name = "user_knn"
    scores_: np.ndarray

    def __init__(self, n_neighbors: int = 40) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, train: csr_matrix) -> UserKNNRecommender:
        super().fit(train)
        # similarity[u, v] = cosine(user u, user v), top-k neighbours only
        similarity = topk_cosine(self.train_, self.n_neighbors)
        # score[u, item] = sum over neighbours v of sim(u, v) * rating(v, item)
        self.scores_ = np.asarray((similarity @ self.train_).toarray(), dtype=np.float64)
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.scores_[user_indices]
