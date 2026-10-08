"""Shared recommender interface.

Every recommender produces, per user, a dense score over all items; the base
class turns those scores into a ranked top-k list, optionally masking items the
user has already rated in the training matrix. Subclasses implement
``fit`` (storing whatever they need) and ``score_users``.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores per row, ranked best-first. ``scores`` is (m, n)."""
    n_items = scores.shape[1]
    k = min(k, n_items)
    # argpartition pulls the k largest to the front cheaply; then we sort those k.
    partitioned = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, partitioned], axis=1)
    return partitioned[row, order]


class BaseRecommender:
    """Base class: fit on a user-item matrix, recommend ranked item indices."""

    name: str = "base"
    train_: csr_matrix
    n_users_: int
    n_items_: int

    def fit(self, train: csr_matrix) -> BaseRecommender:
        self.train_ = train.tocsr()
        self.n_users_, self.n_items_ = self.train_.shape
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        """Return a dense (len(user_indices), n_items) array of item scores."""
        raise NotImplementedError

    def recommend(self, user_index: int, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        """Top-k item indices for a single user."""
        users = np.array([user_index])
        return self.recommend_many(users, k=k, exclude_seen=exclude_seen)[0]

    def recommend_many(
        self, user_indices: np.ndarray, k: int = 10, exclude_seen: bool = True
    ) -> np.ndarray:
        """Top-k item indices for each user in ``user_indices`` (shape (m, k))."""
        user_indices = np.asarray(user_indices)
        scores = np.asarray(self.score_users(user_indices), dtype=np.float64)
        if exclude_seen:
            for row, user in enumerate(user_indices):
                start, end = self.train_.indptr[user], self.train_.indptr[user + 1]
                seen = self.train_.indices[start:end]
                scores[row, seen] = -np.inf
        return top_k_indices(scores, k)

    def recommend_all(self, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        """Top-k item indices for every user (shape (n_users, k))."""
        return self.recommend_many(np.arange(self.n_users_), k=k, exclude_seen=exclude_seen)
