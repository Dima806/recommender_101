"""Item-based collaborative filtering, from scratch — the dual of user-based.

"Find items similar to the ones you liked and recommend those." Similarity is
cosine between items' columns (who rated them). Item similarities change slowly
and are often more stable and scalable than user-based, which is why item-based
CF powered early large-scale systems.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.recommenders.base import BaseRecommender
from src.recommenders.similarity import topk_cosine


class ItemKNNRecommender(BaseRecommender):
    name = "item_knn"
    scores_: np.ndarray

    def __init__(self, n_neighbors: int = 40) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, train: csr_matrix) -> ItemKNNRecommender:
        super().fit(train)
        # similarity[i, j] = cosine(item i, item j) over the users who rated them
        similarity = topk_cosine(self.train_.T.tocsr(), self.n_neighbors)
        # score[u, item] = sum over items j the user rated of sim(item, j) * rating(u, j)
        self.scores_ = np.asarray((self.train_ @ similarity.T).toarray(), dtype=np.float64)
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.scores_[user_indices]
