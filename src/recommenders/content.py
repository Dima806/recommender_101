"""Content-based recommendation using item features (e.g. genres).

Build a taste profile for each user as the (rating-weighted) average of the
feature vectors of items they liked, then score every item by cosine similarity
to that profile. Because it needs no *other* users, it keeps working for
cold-start items that have features but no ratings yet — the complement that
fixes collaborative filtering's blind spot.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.recommenders.base import BaseRecommender


class ContentRecommender(BaseRecommender):
    name = "content"
    scores_: np.ndarray

    def __init__(self, item_features: csr_matrix) -> None:
        self.item_features = csr_matrix(item_features)

    def fit(self, train: csr_matrix) -> ContentRecommender:
        super().fit(train)
        features = normalize(self.item_features.astype(np.float64), norm="l2", axis=1)
        # profile[u] = sum over items of rating(u, item) * feature(item)
        profiles = self.train_ @ features
        # score[u, item] = cosine-like match between the user's profile and the item
        self.scores_ = np.asarray((profiles @ features.T).toarray(), dtype=np.float64)
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.scores_[user_indices]
