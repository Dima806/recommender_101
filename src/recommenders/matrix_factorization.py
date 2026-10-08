"""Matrix factorization via SGD, from scratch.

Compress the sparse ratings matrix into two small tables: a vector of latent
"taste" dimensions per user and the same dimensions per item. A rating is
predicted by the dot product of a user vector and an item vector (plus global,
user, and item biases). The latent dimensions are learned, not labelled, and end
up capturing taste — this is exactly the embeddings idea applied to users and
items, trained with the gradient descent you already know.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.recommenders.base import BaseRecommender


class MatrixFactorization(BaseRecommender):
    name = "matrix_factorization"

    user_factors_: np.ndarray
    item_factors_: np.ndarray
    user_bias_: np.ndarray
    item_bias_: np.ndarray
    global_mean_: float
    history_: list[float]

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 30,
        lr: float = 0.01,
        reg: float = 0.05,
        use_bias: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.use_bias = use_bias
        self.seed = seed

    def fit(self, train: csr_matrix) -> MatrixFactorization:
        super().fit(train)
        coo = self.train_.tocoo()
        users = coo.row.astype(np.intp)
        items = coo.col.astype(np.intp)
        ratings = coo.data.astype(np.float64)
        n_ratings = ratings.shape[0]

        rng = np.random.default_rng(self.seed)
        self.user_factors_ = rng.normal(0.0, 0.1, size=(self.n_users_, self.n_factors))
        self.item_factors_ = rng.normal(0.0, 0.1, size=(self.n_items_, self.n_factors))
        self.user_bias_ = np.zeros(self.n_users_)
        self.item_bias_ = np.zeros(self.n_items_)
        self.global_mean_ = float(ratings.mean()) if self.use_bias else 0.0
        self.history_ = []

        order = np.arange(n_ratings)
        for _ in range(self.n_epochs):
            rng.shuffle(order)
            squared_error = 0.0
            for idx in order:
                user = users[idx]
                item = items[idx]
                user_vec = self.user_factors_[user]
                item_vec = self.item_factors_[item]

                pred = self.global_mean_ + self.user_bias_[user] + self.item_bias_[item]
                pred += float(user_vec @ item_vec)
                err = ratings[idx] - pred
                squared_error += err * err

                if self.use_bias:
                    self.user_bias_[user] += self.lr * (err - self.reg * self.user_bias_[user])
                    self.item_bias_[item] += self.lr * (err - self.reg * self.item_bias_[item])

                # update both factor vectors using the *pre-update* user vector
                self.user_factors_[user] += self.lr * (err * item_vec - self.reg * user_vec)
                self.item_factors_[item] += self.lr * (err * user_vec - self.reg * item_vec)

            self.history_.append(float(np.sqrt(squared_error / n_ratings)))
        return self

    def predict(self, user_index: int, item_index: int) -> float:
        """Predicted rating for a single (user, item) pair."""
        dot = float(self.user_factors_[user_index] @ self.item_factors_[item_index])
        return self.global_mean_ + self.user_bias_[user_index] + self.item_bias_[item_index] + dot

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        user_vecs = self.user_factors_[user_indices]
        scores = user_vecs @ self.item_factors_.T
        scores = scores + self.global_mean_ + self.item_bias_[None, :]
        scores = scores + self.user_bias_[user_indices][:, None]
        return scores

    def reconstruct(self) -> np.ndarray:
        """Full predicted (n_users, n_items) rating matrix."""
        return self.score_users(np.arange(self.n_users_))
