"""Honest train/test splitting: hold out each user's most-recent interactions.

Leave-last-out per user is the realistic protocol for top-N recommendation: we
train on a user's history and ask the model to rank the items they interacted
with *next*. The invariant that makes the evaluation honest — no held-out
interaction may appear in training — is asserted in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Split:
    """A train matrix and the held-out item indices per evaluated user."""

    train: csr_matrix
    test_items: dict[int, np.ndarray]

    @property
    def n_evaluated(self) -> int:
        return len(self.test_items)


def leave_last_out(
    matrix: csr_matrix,
    n_holdout: int = 1,
    min_train: int = 1,
    timestamps: csr_matrix | None = None,
    seed: int = 42,
) -> Split:
    """Hold out each user's ``n_holdout`` most-recent (or, lacking timestamps, random) items.

    Users with fewer than ``n_holdout + min_train`` ratings are left entirely in
    training and not evaluated.
    """
    matrix = matrix.tocsr()
    ts = timestamps.tocsr() if timestamps is not None else None
    rng = np.random.default_rng(seed)

    train = matrix.tolil(copy=True)
    test_items: dict[int, np.ndarray] = {}

    for user in range(matrix.shape[0]):
        start, end = matrix.indptr[user], matrix.indptr[user + 1]
        items = matrix.indices[start:end]
        if len(items) < n_holdout + min_train:
            continue

        if ts is not None:
            times = np.asarray(ts[user].todense()).ravel()[items]
            held = items[np.argsort(times)[::-1][:n_holdout]]
        else:
            held = rng.choice(items, size=n_holdout, replace=False)

        for item in held:
            train[user, item] = 0.0
        test_items[user] = np.sort(held)

    train_csr = train.tocsr()
    train_csr.eliminate_zeros()
    return Split(train=train_csr, test_items=test_items)
