"""The leave-last-out split must not leak any user's holdout into training."""

from __future__ import annotations

import numpy as np

from src.data.ratings import make_synthetic_ratings
from src.evaluation.splits import leave_last_out


def test_no_holdout_leaks_into_training():
    data = make_synthetic_ratings(n_users=100, n_items=70, density=0.3, seed=2)
    original = data.observed.tocsr()
    split = leave_last_out(original, n_holdout=2, min_train=3, seed=2)

    for user, held in split.test_items.items():
        start, end = split.train.indptr[user], split.train.indptr[user + 1]
        train_items = set(split.train.indices[start:end].tolist())
        assert train_items.isdisjoint(set(held.tolist()))


def test_split_partitions_the_original_ratings():
    data = make_synthetic_ratings(n_users=100, n_items=70, density=0.3, seed=3)
    original = data.observed.tocsr()
    split = leave_last_out(original, n_holdout=2, min_train=3, seed=3)

    total_held = sum(len(v) for v in split.test_items.values())
    assert split.train.nnz + total_held == original.nnz

    # every held-out item was genuinely a rating in the original matrix
    for user, held in split.test_items.items():
        start, end = original.indptr[user], original.indptr[user + 1]
        original_items = set(original.indices[start:end].tolist())
        assert set(held.tolist()).issubset(original_items)


def test_holdout_count_is_respected():
    data = make_synthetic_ratings(n_users=60, n_items=50, density=0.4, seed=4)
    split = leave_last_out(data.observed, n_holdout=2, min_train=3, seed=4)
    assert split.n_evaluated > 0
    for held in split.test_items.values():
        assert len(held) == 2


def test_timestamps_hold_out_the_most_recent():
    data = make_synthetic_ratings(n_users=30, n_items=40, density=0.5, seed=5)
    matrix = data.observed.tocsr()
    # give each rating an increasing timestamp in storage order
    timestamps = matrix.copy()
    timestamps.data = np.arange(1, matrix.nnz + 1, dtype=np.float64)

    split = leave_last_out(matrix, n_holdout=1, min_train=3, timestamps=timestamps, seed=0)
    for user, held in split.test_items.items():
        start, end = matrix.indptr[user], matrix.indptr[user + 1]
        user_items = matrix.indices[start:end]
        user_times = timestamps.data[start:end]
        latest_item = user_items[np.argmax(user_times)]
        assert held[0] == latest_item
