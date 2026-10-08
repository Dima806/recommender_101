"""Every recommender must return a valid top-k list for every user."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.ratings import make_synthetic_ratings
from src.evaluation.splits import leave_last_out
from src.recommenders import build_default_recommenders


@pytest.fixture
def synthetic():
    return make_synthetic_ratings(n_users=80, n_items=60, n_groups=4, density=0.25, seed=1)


@pytest.fixture
def split(synthetic):
    return leave_last_out(synthetic.observed, n_holdout=2, min_train=3, seed=1)


@pytest.fixture
def fitted(synthetic, split):
    models = build_default_recommenders(item_features=synthetic.item_features)
    for model in models.values():
        model.fit(split.train)
    return models


K = 10


def test_all_methods_present(fitted):
    assert set(fitted) == {
        "popularity",
        "user_knn",
        "item_knn",
        "matrix_factorization",
        "content",
    }


def test_recommend_all_shape_and_range(fitted, synthetic):
    n_items = synthetic.observed.shape[1]
    for name, model in fitted.items():
        recs = model.recommend_all(k=K)
        assert recs.shape == (synthetic.observed.shape[0], K), name
        assert recs.min() >= 0 and recs.max() < n_items, name
        assert not np.isnan(recs).any(), name


def test_top_k_are_distinct_per_user(fitted):
    for name, model in fitted.items():
        recs = model.recommend_all(k=K)
        for row in recs:
            assert len(np.unique(row)) == K, name


def test_exclude_seen_never_recommends_training_items(fitted, split):
    train = split.train
    for name, model in fitted.items():
        recs = model.recommend_all(k=K, exclude_seen=True)
        for user in range(train.shape[0]):
            start, end = train.indptr[user], train.indptr[user + 1]
            seen = set(train.indices[start:end].tolist())
            recommended = set(recs[user].tolist())
            assert seen.isdisjoint(recommended), f"{name} leaked a seen item for user {user}"


def test_single_user_recommend_matches_batch(fitted):
    for name, model in fitted.items():
        batch = model.recommend_all(k=K)
        for user in (0, 5, 17):
            single = model.recommend(user, k=K)
            assert np.array_equal(single, batch[user]), name
