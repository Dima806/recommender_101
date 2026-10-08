"""Matrix factorization must recover planted structure, beat popularity, be deterministic."""

from __future__ import annotations

import numpy as np

from src.data.ratings import make_synthetic_ratings
from src.evaluation.ranking import ndcg_at_k
from src.evaluation.splits import leave_last_out
from src.recommenders.matrix_factorization import MatrixFactorization
from src.recommenders.popularity import PopularityRecommender


def _grouped():
    # Strong block structure so personalization clearly matters.
    return make_synthetic_ratings(
        n_users=120, n_items=80, n_groups=4, density=0.25, noise=0.2, seed=7
    )


def test_recovers_planted_structure():
    data = _grouped()
    observed = data.observed
    mf = MatrixFactorization(n_factors=8, n_epochs=60, lr=0.02, reg=0.02, seed=0)
    mf.fit(observed)

    predicted = mf.reconstruct()
    held_out = observed.toarray() == 0  # entries we never trained on
    true_vals = data.true_ratings[held_out]
    pred_vals = predicted[held_out]
    corr = np.corrcoef(true_vals, pred_vals)[0, 1]
    assert corr > 0.9, f"MF did not recover planted structure (corr={corr:.3f})"


def test_mf_beats_popularity_on_ranking():
    data = _grouped()
    split = leave_last_out(data.observed, n_holdout=2, min_train=3, seed=0)

    pop = PopularityRecommender().fit(split.train)
    mf = MatrixFactorization(n_factors=8, n_epochs=60, lr=0.02, reg=0.02, seed=0).fit(split.train)

    pop_recs = pop.recommend_all(k=10)
    mf_recs = mf.recommend_all(k=10)

    pop_ndcg = np.mean([ndcg_at_k(pop_recs[u], rel, 10) for u, rel in split.test_items.items()])
    mf_ndcg = np.mean([ndcg_at_k(mf_recs[u], rel, 10) for u, rel in split.test_items.items()])
    assert mf_ndcg > pop_ndcg, f"MF ({mf_ndcg:.3f}) did not beat popularity ({pop_ndcg:.3f})"


def test_training_loss_decreases():
    data = _grouped()
    mf = MatrixFactorization(n_factors=8, n_epochs=40, lr=0.02, reg=0.02, seed=0).fit(
        data.observed
    )
    assert mf.history_[-1] < mf.history_[0]


def test_deterministic_with_fixed_seed():
    data = _grouped()
    a = MatrixFactorization(n_factors=8, n_epochs=20, seed=123).fit(data.observed)
    b = MatrixFactorization(n_factors=8, n_epochs=20, seed=123).fit(data.observed)
    assert np.allclose(a.user_factors_, b.user_factors_)
    assert np.allclose(a.item_factors_, b.item_factors_)
