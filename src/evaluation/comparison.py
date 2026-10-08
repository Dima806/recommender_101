"""The arena: fit every recommender on one split and build the scoreboard.

The scoreboard puts accuracy metrics next to the beyond-accuracy ones, with the
popularity baseline as the first row, so the honest verdict is readable at a
glance: matrix factorization tends to win overall but by a modest margin,
collaborative filtering is inconsistent against popularity, and popularity itself
has poor novelty and diversity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.evaluation.beyond_accuracy import (
    intra_list_diversity,
    novelty,
    popularity_bias,
)
from src.evaluation.ranking import catalog_coverage, evaluate_ranking
from src.evaluation.splits import Split
from src.recommenders.base import BaseRecommender


def run_arena(
    recommenders: dict[str, BaseRecommender],
    split: Split,
    k: int = 10,
    item_features: csr_matrix | None = None,
    fit: bool = True,
) -> pd.DataFrame:
    """Evaluate each recommender and return a per-method metric scoreboard."""
    evaluated_users = np.array(sorted(split.test_items))
    n_items = split.train.shape[1]
    names: list[str] = []
    rows: list[dict[str, float]] = []

    for name, model in recommenders.items():
        if fit:
            model.fit(split.train)
        recommendations = model.recommend_all(k=k)
        evaluated = recommendations[evaluated_users]

        metrics = evaluate_ranking(recommendations, split.test_items, k)
        metrics["coverage"] = catalog_coverage(evaluated, n_items)
        metrics["novelty"] = novelty(evaluated, split.train)
        metrics["pop_bias"] = popularity_bias(evaluated, split.train)
        if item_features is not None:
            metrics["diversity"] = intra_list_diversity(evaluated, item_features)
        names.append(name)
        rows.append(metrics)

    return pd.DataFrame(rows, index=names).rename_axis("method")
