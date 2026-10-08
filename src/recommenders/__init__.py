"""The five recommenders, all built from scratch, plus a default registry."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from src.config import Settings, get_settings
from src.recommenders.base import BaseRecommender, top_k_indices
from src.recommenders.content import ContentRecommender
from src.recommenders.item_knn import ItemKNNRecommender
from src.recommenders.matrix_factorization import MatrixFactorization
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.user_knn import UserKNNRecommender

__all__ = [
    "BaseRecommender",
    "ContentRecommender",
    "ItemKNNRecommender",
    "MatrixFactorization",
    "PopularityRecommender",
    "UserKNNRecommender",
    "build_default_recommenders",
    "top_k_indices",
]


def build_default_recommenders(
    item_features: csr_matrix | None = None,
    settings: Settings | None = None,
) -> dict[str, BaseRecommender]:
    """Construct every recommender with parameters from settings, keyed by name.

    Content-based is included only when ``item_features`` is provided. The
    popularity baseline is listed first on purpose.
    """
    settings = settings or get_settings()
    recommenders: dict[str, BaseRecommender] = {
        "popularity": PopularityRecommender(),
        "user_knn": UserKNNRecommender(n_neighbors=settings.knn.n_neighbors),
        "item_knn": ItemKNNRecommender(n_neighbors=settings.knn.n_neighbors),
        "matrix_factorization": MatrixFactorization(
            n_factors=settings.mf.n_factors,
            n_epochs=settings.mf.n_epochs,
            lr=settings.mf.lr,
            reg=settings.mf.reg,
            use_bias=settings.mf.use_bias,
            seed=settings.seed,
        ),
    }
    if item_features is not None:
        recommenders["content"] = ContentRecommender(item_features)
    return recommenders
