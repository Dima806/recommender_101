"""Data loading: MovieLens-small and a synthetic planted-structure generator."""

from src.data.ratings import (
    RatingsData,
    SyntheticRatings,
    item_popularity,
    load_movielens,
    make_synthetic_ratings,
)

__all__ = [
    "RatingsData",
    "SyntheticRatings",
    "item_popularity",
    "load_movielens",
    "make_synthetic_ratings",
]
