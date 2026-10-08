"""MovieLens-small loader and a synthetic ratings generator.

The real loader downloads MovieLens-small once, caches a parquet copy under
``data/``, and prefers that cache on every subsequent call so the project keeps
working if the download URL rots (commit the cache to freeze it).

The synthetic generator plants *group* structure: users and items each belong to
one of ``n_groups`` latent groups, and a user rates items in their own group
highly and others low. This gives matrix factorization a known taste structure to
recover, and makes personalization genuinely beat the global popularity baseline.
Run ``python -m src.data.ratings`` to download and cache the real dataset.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.config import get_settings

DATA_DIR = Path("data")


@dataclass
class RatingsData:
    """A ratings dataset as a sparse user-item matrix plus item metadata."""

    matrix: csr_matrix  # (n_users, n_items), explicit ratings; 0 means "unrated"
    user_ids: np.ndarray  # original ids, indexed by internal user index
    item_ids: np.ndarray  # original ids, indexed by internal item index
    timestamps: csr_matrix | None = None  # same layout as ``matrix`` when available
    item_features: csr_matrix | None = None  # (n_items, n_genres) multi-hot
    item_titles: list[str] | None = None
    genre_names: list[str] | None = None

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]


@dataclass
class SyntheticRatings:
    """Synthetic ratings with known planted latent (group) structure."""

    observed: csr_matrix  # (n_users, n_items) observed ratings; 0 means missing
    true_ratings: np.ndarray  # (n_users, n_items) full ground-truth rating matrix
    user_group: np.ndarray  # latent group per user
    item_group: np.ndarray  # latent group per item
    item_features: csr_matrix  # (n_items, n_groups) one-hot group features


def item_popularity(matrix: csr_matrix) -> np.ndarray:
    """Support of each item: how many users have rated it."""
    binary = (matrix > 0).astype(np.float64)
    return np.asarray(binary.sum(axis=0)).ravel()


def make_synthetic_ratings(
    n_users: int = 120,
    n_items: int = 80,
    n_groups: int = 4,
    density: float = 0.2,
    noise: float = 0.3,
    high: float = 5.0,
    low: float = 1.0,
    min_per_user: int = 4,
    seed: int = 0,
) -> SyntheticRatings:
    """Generate ratings with block (group) structure and a sampled observation mask."""
    rng = np.random.default_rng(seed)
    user_group = rng.integers(0, n_groups, size=n_users)
    item_group = rng.integers(0, n_groups, size=n_items)

    same_group = user_group[:, None] == item_group[None, :]
    base = np.where(same_group, high, low)
    true_ratings = np.clip(base + rng.normal(0.0, noise, size=base.shape), low, high)

    mask = rng.random(size=(n_users, n_items)) < density
    for user in range(n_users):
        if mask[user].sum() < min_per_user:
            cols = rng.choice(n_items, size=min_per_user, replace=False)
            mask[user, cols] = True

    observed = csr_matrix(np.where(mask, true_ratings, 0.0))
    one_hot = np.zeros((n_items, n_groups), dtype=np.float64)
    one_hot[np.arange(n_items), item_group] = 1.0
    return SyntheticRatings(
        observed=observed,
        true_ratings=true_ratings,
        user_group=user_group,
        item_group=item_group,
        item_features=csr_matrix(one_hot),
    )


def _build_matrix(
    ratings: pd.DataFrame,
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    """Turn a long (userId, movieId, rating, timestamp) frame into sparse matrices."""
    user_ids = np.sort(ratings["userId"].unique())
    item_ids = np.sort(ratings["movieId"].unique())
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    rows = ratings["userId"].map(user_index).to_numpy()
    cols = ratings["movieId"].map(item_index).to_numpy()
    shape = (len(user_ids), len(item_ids))

    matrix = csr_matrix((ratings["rating"].to_numpy(dtype=np.float64), (rows, cols)), shape=shape)
    timestamps = csr_matrix(
        (ratings["timestamp"].to_numpy(dtype=np.float64), (rows, cols)), shape=shape
    )
    return matrix, timestamps, user_ids, item_ids


def _genre_features(
    movies: pd.DataFrame, item_ids: np.ndarray
) -> tuple[csr_matrix, list[str], list[str]]:
    """Multi-hot genre features aligned to ``item_ids`` plus titles and genre names."""
    movies = movies.set_index("movieId")
    genre_lists = [
        movies.loc[iid, "genres"].split("|") if iid in movies.index else [] for iid in item_ids
    ]
    genre_names = sorted({g for gs in genre_lists for g in gs if g != "(no genres listed)"})
    genre_index = {g: j for j, g in enumerate(genre_names)}

    features = np.zeros((len(item_ids), len(genre_names)), dtype=np.float64)
    for i, genres in enumerate(genre_lists):
        for g in genres:
            if g in genre_index:
                features[i, genre_index[g]] = 1.0

    titles = [
        str(movies.loc[iid, "title"]) if iid in movies.index else str(iid) for iid in item_ids
    ]
    return csr_matrix(features), titles, genre_names


def load_movielens(
    data_dir: Path = DATA_DIR,
    url: str | None = None,
    force_download: bool = False,
) -> RatingsData:
    """Load MovieLens-small, preferring the cached frozen copy under ``data_dir``."""
    settings = get_settings()
    url = url or settings.data.url
    cache_ratings = data_dir / "movielens_ratings.parquet"
    cache_movies = data_dir / "movielens_movies.parquet"

    if force_download or not (cache_ratings.exists() and cache_movies.exists()):
        ratings, movies = _download_movielens(url)
        data_dir.mkdir(parents=True, exist_ok=True)
        ratings.to_parquet(cache_ratings)
        movies.to_parquet(cache_movies)
    else:
        ratings = pd.read_parquet(cache_ratings)
        movies = pd.read_parquet(cache_movies)

    matrix, timestamps, user_ids, item_ids = _build_matrix(ratings)
    features, titles, genre_names = _genre_features(movies, item_ids)
    return RatingsData(
        matrix=matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        timestamps=timestamps,
        item_features=features,
        item_titles=titles,
        genre_names=genre_names,
    )


def _download_movielens(url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the zip and read ``ratings.csv`` and ``movies.csv`` out of it."""
    with urlopen(url) as response:  # noqa: S310 - trusted GroupLens URL from config
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        ratings_name = next(n for n in archive.namelist() if n.endswith("ratings.csv"))
        movies_name = next(n for n in archive.namelist() if n.endswith("movies.csv"))
        with archive.open(ratings_name) as handle:
            ratings = pd.read_csv(handle)
        with archive.open(movies_name) as handle:
            movies = pd.read_csv(handle)
    return ratings, movies


def main() -> None:
    """Download and cache MovieLens-small, then print a short summary."""
    data = load_movielens(force_download=True)
    print(
        f"Cached MovieLens-small: {data.n_users} users x {data.n_items} items, "
        f"{data.matrix.nnz} ratings, {len(data.genre_names or [])} genres."
    )


if __name__ == "__main__":
    main()
