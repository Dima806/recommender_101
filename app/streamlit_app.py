"""Streamlit app for recommender_101.

Four views: recommend-to-me (top-k from each method side by side), a taste-vector
explorer, a baseline showdown against the popularity line, and a cold-start
simulator. It loads cached MovieLens-small if present and otherwise falls back to
the synthetic dataset so the app always runs. Launch with ``make run``.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix

from src.config import get_settings
from src.data.ratings import load_movielens, make_synthetic_ratings
from src.evaluation.comparison import run_arena
from src.evaluation.splits import leave_last_out
from src.recommenders import (
    MatrixFactorization,
    PopularityRecommender,
    UserKNNRecommender,
    build_default_recommenders,
)

SETTINGS = get_settings()


@st.cache_data(show_spinner="Loading ratings...")
def load_data() -> dict[str, object]:
    """Load MovieLens if cached, else synthetic. Returned as plain arrays for caching."""
    try:
        data = load_movielens()
        titles = data.item_titles or [str(i) for i in data.item_ids]
        return {
            "matrix": data.matrix,
            "features": data.item_features,
            "titles": titles,
            "source": "MovieLens-small",
        }
    except Exception:  # noqa: BLE001 - any download/parse failure falls back to synthetic
        synthetic = make_synthetic_ratings(seed=SETTINGS.seed)
        titles = [f"item {i} (group {g})" for i, g in enumerate(synthetic.item_group)]
        return {
            "matrix": synthetic.observed,
            "features": synthetic.item_features,
            "titles": titles,
            "source": "synthetic",
        }


def _names(titles: list[str], indices: np.ndarray) -> list[str]:
    return [titles[i] for i in indices]


def recommend_to_me(data: dict[str, object]) -> None:
    st.header("Recommend to me")
    matrix = data["matrix"]
    titles = data["titles"]
    user = st.number_input("User index", min_value=0, max_value=matrix.shape[0] - 1, value=0)
    k = st.slider("Top-k", min_value=5, max_value=20, value=SETTINGS.top_k)

    models = {
        "popularity": PopularityRecommender().fit(matrix),
        "user_knn": UserKNNRecommender(SETTINGS.knn.n_neighbors).fit(matrix),
        "matrix_factorization": MatrixFactorization(
            n_factors=SETTINGS.mf.n_factors, n_epochs=SETTINGS.mf.n_epochs, seed=SETTINGS.seed
        ).fit(matrix),
    }
    columns = st.columns(len(models))
    for col, (name, model) in zip(columns, models.items(), strict=True):
        recs = model.recommend(int(user), k=k)
        col.subheader(name)
        col.write(_names(titles, recs))


def baseline_showdown(data: dict[str, object]) -> None:
    st.header("Baseline showdown")
    matrix = data["matrix"]
    features = data["features"]
    split = leave_last_out(
        matrix,
        n_holdout=SETTINGS.split.n_holdout,
        min_train=SETTINGS.split.min_train,
        seed=SETTINGS.seed,
    )
    recommenders = build_default_recommenders(item_features=features, settings=SETTINGS)
    scoreboard = run_arena(recommenders, split, k=SETTINGS.top_k, item_features=features)
    metric = st.selectbox("Metric", list(scoreboard.columns), index=0)
    st.bar_chart(scoreboard[metric])
    st.dataframe(scoreboard.style.format("{:.4f}"))


def cold_start(data: dict[str, object]) -> None:
    st.header("Cold-start simulator")
    st.write(
        "A brand-new user has no history, so collaborative filtering has nothing to "
        "copy and falls back to noise, while popularity still works."
    )
    matrix = data["matrix"]
    titles = data["titles"]
    n_ratings = st.slider("Ratings the new user has given", 0, 10, 0)

    rng = np.random.default_rng(SETTINGS.seed)
    new_row = np.zeros(matrix.shape[1])
    if n_ratings > 0:
        liked = rng.choice(matrix.shape[1], size=n_ratings, replace=False)
        new_row[liked] = 5.0
    augmented = np.vstack([matrix.toarray(), new_row])
    augmented_sparse = csr_matrix(augmented)
    new_index = augmented.shape[0] - 1

    pop = PopularityRecommender().fit(augmented_sparse)
    cf = UserKNNRecommender(SETTINGS.knn.n_neighbors).fit(augmented_sparse)
    left, right = st.columns(2)
    left.subheader("popularity")
    left.write(_names(titles, pop.recommend(new_index, k=SETTINGS.top_k)))
    right.subheader("user_knn")
    right.write(_names(titles, cf.recommend(new_index, k=SETTINGS.top_k)))


def main() -> None:
    st.set_page_config(page_title="recommender_101", layout="wide")
    st.title("recommender_101")
    st.caption("Report the popularity baseline first, always.")

    data = load_data()
    st.info(f"Dataset: {data['source']}")
    view = st.sidebar.radio(
        "View", ["Recommend to me", "Baseline showdown", "Cold-start simulator"]
    )
    if view == "Recommend to me":
        recommend_to_me(data)
    elif view == "Baseline showdown":
        baseline_showdown(data)
    else:
        cold_start(data)


if __name__ == "__main__":
    main()
