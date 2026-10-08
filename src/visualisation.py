"""Small plotting helpers shared by the notebooks and the Streamlit app.

Kept deliberately thin: these wrap matplotlib so the notebooks stay about the
ideas, not the plotting boilerplate. Each returns the Matplotlib ``Axes`` so the
caller can tweak or save the figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def plot_training_curve(history: list[float], ax: Axes | None = None) -> Axes:
    """Plot MF training RMSE per epoch to show the loss falling."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(history) + 1), history, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train RMSE")
    ax.set_title("Matrix factorization: training loss")
    return ax


def plot_scoreboard(scoreboard: pd.DataFrame, metric: str, ax: Axes | None = None) -> Axes:
    """Bar chart of one metric across methods, with the popularity line drawn in."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    values = scoreboard[metric]
    ax.bar(values.index, values.to_numpy())
    if "popularity" in values.index:
        ax.axhline(values.loc["popularity"], linestyle="--", label="popularity baseline")
        ax.legend()
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by method")
    ax.tick_params(axis="x", rotation=30)
    return ax


def plot_embedding_2d(coords: np.ndarray, labels: np.ndarray, ax: Axes | None = None) -> Axes:
    """Scatter 2-D item embeddings coloured by a label (e.g. genre/group)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=12, alpha=0.8)
    ax.set_title("Item embedding space")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    legend = ax.legend(*scatter.legend_elements(), title="group", loc="best")
    ax.add_artist(legend)
    return ax
