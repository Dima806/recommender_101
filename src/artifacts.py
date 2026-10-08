"""Persist notebook outputs as files: figures to PNG, results to JSON.

Every notebook figure is saved under ``outputs/figures/`` as a ``.png`` and every
textual/numerical result under ``outputs/data/`` as a ``.json``.
Paths are resolved from this file's location, so saving works no matter what the
current working directory is when a notebook runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DATA_DIR = OUTPUTS_DIR / "data"


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy/pandas/Path objects into plain JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return {str(k): _to_jsonable(v) for k, v in obj.to_dict(orient="index").items()}
    if isinstance(obj, pd.Series):
        return {str(k): _to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_json(obj: Any, name: str) -> Path:
    """Save ``obj`` as pretty-printed JSON under ``outputs/data/`` and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / (name if name.endswith(".json") else f"{name}.json")
    path.write_text(json.dumps(_to_jsonable(obj), indent=2, sort_keys=False))
    return path


def save_figure(figure: Figure | Axes, name: str, dpi: int = 150) -> Path:
    """Save a Matplotlib figure (or axes) as a PNG under ``outputs/figures/``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = figure if isinstance(figure, Figure) else figure.get_figure()
    if not isinstance(fig, Figure):
        raise TypeError("save_figure expected a Matplotlib Figure or Axes")
    path = FIGURES_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
