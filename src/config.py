"""Typed configuration loaded from ``config/settings.yaml``.

All tunable knobs (latent dimensions, neighbours, top-k, seeds) live in the YAML
file and are validated here with pydantic, so nothing is hardcoded in ``src/``.
Environment variables prefixed ``REC101_`` override individual fields
(e.g. ``REC101_SEED=7``), with nested fields using ``__`` (``REC101_MF__LR=0.02``).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")


class DataConfig(BaseModel):
    dataset: str = "ml-latest-small"
    url: str = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    min_ratings_per_user: int = 5


class SplitConfig(BaseModel):
    n_holdout: int = 2
    min_train: int = 3


class KNNConfig(BaseModel):
    n_neighbors: int = 40


class MFConfig(BaseModel):
    n_factors: int = 32
    n_epochs: int = 30
    lr: float = 0.01
    reg: float = 0.05
    use_bias: bool = True


class ContentConfig(BaseModel):
    n_neighbors: int = 40


class Settings(BaseSettings):
    """Top-level project settings."""

    model_config = SettingsConfigDict(env_prefix="REC101_", env_nested_delimiter="__")

    seed: int = 42
    top_k: int = 10
    like_threshold: float = 4.0

    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    knn: KNNConfig = Field(default_factory=KNNConfig)
    mf: MFConfig = Field(default_factory=MFConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)


def get_settings(path: str | Path | None = DEFAULT_SETTINGS_PATH) -> Settings:
    """Load settings from ``path`` (falling back to defaults if it is missing)."""
    if path is None:
        return Settings()
    path = Path(path)
    if not path.exists():
        return Settings()
    raw = yaml.safe_load(path.read_text()) or {}
    return Settings(**raw)
