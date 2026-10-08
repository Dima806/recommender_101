"""Honest evaluation: splits, ranking metrics, beyond-accuracy metrics, the arena."""

from src.evaluation.beyond_accuracy import (
    intra_list_diversity,
    novelty,
    popularity_bias,
)
from src.evaluation.comparison import run_arena
from src.evaluation.ranking import (
    average_precision_at_k,
    catalog_coverage,
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.splits import Split, leave_last_out

__all__ = [
    "Split",
    "average_precision_at_k",
    "catalog_coverage",
    "evaluate_ranking",
    "intra_list_diversity",
    "leave_last_out",
    "ndcg_at_k",
    "novelty",
    "popularity_bias",
    "precision_at_k",
    "recall_at_k",
    "run_arena",
]
