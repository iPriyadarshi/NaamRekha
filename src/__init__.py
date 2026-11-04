"""Shared NaamRekha infrastructure: loader, evaluator, results logger."""

from .data import (
    LABELS,
    REPO_ID,
    SPLITS,
    id2label,
    label2id,
    load_split,
    tag_distribution,
)
from .eval import evaluate
from .results import COLUMNS, RESULTS_PATH, log_result

__all__ = [
    "COLUMNS",
    "LABELS",
    "REPO_ID",
    "RESULTS_PATH",
    "SPLITS",
    "evaluate",
    "id2label",
    "label2id",
    "load_split",
    "log_result",
    "tag_distribution",
]
