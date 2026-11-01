"""seqeval entity-level micro-F1, overall and per type."""

from __future__ import annotations

from typing import cast

from seqeval.metrics import classification_report

# result key -> seqeval entity label
_TYPES = {"F1_PER": "PER", "F1_ORG": "ORG", "F1_LOC": "LOC"}


def evaluate(y_true: list[list[str]], y_pred: list[list[str]]) -> dict[str, float]:
    """Score predicted tags against gold (one tag sequence per sentence).
    Returns F1_overall, F1_PER, F1_ORG, F1_LOC."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred differ in length: {len(y_true)} vs {len(y_pred)}"
        )

    report = cast(dict, classification_report(y_true, y_pred, output_dict=True))
    result = {"F1_overall": float(report["micro avg"]["f1-score"])}
    for col, entity in _TYPES.items():
        result[col] = float(report.get(entity, {}).get("f1-score", 0.0))
    return result
