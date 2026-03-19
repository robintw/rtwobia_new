"""Accuracy assessment for classification results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score


@dataclass
class AccuracyReport:
    """Container for classification accuracy metrics."""

    overall_accuracy: float
    kappa: float
    confusion_matrix: np.ndarray
    class_names: list[str]
    per_class: dict[str, dict[str, float]]

    def summary(self) -> str:
        lines = [
            f"Overall Accuracy: {self.overall_accuracy:.4f}",
            f"Kappa Coefficient: {self.kappa:.4f}",
            "",
            "Per-class metrics:",
        ]
        for cls_name, metrics in self.per_class.items():
            lines.append(
                f"  {cls_name}: precision={metrics['precision']:.3f} "
                f"recall={metrics['recall']:.3f} f1={metrics['f1-score']:.3f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall_accuracy": self.overall_accuracy,
            "kappa": self.kappa,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_names": self.class_names,
            "per_class": self.per_class,
        }


def assess_accuracy(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
) -> AccuracyReport:
    """Compute classification accuracy metrics.

    Args:
        true_labels: Ground truth labels (Series with segment_id index).
        predicted_labels: Predicted labels (Series with segment_id index).

    Returns:
        AccuracyReport with OA, kappa, confusion matrix, per-class metrics.
    """
    common = true_labels.index.intersection(predicted_labels.index)
    y_true = true_labels.loc[common]
    y_pred = predicted_labels.loc[common]

    class_names = sorted(y_true.unique().tolist())

    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = {
        str(cls): {
            "precision": report[str(cls)]["precision"],
            "recall": report[str(cls)]["recall"],
            "f1-score": report[str(cls)]["f1-score"],
            "support": report[str(cls)]["support"],
        }
        for cls in class_names
        if str(cls) in report
    }

    return AccuracyReport(
        overall_accuracy=oa,
        kappa=kappa,
        confusion_matrix=cm,
        class_names=[str(c) for c in class_names],
        per_class=per_class,
    )


def cross_validate(
    classifier,
    features: pd.DataFrame,
    labels: pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    """Run cross-validation on a classifier.

    Args:
        classifier: A classifier with sklearn-compatible interface.
        features: Feature DataFrame.
        labels: Class labels.
        cv: Number of folds.

    Returns:
        Dict with mean and std accuracy scores.
    """
    common = features.index.intersection(labels.index)
    X = features.loc[common].values
    y = np.asarray(labels.loc[common])

    scores = cross_val_score(classifier.model_, X, y, cv=cv, scoring="accuracy")
    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "fold_scores": scores.tolist(),
    }
