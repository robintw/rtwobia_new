"""Classification module with supervised and unsupervised methods."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

from geobia.classification.accuracy import AccuracyReport, assess_accuracy, cross_validate
from geobia.classification.base import BaseClassifier
from geobia.classification.fuzzy import FuzzyClassifier, FuzzyRule
from geobia.classification.supervised import SupervisedClassifier
from geobia.classification.unsupervised import UnsupervisedClassifier


def classify(
    features: pd.DataFrame,
    method: str = "random_forest",
    training_labels: pd.Series | None = None,
    progress: Any | None = None,
    **params,
) -> pd.Series:
    """Convenience function to classify segments.

    Args:
        features: Feature DataFrame with segment_id index.
        method: Classification method name.
        training_labels: Required for supervised methods. Series mapping
            segment_id -> class_label.
        progress: Optional callable(percent: float) for progress reporting.
        **params: Algorithm-specific parameters.

    Returns:
        Series with predicted labels (segment_id index).
    """
    _report = progress or (lambda p: None)

    unsupervised_methods = ("kmeans", "gmm", "dbscan")
    supervised_methods = ("random_forest", "svm", "gradient_boosting")

    logger.info("Classifying with method=%s (%d segments)", method, len(features))
    _report(10)
    if method in unsupervised_methods:
        clf = UnsupervisedClassifier(algorithm=method, **params)
        clf.fit(features)
    elif method == "fuzzy":
        clf = FuzzyClassifier(**params)
        clf.fit(features)
    elif method in supervised_methods:
        clf = SupervisedClassifier(algorithm=method, **params)
        clf.fit(features, training_labels)
    else:
        # Try as supervised (will raise if unknown)
        clf = SupervisedClassifier(algorithm=method, **params)
        clf.fit(features, training_labels)

    _report(70)
    result = clf.predict(features)
    logger.info("Classification complete: %d classes", result.nunique())
    _report(100)
    return result


__all__ = [
    "BaseClassifier",
    "SupervisedClassifier",
    "UnsupervisedClassifier",
    "FuzzyClassifier",
    "FuzzyRule",
    "AccuracyReport",
    "assess_accuracy",
    "cross_validate",
    "classify",
]
