"""Classification module with supervised and unsupervised methods."""

from __future__ import annotations

import pandas as pd

from geobia.classification.base import BaseClassifier
from geobia.classification.supervised import SupervisedClassifier
from geobia.classification.unsupervised import UnsupervisedClassifier
from geobia.classification.accuracy import AccuracyReport, assess_accuracy, cross_validate


def classify(
    features: pd.DataFrame,
    method: str = "random_forest",
    training_labels: pd.Series | None = None,
    **params,
) -> pd.Series:
    """Convenience function to classify segments.

    Args:
        features: Feature DataFrame with segment_id index.
        method: Classification method name.
        training_labels: Required for supervised methods. Series mapping
            segment_id -> class_label.
        **params: Algorithm-specific parameters.

    Returns:
        Series with predicted labels (segment_id index).
    """
    if method in ("kmeans",):
        clf = UnsupervisedClassifier(algorithm=method, **params)
        clf.fit(features)
    else:
        clf = SupervisedClassifier(algorithm=method, **params)
        clf.fit(features, training_labels)

    return clf.predict(features)


__all__ = [
    "BaseClassifier",
    "SupervisedClassifier",
    "UnsupervisedClassifier",
    "AccuracyReport",
    "assess_accuracy",
    "cross_validate",
    "classify",
]
