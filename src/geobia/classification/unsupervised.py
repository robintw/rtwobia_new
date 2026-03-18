"""Unsupervised classification (clustering) wrapping scikit-learn."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

from geobia.classification.base import BaseClassifier


class UnsupervisedClassifier(BaseClassifier):
    """Unsupervised classifier using K-Means clustering.

    Groups segments into clusters based on feature similarity.
    """

    def __init__(
        self,
        algorithm: str = "kmeans",
        **params,
    ):
        self.algorithm = algorithm
        self.params = params
        self.model_ = None
        self.feature_names_ = None

        self.model_ = self._create_model()

    def _create_model(self):
        if self.algorithm == "kmeans":
            defaults = {
                "n_clusters": 8,
                "n_init": 10,
                "max_iter": 300,
                "random_state": 42,
            }
            defaults.update(self.params)
            return KMeans(**defaults)
        raise ValueError(f"Unknown algorithm: {self.algorithm!r}")

    def fit(self, features: pd.DataFrame, labels: pd.Series | None = None) -> None:
        # labels are ignored for unsupervised
        self.feature_names_ = list(features.columns)
        self.model_.fit(features.values)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.feature_names_ is None:
            raise RuntimeError("Classifier has not been fitted")
        X = features[self.feature_names_]
        preds = self.model_.predict(X.values)
        # Cluster labels are 0-indexed; shift to 1-indexed for clarity
        return pd.Series(preds + 1, index=features.index, name="cluster")

    def get_params(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "feature_names": self.feature_names_,
            **self.params,
        }
