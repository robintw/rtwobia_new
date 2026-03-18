"""Unsupervised classification (clustering) wrapping scikit-learn."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from geobia.classification.base import BaseClassifier


class UnsupervisedClassifier(BaseClassifier):
    """Unsupervised classifier supporting K-Means, GMM, and DBSCAN.

    Groups segments into clusters based on feature similarity.
    """

    ALGORITHMS = ("kmeans", "gmm", "dbscan")

    def __init__(
        self,
        algorithm: str = "kmeans",
        **params,
    ):
        self.algorithm = algorithm
        self.params = params
        self.model_ = None
        self.feature_names_ = None
        self.scaler_ = None
        self.labels_ = None  # stored for DBSCAN (no predict method)

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

        elif self.algorithm == "gmm":
            defaults = {
                "n_components": 8,
                "covariance_type": "full",
                "random_state": 42,
                "n_init": 3,
            }
            defaults.update(self.params)
            return GaussianMixture(**defaults)

        elif self.algorithm == "dbscan":
            defaults = {
                "eps": 0.5,
                "min_samples": 5,
            }
            defaults.update(self.params)
            return DBSCAN(**defaults)

        raise ValueError(
            f"Unknown algorithm: {self.algorithm!r}. Available: {self.ALGORITHMS}"
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series | None = None) -> None:
        self.feature_names_ = list(features.columns)

        # DBSCAN benefits from scaled features
        if self.algorithm == "dbscan":
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(features.values)
            self.model_.fit(X)
            # DBSCAN stores labels in .labels_ (-1 = noise)
            self.labels_ = pd.Series(
                self.model_.labels_ + 1,  # shift: -1->0 (noise), 0->1, etc.
                index=features.index,
                name="cluster",
            )
        else:
            self.model_.fit(features.values)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.feature_names_ is None:
            raise RuntimeError("Classifier has not been fitted")
        X = features[self.feature_names_]

        if self.algorithm == "dbscan":
            # DBSCAN has no predict — return stored labels for training data,
            # or assign noise label (0) for unseen data
            if self.labels_ is not None and features.index.equals(self.labels_.index):
                return self.labels_
            # For new data, assign to nearest core sample's cluster
            return pd.Series(
                np.zeros(len(features), dtype=int),
                index=features.index,
                name="cluster",
            )

        if self.algorithm == "gmm":
            preds = self.model_.predict(X.values)
        else:
            preds = self.model_.predict(X.values)

        return pd.Series(preds + 1, index=features.index, name="cluster")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame | None:
        if self.algorithm == "gmm" and self.feature_names_ is not None:
            X = features[self.feature_names_]
            proba = self.model_.predict_proba(X.values)
            columns = [f"prob_cluster_{i + 1}" for i in range(proba.shape[1])]
            return pd.DataFrame(proba, index=features.index, columns=columns)
        return None

    def get_params(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "feature_names": self.feature_names_,
            **self.params,
        }
