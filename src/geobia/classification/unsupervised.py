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
    Cluster labels are 1-indexed.  For DBSCAN, label 0 means *noise*
    (the point did not belong to any cluster).
    """

    ALGORITHMS = ("kmeans", "gmm", "dbscan")
    NOISE_LABEL = 0  # DBSCAN noise / unclassified

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
        self._auto_eps = False

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
            mapped = dict(self.params)
            # Map n_clusters to n_components for GMM
            if "n_clusters" in mapped:
                mapped["n_components"] = mapped.pop("n_clusters")
            defaults.update(mapped)
            return GaussianMixture(**defaults)

        elif self.algorithm == "dbscan":
            # eps=0 means auto-estimate at fit time; strip it so DBSCAN
            # uses its own default until we override in fit().
            defaults = {
                "min_samples": 5,
            }
            defaults.update(self.params)
            self._auto_eps = defaults.pop("eps", 0) in (0, None)
            if not self._auto_eps:
                defaults["eps"] = self.params["eps"]
            return DBSCAN(**defaults)

        raise ValueError(f"Unknown algorithm: {self.algorithm!r}. Available: {self.ALGORITHMS}")

    def fit(self, features: pd.DataFrame, labels: pd.Series | None = None) -> None:
        self.feature_names_ = list(features.columns)

        # DBSCAN benefits from scaled features
        if self.algorithm == "dbscan":
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(features.values)
            # Auto-estimate eps from k-nearest-neighbor distances if not
            # explicitly provided.  Default eps=0.5 is far too small for
            # high-dimensional scaled feature spaces where typical NN
            # distances grow with sqrt(n_features).
            if self._auto_eps:
                from sklearn.neighbors import NearestNeighbors

                k = min(self.model_.min_samples, len(X) - 1)
                nn = NearestNeighbors(n_neighbors=k)
                nn.fit(X)
                dists, _ = nn.kneighbors(X)
                # Use the "knee" of the k-distance curve: the median
                # of the k-th neighbor distances is a robust default.
                self.model_.eps = float(np.median(dists[:, -1]))
            self.model_.fit(X)
            # DBSCAN stores labels in .labels_ (-1 = noise).
            # Shift so that: -1 -> NOISE_LABEL (0), 0 -> 1, 1 -> 2, etc.
            raw = self.model_.labels_
            shifted = np.where(raw == -1, self.NOISE_LABEL, raw + 1)
            self.labels_ = pd.Series(
                shifted,
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
            # For new data, assign noise label (no predict for DBSCAN)
            return pd.Series(
                self.NOISE_LABEL,
                index=features.index,
                name="cluster",
                dtype=int,
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

    @classmethod
    def get_param_schema(cls, algorithm: str = "kmeans") -> dict:
        """Return JSON Schema for the given unsupervised algorithm."""
        schemas = {
            "kmeans": {
                "type": "object",
                "properties": {
                    "n_clusters": {
                        "type": "integer",
                        "default": 8,
                        "minimum": 2,
                        "maximum": 1000,
                        "description": (
                            "Number of clusters to create. Choose based on "
                            "the expected number of land cover types in the "
                            "scene."
                        ),
                    },
                },
            },
            "gmm": {
                "type": "object",
                "properties": {
                    "n_clusters": {
                        "type": "integer",
                        "default": 8,
                        "minimum": 2,
                        "maximum": 1000,
                        "description": (
                            "Number of Gaussian components (clusters). GMM "
                            "allows soft cluster assignments, so segments "
                            "can have partial membership in multiple clusters."
                        ),
                    },
                },
            },
            "dbscan": {
                "type": "object",
                "properties": {
                    "eps": {
                        "type": "number",
                        "default": 0,
                        "minimum": 0,
                        "maximum": 100.0,
                        "description": (
                            "Maximum distance between two samples to be "
                            "considered neighbours. Smaller values create "
                            "tighter, more numerous clusters. Set to 0 for "
                            "automatic estimation from the data."
                        ),
                    },
                    "min_samples": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 1000,
                        "description": (
                            "Minimum number of samples in a neighbourhood "
                            "to form a cluster core. Higher values ignore "
                            "small clusters and classify them as noise."
                        ),
                    },
                },
            },
        }
        return schemas.get(algorithm, {"type": "object", "properties": {}})
