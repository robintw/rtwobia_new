"""Supervised classification wrapping scikit-learn classifiers."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from geobia.classification.base import BaseClassifier


class SupervisedClassifier(BaseClassifier):
    """Supervised classifier wrapping sklearn classifiers.

    Supports Random Forest (default), SVM, and Gradient Boosting.
    """

    def __init__(
        self,
        algorithm: str = "random_forest",
        **params,
    ):
        self.algorithm = algorithm
        self.params = params
        self.model_ = None
        self.feature_names_ = None
        self.classes_ = None

        self.model_ = self._create_model()

    def _create_model(self):
        if self.algorithm == "random_forest":
            defaults = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 5,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": 1,
            }
            defaults.update(self.params)
            return RandomForestClassifier(**defaults)
        elif self.algorithm == "svm":
            defaults = {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "class_weight": "balanced",
                "probability": True,
                "random_state": 42,
            }
            defaults.update(self.params)
            return SVC(**defaults)
        elif self.algorithm == "gradient_boosting":
            defaults = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": 42,
            }
            defaults.update(self.params)
            return GradientBoostingClassifier(**defaults)
        raise ValueError(
            f"Unknown algorithm: {self.algorithm!r}. "
            f"Available: 'random_forest', 'svm', 'gradient_boosting'"
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series | None = None) -> None:
        if labels is None:
            raise ValueError("Supervised classifier requires training labels")

        # Align features and labels on their index
        common = features.index.intersection(labels.index)
        if len(common) == 0:
            raise ValueError("No common segment IDs between features and labels")

        X = features.loc[common]
        y = labels.loc[common]

        self.feature_names_ = list(X.columns)
        self.model_.fit(X.values, y.values)
        self.classes_ = list(self.model_.classes_)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.feature_names_ is None:
            raise RuntimeError("Classifier has not been trained")
        X = features[self.feature_names_]
        preds = self.model_.predict(X.values)
        return pd.Series(preds, index=features.index, name="class_label")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("Classifier has not been trained")
        X = features[self.feature_names_]
        proba = self.model_.predict_proba(X.values)
        columns = [f"prob_{c}" for c in self.classes_]
        return pd.DataFrame(proba, index=features.index, columns=columns)

    def feature_importance(self) -> pd.Series:
        """Return feature importance from the trained model."""
        if self.feature_names_ is None:
            raise RuntimeError("Classifier has not been trained")
        importances = self.model_.feature_importances_
        return pd.Series(importances, index=self.feature_names_, name="importance").sort_values(
            ascending=False
        )

    def get_params(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "feature_names": self.feature_names_,
            "classes": self.classes_,
            **self.params,
        }
