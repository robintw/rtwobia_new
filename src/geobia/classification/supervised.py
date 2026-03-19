"""Supervised classification wrapping scikit-learn classifiers."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
        self.scaler_ = None  # StandardScaler for SVM

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
        X_vals = X.values
        if self.algorithm == "svm":
            self.scaler_ = StandardScaler()
            X_vals = self.scaler_.fit_transform(X_vals)
        self.model_.fit(X_vals, y.values)
        self.classes_ = list(self.model_.classes_)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.feature_names_ is None:
            raise RuntimeError("Classifier has not been trained")
        X = features[self.feature_names_]
        X_vals = X.values
        if self.scaler_ is not None:
            X_vals = self.scaler_.transform(X_vals)
        preds = self.model_.predict(X_vals)
        return pd.Series(preds, index=features.index, name="class_label")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("Classifier has not been trained")
        X = features[self.feature_names_]
        X_vals = X.values
        if self.scaler_ is not None:
            X_vals = self.scaler_.transform(X_vals)
        proba = self.model_.predict_proba(X_vals)
        columns = [f"prob_{c}" for c in self.classes_]
        return pd.DataFrame(proba, index=features.index, columns=columns)

    def feature_importance(self) -> pd.Series:
        """Return feature importance from the trained model.

        For tree-based models (Random Forest, Gradient Boosting) returns
        built-in feature importances. SVM does not provide feature importances;
        raises NotImplementedError in that case.
        """
        if self.feature_names_ is None:
            raise RuntimeError("Classifier has not been trained")
        if not hasattr(self.model_, "feature_importances_"):
            raise NotImplementedError(
                f"Feature importance is not available for '{self.algorithm}'. "
                f"Use a tree-based algorithm (random_forest, gradient_boosting)."
            )
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

    @classmethod
    def get_param_schema(cls, algorithm: str = "random_forest") -> dict:
        """Return JSON Schema for the given supervised algorithm."""
        schemas = {
            "random_forest": {
                "type": "object",
                "properties": {
                    "n_estimators": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 10000,
                        "description": (
                            "Number of decision trees in the ensemble. "
                            "More trees generally improve accuracy but "
                            "increase computation time."
                        ),
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "maximum": 1000,
                        "description": (
                            "Maximum depth of each tree. 0 means unlimited "
                            "(trees grow until leaves are pure). Lower values "
                            "reduce overfitting but may underfit."
                        ),
                    },
                },
            },
            "svm": {
                "type": "object",
                "properties": {
                    "kernel": {
                        "type": "enum",
                        "options": ["rbf", "linear", "poly", "sigmoid"],
                        "default": "rbf",
                        "description": (
                            "Kernel function for the SVM. 'rbf' (radial basis "
                            "function) works well for most cases. 'linear' is "
                            "faster and suits linearly separable classes."
                        ),
                    },
                    "C": {
                        "type": "number",
                        "default": 1.0,
                        "minimum": 0.001,
                        "maximum": 10000.0,
                        "description": (
                            "Regularisation parameter. Higher values fit "
                            "training data more closely (risk of overfitting). "
                            "Lower values produce a smoother decision boundary."
                        ),
                    },
                },
            },
            "gradient_boosting": {
                "type": "object",
                "properties": {
                    "n_estimators": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 10000,
                        "description": (
                            "Number of boosting stages (trees). More stages "
                            "can improve accuracy but increase training time "
                            "and risk of overfitting."
                        ),
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100,
                        "description": (
                            "Maximum depth of each tree. Gradient boosting "
                            "typically uses shallow trees (3-8). Deeper trees "
                            "capture more complex interactions but increase "
                            "overfitting risk."
                        ),
                    },
                    "learning_rate": {
                        "type": "number",
                        "default": 0.1,
                        "minimum": 0.001,
                        "maximum": 10.0,
                        "description": (
                            "Shrinks the contribution of each tree. Lower "
                            "values (e.g. 0.01) need more trees but often "
                            "generalise better. Typical range: 0.01-0.3."
                        ),
                    },
                },
            },
        }
        return schemas.get(algorithm, {"type": "object", "properties": {}})
