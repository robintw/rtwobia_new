"""Base class for classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd


class BaseClassifier(ABC):
    """Base class for all classifiers (supervised and unsupervised)."""

    @abstractmethod
    def fit(self, features: pd.DataFrame, labels: pd.Series | None = None) -> None:
        """Train the classifier or fit clusters.

        Args:
            features: Feature DataFrame with segment_id as index.
            labels: Class labels (Series with segment_id index). Required for
                supervised classifiers, ignored for unsupervised.
        """

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict class labels for segments.

        Args:
            features: Feature DataFrame with segment_id as index.

        Returns:
            Series with segment_id index and predicted class labels.
        """

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame | None:
        """Return class probabilities for each segment.

        Not all classifiers support this. Returns None if not supported.
        """
        return None

    def save(self, path: str | Path) -> None:
        """Serialize the trained model to disk."""
        joblib.dump(self._get_state(), path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseClassifier":
        """Load a trained model from disk."""
        state = joblib.load(path)
        instance = cls.__new__(cls)
        instance._set_state(state)
        return instance

    @abstractmethod
    def get_params(self) -> dict:
        """Return current parameters as a serializable dict."""

    def _get_state(self) -> dict:
        return self.__dict__.copy()

    def _set_state(self, state: dict) -> None:
        self.__dict__.update(state)
