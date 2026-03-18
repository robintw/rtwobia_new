"""Base class for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseExtractor(ABC):
    """Base class for feature extractors.

    Subclasses compute features from image pixels grouped by segment labels.
    """

    @abstractmethod
    def extract(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> pd.DataFrame:
        """Extract features for all segments.

        Args:
            image: Multi-band array (bands, height, width).
            labels: Segment label array (height, width).

        Returns:
            DataFrame with segment_id as index, features as columns.
        """

    @classmethod
    @abstractmethod
    def feature_names(cls, **kwargs) -> list[str]:
        """Return list of feature names this extractor produces."""
