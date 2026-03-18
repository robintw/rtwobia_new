"""Base class for segmentation algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSegmenter(ABC):
    """Base class for all segmentation algorithms.

    Subclasses must implement the segment() method. Images are expected
    in (bands, height, width) format. Returns integer label arrays with
    shape (height, width) where 0 = nodata and 1..N = segment IDs.
    """

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        nodata_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Segment an image into labeled regions.

        Args:
            image: Multi-band image array, shape (bands, height, width).
            nodata_mask: Boolean mask, True where data is invalid.

        Returns:
            Integer array of segment IDs, shape (height, width).
        """

    @abstractmethod
    def get_params(self) -> dict:
        """Return current parameters as a serializable dict."""

    @classmethod
    def get_param_schema(cls) -> dict:
        """Return JSON Schema describing available parameters."""
        return {}
