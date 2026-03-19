"""Watershed segmentation using scikit-image."""

from __future__ import annotations

import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.segmentation import watershed

from geobia.segmentation.base import BaseSegmenter


class WatershedSegmenter(BaseSegmenter):
    """Marker-based watershed segmentation.

    Computes a gradient magnitude image, finds local minima as markers,
    then floods the gradient surface using the watershed transform.
    """

    def __init__(
        self,
        markers: int = 500,
        compactness: float = 0.0,
        min_distance: int = 10,
    ):
        self.markers = markers
        self.compactness = compactness
        self.min_distance = min_distance

    def segment(
        self,
        image: np.ndarray,
        nodata_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Segment using marker-based watershed.

        Args:
            image: (bands, height, width) array.
            nodata_mask: Boolean mask, True where invalid.

        Returns:
            (height, width) integer label array (1-indexed, 0=nodata).
        """
        # Compute per-band gradient and average
        if image.ndim == 3:
            gradients = [sobel(image[b]) for b in range(image.shape[0])]
            gradient = np.mean(gradients, axis=0)
        else:
            gradient = sobel(image)

        # Find markers via local minima of the gradient
        # Use negative gradient so peak_local_max finds minima
        coords = peak_local_max(
            -gradient,
            min_distance=self.min_distance,
            num_peaks=self.markers,
        )
        marker_array = np.zeros(gradient.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            marker_array[r, c] = i

        labels = watershed(
            gradient,
            markers=marker_array,
            compactness=self.compactness,
        )

        if nodata_mask is not None:
            labels[nodata_mask] = 0

        return labels.astype(np.int32)

    def get_params(self) -> dict:
        return {
            "algorithm": "watershed",
            "markers": self.markers,
            "compactness": self.compactness,
            "min_distance": self.min_distance,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "markers": {
                    "type": "integer",
                    "default": 500,
                    "minimum": 1,
                    "description": "Maximum number of seed markers",
                },
                "compactness": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0,
                    "description": "Compactness of watershed basins (0=no shape constraint)",
                },
                "min_distance": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "description": "Minimum distance between markers in pixels",
                },
            },
        }
