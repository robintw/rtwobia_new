"""Felzenszwalb graph-based segmentation using scikit-image."""

from __future__ import annotations

import numpy as np
from skimage.segmentation import felzenszwalb

from geobia.segmentation.base import BaseSegmenter


class FelzenszwalbSegmenter(BaseSegmenter):
    """Felzenszwalb graph-based segmentation.

    Uses minimum spanning tree to merge regions when inter-region difference
    is small relative to internal variation. Produces adaptive-sized segments.
    """

    def __init__(
        self,
        scale: float = 100.0,
        sigma: float = 0.8,
        min_size: int = 50,
    ):
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size

    def segment(
        self,
        image: np.ndarray,
        nodata_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Segment using Felzenszwalb's method.

        Args:
            image: (bands, height, width) array.
            nodata_mask: Boolean mask, True where invalid.

        Returns:
            (height, width) integer label array (1-indexed, 0=nodata).
        """
        # scikit-image expects (height, width, channels)
        if image.ndim == 3:
            img = np.moveaxis(image, 0, -1)
        else:
            img = image

        labels = felzenszwalb(
            img,
            scale=self.scale,
            sigma=self.sigma,
            min_size=self.min_size,
            channel_axis=-1 if img.ndim == 3 else None,
        )

        # Felzenszwalb returns 0-indexed labels; shift to 1-indexed
        labels = labels + 1

        if nodata_mask is not None:
            labels[nodata_mask] = 0

        return labels.astype(np.int32)

    def get_params(self) -> dict:
        return {
            "algorithm": "felzenszwalb",
            "scale": self.scale,
            "sigma": self.sigma,
            "min_size": self.min_size,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "default": 100.0,
                    "minimum": 0,
                    "description": "Controls the coarseness of segmentation. Higher values merge more aggressively, producing fewer, larger segments. Lower values preserve finer detail.",
                },
                "sigma": {
                    "type": "number",
                    "default": 0.8,
                    "minimum": 0,
                    "description": "Width of Gaussian smoothing applied before segmentation. Increase to reduce noise sensitivity.",
                },
                "min_size": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "description": "Minimum segment size in pixels. Segments smaller than this are merged into the most similar neighbour.",
                },
            },
        }
