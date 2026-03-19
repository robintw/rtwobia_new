"""SLIC superpixel segmentation using scikit-image."""

from __future__ import annotations

import numpy as np
from skimage.segmentation import slic

from geobia.segmentation.base import BaseSegmenter


class SLICSegmenter(BaseSegmenter):
    """SLIC superpixel segmentation.

    Performs k-means clustering in a 5D space (x, y, color channels)
    to produce approximately equally-sized superpixels.
    """

    def __init__(
        self,
        n_segments: int = 500,
        compactness: float = 10.0,
        sigma: float = 0.0,
        min_size_factor: float = 0.5,
        enforce_connectivity: bool = True,
    ):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.min_size_factor = min_size_factor
        self.enforce_connectivity = enforce_connectivity

    def segment(
        self,
        image: np.ndarray,
        nodata_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Segment using SLIC superpixels.

        Args:
            image: (bands, height, width) array.
            nodata_mask: Boolean mask, True where invalid.

        Returns:
            (height, width) integer label array (1-indexed, 0=nodata).
        """
        # scikit-image slic expects (height, width, channels)
        if image.ndim == 3:
            img = np.moveaxis(image, 0, -1)
        else:
            img = image

        labels = slic(
            img,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            min_size_factor=self.min_size_factor,
            enforce_connectivity=self.enforce_connectivity,
            start_label=1,
            channel_axis=-1 if img.ndim == 3 else None,
        )

        if nodata_mask is not None:
            labels[nodata_mask] = 0

        return labels.astype(np.int32)

    def get_params(self) -> dict:
        return {
            "algorithm": "slic",
            "n_segments": self.n_segments,
            "compactness": self.compactness,
            "sigma": self.sigma,
            "min_size_factor": self.min_size_factor,
            "enforce_connectivity": self.enforce_connectivity,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "n_segments": {
                    "type": "integer",
                    "default": 500,
                    "minimum": 1,
                    "description": "Approximate number of superpixels to generate. Higher values produce smaller, more detailed segments.",
                },
                "compactness": {
                    "type": "number",
                    "default": 10.0,
                    "minimum": 0,
                    "description": "Balances color similarity vs spatial proximity. Higher values produce more regularly shaped segments; lower values follow spectral boundaries more closely.",
                },
                "sigma": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0,
                    "description": "Width of Gaussian smoothing applied before segmentation. Increase to reduce noise sensitivity.",
                },
                "min_size_factor": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Minimum segment size as a fraction of the average segment size. Small segments below this threshold are merged into neighbours.",
                },
                "enforce_connectivity": {
                    "type": "boolean",
                    "default": True,
                    "description": "When enabled, ensures every segment is a single connected region. Disable for faster results if spatial contiguity is not required.",
                },
            },
        }
