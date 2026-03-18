"""Shepherd segmentation algorithm.

Delegates to pyshepseg for the core algorithm (K-means seeding,
connected component clumping, iterative elimination of small segments).

pyshepseg provides optimized numba JIT implementations and is ~2x faster
than our previous built-in implementation.
"""

from __future__ import annotations

import numpy as np
from pyshepseg import shepseg

from geobia.segmentation.base import BaseSegmenter


class ShepherdSegmenter(BaseSegmenter):
    """Shepherd segmentation algorithm.

    K-means seeded, iterative elimination of small segments. Uses
    pyshepseg as the backend for ~2x better performance.
    """

    def __init__(
        self,
        num_clusters: int = 60,
        min_n_pxls: int = 100,
        dist_thres: float | str = "auto",
        sampling: int = 100,
        bands: list[int] | None = None,
    ):
        self.num_clusters = num_clusters
        self.min_n_pxls = min_n_pxls
        self.dist_thres = dist_thres
        self.sampling = sampling
        self.bands = bands

    def segment(
        self,
        image: np.ndarray,
        nodata_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        n_bands, h, w = image.shape

        if self.bands is not None:
            image = image[self.bands]
            n_bands = len(self.bands)

        # pyshepseg expects integer dtype with shape (bands, rows, cols)
        # Scale float imagery to uint16 range for integer input
        if np.issubdtype(image.dtype, np.floating):
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                scaled = (image - img_min) / (img_max - img_min) * 10000
            else:
                scaled = np.zeros_like(image)
            img_int = scaled.astype(np.uint16)
        else:
            img_int = image

        # Determine null value for pyshepseg
        img_null_val = None
        if nodata_mask is not None and nodata_mask.any():
            # Set nodata pixels to 0 and tell pyshepseg about it
            img_null_val = 0
            img_int[:, nodata_mask] = 0

        # Convert sampling rate to percentage for pyshepseg
        subsample_pcnt = max(1, 100 // max(1, self.sampling))

        result = shepseg.doShepherdSegmentation(
            img_int,
            numClusters=self.num_clusters,
            clusterSubsamplePcnt=subsample_pcnt,
            minSegmentSize=self.min_n_pxls,
            maxSpectralDiff=self.dist_thres,
            imgNullVal=img_null_val,
            fourConnected=True,
            fixedKMeansInit=True,
        )

        # Convert to int32 (our convention: 0=nodata, 1..N=segments)
        labels = result.segimg.astype(np.int32)
        return labels

    def get_params(self) -> dict:
        return {
            "algorithm": "shepherd",
            "backend": "pyshepseg",
            "num_clusters": self.num_clusters,
            "min_n_pxls": self.min_n_pxls,
            "dist_thres": self.dist_thres,
            "sampling": self.sampling,
            "bands": self.bands,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "num_clusters": {
                    "type": "integer",
                    "default": 60,
                    "minimum": 2,
                    "description": "K-means seed count (30-90 recommended)",
                },
                "min_n_pxls": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 1,
                    "description": "Minimum segment size in pixels",
                },
                "dist_thres": {
                    "default": "auto",
                    "description": "Max spectral distance for merging ('auto' recommended)",
                },
                "sampling": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 1,
                    "description": "Subsampling rate for k-means (every Nth pixel)",
                },
                "bands": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Band indices to use (default: all)",
                },
            },
        }
