"""Texture feature extraction using GLCM (Gray Level Co-occurrence Matrix)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

from geobia.features.base import BaseExtractor


class TextureExtractor(BaseExtractor):
    """Extract GLCM/Haralick texture features per segment.

    Computes texture features from the GLCM for each segment's bounding box,
    masked to only segment pixels. Features are computed per-band.
    """

    PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")

    def __init__(
        self,
        distances: list[int] | None = None,
        angles: list[float] | None = None,
        levels: int = 32,
        bands: list[int] | None = None,
    ):
        """
        Args:
            distances: Pixel distances for GLCM (default: [1]).
            angles: Angles in radians for GLCM (default: [0, pi/4, pi/2, 3*pi/4]).
            levels: Number of gray levels for quantization.
            bands: Which bands to compute texture for (default: all).
        """
        self.distances = distances or [1]
        self.angles = angles or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.levels = levels
        self.bands = bands

    def extract(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> pd.DataFrame:
        from scipy.ndimage import find_objects

        n_bands = image.shape[0]
        band_indices = self.bands if self.bands is not None else list(range(n_bands))

        segment_ids = np.unique(labels)
        segment_ids = segment_ids[segment_ids > 0]

        if len(segment_ids) == 0:
            return pd.DataFrame()

        # Quantize each band to [0, levels-1]
        quantized = np.zeros((n_bands, image.shape[1], image.shape[2]), dtype=np.uint8)
        for b in range(n_bands):
            band = image[b].astype(np.float64)
            bmin, bmax = band.min(), band.max()
            if bmax > bmin:
                scaled = (band - bmin) / (bmax - bmin) * (self.levels - 1)
            else:
                scaled = np.zeros_like(band)
            quantized[b] = np.clip(scaled, 0, self.levels - 1).astype(np.uint8)

        # Get bounding slices for each segment
        max_label = int(labels.max())
        slices = find_objects(labels)

        # Build feature columns
        col_names = []
        for b in band_indices:
            for prop in self.PROPERTIES:
                col_names.append(f"glcm_{prop}_b{b}")

        records = {}
        for seg_id in segment_ids:
            idx = seg_id - 1
            if idx >= len(slices) or slices[idx] is None:
                continue

            row_slice, col_slice = slices[idx]
            seg_mask = labels[row_slice, col_slice] == seg_id

            row = []
            for b in band_indices:
                patch = quantized[b][row_slice, col_slice].copy()
                # Mask non-segment pixels to 0 and exclude from GLCM
                patch[~seg_mask] = 0

                if seg_mask.sum() < 4:
                    # Too few pixels for meaningful GLCM
                    row.extend([0.0] * len(self.PROPERTIES))
                    continue

                try:
                    glcm = graycomatrix(
                        patch,
                        distances=self.distances,
                        angles=self.angles,
                        levels=self.levels,
                        symmetric=True,
                        normed=True,
                    )

                    for prop in self.PROPERTIES:
                        values = graycoprops(glcm, prop)
                        # Average over all distances and angles
                        row.append(float(values.mean()))
                except Exception:
                    row.extend([0.0] * len(self.PROPERTIES))

            records[seg_id] = row

        df = pd.DataFrame.from_dict(records, orient="index", columns=col_names)
        df.index.name = "segment_id"
        return df

    @classmethod
    def feature_names(cls, n_bands: int = 3, **kwargs) -> list[str]:
        names = []
        for b in range(n_bands):
            for prop in cls.PROPERTIES:
                names.append(f"glcm_{prop}_b{b}")
        return names
