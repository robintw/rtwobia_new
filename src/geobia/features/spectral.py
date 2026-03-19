"""Spectral feature extraction: per-band statistics and band ratios."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from geobia.features.base import BaseExtractor


class SpectralExtractor(BaseExtractor):
    """Extract per-band spectral statistics for each segment.

    Computes mean, std, min, max, median, range per band, plus brightness
    and optional band ratios (NDVI, NDWI).
    """

    def __init__(
        self,
        band_names: dict[str, int] | list[str] | None = None,
        ratios: bool = True,
    ):
        """
        Args:
            band_names: Mapping of band name -> band index, or list of names.
                If None, bands are named 'band_0', 'band_1', etc.
            ratios: Whether to compute NDVI/NDWI if appropriate bands exist.
        """
        self.band_names = band_names
        self.ratios = ratios

    def _get_band_names(self, n_bands: int) -> list[str]:
        if self.band_names is None:
            return [f"band_{i}" for i in range(n_bands)]
        if isinstance(self.band_names, dict):
            # Invert: index -> name
            idx_to_name = {v: k for k, v in self.band_names.items()}
            return [idx_to_name.get(i, f"band_{i}") for i in range(n_bands)]
        return list(self.band_names)[:n_bands]

    def _get_band_index(self, name: str) -> int | None:
        if isinstance(self.band_names, dict):
            return self.band_names.get(name)
        return None

    def extract(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> pd.DataFrame:
        if image.ndim < 2:
            raise ValueError(f"Expected image with at least 2 dimensions, got {image.ndim}")
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        n_bands = image.shape[0]
        names = self._get_band_names(n_bands)
        nodata = kwargs.get("nodata")

        segment_ids = np.unique(labels)
        segment_ids = segment_ids[segment_ids > 0]
        n_segments = len(segment_ids)

        if n_segments == 0:
            return pd.DataFrame()

        # If nodata value is provided, mask those pixels out of label array
        # so they don't contribute to stats
        effective_labels = labels
        if nodata is not None:
            nodata_mask = np.zeros(labels.shape, dtype=bool)
            for b in range(n_bands):
                nodata_mask |= image[b] == nodata
            if nodata_mask.any():
                effective_labels = labels.copy()
                effective_labels[nodata_mask] = 0

        features = {}

        for b in range(n_bands):
            band_data = image[b].astype(np.float64)
            bname = names[b]

            features[f"mean_{bname}"] = ndimage.mean(band_data, effective_labels, segment_ids)
            features[f"std_{bname}"] = ndimage.standard_deviation(
                band_data, effective_labels, segment_ids
            )
            features[f"min_{bname}"] = ndimage.minimum(band_data, effective_labels, segment_ids)
            features[f"max_{bname}"] = ndimage.maximum(band_data, effective_labels, segment_ids)
            features[f"median_{bname}"] = ndimage.median(band_data, effective_labels, segment_ids)

        # Compute range from min/max
        for b in range(n_bands):
            bname = names[b]
            min_vals = np.asarray(features[f"min_{bname}"])
            max_vals = np.asarray(features[f"max_{bname}"])
            features[f"range_{bname}"] = max_vals - min_vals

        # Brightness: mean across all band means
        mean_arrays = [np.asarray(features[f"mean_{names[b]}"]) for b in range(n_bands)]
        features["brightness"] = np.mean(mean_arrays, axis=0)

        # Band ratios
        if self.ratios:
            nir_idx = self._get_band_index("nir")
            red_idx = self._get_band_index("red")
            green_idx = self._get_band_index("green")

            if nir_idx is not None and red_idx is not None:
                nir_mean = np.asarray(features[f"mean_{names[nir_idx]}"])
                red_mean = np.asarray(features[f"mean_{names[red_idx]}"])
                denom = nir_mean + red_mean
                features["ndvi"] = np.where(denom != 0, (nir_mean - red_mean) / denom, 0.0)

            if green_idx is not None and nir_idx is not None:
                green_mean = np.asarray(features[f"mean_{names[green_idx]}"])
                nir_mean = np.asarray(features[f"mean_{names[nir_idx]}"])
                denom = green_mean + nir_mean
                features["ndwi"] = np.where(denom != 0, (green_mean - nir_mean) / denom, 0.0)

        df = pd.DataFrame(features, index=segment_ids)
        df.index.name = "segment_id"
        return df

    @classmethod
    def feature_names(cls, n_bands: int = 4, **kwargs) -> list[str]:
        names = []
        for i in range(n_bands):
            for stat in ("mean", "std", "min", "max", "median", "range"):
                names.append(f"{stat}_band_{i}")
        names.append("brightness")
        names.extend(["ndvi", "ndwi"])
        return names
