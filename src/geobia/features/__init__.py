"""Feature extraction module with composable extractors."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from geobia.features.base import BaseExtractor
from geobia.features.spectral import SpectralExtractor
from geobia.features.geometry import GeometryExtractor
from geobia.features.texture import TextureExtractor
from geobia.features.context import ContextExtractor

_REGISTRY: dict[str, type[BaseExtractor]] = {
    "spectral": SpectralExtractor,
    "geometry": GeometryExtractor,
    "texture": TextureExtractor,
    "context": ContextExtractor,
}


def extract(
    image: np.ndarray,
    labels: np.ndarray,
    categories: list[str] | None = None,
    progress: Any | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Extract features using one or more extractors and merge results.

    Args:
        image: Multi-band array (bands, height, width).
        labels: Segment label array (height, width).
        categories: List of extractor names to use. Defaults to all available.
        progress: Optional callable(percent: float) for progress reporting.
        **kwargs: Passed to extractors (e.g., band_names, pixel_size, nodata).

    Returns:
        Merged DataFrame with segment_id as index.
    """
    _report = progress or (lambda p: None)

    if categories is None:
        categories = list(_REGISTRY.keys())

    frames = []
    n_cats = len(categories)
    for i, cat in enumerate(categories):
        cat = cat.lower()
        if cat not in _REGISTRY:
            raise ValueError(f"Unknown feature category: {cat!r}. Available: {list(_REGISTRY)}")

        # Route kwargs to the right extractor
        extractor = _create_extractor(cat, **kwargs)
        logger.info("Extracting %s features", cat)
        df = extractor.extract(image, labels, **kwargs)
        logger.debug("Extracted %d features for %d segments", len(df.columns), len(df))
        frames.append(df)
        _report((i + 1) / n_cats * 100)

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for df in frames[1:]:
        result = result.join(df, how="outer")

    return result


def _create_extractor(category: str, **kwargs) -> BaseExtractor:
    """Create an extractor with appropriate kwargs."""
    cls = _REGISTRY[category]
    if category == "spectral":
        return cls(
            band_names=kwargs.get("band_names"),
            ratios=kwargs.get("ratios", True),
        )
    elif category == "geometry":
        return cls(pixel_size=kwargs.get("pixel_size"))
    elif category == "texture":
        return cls(
            bands=kwargs.get("texture_bands"),
            distances=kwargs.get("texture_distances"),
            levels=kwargs.get("texture_levels", 32),
        )
    elif category == "context":
        return cls()
    return cls()


def list_categories() -> list[str]:
    """Return available feature category names."""
    return list(_REGISTRY.keys())


__all__ = [
    "BaseExtractor",
    "SpectralExtractor",
    "GeometryExtractor",
    "TextureExtractor",
    "ContextExtractor",
    "extract",
    "list_categories",
]
