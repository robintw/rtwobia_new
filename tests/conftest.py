"""Shared test fixtures: synthetic imagery and segment labels."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)

SPOT_URL = "https://rtwilson.com/downloads/SPOT_ROI.tif"
SPOT_PATH = Path(__file__).parent / "data" / "SPOT_ROI.tif"


@pytest.fixture(scope="session")
def spot_path():
    """Return path to SPOT_ROI.tif, downloading it if not already cached.

    The file is stored in tests/data/ (gitignored) and reused across runs.
    Tests that need this fixture will fail with a skip if the download fails.
    """
    if SPOT_PATH.exists():
        return SPOT_PATH

    SPOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading SPOT_ROI.tif from %s ...", SPOT_URL)
    try:
        urllib.request.urlretrieve(SPOT_URL, SPOT_PATH)
    except Exception as exc:
        pytest.skip(f"Could not download SPOT_ROI.tif: {exc}")

    return SPOT_PATH


@pytest.fixture
def synthetic_image():
    """4-band synthetic image (bands, height, width) with spatial structure."""
    rng = np.random.RandomState(42)
    h, w = 100, 100
    image = np.zeros((4, h, w), dtype=np.float32)

    # Create spatial structure: 4 quadrants with distinct spectral signatures
    # Top-left: high NIR (vegetation-like)
    image[0, :50, :50] = rng.normal(0.15, 0.02, (50, 50))  # red
    image[1, :50, :50] = rng.normal(0.20, 0.02, (50, 50))  # green
    image[2, :50, :50] = rng.normal(0.10, 0.02, (50, 50))  # blue
    image[3, :50, :50] = rng.normal(0.55, 0.03, (50, 50))  # nir

    # Top-right: bright (urban-like)
    image[0, :50, 50:] = rng.normal(0.40, 0.03, (50, 50))
    image[1, :50, 50:] = rng.normal(0.38, 0.03, (50, 50))
    image[2, :50, 50:] = rng.normal(0.35, 0.03, (50, 50))
    image[3, :50, 50:] = rng.normal(0.42, 0.03, (50, 50))

    # Bottom-left: dark (water-like)
    image[0, 50:, :50] = rng.normal(0.05, 0.01, (50, 50))
    image[1, 50:, :50] = rng.normal(0.08, 0.01, (50, 50))
    image[2, 50:, :50] = rng.normal(0.12, 0.02, (50, 50))
    image[3, 50:, :50] = rng.normal(0.03, 0.01, (50, 50))

    # Bottom-right: moderate (soil-like)
    image[0, 50:, 50:] = rng.normal(0.30, 0.02, (50, 50))
    image[1, 50:, 50:] = rng.normal(0.25, 0.02, (50, 50))
    image[2, 50:, 50:] = rng.normal(0.20, 0.02, (50, 50))
    image[3, 50:, 50:] = rng.normal(0.28, 0.02, (50, 50))

    return np.clip(image, 0, 1)


@pytest.fixture
def synthetic_meta():
    """Metadata dict for a 100x100 image in UTM zone 33N."""
    return {
        "crs": CRS.from_epsg(32633),
        "transform": from_bounds(500000, 5000000, 501000, 5001000, 100, 100),
        "nodata": None,
        "dtype": "float32",
        "width": 100,
        "height": 100,
        "count": 4,
    }


@pytest.fixture
def synthetic_labels():
    """Simple segment label array with 4 segments matching quadrants."""
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[:50, :50] = 1
    labels[:50, 50:] = 2
    labels[50:, :50] = 3
    labels[50:, 50:] = 4
    return labels


@pytest.fixture
def raster_file(tmp_path, synthetic_image, synthetic_meta):
    """Write synthetic image to a temporary GeoTIFF and return its path."""
    path = str(tmp_path / "test_image.tif")
    bands, h, w = synthetic_image.shape
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": w,
        "height": h,
        "count": bands,
        "crs": synthetic_meta["crs"],
        "transform": synthetic_meta["transform"],
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(synthetic_image)
    return path


@pytest.fixture
def labels_file(tmp_path, synthetic_labels, synthetic_meta):
    """Write synthetic labels to a temporary GeoTIFF and return its path."""
    path = str(tmp_path / "test_labels.tif")
    h, w = synthetic_labels.shape
    profile = {
        "driver": "GTiff",
        "dtype": "int32",
        "width": w,
        "height": h,
        "count": 1,
        "crs": synthetic_meta["crs"],
        "transform": synthetic_meta["transform"],
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(synthetic_labels[np.newaxis, ...])
    return path
