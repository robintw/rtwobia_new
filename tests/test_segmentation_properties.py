"""Common-sense segmentation property tests using SPOT_ROI.tif.

These tests verify intuitive relationships between segmentation parameters
and their outputs — e.g., requesting more segments should produce more
segments, and a larger minimum size should produce fewer segments.

Marked with 'spot_image' and 'slow' to skip when SPOT_ROI.tif is absent:

    pytest tests/ -v -m "not spot_image"      # skip these
    pytest tests/ -v -m "spot_image"           # run these only
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geobia.segmentation import segment

SPOT_PATH = Path(__file__).parent / "data" / "SPOT_ROI.tif"
SPOT_AVAILABLE = SPOT_PATH.exists()

pytestmark = [
    pytest.mark.spot_image,
    pytest.mark.slow,
    pytest.mark.skipif(not SPOT_AVAILABLE, reason="SPOT_ROI.tif not found in tests/data/"),
]


@pytest.fixture(scope="module")
def spot_image():
    from geobia.io.raster import read_raster
    image, meta = read_raster(str(SPOT_PATH))
    return image


def _count_segments(labels):
    """Count non-zero unique segment IDs."""
    unique = np.unique(labels)
    return len(unique[unique != 0])


# ---------------------------------------------------------------------------
# SLIC: more n_segments → more actual segments
# ---------------------------------------------------------------------------

class TestSLICProperties:
    def test_more_n_segments_gives_more_segments(self, spot_image):
        labels_few = segment(spot_image, method="slic", n_segments=100, compactness=10.0)
        labels_many = segment(spot_image, method="slic", n_segments=1000, compactness=10.0)
        n_few = _count_segments(labels_few)
        n_many = _count_segments(labels_many)
        assert n_many > n_few, (
            f"Expected more segments with n_segments=1000 ({n_many}) "
            f"than n_segments=100 ({n_few})"
        )

    def test_higher_compactness_still_produces_segments(self, spot_image):
        labels = segment(spot_image, method="slic", n_segments=500, compactness=100.0)
        n = _count_segments(labels)
        assert n > 10, f"Expected >10 segments, got {n}"

    def test_produces_valid_labels(self, spot_image):
        labels = segment(spot_image, method="slic", n_segments=500)
        assert labels.dtype == np.int32
        assert labels.shape == spot_image.shape[1:]
        assert labels.min() >= 0


# ---------------------------------------------------------------------------
# Felzenszwalb: larger min_size → fewer segments
# ---------------------------------------------------------------------------

class TestFelzenszwalbProperties:
    def test_larger_min_size_gives_fewer_segments(self, spot_image):
        labels_small = segment(spot_image, method="felzenszwalb",
                               scale=100, min_size=20)
        labels_large = segment(spot_image, method="felzenszwalb",
                               scale=100, min_size=200)
        n_small = _count_segments(labels_small)
        n_large = _count_segments(labels_large)
        assert n_small > n_large, (
            f"Expected fewer segments with min_size=200 ({n_large}) "
            f"than min_size=20 ({n_small})"
        )

    def test_larger_scale_gives_fewer_segments(self, spot_image):
        labels_low = segment(spot_image, method="felzenszwalb",
                             scale=50, min_size=50)
        labels_high = segment(spot_image, method="felzenszwalb",
                              scale=500, min_size=50)
        n_low = _count_segments(labels_low)
        n_high = _count_segments(labels_high)
        assert n_low > n_high, (
            f"Expected fewer segments with scale=500 ({n_high}) "
            f"than scale=50 ({n_low})"
        )

    def test_produces_valid_labels(self, spot_image):
        labels = segment(spot_image, method="felzenszwalb", scale=100, min_size=50)
        assert labels.dtype == np.int32
        assert labels.shape == spot_image.shape[1:]
        assert labels.min() >= 0


# ---------------------------------------------------------------------------
# Shepherd: larger min_n_pxls → fewer segments
# ---------------------------------------------------------------------------

class TestShepherdProperties:
    def test_larger_min_pixels_gives_fewer_segments(self, spot_image):
        labels_small = segment(spot_image, method="shepherd",
                               num_clusters=60, min_n_pxls=50)
        labels_large = segment(spot_image, method="shepherd",
                               num_clusters=60, min_n_pxls=500)
        n_small = _count_segments(labels_small)
        n_large = _count_segments(labels_large)
        assert n_small > n_large, (
            f"Expected fewer segments with min_n_pxls=500 ({n_large}) "
            f"than min_n_pxls=50 ({n_small})"
        )

    def test_produces_valid_labels(self, spot_image):
        labels = segment(spot_image, method="shepherd",
                         num_clusters=60, min_n_pxls=100)
        assert labels.dtype == np.int32
        assert labels.shape == spot_image.shape[1:]
        assert labels.min() >= 0


# ---------------------------------------------------------------------------
# Watershed: fewer markers → fewer segments
# ---------------------------------------------------------------------------

class TestWatershedProperties:
    def test_fewer_markers_gives_fewer_segments(self, spot_image):
        labels_few = segment(spot_image, method="watershed",
                             markers=100, min_distance=20)
        labels_many = segment(spot_image, method="watershed",
                              markers=1000, min_distance=5)
        n_few = _count_segments(labels_few)
        n_many = _count_segments(labels_many)
        assert n_many > n_few, (
            f"Expected more segments with markers=1000 ({n_many}) "
            f"than markers=100 ({n_few})"
        )

    def test_produces_valid_labels(self, spot_image):
        labels = segment(spot_image, method="watershed",
                         markers=500, min_distance=10)
        assert labels.dtype == np.int32
        assert labels.shape == spot_image.shape[1:]
        assert labels.min() >= 0


# ---------------------------------------------------------------------------
# Cross-method: different methods give different results
# ---------------------------------------------------------------------------

class TestCrossMethodProperties:
    def test_different_methods_produce_different_results(self, spot_image):
        labels_slic = segment(spot_image, method="slic", n_segments=500)
        labels_fz = segment(spot_image, method="felzenszwalb", scale=100, min_size=50)
        # Different methods should not produce identical label arrays
        assert not np.array_equal(labels_slic, labels_fz), (
            "SLIC and Felzenszwalb produced identical label arrays"
        )

    def test_all_methods_produce_nonzero_labels(self, spot_image):
        methods_params = [
            ("slic", {"n_segments": 200}),
            ("felzenszwalb", {"scale": 100, "min_size": 50}),
            ("shepherd", {"num_clusters": 40, "min_n_pxls": 100}),
            ("watershed", {"markers": 200, "min_distance": 10}),
        ]
        for method, params in methods_params:
            labels = segment(spot_image, method=method, **params)
            n = _count_segments(labels)
            assert n > 0, f"{method} produced zero segments"
