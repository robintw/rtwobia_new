"""Tests for tiled segmentation."""

import numpy as np
import pytest
import rasterio

from geobia.segmentation import segment_tiled


@pytest.fixture
def large_raster(tmp_path, synthetic_meta):
    """Create a larger raster (200x200) for tiled processing tests."""
    rng = np.random.RandomState(42)
    h, w = 200, 200
    image = np.zeros((4, h, w), dtype=np.float32)

    # 4 quadrant pattern
    image[:, :100, :100] = rng.normal(0.2, 0.02, (4, 100, 100))
    image[:, :100, 100:] = rng.normal(0.5, 0.02, (4, 100, 100))
    image[:, 100:, :100] = rng.normal(0.1, 0.02, (4, 100, 100))
    image[:, 100:, 100:] = rng.normal(0.4, 0.02, (4, 100, 100))
    image = np.clip(image, 0, 1)

    path = str(tmp_path / "large.tif")
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": w,
        "height": h,
        "count": 4,
        "crs": synthetic_meta["crs"],
        "transform": synthetic_meta["transform"],
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(image)
    return path


class TestSegmentTiled:
    def test_produces_labels(self, large_raster):
        labels = segment_tiled(
            large_raster, method="slic", tile_size=128, overlap=32, n_segments=20
        )
        assert labels.shape == (200, 200)
        assert labels.dtype == np.int32
        assert labels.max() > 0

    def test_no_gaps_in_output(self, large_raster):
        """Every pixel should be assigned to a segment."""
        labels = segment_tiled(
            large_raster, method="slic", tile_size=128, overlap=32, n_segments=20
        )
        assert (labels > 0).all()

    def test_writes_output_file(self, large_raster, tmp_path):
        output = str(tmp_path / "tiled_labels.tif")
        labels = segment_tiled(
            large_raster,
            method="slic",
            tile_size=128,
            overlap=32,
            output_path=output,
            n_segments=20,
        )
        with rasterio.open(output) as ds:
            assert ds.width == 200
            assert ds.height == 200
            data = ds.read(1)
            np.testing.assert_array_equal(data, labels)

    def test_tile_size_larger_than_image(self, large_raster):
        """Should work when tile_size >= image size (single tile)."""
        labels = segment_tiled(
            large_raster, method="slic", tile_size=1024, overlap=0, n_segments=20
        )
        assert labels.shape == (200, 200)
        assert labels.max() > 0

    def test_different_methods(self, large_raster):
        labels = segment_tiled(
            large_raster, method="felzenszwalb", tile_size=128, overlap=32, scale=50, min_size=20
        )
        assert labels.shape == (200, 200)
        assert labels.max() > 0

    def test_labels_globally_unique(self, large_raster):
        """Each tile's labels should be offset to be globally unique."""
        labels = segment_tiled(
            large_raster, method="slic", tile_size=128, overlap=32, n_segments=20
        )
        # The maximum label should be larger than what a single tile would produce
        assert labels.max() > 10
