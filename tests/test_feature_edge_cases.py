"""Edge case tests for feature extraction."""

import numpy as np
import pytest

from geobia.features import extract
from geobia.features.context import ContextExtractor
from geobia.features.geometry import GeometryExtractor
from geobia.features.spectral import SpectralExtractor
from geobia.features.texture import TextureExtractor


class TestSpectralEdgeCases:
    def test_single_band_image(self):
        image = np.random.RandomState(0).rand(1, 50, 50).astype(np.float32)
        labels = np.ones((50, 50), dtype=np.int32)
        ext = SpectralExtractor()
        df = ext.extract(image, labels)
        assert len(df) == 1
        assert "mean_band_0" in df.columns

    def test_2d_image_auto_expand(self):
        """A 2D image should be treated as single-band."""
        image = np.random.RandomState(0).rand(50, 50).astype(np.float32)
        labels = np.ones((50, 50), dtype=np.int32)
        ext = SpectralExtractor()
        df = ext.extract(image, labels)
        assert len(df) == 1
        assert "mean_band_0" in df.columns

    def test_single_pixel_segment(self):
        image = np.random.RandomState(0).rand(2, 10, 10).astype(np.float32)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[5, 5] = 1
        ext = SpectralExtractor()
        df = ext.extract(image, labels)
        assert len(df) == 1
        # Mean of a single pixel should equal the pixel value
        np.testing.assert_almost_equal(df.loc[1, "mean_band_0"], image[0, 5, 5])

    def test_nodata_excludes_pixels(self):
        image = np.ones((2, 10, 10), dtype=np.float32) * 5.0
        image[:, 0, 0] = -9999
        labels = np.ones((10, 10), dtype=np.int32)
        ext = SpectralExtractor()
        df = ext.extract(image, labels, nodata=-9999)
        # Mean should still be 5.0 since nodata pixel is excluded
        assert abs(df.loc[1, "mean_band_0"] - 5.0) < 0.01

    def test_all_nodata_segment(self):
        image = np.full((2, 10, 10), -9999, dtype=np.float32)
        labels = np.ones((10, 10), dtype=np.int32)
        ext = SpectralExtractor()
        # All pixels are nodata so effective_labels will be all 0,
        # but segment_ids are computed from original labels before masking
        df = ext.extract(image, labels, nodata=-9999)
        # The segment still appears in output since IDs come from original labels
        # but feature values will be NaN since all pixels are masked
        assert len(df) <= 1

    def test_nan_values_in_image(self):
        image = np.ones((2, 10, 10), dtype=np.float32)
        image[0, 5, 5] = np.nan
        labels = np.ones((10, 10), dtype=np.int32)
        ext = SpectralExtractor()
        df = ext.extract(image, labels)
        # Should still produce a result, NaN propagates through stats
        assert len(df) == 1

    def test_many_bands(self):
        """Image with more than 4 bands."""
        image = np.random.RandomState(0).rand(8, 20, 20).astype(np.float32)
        labels = np.ones((20, 20), dtype=np.int32)
        ext = SpectralExtractor()
        df = ext.extract(image, labels)
        assert "mean_band_7" in df.columns


class TestGeometryEdgeCases:
    def test_invalid_labels_dimension(self):
        image = np.zeros((2, 10, 10))
        labels = np.ones((2, 10, 10), dtype=np.int32)
        ext = GeometryExtractor()
        with pytest.raises(ValueError, match="2D"):
            ext.extract(image, labels)

    def test_single_pixel_segment(self):
        image = np.zeros((2, 10, 10))
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[5, 5] = 1
        ext = GeometryExtractor()
        df = ext.extract(image, labels)
        assert df.loc[1, "area_px"] == 1

    def test_large_segment(self):
        image = np.zeros((2, 100, 100))
        labels = np.ones((100, 100), dtype=np.int32)
        ext = GeometryExtractor()
        df = ext.extract(image, labels)
        assert df.loc[1, "area_px"] == 10000


class TestTextureEdgeCases:
    def test_2d_image_auto_expand(self):
        image = np.random.RandomState(0).rand(20, 20).astype(np.float32)
        labels = np.ones((20, 20), dtype=np.int32)
        ext = TextureExtractor()
        df = ext.extract(image, labels)
        assert len(df) == 1

    def test_small_segment_returns_zeros(self):
        """Segments with < 4 pixels should get zero texture values."""
        image = np.random.RandomState(0).rand(2, 20, 20).astype(np.float32)
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[0, 0] = 1  # single pixel
        ext = TextureExtractor()
        df = ext.extract(image, labels)
        assert len(df) == 1
        assert (df.iloc[0] == 0.0).all()

    def test_constant_image(self):
        """GLCM on constant image should have zero contrast."""
        image = np.full((1, 20, 20), 0.5, dtype=np.float32)
        labels = np.ones((20, 20), dtype=np.int32)
        ext = TextureExtractor()
        df = ext.extract(image, labels)
        assert df.loc[1, "glcm_contrast_b0"] == 0.0


class TestContextEdgeCases:
    def test_single_segment_no_neighbors(self):
        image = np.random.RandomState(0).rand(2, 20, 20).astype(np.float32)
        labels = np.ones((20, 20), dtype=np.int32)
        ext = ContextExtractor()
        df = ext.extract(image, labels)
        assert df.loc[1, "n_neighbors"] == 0

    def test_two_adjacent_segments(self):
        image = np.random.RandomState(0).rand(2, 20, 20).astype(np.float32)
        labels = np.ones((20, 20), dtype=np.int32)
        labels[:, 10:] = 2
        ext = ContextExtractor()
        df = ext.extract(image, labels)
        assert df.loc[1, "n_neighbors"] == 1
        assert df.loc[2, "n_neighbors"] == 1


class TestCompositeEdgeCases:
    def test_extract_with_band_names(self):
        """Band names should propagate through the extract function."""
        image = np.random.RandomState(0).rand(4, 20, 20).astype(np.float32)
        labels = np.ones((20, 20), dtype=np.int32)
        df = extract(
            image,
            labels,
            categories=["spectral"],
            band_names={"red": 0, "green": 1, "blue": 2, "nir": 3},
        )
        assert "mean_red" in df.columns
        assert "ndvi" in df.columns

    def test_extract_with_pixel_size(self):
        """pixel_size should propagate to geometry extractor."""
        image = np.random.RandomState(0).rand(2, 20, 20).astype(np.float32)
        labels = np.ones((20, 20), dtype=np.int32)
        df = extract(image, labels, categories=["geometry"], pixel_size=10.0)
        # area = 400 pixels * 100 m²/px = 40000
        assert abs(df.loc[1, "area"] - 40000) < 1
