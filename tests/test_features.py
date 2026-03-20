"""Tests for feature extraction."""

import numpy as np
import pandas as pd
import pytest

from geobia.features import extract
from geobia.features.geometry import GeometryExtractor
from geobia.features.spectral import SpectralExtractor


class TestSpectralExtractor:
    def test_produces_dataframe(self, synthetic_image, synthetic_labels):
        ext = SpectralExtractor()
        df = ext.extract(synthetic_image, synthetic_labels)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 4 segments
        assert df.index.name == "segment_id"

    def test_has_expected_columns(self, synthetic_image, synthetic_labels):
        ext = SpectralExtractor()
        df = ext.extract(synthetic_image, synthetic_labels)
        assert "mean_band_0" in df.columns
        assert "std_band_0" in df.columns
        assert "brightness" in df.columns

    def test_band_names(self, synthetic_image, synthetic_labels):
        ext = SpectralExtractor(band_names={"red": 0, "green": 1, "blue": 2, "nir": 3})
        df = ext.extract(synthetic_image, synthetic_labels)
        assert "mean_red" in df.columns
        assert "mean_nir" in df.columns
        assert "ndvi" in df.columns
        assert "ndwi" in df.columns

    def test_ndvi_values_reasonable(self, synthetic_image, synthetic_labels):
        ext = SpectralExtractor(band_names={"red": 0, "green": 1, "blue": 2, "nir": 3})
        df = ext.extract(synthetic_image, synthetic_labels)
        # Segment 1 (top-left) has high NIR, low red -> high NDVI
        assert df.loc[1, "ndvi"] > 0.3
        # All NDVI values should be in [-1, 1]
        assert df["ndvi"].between(-1, 1).all()

    def test_mean_values_match_quadrant(self, synthetic_image, synthetic_labels):
        ext = SpectralExtractor()
        df = ext.extract(synthetic_image, synthetic_labels)
        # Top-left (segment 1) red band mean should be ~0.15
        assert abs(df.loc[1, "mean_band_0"] - 0.15) < 0.05

    def test_empty_labels(self, synthetic_image):
        labels = np.zeros((100, 100), dtype=np.int32)
        ext = SpectralExtractor()
        df = ext.extract(synthetic_image, labels)
        assert len(df) == 0


class TestGeometryExtractor:
    def test_produces_dataframe(self, synthetic_image, synthetic_labels):
        ext = GeometryExtractor()
        df = ext.extract(synthetic_image, synthetic_labels)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_area_is_correct(self, synthetic_image, synthetic_labels):
        ext = GeometryExtractor()
        df = ext.extract(synthetic_image, synthetic_labels)
        # Each quadrant is 50x50 = 2500 pixels
        assert df.loc[1, "area_px"] == 2500

    def test_compactness_range(self, synthetic_image, synthetic_labels):
        ext = GeometryExtractor()
        df = ext.extract(synthetic_image, synthetic_labels)
        assert df["compactness"].between(0, 1).all()

    def test_pixel_size_scales_area(self, synthetic_image, synthetic_labels):
        ext = GeometryExtractor(pixel_size=10.0)
        df = ext.extract(synthetic_image, synthetic_labels)
        # 2500 pixels * 100 m^2/pixel = 250000 m^2
        assert abs(df.loc[1, "area"] - 250000) < 1


class TestCompositeExtract:
    def test_combines_spectral_and_geometry(self, synthetic_image, synthetic_labels):
        df = extract(synthetic_image, synthetic_labels, categories=["spectral", "geometry"])
        assert "mean_band_0" in df.columns
        assert "area_px" in df.columns
        assert len(df) == 4

    def test_default_uses_all_categories(self, synthetic_image, synthetic_labels):
        df = extract(synthetic_image, synthetic_labels)
        assert "mean_band_0" in df.columns
        assert "area_px" in df.columns

    def test_unknown_category_raises(self, synthetic_image, synthetic_labels):
        with pytest.raises(ValueError, match="Unknown"):
            extract(synthetic_image, synthetic_labels, categories=["nonexistent"])
