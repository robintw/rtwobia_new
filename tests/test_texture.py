"""Tests for GLCM texture feature extraction."""

import numpy as np
import pandas as pd
import pytest

from geobia.features.texture import TextureExtractor
from geobia.features import extract


class TestTextureExtractor:
    def test_produces_dataframe(self, synthetic_image, synthetic_labels):
        ext = TextureExtractor()
        df = ext.extract(synthetic_image, synthetic_labels)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert df.index.name == "segment_id"

    def test_has_glcm_columns(self, synthetic_image, synthetic_labels):
        ext = TextureExtractor()
        df = ext.extract(synthetic_image, synthetic_labels)
        assert "glcm_contrast_b0" in df.columns
        assert "glcm_homogeneity_b0" in df.columns
        assert "glcm_energy_b0" in df.columns
        assert "glcm_correlation_b0" in df.columns

    def test_band_selection(self, synthetic_image, synthetic_labels):
        ext = TextureExtractor(bands=[0, 2])
        df = ext.extract(synthetic_image, synthetic_labels)
        assert "glcm_contrast_b0" in df.columns
        assert "glcm_contrast_b2" in df.columns
        assert "glcm_contrast_b1" not in df.columns

    def test_values_are_finite(self, synthetic_image, synthetic_labels):
        ext = TextureExtractor()
        df = ext.extract(synthetic_image, synthetic_labels)
        assert df.notna().all().all()
        assert np.isfinite(df.values).all()

    def test_empty_labels(self, synthetic_image):
        labels = np.zeros((100, 100), dtype=np.int32)
        ext = TextureExtractor()
        df = ext.extract(synthetic_image, labels)
        assert len(df) == 0

    def test_registered_in_extract(self, synthetic_image, synthetic_labels):
        df = extract(synthetic_image, synthetic_labels, categories=["texture"])
        assert "glcm_contrast_b0" in df.columns
        assert len(df) == 4

    def test_combined_with_other_features(self, synthetic_image, synthetic_labels):
        df = extract(
            synthetic_image,
            synthetic_labels,
            categories=["spectral", "geometry", "texture"],
        )
        assert "mean_band_0" in df.columns
        assert "area_px" in df.columns
        assert "glcm_contrast_b0" in df.columns
