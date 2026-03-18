"""Tests for segmentation algorithms."""

import numpy as np
import pytest

from geobia.segmentation import create, segment, list_methods
from geobia.segmentation.slic import SLICSegmenter
from geobia.segmentation.felzenszwalb import FelzenszwalbSegmenter


class TestSLICSegmenter:
    def test_produces_labeled_output(self, synthetic_image):
        seg = SLICSegmenter(n_segments=20)
        labels = seg.segment(synthetic_image)
        assert labels.shape == (100, 100)
        assert labels.dtype == np.int32
        assert labels.min() >= 0
        assert labels.max() > 0

    def test_respects_nodata_mask(self, synthetic_image):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:10, :10] = True
        seg = SLICSegmenter(n_segments=20)
        labels = seg.segment(synthetic_image, nodata_mask=mask)
        assert np.all(labels[:10, :10] == 0)

    def test_get_params(self):
        seg = SLICSegmenter(n_segments=100, compactness=5.0)
        params = seg.get_params()
        assert params["algorithm"] == "slic"
        assert params["n_segments"] == 100
        assert params["compactness"] == 5.0


class TestFelzenszwalbSegmenter:
    def test_produces_labeled_output(self, synthetic_image):
        seg = FelzenszwalbSegmenter(scale=50, min_size=20)
        labels = seg.segment(synthetic_image)
        assert labels.shape == (100, 100)
        assert labels.dtype == np.int32
        assert labels.min() >= 1  # 1-indexed

    def test_respects_nodata_mask(self, synthetic_image):
        mask = np.zeros((100, 100), dtype=bool)
        mask[90:, 90:] = True
        seg = FelzenszwalbSegmenter(scale=50)
        labels = seg.segment(synthetic_image, nodata_mask=mask)
        assert np.all(labels[90:, 90:] == 0)

    def test_get_params(self):
        seg = FelzenszwalbSegmenter(scale=200, sigma=1.0, min_size=100)
        params = seg.get_params()
        assert params["algorithm"] == "felzenszwalb"
        assert params["scale"] == 200


class TestFactory:
    def test_create_slic(self):
        seg = create("slic", n_segments=100)
        assert isinstance(seg, SLICSegmenter)

    def test_create_felzenszwalb(self):
        seg = create("felzenszwalb", scale=50)
        assert isinstance(seg, FelzenszwalbSegmenter)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create("nonexistent")

    def test_list_methods(self):
        methods = list_methods()
        assert "slic" in methods
        assert "felzenszwalb" in methods


class TestConvenienceSegment:
    def test_segment_function(self, synthetic_image):
        labels = segment(synthetic_image, method="slic", n_segments=20)
        assert labels.shape == (100, 100)
        assert labels.max() > 0
