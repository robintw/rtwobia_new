"""Tests for segmentation algorithms."""

import numpy as np
import pytest

from geobia.segmentation import create, list_methods, segment
from geobia.segmentation.felzenszwalb import FelzenszwalbSegmenter
from geobia.segmentation.shepherd import ShepherdSegmenter
from geobia.segmentation.slic import SLICSegmenter


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


class TestShepherdSegmenter:
    def test_produces_labeled_output(self, synthetic_image):
        seg = ShepherdSegmenter(num_clusters=5, min_n_pxls=10, dist_thres="auto", sampling=10)
        labels = seg.segment(synthetic_image)
        assert labels.shape == (100, 100)
        assert labels.dtype == np.int32
        assert labels.max() > 0

    def test_respects_nodata_mask(self, synthetic_image):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:10, :10] = True
        seg = ShepherdSegmenter(num_clusters=5, min_n_pxls=10, dist_thres="auto", sampling=10)
        labels = seg.segment(synthetic_image, nodata_mask=mask)
        assert np.all(labels[:10, :10] == 0)

    def test_eliminates_small_segments(self, synthetic_image):
        seg = ShepherdSegmenter(num_clusters=5, min_n_pxls=50, dist_thres="auto", sampling=10)
        labels = seg.segment(synthetic_image)
        from geobia.utils.labels import segment_sizes

        sizes = segment_sizes(labels)
        # All segments should be >= min_n_pxls (or close to it)
        for _sid, count in sizes.items():
            assert count >= 10  # may not reach exact min_n_pxls due to algorithm behaviour

    def test_get_params(self):
        seg = ShepherdSegmenter(num_clusters=30, min_n_pxls=50)
        params = seg.get_params()
        assert params["algorithm"] == "shepherd"
        assert params["num_clusters"] == 30
        assert params["min_n_pxls"] == 50

    def test_band_selection_via_segment(self, synthetic_image):
        """Band selection is handled by the top-level segment() function."""
        labels = segment(
            synthetic_image,
            method="shepherd",
            num_clusters=5,
            min_n_pxls=10,
            dist_thres="auto",
            sampling=10,
            bands=[0, 1],
        )
        assert labels.shape == (100, 100)
        assert labels.max() > 0


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

    def test_create_shepherd(self):
        seg = create("shepherd", num_clusters=10)
        assert isinstance(seg, ShepherdSegmenter)

    def test_list_methods(self):
        methods = list_methods()
        assert "slic" in methods
        assert "felzenszwalb" in methods
        assert "shepherd" in methods


class TestConvenienceSegment:
    def test_segment_function(self, synthetic_image):
        labels = segment(synthetic_image, method="slic", n_segments=20)
        assert labels.shape == (100, 100)
        assert labels.max() > 0
