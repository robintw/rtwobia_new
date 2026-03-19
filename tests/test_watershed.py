"""Tests for watershed segmentation."""

import numpy as np
import pytest

from geobia.segmentation.watershed import WatershedSegmenter
from geobia.segmentation import segment, list_methods


def test_watershed_registered():
    assert "watershed" in list_methods()


def test_watershed_basic(synthetic_image):
    seg = WatershedSegmenter(markers=20, min_distance=5)
    labels = seg.segment(synthetic_image)
    assert labels.shape == synthetic_image.shape[1:]
    assert labels.dtype == np.int32
    assert labels.min() >= 0
    assert labels.max() > 0


def test_watershed_via_convenience(synthetic_image):
    labels = segment(synthetic_image, method="watershed", markers=20, min_distance=5)
    assert labels.shape == synthetic_image.shape[1:]
    assert labels.max() > 0


def test_watershed_respects_nodata(synthetic_image):
    nodata = np.zeros(synthetic_image.shape[1:], dtype=bool)
    nodata[:10, :10] = True
    seg = WatershedSegmenter(markers=20, min_distance=5)
    labels = seg.segment(synthetic_image, nodata_mask=nodata)
    assert np.all(labels[:10, :10] == 0)


def test_watershed_params():
    seg = WatershedSegmenter(markers=100, compactness=0.5, min_distance=8)
    params = seg.get_params()
    assert params["algorithm"] == "watershed"
    assert params["markers"] == 100
    assert params["compactness"] == 0.5
    assert params["min_distance"] == 8


def test_watershed_schema():
    schema = WatershedSegmenter.get_param_schema()
    assert "markers" in schema["properties"]
    assert "compactness" in schema["properties"]
    assert "min_distance" in schema["properties"]


def test_watershed_fewer_markers(synthetic_image):
    labels_many = segment(synthetic_image, method="watershed", markers=100, min_distance=3)
    labels_few = segment(synthetic_image, method="watershed", markers=10, min_distance=15)
    assert labels_many.max() > labels_few.max()
