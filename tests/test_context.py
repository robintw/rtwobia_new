"""Tests for contextual feature extraction."""

import numpy as np
import pytest

from geobia.features.context import ContextExtractor, _find_neighbors
from geobia.features import extract


def test_find_neighbors(synthetic_labels):
    neighbors = _find_neighbors(synthetic_labels)
    # 4 quadrant segments: 1 touches 2 and 3, 2 touches 1 and 4, etc.
    assert 2 in neighbors[1]
    assert 3 in neighbors[1]
    assert 1 in neighbors[2]
    assert 4 in neighbors[2]


def test_context_basic(synthetic_image, synthetic_labels):
    ext = ContextExtractor()
    df = ext.extract(synthetic_image, synthetic_labels)
    assert len(df) == 4
    assert "n_neighbors" in df.columns
    assert "nbr_brightness_mean" in df.columns
    assert "nbr_brightness_std" in df.columns
    assert "border_contrast" in df.columns


def test_context_neighbor_count(synthetic_image, synthetic_labels):
    ext = ContextExtractor()
    df = ext.extract(synthetic_image, synthetic_labels)
    # Corner segments have 2 neighbors, not 3
    for sid in [1, 2, 3, 4]:
        assert df.loc[sid, "n_neighbors"] == 2


def test_context_via_registry(synthetic_image, synthetic_labels):
    df = extract(synthetic_image, synthetic_labels, categories=["context"])
    assert "n_neighbors" in df.columns
    assert "border_contrast" in df.columns


def test_context_border_contrast_positive(synthetic_image, synthetic_labels):
    ext = ContextExtractor()
    df = ext.extract(synthetic_image, synthetic_labels)
    # Our quadrants have different spectral signatures, so contrast > 0
    assert (df["border_contrast"] > 0).all()


def test_context_empty_labels(synthetic_image):
    empty = np.zeros((100, 100), dtype=np.int32)
    ext = ContextExtractor()
    df = ext.extract(synthetic_image, empty)
    assert len(df) == 0


def test_context_feature_names():
    names = ContextExtractor.feature_names()
    assert "n_neighbors" in names
    assert "border_contrast" in names
