"""Tests for multi-scale / hierarchical segmentation."""

import numpy as np

from geobia.segmentation.multiscale import (
    segment_multiscale,
)


def test_multiscale_slic(synthetic_image):
    hier = segment_multiscale(
        synthetic_image,
        method="slic",
        scales=[
            {"n_segments": 100, "compactness": 10},
            {"n_segments": 10, "compactness": 10},
        ],
    )
    assert hier.n_levels == 2
    assert hier.finest.n_segments > hier.coarsest.n_segments


def test_multiscale_default_scales(synthetic_image):
    hier = segment_multiscale(synthetic_image, method="slic")
    assert hier.n_levels == 3
    # Ordered finest to coarsest
    for i in range(hier.n_levels - 1):
        assert hier.levels[i].n_segments >= hier.levels[i + 1].n_segments


def test_parent_map(synthetic_image):
    hier = segment_multiscale(
        synthetic_image,
        method="slic",
        scales=[
            {"n_segments": 50, "compactness": 10},
            {"n_segments": 5, "compactness": 10},
        ],
    )
    pmap = hier.parent_map(fine_idx=0, coarse_idx=1)
    # Every fine segment should have a parent
    fine_ids = set(np.unique(hier.finest.labels))
    fine_ids.discard(0)
    assert set(pmap.keys()) == fine_ids
    # Parents should be valid coarse segment IDs
    coarse_ids = set(np.unique(hier.coarsest.labels))
    coarse_ids.discard(0)
    assert set(pmap.values()).issubset(coarse_ids)


def test_cross_scale_features(synthetic_image):
    hier = segment_multiscale(
        synthetic_image,
        method="slic",
        scales=[
            {"n_segments": 50, "compactness": 10},
            {"n_segments": 5, "compactness": 10},
        ],
    )
    df = hier.cross_scale_features(fine_idx=0, coarse_idx=1)
    assert "parent_id" in df.columns
    assert "area_ratio" in df.columns
    assert "n_siblings" in df.columns
    # Area ratios should be between 0 and 1
    assert (df["area_ratio"] > 0).all()
    assert (df["area_ratio"] <= 1).all()


def test_multiscale_watershed(synthetic_image):
    hier = segment_multiscale(
        synthetic_image,
        method="watershed",
        scales=[
            {"markers": 50, "min_distance": 5},
            {"markers": 10, "min_distance": 15},
        ],
    )
    assert hier.n_levels == 2
    assert hier.finest.n_segments > hier.coarsest.n_segments


def test_level_labels_shape(synthetic_image):
    hier = segment_multiscale(
        synthetic_image,
        method="slic",
        scales=[{"n_segments": 20}],
    )
    assert hier.levels[0].labels.shape == synthetic_image.shape[1:]
