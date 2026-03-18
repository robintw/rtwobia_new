"""Tests for utility functions."""

import numpy as np
import pytest

from geobia.utils.labels import count_segments, segment_sizes, relabel_sequential, get_segment_slices
from geobia.utils.vectorize import vectorize_labels


class TestLabelUtils:
    def test_count_segments(self, synthetic_labels):
        assert count_segments(synthetic_labels) == 4

    def test_count_excludes_zero(self):
        labels = np.array([[0, 1], [2, 0]], dtype=np.int32)
        assert count_segments(labels) == 2

    def test_segment_sizes(self, synthetic_labels):
        sizes = segment_sizes(synthetic_labels)
        assert sizes[1] == 2500
        assert sizes[2] == 2500
        assert len(sizes) == 4

    def test_relabel_sequential(self):
        labels = np.array([[0, 5], [10, 5]], dtype=np.int32)
        result = relabel_sequential(labels)
        assert result[0, 0] == 0  # nodata stays 0
        assert result.max() == 2
        assert set(result.flat) == {0, 1, 2}

    def test_get_segment_slices(self, synthetic_labels):
        slices = get_segment_slices(synthetic_labels)
        assert len(slices) == 4
        assert 1 in slices
        row_slice, col_slice = slices[1]
        assert row_slice == slice(0, 50)
        assert col_slice == slice(0, 50)


class TestVectorize:
    def test_vectorize_produces_geodataframe(self, synthetic_labels, synthetic_meta):
        from rasterio.transform import from_bounds
        gdf = vectorize_labels(
            synthetic_labels,
            synthetic_meta["transform"],
            synthetic_meta["crs"],
        )
        assert len(gdf) == 4
        assert "segment_id" in gdf.columns
        assert gdf.geometry is not None

    def test_vectorize_excludes_nodata(self, synthetic_meta):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:8, 2:8] = 1
        gdf = vectorize_labels(labels, synthetic_meta["transform"], synthetic_meta["crs"])
        assert len(gdf) == 1
        assert gdf.iloc[0]["segment_id"] == 1
