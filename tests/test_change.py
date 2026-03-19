"""Tests for change detection."""

import numpy as np
import pandas as pd
import pytest

from geobia.change import (
    change_magnitude,
    change_summary,
    detect_changes,
    feature_difference,
)


@pytest.fixture
def features_t1():
    return pd.DataFrame(
        {
            "brightness": [0.2, 0.3, 0.4, 0.5],
            "ndvi": [0.6, 0.7, 0.1, 0.2],
        },
        index=pd.RangeIndex(1, 5, name="segment_id"),
    )


@pytest.fixture
def features_t2():
    return pd.DataFrame(
        {
            "brightness": [0.2, 0.3, 0.8, 0.9],  # segs 3,4 changed
            "ndvi": [0.6, 0.7, 0.5, 0.6],  # segs 3,4 changed
        },
        index=pd.RangeIndex(1, 5, name="segment_id"),
    )


def test_feature_difference(features_t1, features_t2):
    diff = feature_difference(features_t1, features_t2)
    assert len(diff) == 4
    assert diff.loc[1, "brightness"] == pytest.approx(0.0)
    assert diff.loc[3, "brightness"] == pytest.approx(0.4)


def test_change_magnitude(features_t1, features_t2):
    mag = change_magnitude(features_t1, features_t2, normalize=False)
    assert len(mag) == 4
    # Segments 1,2 unchanged
    assert mag.iloc[0] == pytest.approx(0.0)
    assert mag.iloc[1] == pytest.approx(0.0)
    # Segments 3,4 changed
    assert mag.iloc[2] > 0
    assert mag.iloc[3] > 0


def test_change_magnitude_normalized(features_t1, features_t2):
    mag = change_magnitude(features_t1, features_t2, normalize=True)
    assert len(mag) == 4
    assert mag.iloc[0] == pytest.approx(0.0)


def test_detect_changes_fixed_threshold(features_t1, features_t2):
    changed = detect_changes(features_t1, features_t2, threshold=0.1, normalize=False)
    assert changed.dtype == bool
    assert not changed.iloc[0]  # seg 1 unchanged
    assert not changed.iloc[1]  # seg 2 unchanged
    assert changed.iloc[2]  # seg 3 changed
    assert changed.iloc[3]  # seg 4 changed


def test_detect_changes_otsu(features_t1, features_t2):
    changed = detect_changes(features_t1, features_t2, threshold="otsu", normalize=False)
    assert changed.dtype == bool
    # At least some changed, some unchanged
    assert changed.sum() > 0
    assert changed.sum() < len(changed)


def test_change_summary(features_t1, features_t2):
    changed = detect_changes(features_t1, features_t2, threshold=0.1, normalize=False)
    summary = change_summary(changed, features_t1, features_t2)
    assert summary["total_segments"] == 4
    assert summary["changed"] == 2
    assert summary["unchanged"] == 2
    assert summary["pct_changed"] == 50.0
    assert "mean_diff_changed" in summary


def test_partial_overlap():
    """Features with different segment sets should use intersection."""
    t1 = pd.DataFrame({"f": [1, 2, 3]}, index=[1, 2, 3])
    t2 = pd.DataFrame({"f": [1, 5, 3]}, index=[1, 2, 4])
    diff = feature_difference(t1, t2)
    assert set(diff.index) == {1, 2}
