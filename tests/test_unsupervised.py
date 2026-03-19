"""Tests for extended unsupervised classification (GMM, DBSCAN)."""

import numpy as np
import pandas as pd
import pytest

from geobia.classification.unsupervised import UnsupervisedClassifier
from geobia.classification import classify


@pytest.fixture
def clustered_features():
    """Features with 4 clearly separated clusters."""
    rng = np.random.RandomState(42)
    n = 25
    data = {
        "feat_a": np.concatenate(
            [
                rng.normal(0, 0.3, n),
                rng.normal(5, 0.3, n),
                rng.normal(0, 0.3, n),
                rng.normal(5, 0.3, n),
            ]
        ),
        "feat_b": np.concatenate(
            [
                rng.normal(0, 0.3, n),
                rng.normal(0, 0.3, n),
                rng.normal(5, 0.3, n),
                rng.normal(5, 0.3, n),
            ]
        ),
    }
    df = pd.DataFrame(data, index=range(1, 101))
    df.index.name = "segment_id"
    return df


class TestGMM:
    def test_fit_and_predict(self, clustered_features):
        clf = UnsupervisedClassifier("gmm", n_components=4)
        clf.fit(clustered_features)
        preds = clf.predict(clustered_features)
        assert len(preds) == 100
        assert preds.nunique() == 4

    def test_predict_proba(self, clustered_features):
        clf = UnsupervisedClassifier("gmm", n_components=4)
        clf.fit(clustered_features)
        proba = clf.predict_proba(clustered_features)
        assert proba is not None
        assert proba.shape == (100, 4)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0, decimal=5)

    def test_via_classify_convenience(self, clustered_features):
        preds = classify(clustered_features, method="gmm", n_components=4)
        assert len(preds) == 100


class TestDBSCAN:
    def test_fit_and_predict(self, clustered_features):
        clf = UnsupervisedClassifier("dbscan", eps=1.0, min_samples=3)
        clf.fit(clustered_features)
        preds = clf.predict(clustered_features)
        assert len(preds) == 100
        # Should find clusters (not all noise)
        assert preds.max() > 0

    def test_via_classify_convenience(self, clustered_features):
        preds = classify(clustered_features, method="dbscan", eps=1.0, min_samples=3)
        assert len(preds) == 100


class TestUnknownAlgorithm:
    def test_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown"):
            UnsupervisedClassifier("nonexistent")
