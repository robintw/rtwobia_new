"""Tests for accuracy assessment and cross-validation."""

import numpy as np
import pandas as pd
import pytest

from geobia.classification.accuracy import assess_accuracy, cross_validate
from geobia.classification.supervised import SupervisedClassifier


@pytest.fixture
def cv_features():
    """Separable features for cross-validation testing."""
    rng = np.random.RandomState(42)
    n = 40
    data = {
        "f1": np.concatenate([rng.normal(0, 0.3, n), rng.normal(5, 0.3, n)]),
        "f2": np.concatenate([rng.normal(0, 0.3, n), rng.normal(5, 0.3, n)]),
    }
    df = pd.DataFrame(data, index=range(1, 2 * n + 1))
    df.index.name = "segment_id"
    return df


@pytest.fixture
def cv_labels():
    """Labels for cross-validation testing."""
    labels = ["A"] * 40 + ["B"] * 40
    return pd.Series(labels, index=range(1, 81), name="class_label")


class TestCrossValidate:
    def test_returns_expected_keys(self, cv_features, cv_labels):
        clf = SupervisedClassifier("random_forest", n_estimators=20)
        result = cross_validate(clf, cv_features, cv_labels, cv=3)
        assert "mean_accuracy" in result
        assert "std_accuracy" in result
        assert "fold_scores" in result

    def test_fold_count_matches_cv(self, cv_features, cv_labels):
        clf = SupervisedClassifier("random_forest", n_estimators=20)
        result = cross_validate(clf, cv_features, cv_labels, cv=5)
        assert len(result["fold_scores"]) == 5

    def test_high_accuracy_on_separable_data(self, cv_features, cv_labels):
        clf = SupervisedClassifier("random_forest", n_estimators=50)
        result = cross_validate(clf, cv_features, cv_labels, cv=3)
        assert result["mean_accuracy"] > 0.8

    def test_handles_partial_index_overlap(self, cv_features, cv_labels):
        """Should align features and labels on common indices."""
        # Use a subset of labels
        partial_labels = cv_labels.iloc[:60]
        clf = SupervisedClassifier("random_forest", n_estimators=20)
        result = cross_validate(clf, cv_features, partial_labels, cv=3)
        assert result["mean_accuracy"] > 0


class TestAssessAccuracy:
    def test_partial_overlap_indices(self):
        true = pd.Series(["a", "b", "a"], index=[1, 2, 3])
        pred = pd.Series(["a", "b", "b"], index=[1, 2, 4])
        report = assess_accuracy(true, pred)
        # Only indices 1 and 2 are common
        assert report.overall_accuracy == 1.0

    def test_multiclass(self):
        true = pd.Series(["a", "b", "c", "a", "b", "c"], index=range(6))
        pred = pd.Series(["a", "b", "c", "a", "b", "a"], index=range(6))
        report = assess_accuracy(true, pred)
        assert report.overall_accuracy == pytest.approx(5 / 6)
        assert len(report.class_names) == 3
