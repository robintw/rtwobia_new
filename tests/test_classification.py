"""Tests for classification module."""

import numpy as np
import pandas as pd
import pytest

from geobia.classification import classify
from geobia.classification.supervised import SupervisedClassifier
from geobia.classification.unsupervised import UnsupervisedClassifier
from geobia.classification.accuracy import assess_accuracy, AccuracyReport


@pytest.fixture
def sample_features():
    """Feature DataFrame with 4 distinct clusters."""
    rng = np.random.RandomState(42)
    n_per_class = 25
    data = {
        "feat_a": np.concatenate([
            rng.normal(0, 0.5, n_per_class),
            rng.normal(5, 0.5, n_per_class),
            rng.normal(0, 0.5, n_per_class),
            rng.normal(5, 0.5, n_per_class),
        ]),
        "feat_b": np.concatenate([
            rng.normal(0, 0.5, n_per_class),
            rng.normal(0, 0.5, n_per_class),
            rng.normal(5, 0.5, n_per_class),
            rng.normal(5, 0.5, n_per_class),
        ]),
    }
    df = pd.DataFrame(data, index=range(1, 101))
    df.index.name = "segment_id"
    return df


@pytest.fixture
def sample_labels():
    """Training labels for 4 classes."""
    labels = (
        ["vegetation"] * 25
        + ["urban"] * 25
        + ["water"] * 25
        + ["soil"] * 25
    )
    return pd.Series(labels, index=range(1, 101), name="class_label")


class TestSupervisedClassifier:
    def test_fit_and_predict(self, sample_features, sample_labels):
        clf = SupervisedClassifier("random_forest", n_estimators=20)
        clf.fit(sample_features, sample_labels)
        preds = clf.predict(sample_features)
        assert len(preds) == 100
        assert preds.index.equals(sample_features.index)

    def test_high_accuracy_on_separable_data(self, sample_features, sample_labels):
        clf = SupervisedClassifier("random_forest", n_estimators=50)
        clf.fit(sample_features, sample_labels)
        preds = clf.predict(sample_features)
        accuracy = (preds == sample_labels).mean()
        assert accuracy > 0.9

    def test_predict_proba(self, sample_features, sample_labels):
        clf = SupervisedClassifier("random_forest", n_estimators=20)
        clf.fit(sample_features, sample_labels)
        proba = clf.predict_proba(sample_features)
        assert isinstance(proba, pd.DataFrame)
        assert len(proba) == 100
        # Probabilities sum to ~1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0, decimal=5)

    def test_feature_importance(self, sample_features, sample_labels):
        clf = SupervisedClassifier("random_forest", n_estimators=20)
        clf.fit(sample_features, sample_labels)
        imp = clf.feature_importance()
        assert len(imp) == 2
        assert imp.index[0] in ("feat_a", "feat_b")

    def test_requires_labels(self, sample_features):
        clf = SupervisedClassifier("random_forest")
        with pytest.raises(ValueError, match="requires training labels"):
            clf.fit(sample_features)

    def test_save_and_load(self, tmp_path, sample_features, sample_labels):
        clf = SupervisedClassifier("random_forest", n_estimators=20)
        clf.fit(sample_features, sample_labels)
        preds_before = clf.predict(sample_features)

        path = str(tmp_path / "model.joblib")
        clf.save(path)
        loaded = SupervisedClassifier.load(path)
        preds_after = loaded.predict(sample_features)

        pd.testing.assert_series_equal(preds_before, preds_after)


class TestUnsupervisedClassifier:
    def test_fit_and_predict(self, sample_features):
        clf = UnsupervisedClassifier("kmeans", n_clusters=4)
        clf.fit(sample_features)
        preds = clf.predict(sample_features)
        assert len(preds) == 100
        assert preds.min() >= 1  # 1-indexed clusters

    def test_finds_correct_cluster_count(self, sample_features):
        clf = UnsupervisedClassifier("kmeans", n_clusters=4)
        clf.fit(sample_features)
        preds = clf.predict(sample_features)
        assert preds.nunique() == 4


class TestAccuracy:
    def test_perfect_accuracy(self):
        true = pd.Series(["a", "b", "a", "b"], index=[1, 2, 3, 4])
        pred = pd.Series(["a", "b", "a", "b"], index=[1, 2, 3, 4])
        report = assess_accuracy(true, pred)
        assert report.overall_accuracy == 1.0
        assert report.kappa == 1.0

    def test_partial_accuracy(self):
        true = pd.Series(["a", "b", "a", "b"], index=[1, 2, 3, 4])
        pred = pd.Series(["a", "a", "a", "b"], index=[1, 2, 3, 4])
        report = assess_accuracy(true, pred)
        assert report.overall_accuracy == 0.75
        assert report.kappa < 1.0

    def test_report_summary(self):
        true = pd.Series(["a", "b"], index=[1, 2])
        pred = pd.Series(["a", "b"], index=[1, 2])
        report = assess_accuracy(true, pred)
        summary = report.summary()
        assert "Overall Accuracy" in summary

    def test_to_dict(self):
        true = pd.Series(["a", "b"], index=[1, 2])
        pred = pd.Series(["a", "b"], index=[1, 2])
        report = assess_accuracy(true, pred)
        d = report.to_dict()
        assert "overall_accuracy" in d
        assert "kappa" in d


class TestClassifyConvenience:
    def test_kmeans(self, sample_features):
        preds = classify(sample_features, method="kmeans", n_clusters=4)
        assert len(preds) == 100

    def test_random_forest(self, sample_features, sample_labels):
        preds = classify(sample_features, method="random_forest",
                         training_labels=sample_labels, n_estimators=20)
        assert len(preds) == 100
