"""Edge case tests for classification module."""

import numpy as np
import pandas as pd
import pytest

from geobia.classification.supervised import SupervisedClassifier
from geobia.classification.unsupervised import UnsupervisedClassifier
from geobia.classification.fuzzy import FuzzyClassifier, FuzzyRule
from geobia.classification.base import BaseClassifier


@pytest.fixture
def small_features():
    """Very small feature set for edge case testing."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {"f1": rng.normal(0, 1, 10), "f2": rng.normal(0, 1, 10)},
        index=range(1, 11),
    )


@pytest.fixture
def small_labels():
    return pd.Series(["A"] * 5 + ["B"] * 5, index=range(1, 11), name="class_label")


class TestSupervisedEdgeCases:
    def test_predict_without_fit_raises(self, small_features):
        clf = SupervisedClassifier("random_forest")
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict(small_features)

    def test_no_common_indices_raises(self, small_features):
        labels = pd.Series(["A", "B"], index=[99, 100], name="class_label")
        clf = SupervisedClassifier("random_forest")
        with pytest.raises(ValueError, match="No common segment IDs"):
            clf.fit(small_features, labels)

    def test_two_samples(self):
        """Minimum viable training set: 2 samples, 2 classes."""
        features = pd.DataFrame({"f1": [0, 10], "f2": [0, 10]}, index=[1, 2])
        labels = pd.Series(["A", "B"], index=[1, 2])
        clf = SupervisedClassifier("random_forest", n_estimators=10)
        clf.fit(features, labels)
        preds = clf.predict(features)
        assert len(preds) == 2

    def test_single_class_training(self):
        """All training samples same class."""
        features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]}, index=[1, 2, 3])
        labels = pd.Series(["A", "A", "A"], index=[1, 2, 3])
        clf = SupervisedClassifier("random_forest", n_estimators=10)
        clf.fit(features, labels)
        preds = clf.predict(features)
        assert (preds == "A").all()


class TestUnsupervisedEdgeCases:
    def test_single_cluster(self, small_features):
        clf = UnsupervisedClassifier("kmeans", n_clusters=1)
        clf.fit(small_features)
        preds = clf.predict(small_features)
        assert preds.nunique() == 1

    def test_more_clusters_than_samples(self):
        """Requesting more clusters than samples should raise ValueError."""
        features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]}, index=[1, 2, 3])
        clf = UnsupervisedClassifier("kmeans", n_clusters=10)
        with pytest.raises(ValueError, match="n_samples.*should be >= n_clusters"):
            clf.fit(features)

    def test_dbscan_all_noise(self):
        """DBSCAN with very small eps should classify all as noise."""
        rng = np.random.RandomState(42)
        features = pd.DataFrame(
            {"f1": rng.normal(0, 10, 20), "f2": rng.normal(0, 10, 20)},
            index=range(1, 21),
        )
        clf = UnsupervisedClassifier("dbscan", eps=0.0001, min_samples=10)
        clf.fit(features)
        preds = clf.predict(features)
        assert (preds == UnsupervisedClassifier.NOISE_LABEL).all()

    def test_dbscan_noise_label_is_zero(self, small_features):
        clf = UnsupervisedClassifier("dbscan")
        clf.fit(small_features)
        preds = clf.predict(small_features)
        # Noise should be 0, clusters should be >= 1
        assert preds.min() >= 0

    def test_gmm_predict_proba(self, small_features):
        clf = UnsupervisedClassifier("gmm", n_clusters=2)
        clf.fit(small_features)
        proba = clf.predict_proba(small_features)
        assert isinstance(proba, pd.DataFrame)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_dbscan_unseen_data(self, small_features):
        """DBSCAN predict on unseen data should return noise label."""
        clf = UnsupervisedClassifier("dbscan")
        clf.fit(small_features)
        new_features = pd.DataFrame(
            {"f1": [100, 200], "f2": [100, 200]},
            index=[99, 100],
        )
        preds = clf.predict(new_features)
        assert (preds == UnsupervisedClassifier.NOISE_LABEL).all()


class TestFuzzyEdgeCases:
    def test_no_rules_raises(self):
        clf = FuzzyClassifier()
        features = pd.DataFrame({"f1": [1, 2]}, index=[1, 2])
        with pytest.raises(ValueError, match="No rules"):
            clf.predict(features)

    def test_missing_feature_raises(self):
        rules = {"cls": [FuzzyRule("nonexistent", 0, 1)]}
        clf = FuzzyClassifier(rules=rules)
        features = pd.DataFrame({"f1": [1, 2]}, index=[1, 2])
        with pytest.raises(ValueError, match="not found"):
            clf.predict(features)

    def test_all_zero_membership(self):
        """All segments outside all rule ranges -> unclassified."""
        rules = {"cls": [FuzzyRule("f1", 100, 200)]}
        clf = FuzzyClassifier(rules=rules)
        features = pd.DataFrame({"f1": [0, 1]}, index=[1, 2])
        preds = clf.predict(features)
        assert (preds == "unclassified").all()


class TestClassifierLoadValidation:
    def test_load_non_dict_raises(self, tmp_path):
        import joblib
        path = str(tmp_path / "bad_model.joblib")
        joblib.dump("not_a_dict", path)
        with pytest.raises(TypeError, match="Expected a dict"):
            SupervisedClassifier.load(path)

    def test_save_and_load_roundtrip(self, tmp_path, small_features, small_labels):
        clf = SupervisedClassifier("random_forest", n_estimators=10)
        clf.fit(small_features, small_labels)
        path = str(tmp_path / "model.joblib")
        clf.save(path)
        loaded = SupervisedClassifier.load(path)
        preds = loaded.predict(small_features)
        assert len(preds) == 10
