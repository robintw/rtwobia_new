"""Tests for SVM and Gradient Boosting classifiers."""

import numpy as np
import pandas as pd
import pytest

from geobia.classification import classify
from geobia.classification.supervised import SupervisedClassifier


@pytest.fixture
def training_data():
    """Create simple training features and labels."""
    rng = np.random.RandomState(42)
    n = 100
    features = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)]),
            "f2": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)]),
        },
        index=range(1, n + 1),
    )
    features.index.name = "segment_id"

    labels = pd.Series(
        ["A"] * (n // 2) + ["B"] * (n // 2),
        index=range(1, n + 1),
        name="class_label",
    )
    return features, labels


def test_svm_fit_predict(training_data):
    features, labels = training_data
    clf = SupervisedClassifier(algorithm="svm")
    clf.fit(features, labels)
    preds = clf.predict(features)
    assert len(preds) == len(features)
    assert set(preds.unique()) == {"A", "B"}


def test_svm_predict_proba(training_data):
    features, labels = training_data
    clf = SupervisedClassifier(algorithm="svm", probability=True)
    clf.fit(features, labels)
    proba = clf.predict_proba(features)
    assert proba.shape == (len(features), 2)
    assert all(col.startswith("prob_") for col in proba.columns)


def test_gradient_boosting_fit_predict(training_data):
    features, labels = training_data
    clf = SupervisedClassifier(algorithm="gradient_boosting", n_estimators=20)
    clf.fit(features, labels)
    preds = clf.predict(features)
    assert len(preds) == len(features)
    assert set(preds.unique()) == {"A", "B"}


def test_gradient_boosting_feature_importance(training_data):
    features, labels = training_data
    clf = SupervisedClassifier(algorithm="gradient_boosting", n_estimators=20)
    clf.fit(features, labels)
    imp = clf.feature_importance()
    assert len(imp) == 2
    assert imp.sum() > 0


def test_svm_via_convenience(training_data):
    features, labels = training_data
    preds = classify(features, method="svm", training_labels=labels)
    assert len(preds) == len(features)


def test_gradient_boosting_via_convenience(training_data):
    features, labels = training_data
    preds = classify(features, method="gradient_boosting", training_labels=labels, n_estimators=20)
    assert len(preds) == len(features)


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        SupervisedClassifier(algorithm="nonexistent")


def test_svm_params():
    clf = SupervisedClassifier(algorithm="svm", C=2.0, kernel="linear")
    params = clf.get_params()
    assert params["algorithm"] == "svm"
    assert params["C"] == 2.0
    assert params["kernel"] == "linear"
