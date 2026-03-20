"""Parametrized tests for broader coverage across methods and configurations."""

import numpy as np
import pandas as pd
import pytest

from geobia.classification import classify
from geobia.features import extract
from geobia.segmentation import segment

# --- Segmentation ---


@pytest.mark.parametrize("method", ["slic", "felzenszwalb", "watershed"])
def test_segmentation_produces_valid_labels(method, synthetic_image):
    params = {
        "slic": {"n_segments": 20},
        "felzenszwalb": {"scale": 50, "min_size": 20},
        "watershed": {"markers": 20, "min_distance": 5},
    }
    labels = segment(synthetic_image, method=method, **params[method])
    assert labels.shape == (100, 100)
    assert labels.dtype == np.int32
    assert labels.max() > 0
    assert labels.min() >= 0


@pytest.mark.parametrize("method", ["slic", "felzenszwalb", "watershed"])
def test_segmentation_respects_nodata(method, synthetic_image):
    nodata_mask = np.zeros((100, 100), dtype=bool)
    nodata_mask[:10, :10] = True
    params = {
        "slic": {"n_segments": 20},
        "felzenszwalb": {"scale": 50, "min_size": 20},
        "watershed": {"markers": 20, "min_distance": 5},
    }
    labels = segment(synthetic_image, method=method, nodata_mask=nodata_mask, **params[method])
    assert labels[0, 0] == 0


# --- Feature extraction ---


@pytest.mark.parametrize("category", ["spectral", "geometry", "texture", "context"])
def test_feature_category_produces_output(category, synthetic_image, synthetic_labels):
    df = extract(synthetic_image, synthetic_labels, categories=[category])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert df.index.name == "segment_id"


@pytest.mark.parametrize(
    "categories",
    [
        ["spectral"],
        ["spectral", "geometry"],
        ["spectral", "geometry", "texture"],
        ["spectral", "geometry", "texture", "context"],
    ],
)
def test_feature_category_combinations(categories, synthetic_image, synthetic_labels):
    df = extract(synthetic_image, synthetic_labels, categories=categories)
    assert len(df) == 4
    assert len(df.columns) > 0


# --- Classification ---


@pytest.fixture
def class_features():
    rng = np.random.RandomState(42)
    n = 20
    data = {
        "f1": np.concatenate([rng.normal(0, 0.5, n), rng.normal(5, 0.5, n)]),
        "f2": np.concatenate([rng.normal(0, 0.5, n), rng.normal(5, 0.5, n)]),
    }
    df = pd.DataFrame(data, index=range(1, 2 * n + 1))
    df.index.name = "segment_id"
    return df


@pytest.fixture
def class_labels():
    return pd.Series(
        ["A"] * 20 + ["B"] * 20,
        index=range(1, 41),
        name="class_label",
    )


@pytest.mark.parametrize("method", ["kmeans", "gmm"])
def test_unsupervised_methods(method, class_features):
    preds = classify(class_features, method=method, n_clusters=2)
    assert len(preds) == 40
    assert preds.nunique() >= 1


@pytest.mark.parametrize(
    "method,params",
    [
        ("random_forest", {"n_estimators": 10}),
        ("svm", {}),
        ("gradient_boosting", {"n_estimators": 10}),
    ],
)
def test_supervised_methods(method, params, class_features, class_labels):
    preds = classify(
        class_features,
        method=method,
        training_labels=class_labels,
        **params,
    )
    assert len(preds) == 40


@pytest.mark.parametrize(
    "method,params",
    [
        ("random_forest", {"n_estimators": 20}),
        ("svm", {}),
        ("gradient_boosting", {"n_estimators": 20}),
    ],
)
def test_supervised_accuracy(method, params, class_features, class_labels):
    preds = classify(
        class_features,
        method=method,
        training_labels=class_labels,
        **params,
    )
    accuracy = (preds == class_labels).mean()
    assert accuracy > 0.8
