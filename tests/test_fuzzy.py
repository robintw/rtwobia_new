"""Tests for fuzzy membership function classification."""

import pandas as pd
import pytest

from geobia.classification import classify
from geobia.classification.fuzzy import FuzzyClassifier, FuzzyRule


@pytest.fixture
def feature_df():
    """Feature DataFrame with known spectral values."""
    return pd.DataFrame(
        {
            "ndvi": [0.7, 0.8, -0.1, 0.0, 0.25, 0.5],
            "brightness_mean": [0.2, 0.15, 0.1, 0.08, 0.35, 0.3],
        },
        index=pd.RangeIndex(1, 7, name="segment_id"),
    )


@pytest.fixture
def rules():
    return {
        "vegetation": [
            FuzzyRule("ndvi", 0.3, 1.0, low_edge=0.2),
        ],
        "water": [
            FuzzyRule("ndvi", -1.0, 0.1, high_edge=0.15),
            FuzzyRule("brightness_mean", 0.0, 0.12, high_edge=0.15),
        ],
    }


def test_fuzzy_rule_full_membership():
    rule = FuzzyRule("x", 0.3, 0.7)
    vals = pd.Series([0.5], index=[1])
    assert rule.evaluate(vals).iloc[0] == 1.0


def test_fuzzy_rule_zero_outside():
    rule = FuzzyRule("x", 0.3, 0.7)
    vals = pd.Series([0.1, 0.9], index=[1, 2])
    result = rule.evaluate(vals)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.0


def test_fuzzy_rule_ramp_up():
    rule = FuzzyRule("x", 0.4, 0.8, low_edge=0.2)
    vals = pd.Series([0.3], index=[1])
    result = rule.evaluate(vals)
    assert 0 < result.iloc[0] < 1  # Should be 0.5


def test_fuzzy_rule_ramp_down():
    rule = FuzzyRule("x", 0.2, 0.6, high_edge=0.8)
    vals = pd.Series([0.7], index=[1])
    result = rule.evaluate(vals)
    assert 0 < result.iloc[0] < 1  # Should be 0.5


def test_fuzzy_classifier_basic(feature_df, rules):
    clf = FuzzyClassifier(rules=rules)
    preds = clf.predict(feature_df)
    assert len(preds) == 6
    # High NDVI segments should be vegetation
    assert preds.iloc[0] == "vegetation"
    assert preds.iloc[1] == "vegetation"
    # Negative NDVI + low brightness should be water
    assert preds.iloc[2] == "water"


def test_fuzzy_classifier_proba(feature_df, rules):
    clf = FuzzyClassifier(rules=rules)
    clf.predict(feature_df)
    proba = clf.predict_proba(feature_df)
    assert "vegetation" in proba.columns
    assert "water" in proba.columns
    assert (proba >= 0).all().all()
    assert (proba <= 1).all().all()


def test_fuzzy_no_rules():
    clf = FuzzyClassifier()
    with pytest.raises(ValueError, match="No rules"):
        clf.predict(pd.DataFrame({"x": [1]}, index=[1]))


def test_fuzzy_missing_feature(feature_df, rules):
    clf = FuzzyClassifier(rules={"test": [FuzzyRule("nonexistent", 0, 1)]})
    with pytest.raises(ValueError, match="not found"):
        clf.predict(feature_df)


def test_fuzzy_unclassified(feature_df):
    """Segments with 0 membership in all classes get 'unclassified'."""
    rules = {
        "extreme_high": [FuzzyRule("ndvi", 10.0, 20.0)],
    }
    clf = FuzzyClassifier(rules=rules)
    preds = clf.predict(feature_df)
    assert (preds == "unclassified").all()


def test_fuzzy_via_convenience(feature_df, rules):
    preds = classify(feature_df, method="fuzzy", rules=rules)
    assert len(preds) == 6


def test_fuzzy_get_params(rules):
    clf = FuzzyClassifier(rules=rules)
    params = clf.get_params()
    assert params["algorithm"] == "fuzzy"
    assert "vegetation" in params["rules"]
