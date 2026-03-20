"""Fuzzy membership function classification (eCognition-style rule-based)."""

from __future__ import annotations

import pandas as pd

from geobia.classification.base import BaseClassifier


class FuzzyRule:
    """A single fuzzy membership rule for one feature.

    Defines a trapezoidal membership function with four control points:
    (a, b, c, d) where membership is 0 below a, ramps to 1 between a-b,
    stays 1 between b-c, ramps to 0 between c-d, and 0 above d.
    """

    def __init__(
        self,
        feature: str,
        low: float,
        high: float,
        low_edge: float | None = None,
        high_edge: float | None = None,
    ):
        """
        Args:
            feature: Feature column name.
            low: Lower bound of full membership.
            high: Upper bound of full membership.
            low_edge: Start of ramp-up (defaults to low, giving a step).
            high_edge: End of ramp-down (defaults to high, giving a step).
        """
        self.feature = feature
        self.a = low_edge if low_edge is not None else low
        self.b = low
        self.c = high
        self.d = high_edge if high_edge is not None else high

    def evaluate(self, values: pd.Series) -> pd.Series:
        """Compute membership degree for each segment."""
        result = pd.Series(0.0, index=values.index)

        # Ramp up: a -> b
        if self.b > self.a:
            ramp_up = (values - self.a) / (self.b - self.a)
            mask = (values >= self.a) & (values < self.b)
            result[mask] = ramp_up[mask].clip(0, 1)

        # Full membership: b -> c
        result[(values >= self.b) & (values <= self.c)] = 1.0

        # Ramp down: c -> d
        if self.d > self.c:
            ramp_down = 1.0 - (values - self.c) / (self.d - self.c)
            mask = (values > self.c) & (values <= self.d)
            result[mask] = ramp_down[mask].clip(0, 1)

        return result


class FuzzyClassifier(BaseClassifier):
    """Rule-based classifier using fuzzy membership functions.

    Each class is defined by a set of rules. A segment's membership in a
    class is the minimum membership across all rules for that class
    (fuzzy AND). The segment is assigned to the class with the highest
    membership.

    Example:
        rules = {
            "vegetation": [
                FuzzyRule("ndvi", 0.3, 1.0, low_edge=0.2),
                FuzzyRule("brightness_mean", 0.0, 0.4),
            ],
            "water": [
                FuzzyRule("ndvi", -1.0, 0.0, high_edge=0.1),
                FuzzyRule("brightness_mean", 0.0, 0.15, high_edge=0.2),
            ],
        }
        clf = FuzzyClassifier(rules=rules)
        predictions = clf.predict(features)
    """

    def __init__(self, rules: dict[str, list[FuzzyRule]] | None = None):
        self.rules = rules or {}
        self._memberships: pd.DataFrame | None = None

    def fit(self, features: pd.DataFrame, labels: pd.Series | None = None) -> None:
        """No-op for rule-based classifier. Rules are set at construction."""
        pass

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Assign each segment to the class with highest fuzzy membership."""
        if not self.rules:
            raise ValueError("No rules defined. Pass rules to FuzzyClassifier()")

        memberships = {}
        for class_name, rule_list in self.rules.items():
            # Fuzzy AND: minimum membership across all rules
            class_membership = pd.Series(1.0, index=features.index)
            for rule in rule_list:
                if rule.feature not in features.columns:
                    raise ValueError(
                        f"Feature {rule.feature!r} not found. Available: {list(features.columns)}"
                    )
                rule_membership = rule.evaluate(features[rule.feature])
                class_membership = class_membership.clip(upper=rule_membership)
            memberships[class_name] = class_membership

        membership_df = pd.DataFrame(memberships)
        self._memberships = membership_df

        # Assign to class with highest membership; "unclassified" if all are 0
        predictions = membership_df.idxmax(axis=1)
        max_vals = membership_df.max(axis=1)
        predictions[max_vals == 0] = "unclassified"

        return predictions.rename("class_label")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return fuzzy membership degrees for each class."""
        if self._memberships is not None:
            return self._memberships
        # Compute if not cached
        self.predict(features)
        return self._memberships

    def get_params(self) -> dict:
        rule_dict = {}
        for cls_name, rule_list in self.rules.items():
            rule_dict[cls_name] = [
                {"feature": r.feature, "a": r.a, "b": r.b, "c": r.c, "d": r.d} for r in rule_list
            ]
        return {"algorithm": "fuzzy", "rules": rule_dict}
