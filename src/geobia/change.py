"""Change detection between two time periods.

Compares per-segment feature vectors from two dates to identify changed
segments using difference magnitude thresholding or classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def feature_difference(
    features_t1: pd.DataFrame,
    features_t2: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-feature difference between two time periods.

    Args:
        features_t1: Feature DataFrame for time 1 (segment_id index).
        features_t2: Feature DataFrame for time 2 (segment_id index).

    Returns:
        DataFrame of (t2 - t1) differences, aligned on common segments
        and numeric features.
    """
    common_idx = features_t1.index.intersection(features_t2.index)
    common_cols = features_t1.columns.intersection(features_t2.columns)
    # Only numeric columns
    numeric_cols = [
        c for c in common_cols
        if pd.api.types.is_numeric_dtype(features_t1[c])
    ]

    diff = features_t2.loc[common_idx, numeric_cols] - features_t1.loc[common_idx, numeric_cols]
    return diff


def change_magnitude(
    features_t1: pd.DataFrame,
    features_t2: pd.DataFrame,
    normalize: bool = True,
) -> pd.Series:
    """Compute per-segment change magnitude (Euclidean distance in feature space).

    Args:
        features_t1: Feature DataFrame for time 1.
        features_t2: Feature DataFrame for time 2.
        normalize: If True, z-score normalize features before computing distance.

    Returns:
        Series of change magnitude per segment.
    """
    diff = feature_difference(features_t1, features_t2)

    if normalize and len(diff) > 1:
        # Normalize by the pooled standard deviation from both periods
        common_idx = diff.index
        cols = diff.columns
        pooled = pd.concat([
            features_t1.loc[common_idx, cols],
            features_t2.loc[common_idx, cols],
        ])
        stds = pooled.std()
        stds = stds.replace(0, 1)  # avoid division by zero
        diff = diff / stds

    magnitude = np.sqrt((diff ** 2).sum(axis=1))
    return magnitude.rename("change_magnitude")


def detect_changes(
    features_t1: pd.DataFrame,
    features_t2: pd.DataFrame,
    threshold: float | str = "otsu",
    normalize: bool = True,
) -> pd.Series:
    """Classify segments as changed/unchanged based on feature difference magnitude.

    Args:
        features_t1: Feature DataFrame for time 1.
        features_t2: Feature DataFrame for time 2.
        threshold: Change magnitude threshold. Use a float for a fixed
            threshold or "otsu" for automatic Otsu thresholding.
        normalize: Whether to normalize features before computing distance.

    Returns:
        Boolean Series (True = changed) indexed by segment_id.
    """
    mag = change_magnitude(features_t1, features_t2, normalize=normalize)

    if threshold == "otsu":
        from skimage.filters import threshold_otsu
        if mag.nunique() < 2:
            thresh_val = float(mag.max()) + 1
        else:
            thresh_val = threshold_otsu(mag.values)
    else:
        thresh_val = float(threshold)

    changed = mag > thresh_val
    return changed.rename("changed")


def change_summary(
    changed: pd.Series,
    features_t1: pd.DataFrame | None = None,
    features_t2: pd.DataFrame | None = None,
) -> dict:
    """Summarize change detection results.

    Args:
        changed: Boolean Series from detect_changes().
        features_t1: Optional feature DataFrame for time 1.
        features_t2: Optional feature DataFrame for time 2.

    Returns:
        Dict with summary statistics.
    """
    total = len(changed)
    n_changed = int(changed.sum())
    result = {
        "total_segments": total,
        "changed": n_changed,
        "unchanged": total - n_changed,
        "pct_changed": round(n_changed / total * 100, 1) if total > 0 else 0.0,
    }

    if features_t1 is not None and features_t2 is not None:
        diff = feature_difference(features_t1, features_t2)
        changed_idx = changed[changed].index
        if len(changed_idx) > 0:
            result["mean_diff_changed"] = diff.loc[changed_idx].mean().to_dict()

    return result
