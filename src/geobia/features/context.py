"""Contextual feature extraction based on segment neighbor relationships."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.ndimage import labeled_comprehension, mean as nd_mean

from geobia.features.base import BaseExtractor


def _find_neighbors(labels: np.ndarray) -> dict[int, set[int]]:
    """Find adjacent segment pairs using vectorized numpy operations.

    Returns:
        Dict mapping segment_id -> set of neighbor segment_ids.
    """
    neighbors: dict[int, set[int]] = {}

    # Horizontal adjacency — vectorized extraction of border pairs
    diff_h = labels[:, :-1] != labels[:, 1:]
    rows_h, cols_h = np.where(diff_h)
    if len(rows_h) > 0:
        a_vals = labels[rows_h, cols_h]
        b_vals = labels[rows_h, cols_h + 1]
        # Filter out nodata (0)
        valid = (a_vals > 0) & (b_vals > 0)
        a_vals = a_vals[valid]
        b_vals = b_vals[valid]
        # Build unique pairs
        pairs = np.column_stack([a_vals, b_vals])
        # Also add reverse direction
        pairs_rev = np.column_stack([b_vals, a_vals])
        pairs = np.vstack([pairs, pairs_rev])

    # Vertical adjacency
    diff_v = labels[:-1, :] != labels[1:, :]
    rows_v, cols_v = np.where(diff_v)
    if len(rows_v) > 0:
        a_vals_v = labels[rows_v, cols_v]
        b_vals_v = labels[rows_v + 1, cols_v]
        valid_v = (a_vals_v > 0) & (b_vals_v > 0)
        a_vals_v = a_vals_v[valid_v]
        b_vals_v = b_vals_v[valid_v]
        pairs_v = np.column_stack([a_vals_v, b_vals_v])
        pairs_v_rev = np.column_stack([b_vals_v, a_vals_v])
        v_pairs = np.vstack([pairs_v, pairs_v_rev])
        if len(rows_h) > 0:
            pairs = np.vstack([pairs, v_pairs])
        else:
            pairs = v_pairs
    elif len(rows_h) == 0:
        return neighbors

    # Deduplicate and build dict
    unique_pairs = np.unique(pairs, axis=0)
    for a, b in unique_pairs:
        neighbors.setdefault(int(a), set()).add(int(b))

    return neighbors


class ContextExtractor(BaseExtractor):
    """Extract contextual features based on neighbor relationships.

    Computes features describing each segment's relationship to its
    neighbors: number of neighbors, mean/std of neighbor spectral
    values, and border contrast.
    """

    def extract(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> pd.DataFrame:
        segment_ids = np.unique(labels)
        segment_ids = segment_ids[segment_ids > 0]

        if len(segment_ids) == 0:
            return pd.DataFrame()

        n_bands = image.shape[0] if image.ndim == 3 else 1
        neighbors = _find_neighbors(labels)

        # Pre-compute per-segment mean brightness per band using scipy.ndimage
        seg_means: dict[int, np.ndarray] = {}
        if image.ndim == 3:
            band_means = []
            for b in range(n_bands):
                means = nd_mean(image[b].astype(np.float64), labels, segment_ids)
                band_means.append(np.asarray(means))
            # band_means[b][i] = mean of band b for segment_ids[i]
            for i, sid in enumerate(segment_ids):
                seg_means[int(sid)] = np.array([band_means[b][i] for b in range(n_bands)])
        else:
            means = nd_mean(image.astype(np.float64), labels, segment_ids)
            means = np.asarray(means)
            for i, sid in enumerate(segment_ids):
                seg_means[int(sid)] = np.array([means[i]])

        records = []
        for sid in segment_ids:
            sid = int(sid)
            nbrs = neighbors.get(sid, set())
            n_neighbors = len(nbrs)

            if n_neighbors > 0:
                nbr_means = np.array([seg_means.get(n, seg_means[sid]) for n in nbrs])
                # Mean of neighbor means (averaged across bands)
                nbr_brightness_mean = nbr_means.mean()
                nbr_brightness_std = nbr_means.std()

                # Border contrast: mean absolute difference to neighbors
                own = seg_means[sid]
                diffs = np.array([np.abs(own - seg_means.get(n, own)).mean() for n in nbrs])
                border_contrast = diffs.mean()
            else:
                nbr_brightness_mean = 0.0
                nbr_brightness_std = 0.0
                border_contrast = 0.0

            records.append(
                {
                    "segment_id": sid,
                    "n_neighbors": n_neighbors,
                    "nbr_brightness_mean": nbr_brightness_mean,
                    "nbr_brightness_std": nbr_brightness_std,
                    "border_contrast": border_contrast,
                }
            )

        df = pd.DataFrame(records).set_index("segment_id")
        return df

    @classmethod
    def feature_names(cls, **kwargs) -> list[str]:
        return [
            "n_neighbors",
            "nbr_brightness_mean",
            "nbr_brightness_std",
            "border_contrast",
        ]
