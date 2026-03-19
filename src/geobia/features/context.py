"""Contextual feature extraction based on segment neighbor relationships."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.ndimage import labeled_comprehension, mean as nd_mean

from geobia.features.base import BaseExtractor


def _find_neighbors(labels: np.ndarray) -> dict[int, set[int]]:
    """Find adjacent segment pairs by checking horizontal and vertical borders.

    Returns:
        Dict mapping segment_id -> set of neighbor segment_ids.
    """
    neighbors: dict[int, set[int]] = {}
    h, w = labels.shape

    # Horizontal adjacency
    diff_h = labels[:, :-1] != labels[:, 1:]
    rows, cols = np.where(diff_h)
    for r, c in zip(rows, cols):
        a, b = int(labels[r, c]), int(labels[r, c + 1])
        if a > 0 and b > 0:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)

    # Vertical adjacency
    diff_v = labels[:-1, :] != labels[1:, :]
    rows, cols = np.where(diff_v)
    for r, c in zip(rows, cols):
        a, b = int(labels[r, c]), int(labels[r + 1, c])
        if a > 0 and b > 0:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)

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

        # Pre-compute per-segment mean brightness per band
        seg_means: dict[int, np.ndarray] = {}
        for sid in segment_ids:
            mask = labels == sid
            if image.ndim == 3:
                seg_means[int(sid)] = np.array(
                    [image[b][mask].mean() for b in range(n_bands)]
                )
            else:
                seg_means[int(sid)] = np.array([image[mask].mean()])

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

            records.append({
                "segment_id": sid,
                "n_neighbors": n_neighbors,
                "nbr_brightness_mean": nbr_brightness_mean,
                "nbr_brightness_std": nbr_brightness_std,
                "border_contrast": border_contrast,
            })

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
