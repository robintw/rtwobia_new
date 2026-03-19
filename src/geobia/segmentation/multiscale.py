"""Multi-scale / hierarchical segmentation.

Produces segmentations at multiple scales and builds a hierarchy where
fine-scale segments nest within coarse-scale segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from geobia.segmentation.base import BaseSegmenter


@dataclass
class SegmentationLevel:
    """One level of a multi-scale segmentation."""

    scale: int | float
    labels: np.ndarray
    n_segments: int
    params: dict = field(default_factory=dict)


@dataclass
class HierarchicalSegmentation:
    """Multi-scale segmentation hierarchy.

    Levels are ordered from finest (most segments) to coarsest (fewest).
    """

    levels: list[SegmentationLevel]

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def finest(self) -> SegmentationLevel:
        return self.levels[0]

    @property
    def coarsest(self) -> SegmentationLevel:
        return self.levels[-1]

    def parent_map(self, fine_idx: int = 0, coarse_idx: int = 1) -> dict[int, int]:
        """Map fine segment IDs to their parent coarse segment IDs.

        A fine segment's parent is the coarse segment that covers the
        majority of its pixels.

        Args:
            fine_idx: Index of the fine level.
            coarse_idx: Index of the coarse level.

        Returns:
            Dict mapping fine_segment_id -> coarse_segment_id.
        """
        fine = self.levels[fine_idx].labels
        coarse = self.levels[coarse_idx].labels

        fine_ids = np.unique(fine)
        fine_ids = fine_ids[fine_ids > 0]

        mapping = {}
        for fid in fine_ids:
            mask = fine == fid
            coarse_vals = coarse[mask]
            coarse_vals = coarse_vals[coarse_vals > 0]
            if len(coarse_vals) > 0:
                # Majority vote
                vals, counts = np.unique(coarse_vals, return_counts=True)
                mapping[int(fid)] = int(vals[counts.argmax()])

        return mapping

    def cross_scale_features(self, fine_idx: int = 0, coarse_idx: int = 1) -> pd.DataFrame:
        """Compute cross-scale features for fine segments.

        Features: parent segment ID, ratio of fine area to parent area,
        number of siblings (fine segments sharing the same parent).

        Returns:
            DataFrame indexed by fine segment_id.
        """
        parent_map = self.parent_map(fine_idx, coarse_idx)

        fine = self.levels[fine_idx].labels
        coarse = self.levels[coarse_idx].labels

        # Compute sizes
        fine_ids, fine_counts = np.unique(fine, return_counts=True)
        fine_sizes = dict(zip(fine_ids.astype(int), fine_counts.astype(int)))

        coarse_ids, coarse_counts = np.unique(coarse, return_counts=True)
        coarse_sizes = dict(zip(coarse_ids.astype(int), coarse_counts.astype(int)))

        # Count siblings per parent
        from collections import Counter

        parent_child_counts = Counter(parent_map.values())

        records = []
        for fid, pid in parent_map.items():
            f_size = fine_sizes.get(fid, 1)
            p_size = coarse_sizes.get(pid, 1)
            records.append(
                {
                    "segment_id": fid,
                    "parent_id": pid,
                    "area_ratio": f_size / p_size,
                    "n_siblings": parent_child_counts.get(pid, 1),
                }
            )

        return pd.DataFrame(records).set_index("segment_id")


def segment_multiscale(
    image: np.ndarray,
    method: str = "slic",
    scales: list[dict] | None = None,
    nodata_mask: np.ndarray | None = None,
) -> HierarchicalSegmentation:
    """Segment an image at multiple scales.

    Args:
        image: (bands, height, width) array.
        method: Segmentation algorithm name.
        scales: List of parameter dicts for each scale, ordered from
            finest to coarsest. If None, uses default 3-level scales.
        nodata_mask: Boolean mask, True where invalid.

    Returns:
        HierarchicalSegmentation with ordered levels.
    """
    from geobia.segmentation import create

    if scales is None:
        scales = _default_scales(method)

    levels = []
    for scale_params in scales:
        segmenter = create(method, **scale_params)
        labels = segmenter.segment(image, nodata_mask=nodata_mask)
        n_segments = int(labels.max())

        # Use a representative "scale" value
        scale_val = (
            scale_params.get("n_segments")
            or scale_params.get("scale")
            or scale_params.get("markers")
            or n_segments
        )

        levels.append(
            SegmentationLevel(
                scale=scale_val,
                labels=labels,
                n_segments=n_segments,
                params=scale_params,
            )
        )

    # Sort finest (most segments) to coarsest
    levels.sort(key=lambda lv: -lv.n_segments)

    return HierarchicalSegmentation(levels=levels)


def _default_scales(method: str) -> list[dict]:
    """Return default multi-scale parameters for a given method."""
    if method == "slic":
        return [
            {"n_segments": 1000, "compactness": 10},
            {"n_segments": 200, "compactness": 10},
            {"n_segments": 50, "compactness": 10},
        ]
    elif method == "felzenszwalb":
        return [
            {"scale": 50, "min_size": 20},
            {"scale": 150, "min_size": 50},
            {"scale": 400, "min_size": 100},
        ]
    elif method == "watershed":
        return [
            {"markers": 1000, "min_distance": 5},
            {"markers": 200, "min_distance": 10},
            {"markers": 50, "min_distance": 20},
        ]
    else:
        # Generic fallback for methods with n_segments
        return [
            {"n_segments": 1000},
            {"n_segments": 200},
            {"n_segments": 50},
        ]
