"""Label array utilities for segment rasters."""

from __future__ import annotations

import numpy as np


def count_segments(labels: np.ndarray) -> int:
    """Count unique segment IDs (excluding 0/nodata)."""
    unique = np.unique(labels)
    return int((unique > 0).sum())


def segment_sizes(labels: np.ndarray) -> dict[int, int]:
    """Compute pixel count per segment.

    Args:
        labels: Segment label array (height, width).

    Returns:
        Dict mapping segment_id -> pixel count.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return {int(sid): int(cnt) for sid, cnt in zip(unique, counts) if sid > 0}


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Relabel segments to sequential IDs starting from 1.

    Preserves 0 as nodata. Returns a new array.
    Uses a vectorized lookup table for O(n) performance.
    """
    max_label = int(labels.max())
    if max_label == 0:
        return np.zeros_like(labels)

    # Build a lookup table: old_id -> new_id
    lut = np.zeros(max_label + 1, dtype=labels.dtype)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]
    lut[unique_ids] = np.arange(1, len(unique_ids) + 1, dtype=labels.dtype)

    return lut[labels]


def get_segment_slices(labels: np.ndarray) -> dict[int, tuple[slice, slice]]:
    """Get bounding box slices for each segment.

    Useful for efficiently extracting per-segment pixel regions.

    Args:
        labels: Segment label array (height, width).

    Returns:
        Dict mapping segment_id -> (row_slice, col_slice).
    """
    from scipy.ndimage import find_objects

    max_label = labels.max()
    if max_label == 0:
        return {}

    slices = find_objects(labels)
    result = {}
    for i, s in enumerate(slices):
        if s is not None:
            result[i + 1] = s
    return result
