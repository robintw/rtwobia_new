"""Segmentation module with pluggable algorithms and tiled processing."""

from __future__ import annotations

import numpy as np

from geobia.segmentation.base import BaseSegmenter
from geobia.segmentation.slic import SLICSegmenter
from geobia.segmentation.felzenszwalb import FelzenszwalbSegmenter
from geobia.segmentation.shepherd import ShepherdSegmenter
from geobia.segmentation.watershed import WatershedSegmenter

_REGISTRY: dict[str, type[BaseSegmenter]] = {
    "slic": SLICSegmenter,
    "felzenszwalb": FelzenszwalbSegmenter,
    "shepherd": ShepherdSegmenter,
    "watershed": WatershedSegmenter,
}

# Register SAM if segment-geospatial is installed
try:
    from geobia.segmentation.sam import SAMSegmenter
    _REGISTRY["sam"] = SAMSegmenter
except ImportError:
    pass


def create(method: str, **params) -> BaseSegmenter:
    """Factory function to create a segmenter by name.

    Args:
        method: Algorithm name ('slic', 'felzenszwalb').
        **params: Algorithm-specific parameters.

    Returns:
        Configured segmenter instance.
    """
    method = method.lower()
    if method not in _REGISTRY:
        raise ValueError(f"Unknown segmentation method: {method!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[method](**params)


def segment(
    image: np.ndarray,
    method: str = "slic",
    nodata_mask: np.ndarray | None = None,
    **params,
) -> np.ndarray:
    """Convenience function to segment an image in one call.

    Args:
        image: (bands, height, width) array.
        method: Algorithm name.
        nodata_mask: Boolean mask, True where invalid.
        **params: Algorithm-specific parameters.

    Returns:
        (height, width) integer label array.
    """
    segmenter = create(method, **params)
    return segmenter.segment(image, nodata_mask=nodata_mask)


def segment_tiled(
    path: str,
    method: str = "slic",
    tile_size: int = 2048,
    overlap: int = 128,
    output_path: str | None = None,
    **params,
) -> np.ndarray:
    """Segment a large raster using tiled processing.

    Reads tiles with overlap, segments each, then stitches results.
    For the overlap zone, uses the labels from the tile that contains
    the pixel in its non-overlap (core) region.

    Args:
        path: Input raster file path.
        method: Segmentation algorithm name.
        tile_size: Size of each tile in pixels.
        overlap: Overlap between tiles in pixels.
        output_path: If provided, writes result to this GeoTIFF.
        **params: Algorithm-specific parameters.

    Returns:
        Full (height, width) label array.
    """
    from geobia.io.raster import read_raster_windows, write_raster

    import rasterio

    with rasterio.open(path) as ds:
        full_height = ds.height
        full_width = ds.width
        meta = {
            "crs": ds.crs,
            "transform": ds.transform,
            "nodata": 0,
            "width": full_width,
            "height": full_height,
        }

    result = np.zeros((full_height, full_width), dtype=np.int32)
    max_label = 0
    segmenter = create(method, **params)
    step = tile_size - overlap

    for tile, tile_meta, window in read_raster_windows(path, tile_size=tile_size, overlap=overlap):
        labels = segmenter.segment(tile)

        # Offset labels to be globally unique
        tile_mask = labels > 0
        labels[tile_mask] += max_label

        # Determine the core region (non-overlap) for this tile
        row_off = int(window.row_off)
        col_off = int(window.col_off)
        tile_h = int(window.height)
        tile_w = int(window.width)

        # Core region bounds within the tile
        core_r_start = overlap // 2 if row_off > 0 else 0
        core_c_start = overlap // 2 if col_off > 0 else 0
        core_r_end = tile_h - overlap // 2 if (row_off + tile_h) < full_height else tile_h
        core_c_end = tile_w - overlap // 2 if (col_off + tile_w) < full_width else tile_w

        # Write core region to result
        out_r_start = row_off + core_r_start
        out_c_start = col_off + core_c_start
        out_r_end = row_off + core_r_end
        out_c_end = col_off + core_c_end

        result[out_r_start:out_r_end, out_c_start:out_c_end] = (
            labels[core_r_start:core_r_end, core_c_start:core_c_end]
        )

        if tile_mask.any():
            max_label = int(labels.max())

    if output_path:
        write_raster(output_path, result, meta, dtype="int32")

    return result


def list_methods() -> list[str]:
    """Return available segmentation method names."""
    return list(_REGISTRY.keys())


__all__ = [
    "BaseSegmenter",
    "SLICSegmenter",
    "FelzenszwalbSegmenter",
    "ShepherdSegmenter",
    "WatershedSegmenter",
    "create",
    "segment",
    "segment_tiled",
    "list_methods",
]
