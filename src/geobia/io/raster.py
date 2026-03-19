"""Raster I/O using rasterio. Supports windowed reading for large images."""

from __future__ import annotations

import warnings
from collections.abc import Generator
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window


def read_raster(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an entire raster file into memory.

    Args:
        path: Path to the raster file.

    Returns:
        Tuple of (image array with shape (bands, height, width), metadata dict).
        Metadata includes: crs, transform, nodata, dtype, width, height, count.
    """
    with rasterio.open(path) as ds:
        image = ds.read()
        meta = _extract_meta(ds)
    return image, meta


def read_raster_windows(
    path: str,
    tile_size: int = 1024,
    overlap: int = 0,
) -> Generator[tuple[np.ndarray, dict[str, Any], Window], None, None]:
    """Read a raster in tiles using rasterio windowed I/O.

    Yields tiles of the specified size with optional overlap. Each tile
    includes the window object so results can be stitched back together.

    Args:
        path: Path to the raster file.
        tile_size: Size of each square tile in pixels.
        overlap: Pixel overlap between adjacent tiles.

    Yields:
        Tuple of (tile array (bands, tile_h, tile_w), metadata dict, Window).
    """
    with rasterio.open(path) as ds:
        meta = _extract_meta(ds)
        step = tile_size - overlap

        for row_off in range(0, ds.height, step):
            for col_off in range(0, ds.width, step):
                win_height = min(tile_size, ds.height - row_off)
                win_width = min(tile_size, ds.width - col_off)
                window = Window(col_off, row_off, win_width, win_height)

                tile = ds.read(window=window)
                tile_meta = meta.copy()
                tile_meta["transform"] = ds.window_transform(window)
                tile_meta["width"] = win_width
                tile_meta["height"] = win_height
                yield tile, tile_meta, window


def write_raster(
    path: str,
    data: np.ndarray,
    meta: dict[str, Any],
    *,
    dtype: str | None = None,
) -> None:
    """Write a raster array to disk.

    Args:
        path: Output file path (GeoTIFF).
        data: Array with shape (bands, height, width) or (height, width).
        meta: Metadata dict with at least 'crs' and 'transform'.
        dtype: Override output dtype. If None, uses data.dtype.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    bands, height, width = data.shape
    out_dtype = dtype or str(data.dtype)

    if meta.get("crs") is None:
        warnings.warn(
            f"Writing {path} without a CRS — the output GeoTIFF will "
            f"have no coordinate reference system.",
            stacklevel=2,
        )
    if meta.get("transform") is None:
        warnings.warn(
            f"Writing {path} without a geotransform — the output GeoTIFF "
            f"will have no georeferencing.",
            stacklevel=2,
        )

    profile = {
        "driver": "GTiff",
        "dtype": out_dtype,
        "width": width,
        "height": height,
        "count": bands,
        "crs": meta.get("crs"),
        "transform": meta.get("transform"),
        "nodata": meta.get("nodata"),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }

    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data)


def _extract_meta(ds: rasterio.DatasetReader) -> dict[str, Any]:
    """Extract metadata from an open rasterio dataset."""
    return {
        "crs": ds.crs,
        "transform": ds.transform,
        "nodata": ds.nodata,
        "dtype": str(ds.dtypes[0]),
        "width": ds.width,
        "height": ds.height,
        "count": ds.count,
        "bounds": ds.bounds,
    }
