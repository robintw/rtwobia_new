"""Vector I/O using geopandas. Read/write GeoPackage and other vector formats."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


def read_vector(path: str, **kwargs) -> gpd.GeoDataFrame:
    """Read a vector file into a GeoDataFrame.

    Args:
        path: Path to vector file (GeoPackage, Shapefile, etc.).
        **kwargs: Additional arguments passed to geopandas.read_file.

    Returns:
        GeoDataFrame with geometry and attributes.
    """
    return gpd.read_file(path, **kwargs)


def write_vector(
    path: str,
    labels: np.ndarray,
    attributes: pd.DataFrame | pd.Series | None = None,
    meta: dict[str, Any] | None = None,
    *,
    polygons: gpd.GeoDataFrame | None = None,
) -> None:
    """Write classified segments to a vector file.

    If polygons are provided, uses them directly. Otherwise vectorizes labels.

    Args:
        path: Output file path (GeoPackage recommended).
        labels: Segment label array (height, width) — used if polygons is None.
        attributes: DataFrame or Series with segment_id index containing attributes/classes.
        meta: Metadata dict with 'crs' and 'transform' (needed for vectorization).
        polygons: Pre-computed GeoDataFrame of segment polygons.
    """
    from geobia.utils.vectorize import vectorize_labels

    if polygons is None:
        if meta is None:
            raise ValueError("meta dict with 'crs' and 'transform' required for vectorization")
        polygons = vectorize_labels(labels, meta["transform"], meta.get("crs"))

    if attributes is not None:
        if isinstance(attributes, pd.Series):
            attributes = attributes.to_frame("class_label")
        polygons = polygons.merge(attributes, left_on="segment_id", right_index=True, how="left")

    driver = "GPKG" if path.endswith(".gpkg") else None
    polygons.to_file(path, driver=driver)


def read_training_samples(
    path: str,
    labels: np.ndarray,
    meta: dict[str, Any],
    class_column: str = "class",
) -> pd.Series:
    """Read training samples and map them to segment IDs.

    Supports polygon or point training data. For polygons, segments whose
    centroids fall within a training polygon are labeled. For points, the
    segment containing each point is labeled.

    Args:
        path: Path to training sample file (GeoPackage, Shapefile).
        labels: Segment label array (height, width).
        meta: Metadata with 'crs' and 'transform'.
        class_column: Name of the class label column.

    Returns:
        Series mapping segment_id -> class_label.
    """
    from rasterio.transform import rowcol

    samples = gpd.read_file(path)

    if meta.get("crs") and samples.crs and samples.crs != meta["crs"]:
        samples = samples.to_crs(meta["crs"])

    transform = meta["transform"]
    height, width = labels.shape
    result = {}

    for _, row in samples.iterrows():
        geom = row.geometry
        class_label = row[class_column]

        if geom.geom_type == "Point":
            r, c = rowcol(transform, geom.x, geom.y)
            if 0 <= r < height and 0 <= c < width:
                seg_id = int(labels[r, c])
                if seg_id > 0:
                    result[seg_id] = class_label
        else:
            # For polygons, find all segment IDs whose pixels overlap
            minx, miny, maxx, maxy = geom.bounds
            r_min, c_min = rowcol(transform, minx, maxy)
            r_max, c_max = rowcol(transform, maxx, miny)
            r_min = max(0, r_min)
            c_min = max(0, c_min)
            r_max = min(height - 1, r_max)
            c_max = min(width - 1, c_max)

            sub_labels = labels[r_min : r_max + 1, c_min : c_max + 1]
            unique_ids = np.unique(sub_labels)
            for seg_id in unique_ids:
                if seg_id > 0:
                    result[int(seg_id)] = class_label

    return pd.Series(result, name="class_label")
