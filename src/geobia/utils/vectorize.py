"""Vectorize raster segment labels to polygon geometries."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape


def vectorize_labels(
    labels: np.ndarray,
    transform: Any,
    crs: Any = None,
) -> gpd.GeoDataFrame:
    """Convert a segment label raster to polygon geometries.

    Uses rasterio.features.shapes for efficient vectorization.

    Args:
        labels: Integer label array (height, width). 0 = nodata.
        transform: Affine transform from the source raster.
        crs: Coordinate reference system.

    Returns:
        GeoDataFrame with 'segment_id' and 'geometry' columns.
    """
    mask = labels > 0
    label_data = labels.astype(np.int32)

    geometries = []
    segment_ids = []

    for geom_dict, value in shapes(label_data, mask=mask, transform=transform):
        geometries.append(shape(geom_dict))
        segment_ids.append(int(value))

    gdf = gpd.GeoDataFrame(
        {"segment_id": segment_ids},
        geometry=geometries,
        crs=crs,
    )

    # Merge multipolygons for the same segment ID (shapes may split segments)
    gdf = gdf.dissolve(by="segment_id", as_index=False)
    return gdf
