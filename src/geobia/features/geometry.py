"""Geometric feature extraction using skimage.measure.regionprops."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from geobia.features.base import BaseExtractor


class GeometryExtractor(BaseExtractor):
    """Extract geometric/shape features for each segment.

    Uses skimage.measure.regionprops for fast raster-based computation.
    Optionally computes map-unit features if pixel_size is provided.
    """

    def __init__(self, pixel_size: float | None = None):
        """
        Args:
            pixel_size: Pixel size in map units (e.g., meters). If provided,
                area and perimeter are computed in map units. Otherwise in pixels.
        """
        self.pixel_size = pixel_size

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

        props = regionprops_table(
            labels,
            properties=[
                "label",
                "area",
                "perimeter",
                "eccentricity",
                "solidity",
                "major_axis_length",
                "minor_axis_length",
                "orientation",
                "centroid",
                "bbox",
            ],
        )

        df = pd.DataFrame(props)
        df = df.rename(columns={"label": "segment_id"})
        df = df.set_index("segment_id")

        # Area in pixels
        df = df.rename(columns={"area": "area_px"})

        # Area and perimeter in map units
        px = self.pixel_size or 1.0
        df["area"] = df["area_px"] * (px ** 2)
        df["perimeter"] = df["perimeter"] * px

        # Compactness: 4 * pi * area / perimeter^2 (1.0 = circle)
        df["compactness"] = np.where(
            df["perimeter"] > 0,
            4 * math.pi * df["area"] / (df["perimeter"] ** 2),
            0.0,
        )

        # Elongation: major_axis / minor_axis
        df["elongation"] = np.where(
            df["minor_axis_length"] > 0,
            df["major_axis_length"] / df["minor_axis_length"],
            0.0,
        )

        # Rectangularity: area / bounding_box_area
        bbox_h = df["bbox-2"] - df["bbox-0"]
        bbox_w = df["bbox-3"] - df["bbox-1"]
        bbox_area = bbox_h * bbox_w
        df["rectangularity"] = np.where(
            bbox_area > 0,
            df["area_px"] / bbox_area,
            0.0,
        )

        # Rename centroid columns
        df = df.rename(columns={
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
        })

        # Drop bbox columns (intermediate)
        df = df.drop(columns=["bbox-0", "bbox-1", "bbox-2", "bbox-3"], errors="ignore")

        return df

    @classmethod
    def feature_names(cls, **kwargs) -> list[str]:
        return [
            "area_px",
            "area",
            "perimeter",
            "compactness",
            "eccentricity",
            "solidity",
            "elongation",
            "rectangularity",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "centroid_x",
            "centroid_y",
        ]
