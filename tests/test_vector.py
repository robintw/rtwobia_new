"""Tests for vector I/O module."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from geobia.io.vector import read_training_samples, read_vector, write_vector


@pytest.fixture
def sample_gpkg(tmp_path, synthetic_labels, synthetic_meta):
    """Write a simple GeoPackage with point training samples."""
    transform = synthetic_meta["transform"]
    crs = synthetic_meta["crs"]

    # Create points in each quadrant
    points = []
    classes = []
    for seg_id, (row, col) in enumerate([(25, 25), (25, 75), (75, 25), (75, 75)], start=1):
        x, y = transform * (col, row)
        points.append(Point(x, y))
        classes.append(f"class_{seg_id}")

    gdf = gpd.GeoDataFrame({"class": classes, "geometry": points}, crs=crs)
    path = str(tmp_path / "samples.gpkg")
    gdf.to_file(path, driver="GPKG")
    return path


@pytest.fixture
def polygon_gpkg(tmp_path, synthetic_meta):
    """Write a GeoPackage with polygon training samples."""
    transform = synthetic_meta["transform"]
    crs = synthetic_meta["crs"]

    # Create polygon covering top-left quadrant
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (50, 50)
    poly = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    gdf = gpd.GeoDataFrame({"class": ["vegetation"], "geometry": [poly]}, crs=crs)
    path = str(tmp_path / "poly_samples.gpkg")
    gdf.to_file(path, driver="GPKG")
    return path


class TestReadVector:
    def test_reads_gpkg(self, sample_gpkg):
        gdf = read_vector(sample_gpkg)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 4
        assert "class" in gdf.columns

    def test_reads_with_kwargs(self, sample_gpkg):
        gdf = read_vector(sample_gpkg, rows=2)
        assert len(gdf) == 2


class TestWriteVector:
    def test_writes_gpkg(self, tmp_path, synthetic_labels, synthetic_meta):
        path = str(tmp_path / "output.gpkg")
        write_vector(path, synthetic_labels, meta=synthetic_meta)

        gdf = gpd.read_file(path)
        assert len(gdf) > 0
        assert "segment_id" in gdf.columns

    def test_writes_with_attributes(self, tmp_path, synthetic_labels, synthetic_meta):
        path = str(tmp_path / "output.gpkg")
        attrs = pd.Series(
            ["veg", "urban", "water", "soil"],
            index=[1, 2, 3, 4],
            name="class_label",
        )
        write_vector(path, synthetic_labels, attributes=attrs, meta=synthetic_meta)

        gdf = gpd.read_file(path)
        assert "class_label" in gdf.columns

    def test_writes_with_dataframe_attributes(self, tmp_path, synthetic_labels, synthetic_meta):
        path = str(tmp_path / "output.gpkg")
        attrs = pd.DataFrame(
            {"class_label": ["veg", "urban", "water", "soil"], "score": [0.9, 0.8, 0.95, 0.7]},
            index=[1, 2, 3, 4],
        )
        write_vector(path, synthetic_labels, attributes=attrs, meta=synthetic_meta)

        gdf = gpd.read_file(path)
        assert "class_label" in gdf.columns
        assert "score" in gdf.columns

    def test_requires_meta_without_polygons(self, tmp_path, synthetic_labels):
        path = str(tmp_path / "output.gpkg")
        with pytest.raises(ValueError, match="meta dict"):
            write_vector(path, synthetic_labels)

    def test_writes_shapefile(self, tmp_path, synthetic_labels, synthetic_meta):
        path = str(tmp_path / "output.shp")
        write_vector(path, synthetic_labels, meta=synthetic_meta)

        gdf = gpd.read_file(path)
        assert len(gdf) > 0


class TestReadTrainingSamples:
    def test_point_samples(self, sample_gpkg, synthetic_labels, synthetic_meta):
        result = read_training_samples(sample_gpkg, synthetic_labels, synthetic_meta)
        assert isinstance(result, pd.Series)
        assert len(result) == 4
        assert result.name == "class_label"

    def test_polygon_samples(self, polygon_gpkg, synthetic_labels, synthetic_meta):
        result = read_training_samples(polygon_gpkg, synthetic_labels, synthetic_meta)
        assert isinstance(result, pd.Series)
        assert len(result) >= 1
        assert "vegetation" in result.values

    def test_out_of_bounds_points_skipped(self, tmp_path, synthetic_labels, synthetic_meta):
        """Points outside raster bounds should be silently skipped."""
        crs = synthetic_meta["crs"]
        # Create a point far outside the raster bounds
        gdf = gpd.GeoDataFrame(
            {"class": ["outside"]},
            geometry=[Point(0, 0)],
            crs=crs,
        )
        path = str(tmp_path / "oob.gpkg")
        gdf.to_file(path, driver="GPKG")

        result = read_training_samples(path, synthetic_labels, synthetic_meta)
        assert len(result) == 0

    def test_custom_class_column(self, tmp_path, synthetic_meta, synthetic_labels):
        transform = synthetic_meta["transform"]
        crs = synthetic_meta["crs"]
        x, y = transform * (25, 25)
        gdf = gpd.GeoDataFrame(
            {"landcover": ["forest"]},
            geometry=[Point(x, y)],
            crs=crs,
        )
        path = str(tmp_path / "custom.gpkg")
        gdf.to_file(path, driver="GPKG")

        result = read_training_samples(
            path, synthetic_labels, synthetic_meta, class_column="landcover"
        )
        assert len(result) == 1
        assert result.iloc[0] == "forest"
