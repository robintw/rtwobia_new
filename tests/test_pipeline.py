"""Tests for the pipeline engine."""

import json

import pandas as pd
import pytest

from geobia.pipeline import Pipeline


class TestPipeline:
    def test_basic_pipeline(self, synthetic_image, synthetic_meta):
        pipeline = Pipeline(
            [
                ("segment", "slic", {"n_segments": 20}),
                ("extract", ["spectral", "geometry"], {}),
                ("classify", "kmeans", {"n_clusters": 3}),
            ]
        )
        result = pipeline.run(image=synthetic_image, meta=synthetic_meta)
        assert result.labels is not None
        assert result.features is not None
        assert result.predictions is not None
        assert result.predictions.nunique() == 3

    def test_pipeline_from_file(self, synthetic_image, synthetic_meta, raster_file):
        pipeline = Pipeline(
            [
                ("segment", "slic", {"n_segments": 20}),
                ("extract", ["spectral"], {}),
                ("classify", "kmeans", {"n_clusters": 2}),
            ]
        )
        result = pipeline.run(input_path=raster_file)
        assert result.labels is not None
        assert result.predictions is not None

    def test_save_and_load(self, tmp_path):
        pipeline = Pipeline(
            [
                ("segment", "slic", {"n_segments": 500}),
                ("extract", ["spectral", "geometry"], {"pixel_size": 10}),
                ("classify", "kmeans", {"n_clusters": 5}),
            ]
        )
        path = str(tmp_path / "pipeline.json")
        pipeline.save(path)

        loaded = Pipeline.load(path)
        assert len(loaded.steps) == 3
        assert loaded.steps[0][0] == "segment"
        assert loaded.steps[1][0] == "extract"
        assert loaded.steps[2][0] == "classify"

    def test_provenance(self, synthetic_image, synthetic_meta):
        pipeline = Pipeline(
            [
                ("segment", "slic", {"n_segments": 20}),
                ("extract", ["spectral"], {}),
                ("classify", "kmeans", {"n_clusters": 3}),
            ]
        )
        result = pipeline.run(image=synthetic_image, meta=synthetic_meta)
        prov = result.provenance()
        assert "steps" in prov
        assert "total_duration_s" in prov
        assert len(prov["steps"]) == 3

    def test_export_parquet(self, tmp_path, synthetic_image, synthetic_meta):
        pipeline = Pipeline(
            [
                ("segment", "slic", {"n_segments": 20}),
                ("extract", ["spectral"], {}),
                ("classify", "kmeans", {"n_clusters": 3}),
            ]
        )
        result = pipeline.run(image=synthetic_image, meta=synthetic_meta)
        output = str(tmp_path / "result.parquet")
        result.export(output)
        df = pd.read_parquet(output)
        assert "class_label" in df.columns

    def test_export_gpkg(self, tmp_path, synthetic_image, synthetic_meta):
        pipeline = Pipeline(
            [
                ("segment", "slic", {"n_segments": 20}),
                ("extract", ["spectral"], {}),
                ("classify", "kmeans", {"n_clusters": 3}),
            ]
        )
        result = pipeline.run(image=synthetic_image, meta=synthetic_meta)
        output = str(tmp_path / "result.gpkg")
        result.export(output)
        import geopandas as gpd

        gdf = gpd.read_file(output)
        assert len(gdf) > 0

    def test_unknown_step_raises(self, synthetic_image, synthetic_meta):
        with pytest.raises(ValueError, match="Unknown pipeline step"):
            Pipeline([("bogus", "slic", {})])

    def test_pipeline_json_roundtrip(self, tmp_path):
        path = str(tmp_path / "test.json")
        original = Pipeline(
            [
                ("segment", "felzenszwalb", {"scale": 100}),
                ("extract", ["spectral", "geometry", "texture"], {}),
                ("classify", "random_forest", {"n_estimators": 50}),
            ]
        )
        original.save(path)

        with open(path) as f:
            data = json.load(f)
        assert data["version"] == "1.0"
        assert len(data["steps"]) == 3
