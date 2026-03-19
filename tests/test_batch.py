"""Tests for batch processing."""

import numpy as np
import pytest
import rasterio

from geobia.batch import BatchResult, batch_summary, process_batch
from geobia.pipeline import Pipeline


@pytest.fixture
def batch_rasters(tmp_path, synthetic_image, synthetic_meta):
    """Create 3 small test rasters."""
    paths = []
    for i in range(3):
        path = str(tmp_path / f"image_{i}.tif")
        bands, h, w = synthetic_image.shape
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": w,
            "height": h,
            "count": bands,
            "crs": synthetic_meta["crs"],
            "transform": synthetic_meta["transform"],
        }
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(synthetic_image)
        paths.append(path)
    return paths


def test_batch_segment_only(batch_rasters, tmp_path):
    pipeline = Pipeline([
        ("segment", "slic", {"n_segments": 20}),
    ])
    output_dir = str(tmp_path / "output")
    results = process_batch(
        batch_rasters, output_dir, pipeline=pipeline, max_workers=1,
    )
    assert len(results) == 3
    assert all(r.success for r in results)
    assert all(r.n_segments > 0 for r in results)
    assert all(r.labels_path is not None for r in results)


def test_batch_with_features(batch_rasters, tmp_path):
    pipeline = Pipeline([
        ("segment", "slic", {"n_segments": 20}),
        ("extract", ["spectral"], {}),
    ])
    output_dir = str(tmp_path / "output")
    results = process_batch(
        batch_rasters, output_dir, pipeline=pipeline, max_workers=1,
    )
    assert all(r.features is not None for r in results)
    assert all(len(r.features.columns) > 0 for r in results)


def test_batch_with_classification(batch_rasters, tmp_path):
    pipeline = Pipeline([
        ("segment", "slic", {"n_segments": 20}),
        ("extract", ["spectral"], {}),
        ("classify", "kmeans", {"n_clusters": 3}),
    ])
    output_dir = str(tmp_path / "output")
    results = process_batch(
        batch_rasters, output_dir, pipeline=pipeline, max_workers=1,
    )
    assert all(r.predictions is not None for r in results)
    assert all(r.predictions.nunique() > 0 for r in results)


def test_batch_summary(batch_rasters, tmp_path):
    pipeline = Pipeline([
        ("segment", "slic", {"n_segments": 20}),
    ])
    output_dir = str(tmp_path / "output")
    results = process_batch(
        batch_rasters, output_dir, pipeline=pipeline, max_workers=1,
    )
    summary = batch_summary(results)
    assert summary["total"] == 3
    assert summary["succeeded"] == 3
    assert summary["failed"] == 0
    assert summary["total_segments"] > 0
    assert summary["total_duration_s"] > 0


def test_batch_handles_errors(tmp_path):
    pipeline = Pipeline([
        ("segment", "slic", {"n_segments": 20}),
    ])
    output_dir = str(tmp_path / "output")
    results = process_batch(
        ["/nonexistent/file.tif"], output_dir, pipeline=pipeline, max_workers=1,
    )
    assert len(results) == 1
    assert not results[0].success
    assert results[0].error is not None


def test_batch_progress_callback(batch_rasters, tmp_path):
    pipeline = Pipeline([
        ("segment", "slic", {"n_segments": 20}),
    ])
    output_dir = str(tmp_path / "output")
    progress_calls = []
    results = process_batch(
        batch_rasters, output_dir, pipeline=pipeline, max_workers=1,
        progress_callback=lambda done, total: progress_calls.append((done, total)),
    )
    assert len(progress_calls) == 3
    assert progress_calls[-1] == (3, 3)


def test_batch_result_properties():
    r = BatchResult(input_path="test.tif")
    assert r.success
    r_err = BatchResult(input_path="test.tif", error="boom")
    assert not r_err.success


def test_batch_from_saved_pipeline(batch_rasters, tmp_path):
    """Pipeline can be saved to JSON and used for batch processing."""
    pipeline = Pipeline([
        ("segment", "slic", {"n_segments": 20}),
        ("extract", ["spectral"], {}),
    ])
    pipeline.save(str(tmp_path / "pipeline.json"))

    loaded = Pipeline.load(str(tmp_path / "pipeline.json"))
    output_dir = str(tmp_path / "output")
    results = process_batch(
        batch_rasters, output_dir, pipeline=loaded, max_workers=1,
    )
    assert all(r.success for r in results)
    assert all(r.features is not None for r in results)
