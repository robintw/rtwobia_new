"""Edge case tests for batch processing."""

import numpy as np
import pytest
import rasterio

from geobia.batch import BatchResult, batch_summary, process_batch
from geobia.pipeline import Pipeline


def test_empty_batch(tmp_path):
    """Empty input list should return empty results."""
    pipeline = Pipeline([("segment", "slic", {"n_segments": 20})])
    output_dir = str(tmp_path / "output")
    results = process_batch([], output_dir, pipeline=pipeline, max_workers=1)
    assert results == []


def test_all_files_fail(tmp_path):
    """All files failing should produce all-error results."""
    pipeline = Pipeline([("segment", "slic", {"n_segments": 20})])
    output_dir = str(tmp_path / "output")
    bad_paths = ["/nonexistent/a.tif", "/nonexistent/b.tif"]
    results = process_batch(bad_paths, output_dir, pipeline=pipeline, max_workers=1)
    assert len(results) == 2
    assert all(not r.success for r in results)

    summary = batch_summary(results)
    assert summary["succeeded"] == 0
    assert summary["failed"] == 2
    assert "errors" in summary


def test_mixed_success_failure(tmp_path, synthetic_image, synthetic_meta):
    """Mix of valid and invalid files."""
    # Create one valid raster
    valid_path = str(tmp_path / "valid.tif")
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
    with rasterio.open(valid_path, "w", **profile) as ds:
        ds.write(synthetic_image)

    pipeline = Pipeline([("segment", "slic", {"n_segments": 20})])
    output_dir = str(tmp_path / "output")
    results = process_batch(
        [valid_path, "/nonexistent/bad.tif"],
        output_dir,
        pipeline=pipeline,
        max_workers=1,
    )
    assert len(results) == 2
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    assert len(successes) == 1
    assert len(failures) == 1


def test_batch_creates_output_dir(tmp_path, synthetic_image, synthetic_meta):
    """Output directory should be created if it doesn't exist."""
    path = str(tmp_path / "image.tif")
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

    output_dir = str(tmp_path / "new" / "nested" / "dir")
    pipeline = Pipeline([("segment", "slic", {"n_segments": 20})])
    results = process_batch([path], output_dir, pipeline=pipeline, max_workers=1)
    assert results[0].success
    assert results[0].labels_path is not None


def test_batch_summary_empty():
    """Summary of empty results."""
    summary = batch_summary([])
    assert summary["total"] == 0
    assert summary["succeeded"] == 0
    assert summary["failed"] == 0


def test_batch_result_defaults():
    r = BatchResult(input_path="test.tif")
    assert r.success
    assert r.n_segments == 0
    assert r.duration_s == 0.0
    assert r.features is None
    assert r.predictions is None
    assert r.labels_path is None
