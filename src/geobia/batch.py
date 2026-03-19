"""Batch processing: run a Pipeline across multiple files in parallel."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BatchResult:
    """Result from processing a single file."""
    input_path: str
    labels_path: str | None = None
    features: pd.DataFrame | None = None
    predictions: pd.Series | None = None
    n_segments: int = 0
    duration_s: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


def _process_single(
    input_path: str,
    output_dir: str,
    pipeline_json: str,
    training_labels: pd.Series | None,
) -> BatchResult:
    """Process a single raster file (runs in worker process)."""
    import time

    try:
        from geobia.pipeline import Pipeline
        from geobia.io.raster import write_raster

        pipeline = Pipeline.load_string(pipeline_json)

        t0 = time.perf_counter()
        result = pipeline.run(input_path=input_path, training=training_labels)
        duration = time.perf_counter() - t0

        # Save labels
        labels_path = None
        if result.labels is not None and result.meta is not None:
            stem = Path(input_path).stem
            labels_path = os.path.join(output_dir, f"{stem}_labels.tif")
            write_raster(labels_path, result.labels, result.meta, dtype="int32")

        n_segments = int(result.labels.max()) if result.labels is not None else 0

        return BatchResult(
            input_path=input_path,
            labels_path=labels_path,
            features=result.features,
            predictions=result.predictions,
            n_segments=n_segments,
            duration_s=duration,
        )

    except Exception as e:
        return BatchResult(input_path=input_path, error=str(e))


def process_batch(
    input_paths: list[str],
    output_dir: str,
    pipeline: Any,
    training_labels: pd.Series | None = None,
    max_workers: int | None = None,
    progress_callback: Any | None = None,
) -> list[BatchResult]:
    """Run a Pipeline on multiple raster files in parallel.

    Args:
        input_paths: List of input raster file paths.
        output_dir: Directory for output label rasters.
        pipeline: A Pipeline instance defining the workflow.
        training_labels: Training labels for supervised classification
            (shared across all files).
        max_workers: Max parallel processes (None = CPU count).
        progress_callback: Optional callable(completed, total) for progress.

    Returns:
        List of BatchResult objects, one per input file.

    Example:
        from geobia.pipeline import Pipeline
        from geobia.batch import process_batch

        pipeline = Pipeline([
            ("segment", "slic", {"n_segments": 500}),
            ("extract", ["spectral", "geometry"], {}),
            ("classify", "kmeans", {"n_clusters": 6}),
        ])

        results = process_batch(
            ["tile_01.tif", "tile_02.tif", "tile_03.tif"],
            output_dir="output/",
            pipeline=pipeline,
            max_workers=4,
        )
    """
    os.makedirs(output_dir, exist_ok=True)

    # Serialize pipeline to JSON so it can be sent to worker processes
    pipeline_json = pipeline.to_json()

    results: list[BatchResult] = []
    total = len(input_paths)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single,
                path, output_dir, pipeline_json, training_labels,
            ): path
            for path in input_paths
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_callback:
                progress_callback(len(results), total)

    # Sort by input order
    path_order = {p: i for i, p in enumerate(input_paths)}
    results.sort(key=lambda r: path_order.get(r.input_path, 0))

    return results


def batch_summary(results: list[BatchResult]) -> dict:
    """Summarize batch processing results."""
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    summary = {
        "total": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "total_segments": sum(r.n_segments for r in succeeded),
        "total_duration_s": round(sum(r.duration_s for r in succeeded), 2),
    }

    if failed:
        summary["errors"] = {r.input_path: r.error for r in failed}

    return summary
