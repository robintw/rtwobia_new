"""Benchmark tests for segmentation performance on SPOT image.

Run with: pytest tests/test_benchmark.py -v -m spot_image
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

SPOT_PATH = Path(__file__).parent / "data" / "SPOT_ROI.tif"
SPOT_AVAILABLE = SPOT_PATH.exists()

pytestmark = [
    pytest.mark.spot_image,
    pytest.mark.slow,
    pytest.mark.benchmark,
    pytest.mark.skipif(not SPOT_AVAILABLE, reason="SPOT_ROI.tif not found in tests/data/"),
]


@pytest.fixture(scope="module")
def spot_image():
    from geobia.io.raster import read_raster

    return read_raster(str(SPOT_PATH))


class TestShepherdBenchmark:
    def test_shepherd_spot_performance(self, spot_image):
        """Benchmark Shepherd segmentation on SPOT image.

        Asserts it completes in under 30 seconds on a typical machine.
        """
        from geobia.segmentation import segment

        image, meta = spot_image

        t0 = time.perf_counter()
        labels = segment(
            image,
            method="shepherd",
            num_clusters=60,
            min_n_pxls=100,
            dist_thres=100.0,
            sampling=100,
        )
        elapsed = time.perf_counter() - t0

        n_segments = len(np.unique(labels)) - (1 if 0 in labels else 0)

        print(
            f"\nShepherd on SPOT ({image.shape[1]}x{image.shape[2]}): "
            f"{elapsed:.1f}s, {n_segments} segments"
        )

        assert labels.shape == (image.shape[1], image.shape[2])
        assert n_segments > 100
        assert elapsed < 30, f"Shepherd took {elapsed:.1f}s, expected < 30s"

    def test_shepherd_vs_slic_speed(self, spot_image):
        """Compare Shepherd and SLIC speeds on the same image."""
        from geobia.segmentation import segment

        image, meta = spot_image

        t0 = time.perf_counter()
        slic_labels = segment(image, method="slic", n_segments=500)
        slic_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        shep_labels = segment(
            image,
            method="shepherd",
            num_clusters=60,
            min_n_pxls=100,
            dist_thres=100.0,
            sampling=100,
        )
        shep_time = time.perf_counter() - t0

        print(f"\nSLIC: {slic_time:.1f}s ({len(np.unique(slic_labels))} segments)")
        print(f"Shepherd: {shep_time:.1f}s ({len(np.unique(shep_labels))} segments)")

        # Both should complete
        assert slic_labels.max() > 0
        assert shep_labels.max() > 0


class TestFullPipelineBenchmark:
    def test_pipeline_spot_performance(self, spot_image):
        """Benchmark full pipeline on SPOT image."""
        from geobia.pipeline import Pipeline

        image, meta = spot_image

        pipeline = Pipeline(
            [
                ("segment", "slic", {"n_segments": 500}),
                ("extract", ["spectral", "geometry"], {}),
                ("classify", "kmeans", {"n_clusters": 5}),
            ]
        )

        t0 = time.perf_counter()
        result = pipeline.run(image=image, meta=meta)
        elapsed = time.perf_counter() - t0

        prov = result.provenance()
        print(f"\nFull pipeline: {elapsed:.1f}s total")
        for step in prov["steps"]:
            print(f"  {step['name']}: {step['duration_s']:.1f}s")

        assert result.predictions is not None
        assert elapsed < 120, f"Pipeline took {elapsed:.1f}s, expected < 120s"
