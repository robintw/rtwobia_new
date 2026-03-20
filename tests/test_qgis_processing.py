"""Tests for QGIS Processing Framework algorithms.

These tests require QGIS Python bindings to be installed. They are marked
with 'qgis' so they can be skipped in environments without QGIS:

    pytest tests/ -v -m "not qgis"      # skip QGIS tests
    pytest tests/ -v -m "qgis"           # run QGIS tests only
"""

from __future__ import annotations

import os

import numpy as np
import pytest

qgis_core = pytest.importorskip("qgis.core", reason="QGIS Python bindings not available")

from qgis.core import (
    QgsApplication,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

pytestmark = pytest.mark.qgis


# ---------------------------------------------------------------------------
# QGIS application fixture (session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def qgis_app():
    """Initialize a QGIS application for the test session."""
    app = QgsApplication([], False)
    app.initQgis()
    yield app
    app.exitQgis()


@pytest.fixture
def processing_context():
    return QgsProcessingContext()


@pytest.fixture
def processing_feedback():
    return QgsProcessingFeedback()


# ---------------------------------------------------------------------------
# Helper: create a temporary 4-band GeoTIFF
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_raster(tmp_path):
    """Write a small 4-band raster to disk and return the path."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    path = str(tmp_path / "sample.tif")
    rng = np.random.RandomState(42)
    image = rng.rand(4, 50, 50).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 50,
        "height": 50,
        "count": 4,
        "crs": CRS.from_epsg(32633),
        "transform": from_bounds(500000, 5000000, 500500, 5000500, 50, 50),
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(image)
    return path


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------


class TestGeobiaProvider:
    def test_provider_loads(self):
        from qgis_plugin.processing.provider import GeobiaProvider

        provider = GeobiaProvider()
        assert provider.id() == "geobia"
        assert provider.name() == "GeoOBIA"

    def test_provider_algorithms(self):
        from qgis_plugin.processing.provider import GeobiaProvider

        provider = GeobiaProvider()
        provider.loadAlgorithms()
        alg_ids = [a.name() for a in provider.algorithms()]
        assert "segment" in alg_ids
        assert "extract_features" in alg_ids
        assert "classify" in alg_ids
        assert "batch" in alg_ids
        assert "change_detection" in alg_ids
        assert "multiscale_segment" in alg_ids


# ---------------------------------------------------------------------------
# Algorithm metadata tests
# ---------------------------------------------------------------------------


class TestSegmentationAlgorithmMeta:
    def test_metadata(self):
        from qgis_plugin.processing.segmentation_alg import SegmentationAlgorithm

        alg = SegmentationAlgorithm()
        assert alg.name() == "segment"
        assert alg.displayName() == "Segment Image"
        assert alg.group() == "Segmentation"
        assert alg.groupId() == "segmentation"

    def test_parameters_defined(self):
        from qgis_plugin.processing.segmentation_alg import SegmentationAlgorithm

        alg = SegmentationAlgorithm()
        alg.initAlgorithm()
        param_names = [p.name() for p in alg.parameterDefinitions()]
        assert "INPUT" in param_names
        assert "METHOD" in param_names
        assert "N_SEGMENTS" in param_names
        assert "OUTPUT" in param_names


class TestFeatureExtractionAlgorithmMeta:
    def test_metadata(self):
        from qgis_plugin.processing.features_alg import FeatureExtractionAlgorithm

        alg = FeatureExtractionAlgorithm()
        assert alg.name() == "extract_features"
        assert alg.displayName() == "Extract Features"
        assert alg.group() == "Features"

    def test_parameters_defined(self):
        from qgis_plugin.processing.features_alg import FeatureExtractionAlgorithm

        alg = FeatureExtractionAlgorithm()
        alg.initAlgorithm()
        param_names = [p.name() for p in alg.parameterDefinitions()]
        assert "INPUT" in param_names
        assert "SEGMENTS" in param_names
        assert "SPECTRAL" in param_names
        assert "GEOMETRY" in param_names
        assert "TEXTURE" in param_names
        assert "OUTPUT" in param_names


class TestClassificationAlgorithmMeta:
    def test_metadata(self):
        from qgis_plugin.processing.classification_alg import ClassificationAlgorithm

        alg = ClassificationAlgorithm()
        assert alg.name() == "classify"
        assert alg.displayName() == "Classify Segments"
        assert alg.group() == "Classification"

    def test_parameters_defined(self):
        from qgis_plugin.processing.classification_alg import ClassificationAlgorithm

        alg = ClassificationAlgorithm()
        alg.initAlgorithm()
        param_names = [p.name() for p in alg.parameterDefinitions()]
        assert "FEATURES" in param_names
        assert "METHOD" in param_names
        assert "N_CLUSTERS" in param_names
        assert "OUTPUT" in param_names


class TestBatchAlgorithmMeta:
    def test_metadata(self):
        from qgis_plugin.processing.batch_alg import BatchProcessingAlgorithm

        alg = BatchProcessingAlgorithm()
        assert alg.name() == "batch"
        assert alg.displayName() == "Batch Process Multiple Images"
        assert alg.group() == "Batch"

    def test_parameters_defined(self):
        from qgis_plugin.processing.batch_alg import BatchProcessingAlgorithm

        alg = BatchProcessingAlgorithm()
        alg.initAlgorithm()
        param_names = [p.name() for p in alg.parameterDefinitions()]
        assert "INPUT" in param_names
        assert "SEG_METHOD" in param_names
        assert "OUTPUT" in param_names


class TestChangeDetectionAlgorithmMeta:
    def test_metadata(self):
        from qgis_plugin.processing.change_detection_alg import ChangeDetectionAlgorithm

        alg = ChangeDetectionAlgorithm()
        assert alg.name() == "change_detection"
        assert alg.displayName() == "Change Detection"
        assert alg.group() == "Analysis"

    def test_parameters_defined(self):
        from qgis_plugin.processing.change_detection_alg import ChangeDetectionAlgorithm

        alg = ChangeDetectionAlgorithm()
        alg.initAlgorithm()
        param_names = [p.name() for p in alg.parameterDefinitions()]
        assert "FEATURES_T1" in param_names
        assert "FEATURES_T2" in param_names
        assert "THRESHOLD_METHOD" in param_names
        assert "OUTPUT" in param_names


class TestMultiscaleAlgorithmMeta:
    def test_metadata(self):
        from qgis_plugin.processing.multiscale_alg import MultiscaleSegmentationAlgorithm

        alg = MultiscaleSegmentationAlgorithm()
        assert alg.name() == "multiscale_segment"
        assert alg.displayName() == "Multi-Scale Segmentation"
        assert alg.group() == "Segmentation"

    def test_parameters_defined(self):
        from qgis_plugin.processing.multiscale_alg import MultiscaleSegmentationAlgorithm

        alg = MultiscaleSegmentationAlgorithm()
        alg.initAlgorithm()
        param_names = [p.name() for p in alg.parameterDefinitions()]
        assert "INPUT" in param_names
        assert "METHOD" in param_names
        assert "N_LEVELS" in param_names
        assert "OUTPUT_FINE" in param_names
        assert "OUTPUT_COARSE" in param_names


# ---------------------------------------------------------------------------
# End-to-end algorithm tests
# ---------------------------------------------------------------------------


class TestSegmentationAlgorithmRun:
    def test_segment_slic(self, sample_raster, tmp_path, processing_context, processing_feedback):
        from qgis_plugin.processing.segmentation_alg import SegmentationAlgorithm

        alg = SegmentationAlgorithm()
        alg.initAlgorithm()

        output_path = str(tmp_path / "labels.tif")
        params = {
            "INPUT": sample_raster,
            "METHOD": 0,  # slic
            "N_SEGMENTS": 20,
            "COMPACTNESS": 10.0,
            "SIGMA": 0.0,
            "OUTPUT": output_path,
        }
        result = alg.processAlgorithm(params, processing_context, processing_feedback)
        assert os.path.exists(result["OUTPUT"])

        import rasterio

        with rasterio.open(result["OUTPUT"]) as ds:
            labels = ds.read(1)
            assert labels.shape == (50, 50)
            assert labels.max() > 0

    def test_segment_felzenszwalb(
        self, sample_raster, tmp_path, processing_context, processing_feedback
    ):
        from qgis_plugin.processing.segmentation_alg import SegmentationAlgorithm

        alg = SegmentationAlgorithm()
        alg.initAlgorithm()

        output_path = str(tmp_path / "labels_fz.tif")
        params = {
            "INPUT": sample_raster,
            "METHOD": 1,  # felzenszwalb
            "SCALE": 50.0,
            "FZ_SIGMA": 0.8,
            "MIN_SIZE": 20,
            "OUTPUT": output_path,
        }
        result = alg.processAlgorithm(params, processing_context, processing_feedback)
        assert os.path.exists(result["OUTPUT"])


class TestFeatureExtractionAlgorithmRun:
    def test_extract_spectral(
        self, sample_raster, tmp_path, processing_context, processing_feedback
    ):
        from qgis_plugin.processing.features_alg import FeatureExtractionAlgorithm
        from qgis_plugin.processing.segmentation_alg import SegmentationAlgorithm

        # First segment
        seg_alg = SegmentationAlgorithm()
        seg_alg.initAlgorithm()
        labels_path = str(tmp_path / "labels.tif")
        seg_result = seg_alg.processAlgorithm(
            {
                "INPUT": sample_raster,
                "METHOD": 0,
                "N_SEGMENTS": 20,
                "COMPACTNESS": 10.0,
                "SIGMA": 0.0,
                "OUTPUT": labels_path,
            },
            processing_context,
            processing_feedback,
        )

        # Then extract features
        feat_alg = FeatureExtractionAlgorithm()
        feat_alg.initAlgorithm()
        output_path = str(tmp_path / "features.parquet")
        result = feat_alg.processAlgorithm(
            {
                "INPUT": sample_raster,
                "SEGMENTS": seg_result["OUTPUT"],
                "SPECTRAL": True,
                "GEOMETRY": True,
                "TEXTURE": False,
                "BAND_NAMES": "",
                "OUTPUT": output_path,
            },
            processing_context,
            processing_feedback,
        )

        assert os.path.exists(result["OUTPUT"])
        import pandas as pd

        df = pd.read_parquet(result["OUTPUT"])
        assert len(df) > 0
        assert "mean_band_0" in df.columns


class TestClassificationAlgorithmRun:
    def test_unsupervised_kmeans(
        self, sample_raster, tmp_path, processing_context, processing_feedback
    ):
        from qgis_plugin.processing.classification_alg import ClassificationAlgorithm
        from qgis_plugin.processing.features_alg import FeatureExtractionAlgorithm
        from qgis_plugin.processing.segmentation_alg import SegmentationAlgorithm

        # Segment
        seg_alg = SegmentationAlgorithm()
        seg_alg.initAlgorithm()
        labels_path = str(tmp_path / "labels.tif")
        seg_alg.processAlgorithm(
            {
                "INPUT": sample_raster,
                "METHOD": 0,
                "N_SEGMENTS": 20,
                "COMPACTNESS": 10.0,
                "SIGMA": 0.0,
                "OUTPUT": labels_path,
            },
            processing_context,
            processing_feedback,
        )

        # Extract features
        feat_alg = FeatureExtractionAlgorithm()
        feat_alg.initAlgorithm()
        features_path = str(tmp_path / "features.parquet")
        feat_alg.processAlgorithm(
            {
                "INPUT": sample_raster,
                "SEGMENTS": labels_path,
                "SPECTRAL": True,
                "GEOMETRY": False,
                "TEXTURE": False,
                "BAND_NAMES": "",
                "OUTPUT": features_path,
            },
            processing_context,
            processing_feedback,
        )

        # Classify
        cls_alg = ClassificationAlgorithm()
        cls_alg.initAlgorithm()
        output_path = str(tmp_path / "classified.parquet")
        result = cls_alg.processAlgorithm(
            {
                "FEATURES": features_path,
                "METHOD": 3,  # kmeans
                "N_CLUSTERS": 3,
                "OUTPUT": output_path,
            },
            processing_context,
            processing_feedback,
        )

        assert os.path.exists(result["OUTPUT"])
        import pandas as pd

        df = pd.read_parquet(result["OUTPUT"])
        assert "class_label" in df.columns
        assert df["class_label"].nunique() == 3
