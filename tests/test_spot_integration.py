"""Integration tests using real SPOT satellite imagery.

These tests run the full pipeline on a real SPOT image and are slower
than the unit tests. They are marked with 'spot_image' and 'slow' markers
so they can be excluded:

    pytest tests/ -v -m "not spot_image"      # skip SPOT tests
    pytest tests/ -v -m "spot_image"           # run SPOT tests only

The SPOT_ROI.tif file is automatically downloaded on first run by the
``spot_path`` fixture in conftest.py and cached in tests/data/.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytestmark = [
    pytest.mark.spot_image,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def spot_image(spot_path):
    from geobia.io.raster import read_raster

    image, meta = read_raster(str(spot_path))
    return image, meta


@pytest.fixture(scope="module")
def spot_slic_labels(spot_image):
    from geobia.segmentation import segment

    image, meta = spot_image
    labels = segment(image, method="slic", n_segments=500, compactness=10.0)
    return labels


@pytest.fixture(scope="module")
def spot_felz_labels(spot_image):
    from geobia.segmentation import segment

    image, meta = spot_image
    labels = segment(image, method="felzenszwalb", scale=100, min_size=50)
    return labels


class TestSpotSegmentation:
    def test_slic_produces_segments(self, spot_slic_labels):
        labels = spot_slic_labels
        assert labels.shape == (1644, 1435)
        assert labels.dtype == np.int32
        n = len(np.unique(labels)) - (1 if 0 in labels else 0)
        assert 100 < n < 2000

    def test_felzenszwalb_produces_segments(self, spot_felz_labels):
        labels = spot_felz_labels
        assert labels.shape == (1644, 1435)
        n = len(np.unique(labels)) - (1 if 0 in labels else 0)
        assert n > 10


class TestSpotFeatureExtraction:
    def test_spectral_features(self, spot_image, spot_slic_labels):
        from geobia.features.spectral import SpectralExtractor

        image, meta = spot_image
        ext = SpectralExtractor()
        df = ext.extract(image, spot_slic_labels)
        assert len(df) > 100
        assert "mean_band_0" in df.columns
        assert "brightness" in df.columns
        assert df["mean_band_0"].notna().all()

    def test_geometry_features(self, spot_image, spot_slic_labels):
        from geobia.features.geometry import GeometryExtractor

        image, meta = spot_image
        ext = GeometryExtractor(pixel_size=10.0)
        df = ext.extract(image, spot_slic_labels)
        assert len(df) > 100
        assert "area" in df.columns
        assert "compactness" in df.columns
        assert (df["area"] > 0).all()

    def test_combined_features(self, spot_image, spot_slic_labels):
        from geobia.features import extract

        image, meta = spot_image
        df = extract(image, spot_slic_labels, pixel_size=10.0)
        assert "mean_band_0" in df.columns
        assert "area_px" in df.columns
        assert len(df) > 100


class TestSpotClassification:
    def test_unsupervised_kmeans(self, spot_image, spot_slic_labels):
        from geobia.classification import classify
        from geobia.features import extract

        image, meta = spot_image
        features = extract(image, spot_slic_labels)
        predictions = classify(features, method="kmeans", n_clusters=5)
        assert len(predictions) == len(features)
        assert predictions.nunique() == 5

    def test_supervised_random_forest(self, spot_image, spot_slic_labels):
        from geobia.classification import SupervisedClassifier, assess_accuracy
        from geobia.features import extract

        image, meta = spot_image
        features = extract(image, spot_slic_labels)

        # Create synthetic training labels from K-Means clusters
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = km.fit_predict(features.values)
        class_names = ["water", "vegetation", "urban", "soil"]
        training_labels = pd.Series(
            [class_names[c] for c in clusters],
            index=features.index,
            name="class_label",
        )

        # Use 50% for training
        train_idx = features.index[: len(features) // 2]
        train_labels = training_labels.loc[train_idx]

        clf = SupervisedClassifier("random_forest", n_estimators=50)
        clf.fit(features, train_labels)
        predictions = clf.predict(features)

        assert len(predictions) == len(features)

        # Assess accuracy on training set
        report = assess_accuracy(train_labels, predictions.loc[train_idx])
        assert report.overall_accuracy > 0.5


class TestSpotExport:
    def test_export_vector(self, tmp_path, spot_image, spot_slic_labels):
        from geobia.features import extract
        from geobia.io.vector import write_vector

        image, meta = spot_image
        features = extract(image, spot_slic_labels)

        output = str(tmp_path / "spot_result.gpkg")
        write_vector(output, spot_slic_labels, attributes=features, meta=meta)
        assert os.path.exists(output)

    def test_export_classified_raster(self, tmp_path, spot_image, spot_slic_labels):
        from geobia.classification import classify
        from geobia.features import extract
        from geobia.io.raster import write_raster

        image, meta = spot_image
        features = extract(image, spot_slic_labels)
        predictions = classify(features, method="kmeans", n_clusters=5)

        # Create classified raster: map segment labels to class IDs
        classified = np.zeros_like(spot_slic_labels)
        for seg_id, cls in predictions.items():
            classified[spot_slic_labels == seg_id] = cls

        output = str(tmp_path / "spot_classified.tif")
        write_raster(output, classified, meta, dtype="int32")
        assert os.path.exists(output)


class TestSpotFullPipeline:
    def test_end_to_end(self, tmp_path, spot_image):
        """Full pipeline: read -> segment -> extract -> classify -> export."""
        import geobia

        image, meta = spot_image

        # Segment
        labels = geobia.segment(image, method="slic", n_segments=300)
        assert labels.shape == image.shape[1:]

        # Extract features
        features = geobia.extract_features(image, labels, pixel_size=10.0)
        assert len(features) > 50

        # Classify
        predictions = geobia.classify(features, method="kmeans", n_clusters=5)
        assert len(predictions) == len(features)

        # Export
        output = str(tmp_path / "pipeline_result.gpkg")
        geobia.io.write_vector(output, labels, attributes=predictions, meta=meta)
        assert os.path.exists(output)
