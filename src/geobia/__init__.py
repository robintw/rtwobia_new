"""geobia: Geographic Object-Based Image Analysis for geospatial imagery.

A complete pipeline for segmenting imagery into objects, extracting rich
per-object features, and classifying objects using supervised or unsupervised
methods.

Example:
    >>> import geobia
    >>> image, meta = geobia.io.read_raster("satellite.tif")
    >>> labels = geobia.segment(image, method="slic", n_segments=1000)
    >>> features = geobia.extract_features(image, labels)
    >>> classes = geobia.classify(features, method="kmeans", n_clusters=5)
"""

from __future__ import annotations

__version__ = "0.1.0"

from geobia import io
from geobia.segmentation import segment, segment_tiled
from geobia.features import extract as extract_features
from geobia.classification import classify

__all__ = [
    "__version__",
    "io",
    "segment",
    "segment_tiled",
    "extract_features",
    "classify",
]
