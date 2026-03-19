# geobia

Geographic Object-Based Image Analysis for geospatial imagery.

A complete pipeline for segmenting imagery into objects, extracting rich
per-object features, and classifying objects using supervised or unsupervised
methods.

## Quick Start

```python
import geobia

image, meta = geobia.io.read_raster("satellite.tif")
labels = geobia.segment(image, method="slic", n_segments=1000)
features = geobia.extract_features(image, labels)
classes = geobia.classify(features, method="kmeans", n_clusters=5)
```

## Installation

```bash
pip install geobia
```

## CLI Usage

```bash
geobia segment image.tif -o segments.tif --method slic --n-segments 500
geobia extract image.tif segments.tif -o features.parquet
geobia classify features.parquet -o classified.parquet --method kmeans
geobia export segments.tif -o output.gpkg --features features.parquet
geobia info image.tif
```
