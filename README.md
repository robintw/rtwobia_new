# geobia

Geographic Object-Based Image Analysis (GEOBIA) for geospatial imagery.

A Python library and CLI for the complete OBIA pipeline: **segment** imagery into objects, **extract** rich per-object features, and **classify** objects using supervised or unsupervised methods. Inspired by eCognition (circa 2008-2011), built on open-source tools.

## Features

- **Segmentation** — SLIC superpixels, Felzenszwalb graph-based, Shepherd K-means+elimination (numba JIT accelerated), and optional SAM (segment-geospatial) — with tiled processing for large images
- **Feature extraction** — Per-band spectral statistics (mean, std, min, max, median, range), band ratios (NDVI, NDWI), geometric shape features (area, perimeter, compactness, elongation, eccentricity, and more), GLCM texture features (contrast, homogeneity, energy, correlation, dissimilarity)
- **Classification** — Supervised (Random Forest) and unsupervised (K-Means, Gaussian Mixture Model, DBSCAN), with accuracy assessment (overall accuracy, kappa, confusion matrix, cross-validation)
- **Pipeline engine** — Define segment→extract→classify pipelines as JSON, run with provenance tracking, export results to Parquet or GeoPackage
- **Export** — GeoTIFF rasters, GeoPackage vectors, Parquet feature tables
- **Large image support** — Tiled windowed I/O and tiled segmentation with overlap stitching, powered by rasterio
- **No GDAL Python bindings required** — Uses rasterio and geopandas for all geospatial I/O

## Installation

### Setting up a conda environment

The geospatial stack (rasterio, GDAL, PROJ) is easiest to install via conda:

```bash
conda create -n geobia python=3.13 \
    rasterio geopandas scikit-image scikit-learn numba \
    click numpy pandas scipy pyarrow joblib shapely \
    -c conda-forge

conda activate geobia
pip install geobia
```

### With pip (if GDAL is already available)

```bash
pip install geobia
```

### With uv

```bash
uv add geobia
```

### Requirements

- Python 3.13+
- rasterio, geopandas, scikit-image, scikit-learn, numba, click, numpy, pandas, scipy, pyarrow, joblib, shapely
- Optional: `pip install geobia[sam]` for SAM segmentation (segment-geospatial)

## Quick start — Python API

```python
import geobia

# 1. Load a multi-band satellite image
image, meta = geobia.io.read_raster("satellite.tif")

# 2. Segment into objects
labels = geobia.segment(image, method="slic", n_segments=1000)

# 3. Extract spectral and geometric features
features = geobia.extract_features(
    image, labels,
    categories=["spectral", "geometry"],
    band_names={"red": 0, "green": 1, "blue": 2, "nir": 3},
)

# 4. Classify — unsupervised (no training data needed)
clusters = geobia.classify(features, method="kmeans", n_clusters=6)

# 5. Export to GeoPackage
geobia.io.write_vector("classified.gpkg", labels, attributes=clusters, meta=meta)
```

### Supervised classification with training data

```python
import geobia
from geobia.classification import SupervisedClassifier, assess_accuracy

image, meta = geobia.io.read_raster("satellite.tif")
labels = geobia.segment(image, method="felzenszwalb", scale=100, min_size=50)

features = geobia.extract_features(
    image, labels,
    band_names={"red": 0, "green": 1, "blue": 2, "nir": 3},
)

# Load training samples (GeoPackage with a "class" column)
training_labels = geobia.io.read_training_samples("training.gpkg", labels, meta)

# Train a Random Forest classifier
clf = SupervisedClassifier("random_forest", n_estimators=200)
clf.fit(features, training_labels)

# Predict all segments
predictions = clf.predict(features)

# Assess accuracy
report = assess_accuracy(training_labels, predictions)
print(report.summary())

# Export results
geobia.io.write_vector("classified.gpkg", labels, attributes=predictions, meta=meta)
clf.save("model.joblib")
```

### Processing large images with tiling

```python
import geobia

# Segment a large image tile-by-tile (reads windows via rasterio)
labels = geobia.segment_tiled(
    "large_image.tif",
    method="slic",
    tile_size=2048,
    overlap=128,
    output_path="segments.tif",
    n_segments=500,
)
```

## Quick start — CLI

The `geobia` command provides the full pipeline from the terminal.

### Segment an image

```bash
# SLIC superpixels
geobia segment satellite.tif -o segments.tif --method slic --n-segments 1000

# Felzenszwalb graph-based
geobia segment satellite.tif -o segments.tif --method felzenszwalb --scale 150 --min-size 80

# Shepherd K-means + elimination (fast, numba accelerated)
geobia segment satellite.tif -o segments.tif --method shepherd --num-clusters 60 --min-size 100

# Tiled processing for large images
geobia segment satellite.tif -o segments.tif --method slic --n-segments 500 --tiled --tile-size 4096
```

### Extract features

```bash
# Spectral and geometric features with named bands
geobia extract satellite.tif segments.tif -o features.parquet \
    --band-names red,green,blue,nir

# Spectral features only
geobia extract satellite.tif segments.tif -o features.parquet --no-geometry

# Include GLCM texture features
geobia extract satellite.tif segments.tif -o features.parquet --texture
```

### Classify segments

```bash
# Unsupervised (K-Means clustering)
geobia classify features.parquet -o classified.parquet --method kmeans --n-clusters 6

# Gaussian Mixture Model
geobia classify features.parquet -o classified.parquet --method gmm --n-clusters 6

# DBSCAN density-based clustering
geobia classify features.parquet -o classified.parquet --method dbscan

# Supervised (Random Forest with training samples)
geobia classify features.parquet -o classified.parquet \
    --method random_forest \
    --training training_samples.gpkg \
    --segments segments.tif \
    --n-estimators 200
```

### Export to GeoPackage

```bash
geobia export segments.tif -o result.gpkg \
    --features features.parquet \
    --classification classified.parquet
```

### Inspect any dataset

```bash
geobia info satellite.tif
# File: satellite.tif
# Size: 5000 x 5000
# Bands: 4
# CRS: EPSG:32633
# Resolution: (10.0, 10.0)

geobia info features.parquet
# Segments: 1247
# Features: 38
# Feature names: mean_red, std_red, min_red, ...
```

### Full CLI pipeline example

```bash
# End-to-end: segment, extract, classify, export
geobia segment satellite.tif -o segments.tif --method slic --n-segments 1000
geobia extract satellite.tif segments.tif -o features.parquet --band-names red,green,blue,nir
geobia classify features.parquet -o classified.parquet --method kmeans --n-clusters 6
geobia export segments.tif -o result.gpkg --features features.parquet --classification classified.parquet
```

## Segmentation algorithms

| Algorithm | Strengths | Key parameters |
|-----------|-----------|----------------|
| **SLIC** | Fast, predictable segment count, good for exploration | `n_segments`, `compactness`, `sigma` |
| **Felzenszwalb** | Adaptive sizes, follows natural boundaries | `scale`, `sigma`, `min_size` |
| **Shepherd** | K-means seeded, numba JIT accelerated, good for remote sensing | `num_clusters`, `min_n_pxls`, `dist_thres`, `sampling` |
| **SAM** | Deep learning foundation model (optional: `pip install geobia[sam]`) | `points_per_side`, `pred_iou_thresh` |

## Extracted features

**Spectral** (per band): mean, std, min, max, median, range, plus brightness (cross-band mean) and band ratios (NDVI, NDWI).

**Geometric**: area (pixels and map units), perimeter, compactness, elongation, rectangularity, eccentricity, solidity, major/minor axis lengths, orientation, centroid coordinates.

**Texture** (GLCM per band): contrast, dissimilarity, homogeneity, energy, correlation.

## Pipeline engine

Define and run a full segment→extract→classify pipeline programmatically:

```python
from geobia.pipeline import Pipeline

pipeline = Pipeline([
    ("segment", "slic", {"n_segments": 1000}),
    ("extract", ["spectral", "geometry", "texture"], {}),
    ("classify", "kmeans", {"n_clusters": 6}),
])

result = pipeline.run(input_path="satellite.tif")
result.export("output.gpkg")

# Save pipeline definition for reproducibility
pipeline.save("pipeline.json")

# Check timing provenance
prov = result.provenance()
for step in prov["steps"]:
    print(f"{step['name']}: {step['duration_s']:.1f}s")
```

## Development

```bash
git clone <repo-url>
cd geobia
uv sync
uv run pytest tests/ -v
```

## License

TBD
