# geobia

Geographic Object-Based Image Analysis (GEOBIA) for geospatial imagery.

A Python library and CLI for the complete OBIA pipeline: **segment** imagery into objects, **extract** rich per-object features, and **classify** objects using supervised or unsupervised methods. Inspired by eCognition (circa 2008-2011), built on open-source tools.

## Features

- **Segmentation** — SLIC superpixels, Felzenszwalb graph-based, Shepherd K-means+elimination (pyshepseg), Watershed, and optional SAM (segment-geospatial) — with tiled processing for large images and multi-scale hierarchical segmentation
- **Feature extraction** — Per-band spectral statistics (mean, std, min, max, median, range), band ratios (NDVI, NDWI), geometric shape features (area, perimeter, compactness, elongation, eccentricity, and more), GLCM texture features (contrast, homogeneity, energy, correlation, dissimilarity), contextual features (neighbor relationships, border contrast)
- **Classification** — Supervised (Random Forest, SVM, Gradient Boosting), unsupervised (K-Means, GMM, DBSCAN), and rule-based fuzzy membership functions (eCognition-style), with accuracy assessment (overall accuracy, kappa, confusion matrix, cross-validation)
- **Change detection** — Compare per-segment features between two time periods, Otsu or fixed threshold, change summary statistics
- **Batch processing** — Process multiple raster files in parallel with `ProcessPoolExecutor`
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

# Shepherd K-means + elimination (pyshepseg)
geobia segment satellite.tif -o segments.tif --method shepherd --num-clusters 60 --min-size 100

# Watershed (marker-based)
geobia segment satellite.tif -o segments.tif --method watershed --markers 500 --min-distance 10

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

# SVM
geobia classify features.parquet -o classified.parquet \
    --method svm \
    --training training_samples.gpkg \
    --segments segments.tif

# Gradient Boosting
geobia classify features.parquet -o classified.parquet \
    --method gradient_boosting \
    --training training_samples.gpkg \
    --segments segments.tif
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
| **Shepherd** | K-means seeded, pyshepseg backend, good for remote sensing | `num_clusters`, `min_n_pxls`, `dist_thres`, `sampling` |
| **Watershed** | Marker-based, follows gradient boundaries | `markers`, `compactness`, `min_distance` |
| **SAM** | Deep learning foundation model (optional: `pip install geobia[sam]`) | `points_per_side`, `pred_iou_thresh` |

## Extracted features

**Spectral** (per band): mean, std, min, max, median, range, plus brightness (cross-band mean) and band ratios (NDVI, NDWI).

**Geometric**: area (pixels and map units), perimeter, compactness, elongation, rectangularity, eccentricity, solidity, major/minor axis lengths, orientation, centroid coordinates.

**Texture** (GLCM per band): contrast, dissimilarity, homogeneity, energy, correlation.

**Contextual**: number of neighbors, mean/std of neighbor brightness, border contrast.

## Multi-scale / hierarchical segmentation

Segment at multiple scales and analyse cross-scale relationships:

```python
from geobia.segmentation.multiscale import segment_multiscale

image, meta = geobia.io.read_raster("satellite.tif")

# Segment at 3 scales — finest to coarsest
hierarchy = segment_multiscale(
    image,
    method="slic",
    scales=[
        {"n_segments": 2000, "compactness": 10},  # fine
        {"n_segments": 500, "compactness": 10},    # medium
        {"n_segments": 50, "compactness": 10},     # coarse
    ],
)

print(f"{hierarchy.n_levels} levels:")
for level in hierarchy.levels:
    print(f"  scale={level.scale}, {level.n_segments} segments")

# Map each fine segment to its parent coarse segment
parent_map = hierarchy.parent_map(fine_idx=0, coarse_idx=2)
# {1: 3, 2: 3, 3: 7, ...} — fine segment 1 is inside coarse segment 3

# Compute cross-scale features (area ratio, sibling count)
cross_features = hierarchy.cross_scale_features(fine_idx=0, coarse_idx=1)
print(cross_features.head())
#              parent_id  area_ratio  n_siblings
# segment_id
# 1                   12        0.08          14
# 2                   12        0.06          14
# 3                    5        0.15           8

# Use the finest level for feature extraction, enrich with cross-scale info
features = geobia.extract_features(image, hierarchy.finest.labels)
features = features.join(cross_features)
```

Default scales are provided if you omit the `scales` parameter — it will produce 3 levels automatically.

## Fuzzy / rule-based classification

Define classes using fuzzy membership functions (eCognition-style):

```python
from geobia.classification import FuzzyClassifier, FuzzyRule

# Define rules for each class using trapezoidal membership functions.
# FuzzyRule(feature, low, high) — full membership between low and high.
# Optional low_edge/high_edge for gradual ramp-up/down.
rules = {
    "vegetation": [
        FuzzyRule("ndvi", 0.3, 1.0, low_edge=0.2),         # high NDVI
        FuzzyRule("brightness_mean", 0.0, 0.4),             # not too bright
    ],
    "water": [
        FuzzyRule("ndvi", -1.0, 0.0, high_edge=0.1),       # low NDVI
        FuzzyRule("brightness_mean", 0.0, 0.15, high_edge=0.2),  # dark
    ],
    "urban": [
        FuzzyRule("ndvi", -0.1, 0.2),                       # low NDVI
        FuzzyRule("brightness_mean", 0.3, 1.0, low_edge=0.25),   # bright
    ],
}

clf = FuzzyClassifier(rules=rules)

# Predict — each segment gets the class with highest membership (fuzzy AND of rules)
predictions = clf.predict(features)

# Inspect membership degrees per class
memberships = clf.predict_proba(features)
print(memberships.head())
#              vegetation  water  urban
# segment_id
# 1                  0.95   0.00   0.00
# 2                  0.80   0.00   0.10
# 3                  0.00   0.90   0.00

# Or use via the convenience function:
predictions = geobia.classify(features, method="fuzzy", rules=rules)
```

Segments with zero membership in all classes are labelled `"unclassified"`.

## Batch processing

Batch processing takes a `Pipeline` and runs it on multiple files in parallel. Define your workflow once, apply it everywhere:

```python
from geobia.pipeline import Pipeline
from geobia.batch import process_batch, batch_summary

# Define the workflow once
pipeline = Pipeline([
    ("segment", "slic", {"n_segments": 500}),
    ("extract", ["spectral", "geometry"], {}),
    ("classify", "kmeans", {"n_clusters": 6}),
])

# Or load a previously saved pipeline
# pipeline = Pipeline.load("my_pipeline.json")

# Run on multiple files in parallel
results = process_batch(
    ["tile_01.tif", "tile_02.tif", "tile_03.tif", "tile_04.tif"],
    output_dir="output/",
    pipeline=pipeline,
    max_workers=4,  # None = use all CPU cores
    progress_callback=lambda done, total: print(f"{done}/{total}"),
)

# Check results
summary = batch_summary(results)
print(f"Processed {summary['succeeded']}/{summary['total']} files")
print(f"Total segments: {summary['total_segments']}")
print(f"Total time: {summary['total_duration_s']}s")

# Access individual results
for r in results:
    if r.success:
        print(f"{r.input_path}: {r.n_segments} segments, labels at {r.labels_path}")
        print(f"  Features shape: {r.features.shape}")
        print(f"  Classes: {r.predictions.nunique()}")
    else:
        print(f"{r.input_path}: FAILED — {r.error}")
```

Each file is processed independently in a separate process. Segmentation labels are saved as GeoTIFFs in the output directory. The pipeline can include any subset of steps — segmentation only, segmentation + extraction, or the full workflow.

## Change detection

Compare per-segment features between two time periods:

```python
from geobia.change import detect_changes, change_magnitude, change_summary

# Extract features from two dates using the same segmentation
features_2020 = geobia.extract_features(image_2020, labels)
features_2023 = geobia.extract_features(image_2023, labels)

# Compute per-segment change magnitude (Euclidean distance in feature space)
magnitude = change_magnitude(features_2020, features_2023, normalize=True)

# Classify as changed/unchanged using Otsu thresholding
changed = detect_changes(features_2020, features_2023, threshold="otsu")
print(f"Changed segments: {changed.sum()} / {len(changed)}")

# Or use a fixed threshold
changed = detect_changes(features_2020, features_2023, threshold=2.5)

# Get a summary
summary = change_summary(changed, features_2020, features_2023)
print(f"{summary['pct_changed']}% of segments changed")
```

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

### Setup

```bash
git clone <repo-url>
cd geobia
uv sync
```

### Running tests

```bash
# Run all fast tests (excludes slow, SPOT, and QGIS tests)
uv run pytest tests/ -v -m "not spot_image and not slow and not qgis"

# Run the full fast test suite (236 tests)
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_features.py -v

# Run a specific test
uv run pytest tests/test_features.py::TestSpectralExtractor::test_extracts_mean_per_band -v
```

### Test markers

Tests are tagged with pytest markers so you can include or exclude groups:

| Marker | Description | When to use |
|--------|-------------|-------------|
| `slow` | Tests that take more than a few seconds | Skip during rapid iteration: `-m "not slow"` |
| `spot_image` | Integration tests using a real SPOT satellite image | Requires `tests/data/SPOT_ROI.tif` (see below) |
| `benchmark` | Performance benchmarks | Run separately: `-m benchmark` |
| `qgis` | Tests for the QGIS Processing provider | Requires QGIS Python bindings installed |

Combine markers with boolean expressions:

```bash
# Everything except slow and QGIS
uv run pytest tests/ -m "not slow and not qgis"

# Only SPOT integration tests
uv run pytest tests/ -m spot_image
```

### SPOT integration test image

The SPOT integration tests (`tests/test_spot_integration.py`) use a real 6.8 MB SPOT satellite image. The file is **not committed** to the repository (it is gitignored) but is **automatically downloaded** on first run from `https://rtwilson.com/downloads/SPOT_ROI.tif` and cached at `tests/data/SPOT_ROI.tif`.

If the download fails (e.g. no network), the SPOT tests are skipped.

### Linting and formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

```bash
# Check for lint errors
uv run ruff check src/ tests/

# Auto-fix lint errors where possible
uv run ruff check src/ tests/ --fix

# Format code
uv run ruff format src/ tests/

# Check formatting without changing files
uv run ruff format src/ tests/ --check
```

The ruff configuration is in `pyproject.toml` and enables pycodestyle, pyflakes, isort, pyupgrade, flake8-bugbear, and flake8-simplify rules.

### CI

GitHub Actions runs lint checks and the full test suite on every push and pull request, across Python 3.10–3.13. See `.github/workflows/ci.yml`.

## License

TBD
