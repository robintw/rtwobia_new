# Python API and CLI Design

## Python API

### Design Principles

1. **Simple things should be simple:** A complete OBIA workflow in <10 lines of code
2. **Complex things should be possible:** Full control over every parameter
3. **Composable:** Each function works independently and composes with others
4. **NumPy-native:** Core functions accept/return NumPy arrays
5. **GeoSpatial-aware:** I/O functions handle CRS, transforms, and metadata

### Package Structure

```
geoobia/
├── __init__.py              # Top-level convenience functions
├── io/
│   ├── __init__.py
│   ├── raster.py            # Read/write raster data (wraps rasterio)
│   └── vector.py            # Read/write vector data (wraps geopandas)
├── segmentation/
│   ├── __init__.py          # create() factory, convenience segment() function
│   ├── base.py              # BaseSegmenter interface
│   ├── slic.py              # SLIC superpixels
│   ├── felzenszwalb.py      # Felzenszwalb graph-based
│   ├── shepherd.py          # Shepherd algorithm
│   ├── watershed.py         # Watershed
│   ├── sam.py               # Segment Anything Model
│   └── postprocess.py       # Merge small, spectral merge, smooth
├── features/
│   ├── __init__.py          # extract() convenience function
│   ├── base.py              # BaseExtractor interface
│   ├── spectral.py          # Per-band statistics
│   ├── geometry.py          # Shape/size features
│   ├── texture.py           # GLCM/Haralick features
│   ├── context.py           # Neighbor/contextual features
│   └── embeddings.py        # Foundation model embeddings
├── classification/
│   ├── __init__.py          # classify() convenience function
│   ├── base.py              # BaseClassifier interface
│   ├── supervised.py        # Wrappers for sklearn classifiers
│   ├── unsupervised.py      # Wrappers for sklearn clustering
│   ├── rules.py             # Rule-based / fuzzy membership
│   └── accuracy.py          # Accuracy assessment
├── pipeline/
│   ├── __init__.py
│   ├── engine.py            # Pipeline execution engine
│   └── definitions.py       # Pipeline step definitions
├── project/
│   ├── __init__.py
│   └── manager.py           # Project create/load/save
└── utils/
    ├── __init__.py
    ├── labels.py             # Label array utilities
    ├── adjacency.py          # Segment adjacency graph
    └── vectorize.py          # Raster labels -> vector polygons
```

### Usage Examples

#### Minimal Example (5 lines)

```python
import geoobia

image, meta = geoobia.io.read_raster("satellite.tif")
labels = geoobia.segment(image, method="slic", n_segments=1000)
features = geoobia.extract_features(image, labels, categories=["spectral", "geometry"])
classes = geoobia.classify(features, method="kmeans", n_clusters=5)
geoobia.io.write_vector("result.gpkg", labels, classes, meta)
```

#### Full Control Example

```python
import geoobia
from geoobia.segmentation import create as create_segmenter
from geoobia.features import SpectralExtractor, GeometryExtractor, TextureExtractor
from geoobia.classification import SupervisedClassifier

# Load data
image, meta = geoobia.io.read_raster("satellite.tif")

# Segment with Shepherd algorithm
segmenter = create_segmenter("shepherd", num_clusters=60, min_n_pxls=100)
labels = segmenter.segment(image)

# Extract features
spectral = SpectralExtractor(bands={"red": 2, "nir": 3}).extract(image, labels)
geometry = GeometryExtractor().extract(image, labels)
texture = TextureExtractor(distances=[1, 2], bands=[0, 1, 2]).extract(image, labels)
features = spectral.join(geometry).join(texture)

# Train classifier with training samples
training_labels = geoobia.io.read_training_samples("training.gpkg", labels)
classifier = SupervisedClassifier("random_forest", n_estimators=200)
classifier.fit(features.loc[training_labels.index], training_labels)

# Predict and assess accuracy
predictions = classifier.predict(features)
accuracy = classifier.validate(features, training_labels, method="cross_val", cv=5)
print(accuracy.report())

# Export
geoobia.io.write_vector("classified.gpkg", labels, predictions, meta)
geoobia.io.write_raster("classified.tif", labels, predictions, meta)
classifier.save("model.joblib")
```

#### Pipeline Example

```python
import geoobia

pipeline = geoobia.Pipeline([
    ("segment", "slic", {"n_segments": 1000}),
    ("extract", ["spectral", "geometry", "texture"], {"bands": {"nir": 3}}),
    ("classify", "random_forest", {"n_estimators": 100}),
])

# Save pipeline definition
pipeline.save("my_workflow.json")

# Load and run on new data
pipeline = geoobia.Pipeline.load("my_workflow.json")
result = pipeline.run("new_image.tif", training="samples.gpkg")
result.export("output.gpkg")
```

### Key API Design Details

#### I/O Layer

```python
# Raster I/O
image, meta = geoobia.io.read_raster("path.tif")
# image: np.ndarray, shape (bands, height, width)
# meta: dict with crs, transform, nodata, dtype, bounds

geoobia.io.write_raster("path.tif", data, meta)

# Vector I/O
gdf = geoobia.io.read_vector("path.gpkg")
geoobia.io.write_vector("path.gpkg", labels, attributes, meta)

# Training samples
samples = geoobia.io.read_training_samples("samples.gpkg", labels)
# Returns: pd.Series mapping segment_id -> class_label
# Segments overlapping training polygons are labeled
```

#### Label Utilities

```python
from geoobia.utils import labels as lbl

# Basic info
n_segments = lbl.count(labels)
sizes = lbl.sizes(labels)  # dict: segment_id -> pixel_count

# Vectorize
polygons = lbl.vectorize(labels, meta["transform"], meta["crs"])
# Returns: GeoDataFrame with segment_id and geometry

# Rasterize (opposite direction)
labels = lbl.rasterize(polygons, meta["transform"], shape)

# Adjacency
adj = lbl.adjacency(labels)
# Returns: dict[int, set[int]] mapping segment_id -> neighbor_ids
```

---

## CLI Design

### Command Structure

```
geoobia <command> [options]
```

Top-level commands follow the OBIA workflow:

```
geoobia segment    -- Segment an image
geoobia extract    -- Extract features from segments
geoobia classify   -- Classify segments
geoobia export     -- Export results
geoobia pipeline   -- Run a saved pipeline
geoobia info       -- Show info about a dataset
geoobia project    -- Manage projects
```

### Command Examples

#### Segment

```bash
# Basic segmentation
geoobia segment input.tif -o segments.tif --method slic --n-segments 1000

# Shepherd with custom parameters
geoobia segment input.tif -o segments.tif \
    --method shepherd \
    --num-clusters 60 \
    --min-size 100 \
    --bands 1,2,3,4

# With preview (segment small region, display stats)
geoobia segment input.tif -o segments.tif --method felzenszwalb --preview
```

#### Extract Features

```bash
# Extract all feature categories
geoobia extract input.tif segments.tif -o features.parquet --all

# Specific categories
geoobia extract input.tif segments.tif -o features.parquet \
    --spectral --geometry --texture \
    --band-names red,green,blue,nir

# With band ratios
geoobia extract input.tif segments.tif -o features.parquet \
    --spectral --geometry \
    --ratios "ndvi=(nir-red)/(nir+red)"
```

#### Classify

```bash
# Supervised classification
geoobia classify features.parquet \
    --training samples.gpkg \
    --method random-forest \
    --n-estimators 200 \
    -o classified.gpkg

# Unsupervised classification
geoobia classify features.parquet \
    --method kmeans \
    --n-clusters 8 \
    -o classified.gpkg

# Rule-based classification
geoobia classify features.parquet \
    --rules rules.yaml \
    -o classified.gpkg
```

#### Pipeline

```bash
# Run a saved pipeline
geoobia pipeline run workflow.json --input image.tif --training samples.gpkg -o result.gpkg

# Run on multiple images (batch)
geoobia pipeline batch workflow.json --input-dir images/ --training samples.gpkg --output-dir results/
```

#### Info

```bash
# Image info
geoobia info image.tif
# Output: dimensions, bands, CRS, resolution, bounds

# Segmentation info
geoobia info segments.tif
# Output: number of segments, size statistics, algorithm used

# Feature info
geoobia info features.parquet
# Output: number of features, feature names, basic statistics
```

### CLI Implementation

Using the `click` library:

```python
import click

@click.group()
@click.version_option()
def cli():
    """GeoOBIA: Object-Based Image Analysis for Geospatial Imagery."""

@cli.command()
@click.argument("input_image")
@click.option("-o", "--output", required=True, help="Output segments file")
@click.option("--method", default="slic", help="Segmentation algorithm")
@click.option("--n-segments", type=int, default=1000)
# ... more options per algorithm
def segment(input_image, output, method, n_segments, **kwargs):
    """Segment an image into objects."""
    ...
```

---

## Pipeline Engine

### Definition Format

Pipelines are defined in JSON or YAML:

```json
{
    "name": "Land Cover Classification",
    "version": "1.0",
    "steps": [
        {
            "name": "segment",
            "type": "segmentation",
            "algorithm": "shepherd",
            "params": {
                "num_clusters": 60,
                "min_n_pxls": 100
            }
        },
        {
            "name": "extract",
            "type": "feature_extraction",
            "categories": ["spectral", "geometry", "texture"],
            "params": {
                "bands": {"red": 2, "green": 1, "blue": 0, "nir": 3},
                "texture_distances": [1, 2]
            }
        },
        {
            "name": "classify",
            "type": "classification",
            "algorithm": "random_forest",
            "params": {
                "n_estimators": 200,
                "max_features": "sqrt"
            }
        }
    ]
}
```

### Pipeline Execution

The engine:
1. Validates the pipeline definition
2. Checks that inputs exist and parameters are valid
3. Executes steps in order, passing outputs as inputs to the next step
4. Caches intermediate results to disk (can resume from any step)
5. Logs progress and timing
6. Saves provenance metadata (parameters, versions, timestamps)

### Reproducibility

Every pipeline run produces a provenance record:
```json
{
    "pipeline": "land_cover.json",
    "input": "image.tif",
    "timestamp": "2026-03-17T10:30:00Z",
    "geoobia_version": "0.1.0",
    "steps": [
        {"name": "segment", "duration_s": 45.2, "n_segments": 4523},
        {"name": "extract", "duration_s": 12.1, "n_features": 47},
        {"name": "classify", "duration_s": 3.4, "accuracy": 0.91}
    ]
}
```
