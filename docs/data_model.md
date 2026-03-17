# Data Model Design

## Overview

The data model defines how imagery, segments, features, and classification results are stored and related. It prioritizes interoperability (standard formats), performance (fast per-segment operations), and simplicity (minimal custom formats).

## Core Data Entities

### 1. Imagery (Input)

Multi-band geospatial raster data.

**Format:** Any GDAL-supported format. GeoTIFF is the primary target. Cloud-Optimized GeoTIFF (COG) supported for remote/large datasets.

**In-memory representation:** NumPy array, shape `(bands, height, width)`, with metadata dict containing CRS, affine transform, nodata value, band names/descriptions.

**Requirements:**
- All bands must have the same spatial extent and resolution
- A valid CRS (coordinate reference system) should be present
- Nodata values should be defined (default: 0 or NaN)

### 2. Segments (Labels)

Integer raster where each pixel's value is its segment ID.

**Raster representation (primary):**
- Format: GeoTIFF (uint32) or KEA
- Value 0 = nodata/no segment
- Values 1..N = segment IDs (sequential, no gaps)
- Same spatial extent, resolution, and CRS as source imagery
- Stored with overviews/pyramids for fast visualization

**Vector representation (derived):**
- Format: GeoPackage
- One polygon per segment, with `segment_id` attribute
- Generated on demand from raster labels via `rasterio.features.shapes`
- Cached to disk after first generation

**Why dual representation:**
- **Raster for computation:** Pixel-to-segment lookup is O(1) -- just read the pixel value. Feature extraction iterates over pixels and accumulates per-segment. RSGISLib demonstrated 33.7M segments attributed in 18 minutes using raster approach.
- **Vector for display/export:** GIS software expects polygons for overlay/styling. Spatial queries need vector geometry. Export formats (Shapefile, GeoPackage) are vector.

### 3. Features (Attributes)

Per-segment attribute table with computed features.

**In-memory:** pandas DataFrame with `segment_id` as index, feature values as columns.

**On-disk:** Apache Parquet (columnar, compressed, fast). Also exportable as:
- CSV (universal but slow/large)
- GeoPackage attribute table (with geometry if needed)

**Schema example:**
```
segment_id | mean_red | mean_green | mean_blue | mean_nir | std_red | ... | area | perimeter | compactness | ndvi | glcm_entropy_b0 | ...
1          | 0.234    | 0.312      | 0.189     | 0.567    | 0.045   | ... | 2340 | 245.3     | 0.78        | 0.41 | 2.34            | ...
2          | 0.178    | 0.256      | 0.145     | 0.612    | 0.032   | ... | 1890 | 198.7     | 0.82        | 0.55 | 1.98            | ...
```

**Feature metadata** (sidecar JSON):
```json
{
    "features": {
        "mean_red": {"category": "spectral", "band": "red", "unit": "reflectance"},
        "area": {"category": "geometry", "unit": "m^2"},
        "compactness": {"category": "geometry", "unit": "ratio", "range": [0, 1]},
        "glcm_entropy_b0": {"category": "texture", "band": 0, "distance": 1}
    },
    "n_segments": 4523,
    "extraction_params": { ... }
}
```

### 4. Training Samples

User-provided class labels for a subset of segments.

**Input formats:**
- **Vector polygons:** GeoPackage/Shapefile with a `class` column. Segments overlapping each polygon are assigned that class.
- **Point samples:** GeoPackage with points. The segment containing each point is assigned that class.
- **CSV:** Mapping `segment_id` to `class_label`.
- **Interactive:** Selected via GUI (stored as GeoPackage).

**In-memory:** pandas Series mapping `segment_id -> class_label`.

### 5. Classification Results

Class labels and probabilities per segment.

**In-memory:** DataFrame with columns:
- `class_label`: Predicted class (string or int)
- `class_prob_{name}`: Probability for each class (float, 0-1)
- `confidence`: Max probability across classes

**On-disk:**
- Added as columns to the feature Parquet file
- Classified vector: GeoPackage with geometry + class_label + probabilities
- Classified raster: GeoTIFF where pixel values are class codes (remapped from segment labels)

### 6. Trained Model

Serialized classifier for reuse.

**Format:** joblib (sklearn standard). Includes:
- Trained model object
- Feature names used for training
- Class names and codes
- Training parameters
- Accuracy metrics from validation

## Project Structure

A GeoOBIA project groups all data for an analysis:

```
my_project/
├── project.json              # Project metadata
├── imagery/
│   └── satellite.tif         # Source imagery (or symlinks)
├── segments/
│   ├── slic_1000.tif         # Segment raster (can have multiple)
│   └── slic_1000.gpkg        # Cached vector representation
├── features/
│   ├── slic_1000_features.parquet  # Feature table
│   └── slic_1000_features.json     # Feature metadata
├── training/
│   └── samples.gpkg          # Training samples
├── classifications/
│   ├── rf_v1.gpkg            # Classification result
│   ├── rf_v1_model.joblib    # Trained model
│   └── rf_v1_accuracy.json   # Accuracy report
├── pipelines/
│   └── workflow.json          # Pipeline definition
└── provenance/
    └── run_20260317_103000.json  # Execution provenance
```

### project.json

```json
{
    "name": "My OBIA Project",
    "created": "2026-03-17T10:00:00Z",
    "crs": "EPSG:32633",
    "imagery": {
        "main": "imagery/satellite.tif",
        "bands": {
            "blue": 0, "green": 1, "red": 2, "nir": 3
        }
    },
    "active_segmentation": "segments/slic_1000.tif",
    "active_classification": "classifications/rf_v1.gpkg"
}
```

Users don't have to use the project system -- all functions accept plain file paths. The project is a convenience layer for the GUI.

## Data Flow

```
                    ┌─────────┐
                    │  Input  │
                    │  Image  │
                    │ (bands, │
                    │  h, w)  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ Segment │──────────────────┐
                    │         │                  │
                    └────┬────┘                  │
                         │                       │
              ┌──────────▼──────────┐     ┌──────▼──────┐
              │  Segment Labels     │     │  Vectorized │
              │  (h, w) uint32     │     │  Polygons   │
              │  raster             │     │  GeoPackage │
              └──────────┬──────────┘     └─────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐   ┌──────▼─────┐  ┌──────▼──────┐
    │Spectral │   │ Geometric  │  │  Texture    │
    │Features │   │ Features   │  │  Features   │
    └────┬────┘   └──────┬─────┘  └──────┬──────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                  ┌──────▼──────┐
                  │  Feature    │
                  │  Table      │
                  │  (DataFrame)│
                  └──────┬──────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
       ┌──────▼────┐ ┌──▼──┐ ┌────▼─────┐
       │Supervised │ │Unsu-│ │Rule-     │
       │ML         │ │perv.│ │Based     │
       └──────┬────┘ └──┬──┘ └────┬─────┘
              │          │         │
              └──────────┼─────────┘
                         │
                  ┌──────▼──────┐
                  │  Classified │
                  │  Segments   │
                  └──────┬──────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
       ┌──────▼────┐ ┌──▼──────┐ ┌─▼────────┐
       │ GeoPackage│ │Classified│ │ Accuracy │
       │ (vector)  │ │Raster   │ │ Report   │
       └───────────┘ └─────────┘ └──────────┘
```

## File Format Rationale

| Format | Used For | Why |
|--------|----------|-----|
| GeoTIFF | Imagery, segment rasters, classified rasters | Universal, GDAL-native, COG support, overviews |
| GeoPackage | Segment vectors, training samples, classified vectors | Open standard, replaces Shapefile, single file, no limitations |
| Parquet | Feature tables | Fast columnar reads, compressed, supports millions of rows, pandas-native |
| JSON | Project metadata, pipeline definitions, provenance, feature metadata | Human-readable, universally parseable |
| joblib | Trained models | sklearn standard, handles NumPy arrays efficiently |

### Why not KEA format?

KEA (HDF5-based) with embedded RAT is RSGISLib's preferred format and has advantages for attribute-heavy raster workflows. However:
- Limited ecosystem support outside RSGISLib
- GeoTIFF is universally supported
- We separate segment raster (GeoTIFF) from attributes (Parquet) for flexibility
- Users working with RSGISLib can still use KEA as an intermediate format

### Why Parquet for features?

- **Columnar:** Reading a single feature across all segments is fast (important for classification with feature selection)
- **Compressed:** 5-10x smaller than CSV
- **Typed:** Preserves dtypes without parsing
- **Scalable:** Handles millions of segments without issues
- **Ecosystem:** pandas, polars, DuckDB, Arrow all read Parquet natively
