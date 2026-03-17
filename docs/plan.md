# GeoOBIA: Open-Source Object-Based Image Analysis System

## Overall Plan

### Vision

Build an open-source Python system for object-based image analysis (OBIA) of geospatial imagery, inspired by eCognition (circa 2008-2011). The system provides a complete pipeline: **segment** imagery into objects, **extract** rich per-object features, and **classify** objects using supervised or unsupervised methods. It ships with a cross-platform GUI, a comprehensive Python API, and a CLI.

### Project Name

**GeoOBIA** (working name -- Geographic Object-Based Image Analysis)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     User Interfaces                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   CLI    │  │  Python API  │  │   GUI (Qt-based)  │  │
│  └────┬─────┘  └──────┬───────┘  └────────┬──────────┘  │
│       │               │                   │              │
│       └───────────────┼───────────────────┘              │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Core Engine (geoobia)                  │ │
│  │                                                     │ │
│  │  ┌────────────┐ ┌────────────┐ ┌─────────────────┐ │ │
│  │  │ Segmenters │ │ Extractors │ │  Classifiers    │ │ │
│  │  │            │ │            │ │                 │ │ │
│  │  │ - Shepherd │ │ - Spectral │ │ - Random Forest │ │ │
│  │  │ - SLIC     │ │ - Geometry │ │ - SVM           │ │ │
│  │  │ - SAM      │ │ - Texture  │ │ - K-Means       │ │ │
│  │  │ - Watershed│ │ - Embedding│ │ - Fuzzy Rules   │ │ │
│  │  │ - Felzensw.│ │ - Context  │ │ - Neural Net    │ │ │
│  │  └────────────┘ └────────────┘ └─────────────────┘ │ │
│  │                                                     │ │
│  │  ┌────────────┐ ┌────────────┐ ┌─────────────────┐ │ │
│  │  │  Project   │ │  Pipeline  │ │   I/O Layer     │ │ │
│  │  │  Manager   │ │  Engine    │ │  (rasterio/     │ │ │
│  │  │            │ │            │ │   geopandas)    │ │ │
│  │  └────────────┘ └────────────┘ └─────────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Segmentation Engine
**Goal:** Partition raster imagery into spatially contiguous, spectrally homogeneous objects.

Multiple algorithms behind a common interface:
- **Shepherd algorithm** (via RSGISLib or reimplemented) -- k-means seeded, iterative elimination
- **SLIC superpixels** (via scikit-image) -- fast, well-understood
- **Felzenszwalb** (via scikit-image) -- graph-based
- **Watershed** (via scikit-image) -- marker-based
- **Segment Anything Model (SAM)** (via segment-geospatial) -- foundation model approach

See: [segmentation.md](segmentation.md)

### 2. Feature Extraction Engine
**Goal:** Compute rich per-object attribute vectors for classification.

Feature categories:
- **Spectral:** mean, std, min, max, median per band; band ratios (NDVI etc.)
- **Geometric:** area, perimeter, compactness, elongation, rectangular fit
- **Texture:** GLCM-based (Haralick features) -- contrast, entropy, homogeneity, etc.
- **Contextual:** neighbor statistics, border relationships
- **Embedding:** aggregated per-object embeddings from foundation models (DINOv2, SAM)

See: [feature_extraction.md](feature_extraction.md)

### 3. Classification Engine
**Goal:** Assign class labels to objects using their feature vectors.

Approaches:
- **Supervised ML:** Random Forest, SVM, Gradient Boosting, KNN (via scikit-learn)
- **Unsupervised:** K-Means, DBSCAN, Gaussian Mixture (via scikit-learn)
- **Rule-based:** User-defined thresholds and fuzzy membership functions
- **Deep learning:** MLP classifier, optional CNN on per-object image chips

See: [classification.md](classification.md)

### 4. GUI Application
**Goal:** Cross-platform visual interface for the full OBIA workflow.

Two-track approach:
- **Primary:** QGIS plugin for users in the GIS ecosystem
- **Secondary:** Standalone PySide6 application for lightweight use

See: [gui.md](gui.md)

### 5. Python API and CLI
**Goal:** Scriptable, automatable access to all functionality.

See: [api_and_cli.md](api_and_cli.md)

---

## Data Model

### Segment Storage Strategy

We use a **dual representation** approach:

1. **Raster clumps** (primary, for computation) -- each pixel stores its segment ID. Fast for feature extraction (pixel-in-polygon is just an array lookup). Stored as GeoTIFF or KEA format.

2. **Vector polygons** (secondary, for display/export) -- GeoDataFrame with geometry + all attributes. Used for GUI display, spatial queries, and export. Stored as GeoPackage.

The system keeps both representations synchronized. The raster representation is the source of truth; vectors are derived on demand.

### Project Structure

A GeoOBIA project is a directory containing:
```
my_project/
├── project.json          # Project metadata, CRS, paths
├── imagery/              # Source raster files (or symlinks)
├── segments/             # Segmentation results (raster + vector)
├── features/             # Extracted feature tables (Parquet/GeoPackage)
├── classifications/      # Classification results
├── training/             # Training samples
└── pipelines/            # Saved pipeline definitions (JSON/YAML)
```

See: [data_model.md](data_model.md)

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Raster I/O | rasterio + GDAL | Standard, fast, all formats |
| Vector I/O | geopandas + shapely | Pythonic, well-maintained |
| Segmentation | scikit-image, RSGISLib, segment-geospatial | Best-of-breed per algorithm |
| Feature extraction | numpy, scipy, scikit-image (GLCM) | Fast, pure Python/C |
| Classification | scikit-learn | Comprehensive, standard |
| Embeddings | torch + transformers (optional) | Foundation model support |
| GUI framework | PySide6 (standalone) + PyQGIS (plugin) | Cross-platform, mature |
| CLI | click | Clean, well-documented |
| Data format | GeoTIFF, GeoPackage, Parquet, KEA | Interoperable standards |
| Packaging | conda (conda-forge) | Handles GDAL/PROJ dependencies |

---

## Development Phases

### Phase 1: Core Library (MVP)
**Goal:** Working Python API for segment-extract-classify pipeline.

1. Project structure and I/O layer (rasterio/geopandas wrappers)
2. Segmentation interface + 2 algorithms (SLIC, Felzenszwalb)
3. Spectral + geometric feature extraction
4. Classification with Random Forest (supervised) + K-Means (unsupervised)
5. Export results (GeoPackage, classified raster)
6. Basic CLI
7. Tests and documentation

**Deliverable:** `pip install geoobia` works; users can run a complete OBIA pipeline from Python or CLI.

### Phase 2: Algorithm Expansion
**Goal:** Full algorithm suite.

1. Add Shepherd segmentation (wrap RSGISLib or reimplement)
2. Add Watershed segmentation
3. Add SAM/segment-geospatial integration
4. Texture features (GLCM/Haralick)
5. Contextual features (neighbor relationships)
6. Additional classifiers (SVM, Gradient Boosting, rule-based)
7. Unsupervised classification algorithms
8. Pipeline engine (save/load/replay workflows)

### Phase 3: GUI -- QGIS Plugin
**Goal:** Visual interface inside QGIS.

1. Plugin skeleton with QGIS Processing integration
2. Segmentation tool with parameter panel + live preview
3. Feature extraction configuration UI
4. Classification workflow (sample selection, training, prediction)
5. Results visualization (color-coded segments)
6. Pipeline builder (leveraging QGIS Processing Modeler)

### Phase 4: Standalone GUI
**Goal:** Lightweight standalone app for users without QGIS.

1. PySide6 application shell with map viewer
2. Raster display with pan/zoom (tiled rendering)
3. Segment overlay visualization
4. Parameter panels mirroring QGIS plugin functionality
5. Packaging with conda-constructor (Windows/Mac/Linux installers)

### Phase 5: Advanced Features
**Goal:** Cutting-edge capabilities.

1. Foundation model embeddings as features (DINOv2, SAM encoder)
2. Multi-scale / hierarchical segmentation
3. Change detection between time periods
4. Batch processing / parallel execution
5. Large-scale tiled processing for huge datasets
6. Fuzzy membership function classification (eCognition-style)

---

## Key Design Decisions

### Why not just wrap RSGISLib?

RSGISLib is excellent but:
- Limited to the Shepherd algorithm for segmentation
- KEA format dependency for RAT-based workflows
- Unix-only (not tested on Windows)
- No GUI
- We want a pluggable architecture supporting many algorithms

We'll integrate RSGISLib as one backend but build a broader framework.

### Why dual GUI approach (QGIS + standalone)?

- **QGIS plugin** gets us to market fast -- raster display, vector overlay, Processing framework, and distribution via plugin repo are all free
- **Standalone app** serves users who don't want 500MB+ QGIS installed and want a simpler, focused UI
- Core logic is shared; only the UI layer differs

### Why raster clumps + vector polygons?

- Raster clumps enable O(1) pixel-to-segment lookup -- critical for fast feature extraction
- Vector polygons are needed for GUI display, spatial queries, and interoperability
- The RSGISLib paper demonstrated that raster-based attribution of 33.7M segments takes ~18 minutes vs hours for vector approaches

### Why conda for packaging?

- GDAL, PROJ, and other C libraries are notoriously difficult to pip install
- conda-forge has reliable cross-platform builds of the entire geospatial stack
- conda-constructor can create standalone installers (like napari does)

---

## Detailed Design Documents

- [segmentation.md](segmentation.md) -- Segmentation algorithms, interface design, parameters
- [feature_extraction.md](feature_extraction.md) -- Feature categories, computation approach, GLCM details
- [classification.md](classification.md) -- Classification methods, training workflow, accuracy assessment
- [gui.md](gui.md) -- GUI architecture, QGIS plugin design, standalone app design
- [api_and_cli.md](api_and_cli.md) -- Python API design, CLI commands, pipeline engine
- [data_model.md](data_model.md) -- Project structure, file formats, data flow
- [embeddings.md](embeddings.md) -- Foundation model integration, embedding extraction, usage patterns
