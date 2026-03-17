# GUI Design

## Overview

The GUI provides a visual interface for the full OBIA workflow. We pursue two tracks:

1. **QGIS Plugin (Phase 3):** For GIS professionals. Leverages QGIS's raster display, vector overlay, symbology, and Processing framework.
2. **Standalone Application (Phase 4):** For non-GIS users. Lighter weight, focused UX, simpler installation.

Both share the same core library; only the UI layer differs.

## QGIS Plugin Design

### Architecture

```
qgis_plugin/
├── __init__.py               # Plugin entry point
├── metadata.txt              # QGIS plugin metadata
├── geoobia_plugin.py         # Main plugin class (toolbar, menus, dock)
├── ui/
│   ├── segmentation_panel.py # Segmentation parameter dock widget
│   ├── features_panel.py     # Feature extraction configuration
│   ├── classification_panel.py # Classification workflow
│   ├── results_panel.py      # Results visualization and export
│   └── sample_selector.py    # Training sample selection tool
├── processing/
│   ├── provider.py           # QGIS Processing provider
│   ├── segmentation_alg.py   # Segmentation as Processing algorithm
│   ├── features_alg.py       # Feature extraction algorithm
│   └── classification_alg.py # Classification algorithm
└── resources/
    ├── icons/
    └── styles/               # Default QGIS symbology styles
```

### Key UI Components

#### 1. GeoOBIA Dock Widget (main panel)

A dockable panel with a tabbed interface:

**Segmentation Tab:**
- Algorithm selector (dropdown: SLIC, Felzenszwalb, Shepherd, SAM, Watershed)
- Dynamic parameter panel (auto-generated from algorithm's param schema)
- Scale slider (normalized 0-1, maps to algorithm-specific size control)
- Band selection checkboxes
- "Preview" button (segments a small viewport region for fast feedback)
- "Run" button (full segmentation)
- Progress bar

**Features Tab:**
- Checkbox groups for feature categories (Spectral, Geometry, Texture, Context, Embedding)
- Expandable sections showing individual features within each category
- Band mapping configuration (which band is red, green, NIR, SWIR)
- Custom band ratio editor
- "Extract" button

**Classification Tab:**
- Mode selector: Supervised / Unsupervised / Rule-Based
- Supervised panel:
  - Class list (add/remove/rename classes, assign colors)
  - "Select Samples" map tool (click segments to assign to classes)
  - Algorithm selector (Random Forest, SVM, etc.)
  - Feature selector (checklist or "use all")
  - Train / Predict / Validate buttons
  - Accuracy metrics display
- Unsupervised panel:
  - Algorithm selector (K-Means, GMM, DBSCAN)
  - Number of clusters
  - Cluster / Assign Names buttons
- Rule-based panel:
  - Rule editor (text-based with syntax highlighting)
  - Fuzzy function visual editor (drag control points)
  - Apply / Preview buttons

**Results Tab:**
- Classification summary table (class name, count, area)
- Export options (GeoPackage, Shapefile, GeoTIFF, CSV)
- Accuracy report viewer

#### 2. Sample Selection Map Tool

A custom QGIS map tool that:
- Highlights segments on hover (thicker outline)
- Click to select a segment
- Assigns to the currently active class in the classification panel
- Selected samples are shown with class-colored fill
- Right-click to remove a sample

#### 3. Segment Visualization

Leverage QGIS's built-in symbology engine:
- **Outline mode:** Transparent fill, colored outlines over imagery
- **Classification mode:** Categorized renderer, one color per class
- **Feature mode:** Graduated renderer, color-ramp based on any feature value
- **Probability mode:** Graduated renderer showing classification confidence

Apply styles programmatically via PyQGIS:
```python
# Categorized renderer for classification results
categories = []
for class_name, color in class_colors.items():
    symbol = QgsSymbol.defaultSymbol(layer.geometryType())
    symbol.setColor(QColor(color))
    categories.append(QgsRendererCategory(class_name, symbol, class_name))
renderer = QgsCategorizedSymbolRenderer("class_label", categories)
layer.setRenderer(renderer)
```

#### 4. Processing Framework Integration

Register all algorithms with QGIS Processing so they can be:
- Used in the QGIS graphical modeler (visual pipeline builder)
- Run in batch mode
- Called from the Python console
- Chained with other QGIS/GDAL/GRASS/OTB tools

This gives us a **pipeline builder for free** via the QGIS Processing Modeler.

### Installation

Distributed via the QGIS Plugin Repository:
1. User opens QGIS Plugin Manager
2. Searches for "GeoOBIA"
3. Clicks Install
4. Plugin appears in toolbar

Dependencies (geoobia core library, scikit-learn, etc.) installed via pip from within the plugin, or bundled.

---

## Standalone Application Design

### Architecture

```
standalone/
├── main.py                   # Application entry point
├── app.py                    # QApplication setup
├── main_window.py            # QMainWindow with layout
├── widgets/
│   ├── map_viewer.py         # Raster/vector display widget
│   ├── layer_panel.py        # Layer list (imagery, segments, classes)
│   ├── workflow_panel.py     # Step-by-step workflow panel
│   ├── params_panel.py       # Dynamic parameter editor
│   ├── feature_table.py      # Feature DataFrame viewer
│   ├── results_panel.py      # Classification results
│   └── rule_editor.py        # Rule-based classification editor
├── tools/
│   ├── pan_tool.py           # Pan/scroll map interaction
│   ├── zoom_tool.py          # Zoom interaction
│   ├── select_tool.py        # Select segments
│   └── sample_tool.py        # Assign training samples
└── resources/
    ├── icons/
    └── styles.qss            # Qt stylesheet
```

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Menu Bar  │  Toolbar (pan, zoom, select, sample)       │
├────────────┼────────────────────────────────┬────────────┤
│            │                                │            │
│  Layer     │                                │  Workflow  │
│  Panel     │       Map Viewer               │  Panel     │
│            │   (raster + segment overlay)    │            │
│  - Imagery │                                │  1. Load   │
│  - Segments│                                │  2. Segment│
│  - Classes │                                │  3. Extract│
│            │                                │  4. Classify│
│            │                                │  5. Export │
│            │                                │            │
├────────────┴────────────────────────────────┴────────────┤
│  Feature Table / Results Panel (tabbed, collapsible)     │
└─────────────────────────────────────────────────────────┘
```

### Map Viewer Implementation

The map viewer is the most complex standalone component. Options:

**Option A: QGraphicsView with tiled rendering (recommended)**
- Use Qt's QGraphicsView framework
- Render raster tiles at multiple zoom levels (pyramid/overview support)
- Draw segment boundaries as QGraphicsPathItems
- Color-code segments by painting filled polygons with transparency
- Smooth pan/zoom with mouse wheel and click-drag

**Option B: Embedded web map (Leaflet/MapLibre)**
- Use QWebEngineView with a web map library
- Good for basemap integration but adds complexity
- Harder to integrate with raster segment overlays

**Option C: Embed QGIS map canvas (QgsMapCanvas)**
- Full QGIS rendering power without the full QGIS UI
- Requires QGIS libraries as a dependency (~adds 300MB+)
- Best rendering quality but heaviest dependency

Recommendation: Start with **Option A** for minimal dependencies. Can upgrade to Option C later if rendering quality or CRS handling becomes a bottleneck.

### Tiled Raster Rendering

For the standalone viewer to handle large rasters:
1. On load, generate image pyramids (overviews) if not present
2. Determine visible viewport in image coordinates
3. Select appropriate pyramid level for current zoom
4. Read only the visible tile(s) via rasterio windowed I/O
5. Render as QPixmap on the QGraphicsScene
6. Cache recently viewed tiles for smooth panning

### Packaging

Using conda-constructor (following the napari model):
- Creates platform-native installers (.exe, .pkg, .sh)
- Bundles Python, all dependencies (including GDAL, PROJ)
- No prior Python knowledge needed from end users
- Target installer size: 200-400MB

---

## Shared UI Patterns

Both QGIS plugin and standalone app share these patterns:

### Dynamic Parameter Panels

Auto-generated from each algorithm's parameter schema (JSON Schema):
- Numeric parameters: spinbox with min/max/step from schema
- Choice parameters: dropdown
- Boolean parameters: checkbox
- With tooltips from schema descriptions

### Workflow-Guided Interface

A step-by-step workflow panel guides users through:
1. **Load** imagery
2. **Segment** with chosen algorithm and parameters
3. **Extract** features
4. **Classify** (supervised, unsupervised, or rule-based)
5. **Validate** accuracy
6. **Export** results

Each step enables the next; users can go back and re-run any step.

### Feature Histogram Viewer

When the user selects a feature, show a histogram of its values across all segments, optionally colored by class. This helps with:
- Understanding feature distributions
- Setting thresholds for rule-based classification
- Identifying discriminating features
- Quality-checking the segmentation
