# geobia - Geographic Object-Based Image Analysis

## Quick Reference

- **Run tests:** `uv run pytest tests/ -v`
- **Run single test:** `uv run pytest tests/test_io.py -v`
- **Install deps:** `uv sync`
- **CLI:** `uv run geobia --help`

## Project Structure

```
src/geobia/
├── __init__.py          # Top-level API: segment(), extract_features(), classify()
├── cli.py               # Click CLI: segment, extract, classify, info, export
├── io/
│   ├── raster.py        # rasterio-based read/write with tiled windowed I/O
│   └── vector.py        # geopandas read/write, training sample loading
├── segmentation/
│   ├── base.py          # BaseSegmenter ABC
│   ├── slic.py          # SLIC superpixels (scikit-image)
│   ├── felzenszwalb.py  # Felzenszwalb graph-based (scikit-image)
│   ├── shepherd.py      # Shepherd K-means+elimination (numba JIT)
│   └── sam.py           # SAM via segment-geospatial (optional)
├── features/
│   ├── base.py          # BaseExtractor ABC
│   ├── spectral.py      # Per-band stats, NDVI/NDWI ratios (scipy.ndimage)
│   ├── geometry.py      # Shape features (skimage.measure.regionprops)
│   └── texture.py       # GLCM texture features (skimage graycomatrix)
├── classification/
│   ├── base.py          # BaseClassifier ABC with save/load (joblib)
│   ├── supervised.py    # Random Forest (sklearn)
│   ├── unsupervised.py  # K-Means, GMM, DBSCAN (sklearn)
│   └── accuracy.py      # OA, kappa, confusion matrix, cross-validation
├── pipeline/
│   ├── __init__.py      # Pipeline public API
│   └── engine.py        # Pipeline engine with JSON save/load, provenance
└── utils/
    ├── labels.py         # count, sizes, relabel, bounding slices
    └── vectorize.py      # rasterio.features.shapes -> GeoDataFrame
```

## Key Design Decisions

- **No `from osgeo import gdal`** — all raster I/O via rasterio
- **Large image support:** `read_raster_windows()` for tiled reading, `segment_tiled()` for tiled segmentation with overlap stitching
- **Segment labels:** int32 raster, 0=nodata, 1..N=segment IDs
- **Feature extraction:** scipy.ndimage for spectral stats (fast), skimage.regionprops for geometry, GLCM texture via skimage
- **Shepherd segmentation:** numba JIT for hot loops (connected components, iterative elimination with packed adjacency lists)
- **SAM segmentation:** optional dependency (`pip install geobia[sam]`), wraps segment-geospatial
- **Pipeline engine:** JSON-serializable pipeline definition with provenance tracking
- **Band format:** (bands, height, width) throughout — matches rasterio convention

## Testing

- **Run fast tests:** `uv run pytest tests/ -v -m "not spot_image and not slow"`
- **Run SPOT integration tests:** `uv run pytest tests/ -v -m spot_image` (requires `tests/data/SPOT_ROI.tif`)
- 85 fast tests + 13 SPOT/benchmark tests
- Synthetic 4-band 100x100 image with 4 quadrants (vegetation, urban, water, soil)
- Shared fixtures in `tests/conftest.py`
