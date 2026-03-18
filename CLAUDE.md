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
│   └── felzenszwalb.py  # Felzenszwalb graph-based (scikit-image)
├── features/
│   ├── base.py          # BaseExtractor ABC
│   ├── spectral.py      # Per-band stats, NDVI/NDWI ratios (scipy.ndimage)
│   └── geometry.py      # Shape features (skimage.measure.regionprops)
├── classification/
│   ├── base.py          # BaseClassifier ABC with save/load (joblib)
│   ├── supervised.py    # Random Forest (sklearn)
│   ├── unsupervised.py  # K-Means (sklearn)
│   └── accuracy.py      # OA, kappa, confusion matrix, cross-validation
└── utils/
    ├── labels.py         # count, sizes, relabel, bounding slices
    └── vectorize.py      # rasterio.features.shapes -> GeoDataFrame
```

## Key Design Decisions

- **No `from osgeo import gdal`** — all raster I/O via rasterio
- **Large image support:** `read_raster_windows()` for tiled reading, `segment_tiled()` for tiled segmentation with overlap stitching
- **Segment labels:** int32 raster, 0=nodata, 1..N=segment IDs
- **Feature extraction:** scipy.ndimage for spectral stats (fast), skimage.regionprops for geometry
- **Band format:** (bands, height, width) throughout — matches rasterio convention

## Testing

- 58 tests covering all modules
- Synthetic 4-band 100x100 image with 4 quadrants (vegetation, urban, water, soil)
- Shared fixtures in `tests/conftest.py`
