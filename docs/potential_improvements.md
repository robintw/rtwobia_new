# Potential Improvements

Ideas for new features, enhancements, tests, UI improvements, and other work.

---

## New Features

### Segmentation

1. **Region-growing segmentation** — Seed-based region growing algorithm. Start from seed points (manually placed or auto-detected), grow regions based on spectral similarity threshold. Useful for targeted extraction of specific land cover types.

2. **Deep learning segmentation backends** — Beyond SAM, support other models: U-Net, DeepLabV3, SegFormer. Wrap pre-trained models from torchgeo or similar libraries behind the BaseSegmenter interface.

3. **Segmentation comparison tool** — Side-by-side or overlay comparison of two segmentation runs on the same image. Compute metrics like over-segmentation rate, under-segmentation rate, boundary recall, and the RASI (Relative Area of Smallest Intersecting segment) index.

4. **Interactive segmentation refinement** — Let users merge or split segments manually in QGIS. Click two adjacent segments to merge, or draw a line through a segment to split it. Update the labels array in-place.

5. **Superpixel boundary smoothing** — Post-processing step to smooth jagged segment boundaries using morphological operations or Douglas-Peucker simplification. Reduces polygon complexity for export.

6. **Minimum mapping unit enforcement** — After any segmentation, merge segments smaller than a user-defined area threshold into their most spectrally similar neighbor. Already partially implemented in Shepherd; generalize to all methods.

### Feature Extraction

7. **Temporal features** — Extract features from multi-temporal image stacks. Compute per-segment temporal statistics: trend, seasonality amplitude, date of maximum/minimum, inter-annual variability. Critical for agricultural monitoring.

8. **LiDAR / DSM height features** — Accept a secondary raster (DSM/DTM) and extract per-segment height statistics: mean height, max height, height variance, canopy height model metrics. Essential for urban and forestry applications.

9. **Deep learning embeddings** — Extract per-segment embeddings from pre-trained vision models (CLIP, DINOv2, SSL4EO). High-dimensional feature vectors that capture semantic content beyond hand-crafted spectral/texture features.

10. **Edge / boundary features** — Per-segment boundary characteristics: mean gradient magnitude along boundary, boundary roughness (fractal dimension), contrast with each neighboring segment by spectral band.

11. **Zonal statistics from external rasters** — Accept additional raster layers (soil type, slope, aspect, distance-to-road) and compute zonal statistics per segment. Common in landscape ecology workflows.

12. **Feature correlation matrix** — After extraction, compute and display a correlation matrix heatmap. Highlight highly correlated features (r > 0.95) and offer to remove redundant ones. Reduces classifier overfitting.

13. **PCA / dimensionality reduction** — Offer PCA, t-SNE, or UMAP as a post-extraction step. Reduce high-dimensional feature spaces to 2-3 components for visualization or as input to clustering.

### Classification

14. **Active learning workflow** — Semi-automated training sample selection. After initial classification, identify segments where the classifier is least confident. Present them to the user for labeling. Iterate until convergence.

15. **Ensemble classification** — Combine predictions from multiple classifiers (RF + SVM + GB) using majority voting, weighted voting, or stacking. Often outperforms any single classifier.

16. **Deep learning classifiers** — MLP, 1D-CNN, or transformer-based classifiers on the feature vectors. May capture non-linear relationships that tree-based methods miss.

17. **Class hierarchy / ontology** — Define a class hierarchy (e.g., vegetation > deciduous > oak). Classify at multiple levels. Support hierarchical accuracy assessment.

18. **Semi-supervised classification** — Use a small set of labeled samples plus a large pool of unlabeled segments. Algorithms: label propagation, self-training, consistency regularization.

19. **Transfer learning between images** — Train a classifier on one image, apply it to another (different date, sensor, or location). Include domain adaptation to handle spectral shifts.

20. **Spatial post-processing** — After classification, apply spatial rules: minimum class area, remove isolated pixels, enforce spatial contiguity. Sieve filter on the classified raster.

### Change Detection

21. **Multi-date change detection** — Extend from 2-date to N-date time series analysis. Detect gradual changes (urbanization, deforestation) vs. abrupt events (fire, flood).

22. **Object-based change detection** — Instead of comparing features at fixed segments, detect segments that appear, disappear, split, or merge between dates. Track object lifecycle.

23. **Change attribution** — After detecting change, classify the type of change: vegetation loss, new construction, water body expansion, etc. Output a change type map.

### I/O and Data

24. **Cloud-optimized GeoTIFF (COG) support** — Read and write COGs natively. Support reading from HTTP/S3 URLs via rasterio's virtual filesystem.

25. **STAC catalog integration** — Search and load imagery from STAC catalogs (Microsoft Planetary Computer, Element 84). Automate the data acquisition step.

26. **Vector tile export** — Export classification results as Mapbox Vector Tiles (MVT) or PMTiles for web visualization.

27. **GeoParquet export** — Export results as GeoParquet with geometry, features, and classification in a single file. More efficient than GeoPackage for large datasets.

---

## Improvements to Existing Features

### Performance

28. **Parallel feature extraction** — Extract features from multiple segments concurrently using multiprocessing or joblib. Currently single-threaded; the segment loop is embarrassingly parallel.

29. **Dask integration for large images** — Support dask arrays for out-of-core processing. Extract features tile-by-tile without loading the full image into memory.

30. **GPU acceleration for GLCM** — GLCM texture computation is the bottleneck for large images. Port to CuPy or use cucim for GPU-accelerated co-occurrence matrices.

31. **Cached feature extraction** — If the user re-runs extraction with the same parameters, skip computation and return cached results. Use content-addressed hashing of (image + labels + params) as the cache key.

32. **Memory-mapped label arrays** — For very large images, keep the labels array memory-mapped (numpy.memmap) instead of fully loaded. The segmentation panel already evicts to disk; extend this to feature extraction.

### Robustness

33. **CRS validation and reprojection** — When loading training samples, auto-reproject if CRS doesn't match the raster. Currently silently produces wrong results if CRS differs.

34. **Nodata propagation** — Track nodata more carefully through the pipeline. Currently, some features produce NaN for nodata segments but others produce zeros. Standardize on NaN and add a `valid_fraction` feature per segment.

35. **Input validation at pipeline boundaries** — The pipeline engine doesn't validate that step outputs match next step inputs. Add schema validation between steps.

36. **Atomic file writes** — Write output files to a temporary location first, then rename. Prevents corrupted output if the process is killed mid-write.

37. **Deterministic segmentation** — SLIC and Felzenszwalb produce slightly different results on different runs due to floating-point ordering. Add explicit random seed parameters for reproducibility.

### API and Architecture

38. **Plugin registry for custom extractors/classifiers** — Let users register their own BaseExtractor or BaseClassifier subclasses at runtime via entry points or a register() function, without modifying geobia source.

39. **Streaming pipeline** — Instead of materializing the full features DataFrame in memory, stream features through the pipeline segment-by-segment. Reduces peak memory usage.

40. **Async pipeline with progress** — Make the pipeline engine async-aware. Report progress per step. Support cancellation. Currently blocks until completion.

41. **Type stubs / py.typed marker** — Add py.typed marker and inline type annotations throughout. Enables mypy checking and IDE autocompletion for library users.

---

## Testing

42. **Property-based testing** — Use Hypothesis to generate random images, labels, and feature values. Test invariants like "spectral mean is always between min and max" or "segment count is always positive."

43. **Regression test suite** — For each segmentation method, store expected output for a fixed input image. Detect when algorithm updates change results unexpectedly.

44. **Cross-platform CI** — Test on Windows and macOS in addition to Linux. Rasterio and GDAL behave differently on different platforms.

45. **Performance benchmarks in CI** — Track feature extraction and segmentation time per commit. Alert on regressions > 20%. Use pytest-benchmark.

46. **Fuzz testing for I/O** — Feed malformed GeoTIFFs, truncated files, and invalid CRS strings to read_raster and read_vector. Ensure graceful errors, not crashes.

47. **QGIS plugin integration tests** — Automated tests that load the plugin in a headless QGIS instance, run the workflow, and verify layer outputs. Use qgis_sketcher or pytest-qgis.

48. **Coverage enforcement** — Add a minimum coverage threshold (e.g., 80%) to CI. The current 236 tests have good coverage but some modules (batch, change, pipeline) are lighter.

49. **Snapshot testing for CLI output** — Use syrupy or similar to snapshot the output of `geobia info` and other CLI commands. Detect formatting regressions.

---

## UI / QGIS Plugin

50. **Undo/redo for training samples** — The sample selector modifies training_samples in-place with no undo. Add an undo stack (QUndoStack) so users can ctrl+z sample assignments.

51. **Keyboard shortcuts** — Add keyboard shortcuts for common actions: R to run segmentation, E to extract, C to classify. Configurable in QGIS's keyboard settings.

52. **Feature scatter plot** — In the Features Explorer, add a scatter plot mode: select two features and plot them against each other, colored by class. Useful for understanding feature separability.

53. **Feature importance visualization on map** — After supervised classification, color each segment by its most discriminating feature. Shows which features drive the classification spatially.

54. **Segment profile tool** — Click a segment and see its full feature profile: a radar/spider chart or bar chart of all normalized feature values. Compare profiles between segments.

55. **Comparison view for segmentation parameters** — Side-by-side map view showing two segmentation runs simultaneously. Help users tune parameters by seeing the effect immediately.

56. **Training sample import/export** — Save and load training samples as GeoPackage. Currently training samples exist only in plugin state and are lost when QGIS closes.

57. **Progress estimation** — Replace the linear progress bar with time-remaining estimates. Use historical timing data to predict completion.

58. **Dark mode support** — The plugin uses hardcoded colors (yellow outlines, white backgrounds). Detect QGIS theme and adjust colors accordingly.

59. **Layer management cleanup** — The plugin creates multiple layers (segments, features, results, training samples). Add a "Remove all GeoOBIA layers" button. Track all created layers in PluginState.

60. **Drag-and-drop workflow** — Let users drag a raster file from their file manager directly onto the plugin dock to set it as the input layer.

61. **Multi-band choropleth** — In the Features Explorer, show RGB composite of three selected features as segment fill colors. Useful for visualizing multi-dimensional feature spaces.

62. **Tooltip preview in segmentation gallery** — When hovering over a run in the gallery, show a small thumbnail preview of the segmentation result.

63. **Classification confusion matrix heatmap** — Replace the text-based confusion matrix in Results with an interactive matplotlib heatmap. Click a cell to highlight those segments on the map.

---

## Documentation

64. **Tutorial notebooks** — Jupyter notebooks walking through common workflows: urban land cover mapping, agricultural field delineation, change detection. Include real data.

65. **Algorithm comparison guide** — Document when to use each segmentation algorithm, with example images showing typical results. Include parameter tuning guidance.

66. **Video tutorials** — Screen recordings of the QGIS plugin workflow. Publish to YouTube and link from the docs.

67. **API cookbook** — Short, copy-paste recipes for common tasks: "extract NDVI per segment", "classify with cross-validation", "batch process a directory."

68. **Architecture diagram** — Visual diagram of the module structure, data flow, and class relationships. Keep updated with each release.

---

## DevOps and Packaging

69. **PyPI publishing** — Set up automated PyPI releases via GitHub Actions on tagged commits. Currently only installable from source.

70. **conda-forge recipe** — Publish to conda-forge for easier installation alongside the geospatial stack.

71. **QGIS plugin repository** — Publish the plugin to the official QGIS Python Plugins Repository. Requires packaging as a zip with metadata.txt.

72. **Pre-commit hooks** — Add a .pre-commit-config.yaml with ruff, ruff-format, and check-yaml. Enforce on all commits.

73. **Changelog automation** — Generate changelogs from conventional commit messages. Use release-please or similar.

74. **Docker development image** — Dockerfile with all dependencies pre-installed (GDAL, QGIS, rasterio). Simplifies onboarding and CI.
