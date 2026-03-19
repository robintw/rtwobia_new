# Codebase Review: geobia

**Date:** 2026-03-19
**Scope:** Full review of source code, tests, configuration, and architecture

---

## Executive Summary

geobia is a well-structured Geographic Object-Based Image Analysis library with ~4,400 lines of source code across 32 Python files, a QGIS plugin, and a test suite of ~164 tests. The core architecture is sound — clean ABC-based plugin patterns, consistent data conventions, and good separation of concerns. However, there are several categories of issues ranging from potential bugs to missing infrastructure that should be addressed.

**Severity breakdown:**
- **Critical (bugs/data-loss risk):** 3
- **Major (incorrect behavior/missing coverage):** 8
- **Moderate (quality/maintainability):** 12
- **Minor (style/consistency):** 9

---

## 1. Potential Bugs

### 1.1 CRITICAL: `read_raster_windows` generator / file handle lifecycle
**File:** `src/geobia/io/raster.py:48-63`

The generator function opens a rasterio dataset inside a `with` block, but yields tiles. Depending on how the generator is consumed, the file handle may be closed before all tiles are read, or held open indefinitely if the generator is never fully consumed. The context manager exits after the function returns the generator object, not after iteration completes.

**Fix:** Restructure so the `with` block wraps the entire iteration, or use `contextlib.contextmanager` to tie the file lifecycle to the generator.

### 1.2 CRITICAL: Feature `__init__.py` only passes `nodata` kwarg to extractors
**File:** `src/geobia/features/__init__.py:56-58`

The `extract()` dispatcher filters kwargs to only pass `nodata` to all extractors. This means:
- `SpectralExtractor` never receives `band_names`
- `GeometryExtractor` never receives `pixel_size`
- `TextureExtractor` never receives `bands` or `distances`

These parameters are silently dropped, causing extractors to use defaults even when the user explicitly passes them.

**Fix:** Each extractor should declare accepted kwargs, or pass all kwargs through and let extractors ignore what they don't need.

### 1.3 CRITICAL: `read_training_samples` has no bounds checking
**File:** `src/geobia/io/vector.py:93-120`

`rowcol()` can return coordinates outside raster bounds when training points fall outside the raster extent. No validation is performed before using these as array indices, causing `IndexError` at runtime.

**Fix:** Add bounds checking after `rowcol()` and either skip out-of-bounds points or raise a clear error.

### 1.4 MAJOR: DBSCAN label ambiguity
**File:** `src/geobia/classification/unsupervised.py:97-130`

DBSCAN returns `-1` for noise points, which is shifted to `0`, then later code adds `1` to all predictions. This creates ambiguity about whether noise points are intentionally classified as class 1 or remain "unclassified." The shift logic differs from KMeans/GMM handling.

**Fix:** Handle DBSCAN noise labels explicitly with a dedicated nodata/unclassified value.

### 1.5 MAJOR: Multi-scale segmentation ignores nodata
**File:** `src/geobia/segmentation/multiscale.py:142-160`

`segment_multiscale()` never passes `nodata_mask` to the underlying segmenters. Nodata regions will be treated as valid pixels and segmented, producing incorrect results for images with nodata areas.

### 1.6 MAJOR: `BaseClassifier.load()` deserializes without validation
**File:** `src/geobia/classification/base.py:43-53`

`joblib.load()` can execute arbitrary code. The loaded object is not validated to be a `BaseClassifier` instance, creating both a correctness and security concern.

**Fix:** Add `isinstance()` check after loading.

---

## 2. Architecture & Design Issues

### 2.1 No structured logging
All modules lack proper Python `logging`. The CLI uses `click.echo()`, but library code has no logging at all. This makes debugging production issues extremely difficult, especially for long-running batch/pipeline operations.

**Impact:** Users cannot enable debug output, trace execution flow, or integrate with logging infrastructure.

### 2.2 Pipeline stores full image in `PipelineResult`
**File:** `src/geobia/pipeline/engine.py:142`

The `PipelineResult` dataclass holds the full image array. During batch processing, each worker stores its own copy. For large satellite imagery, this causes unnecessary memory pressure.

**Fix:** Discard or weakref the image after the feature extraction step.

### 2.3 No configuration file support
**File:** `src/geobia/cli.py`

All parameters are CLI flags with hardcoded defaults. There's no support for a YAML/TOML configuration file, making reproducible workflows difficult without scripting.

### 2.4 Hardcoded GeoTIFF write parameters
**File:** `src/geobia/io/raster.py:109-112`

Block size (256), compression (`deflate`), and tiling are hardcoded in `write_raster()`. Users processing large imagery may need different settings.

### 2.5 Dual version source
Version is defined in both `pyproject.toml` and `src/geobia/__init__.py`. These must be manually synchronized.

**Fix:** Use dynamic versioning (e.g., `importlib.metadata` or `hatch-vcs`).

### 2.6 Git dependency for pyshepseg
**File:** `pyproject.toml`

`pyshepseg` is pinned to a git URL without a commit hash. This is fragile for reproducibility and can break installs if the upstream repo changes.

---

## 3. Test Coverage Gaps

### 3.1 Completely untested modules

| Module | Risk |
|--------|------|
| `io/vector.py` (read_vector, write_vector, read_training_samples) | HIGH — used in export and training pipelines |
| `segmentation/sam.py` (SAMSegmenter) | MEDIUM — optional but documented feature |
| `segmentation/__init__.py` segment_tiled() | HIGH — critical for large image processing |
| `classification/accuracy.py` cross_validate() | MEDIUM — model evaluation function |

### 3.2 Sparse CLI test coverage
**File:** `tests/test_cli.py` — only 5 tests

- `classify` command: 0 tests
- `export` command: 0 tests
- Error handling (missing files, invalid args): 0 tests
- Help/version output: 0 tests

### 3.3 Missing edge case tests

**Feature extraction:**
- Single-band images
- NaN/infinite values in input
- All-zero segments
- Single-pixel segments

**Classification:**
- Single sample per class
- Highly imbalanced classes
- DBSCAN producing all-noise result
- More clusters requested than data points

**I/O:**
- Non-existent file paths
- Corrupt raster files
- Different dtypes (int16, int64)
- CRS edge cases

**Change detection:**
- No-change scenario
- Complete change scenario
- Threshold boundary values (0, 1.0)

**Batch processing:**
- Empty batch
- All files failing
- Parallel execution (max_workers > 1)

### 3.4 Missing test infrastructure

- No pytest-xdist for parallel test execution
- No coverage threshold configured
- `qgis` marker defined but not applied to QGIS tests
- No parametrized tests for combinatorial coverage

---

## 4. Code Quality Issues

### 4.1 Inconsistent parameter naming across segmenters
- `min_size` vs `min_n_pxls` (shepherd)
- `n_clusters` vs `n_components` (GMM remaps internally)
- `bands` means different things in different extractors

### 4.2 Inconsistent error handling patterns
- `segmentation/` and `classification/`: Good — ValueError for bad input, RuntimeError for not-fitted
- `features/` and `io/`: Poor — no input validation, raw exceptions from numpy/rasterio bubble up

### 4.3 Missing `__all__` in some modules
- `SAMSegmenter` available via registry but not in `segmentation/__all__`
- `batch.py` and `change.py` have no `__all__`

### 4.4 Imports inside functions unnecessarily
**File:** `cli.py` — standard library imports inside Click command handlers. Conditional imports are appropriate for optional deps (sam.py), but not for always-available modules.

**File:** `change.py:93` — `threshold_otsu` imported inside function despite `skimage` being a required dependency.

### 4.5 Type annotation gaps
- `batch.py`: Uses `Any` for `pipeline` and `progress_callback` parameters
- `pipeline/engine.py`: Overuses `dict[str, Any]` instead of specific types
- No `TypeAlias` definitions for common types like metadata dicts

---

## 5. Missing Infrastructure

### 5.1 No CI/CD pipeline
No GitHub Actions, GitLab CI, or any automation. Tests, linting, and type checking are only run manually.

### 5.2 No code formatting/linting configuration
No ruff, black, flake8, pylint, or any linter is configured. No `.pre-commit-config.yaml`.

### 5.3 No type checker in dev dependencies
`py.typed` marker exists (PEP 561 compliance), but neither mypy nor pyright is in dev dependencies.

### 5.4 No API documentation generation
No sphinx, mkdocs, or pdoc configuration. The only documentation is CLAUDE.md and code comments.

---

## 6. QGIS Plugin Concerns

### 6.1 Naming collision risk
The plugin docstring warns that the folder must NOT be named "geobia" to avoid shadowing the library. This is fragile and relies on users reading the warning.

### 6.2 No automated plugin tests with QGIS marker
`test_qgis_processing.py` exists with 35+ tests but the `qgis` pytest marker (defined in pyproject.toml) is not applied to these tests, making it impossible to selectively skip them in environments without QGIS.

---

## 7. Task List

### P0 — Critical (fix immediately)

| # | Task | Files | Issue |
|---|------|-------|-------|
| 1 | Fix `read_raster_windows` file handle lifecycle | `io/raster.py` | Bug 1.1 |
| 2 | Fix feature kwargs passthrough in `extract()` | `features/__init__.py` | Bug 1.2 |
| 3 | Add bounds checking in `read_training_samples` | `io/vector.py` | Bug 1.3 |
| 4 | Add type validation in `BaseClassifier.load()` | `classification/base.py` | Bug 1.6 |
| 5 | Fix DBSCAN noise label handling | `classification/unsupervised.py` | Bug 1.4 |
| 6 | Pass nodata_mask through multi-scale segmentation | `segmentation/multiscale.py` | Bug 1.5 |

### P1 — High (address soon)

| # | Task | Files | Issue |
|---|------|-------|-------|
| 7 | Add tests for `io/vector.py` | `tests/test_vector.py` (new) | Gap 3.1 |
| 8 | Add tests for `segment_tiled()` | `tests/test_tiled.py` (new) | Gap 3.1 |
| 9 | Add tests for `cross_validate()` | `tests/test_accuracy.py` (new) | Gap 3.1 |
| 10 | Expand CLI tests (classify, export, error paths) | `tests/test_cli.py` | Gap 3.2 |
| 11 | Add structured logging throughout library | All modules | Issue 2.1 |
| 12 | Add CI/CD pipeline (GitHub Actions) | `.github/workflows/` | Issue 5.1 |
| 13 | Pin pyshepseg to a specific commit or release | `pyproject.toml` | Issue 2.6 |

### P2 — Medium (improve quality)

| # | Task | Files | Issue |
|---|------|-------|-------|
| 14 | Add input validation to feature extractors | `features/*.py` | Issue 4.2 |
| 15 | Add edge case tests for features (NaN, single-band, single-pixel) | `tests/test_features.py` | Gap 3.3 |
| 16 | Add edge case tests for classification | `tests/test_classification.py` | Gap 3.3 |
| 17 | Add linting config (ruff) and pre-commit hooks | `pyproject.toml`, `.pre-commit-config.yaml` | Issue 5.2 |
| 18 | Add mypy to dev deps and fix type errors | `pyproject.toml` | Issue 5.3 |
| 19 | Consolidate version to single source | `pyproject.toml`, `__init__.py` | Issue 2.5 |
| 20 | Make GeoTIFF write parameters configurable | `io/raster.py` | Issue 2.4 |
| 21 | Discard image from PipelineResult after feature extraction | `pipeline/engine.py` | Issue 2.2 |
| 22 | Normalize parameter naming across segmenters | `segmentation/*.py` | Issue 4.1 |
| 23 | Add `qgis` marker to QGIS test file | `tests/test_qgis_processing.py` | Issue 6.2 |
| 24 | Add `__all__` to batch.py and change.py | `batch.py`, `change.py` | Issue 4.3 |
| 25 | Add SAMSegmenter to segmentation `__all__` | `segmentation/__init__.py` | Issue 4.3 |

### P3 — Low (nice to have)

| # | Task | Files | Issue |
|---|------|-------|-------|
| 26 | Move function-level imports to module level where appropriate | `cli.py`, `change.py` | Issue 4.4 |
| 27 | Replace `Any` types with specific types | `batch.py`, `pipeline/engine.py` | Issue 4.5 |
| 28 | Add config file support (YAML/TOML) for CLI | `cli.py` | Issue 2.3 |
| 29 | Add API documentation generation (mkdocs/sphinx) | New config files | Issue 5.4 |
| 30 | Add SAM segmentation tests (or mark as untested optional) | `tests/test_sam.py` (new) | Gap 3.1 |
| 31 | Add parametrized tests for combinatorial coverage | Various test files | Gap 3.4 |
| 32 | Add batch processing edge case tests (empty, all-fail, parallel) | `tests/test_batch.py` | Gap 3.3 |
