# Segmentation Design

## Overview

Segmentation partitions raster imagery into spatially contiguous, spectrally homogeneous regions ("objects" or "segments"). This is the foundation of OBIA -- all subsequent feature extraction and classification operate on these objects rather than individual pixels.

## Interface Design

All segmentation algorithms implement a common interface:

```python
class BaseSegmenter:
    """Base class for all segmentation algorithms."""

    def __init__(self, **params):
        """Initialize with algorithm-specific parameters."""

    def segment(self, image: np.ndarray, nodata_mask: np.ndarray = None) -> np.ndarray:
        """
        Segment an image into labeled regions.

        Args:
            image: Multi-band image array, shape (bands, height, width)
            nodata_mask: Boolean mask, True where data is invalid

        Returns:
            Integer array of segment IDs, shape (height, width).
            0 = no data, 1..N = segment IDs.
        """

    def get_params(self) -> dict:
        """Return current parameters as a serializable dict."""

    @classmethod
    def get_param_schema(cls) -> dict:
        """Return JSON Schema describing available parameters.
        Used by GUI to auto-generate parameter panels."""
```

A factory function creates segmenters by name:

```python
segmenter = geoobia.segmentation.create("slic", n_segments=500, compactness=10)
labels = segmenter.segment(image)
```

## Algorithms

### 1. SLIC Superpixels (Phase 1)

**Source:** scikit-image `skimage.segmentation.slic`

**How it works:** K-means clustering in a 5D space (x, y, L, a, b) where spatial and color dimensions are balanced by a compactness parameter. Produces roughly equally-sized superpixels.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_segments` | 500 | Approximate number of segments |
| `compactness` | 10.0 | Balances color vs spatial proximity. Higher = more compact/regular shapes |
| `sigma` | 0.0 | Gaussian smoothing before segmentation |
| `min_size_factor` | 0.5 | Minimum segment size as fraction of average |
| `enforce_connectivity` | True | Ensure all segments are spatially connected |

**Strengths:** Fast, predictable segment count, good for initial exploration.
**Weaknesses:** Produces grid-like artifacts; poor at following natural boundaries.

### 2. Felzenszwalb (Phase 1)

**Source:** scikit-image `skimage.segmentation.felzenszwalb`

**How it works:** Graph-based method using minimum spanning tree. Merges regions when the inter-region difference is small relative to internal variation. Adapts segment size to image content -- produces smaller segments in high-detail areas and larger segments in homogeneous areas.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale` | 100 | Higher = fewer, larger segments |
| `sigma` | 0.8 | Gaussian pre-smoothing |
| `min_size` | 50 | Minimum segment size in pixels |

**Strengths:** Adaptive segment sizes, follows natural boundaries well.
**Weaknesses:** Less predictable segment count; can produce very large segments in homogeneous areas.

### 3. Shepherd Algorithm (Phase 2)

**Source:** RSGISLib `rsgislib.segmentation.shepherdseg` or custom reimplementation

**How it works (4 steps):**

1. **Seeding (K-Means):** Normalize bands to 2-SD range, subsample pixels (e.g., 1%), run k-means to find `k` cluster centres. Label each pixel to nearest centre.

2. **Clumping:** Connected component labeling (4-connectivity, breadth-first flood fill) groups same-label pixels into spatially unique regions. Produces heavy over-segmentation (many single-pixel clumps).

3. **Iterative Elimination:** Core innovation. Process clumps by size from smallest upward:
   - For each size threshold (1 to min_size): find all clumps at that size
   - For each small clump: find its spectrally closest neighbor (Euclidean distance of mean band values) that is larger than itself
   - Merge if spectral distance < threshold
   - Batch merges at end of each size step (prevents order-dependent bias)
   - Update mean spectral values after merging

4. **Relabeling:** Compact sequential IDs (0 = nodata, 1..N = segments).

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clusters` (k) | 60 | K-means seed count. 30-90 recommended. Lower = larger segments |
| `min_n_pxls` | 100 | Minimum mapping unit in pixels |
| `dist_thres` | 100 | Max spectral distance for merging. Prevents merging dissimilar features |
| `sampling` | 100 | Subsampling rate (every Nth pixel for k-means). Higher = faster |
| `bands` | all | Which bands to use for segmentation |

**Strengths:** Deterministic (same input = same output), scales to continental datasets, few intuitive parameters, top-ranked in comparative studies.
**Weaknesses:** Requires RSGISLib or custom implementation; slower than SLIC for small images.

**Implementation strategy:** First wrap RSGISLib (requires conda install), then consider a pure Python/NumPy reimplementation for pip-installable use. The algorithm is well-documented enough to reimplement.

### 4. Watershed (Phase 2)

**Source:** scikit-image `skimage.segmentation.watershed`

**How it works:** Treats gradient magnitude as a topographic surface and "floods" from seed points (markers). Segment boundaries form at ridges (high gradient). Markers can be auto-generated via local minima detection or user-provided.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `markers` | auto | Number of markers, or explicit marker array |
| `compactness` | 0.0 | Higher values produce more regularly-shaped segments |
| `gradient_sigma` | 1.0 | Gaussian smoothing for gradient computation |

**Strengths:** Follows image gradients well; good for imagery with clear boundaries.
**Weaknesses:** Prone to over-segmentation; sensitive to noise; requires good marker placement.

### 5. Segment Anything Model -- SAM (Phase 2)

**Source:** `segment-geospatial` (samgeo) wrapping Meta's SAM/SAM2/SAM3

**How it works:** A vision transformer encodes the image into dense embeddings. A prompt encoder (points, boxes, or automatic grid) generates prompt tokens. A lightweight mask decoder produces segmentation masks. For OBIA, we use the automatic mask generator which tiles the image with a grid of point prompts.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | "sam2" | Which SAM variant to use |
| `points_per_side` | 32 | Density of automatic point grid |
| `pred_iou_thresh` | 0.88 | Confidence threshold for masks |
| `stability_score_thresh` | 0.95 | Stability threshold for masks |
| `min_mask_region_area` | 100 | Minimum segment area in pixels |

**Strengths:** State-of-the-art zero-shot segmentation; handles complex objects; leverages massive pretraining.
**Weaknesses:** Requires GPU for reasonable speed; large model download; may not respect spectral homogeneity as well as traditional methods; not designed specifically for remote sensing.

**Note:** SAM is particularly interesting for high-resolution imagery where objects have clear visual boundaries (buildings, roads, trees). For lower-resolution multispectral imagery, traditional methods may still be preferable.

## Scale Parameter Concept

eCognition's key insight was the **scale parameter** -- a single number controlling segment size. We adopt this concept across algorithms:

Each segmenter exposes a normalized `scale` parameter (0.0 to 1.0) that maps to the algorithm's native size control:
- SLIC: maps to `n_segments` (inversely)
- Felzenszwalb: maps to `scale`
- Shepherd: maps to `num_clusters` (inversely) and `min_n_pxls`
- Watershed: maps to `markers` (inversely)
- SAM: maps to `points_per_side` (inversely)

This enables the GUI to offer a simple "segment size" slider that works across all algorithms.

## Multi-Scale / Hierarchical Segmentation (Phase 5)

eCognition supported multiple segmentation levels forming a hierarchy (fine segments nested within coarse segments). We support this via:

1. Run segmentation at multiple scale parameters
2. Build parent-child relationships (each fine segment maps to exactly one coarse segment)
3. Features can reference parent/child segments (e.g., "area of parent segment", "count of child segments")

This is stored as multiple label rasters with a lookup table mapping fine-to-coarse IDs.

## Post-Processing

After initial segmentation:
- **Merge small segments:** Eliminate segments below a minimum size by merging with spectrally closest neighbor (similar to Shepherd step 3, applicable to any algorithm)
- **Spectral difference merge:** Merge adjacent segments whose mean spectral values differ by less than a threshold
- **Boundary smoothing:** Optional morphological smoothing of segment boundaries

These are implemented as standalone functions that operate on any label array.

## Handling Large Rasters

For images too large to fit in memory:
1. **Tiled processing:** Segment in overlapping tiles, merge at boundaries
2. **The Shepherd tiling approach:** Segment tiles, remove boundary segments, re-tile with offset, re-segment boundaries, merge
3. **Windowed reading:** Use rasterio's windowed I/O to process tile-by-tile

The tiling strategy from the RSGISLib paper (tested at 141,482 x 130,103 pixels) is our reference implementation.
