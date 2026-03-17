# Feature Extraction Design

## Overview

Feature extraction computes a rich attribute vector for each segment. These attributes are the basis for classification -- the richer and more discriminating the feature set, the better the classification results. This is the key advantage of OBIA over pixel-based analysis: each object has geometry, texture, context, and aggregated spectral information, not just a single pixel's band values.

## Interface Design

```python
class BaseExtractor:
    """Base class for feature extractors."""

    def extract(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract features for all segments.

        Args:
            image: Multi-band array (bands, height, width)
            labels: Segment label array (height, width)

        Returns:
            DataFrame with segment_id as index, features as columns.
        """

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return list of feature names this extractor produces."""
```

Multiple extractors can be composed:

```python
features = geoobia.features.extract(
    image, labels,
    extractors=["spectral", "geometry", "texture"],
    bands={"red": 0, "green": 1, "blue": 2, "nir": 3}
)
# Returns a single DataFrame with all features merged
```

## Feature Categories

### 1. Spectral Features (Phase 1)

Per-band statistics computed from all pixels within each segment.

| Feature | Description | Formula |
|---------|-------------|---------|
| `mean_{band}` | Mean pixel value | `np.mean(pixels)` |
| `std_{band}` | Standard deviation | `np.std(pixels)` |
| `min_{band}` | Minimum value | `np.min(pixels)` |
| `max_{band}` | Maximum value | `np.max(pixels)` |
| `median_{band}` | Median value | `np.median(pixels)` |
| `range_{band}` | Value range | `max - min` |
| `brightness` | Mean across all bands | `mean(band_means)` |

**Band ratios** (computed from band means):
| Feature | Formula | Use |
|---------|---------|-----|
| `ndvi` | `(nir - red) / (nir + red)` | Vegetation |
| `ndwi` | `(green - nir) / (green + nir)` | Water |
| `ndbi` | `(swir - nir) / (swir + nir)` | Built-up |

Users can define custom ratios via expressions.

**Implementation:** Use `scipy.ndimage.labeled_comprehension` or `numpy` advanced indexing with the label array. For large datasets, iterate over segments using rasterio windowed reads. The `rasterstats` library is an alternative for zonal statistics but we'll implement our own for tighter integration and performance.

### 2. Geometric Features (Phase 1)

Computed from the segment polygon geometry (vectorized from raster labels).

| Feature | Description | Formula |
|---------|-------------|---------|
| `area` | Area in map units (m^2) | `polygon.area` |
| `area_px` | Area in pixels | `count of pixels` |
| `perimeter` | Perimeter in map units | `polygon.length` |
| `compactness` | Shape compactness | `4 * pi * area / perimeter^2` (1.0 = circle) |
| `elongation` | Length/width ratio | `length / width` of oriented bounding box |
| `rectangularity` | Rectangular fit | `area / oriented_bbox_area` |
| `convexity` | Convex hull ratio | `area / convex_hull_area` |
| `solidity` | Solidity | Same as convexity |
| `centroid_x` | Centroid X coordinate | `polygon.centroid.x` |
| `centroid_y` | Centroid Y coordinate | `polygon.centroid.y` |
| `major_axis` | Major axis length | From `skimage.measure.regionprops` |
| `minor_axis` | Minor axis length | From `skimage.measure.regionprops` |
| `orientation` | Orientation angle | From `skimage.measure.regionprops` |
| `eccentricity` | Ellipse eccentricity | From `skimage.measure.regionprops` |

**Implementation:** Two paths:
1. **Fast (raster-based):** Use `skimage.measure.regionprops` on the label array. Gives area, perimeter, bounding box, moments, orientation, eccentricity directly. No vectorization needed.
2. **Precise (vector-based):** Vectorize labels to polygons (via `rasterio.features.shapes`), compute geometry features using Shapely. More accurate for perimeter and shape metrics.

Default to raster-based for speed; vector-based available as option.

### 3. Texture Features -- GLCM/Haralick (Phase 2)

Gray Level Co-occurrence Matrix (GLCM) features capture spatial patterns of pixel intensity within each segment. These are critical for distinguishing texturally different land cover types (e.g., forest vs. grassland).

**GLCM computation per segment:**
1. Extract the bounding box of pixels for the segment
2. Mask to only segment pixels
3. Quantize to N gray levels (default: 32)
4. Compute GLCM for specified distances and angles
5. Average across directions for rotation invariance

**Haralick features (from GLCM):**
| Feature | Description |
|---------|-------------|
| `glcm_contrast` | Local intensity variation |
| `glcm_dissimilarity` | Similar to contrast but linear |
| `glcm_homogeneity` | Inverse difference moment -- closeness to diagonal |
| `glcm_energy` | Angular second moment -- textural uniformity |
| `glcm_entropy` | Randomness/disorder of texture |
| `glcm_correlation` | Linear dependency of gray levels |
| `glcm_mean` | Mean of GLCM |
| `glcm_variance` | Variance of GLCM |

**Parameters:**
- `distances`: List of pixel distances (default: [1])
- `angles`: List of angles in radians (default: [0, pi/4, pi/2, 3*pi/4])
- `levels`: Number of gray levels for quantization (default: 32)
- `bands`: Which bands to compute texture for

**Implementation:** Use `skimage.feature.graycomatrix` and `skimage.feature.graycoprops`. Compute per-band, per-segment. This is the most computationally expensive feature extraction step -- consider parallelization with joblib.

### 4. Contextual Features (Phase 2)

Features describing a segment's relationship to its neighbors.

| Feature | Description |
|---------|-------------|
| `n_neighbors` | Number of adjacent segments |
| `mean_diff_{band}` | Difference between segment mean and mean of all neighbors |
| `border_contrast_{band}` | Mean spectral difference along shared borders |
| `border_length_ratio` | Shared border length / total perimeter |
| `mean_neighbor_area` | Average area of neighboring segments |
| `relative_area` | Segment area / mean neighbor area |

**Neighbor detection:** Two segments are neighbors if they share at least one edge (4-connectivity) in the raster label array. Build an adjacency graph by scanning all horizontally and vertically adjacent pixel pairs.

**Implementation:**
```python
def build_adjacency(labels: np.ndarray) -> dict[int, set[int]]:
    """Build segment adjacency graph from label raster."""
    # Check horizontal neighbors
    h_mask = labels[:, :-1] != labels[:, 1:]
    # Check vertical neighbors
    v_mask = labels[:-1, :] != labels[1:, :]
    # Collect unique neighbor pairs
    ...
```

### 5. Embedding Features (Phase 5)

Per-object feature vectors from foundation models. Rather than hand-engineering features, use a pretrained model to produce semantically rich embeddings.

**Approach:**
1. Run a foundation model (DINOv2, SAM encoder, or RS-specific model) on the image to produce per-pixel embeddings
2. For each segment, aggregate pixel embeddings (mean pooling)
3. Use the resulting embedding vector (e.g., 768-dim for DINOv2) as features

**Models:**
- **DINOv2:** Strong general-purpose visual features, proven effective for remote sensing
- **SAM encoder:** Already computed if using SAM for segmentation -- embeddings come "for free"
- **RS-specific models:** Clay, Prithvi, SatMAE -- trained on remote sensing data

**Integration with traditional features:** Embeddings complement (not replace) traditional features. Users can combine spectral + geometry + texture + embedding features in a single feature vector.

See [embeddings.md](embeddings.md) for detailed design.

## Performance Considerations

### Memory
- For large images (>10,000 x 10,000 pixels), features must be computed in tiles
- The label array itself may not fit in memory -- use memory-mapped arrays or chunked processing
- GLCM computation is the bottleneck -- parallelize across segments

### Speed
- Spectral features: O(pixels) -- fast, single pass through image
- Geometric features (raster-based): O(pixels) via regionprops -- fast
- Texture features: O(segments * bbox_area * levels^2) -- slow for many segments
- Contextual features: O(pixels) for adjacency + O(segments * neighbors) for stats

### Parallelization
- Feature extraction is embarrassingly parallel across segments
- Use `joblib.Parallel` for multi-core GLCM computation
- Consider Dask for out-of-core computation on very large datasets

## Output Format

Features are stored as:
1. **In-memory:** pandas DataFrame (segment_id index, feature columns)
2. **On-disk:** Parquet file (fast, columnar, supports large datasets)
3. **With geometry:** GeoPackage (GeoDataFrame with polygon geometry + all feature columns)

Feature metadata (units, descriptions, which band each feature relates to) is stored in a sidecar JSON file.
