# Foundation Model Embeddings Design

## Overview

Foundation model embeddings represent a paradigm shift in OBIA feature extraction. Instead of hand-engineering spectral statistics, shape metrics, and texture measures, we can extract semantically rich feature vectors from pretrained vision models. These embeddings capture high-level visual concepts that complement (or sometimes surpass) traditional features.

This is a Phase 5 feature but is architecturally planned from the start.

## Concept

### Traditional OBIA Features
```
Segment pixels → Statistical aggregation → [mean_red, std_nir, compactness, glcm_entropy, ...]
                                            (hand-engineered, ~20-100 features)
```

### Embedding-Based Features
```
Image → Foundation model encoder → Per-pixel embeddings → Aggregate per segment → [e_0, e_1, ..., e_767]
                                                                                    (learned, ~256-768 features)
```

### Combined Approach (recommended)
```
Traditional features:  [mean_red, std_nir, ndvi, area, glcm_entropy, ...]  (~50 features)
Embedding features:    [e_0, e_1, ..., e_255]                               (~256 features)
Combined vector:       [mean_red, ..., e_0, ..., e_255]                     (~300 features)
```

## Applicable Foundation Models

### General Vision Models

#### DINOv2 (Meta)
- **Architecture:** Vision Transformer (ViT)
- **Output:** 768-dim (ViT-B) or 1024-dim (ViT-L) per-patch embedding
- **Patch size:** 14x14 pixels
- **Why:** Best general-purpose visual features; strong cross-domain transfer to remote sensing without fine-tuning. Frozen DINOv2 has been shown to outperform fine-tuned alternatives for RS semantic segmentation.
- **Package:** `torch` + `transformers` or `timm`

#### SAM Encoder (Meta)
- **Architecture:** ViT-based image encoder
- **Output:** 256-dim per-patch embedding
- **Why:** Already computed if using SAM for segmentation -- embeddings are "free"
- **Package:** `segment-geospatial`

### Remote Sensing-Specific Models

#### Clay Foundation Model
- Trained specifically on multi-spectral satellite imagery (Sentinel-2, Landsat)
- Handles multi-spectral bands natively (not just RGB)
- Location-aware (encodes lat/lon and time)

#### Prithvi (IBM/NASA)
- Geospatial foundation model trained on HLS (Harmonized Landsat-Sentinel) data
- Temporal-aware (handles time series)
- 6 spectral bands

#### SatMAE
- Masked autoencoder pretrained on satellite imagery
- Handles temporal and multi-spectral data

### Integration Library: rs-embed

The `rs-embed` library (Feb 2026) provides a unified interface for obtaining embeddings from diverse RS foundation models with a single API:

```python
from rsembed import DINOv2Embedder

embedder = DINOv2Embedder()
embeddings = embedder.get_embedding(roi, time_range)
```

We should integrate with rs-embed where possible for model-agnostic embedding extraction.

## Implementation Design

### Embedding Extraction Pipeline

```python
class EmbeddingExtractor:
    """Extract foundation model embeddings per segment."""

    def __init__(
        self,
        model: str = "dinov2_vitb14",
        device: str = "auto",         # "cpu", "cuda", "mps", "auto"
        batch_size: int = 4,
        aggregation: str = "mean",    # "mean", "max", "median", "concat_stats"
    ):
        ...

    def extract(
        self,
        image: np.ndarray,          # (bands, H, W) -- may need RGB conversion
        labels: np.ndarray,          # (H, W) segment labels
    ) -> pd.DataFrame:
        """
        Returns DataFrame with segment_id as index,
        embedding dimensions as columns (e_0, e_1, ..., e_N).
        """
```

### Processing Steps

1. **Preprocessing:**
   - Convert multi-spectral to RGB if model requires it (for DINOv2/SAM)
   - Or use all bands if model supports multi-spectral (Clay, Prithvi)
   - Normalize/resize to model's expected input

2. **Tiled Inference:**
   - Large images don't fit in GPU memory
   - Process in overlapping tiles (e.g., 518x518 for DINOv2 with 37x37 patches)
   - Stitch per-patch embeddings back into full spatial grid

3. **Per-Pixel Embedding Map:**
   - Model produces per-patch embeddings (e.g., one 768-dim vector per 14x14 patch)
   - Upsample to per-pixel via bilinear interpolation
   - Result: `(H, W, embedding_dim)` array

4. **Per-Segment Aggregation:**
   - For each segment, gather all pixel embeddings within it
   - Aggregate using chosen strategy:
     - **Mean:** Average embedding (most common, robust)
     - **Max:** Element-wise maximum (captures most activated features)
     - **Concat stats:** Concatenate mean + std (doubles dimensionality but more expressive)
   - Result: DataFrame with one row per segment, embedding_dim columns

### Dimensionality Reduction (Optional)

768 dimensions per segment may be too many for some classifiers. Options:
- **PCA:** Reduce to 32-128 components (retain 95% variance)
- **UMAP:** Non-linear reduction for visualization and clustering
- **Feature selection:** Use classifier's feature importance to prune

This is optional -- Random Forest handles high-dimensional spaces well.

## Usage Patterns

### Pattern 1: Embedding-Only Classification

For quick exploration when you don't want to compute traditional features:

```python
embeddings = geoobia.features.extract_embeddings(image, labels, model="dinov2")
classes = geoobia.classify(embeddings, method="kmeans", n_clusters=10)
```

### Pattern 2: Embedding-Augmented Classification

Combine embeddings with traditional features for best accuracy:

```python
traditional = geoobia.extract_features(image, labels, categories=["spectral", "geometry"])
embeddings = geoobia.features.extract_embeddings(image, labels, model="dinov2")
combined = traditional.join(embeddings)
classes = geoobia.classify(combined, method="random_forest", training=samples)
```

### Pattern 3: SAM Segment + Embed Pipeline

When using SAM for segmentation, reuse the encoder's embeddings:

```python
segmenter = geoobia.segmentation.create("sam")
labels = segmenter.segment(image)
embeddings = segmenter.get_embeddings(labels)  # Reuse encoder output
```

### Pattern 4: Transfer Learning

Use embeddings as a starting point for a task-specific classifier:

```python
embeddings = geoobia.features.extract_embeddings(image, labels, model="dinov2")
# Fine-tune a small MLP on top of frozen embeddings
classifier = geoobia.classification.NeuralClassifier(
    input_dim=768, hidden_dims=[256, 64], n_classes=10
)
classifier.fit(embeddings, training_labels, epochs=50)
```

## Hardware Requirements

| Model | VRAM (inference) | Speed (1000x1000 px) | CPU Fallback |
|-------|-----------------|----------------------|-------------|
| DINOv2 ViT-B/14 | ~2 GB | ~3 seconds | Yes (10x slower) |
| DINOv2 ViT-L/14 | ~4 GB | ~8 seconds | Yes (20x slower) |
| SAM ViT-B | ~2 GB | ~5 seconds | Yes (15x slower) |
| SAM ViT-H | ~8 GB | ~15 seconds | Impractical |

CPU fallback is supported but significantly slower. For large datasets, GPU is strongly recommended.

## Dependency Strategy

Foundation model support is **optional** -- the core library works without torch/transformers:

```
geoobia                  # Core: numpy, scipy, scikit-learn, rasterio, geopandas
geoobia[embeddings]      # Adds: torch, transformers, timm
geoobia[sam]             # Adds: segment-geospatial
geoobia[all]             # Everything
```

This keeps the base installation lightweight (~50MB) while allowing advanced features for users with GPUs.
