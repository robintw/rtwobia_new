"""Shepherd segmentation algorithm.

A pure Python/NumPy/Numba reimplementation of the Shepherd algorithm.

4 steps:
1. K-Means seeding — cluster pixels spectrally, label each to nearest centre
2. Clumping — connected component labelling (4-connectivity)
3. Iterative elimination — merge small clumps into spectrally closest
   neighbour, processing from smallest upward
4. Relabeling — compact sequential IDs (0 = nodata, 1..N)

Performance-critical inner loops use Numba JIT.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from sklearn.cluster import MiniBatchKMeans

from geobia.segmentation.base import BaseSegmenter


@njit(cache=True)
def _connected_components(cluster_labels: np.ndarray) -> np.ndarray:
    """4-connectivity connected component labelling (union-find)."""
    h, w = cluster_labels.shape
    parent = np.arange(h * w, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for r in range(h):
        for c in range(w):
            idx = r * w + c
            val = cluster_labels[r, c]
            if c + 1 < w and cluster_labels[r, c + 1] == val:
                union(idx, idx + 1)
            if r + 1 < h and cluster_labels[r + 1, c] == val:
                union(idx, idx + w)

    for i in range(h * w):
        parent[i] = find(i)

    label_map = np.full(h * w, -1, dtype=np.int64)
    next_label = np.int64(1)
    out = np.empty(h * w, dtype=np.int32)
    for i in range(h * w):
        root = parent[i]
        if label_map[root] == -1:
            label_map[root] = next_label
            next_label += 1
        out[i] = np.int32(label_map[root])

    return out.reshape(h, w)


@njit(cache=True)
def _iterative_elimination(
    labels: np.ndarray,
    image: np.ndarray,
    min_n_pxls: int,
    dist_thres: float,
) -> np.ndarray:
    """Iteratively merge small clumps into spectrally closest neighbour.

    Entire elimination loop runs in numba for performance.

    Args:
        labels: (h, w) int32 label array (modified in-place)
        image: (h, w, n_bands) float64 array
        min_n_pxls: minimum segment size in pixels
        dist_thres: maximum spectral distance for merging

    Returns:
        Modified labels array.
    """
    h, w = labels.shape
    n_bands = image.shape[2]
    n_pixels = h * w
    labels_flat = labels.ravel()

    for size_thresh in range(1, min_n_pxls + 1):
        max_label = 0
        for i in range(n_pixels):
            if labels_flat[i] > max_label:
                max_label = labels_flat[i]

        if max_label == 0:
            break

        # Compute sizes and band sums
        sizes = np.zeros(max_label + 1, dtype=np.int64)
        sums = np.zeros((max_label + 1, n_bands), dtype=np.float64)

        for r in range(h):
            for c in range(w):
                v = labels[r, c]
                if v > 0:
                    sizes[v] += 1
                    for b in range(n_bands):
                        sums[v, b] += image[r, c, b]

        # Compute means
        means = np.zeros((max_label + 1, n_bands), dtype=np.float64)
        for i in range(1, max_label + 1):
            if sizes[i] > 0:
                for b in range(n_bands):
                    means[i, b] = sums[i, b] / sizes[i]

        # Check if any clumps need merging
        has_small = False
        for i in range(1, max_label + 1):
            if 0 < sizes[i] <= size_thresh:
                has_small = True
                break
        if not has_small:
            continue

        # Build adjacency using a dense boolean matrix
        # For very large label counts, use sparse approach
        if max_label <= 50000:
            # Dense adjacency: adj_matrix[a, b] = True if a,b are neighbours
            # Use a packed neighbour list instead of full matrix for memory
            # Build neighbour list: for each label, store up to MAX_NEIGHBOURS
            MAX_NEIGHBOURS = 100
            adj_list = np.zeros((max_label + 1, MAX_NEIGHBOURS), dtype=np.int32)
            adj_count = np.zeros(max_label + 1, dtype=np.int32)

            for r in range(h):
                for c in range(w):
                    v = labels[r, c]
                    if v <= 0:
                        continue
                    # Check right
                    if c + 1 < w:
                        u = labels[r, c + 1]
                        if u > 0 and u != v:
                            _add_neighbour(adj_list, adj_count, v, u, MAX_NEIGHBOURS)
                            _add_neighbour(adj_list, adj_count, u, v, MAX_NEIGHBOURS)
                    # Check down
                    if r + 1 < h:
                        u = labels[r + 1, c]
                        if u > 0 and u != v:
                            _add_neighbour(adj_list, adj_count, v, u, MAX_NEIGHBOURS)
                            _add_neighbour(adj_list, adj_count, u, v, MAX_NEIGHBOURS)
        else:
            # Fallback for very large label counts — same approach but cap neighbours
            MAX_NEIGHBOURS = 50
            adj_list = np.zeros((max_label + 1, MAX_NEIGHBOURS), dtype=np.int32)
            adj_count = np.zeros(max_label + 1, dtype=np.int32)

            for r in range(h):
                for c in range(w):
                    v = labels[r, c]
                    if v <= 0:
                        continue
                    if c + 1 < w:
                        u = labels[r, c + 1]
                        if u > 0 and u != v:
                            _add_neighbour(adj_list, adj_count, v, u, MAX_NEIGHBOURS)
                            _add_neighbour(adj_list, adj_count, u, v, MAX_NEIGHBOURS)
                    if r + 1 < h:
                        u = labels[r + 1, c]
                        if u > 0 and u != v:
                            _add_neighbour(adj_list, adj_count, v, u, MAX_NEIGHBOURS)
                            _add_neighbour(adj_list, adj_count, u, v, MAX_NEIGHBOURS)

        # Find merge targets
        merge_to = np.arange(max_label + 1, dtype=np.int32)
        any_merges = False

        for clump_id in range(1, max_label + 1):
            if sizes[clump_id] <= 0 or sizes[clump_id] > size_thresh:
                continue

            best_dist = dist_thres
            best_target = np.int32(-1)
            n_adj = adj_count[clump_id]

            for ni in range(n_adj):
                neighbour = adj_list[clump_id, ni]
                if sizes[neighbour] <= sizes[clump_id]:
                    continue

                dist_sq = 0.0
                for b in range(n_bands):
                    d = means[clump_id, b] - means[neighbour, b]
                    dist_sq += d * d
                dist = dist_sq ** 0.5

                if dist < best_dist:
                    best_dist = dist
                    best_target = neighbour

            if best_target > 0:
                merge_to[clump_id] = best_target
                any_merges = True

        if not any_merges:
            continue

        # Resolve chains
        for i in range(1, max_label + 1):
            if merge_to[i] == i:
                continue
            root = merge_to[i]
            depth = 0
            while merge_to[root] != root and depth < max_label:
                root = merge_to[root]
                depth += 1
            merge_to[i] = root

        # Apply merges
        for i in range(n_pixels):
            labels_flat[i] = merge_to[labels_flat[i]]

        # Relabel sequential
        mapping = np.zeros(max_label + 1, dtype=np.int32)
        next_id = np.int32(1)
        for i in range(n_pixels):
            old = labels_flat[i]
            if old > 0:
                if mapping[old] == 0:
                    mapping[old] = next_id
                    next_id += 1
                labels_flat[i] = mapping[old]

    return labels


@njit(cache=True)
def _add_neighbour(adj_list, adj_count, a, b, max_n):
    """Add b as neighbour of a if not already present and space available."""
    n = adj_count[a]
    # Check if already present
    for i in range(n):
        if adj_list[a, i] == b:
            return
    if n < max_n:
        adj_list[a, n] = b
        adj_count[a] = n + 1


class ShepherdSegmenter(BaseSegmenter):
    """Shepherd segmentation algorithm.

    K-means seeded, iterative elimination of small segments. Produces
    deterministic results and scales well to large images.
    """

    def __init__(
        self,
        num_clusters: int = 60,
        min_n_pxls: int = 100,
        dist_thres: float = 100.0,
        sampling: int = 100,
        bands: list[int] | None = None,
    ):
        self.num_clusters = num_clusters
        self.min_n_pxls = min_n_pxls
        self.dist_thres = dist_thres
        self.sampling = sampling
        self.bands = bands

    def segment(
        self,
        image: np.ndarray,
        nodata_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        n_bands, h, w = image.shape

        if self.bands is not None:
            image = image[self.bands]
            n_bands = len(self.bands)

        # Reshape to (n_pixels, n_bands) and normalize to 0-1
        pixels = image.reshape(n_bands, -1).T.astype(np.float64)
        for b in range(n_bands):
            bmin = pixels[:, b].min()
            bmax = pixels[:, b].max()
            if bmax > bmin:
                pixels[:, b] = (pixels[:, b] - bmin) / (bmax - bmin)

        valid = ~nodata_mask.ravel() if nodata_mask is not None else np.ones(h * w, dtype=bool)

        # --- Step 1: K-Means seeding ---
        valid_pixels = pixels[valid]
        step = max(1, self.sampling)
        sample = valid_pixels[::step]

        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=min(10000, len(sample)),
            n_init=3,
            random_state=42,
        )
        kmeans.fit(sample)

        cluster_labels = np.zeros(h * w, dtype=np.int32)
        cluster_labels[valid] = kmeans.predict(valid_pixels).astype(np.int32)
        cluster_labels_2d = cluster_labels.reshape(h, w)

        # --- Step 2: Connected component clumping ---
        labels = _connected_components(cluster_labels_2d)
        if nodata_mask is not None:
            labels[nodata_mask] = 0

        # --- Step 3 + 4: Iterative elimination (runs entirely in numba) ---
        image_hwb = pixels.reshape(h, w, n_bands)
        labels = _iterative_elimination(labels, image_hwb, self.min_n_pxls, self.dist_thres)

        return labels

    def get_params(self) -> dict:
        return {
            "algorithm": "shepherd",
            "num_clusters": self.num_clusters,
            "min_n_pxls": self.min_n_pxls,
            "dist_thres": self.dist_thres,
            "sampling": self.sampling,
            "bands": self.bands,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "num_clusters": {
                    "type": "integer",
                    "default": 60,
                    "minimum": 2,
                    "description": "K-means seed count (30-90 recommended)",
                },
                "min_n_pxls": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 1,
                    "description": "Minimum segment size in pixels",
                },
                "dist_thres": {
                    "type": "number",
                    "default": 100.0,
                    "minimum": 0,
                    "description": "Max spectral distance for merging",
                },
                "sampling": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 1,
                    "description": "Subsampling rate for k-means (every Nth pixel)",
                },
                "bands": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Band indices to use (default: all)",
                },
            },
        }
