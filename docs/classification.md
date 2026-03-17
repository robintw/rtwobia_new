# Classification Design

## Overview

Classification assigns a class label to each segment based on its feature vector. The system supports supervised (user-provided training data), unsupervised (automatic clustering), and rule-based (user-defined logic) approaches. Following eCognition's model, users should be able to mix and iterate between these approaches.

## Interface Design

```python
class BaseClassifier:
    """Base class for all classifiers."""

    def fit(self, features: pd.DataFrame, labels: pd.Series = None):
        """Train the classifier (supervised) or fit clusters (unsupervised)."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict class labels for segments."""

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities/membership for each segment."""

    def save(self, path: str):
        """Serialize trained model to disk."""

    @classmethod
    def load(cls, path: str) -> "BaseClassifier":
        """Load a trained model from disk."""

    def get_params(self) -> dict:
        """Return current parameters."""
```

## Supervised Classification (Phase 1)

### Training Workflow

1. **Sample selection:** User selects representative segments for each class, either by:
   - Clicking segments in the GUI and assigning class labels
   - Providing a vector file with training polygons (segments overlapping training polygons are labeled)
   - Providing a CSV mapping segment IDs to class labels

2. **Feature selection:** User selects which features to use (or uses all features). Feature importance from an initial Random Forest run can guide selection.

3. **Training:** Fit the classifier on the training data.

4. **Prediction:** Apply the trained classifier to all segments.

5. **Validation:** Assess accuracy using held-out samples, cross-validation, or a separate test set.

### Algorithms

#### Random Forest (Phase 1, default)

**Source:** `sklearn.ensemble.RandomForestClassifier`

The workhorse of remote sensing classification. Handles high-dimensional feature spaces, provides feature importance, is robust to overfitting, and runs fast.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Number of trees |
| `max_depth` | None | Max tree depth (None = unlimited) |
| `min_samples_split` | 5 | Min samples to split a node |
| `max_features` | "sqrt" | Features considered per split |
| `class_weight` | "balanced" | Handle imbalanced classes |

#### Support Vector Machine (Phase 2)

**Source:** `sklearn.svm.SVC`

Can achieve higher accuracy than Random Forest on small, well-separated feature spaces. Better with fewer, well-chosen features.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel` | "rbf" | Kernel function |
| `C` | 1.0 | Regularization |
| `gamma` | "scale" | Kernel coefficient |
| `class_weight` | "balanced" | Handle imbalanced classes |

#### Gradient Boosting (Phase 2)

**Source:** `sklearn.ensemble.GradientBoostingClassifier` or XGBoost/LightGBM

Often outperforms Random Forest in benchmarks, especially with hyperparameter tuning.

#### K-Nearest Neighbors (Phase 2)

**Source:** `sklearn.neighbors.KNeighborsClassifier`

Similar to eCognition's nearest-neighbor classifier. Simple, interpretable, works well with good training samples.

## Unsupervised Classification (Phase 1)

For exploratory analysis or when training data is unavailable.

### K-Means Clustering

**Source:** `sklearn.cluster.KMeans`

Groups segments into `k` clusters based on feature similarity. User assigns meaning to clusters after the fact.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | 8 | Number of classes |
| `n_init` | 10 | Number of random initializations |
| `max_iter` | 300 | Maximum iterations |

### Gaussian Mixture Model (Phase 2)

**Source:** `sklearn.mixture.GaussianMixture`

Soft clustering -- each segment gets a probability of belonging to each cluster. More flexible than K-Means.

### DBSCAN (Phase 2)

**Source:** `sklearn.cluster.DBSCAN`

Density-based clustering. Discovers clusters of arbitrary shape and identifies outliers. Does not require specifying the number of clusters.

## Rule-Based Classification (Phase 2)

Inspired by eCognition's fuzzy membership function system. Users define rules mapping feature values to class membership.

### Rule Types

1. **Threshold rules:** Simple min/max bounds on feature values
   ```python
   Rule("vegetation", "ndvi > 0.3 AND brightness < 150")
   ```

2. **Fuzzy membership functions:** Continuous membership (0-1) based on feature values
   ```python
   FuzzyRule("water", feature="ndwi", function="sigmoid", params=(0.0, 0.3))
   # 0% membership at ndwi=0, 100% at ndwi=0.3, smooth transition between
   ```

3. **Compound rules:** Combine multiple rules with AND/OR/NOT
   ```python
   CompoundRule("urban", [
       FuzzyRule("ndvi", "low", (-0.1, 0.2)),     # Low vegetation
       FuzzyRule("brightness", "high", (100, 200)), # Bright
       FuzzyRule("compactness", "high", (0.5, 0.9)) # Compact shape
   ], operator="AND")
   ```

### Fuzzy Membership Functions

Supported function shapes:
- **Linear:** Ramp from 0 to 1 between two values
- **Sigmoid:** S-curve transition
- **Gaussian:** Bell curve centered on ideal value
- **Trapezoidal:** Flat top with linear ramps on sides

Each class's total membership is the combination (AND = min, OR = max) of its rule memberships. The segment is assigned to the class with the highest membership value.

### Expression Engine

For power users, a simple expression language for rules:

```
class "Forest":
    ndvi > 0.4
    AND glcm_entropy > 2.0
    AND area > 500

class "Water":
    ndwi > 0.3
    AND brightness < 80
```

This is parsed into a rule tree and evaluated against the feature DataFrame. Uses safe expression evaluation (no arbitrary code execution).

## Accuracy Assessment

### Metrics

| Metric | Description |
|--------|-------------|
| Overall Accuracy | % correctly classified |
| Kappa Coefficient | Agreement corrected for chance |
| Per-class Precision | True positives / predicted positives |
| Per-class Recall | True positives / actual positives |
| F1 Score | Harmonic mean of precision and recall |
| Confusion Matrix | Full breakdown of predicted vs actual |

### Validation Methods

1. **Train/test split:** Random split of training samples (default: 70/30)
2. **Cross-validation:** K-fold CV on training data (default: 5-fold)
3. **Independent validation:** User-provided separate validation dataset
4. **Spatial cross-validation:** Splits that respect spatial autocorrelation (blocks rather than random)

### Feature Importance

For Random Forest and Gradient Boosting:
- Gini importance (built-in to sklearn)
- Permutation importance (more robust)
- Partial dependence plots for understanding feature effects

Output as a ranked table and optionally as a bar chart.

## Iterative Classification Workflow

Following eCognition's approach, classification is not a single step but an iterative process:

1. **Coarse classification:** Separate major land cover types (vegetation, water, built-up, bare)
2. **Sub-classification:** Within each coarse class, further subdivide (e.g., vegetation -> forest, grassland, cropland)
3. **Refinement:** Use contextual rules to fix misclassifications (e.g., small "water" segments surrounded by "forest" are likely shadows)

The pipeline engine (see [api_and_cli.md](api_and_cli.md)) supports this iterative workflow by allowing classification steps to reference and refine previous results.

## Output

Classification results are stored as:
1. **Class label column** added to the feature DataFrame/GeoPackage
2. **Probability columns** (one per class) for soft classification results
3. **Classified raster** -- the segment label raster remapped to class values
4. **Accuracy report** -- JSON/HTML with metrics and confusion matrix
5. **Trained model** -- serialized sklearn model (joblib format)
