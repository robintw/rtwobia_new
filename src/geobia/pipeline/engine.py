"""Pipeline engine: define, save, load, and execute OBIA workflows."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StepResult:
    """Result from a single pipeline step."""

    name: str
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result from a complete pipeline run."""

    labels: np.ndarray | None = None
    features: pd.DataFrame | None = None
    predictions: pd.Series | None = None
    meta: dict[str, Any] | None = None
    steps: list[StepResult] = field(default_factory=list)
    image: np.ndarray | None = None

    def export(self, path: str) -> None:
        """Export results to file (GeoPackage or Parquet)."""
        if path.endswith(".gpkg") and self.labels is not None and self.meta is not None:
            from geobia.io.vector import write_vector
            write_vector(path, self.labels, attributes=self.predictions, meta=self.meta)
        elif path.endswith(".parquet") and self.features is not None:
            df = self.features.copy()
            if self.predictions is not None:
                df["class_label"] = self.predictions
            df.to_parquet(path)

    def provenance(self) -> dict:
        """Return provenance record for this run."""
        return {
            "steps": [
                {
                    "name": s.name,
                    "duration_s": round(s.duration_s, 3),
                    **s.metadata,
                }
                for s in self.steps
            ],
            "total_duration_s": round(sum(s.duration_s for s in self.steps), 3),
        }


class Pipeline:
    """Composable OBIA pipeline: segment -> extract -> classify.

    Pipelines can be defined programmatically or loaded from JSON.

    Example:
        pipeline = Pipeline([
            ("segment", "slic", {"n_segments": 1000}),
            ("extract", ["spectral", "geometry"], {"band_names": {"nir": 3}}),
            ("classify", "kmeans", {"n_clusters": 5}),
        ])
        result = pipeline.run("image.tif")
        result.export("output.gpkg")
    """

    def __init__(self, steps: list[tuple] | None = None):
        """
        Args:
            steps: List of (step_type, method_or_categories, params) tuples.
                step_type is one of: "segment", "extract", "classify".
        """
        self.steps = steps or []

    def run(
        self,
        input_path: str | None = None,
        image: np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
        training: str | pd.Series | None = None,
    ) -> PipelineResult:
        """Execute the pipeline.

        Args:
            input_path: Path to input raster. If None, image/meta must be provided.
            image: Pre-loaded image array (bands, height, width).
            meta: Pre-loaded metadata dict.
            training: Path to training samples or pre-loaded Series.

        Returns:
            PipelineResult with labels, features, predictions, and provenance.
        """
        result = PipelineResult(meta=meta)

        # Load image if path provided
        if input_path is not None and image is None:
            from geobia.io.raster import read_raster
            t0 = time.perf_counter()
            image, meta = read_raster(input_path)
            result.meta = meta
            result.steps.append(StepResult(
                name="load",
                duration_s=time.perf_counter() - t0,
                metadata={"path": input_path},
            ))

        result.image = image

        for step_def in self.steps:
            step_type = step_def[0]
            method_or_cats = step_def[1] if len(step_def) > 1 else None
            params = step_def[2] if len(step_def) > 2 else {}

            if step_type == "segment":
                result = self._run_segment(result, image, method_or_cats, params)
            elif step_type == "extract":
                result = self._run_extract(result, image, method_or_cats, params)
            elif step_type == "classify":
                result = self._run_classify(result, method_or_cats, params, training)
            else:
                raise ValueError(f"Unknown pipeline step type: {step_type!r}")

        return result

    def _run_segment(self, result, image, method, params):
        from geobia.segmentation import segment

        t0 = time.perf_counter()
        labels = segment(image, method=method or "slic", **params)
        duration = time.perf_counter() - t0

        result.labels = labels
        n_segments = int(len(np.unique(labels)) - (1 if 0 in labels else 0))
        result.steps.append(StepResult(
            name="segment",
            duration_s=duration,
            metadata={"method": method, "n_segments": n_segments, **params},
        ))
        return result

    def _run_extract(self, result, image, categories, params):
        from geobia.features import extract

        if isinstance(categories, str):
            categories = [categories]

        t0 = time.perf_counter()
        features = extract(image, result.labels, categories=categories, **params)
        duration = time.perf_counter() - t0

        result.features = features
        result.steps.append(StepResult(
            name="extract",
            duration_s=duration,
            metadata={"categories": categories, "n_features": len(features.columns)},
        ))
        return result

    def _run_classify(self, result, method, params, training):
        from geobia.classification import classify

        training_labels = None
        if training is not None:
            if isinstance(training, str):
                from geobia.io.vector import read_training_samples
                training_labels = read_training_samples(
                    training, result.labels, result.meta
                )
            else:
                training_labels = training

        t0 = time.perf_counter()
        predictions = classify(
            result.features,
            method=method or "kmeans",
            training_labels=training_labels,
            **params,
        )
        duration = time.perf_counter() - t0

        result.predictions = predictions
        result.steps.append(StepResult(
            name="classify",
            duration_s=duration,
            metadata={"method": method, "n_classes": int(predictions.nunique())},
        ))
        return result

    def to_json(self) -> str:
        """Serialize pipeline definition to a JSON string."""
        definition = {
            "version": "1.0",
            "steps": [
                {
                    "type": step[0],
                    "method": step[1] if len(step) > 1 else None,
                    "params": step[2] if len(step) > 2 else {},
                }
                for step in self.steps
            ],
        }
        return json.dumps(definition, indent=2)

    def save(self, path: str | Path) -> None:
        """Save pipeline definition to JSON file."""
        Path(path).write_text(self.to_json())

    @classmethod
    def load_string(cls, json_string: str) -> "Pipeline":
        """Load a pipeline definition from a JSON string."""
        definition = json.loads(json_string)
        steps = []
        for step_def in definition["steps"]:
            steps.append((
                step_def["type"],
                step_def.get("method"),
                step_def.get("params", {}),
            ))
        return cls(steps)

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        """Load a pipeline definition from a JSON file."""
        return cls.load_string(Path(path).read_text())
