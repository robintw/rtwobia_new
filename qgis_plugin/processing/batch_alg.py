"""Batch processing as a QGIS Processing algorithm."""

import os

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessing,
)

METHODS = ["slic", "felzenszwalb", "shepherd", "watershed"]
CLASSIFY_METHODS = ["none", "kmeans", "gmm", "random_forest"]


class BatchProcessingAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    SEG_METHOD = "SEG_METHOD"
    N_SEGMENTS = "N_SEGMENTS"
    CLASSIFY_METHOD = "CLASSIFY_METHOD"
    N_CLUSTERS = "N_CLUSTERS"
    TRAINING = "TRAINING"
    SEGMENTS_REF = "SEGMENTS_REF"
    MAX_WORKERS = "MAX_WORKERS"
    OUTPUT = "OUTPUT"

    def name(self):
        return "batch"

    def displayName(self):
        return "Batch Process Multiple Images"

    def group(self):
        return "Batch"

    def groupId(self):
        return "batch"

    def shortHelpString(self):
        return (
            "Run a segment-extract-classify pipeline on multiple raster "
            "images in parallel.\n\n"
            "Outputs segment label rasters to the output directory."
        )

    def createInstance(self):
        return BatchProcessingAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMultipleLayers(
            self.INPUT, "Input raster images",
            layerType=QgsProcessing.TypeRaster))

        self.addParameter(QgsProcessingParameterEnum(
            self.SEG_METHOD, "Segmentation algorithm",
            options=METHODS, defaultValue=0))

        self.addParameter(QgsProcessingParameterNumber(
            self.N_SEGMENTS, "Number of segments (SLIC/Watershed markers)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=500, minValue=1, optional=True))

        self.addParameter(QgsProcessingParameterEnum(
            self.CLASSIFY_METHOD, "Classification method (optional)",
            options=CLASSIFY_METHODS, defaultValue=0, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.N_CLUSTERS, "Number of clusters (K-Means/GMM)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=8, minValue=2, optional=True))

        self.addParameter(QgsProcessingParameterVectorLayer(
            self.TRAINING, "Training samples (for supervised)",
            optional=True))

        self.addParameter(QgsProcessingParameterRasterLayer(
            self.SEGMENTS_REF, "Reference segment labels (for supervised)",
            optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_WORKERS, "Max parallel workers",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1, minValue=1, optional=True))

        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT, "Output directory"))

    def processAlgorithm(self, parameters, context, feedback):
        from geobia.pipeline import Pipeline
        from geobia.batch import process_batch, batch_summary

        layers = self.parameterAsLayerList(parameters, self.INPUT, context)
        input_paths = [layer.source() for layer in layers]

        seg_method_idx = self.parameterAsEnum(parameters, self.SEG_METHOD, context)
        seg_method = METHODS[seg_method_idx]
        n_segments = self.parameterAsInt(parameters, self.N_SEGMENTS, context)
        clf_method_idx = self.parameterAsEnum(parameters, self.CLASSIFY_METHOD, context)
        clf_method = CLASSIFY_METHODS[clf_method_idx]
        n_clusters = self.parameterAsInt(parameters, self.N_CLUSTERS, context)
        max_workers = self.parameterAsInt(parameters, self.MAX_WORKERS, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT, context)

        # Build pipeline steps
        seg_params = {}
        if seg_method in ("slic", "watershed"):
            key = "n_segments" if seg_method == "slic" else "markers"
            seg_params[key] = n_segments

        steps = [
            ("segment", seg_method, seg_params),
            ("extract", ["spectral", "geometry"], {}),
        ]

        training_labels = None
        if clf_method != "none":
            clf_params = {}
            if clf_method in ("kmeans", "gmm"):
                clf_params["n_clusters"] = n_clusters
            elif clf_method == "random_forest":
                training_layer = self.parameterAsVectorLayer(
                    parameters, self.TRAINING, context)
                seg_layer = self.parameterAsRasterLayer(
                    parameters, self.SEGMENTS_REF, context)
                if training_layer and seg_layer:
                    from geobia.io.raster import read_raster
                    from geobia.io.vector import read_training_samples
                    seg_data, meta = read_raster(seg_layer.source())
                    training_labels = read_training_samples(
                        training_layer.source(), seg_data[0], meta)
            steps.append(("classify", clf_method, clf_params))

        pipeline = Pipeline(steps)

        feedback.pushInfo(
            f"Processing {len(input_paths)} images with {seg_method}...")

        def on_progress(completed, total):
            feedback.setProgress(int(completed / total * 100))

        results = process_batch(
            input_paths, output_dir, pipeline,
            training_labels=training_labels,
            max_workers=max_workers,
            progress_callback=on_progress,
        )

        summary = batch_summary(results)
        feedback.pushInfo(
            f"Done: {summary['succeeded']}/{summary['total']} succeeded, "
            f"{summary['total_segments']} total segments")
        if summary.get("errors"):
            for path, err in summary["errors"].items():
                feedback.reportError(f"{path}: {err}")

        return {self.OUTPUT: output_dir}
