"""Classification as a QGIS Processing algorithm."""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
)

METHODS = ["random_forest", "svm", "gradient_boosting", "kmeans", "gmm", "dbscan"]
SUPERVISED_METHODS = {"random_forest", "svm", "gradient_boosting"}


class ClassificationAlgorithm(QgsProcessingAlgorithm):

    FEATURES = "FEATURES"
    METHOD = "METHOD"
    TRAINING = "TRAINING"
    SEGMENTS = "SEGMENTS"
    N_CLUSTERS = "N_CLUSTERS"
    N_ESTIMATORS = "N_ESTIMATORS"
    OUTPUT = "OUTPUT"

    def name(self):
        return "classify"

    def displayName(self):
        return "Classify Segments"

    def group(self):
        return "Classification"

    def groupId(self):
        return "classification"

    def shortHelpString(self):
        return (
            "Classify segments using their extracted features.\n\n"
            "Supervised methods (Random Forest, SVM, Gradient Boosting) "
            "require training samples.\n"
            "Unsupervised methods (K-Means, GMM, DBSCAN) cluster automatically."
        )

    def createInstance(self):
        return ClassificationAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(
            self.FEATURES, "Input features file (Parquet)",
            extension="parquet"))

        self.addParameter(QgsProcessingParameterEnum(
            self.METHOD, "Classification method",
            options=METHODS, defaultValue=1))

        self.addParameter(QgsProcessingParameterVectorLayer(
            self.TRAINING, "Training samples (for supervised)",
            optional=True))

        self.addParameter(QgsProcessingParameterRasterLayer(
            self.SEGMENTS, "Segment labels raster (for supervised)",
            optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.N_CLUSTERS, "Number of clusters (K-Means/GMM)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=8, minValue=2, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.N_ESTIMATORS, "Number of trees (Random Forest)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=100, minValue=1, optional=True))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output classification file",
            fileFilter="Parquet (*.parquet)"))

    def processAlgorithm(self, parameters, context, feedback):
        import pandas as pd
        from geobia.classification import classify

        features_path = self.parameterAsString(parameters, self.FEATURES, context)
        method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
        method = METHODS[method_idx]
        output_path = self.parameterAsString(parameters, self.OUTPUT, context)

        feedback.pushInfo(f"Reading features from {features_path}...")
        features = pd.read_parquet(features_path)

        params = {}
        training_labels = None

        if method in SUPERVISED_METHODS:
            training_layer = self.parameterAsVectorLayer(
                parameters, self.TRAINING, context)
            seg_layer = self.parameterAsRasterLayer(
                parameters, self.SEGMENTS, context)

            if training_layer is None or seg_layer is None:
                raise ValueError(
                    "Training samples and segment labels are required "
                    "for supervised classification.")

            from geobia.io.raster import read_raster
            from geobia.io.vector import read_training_samples

            seg_data, meta = read_raster(seg_layer.source())
            labels = seg_data[0]
            training_labels = read_training_samples(
                training_layer.source(), labels, meta)
            params["n_estimators"] = self.parameterAsInt(
                parameters, self.N_ESTIMATORS, context)
        elif method in ("kmeans", "gmm"):
            params["n_clusters"] = self.parameterAsInt(
                parameters, self.N_CLUSTERS, context)

        feedback.pushInfo(f"Classifying with {method}...")
        feedback.setProgress(30)

        predictions = classify(
            features, method=method,
            training_labels=training_labels, **params)

        result = features.copy()
        result["class_label"] = predictions

        feedback.pushInfo(f"Writing results to {output_path}...")
        feedback.setProgress(80)
        result.to_parquet(output_path)

        feedback.setProgress(100)
        return {self.OUTPUT: output_path}
