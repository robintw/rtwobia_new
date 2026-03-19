"""Change detection as a QGIS Processing algorithm."""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)

THRESHOLD_METHODS = ["otsu", "fixed"]


class ChangeDetectionAlgorithm(QgsProcessingAlgorithm):

    FEATURES_T1 = "FEATURES_T1"
    FEATURES_T2 = "FEATURES_T2"
    THRESHOLD_METHOD = "THRESHOLD_METHOD"
    THRESHOLD_VALUE = "THRESHOLD_VALUE"
    NORMALIZE = "NORMALIZE"
    OUTPUT = "OUTPUT"

    def name(self):
        return "change_detection"

    def displayName(self):
        return "Change Detection"

    def group(self):
        return "Analysis"

    def groupId(self):
        return "analysis"

    def shortHelpString(self):
        return (
            "Detect changes between two time periods by comparing "
            "per-segment feature vectors.\n\n"
            "Input: two Parquet feature files (same segmentation applied "
            "to imagery from different dates).\n"
            "Output: Parquet file with change magnitude and changed flag."
        )

    def createInstance(self):
        return ChangeDetectionAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(
            self.FEATURES_T1, "Features file — Time 1 (Parquet)",
            extension="parquet"))

        self.addParameter(QgsProcessingParameterFile(
            self.FEATURES_T2, "Features file — Time 2 (Parquet)",
            extension="parquet"))

        self.addParameter(QgsProcessingParameterEnum(
            self.THRESHOLD_METHOD, "Threshold method",
            options=THRESHOLD_METHODS, defaultValue=0))

        self.addParameter(QgsProcessingParameterNumber(
            self.THRESHOLD_VALUE, "Fixed threshold value (if method=fixed)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=2.0, minValue=0.0, optional=True))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output change detection file",
            fileFilter="Parquet (*.parquet)"))

    def processAlgorithm(self, parameters, context, feedback):
        import pandas as pd
        from geobia.change import (
            change_magnitude,
            detect_changes,
            change_summary,
        )

        t1_path = self.parameterAsString(parameters, self.FEATURES_T1, context)
        t2_path = self.parameterAsString(parameters, self.FEATURES_T2, context)
        thresh_idx = self.parameterAsEnum(parameters, self.THRESHOLD_METHOD, context)
        thresh_method = THRESHOLD_METHODS[thresh_idx]
        thresh_value = self.parameterAsDouble(parameters, self.THRESHOLD_VALUE, context)
        output_path = self.parameterAsString(parameters, self.OUTPUT, context)

        feedback.pushInfo(f"Reading features from {t1_path} and {t2_path}...")
        features_t1 = pd.read_parquet(t1_path)
        features_t2 = pd.read_parquet(t2_path)

        feedback.pushInfo("Computing change magnitude...")
        feedback.setProgress(20)

        threshold = thresh_method if thresh_method == "otsu" else thresh_value
        magnitude = change_magnitude(features_t1, features_t2, normalize=True)
        changed = detect_changes(
            features_t1, features_t2, threshold=threshold, normalize=True)

        feedback.setProgress(70)

        result = pd.DataFrame({
            "change_magnitude": magnitude,
            "changed": changed,
        })

        summary = change_summary(changed, features_t1, features_t2)
        feedback.pushInfo(
            f"Done: {summary['changed']}/{summary['total_segments']} segments "
            f"changed ({summary['pct_changed']}%)")

        result.to_parquet(output_path)
        feedback.setProgress(100)

        return {self.OUTPUT: output_path}
