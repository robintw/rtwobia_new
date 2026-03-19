"""Multi-scale segmentation as a QGIS Processing algorithm."""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

METHODS = ["slic", "felzenszwalb", "watershed"]


class MultiscaleSegmentationAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    METHOD = "METHOD"
    N_LEVELS = "N_LEVELS"
    SCALE_PARAMS = "SCALE_PARAMS"
    OUTPUT_FINE = "OUTPUT_FINE"
    OUTPUT_COARSE = "OUTPUT_COARSE"

    def name(self):
        return "multiscale_segment"

    def displayName(self):
        return "Multi-Scale Segmentation"

    def group(self):
        return "Segmentation"

    def groupId(self):
        return "segmentation"

    def shortHelpString(self):
        return (
            "Segment an image at multiple scales to produce a hierarchy "
            "of nested segments.\n\n"
            "Outputs the finest and coarsest scale label rasters. "
            "The hierarchy can be used for cross-scale feature analysis."
        )

    def createInstance(self):
        return MultiscaleSegmentationAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, "Input image"))

        self.addParameter(QgsProcessingParameterEnum(
            self.METHOD, "Algorithm",
            options=METHODS, defaultValue=0))

        self.addParameter(QgsProcessingParameterNumber(
            self.N_LEVELS, "Number of scale levels",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3, minValue=2, maxValue=10))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_FINE, "Output — finest scale labels"))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_COARSE, "Output — coarsest scale labels"))

    def processAlgorithm(self, parameters, context, feedback):
        from geobia.io.raster import read_raster, write_raster
        from geobia.segmentation.multiscale import segment_multiscale

        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
        method = METHODS[method_idx]
        n_levels = self.parameterAsInt(parameters, self.N_LEVELS, context)
        output_fine = self.parameterAsOutputLayer(
            parameters, self.OUTPUT_FINE, context)
        output_coarse = self.parameterAsOutputLayer(
            parameters, self.OUTPUT_COARSE, context)

        feedback.pushInfo(f"Reading {layer.source()}...")
        image, meta = read_raster(layer.source())

        feedback.pushInfo(
            f"Running multi-scale {method} segmentation "
            f"with {n_levels} levels...")
        feedback.setProgress(10)

        # Generate scale parameters based on n_levels
        scales = _generate_scales(method, n_levels)

        hierarchy = segment_multiscale(image, method=method, scales=scales)

        feedback.pushInfo(
            f"Done: {hierarchy.n_levels} levels, "
            f"finest={hierarchy.finest.n_segments} segments, "
            f"coarsest={hierarchy.coarsest.n_segments} segments")
        feedback.setProgress(80)

        write_raster(output_fine, hierarchy.finest.labels, meta, dtype="int32")
        write_raster(output_coarse, hierarchy.coarsest.labels, meta, dtype="int32")

        feedback.setProgress(100)
        return {
            self.OUTPUT_FINE: output_fine,
            self.OUTPUT_COARSE: output_coarse,
        }


def _generate_scales(method: str, n_levels: int) -> list[dict]:
    """Generate evenly spaced scale parameters for n levels."""
    if method == "slic":
        # Log-spaced from many to few segments
        import numpy as np
        values = np.logspace(3, 1.5, n_levels).astype(int)
        return [{"n_segments": int(v), "compactness": 10} for v in values]
    elif method == "felzenszwalb":
        import numpy as np
        values = np.logspace(1.5, 2.8, n_levels)
        return [
            {"scale": float(v), "min_size": max(20, int(v / 3))}
            for v in values
        ]
    elif method == "watershed":
        import numpy as np
        values = np.logspace(3, 1.5, n_levels).astype(int)
        return [
            {"markers": int(v), "min_distance": max(3, 30 - int(v / 50))}
            for v in values
        ]
    # Fallback
    import numpy as np
    values = np.logspace(3, 1.5, n_levels).astype(int)
    return [{"n_segments": int(v)} for v in values]
