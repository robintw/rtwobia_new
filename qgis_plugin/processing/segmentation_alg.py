"""Segmentation as a QGIS Processing algorithm."""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)

METHODS = ["slic", "felzenszwalb", "shepherd", "watershed"]


class SegmentationAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    METHOD = "METHOD"
    OUTPUT = "OUTPUT"

    # SLIC
    N_SEGMENTS = "N_SEGMENTS"
    COMPACTNESS = "COMPACTNESS"
    SIGMA = "SIGMA"

    # Felzenszwalb
    SCALE = "SCALE"
    FZ_SIGMA = "FZ_SIGMA"
    MIN_SIZE = "MIN_SIZE"

    # Shepherd
    NUM_CLUSTERS = "NUM_CLUSTERS"
    MIN_N_PXLS = "MIN_N_PXLS"
    SAMPLING = "SAMPLING"

    # Watershed
    MARKERS = "MARKERS"
    MIN_DISTANCE = "MIN_DISTANCE"

    def name(self):
        return "segment"

    def displayName(self):
        return "Segment Image"

    def group(self):
        return "Segmentation"

    def groupId(self):
        return "segmentation"

    def shortHelpString(self):
        return (
            "Segment a raster image into spatially contiguous objects.\n\n"
            "Algorithms:\n"
            "  - SLIC: superpixel clustering (fast, regular shapes)\n"
            "  - Felzenszwalb: graph-based adaptive segmentation\n"
            "  - Shepherd: K-means seeded with iterative elimination\n"
            "  - Watershed: gradient-based watershed from local minima"
        )

    def createInstance(self):
        return SegmentationAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, "Input image"))

        self.addParameter(QgsProcessingParameterEnum(
            self.METHOD, "Algorithm", options=METHODS, defaultValue=0))

        # SLIC parameters
        self.addParameter(QgsProcessingParameterNumber(
            self.N_SEGMENTS, "Number of segments (SLIC)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=500, minValue=1, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.COMPACTNESS, "Compactness (SLIC)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=10.0, minValue=0.0, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.SIGMA, "Gaussian sigma (SLIC)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0, minValue=0.0, optional=True))

        # Felzenszwalb parameters
        self.addParameter(QgsProcessingParameterNumber(
            self.SCALE, "Scale (Felzenszwalb)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=100.0, minValue=0.0, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.FZ_SIGMA, "Gaussian sigma (Felzenszwalb)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.8, minValue=0.0, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_SIZE, "Minimum segment size (Felzenszwalb/Shepherd)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=50, minValue=1, optional=True))

        # Shepherd parameters
        self.addParameter(QgsProcessingParameterNumber(
            self.NUM_CLUSTERS, "K-means clusters (Shepherd)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=60, minValue=2, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_N_PXLS, "Minimum segment pixels (Shepherd)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=100, minValue=1, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.SAMPLING, "Subsampling rate (Shepherd)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=100, minValue=1, optional=True))

        # Watershed parameters
        self.addParameter(QgsProcessingParameterNumber(
            self.MARKERS, "Number of markers (Watershed)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=500, minValue=1, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_DISTANCE, "Min distance between markers (Watershed)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=10, minValue=1, optional=True))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, "Output segment labels"))

    def processAlgorithm(self, parameters, context, feedback):
        from geobia.io.raster import read_raster, write_raster
        from geobia.segmentation import segment

        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
        method = METHODS[method_idx]
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        feedback.pushInfo(f"Reading {layer.source()}...")
        image, meta = read_raster(layer.source())

        params = {}
        if method == "slic":
            params["n_segments"] = self.parameterAsInt(parameters, self.N_SEGMENTS, context)
            params["compactness"] = self.parameterAsDouble(parameters, self.COMPACTNESS, context)
            params["sigma"] = self.parameterAsDouble(parameters, self.SIGMA, context)
        elif method == "felzenszwalb":
            params["scale"] = self.parameterAsDouble(parameters, self.SCALE, context)
            params["sigma"] = self.parameterAsDouble(parameters, self.FZ_SIGMA, context)
            params["min_size"] = self.parameterAsInt(parameters, self.MIN_SIZE, context)
        elif method == "shepherd":
            params["num_clusters"] = self.parameterAsInt(parameters, self.NUM_CLUSTERS, context)
            params["min_n_pxls"] = self.parameterAsInt(parameters, self.MIN_N_PXLS, context)
            params["sampling"] = self.parameterAsInt(parameters, self.SAMPLING, context)
        elif method == "watershed":
            params["markers"] = self.parameterAsInt(parameters, self.MARKERS, context)
            params["min_distance"] = self.parameterAsInt(parameters, self.MIN_DISTANCE, context)

        feedback.pushInfo(f"Segmenting with {method}...")
        feedback.setProgress(10)

        labels = segment(image, method=method, **params)

        feedback.pushInfo(f"Writing {labels.max()} segments to {output_path}...")
        feedback.setProgress(80)
        write_raster(output_path, labels, meta, dtype="int32")

        feedback.setProgress(100)
        return {self.OUTPUT: output_path}
