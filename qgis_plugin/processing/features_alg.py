"""Feature extraction as a QGIS Processing algorithm."""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)


class FeatureExtractionAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    SEGMENTS = "SEGMENTS"
    SPECTRAL = "SPECTRAL"
    GEOMETRY = "GEOMETRY"
    TEXTURE = "TEXTURE"
    BAND_NAMES = "BAND_NAMES"
    OUTPUT = "OUTPUT"

    def name(self):
        return "extract_features"

    def displayName(self):
        return "Extract Features"

    def group(self):
        return "Features"

    def groupId(self):
        return "features"

    def shortHelpString(self):
        return (
            "Extract per-segment features from imagery.\n\n"
            "Categories:\n"
            "  - Spectral: per-band mean, std, min, max; NDVI/NDWI ratios\n"
            "  - Geometry: area, perimeter, compactness, elongation\n"
            "  - Texture: GLCM contrast, homogeneity, entropy, etc."
        )

    def createInstance(self):
        return FeatureExtractionAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, "Input image"))

        self.addParameter(QgsProcessingParameterRasterLayer(
            self.SEGMENTS, "Segment labels raster"))

        self.addParameter(QgsProcessingParameterBoolean(
            self.SPECTRAL, "Extract spectral features", defaultValue=True))

        self.addParameter(QgsProcessingParameterBoolean(
            self.GEOMETRY, "Extract geometry features", defaultValue=True))

        self.addParameter(QgsProcessingParameterBoolean(
            self.TEXTURE, "Extract texture features (GLCM)", defaultValue=False))

        self.addParameter(QgsProcessingParameterString(
            self.BAND_NAMES,
            "Band names (comma-separated, e.g. red,green,blue,nir)",
            optional=True))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output features file",
            fileFilter="Parquet (*.parquet)"))

    def processAlgorithm(self, parameters, context, feedback):
        from geobia.io.raster import read_raster
        from geobia.features import extract

        img_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        seg_layer = self.parameterAsRasterLayer(parameters, self.SEGMENTS, context)
        output_path = self.parameterAsString(parameters, self.OUTPUT, context)

        feedback.pushInfo(f"Reading image {img_layer.source()}...")
        image, meta = read_raster(img_layer.source())

        feedback.pushInfo(f"Reading segments {seg_layer.source()}...")
        seg_data, _ = read_raster(seg_layer.source())
        labels = seg_data[0]

        categories = []
        if self.parameterAsBool(parameters, self.SPECTRAL, context):
            categories.append("spectral")
        if self.parameterAsBool(parameters, self.GEOMETRY, context):
            categories.append("geometry")
        if self.parameterAsBool(parameters, self.TEXTURE, context):
            categories.append("texture")

        kwargs = {}
        band_names_str = self.parameterAsString(parameters, self.BAND_NAMES, context)
        if band_names_str:
            names = [n.strip() for n in band_names_str.split(",")]
            kwargs["band_names"] = {name: i for i, name in enumerate(names)}

        pixel_size = abs(meta["transform"].a) if meta.get("transform") else None
        if pixel_size:
            kwargs["pixel_size"] = pixel_size

        feedback.pushInfo(f"Extracting features: {', '.join(categories)}...")
        feedback.setProgress(20)

        features = extract(image, labels, categories=categories, **kwargs)

        feedback.pushInfo(f"Writing {len(features)} segments, {len(features.columns)} features...")
        feedback.setProgress(80)
        features.to_parquet(output_path)

        feedback.setProgress(100)
        return {self.OUTPUT: output_path}
