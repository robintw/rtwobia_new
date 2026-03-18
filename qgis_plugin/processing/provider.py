"""QGIS Processing provider registering all GeoOBIA algorithms."""

from qgis.core import QgsProcessingProvider

from .segmentation_alg import SegmentationAlgorithm
from .features_alg import FeatureExtractionAlgorithm
from .classification_alg import ClassificationAlgorithm


class GeobiaProvider(QgsProcessingProvider):

    def id(self):
        return "geobia"

    def name(self):
        return "GeoOBIA"

    def longName(self):
        return "GeoOBIA -- Object-Based Image Analysis"

    def loadAlgorithms(self):
        self.addAlgorithm(SegmentationAlgorithm())
        self.addAlgorithm(FeatureExtractionAlgorithm())
        self.addAlgorithm(ClassificationAlgorithm())
