"""Shared vector layer utilities for the GeoOBIA QGIS plugin."""

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsProject,
    QgsSimpleFillSymbolLayer,
    QgsSingleSymbolRenderer,
    QgsSymbol,
    QgsVectorLayer,
)


def is_layer_alive(layer):
    """Check if a QgsMapLayer C++ object is still valid."""
    if layer is None:
        return False
    try:
        layer.id()
        return True
    except RuntimeError:
        return False


def remove_layer(layer):
    """Safely remove a layer from the project."""
    if layer is None:
        return
    try:
        layer.id()
        QgsProject.instance().removeMapLayer(layer.id())
    except (RuntimeError, Exception):
        pass


def remove_layers_by_name(name):
    """Remove all layers with the given name from the project."""
    for lyr in QgsProject.instance().mapLayersByName(name):
        try:
            QgsProject.instance().removeMapLayer(lyr.id())
        except (RuntimeError, Exception):
            pass


def create_polygon_layer(name, crs, fields=None):
    """Create an in-memory polygon vector layer.

    Args:
        name: Layer name.
        crs: CRS object or string (e.g. "EPSG:4326").
        fields: Optional list of QgsField objects.

    Returns:
        QgsVectorLayer with fields added and ready for features.
    """
    crs_str = str(crs) if crs else "EPSG:4326"
    vlayer = QgsVectorLayer(f"Polygon?crs={crs_str}", name, "memory")
    if fields:
        vlayer.dataProvider().addAttributes(fields)
        vlayer.updateFields()
    return vlayer


def add_gdf_features(vlayer, gdf, attr_columns=None):
    """Add features from a GeoDataFrame to a QgsVectorLayer.

    Args:
        vlayer: Target vector layer.
        gdf: GeoDataFrame with geometry column.
        attr_columns: List of column names to include as attributes.
            If None, includes only 'segment_id'.
    """
    provider = vlayer.dataProvider()
    if attr_columns is None:
        attr_columns = ["segment_id"]

    features = []
    for _, row_data in gdf.iterrows():
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromWkt(row_data.geometry.wkt))
        attrs = [int(row_data[c]) if c == "segment_id" else row_data.get(c)
                 for c in attr_columns]
        feat.setAttributes(attrs)
        features.append(feat)

    provider.addFeatures(features)
    vlayer.updateExtents()


def apply_outline_style(vlayer, stroke_color=QColor(255, 255, 0),
                        stroke_width=0.5):
    """Apply transparent fill with colored outline."""
    symbol = QgsSymbol.defaultSymbol(vlayer.geometryType())
    symbol.deleteSymbolLayer(0)
    outline = QgsSimpleFillSymbolLayer()
    outline.setFillColor(QColor(0, 0, 0, 0))
    outline.setStrokeColor(stroke_color)
    outline.setStrokeWidth(stroke_width)
    symbol.appendSymbolLayer(outline)
    vlayer.setRenderer(QgsSingleSymbolRenderer(symbol))
