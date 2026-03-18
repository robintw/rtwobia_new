"""GeoOBIA QGIS Plugin -- Object-Based Image Analysis."""


def classFactory(iface):  # noqa: N802 — QGIS convention
    from .geoobia_plugin import GeobiaPlugin
    return GeobiaPlugin(iface)
