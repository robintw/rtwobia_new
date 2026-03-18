"""GeoOBIA QGIS Plugin -- Object-Based Image Analysis.

IMPORTANT: When installing this plugin into QGIS, the folder name in
~/.../python/plugins/ must NOT be "geobia" — that would shadow the
geobia Python library.  Use "geobia_sketcher" or any other name that
differs from the library package name.

Example install:
    cp -r qgis_plugin  ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geobia_sketcher
"""


def classFactory(iface):  # noqa: N802 — QGIS convention
    from .geoobia_plugin import GeobiaPlugin
    return GeobiaPlugin(iface)
