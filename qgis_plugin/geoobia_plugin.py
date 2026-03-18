"""Main QGIS plugin class -- toolbar, dock widget, Processing provider."""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QDockWidget, QMessageBox, QTabWidget
from qgis.core import QgsApplication, QgsMessageLog, Qgis

TAG = "GeoOBIA"


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


def _check_geobia():
    """Return True if the geobia library is importable."""
    try:
        import geobia.segmentation  # noqa: F401
        return True
    except ImportError:
        return False


class SegmentationRun:
    """One segmentation result with its parameters and data."""

    def __init__(self, method, params, labels_array, meta, raster_path,
                 gdf, n_segments):
        self.method = method
        self.params = params
        self.labels_array = labels_array
        self.meta = meta
        self.raster_path = raster_path
        self.gdf = gdf  # vectorized GeoDataFrame
        self.n_segments = n_segments

    @property
    def summary(self):
        """Short human-readable description for the gallery list."""
        parts = [self.method.upper()]
        for k, v in self.params.items():
            parts.append(f"{k}={v}")
        return " | ".join(parts) + f"  [{self.n_segments} segs]"


class PluginState:
    """Shared mutable state passed between panels."""

    def __init__(self):
        self.input_layer = None
        # Segmentation gallery
        self.seg_runs = []          # list[SegmentationRun]
        self.active_seg_index = -1  # index into seg_runs, -1 = none
        # Convenience accessors for the active segmentation
        self.labels_layer = None    # QgsRasterLayer (hidden)
        self.features_df = None
        self.predictions = None
        self.training_samples = {}  # segment_id -> class_name
        self.class_colors = {}      # class_name -> QColor

    @property
    def active_seg(self):
        if 0 <= self.active_seg_index < len(self.seg_runs):
            return self.seg_runs[self.active_seg_index]
        return None


class GeobiaPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dock = None
        self.provider = None
        self.state = PluginState()

    def initGui(self):
        log("initGui starting")
        icon_path = QgsApplication.iconPath("mIconRaster.svg")
        self.action = QAction(QIcon(icon_path), "GeoOBIA", self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self._toggle_dock)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("&GeoOBIA", self.action)

        # Register Processing provider (defers geobia imports to algorithm runtime)
        from .processing.provider import GeobiaProvider
        self.provider = GeobiaProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)
        log("initGui complete")

    def _toggle_dock(self, checked):
        log(f"_toggle_dock called: checked={checked}")
        if checked:
            if self.dock is None:
                if not self._create_dock():
                    self.action.setChecked(False)
                    return
            self.dock.show()
        elif self.dock:
            self.dock.hide()

    def _create_dock(self) -> bool:
        """Create the dock widget. Returns False if geobia is not available."""
        log("_create_dock starting")
        if not _check_geobia():
            QMessageBox.critical(
                self.iface.mainWindow(),
                "GeoOBIA",
                "The 'geobia' Python library is not installed in this "
                "Python environment.\n\n"
                "Install it with:\n"
                "  pip install -e /path/to/geobia\n\n"
                "Also ensure the plugin folder is NOT named 'geobia' — "
                "that shadows the library. Use 'geobia_sketcher' or similar.",
            )
            return False

        from .ui.segmentation_panel import SegmentationPanel
        from .ui.features_panel import FeaturesPanel
        from .ui.classification_panel import ClassificationPanel
        from .ui.results_panel import ResultsPanel

        self.dock = QDockWidget("GeoOBIA", self.iface.mainWindow())
        self.dock.setObjectName("GeobiaDock")

        tabs = QTabWidget()
        tabs.addTab(SegmentationPanel(self.iface, self.state), "Segmentation")
        tabs.addTab(FeaturesPanel(self.iface, self.state), "Features")
        tabs.addTab(ClassificationPanel(self.iface, self.state), "Classification")
        tabs.addTab(ResultsPanel(self.iface, self.state), "Results")
        self.dock.setWidget(tabs)

        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        log("_create_dock complete — dock added")
        return True

    def unload(self):
        if self.action:
            self.iface.removeToolBarIcon(self.action)
            self.iface.removePluginRasterMenu("&GeoOBIA", self.action)
        if self.dock:
            self.iface.removeDockWidget(self.dock)
            self.dock = None
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
