"""Main QGIS plugin class -- toolbar, dock widget, Processing provider."""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QDockWidget, QTabWidget
from qgis.core import QgsApplication

from .processing.provider import GeobiaProvider
from .ui.segmentation_panel import SegmentationPanel
from .ui.features_panel import FeaturesPanel
from .ui.classification_panel import ClassificationPanel
from .ui.results_panel import ResultsPanel


class PluginState:
    """Shared mutable state passed between panels."""

    def __init__(self):
        self.input_layer = None
        self.labels_layer = None
        self.labels_array = None
        self.meta = None
        self.features_df = None
        self.predictions = None
        self.training_samples = {}  # segment_id -> class_name
        self.class_colors = {}      # class_name -> QColor


class GeobiaPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dock = None
        self.provider = None
        self.state = PluginState()

    def initGui(self):
        icon_path = QgsApplication.iconPath("mIconRaster.svg")
        self.action = QAction(QIcon(icon_path), "GeoOBIA", self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self._toggle_dock)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("&GeoOBIA", self.action)

        self.provider = GeobiaProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

        self._create_dock()

    def _create_dock(self):
        self.dock = QDockWidget("GeoOBIA", self.iface.mainWindow())
        self.dock.setObjectName("GeobiaDock")

        tabs = QTabWidget()
        tabs.addTab(SegmentationPanel(self.iface, self.state), "Segmentation")
        tabs.addTab(FeaturesPanel(self.iface, self.state), "Features")
        tabs.addTab(ClassificationPanel(self.iface, self.state), "Classification")
        tabs.addTab(ResultsPanel(self.iface, self.state), "Results")
        self.dock.setWidget(tabs)

        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.hide()

    def _toggle_dock(self, checked):
        if self.dock:
            self.dock.setVisible(checked)

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
