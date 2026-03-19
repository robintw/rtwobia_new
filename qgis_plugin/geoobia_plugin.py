"""Main QGIS plugin class -- toolbar, dock widget, Processing provider."""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (
    QAction, QDockWidget, QFileDialog, QHBoxLayout, QMessageBox,
    QPushButton, QTabWidget, QVBoxLayout, QWidget,
)
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
    """One segmentation result with its parameters and data.

    Labels are evicted from memory after initial creation and reloaded
    from disk on demand to save memory for large images.
    """

    def __init__(self, method, params, labels_array, meta, raster_path,
                 gdf, n_segments):
        self.method = method
        self.params = params
        self._labels_array = labels_array  # may be evicted to None
        self.meta = meta
        self.raster_path = raster_path
        self.gdf = gdf  # vectorized GeoDataFrame
        self.n_segments = n_segments

    @property
    def labels_array(self):
        """Load labels from disk if evicted from memory."""
        if self._labels_array is None and self.raster_path:
            import os
            if os.path.exists(self.raster_path):
                from geobia.io.raster import read_raster
                data, _ = read_raster(self.raster_path)
                self._labels_array = data[0]
        return self._labels_array

    @labels_array.setter
    def labels_array(self, value):
        self._labels_array = value

    def evict_labels(self):
        """Release the labels array from memory (will reload from disk)."""
        self._labels_array = None

    @property
    def summary(self):
        """Short human-readable description for the gallery list."""
        parts = [f"{self.method.upper()} [{self.n_segments} segs]"]
        for k, v in self.params.items():
            parts.append(f"{k}={v}")
        return " | ".join(parts)


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

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(2, 2, 2, 2)

        tabs = QTabWidget()
        tabs.addTab(SegmentationPanel(self.iface, self.state), "Segmentation")
        tabs.addTab(FeaturesPanel(self.iface, self.state), "Features")
        tabs.addTab(ClassificationPanel(self.iface, self.state), "Classification")
        tabs.addTab(ResultsPanel(self.iface, self.state), "Results")
        container_layout.addWidget(tabs)

        # Pipeline save/load
        pipeline_row = QHBoxLayout()
        save_pipe_btn = QPushButton("Save Pipeline...")
        save_pipe_btn.setToolTip("Save the current workflow as a reusable JSON pipeline")
        save_pipe_btn.clicked.connect(self._on_save_pipeline)
        pipeline_row.addWidget(save_pipe_btn)
        load_pipe_btn = QPushButton("Load Pipeline...")
        load_pipe_btn.setToolTip("Load a pipeline JSON file to reproduce a workflow")
        load_pipe_btn.clicked.connect(self._on_load_pipeline)
        pipeline_row.addWidget(load_pipe_btn)
        container_layout.addLayout(pipeline_row)

        container.setLayout(container_layout)
        self.dock.setWidget(container)

        self.dock.visibilityChanged.connect(self._on_dock_visibility_changed)
        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        log("_create_dock complete — dock added")
        return True

    def _on_dock_visibility_changed(self, visible):
        """Sync toolbar button state when dock is closed via the X button."""
        if self.action is not None:
            self.action.setChecked(visible)

    def _on_save_pipeline(self):
        """Save the current workflow configuration as a Pipeline JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(), "Save Pipeline", "",
            "JSON Pipeline (*.json)")
        if not path:
            return
        try:
            from geobia.pipeline import Pipeline
            steps = self._build_pipeline_steps()
            if not steps:
                QMessageBox.warning(
                    self.iface.mainWindow(), "GeoOBIA",
                    "No workflow steps to save. Run segmentation, "
                    "extraction, or classification first.")
                return
            pipeline = Pipeline(steps)
            pipeline.save(path)
            log(f"Pipeline saved to {path}")
        except Exception as e:
            log(f"Save pipeline failed: {e}", Qgis.Critical)
            QMessageBox.warning(
                self.iface.mainWindow(), "GeoOBIA",
                f"Failed to save pipeline: {e}")

    def _on_load_pipeline(self):
        """Load a Pipeline JSON and display its configuration."""
        path, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(), "Load Pipeline", "",
            "JSON Pipeline (*.json)")
        if not path:
            return
        try:
            from geobia.pipeline import Pipeline
            pipeline = Pipeline.load(path)
            info_parts = []
            for step in pipeline.steps:
                step_type = step[0]
                method = step[1] if len(step) > 1 else "default"
                params = step[2] if len(step) > 2 else {}
                info_parts.append(f"{step_type}: {method} {params}")
            info = "\n".join(info_parts)
            QMessageBox.information(
                self.iface.mainWindow(), "GeoOBIA — Pipeline Loaded",
                f"Pipeline from {path}:\n\n{info}")
            log(f"Pipeline loaded from {path}: {len(pipeline.steps)} steps")
        except Exception as e:
            log(f"Load pipeline failed: {e}", Qgis.Critical)
            QMessageBox.warning(
                self.iface.mainWindow(), "GeoOBIA",
                f"Failed to load pipeline: {e}")

    def _build_pipeline_steps(self):
        """Build pipeline step tuples from the current plugin state."""
        steps = []
        seg = self.state.active_seg
        if seg is not None:
            steps.append(("segment", seg.method, seg.params))

        if self.state.features_df is not None:
            steps.append(("extract", ["spectral", "geometry"], {}))

        if self.state.predictions is not None:
            steps.append(("classify", "kmeans", {}))

        return steps

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
