"""Segmentation panel with algorithm selection, dynamic params, and preview."""

import os
import tempfile
import traceback
from collections import OrderedDict

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsMapLayerProxyModel,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsTask,
    QgsApplication,
    Qgis,
)
from qgis.gui import QgsMapLayerComboBox

from .schema_widgets import build_param_widgets, collect_param_values, create_param_group

TAG = "GeoOBIA"


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


class SegmentationPanel(QWidget):
    def __init__(self, iface, state, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.state = state
        self._param_widgets = OrderedDict()
        self._param_group = None
        log("SegmentationPanel.__init__ starting")
        try:
            self._setup_ui()
            log("SegmentationPanel.__init__ completed OK")
        except Exception:
            log(f"SegmentationPanel.__init__ FAILED:\n{traceback.format_exc()}",
                Qgis.Critical)

    def _setup_ui(self):
        log("_setup_ui starting")
        layout = QVBoxLayout()

        # Input layer selector
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        self._layer_combo = QgsMapLayerComboBox()
        self._layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        input_layout.addWidget(QLabel("Raster layer:"))
        input_layout.addWidget(self._layer_combo)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Algorithm selector
        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout()
        self._method_combo = QComboBox()
        self._method_combo.addItems(["slic", "felzenszwalb", "shepherd"])
        algo_layout.addWidget(self._method_combo)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # Dynamic parameter area (placeholder, rebuilt on method change)
        self._params_container = QVBoxLayout()
        layout.addLayout(self._params_container)

        # Buttons — connect BEFORE anything that might fail
        btn_layout = QHBoxLayout()
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setToolTip(
            "Segment only the current map canvas extent for quick feedback")
        self._preview_btn.clicked.connect(self._on_preview)
        btn_layout.addWidget(self._preview_btn)

        self._run_btn = QPushButton("Run")
        self._run_btn.setToolTip("Segment the full image")
        self._run_btn.clicked.connect(self._on_run)
        btn_layout.addWidget(self._run_btn)
        layout.addLayout(btn_layout)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Status
        self._status = QLabel("")
        layout.addWidget(self._status)

        layout.addStretch()
        self.setLayout(layout)
        log("_setup_ui layout done, connecting signals and loading params")

        # Connect algorithm change AFTER layout is complete
        self._method_combo.currentTextChanged.connect(self._on_method_changed)

        # Initialize parameter panel for default method
        self._on_method_changed(self._method_combo.currentText())
        log("_setup_ui complete")

    def _on_method_changed(self, method: str):
        """Rebuild the parameter panel for the selected algorithm."""
        try:
            from geobia.segmentation import _REGISTRY
        except Exception:
            log("Cannot import geobia.segmentation — is it installed?", Qgis.Warning)
            return

        # Remove old parameter group
        if self._param_group is not None:
            self._params_container.removeWidget(self._param_group)
            self._param_group.deleteLater()
            self._param_group = None

        # Build new widgets from the algorithm's JSON Schema
        cls = _REGISTRY.get(method)
        if cls is None:
            log(f"Method '{method}' not found in registry", Qgis.Warning)
            return

        schema = cls.get_param_schema()
        self._param_widgets = build_param_widgets(schema)
        self._param_group = create_param_group("Parameters", self._param_widgets)
        self._params_container.addWidget(self._param_group)
        log(f"Loaded params for {method}: {list(self._param_widgets.keys())}")

    def _get_input_layer(self):
        layer = self._layer_combo.currentLayer()
        if layer is None:
            QMessageBox.warning(self, "GeoOBIA", "Select an input raster layer.")
            log("No input layer selected", Qgis.Warning)
        return layer

    def _collect_params(self) -> dict:
        """Collect algorithm parameters from the dynamic widgets."""
        params = collect_param_values(self._param_widgets)
        # Remove None values
        return {k: v for k, v in params.items() if v is not None}

    def _on_preview(self):
        """Segment a small region matching the current canvas extent."""
        log("Preview button clicked")
        layer = self._get_input_layer()
        if layer is None:
            return

        method = self._method_combo.currentText()
        params = self._collect_params()
        log(f"Preview: method={method}, params={params}, source={layer.source()}")

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Running preview...")

        task = _PreviewTask(
            self.iface, layer, method, params, self._on_task_done)
        QgsApplication.taskManager().addTask(task)
        log("Preview task submitted to task manager")

    def _on_run(self):
        """Segment the full image."""
        log("Run button clicked")
        layer = self._get_input_layer()
        if layer is None:
            return

        method = self._method_combo.currentText()
        params = self._collect_params()
        log(f"Run: method={method}, params={params}, source={layer.source()}")

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._status.setText("Segmenting full image...")

        task = _SegmentTask(
            self.iface, self.state, layer, method, params, self._on_task_done)
        QgsApplication.taskManager().addTask(task)
        log("Segment task submitted to task manager")

    def _on_task_done(self, success: bool, message: str):
        """Called when a segmentation task finishes."""
        log(f"Task done: success={success}, message={message}")
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText(message)
        if not success:
            QMessageBox.warning(self, "GeoOBIA", message)


class _SegmentTask(QgsTask):
    """Run segmentation in a background thread."""

    def __init__(self, iface, state, layer, method, params, callback):
        super().__init__(f"GeoOBIA: Segment ({method})")
        self.iface = iface
        self.state = state
        self.layer_source = layer.source()
        self.layer = layer
        self.method = method
        self.params = params
        self.callback = callback
        self.output_path = None
        self.n_segments = 0
        self.error_msg = ""

    def run(self):
        log(f"SegmentTask.run() started: method={self.method}, source={self.layer_source}")
        try:
            from geobia.io.raster import read_raster, write_raster
            from geobia.segmentation import segment

            self.setProgress(5)
            log("Reading raster...")
            image, meta = read_raster(self.layer_source)
            log(f"Image shape: {image.shape}, dtype: {image.dtype}")

            self.setProgress(10)
            log(f"Segmenting with {self.method}, params={self.params}...")
            labels = segment(image, method=self.method, **self.params)

            self.setProgress(80)
            self.n_segments = int(labels.max())
            log(f"Segmentation done: {self.n_segments} segments")

            # Write to a temp file next to the input
            fd, self.output_path = tempfile.mkstemp(
                suffix="_segments.tif",
                prefix=f"geobia_{self.method}_",
                dir=os.path.dirname(self.layer_source) or tempfile.gettempdir(),
            )
            os.close(fd)
            write_raster(self.output_path, labels, meta, dtype="int32")
            log(f"Wrote segments to {self.output_path}")

            # Stash for other panels
            self.state.labels_array = labels
            self.state.meta = meta
            self.state.input_layer = self.layer

            self.setProgress(100)
            return True
        except Exception as e:
            self.error_msg = str(e)
            log(f"SegmentTask.run() FAILED:\n{traceback.format_exc()}", Qgis.Critical)
            return False

    def finished(self, result):
        log(f"SegmentTask.finished() called: result={result}")
        if result and self.output_path:
            name = f"Segments ({self.method})"
            rlayer = QgsRasterLayer(self.output_path, name)
            if rlayer.isValid():
                QgsProject.instance().addMapLayer(rlayer)
                self.state.labels_layer = rlayer
                _apply_segment_style(rlayer)
                log(f"Added layer '{name}' to project")
            else:
                log(f"Created raster layer is invalid: {self.output_path}", Qgis.Warning)
            self.callback(True, f"{self.n_segments} segments created.")
        else:
            self.callback(False, f"Segmentation failed: {self.error_msg}")


class _PreviewTask(QgsTask):
    """Run segmentation on the visible canvas extent only."""

    def __init__(self, iface, layer, method, params, callback):
        super().__init__(f"GeoOBIA: Preview ({method})")
        self.iface = iface
        self.layer = layer
        self.layer_source = layer.source()
        self.method = method
        self.params = params
        self.callback = callback
        self.output_path = None
        self.n_segments = 0
        self.error_msg = ""

        # Capture canvas extent on the main thread
        canvas = iface.mapCanvas()
        self.canvas_extent = canvas.extent()
        self.canvas_crs = canvas.mapSettings().destinationCrs()
        log(f"Preview extent: {self.canvas_extent.toString()}")

    def run(self):
        log(f"PreviewTask.run() started: method={self.method}, source={self.layer_source}")
        try:
            import numpy as np
            import rasterio
            from rasterio.windows import from_bounds
            from geobia.io.raster import write_raster
            from geobia.segmentation import segment

            self.setProgress(5)

            with rasterio.open(self.layer_source) as ds:
                log(f"Raster bounds: {ds.bounds}, size: {ds.width}x{ds.height}")
                # Transform canvas extent to raster pixel window
                try:
                    window = from_bounds(
                        self.canvas_extent.xMinimum(),
                        self.canvas_extent.yMinimum(),
                        self.canvas_extent.xMaximum(),
                        self.canvas_extent.yMaximum(),
                        transform=ds.transform,
                    )
                    log(f"Window from bounds: {window}")
                except Exception as e:
                    log(f"from_bounds failed ({e}), using full image", Qgis.Warning)
                    window = rasterio.windows.Window(0, 0, ds.width, ds.height)

                # Clip to image bounds and limit preview size
                window = window.intersection(
                    rasterio.windows.Window(0, 0, ds.width, ds.height))

                max_preview = 512
                col_off = int(window.col_off)
                row_off = int(window.row_off)
                win_w = min(int(window.width), max_preview)
                win_h = min(int(window.height), max_preview)
                window = rasterio.windows.Window(col_off, row_off, win_w, win_h)
                log(f"Final preview window: {window}")

                image = ds.read(window=window)
                meta = {
                    "crs": ds.crs,
                    "transform": ds.window_transform(window),
                    "nodata": ds.nodata,
                    "width": win_w,
                    "height": win_h,
                }

            log(f"Preview image shape: {image.shape}, dtype: {image.dtype}")

            if image.size == 0:
                self.error_msg = "Preview region is empty."
                log("Preview region is empty", Qgis.Warning)
                return False

            self.setProgress(20)
            log(f"Segmenting preview with {self.method}, params={self.params}...")
            labels = segment(image, method=self.method, **self.params)

            self.setProgress(80)
            self.n_segments = int(labels.max())
            log(f"Preview segmentation done: {self.n_segments} segments")

            fd, self.output_path = tempfile.mkstemp(
                suffix="_preview.tif", prefix="geobia_preview_")
            os.close(fd)
            write_raster(self.output_path, labels, meta, dtype="int32")
            log(f"Wrote preview to {self.output_path}")

            self.setProgress(100)
            return True
        except Exception as e:
            self.error_msg = str(e)
            log(f"PreviewTask.run() FAILED:\n{traceback.format_exc()}", Qgis.Critical)
            return False

    def finished(self, result):
        log(f"PreviewTask.finished() called: result={result}")
        if result and self.output_path:
            name = f"Preview ({self.method})"
            # Remove previous preview layers
            for lyr in QgsProject.instance().mapLayersByName(name):
                QgsProject.instance().removeMapLayer(lyr.id())

            rlayer = QgsRasterLayer(self.output_path, name)
            if rlayer.isValid():
                QgsProject.instance().addMapLayer(rlayer)
                _apply_segment_style(rlayer)
                log(f"Added preview layer '{name}' to project")
            else:
                log(f"Preview raster layer is invalid: {self.output_path}", Qgis.Warning)
            self.callback(True, f"Preview: {self.n_segments} segments")
        else:
            self.callback(False, f"Preview failed: {self.error_msg}")


def _apply_segment_style(layer: QgsRasterLayer):
    """Apply a singleband pseudocolor style to make segment boundaries visible."""
    from qgis.core import (
        QgsRasterShader,
        QgsSingleBandPseudoColorRenderer,
        QgsColorRampShader,
        QgsStyle,
    )

    shader = QgsRasterShader()
    color_ramp_shader = QgsColorRampShader()
    color_ramp_shader.setColorRampType(QgsColorRampShader.Interpolated)

    # Use a spectral color ramp for visual distinction
    style = QgsStyle.defaultStyle()
    ramp = style.colorRamp("Spectral")
    if ramp:
        color_ramp_shader.setSourceColorRamp(ramp)
    color_ramp_shader.classifyColorRamp(255)

    shader.setRasterShaderFunction(color_ramp_shader)
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    layer.setRenderer(renderer)
    layer.setOpacity(0.5)
    layer.triggerRepaint()
