"""Segmentation panel with algorithm selection, dynamic params, and preview."""

import os
import tempfile
import traceback
from collections import OrderedDict

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMapLayerProxyModel,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsSimpleFillSymbolLayer,
    QgsSingleSymbolRenderer,
    QgsSymbol,
    QgsVectorLayer,
    Qgis,
)
from qgis.PyQt.QtCore import QVariant
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

        # Buttons
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

        # Status
        self._status = QLabel("")
        layout.addWidget(self._status)

        layout.addStretch()
        self.setLayout(layout)

        # Connect algorithm change AFTER layout is complete
        self._method_combo.currentTextChanged.connect(self._on_method_changed)

        # Initialize parameter panel for default method
        self._on_method_changed(self._method_combo.currentText())

    def _on_method_changed(self, method: str):
        """Rebuild the parameter panel for the selected algorithm."""
        try:
            from geobia.segmentation import _REGISTRY
        except Exception:
            log("Cannot import geobia.segmentation — is it installed?", Qgis.Warning)
            return

        if self._param_group is not None:
            self._params_container.removeWidget(self._param_group)
            self._param_group.deleteLater()
            self._param_group = None

        cls = _REGISTRY.get(method)
        if cls is None:
            return

        schema = cls.get_param_schema()
        self._param_widgets = build_param_widgets(schema)
        self._param_group = create_param_group("Parameters", self._param_widgets)
        self._params_container.addWidget(self._param_group)

    def _get_input_layer(self):
        layer = self._layer_combo.currentLayer()
        if layer is None:
            QMessageBox.warning(self, "GeoOBIA", "Select an input raster layer.")
        return layer

    def _collect_params(self) -> dict:
        params = collect_param_values(self._param_widgets)
        return {k: v for k, v in params.items() if v is not None}

    def _on_preview(self):
        """Segment the current canvas extent and show outlines."""
        log("Preview button clicked")
        layer = self._get_input_layer()
        if layer is None:
            return

        method = self._method_combo.currentText()
        params = self._collect_params()
        log(f"Preview: method={method}, params={params}")

        self._status.setText("Running preview...")
        self._preview_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            import rasterio
            from rasterio.windows import from_bounds
            from geobia.segmentation import segment
            from geobia.utils.vectorize import vectorize_labels

            source = layer.source()
            with rasterio.open(source) as ds:
                canvas = self.iface.mapCanvas()
                extent = canvas.extent()
                log(f"Canvas extent: {extent.toString()}")
                log(f"Raster bounds: {ds.bounds}, size: {ds.width}x{ds.height}")

                try:
                    window = from_bounds(
                        extent.xMinimum(), extent.yMinimum(),
                        extent.xMaximum(), extent.yMaximum(),
                        transform=ds.transform,
                    )
                except Exception:
                    window = rasterio.windows.Window(0, 0, ds.width, ds.height)

                window = window.intersection(
                    rasterio.windows.Window(0, 0, ds.width, ds.height))

                # Limit preview size
                max_px = 512
                col_off = max(0, int(window.col_off))
                row_off = max(0, int(window.row_off))
                win_w = min(int(window.width), max_px)
                win_h = min(int(window.height), max_px)
                if win_w <= 0 or win_h <= 0:
                    self._status.setText("Preview region doesn't overlap the raster.")
                    return
                window = rasterio.windows.Window(col_off, row_off, win_w, win_h)
                log(f"Preview window: {window}")

                image = ds.read(window=window)
                meta = {
                    "crs": ds.crs,
                    "transform": ds.window_transform(window),
                    "nodata": ds.nodata,
                }

            log(f"Preview image: shape={image.shape}, dtype={image.dtype}")
            labels = segment(image, method=method, **params)
            n_segments = int(labels.max())
            log(f"Preview: {n_segments} segments")

            crs = meta.get("crs")
            gdf = vectorize_labels(labels, meta["transform"], crs)
            log(f"Vectorized {len(gdf)} polygons")

            name = f"Preview ({method})"
            _remove_layers_by_name(name)
            vlayer = _create_outline_layer(name, crs, gdf)
            QgsProject.instance().addMapLayer(vlayer)
            self.iface.mapCanvas().refresh()

            self._status.setText(f"Preview: {n_segments} segments")
            log(f"Preview layer '{name}' added to map")

        except Exception:
            msg = traceback.format_exc()
            log(f"Preview FAILED:\n{msg}", Qgis.Critical)
            self._status.setText("Preview failed — see Log Messages.")
            QMessageBox.warning(self, "GeoOBIA", f"Preview failed:\n{msg}")
        finally:
            QApplication.restoreOverrideCursor()
            self._preview_btn.setEnabled(True)

    def _on_run(self):
        """Segment the full image and show outlines."""
        log("Run button clicked")
        layer = self._get_input_layer()
        if layer is None:
            return

        method = self._method_combo.currentText()
        params = self._collect_params()
        log(f"Run: method={method}, params={params}")

        self._status.setText("Segmenting full image...")
        self._run_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            from geobia.io.raster import read_raster, write_raster
            from geobia.segmentation import segment
            from geobia.utils.vectorize import vectorize_labels

            source = layer.source()
            log(f"Reading {source}...")
            image, meta = read_raster(source)
            log(f"Image: shape={image.shape}, dtype={image.dtype}")

            log("Segmenting...")
            labels = segment(image, method=method, **params)
            n_segments = int(labels.max())
            log(f"Segmentation done: {n_segments} segments")

            # Save raster labels
            fd, output_path = tempfile.mkstemp(
                suffix="_segments.tif",
                prefix=f"geobia_{method}_",
                dir=os.path.dirname(source) or tempfile.gettempdir(),
            )
            os.close(fd)
            write_raster(output_path, labels, meta, dtype="int32")
            log(f"Wrote labels raster: {output_path}")

            # Store in state for other panels
            self.state.labels_array = labels
            self.state.meta = meta
            self.state.input_layer = layer

            # Add raster labels layer (hidden — used by classification panel)
            rlayer = QgsRasterLayer(output_path, f"Segments [raster]")
            if rlayer.isValid():
                QgsProject.instance().addMapLayer(rlayer, addToLegend=False)
                self.state.labels_layer = rlayer

            # Vectorize and display outlines
            log("Vectorizing...")
            crs = meta.get("crs")
            gdf = vectorize_labels(labels, meta["transform"], crs)
            log(f"Vectorized {len(gdf)} polygons")

            name = f"Segments ({method})"
            _remove_layers_by_name(name)
            vlayer = _create_outline_layer(name, crs, gdf)
            QgsProject.instance().addMapLayer(vlayer)
            self.iface.mapCanvas().refresh()

            self._status.setText(f"{n_segments} segments created.")
            log(f"Segment layer '{name}' added to map")

        except Exception:
            msg = traceback.format_exc()
            log(f"Run FAILED:\n{msg}", Qgis.Critical)
            self._status.setText("Segmentation failed — see Log Messages.")
            QMessageBox.warning(self, "GeoOBIA", f"Segmentation failed:\n{msg}")
        finally:
            QApplication.restoreOverrideCursor()
            self._run_btn.setEnabled(True)


def _remove_layers_by_name(name):
    """Remove all map layers with the given name."""
    for lyr in QgsProject.instance().mapLayersByName(name):
        QgsProject.instance().removeMapLayer(lyr.id())


def _create_outline_layer(name, crs, gdf):
    """Create a memory vector layer with bright yellow outlines.

    Args:
        name: Layer name.
        crs: rasterio CRS object or None.
        gdf: GeoDataFrame with segment_id and geometry columns.

    Returns:
        QgsVectorLayer ready to add to the project.
    """
    crs_str = str(crs) if crs else "EPSG:4326"
    vlayer = QgsVectorLayer(f"Polygon?crs={crs_str}", name, "memory")
    provider = vlayer.dataProvider()
    provider.addAttributes([QgsField("segment_id", QVariant.Int)])
    vlayer.updateFields()

    features = []
    for _, row in gdf.iterrows():
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromWkt(row.geometry.wkt))
        feat.setAttributes([int(row["segment_id"])])
        features.append(feat)

    provider.addFeatures(features)
    vlayer.updateExtents()

    # Bright yellow outline, fully transparent fill
    symbol = QgsSymbol.defaultSymbol(vlayer.geometryType())
    symbol.deleteSymbolLayer(0)
    outline = QgsSimpleFillSymbolLayer()
    outline.setFillColor(QColor(0, 0, 0, 0))
    outline.setStrokeColor(QColor(255, 255, 0))
    outline.setStrokeWidth(0.5)
    symbol.appendSymbolLayer(outline)
    vlayer.setRenderer(QgsSingleSymbolRenderer(symbol))

    log(f"Created outline layer '{name}': {len(features)} features, crs={crs_str}")
    return vlayer
