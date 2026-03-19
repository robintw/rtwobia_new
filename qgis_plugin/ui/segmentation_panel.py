"""Segmentation panel with algorithm selection, dynamic params, and gallery."""

import os
import tempfile
import traceback
from collections import OrderedDict

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
_OUTLINE_LAYER_NAME = "GeoOBIA Segments"


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


class SegmentationPanel(QWidget):
    def __init__(self, iface, state, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.state = state
        self._param_widgets = OrderedDict()
        self._param_group = None
        self._outline_layer = None  # current vector layer on the map
        try:
            self._setup_ui()
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
        self._method_combo.addItems(["slic", "felzenszwalb", "shepherd", "watershed"])
        algo_layout.addWidget(self._method_combo)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # Dynamic parameter area
        self._params_container = QVBoxLayout()
        layout.addLayout(self._params_container)

        # Run button
        self._run_btn = QPushButton("Run Segmentation")
        self._run_btn.setToolTip("Segment the full image")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        # Status
        self._status = QLabel("")
        layout.addWidget(self._status)

        # --- Gallery ---
        gallery_group = QGroupBox("Segmentation Runs")
        gallery_layout = QVBoxLayout()

        self._gallery_list = QListWidget()
        self._gallery_list.currentRowChanged.connect(self._on_gallery_select)
        gallery_layout.addWidget(self._gallery_list)

        gallery_btn_layout = QHBoxLayout()

        self._use_btn = QPushButton("Use for Extraction")
        self._use_btn.setToolTip("Set the selected segmentation as active for feature extraction and classification")
        self._use_btn.clicked.connect(self._on_use)
        self._use_btn.setEnabled(False)
        gallery_btn_layout.addWidget(self._use_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setToolTip("Remove the selected segmentation run")
        self._delete_btn.clicked.connect(self._on_delete)
        self._delete_btn.setEnabled(False)
        gallery_btn_layout.addWidget(self._delete_btn)

        gallery_layout.addLayout(gallery_btn_layout)
        gallery_group.setLayout(gallery_layout)
        layout.addWidget(gallery_group)

        layout.addStretch()
        self.setLayout(layout)

        # Connect algorithm change AFTER layout is complete
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        self._on_method_changed(self._method_combo.currentText())

    def _on_method_changed(self, method: str):
        """Rebuild the parameter panel for the selected algorithm."""
        try:
            from geobia.segmentation import _REGISTRY
        except Exception:
            log("Cannot import geobia.segmentation", Qgis.Warning)
            return

        # Clear old widgets first — they'll be destroyed by deleteLater
        self._param_widgets = OrderedDict()
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

    # --- Run segmentation ---

    def _on_run(self):
        layer = self._get_input_layer()
        if layer is None:
            return

        method = self._method_combo.currentText()
        params = self._collect_params()
        log(f"Run: method={method}, params={params}")

        self._status.setText("Segmenting...")
        self._run_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            from geobia.io.raster import read_raster, write_raster
            from geobia.segmentation import segment
            from geobia.utils.vectorize import vectorize_labels
            from ..geoobia_plugin import SegmentationRun

            source = layer.source()
            image, meta = read_raster(source)
            log(f"Image: shape={image.shape}, dtype={image.dtype}")

            labels = segment(image, method=method, **params)
            n_segments = int(labels.max())
            log(f"Segmentation done: {n_segments} segments")

            # Save raster labels
            fd, raster_path = tempfile.mkstemp(
                suffix="_segments.tif",
                prefix=f"geobia_{method}_",
                dir=os.path.dirname(source) or tempfile.gettempdir(),
            )
            os.close(fd)
            write_raster(raster_path, labels, meta, dtype="int32")

            # Vectorize
            crs = meta.get("crs")
            gdf = vectorize_labels(labels, meta["transform"], crs)
            log(f"Vectorized {len(gdf)} polygons")

            # Create run record
            run = SegmentationRun(
                method=method, params=params, labels_array=labels,
                meta=meta, raster_path=raster_path, gdf=gdf,
                n_segments=n_segments,
            )
            self.state.seg_runs.append(run)
            self.state.input_layer = layer

            # Add to gallery and select it
            self._gallery_list.addItem(run.summary)
            new_idx = self._gallery_list.count() - 1
            self._gallery_list.setCurrentRow(new_idx)
            # _on_gallery_select will show it on the map

            self._status.setText(f"{n_segments} segments created.")

        except Exception:
            msg = traceback.format_exc()
            log(f"Run FAILED:\n{msg}", Qgis.Critical)
            self._status.setText("Segmentation failed — see Log Messages.")
            QMessageBox.warning(self, "GeoOBIA", f"Segmentation failed:\n{msg}")
        finally:
            QApplication.restoreOverrideCursor()
            self._run_btn.setEnabled(True)

    # --- Gallery interactions ---

    def _on_gallery_select(self, row):
        """Show the selected run's outlines on the map."""
        has_selection = 0 <= row < len(self.state.seg_runs)
        self._use_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)

        if not has_selection:
            return

        run = self.state.seg_runs[row]
        self._show_outlines(run)

        # Bold the active run in the list
        self._update_gallery_styling()

    def _on_use(self):
        """Mark the selected run as the active segmentation for downstream."""
        row = self._gallery_list.currentRow()
        if row < 0 or row >= len(self.state.seg_runs):
            return

        run = self.state.seg_runs[row]
        self.state.active_seg_index = row

        # Set up raster labels layer for classification panel
        if self.state.labels_layer:
            # Remove old hidden raster layer
            try:
                QgsProject.instance().removeMapLayer(self.state.labels_layer.id())
            except Exception:
                pass
        rlayer = QgsRasterLayer(run.raster_path, "GeoOBIA Segments [raster]")
        if rlayer.isValid():
            QgsProject.instance().addMapLayer(rlayer, addToLegend=False)
            self.state.labels_layer = rlayer

        # Clear downstream results since segmentation changed
        self.state.features_df = None
        self.state.predictions = None
        self.state.training_samples = {}

        self._update_gallery_styling()
        self._status.setText(f"Active: {run.summary}")
        log(f"Active segmentation set to run {row}: {run.summary}")

    def _on_delete(self):
        """Delete the selected segmentation run."""
        row = self._gallery_list.currentRow()
        if row < 0 or row >= len(self.state.seg_runs):
            return

        # Remove the run
        self.state.seg_runs.pop(row)
        self._gallery_list.takeItem(row)

        # Adjust active index
        if self.state.active_seg_index == row:
            self.state.active_seg_index = -1
            self.state.labels_layer = None
            self.state.features_df = None
            self.state.predictions = None
            self._status.setText("Active segmentation deleted.")
        elif self.state.active_seg_index > row:
            self.state.active_seg_index -= 1

        # Update display
        if self._gallery_list.count() > 0:
            new_row = min(row, self._gallery_list.count() - 1)
            self._gallery_list.setCurrentRow(new_row)
        else:
            self._remove_outline_layer()
            self._use_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)

        self._update_gallery_styling()

    def _update_gallery_styling(self):
        """Bold the active run, normal font for others."""
        for i in range(self._gallery_list.count()):
            item = self._gallery_list.item(i)
            font = item.font()
            if i == self.state.active_seg_index:
                font.setBold(True)
                item.setText("\u2605 " + self.state.seg_runs[i].summary)  # star
            else:
                font.setBold(False)
                item.setText("  " + self.state.seg_runs[i].summary)
            item.setFont(font)

    def _show_outlines(self, run):
        """Replace the outline layer on the map with this run's polygons."""
        self._remove_outline_layer()

        crs = run.meta.get("crs")
        crs_str = str(crs) if crs else "EPSG:4326"

        vlayer = QgsVectorLayer(
            f"Polygon?crs={crs_str}", _OUTLINE_LAYER_NAME, "memory")
        provider = vlayer.dataProvider()
        provider.addAttributes([QgsField("segment_id", QVariant.Int)])
        vlayer.updateFields()

        features = []
        for _, row_data in run.gdf.iterrows():
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromWkt(row_data.geometry.wkt))
            feat.setAttributes([int(row_data["segment_id"])])
            features.append(feat)
        provider.addFeatures(features)
        vlayer.updateExtents()

        # Bright yellow outline, transparent fill
        symbol = QgsSymbol.defaultSymbol(vlayer.geometryType())
        symbol.deleteSymbolLayer(0)
        outline = QgsSimpleFillSymbolLayer()
        outline.setFillColor(QColor(0, 0, 0, 0))
        outline.setStrokeColor(QColor(255, 255, 0))
        outline.setStrokeWidth(0.5)
        symbol.appendSymbolLayer(outline)
        vlayer.setRenderer(QgsSingleSymbolRenderer(symbol))

        QgsProject.instance().addMapLayer(vlayer)
        self._outline_layer = vlayer
        self.iface.mapCanvas().refresh()

    def _remove_outline_layer(self):
        """Remove the current outline layer from the map."""
        if self._outline_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(self._outline_layer.id())
            except Exception:
                pass
            self._outline_layer = None
        # Also clean up any stale layers with our name
        for lyr in QgsProject.instance().mapLayersByName(_OUTLINE_LAYER_NAME):
            QgsProject.instance().removeMapLayer(lyr.id())
