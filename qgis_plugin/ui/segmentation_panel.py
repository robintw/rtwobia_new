"""Segmentation panel with algorithm selection, dynamic params, and gallery."""

import os
import tempfile
import traceback
from collections import OrderedDict

from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QFileDialog,
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
        self._outline_layers = {}  # run index -> cached QgsVectorLayer
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

        # Run button row
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Segmentation")
        self._run_btn.setToolTip("Segment the full image")
        self._run_btn.clicked.connect(self._on_run)
        run_row.addWidget(self._run_btn)

        self._load_labels_btn = QPushButton("Load Labels...")
        self._load_labels_btn.setToolTip(
            "Load a pre-existing segment labels raster (skip segmentation)")
        self._load_labels_btn.clicked.connect(self._on_load_labels)
        run_row.addWidget(self._load_labels_btn)
        layout.addLayout(run_row)

        # Progress
        from .tasks import TaskProgressWidget
        self._progress = TaskProgressWidget()
        layout.addWidget(self._progress)

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
        source = layer.source()
        log(f"Run: method={method}, params={params}")

        self._status.setText("Segmenting...")
        self._run_btn.setEnabled(False)

        # Define work function — raster I/O + segmentation in background
        def work(set_progress, is_canceled):
            from geobia.io.raster import read_raster, write_raster
            from geobia.segmentation import segment
            from geobia.utils.vectorize import vectorize_labels

            set_progress(2)
            image, meta = read_raster(source)
            log(f"Image: shape={image.shape}, dtype={image.dtype}")
            set_progress(5)

            if is_canceled():
                return None

            labels = segment(image, method=method, **params)
            n_segments = int(labels.max())
            log(f"Segmentation done: {n_segments} segments")
            set_progress(60)

            if is_canceled():
                return None

            # Save raster labels
            fd, raster_path = tempfile.mkstemp(
                suffix="_segments.tif",
                prefix=f"geobia_{method}_",
                dir=os.path.dirname(source) or tempfile.gettempdir(),
            )
            os.close(fd)
            write_raster(raster_path, labels, meta, dtype="int32")
            set_progress(80)

            if is_canceled():
                return None

            # Vectorize
            crs = meta.get("crs")
            gdf = vectorize_labels(labels, meta["transform"], crs)
            log(f"Vectorized {len(gdf)} polygons")
            set_progress(100)

            return {
                "labels": labels, "meta": meta, "raster_path": raster_path,
                "gdf": gdf, "n_segments": n_segments,
            }

        def on_success(result):
            self._run_btn.setEnabled(True)
            if result is None:
                self._status.setText("Segmentation canceled.")
                return

            from ..geoobia_plugin import SegmentationRun
            run = SegmentationRun(
                method=method, params=params,
                labels_array=result["labels"], meta=result["meta"],
                raster_path=result["raster_path"], gdf=result["gdf"],
                n_segments=result["n_segments"],
            )
            # Evict labels from non-active runs to save memory
            for old_run in self.state.seg_runs:
                old_run.evict_labels()
            self.state.seg_runs.append(run)
            self.state.input_layer = layer

            self._gallery_list.addItem(run.summary)
            new_idx = self._gallery_list.count() - 1
            self._gallery_list.setCurrentRow(new_idx)
            self._status.setText(f"{result['n_segments']} segments created.")

        def on_failure(error_msg):
            self._run_btn.setEnabled(True)
            log(f"Run FAILED:\n{error_msg}", Qgis.Critical)
            self._status.setText("Segmentation failed — see Log Messages.")
            QMessageBox.warning(self, "GeoOBIA",
                                f"Segmentation failed:\n{error_msg}")

        from .tasks import BackgroundTask, run_task
        task = BackgroundTask(
            f"GeoOBIA: {method} segmentation",
            work, on_success, on_failure,
        )
        run_task(self, task, progress_widget=self._progress)

    def _on_load_labels(self):
        """Load a pre-existing segment labels raster."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Segment Labels", "",
            "GeoTIFF (*.tif *.tiff);;All files (*)")
        if not path:
            return

        try:
            from geobia.io.raster import read_raster
            from geobia.utils.vectorize import vectorize_labels

            self._status.setText("Loading labels...")
            data, meta = read_raster(path)
            labels = data[0]
            n_segments = int(labels.max())

            if n_segments == 0:
                QMessageBox.warning(self, "GeoOBIA",
                                    "No segments found in this raster (all zeros).")
                self._status.setText("")
                return

            crs = meta.get("crs")
            gdf = vectorize_labels(labels, meta["transform"], crs)
            log(f"Loaded {n_segments} segments from {path}")

            from ..geoobia_plugin import SegmentationRun
            run = SegmentationRun(
                method="loaded", params={"path": path},
                labels_array=labels, meta=meta,
                raster_path=path, gdf=gdf,
                n_segments=n_segments,
            )
            for old_run in self.state.seg_runs:
                old_run.evict_labels()
            self.state.seg_runs.append(run)

            # Set input layer if one is selected
            layer = self._layer_combo.currentLayer()
            if layer is not None:
                self.state.input_layer = layer

            self._gallery_list.addItem(run.summary)
            new_idx = self._gallery_list.count() - 1
            self._gallery_list.setCurrentRow(new_idx)
            self._status.setText(f"Loaded {n_segments} segments from file.")
        except Exception:
            msg = traceback.format_exc()
            log(f"Load labels FAILED:\n{msg}", Qgis.Critical)
            self._status.setText("Failed to load labels — see Log Messages.")
            QMessageBox.warning(self, "GeoOBIA",
                                f"Failed to load labels:\n{msg}")

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
        if self.state.labels_layer is not None:
            # Remove old hidden raster layer
            try:
                self.state.labels_layer.id()
                QgsProject.instance().removeMapLayer(self.state.labels_layer.id())
            except (RuntimeError, Exception):
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

        # Remove cached outline layer for this run
        vlayer = self._outline_layers.pop(row, None)
        if self._is_layer_alive(vlayer):
            try:
                QgsProject.instance().removeMapLayer(vlayer.id())
            except (RuntimeError, Exception):
                pass
        # Re-key outline layers above the deleted row
        self._outline_layers = {
            (k - 1 if k > row else k): v
            for k, v in self._outline_layers.items()
        }

        # Remove the run and clean up temp file
        run = self.state.seg_runs.pop(row)
        if run.raster_path:
            try:
                os.remove(run.raster_path)
                log(f"Deleted temp file: {run.raster_path}")
            except OSError:
                pass
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
            self._remove_outline_layers()
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
        """Show this run's outline layer, hiding all others.

        Layers are cached per run so switching is instant after the
        first view — no geometry rebuilding needed.
        """
        row = self.state.seg_runs.index(run) if run in self.state.seg_runs else -1
        root = QgsProject.instance().layerTreeRoot()

        # Hide all cached outline layers
        for idx, vlayer in list(self._outline_layers.items()):
            if not self._is_layer_alive(vlayer):
                self._outline_layers.pop(idx, None)
                continue
            node = root.findLayer(vlayer.id())
            if node is not None:
                node.setItemVisibilityChecked(False)

        # Reuse cached layer if available
        if row >= 0 and row in self._outline_layers:
            vlayer = self._outline_layers[row]
            if self._is_layer_alive(vlayer):
                node = root.findLayer(vlayer.id())
                if node is not None:
                    node.setItemVisibilityChecked(True)
                    self.iface.mapCanvas().refresh()
                    return
                # Layer was removed from project — rebuild
                self._outline_layers.pop(row, None)

        # Build a new layer for this run
        vlayer = self._build_outline_layer(run)
        QgsProject.instance().addMapLayer(vlayer)
        if row >= 0:
            self._outline_layers[row] = vlayer
        self.iface.mapCanvas().refresh()

    @staticmethod
    def _is_layer_alive(layer):
        if layer is None:
            return False
        try:
            layer.id()
            return True
        except RuntimeError:
            return False

    def _build_outline_layer(self, run):
        """Create a styled vector layer from a run's GeoDataFrame."""
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
        return vlayer

    def _remove_outline_layers(self):
        """Remove all cached outline layers from the map."""
        for idx, vlayer in list(self._outline_layers.items()):
            if self._is_layer_alive(vlayer):
                try:
                    QgsProject.instance().removeMapLayer(vlayer.id())
                except (RuntimeError, Exception):
                    pass
        self._outline_layers.clear()
        # Also clean up any stale layers with our name
        for lyr in QgsProject.instance().mapLayersByName(_OUTLINE_LAYER_NAME):
            try:
                QgsProject.instance().removeMapLayer(lyr.id())
            except (RuntimeError, Exception):
                pass
