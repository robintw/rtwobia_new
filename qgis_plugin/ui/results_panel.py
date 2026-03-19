"""Results visualization and export panel."""

import os
import traceback

from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsField,
    QgsProject,
    QgsVectorLayer,
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsSymbol,
    QgsMessageLog,
    Qgis,
)
from qgis.PyQt.QtCore import QVariant

TAG = "GeoOBIA"

# Default palette for classes
_SAMPLES_LAYER_NAME = "GeoOBIA Training Samples"

_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
]


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


def _is_layer_alive(layer):
    """Check if a QgsVectorLayer C++ object is still valid."""
    if layer is None:
        return False
    try:
        layer.id()
        return True
    except RuntimeError:
        return False


class ResultsPanel(QWidget):
    def __init__(self, iface, state, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.state = state
        self._vector_layer = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Refresh button
        refresh_btn = QPushButton("Refresh Results")
        refresh_btn.setToolTip("Update summary from latest classification")
        refresh_btn.clicked.connect(self._refresh)
        layout.addWidget(refresh_btn)

        # Summary table
        summary_group = QGroupBox("Classification Summary")
        summary_layout = QVBoxLayout()
        self._summary_table = QTableWidget(0, 3)
        self._summary_table.setHorizontalHeaderLabels(
            ["Class", "Segments", "% of Total"])
        self._summary_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        summary_layout.addWidget(self._summary_table)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # Visualization
        vis_group = QGroupBox("Visualization")
        vis_layout = QVBoxLayout()

        self._vis_mode = QComboBox()
        self._vis_mode.addItems([
            "Classification (categorized)",
            "Segment outlines",
        ])
        vis_layout.addWidget(QLabel("Display mode:"))
        vis_layout.addWidget(self._vis_mode)

        vis_btn = QPushButton("Apply Visualization")
        vis_btn.clicked.connect(self._apply_visualization)
        vis_layout.addWidget(vis_btn)

        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()

        self._export_format = QComboBox()
        self._export_format.addItems([
            "GeoPackage (.gpkg)",
            "GeoTIFF (.tif) — segment labels",
            "GeoTIFF (.tif) — classified raster",
            "Parquet (.parquet) — features + classes",
            "CSV (.csv) — features + classes",
        ])
        export_layout.addWidget(self._export_format)

        self._add_to_qgis_cb = QCheckBox("Add exported layer to QGIS")
        self._add_to_qgis_cb.setChecked(True)
        export_layout.addWidget(self._add_to_qgis_cb)

        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._on_export)
        export_layout.addWidget(export_btn)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Status
        self._status = QLabel("")
        layout.addWidget(self._status)

        layout.addStretch()
        self.setLayout(layout)

    def _get_active_seg(self):
        """Get the active segmentation run, or warn the user."""
        seg = self.state.active_seg
        if seg is None:
            QMessageBox.warning(self, "GeoOBIA", "No active segmentation selected.")
        return seg

    def _refresh(self):
        """Rebuild the summary table from current predictions."""
        predictions = self.state.predictions
        if predictions is None:
            self._status.setText("No classification results yet.")
            return

        from collections import Counter
        counts = Counter(predictions)
        total = sum(counts.values())

        self._summary_table.setRowCount(0)
        for cls_name, count in sorted(counts.items(), key=lambda x: -x[1]):
            row = self._summary_table.rowCount()
            self._summary_table.insertRow(row)
            self._summary_table.setItem(row, 0, QTableWidgetItem(str(cls_name)))
            self._summary_table.setItem(row, 1, QTableWidgetItem(str(count)))
            pct = f"{count / total * 100:.1f}%"
            self._summary_table.setItem(row, 2, QTableWidgetItem(pct))

        self._status.setText(
            f"{total} segments in {len(counts)} classes.")

    def _apply_visualization(self):
        """Create/update a vector layer showing classification results."""
        mode = self._vis_mode.currentText()

        seg = self._get_active_seg()
        if seg is None:
            return

        # Hide training samples layer so it doesn't obscure results
        self._hide_training_samples()

        # Create or reuse vector layer from labels
        vlayer = self._get_or_create_vector_layer(seg)
        if vlayer is None:
            return

        if "Classification" in mode:
            self._apply_categorized_style(vlayer)
        else:
            self._apply_outline_style(vlayer)

    def _hide_training_samples(self):
        """Turn off the training samples layer in the layer tree."""
        root = QgsProject.instance().layerTreeRoot()
        for lyr in QgsProject.instance().mapLayersByName(_SAMPLES_LAYER_NAME):
            try:
                node = root.findLayer(lyr.id())
                if node is not None:
                    node.setItemVisibilityChecked(False)
            except (RuntimeError, Exception):
                pass

    def _get_or_create_vector_layer(self, seg):
        """Vectorize labels into a memory vector layer."""
        # Check if the cached layer is still alive and in the project
        if _is_layer_alive(self._vector_layer):
            if QgsProject.instance().mapLayer(self._vector_layer.id()):
                return self._vector_layer

        # Layer was deleted or removed — clear reference and recreate
        self._vector_layer = None

        try:
            gdf = seg.gdf.copy()

            # Join predictions if available
            if self.state.predictions is not None:
                import pandas as pd
                pred_df = pd.DataFrame({
                    "segment_id": self.state.predictions.index,
                    "class_label": self.state.predictions.values,
                })
                gdf = gdf.merge(pred_df, on="segment_id", how="left")

            # Create memory vector layer
            crs = seg.meta.get("crs")
            crs_str = str(crs) if crs else "EPSG:4326"
            vlayer = QgsVectorLayer(
                f"Polygon?crs={crs_str}", "GeoOBIA Results", "memory")
            provider = vlayer.dataProvider()

            # Add fields
            fields = [QgsField("segment_id", QVariant.Int)]
            if "class_label" in gdf.columns:
                fields.append(QgsField("class_label", QVariant.String))
            provider.addAttributes(fields)
            vlayer.updateFields()

            # Add features
            from qgis.core import QgsFeature, QgsGeometry
            features = []
            for _, row in gdf.iterrows():
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromWkt(row.geometry.wkt))
                attrs = [int(row["segment_id"])]
                if "class_label" in gdf.columns:
                    attrs.append(str(row.get("class_label", "")))
                feat.setAttributes(attrs)
                features.append(feat)

            provider.addFeatures(features)
            vlayer.updateExtents()

            QgsProject.instance().addMapLayer(vlayer)
            self._vector_layer = vlayer
            return vlayer

        except Exception as e:
            log(f"Failed to create vector layer: {e}", Qgis.Warning)
            QMessageBox.warning(
                self, "GeoOBIA", f"Failed to create results layer: {e}")
            return None

    def _apply_categorized_style(self, vlayer):
        """Apply categorized symbology by class_label."""
        if "class_label" not in [f.name() for f in vlayer.fields()]:
            self._status.setText("No classification data to visualize.")
            return

        # Get unique class labels
        classes = sorted(set(
            f.attribute("class_label") for f in vlayer.getFeatures()
            if f.attribute("class_label")))

        categories = []
        for i, cls_name in enumerate(classes):
            symbol = QgsSymbol.defaultSymbol(vlayer.geometryType())
            color = self.state.class_colors.get(
                cls_name, QColor(_PALETTE[i % len(_PALETTE)]))
            symbol.setColor(color)
            symbol.setOpacity(0.6)
            categories.append(QgsRendererCategory(cls_name, symbol, str(cls_name)))

        renderer = QgsCategorizedSymbolRenderer("class_label", categories)
        vlayer.setRenderer(renderer)
        vlayer.triggerRepaint()
        self._status.setText("Applied categorized classification style.")

    def _apply_outline_style(self, vlayer):
        """Apply outline-only style to show segment boundaries."""
        from qgis.core import QgsSimpleFillSymbolLayer
        symbol = QgsSymbol.defaultSymbol(vlayer.geometryType())
        symbol.deleteSymbolLayer(0)
        outline = QgsSimpleFillSymbolLayer()
        outline.setFillColor(QColor(0, 0, 0, 0))
        outline.setStrokeColor(QColor(255, 255, 0))
        outline.setStrokeWidth(0.3)
        symbol.appendSymbolLayer(outline)

        from qgis.core import QgsSingleSymbolRenderer
        renderer = QgsSingleSymbolRenderer(symbol)
        vlayer.setRenderer(renderer)
        vlayer.triggerRepaint()
        self._status.setText("Applied segment outline style.")

    def _on_export(self):
        fmt = self._export_format.currentText()

        if ".gpkg" in fmt:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export GeoPackage", "", "GeoPackage (*.gpkg)")
            if path:
                self._export_gpkg(path)
        elif "classified raster" in fmt:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Classified Raster", "", "GeoTIFF (*.tif)")
            if path:
                self._export_classified_raster(path)
        elif ".tif" in fmt:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export GeoTIFF", "", "GeoTIFF (*.tif)")
            if path:
                self._export_geotiff(path)
        elif ".parquet" in fmt:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Parquet", "", "Parquet (*.parquet)")
            if path:
                self._export_parquet(path)
        elif ".csv" in fmt:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "", "CSV (*.csv)")
            if path:
                self._export_csv(path)

    def _maybe_add_to_qgis(self, path):
        """Add the exported file as a QGIS layer if the checkbox is ticked."""
        if not self._add_to_qgis_cb.isChecked():
            return

        name = os.path.splitext(os.path.basename(path))[0]

        if path.endswith((".gpkg",)):
            from qgis.core import QgsVectorLayer as QVL
            layer = QVL(path, name, "ogr")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                log(f"Added vector layer: {name}")
            else:
                log(f"Failed to load exported layer: {path}", Qgis.Warning)

        elif path.endswith((".tif", ".tiff")):
            from qgis.core import QgsRasterLayer
            layer = QgsRasterLayer(path, name)
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                log(f"Added raster layer: {name}")
            else:
                log(f"Failed to load exported layer: {path}", Qgis.Warning)

    def _export_gpkg(self, path):
        try:
            seg = self._get_active_seg()
            if seg is None:
                return
            gdf = seg.gdf.copy()

            if self.state.features_df is not None:
                gdf = gdf.merge(
                    self.state.features_df, left_on="segment_id",
                    right_index=True, how="left")
            if self.state.predictions is not None:
                import pandas as pd
                pred_df = pd.DataFrame({
                    "segment_id": self.state.predictions.index,
                    "class_label": self.state.predictions.values,
                })
                gdf = gdf.merge(pred_df, on="segment_id", how="left")

            gdf.to_file(path, driver="GPKG")
            self._status.setText(f"Exported to {path}")
            self._maybe_add_to_qgis(path)
        except Exception as e:
            log(f"Export GPKG failed: {traceback.format_exc()}", Qgis.Critical)
            QMessageBox.warning(self, "GeoOBIA", f"Export failed: {e}")

    def _export_classified_raster(self, path):
        """Export a raster where each pixel has its segment's class value."""
        try:
            import numpy as np
            seg = self._get_active_seg()
            if seg is None:
                return
            predictions = self.state.predictions
            if predictions is None:
                QMessageBox.warning(self, "GeoOBIA", "No classification results to export.")
                return

            labels = seg.labels_array
            # Map class names to integer codes
            unique_classes = sorted(predictions.unique())
            class_to_int = {c: i + 1 for i, c in enumerate(unique_classes)}
            log(f"Class mapping: {class_to_int}")

            classified = np.zeros_like(labels, dtype=np.int32)
            for seg_id, cls_name in predictions.items():
                classified[labels == seg_id] = class_to_int.get(cls_name, 0)

            from geobia.io.raster import write_raster
            write_raster(path, classified, seg.meta, dtype="int32")
            self._status.setText(f"Classified raster exported to {path}")
            self._maybe_add_to_qgis(path)
        except Exception as e:
            log(f"Export classified raster failed: {traceback.format_exc()}", Qgis.Critical)
            QMessageBox.warning(self, "GeoOBIA", f"Export failed: {e}")

    def _export_geotiff(self, path):
        try:
            seg = self._get_active_seg()
            if seg is None:
                return
            from geobia.io.raster import write_raster
            write_raster(path, seg.labels_array, seg.meta, dtype="int32")
            self._status.setText(f"Exported to {path}")
            self._maybe_add_to_qgis(path)
        except Exception as e:
            log(f"Export GeoTIFF failed: {traceback.format_exc()}", Qgis.Critical)
            QMessageBox.warning(self, "GeoOBIA", f"Export failed: {e}")

    def _export_parquet(self, path):
        try:
            df = self.state.features_df
            if df is None:
                QMessageBox.warning(self, "GeoOBIA", "No features to export.")
                return
            if self.state.predictions is not None:
                df = df.copy()
                df["class_label"] = self.state.predictions
            df.to_parquet(path)
            self._status.setText(f"Exported to {path}")
        except Exception as e:
            log(f"Export Parquet failed: {traceback.format_exc()}", Qgis.Critical)
            QMessageBox.warning(self, "GeoOBIA", f"Export failed: {e}")

    def _export_csv(self, path):
        try:
            df = self.state.features_df
            if df is None:
                QMessageBox.warning(self, "GeoOBIA", "No features to export.")
                return
            if self.state.predictions is not None:
                df = df.copy()
                df["class_label"] = self.state.predictions
            df.to_csv(path)
            self._status.setText(f"Exported to {path}")
        except Exception as e:
            log(f"Export CSV failed: {traceback.format_exc()}", Qgis.Critical)
            QMessageBox.warning(self, "GeoOBIA", f"Export failed: {e}")
