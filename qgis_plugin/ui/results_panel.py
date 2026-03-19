"""Results visualization and export panel."""

import os

from qgis.PyQt.QtWidgets import (
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
)
from qgis.PyQt.QtCore import QVariant

# Default palette for classes
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
]


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
            "GeoTIFF (.tif) — labels",
            "Parquet (.parquet) — features + classes",
            "CSV (.csv) — features + classes",
        ])
        export_layout.addWidget(self._export_format)

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

        # Create or reuse vector layer from labels
        vlayer = self._get_or_create_vector_layer(seg)
        if vlayer is None:
            return

        if "Classification" in mode:
            self._apply_categorized_style(vlayer)
        else:
            self._apply_outline_style(vlayer)

    def _get_or_create_vector_layer(self, seg):
        """Vectorize labels into a memory vector layer."""
        if self._vector_layer is not None:
            # Check if still in project
            if QgsProject.instance().mapLayer(self._vector_layer.id()):
                return self._vector_layer

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
            QMessageBox.warning(
                self, "GeoOBIA", f"Failed to vectorize segments: {e}")
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
        except Exception as e:
            QMessageBox.warning(self, "GeoOBIA", f"Export failed: {e}")

    def _export_geotiff(self, path):
        try:
            seg = self._get_active_seg()
            if seg is None:
                return
            from geobia.io.raster import write_raster
            write_raster(path, seg.labels_array, seg.meta, dtype="int32")
            self._status.setText(f"Exported to {path}")
        except Exception as e:
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
            QMessageBox.warning(self, "GeoOBIA", f"Export failed: {e}")
