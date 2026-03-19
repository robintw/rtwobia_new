"""Feature extraction configuration panel."""

import traceback

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsSimpleFillSymbolLayer,
    QgsSingleSymbolRenderer,
    QgsSymbol,
    QgsVectorLayer,
    Qgis,
)

TAG = "GeoOBIA"
_FEATURES_LAYER_NAME = "GeoOBIA Features"


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


class FeaturesPanel(QWidget):
    def __init__(self, iface, state, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.state = state
        self._features_layer = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Active segmentation indicator
        seg_group = QGroupBox("Active Segmentation")
        seg_layout = QVBoxLayout()
        self._seg_label = QLabel("No segmentation selected.")
        self._seg_label.setWordWrap(True)
        seg_layout.addWidget(self._seg_label)
        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)

        # Feature categories
        cat_group = QGroupBox("Feature Categories")
        cat_layout = QVBoxLayout()

        self._spectral_cb = QCheckBox("Spectral (mean, std, min, max per band; NDVI/NDWI)")
        self._spectral_cb.setChecked(True)
        cat_layout.addWidget(self._spectral_cb)

        self._geometry_cb = QCheckBox("Geometry (area, perimeter, compactness, elongation)")
        self._geometry_cb.setChecked(True)
        cat_layout.addWidget(self._geometry_cb)

        self._texture_cb = QCheckBox("Texture (GLCM: contrast, dissimilarity, homogeneity, energy, correlation per band)")
        self._texture_cb.setChecked(False)
        cat_layout.addWidget(self._texture_cb)

        self._context_cb = QCheckBox("Context (neighbor count, border contrast)")
        self._context_cb.setChecked(False)
        cat_layout.addWidget(self._context_cb)

        cat_group.setLayout(cat_layout)
        layout.addWidget(cat_group)

        # Band names
        band_group = QGroupBox("Band Configuration")
        band_layout = QFormLayout()
        self._band_names_edit = QLineEdit()
        self._band_names_edit.setPlaceholderText("e.g. red, green, blue, nir")
        self._band_names_edit.setToolTip(
            "Comma-separated band names. Enables NDVI/NDWI ratio computation "
            "when 'nir', 'red', 'green' are specified.")
        band_layout.addRow("Band names:", self._band_names_edit)
        band_group.setLayout(band_layout)
        layout.addWidget(band_group)

        # Extract button
        self._extract_btn = QPushButton("Extract Features")
        self._extract_btn.setToolTip("Extract features from the active segmentation")
        self._extract_btn.clicked.connect(self._on_extract)
        layout.addWidget(self._extract_btn)

        # Progress
        from .tasks import TaskProgressWidget
        self._progress = TaskProgressWidget()
        layout.addWidget(self._progress)

        # Status
        self._status = QLabel("")
        layout.addWidget(self._status)

        layout.addStretch()
        self.setLayout(layout)

    def showEvent(self, event):
        """Update the active segmentation label when the panel becomes visible."""
        super().showEvent(event)
        self._update_seg_label()

    def _update_seg_label(self):
        seg = self.state.active_seg
        if seg is not None:
            self._seg_label.setText(f"Using: {seg.summary}")
        else:
            self._seg_label.setText("No segmentation selected. "
                                    "Use the Segmentation tab to run and activate one.")

    def _on_extract(self):
        self._update_seg_label()

        seg = self.state.active_seg
        if seg is None:
            QMessageBox.warning(
                self, "GeoOBIA",
                "No active segmentation. Go to the Segmentation tab, "
                "run a segmentation, and click 'Use for Extraction'.")
            return

        if self.state.input_layer is None:
            QMessageBox.warning(self, "GeoOBIA", "No input raster layer set.")
            return

        categories = []
        if self._spectral_cb.isChecked():
            categories.append("spectral")
        if self._geometry_cb.isChecked():
            categories.append("geometry")
        if self._texture_cb.isChecked():
            categories.append("texture")
        if self._context_cb.isChecked():
            categories.append("context")

        if not categories:
            QMessageBox.warning(self, "GeoOBIA", "Select at least one feature category.")
            return

        band_names_str = self._band_names_edit.text().strip()
        log(f"Extract: categories={categories}, bands={band_names_str!r}")

        self._status.setText("Extracting features...")
        self._extract_btn.setEnabled(False)

        source = self.state.input_layer.source()
        labels = seg.labels_array.copy()
        seg_meta = seg.meta

        kwargs = {}
        if band_names_str:
            names = [n.strip() for n in band_names_str.split(",")]
            kwargs["band_names"] = {name: i for i, name in enumerate(names)}

        def work(set_progress, is_canceled):
            from geobia.io.raster import read_raster
            from geobia.features import extract

            set_progress(2)
            image, meta = read_raster(source)
            log(f"Image: shape={image.shape}, dtype={image.dtype}")
            log(f"Labels: shape={labels.shape}, max={labels.max()}")

            pixel_size = abs(meta["transform"].a) if meta.get("transform") else None
            if pixel_size:
                kwargs["pixel_size"] = pixel_size

            set_progress(5)
            features = extract(image, labels, categories=categories, **kwargs)
            set_progress(100)
            return features

        def on_success(features):
            self._extract_btn.setEnabled(True)
            self.state.features_df = features
            n_segs = len(features)
            n_feats = len(features.columns)
            log(f"Extraction done: {n_feats} features x {n_segs} segments")
            log(f"Feature columns: {list(features.columns)}")
            self._update_features_layer(seg, features)
            self._status.setText(
                f"{n_feats} features extracted for {n_segs} segments.")

        def on_failure(error_msg):
            self._extract_btn.setEnabled(True)
            log(f"Extract FAILED:\n{error_msg}", Qgis.Critical)
            self._status.setText("Feature extraction failed — see Log Messages.")
            QMessageBox.warning(self, "GeoOBIA",
                                f"Feature extraction failed:\n{error_msg}")

        from .tasks import BackgroundTask, run_task
        task = BackgroundTask(
            "GeoOBIA: Feature extraction",
            work, on_success, on_failure,
        )
        run_task(self, task, progress_widget=self._progress)

    def _update_features_layer(self, seg, features_df):
        """Create/replace a vector layer with all extracted feature attributes.

        This makes features visible in QGIS's Identify Features tool.
        """
        import numpy as np
        import pandas as pd

        # Remove old features layer
        self._remove_features_layer()

        crs = seg.meta.get("crs")
        crs_str = str(crs) if crs else "EPSG:4326"

        vlayer = QgsVectorLayer(
            f"Polygon?crs={crs_str}", _FEATURES_LAYER_NAME, "memory")
        provider = vlayer.dataProvider()

        # Build fields: segment_id + all feature columns
        qgs_fields = [QgsField("segment_id", QVariant.Int)]
        for col in features_df.columns:
            dtype = features_df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                qgs_fields.append(QgsField(col, QVariant.Int))
            elif pd.api.types.is_float_dtype(dtype):
                qgs_fields.append(QgsField(col, QVariant.Double))
            else:
                qgs_fields.append(QgsField(col, QVariant.String))
        provider.addAttributes(qgs_fields)
        vlayer.updateFields()

        # Add features with all attributes
        qgs_features = []
        for _, row_data in seg.gdf.iterrows():
            sid = int(row_data["segment_id"])
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromWkt(row_data.geometry.wkt))

            attrs = [sid]
            if sid in features_df.index:
                for col in features_df.columns:
                    val = features_df.loc[sid, col]
                    if pd.isna(val):
                        attrs.append(None)
                    elif isinstance(val, (np.integer,)):
                        attrs.append(int(val))
                    elif isinstance(val, (np.floating, float)):
                        attrs.append(float(val))
                    else:
                        attrs.append(str(val))
            else:
                attrs.extend([None] * len(features_df.columns))

            feat.setAttributes(attrs)
            qgs_features.append(feat)

        provider.addFeatures(qgs_features)
        vlayer.updateExtents()

        # Transparent fill, thin outline so it doesn't obscure the image
        symbol = QgsSymbol.defaultSymbol(vlayer.geometryType())
        symbol.deleteSymbolLayer(0)
        outline = QgsSimpleFillSymbolLayer()
        outline.setFillColor(QColor(0, 0, 0, 0))
        outline.setStrokeColor(QColor(100, 200, 255))
        outline.setStrokeWidth(0.3)
        symbol.appendSymbolLayer(outline)
        vlayer.setRenderer(QgsSingleSymbolRenderer(symbol))

        QgsProject.instance().addMapLayer(vlayer)
        self._features_layer = vlayer
        self.iface.mapCanvas().refresh()
        log(f"Features layer added with {len(qgs_features)} features, "
            f"{len(features_df.columns)} attributes")

    def _remove_features_layer(self):
        """Remove the current features layer from the map."""
        if self._features_layer is not None:
            try:
                self._features_layer.id()
                QgsProject.instance().removeMapLayer(self._features_layer.id())
            except (RuntimeError, Exception):
                pass
            self._features_layer = None
        # Clean up any stale layers with our name
        for lyr in QgsProject.instance().mapLayersByName(_FEATURES_LAYER_NAME):
            try:
                QgsProject.instance().removeMapLayer(lyr.id())
            except Exception:
                pass
