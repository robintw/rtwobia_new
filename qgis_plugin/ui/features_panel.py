"""Feature extraction configuration panel."""

import traceback

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QApplication,
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
from qgis.core import QgsMessageLog, Qgis

TAG = "GeoOBIA"


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


class FeaturesPanel(QWidget):
    def __init__(self, iface, state, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.state = state
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

        self._texture_cb = QCheckBox("Texture (GLCM contrast, homogeneity, entropy)")
        self._texture_cb.setChecked(False)
        cat_layout.addWidget(self._texture_cb)

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

        if not categories:
            QMessageBox.warning(self, "GeoOBIA", "Select at least one feature category.")
            return

        band_names_str = self._band_names_edit.text().strip()
        log(f"Extract: categories={categories}, bands={band_names_str!r}")

        self._status.setText("Extracting features...")
        self._extract_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            from geobia.io.raster import read_raster
            from geobia.features import extract

            source = self.state.input_layer.source()
            image, meta = read_raster(source)
            log(f"Image: shape={image.shape}, dtype={image.dtype}")

            labels = seg.labels_array
            log(f"Labels: shape={labels.shape}, max={labels.max()}")

            kwargs = {}
            if band_names_str:
                names = [n.strip() for n in band_names_str.split(",")]
                kwargs["band_names"] = {name: i for i, name in enumerate(names)}

            pixel_size = abs(meta["transform"].a) if meta.get("transform") else None
            if pixel_size:
                kwargs["pixel_size"] = pixel_size

            features = extract(image, labels, categories=categories, **kwargs)

            self.state.features_df = features
            n_segs = len(features)
            n_feats = len(features.columns)
            self._status.setText(f"{n_feats} features extracted for {n_segs} segments.")
            log(f"Extraction done: {n_feats} features x {n_segs} segments")

        except Exception:
            msg = traceback.format_exc()
            log(f"Extract FAILED:\n{msg}", Qgis.Critical)
            self._status.setText("Feature extraction failed — see Log Messages.")
            QMessageBox.warning(self, "GeoOBIA", f"Feature extraction failed:\n{msg}")
        finally:
            QApplication.restoreOverrideCursor()
            self._extract_btn.setEnabled(True)
