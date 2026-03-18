"""Feature extraction configuration panel."""

import os
import tempfile

from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsApplication,
    QgsMapLayerProxyModel,
    QgsTask,
)
from qgis.gui import QgsMapLayerComboBox


class FeaturesPanel(QWidget):
    def __init__(self, iface, state, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.state = state
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Input layers
        input_group = QGroupBox("Input Layers")
        input_layout = QFormLayout()

        self._image_combo = QgsMapLayerComboBox()
        self._image_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        input_layout.addRow("Image:", self._image_combo)

        self._segments_combo = QgsMapLayerComboBox()
        self._segments_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        input_layout.addRow("Segments:", self._segments_combo)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

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
        btn_layout = QHBoxLayout()
        self._extract_btn = QPushButton("Extract Features")
        self._extract_btn.clicked.connect(self._on_extract)
        btn_layout.addWidget(self._extract_btn)
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

    def _on_extract(self):
        img_layer = self._image_combo.currentLayer()
        seg_layer = self._segments_combo.currentLayer()
        if img_layer is None or seg_layer is None:
            QMessageBox.warning(
                self, "GeoOBIA",
                "Select both an image layer and a segment labels layer.")
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

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._extract_btn.setEnabled(False)
        self._status.setText("Extracting features...")

        task = _ExtractTask(
            self.state, img_layer, seg_layer,
            categories, band_names_str, self._on_done)
        QgsApplication.taskManager().addTask(task)

    def _on_done(self, success: bool, message: str):
        self._progress.setVisible(False)
        self._extract_btn.setEnabled(True)
        self._status.setText(message)
        if not success:
            QMessageBox.warning(self, "GeoOBIA", message)


class _ExtractTask(QgsTask):
    def __init__(self, state, img_layer, seg_layer,
                 categories, band_names_str, callback):
        super().__init__("GeoOBIA: Extract Features")
        self.state = state
        self.img_source = img_layer.source()
        self.seg_source = seg_layer.source()
        self.categories = categories
        self.band_names_str = band_names_str
        self.callback = callback
        self.error_msg = ""

    def run(self):
        try:
            from geobia.io.raster import read_raster
            from geobia.features import extract

            self.setProgress(5)
            image, meta = read_raster(self.img_source)

            self.setProgress(10)
            seg_data, _ = read_raster(self.seg_source)
            labels = seg_data[0]

            kwargs = {}
            if self.band_names_str:
                names = [n.strip() for n in self.band_names_str.split(",")]
                kwargs["band_names"] = {name: i for i, name in enumerate(names)}

            pixel_size = abs(meta["transform"].a) if meta.get("transform") else None
            if pixel_size:
                kwargs["pixel_size"] = pixel_size

            self.setProgress(20)
            features = extract(image, labels, categories=self.categories, **kwargs)

            self.state.features_df = features
            self.state.meta = meta

            self.setProgress(100)
            n_segs = len(features)
            n_feats = len(features.columns)
            self.callback(True, f"{n_feats} features extracted for {n_segs} segments.")
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def finished(self, result):
        if not result:
            self.callback(False, f"Feature extraction failed: {self.error_msg}")
