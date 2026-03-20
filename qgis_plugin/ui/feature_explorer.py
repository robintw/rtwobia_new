"""Features Explorer — choropleth map, statistics, histogram, and inspect tool.

Embedded in the Features tab after extraction. Lets users explore individual
feature distributions across segments without leaving the dock widget.
"""

import numpy as np
from qgis.core import (
    Qgis,
    QgsFeatureRequest,
    QgsGradientColorRamp,
    QgsGraduatedSymbolRenderer,
    QgsMessageLog,
    QgsRectangle,
    QgsSymbol,
)
from qgis.gui import QgsMapTool
from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

TAG = "GeoOBIA"

_LARGE_SEGMENT_THRESHOLD = 50_000


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


# ---------------------------------------------------------------------------
# Histogram dialog (matplotlib in a QDialog)
# ---------------------------------------------------------------------------


class HistogramDialog(QDialog):
    """Pop-up dialog showing a matplotlib histogram of feature values."""

    def __init__(self, values, feature_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Histogram — {feature_name}")
        self.setMinimumSize(520, 380)
        self.setAttribute(Qt.WA_DeleteOnClose)

        layout = QVBoxLayout()

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        except ImportError:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        clean = values.dropna().values
        n_bins = min(50, max(10, len(clean) // 20))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        if len(clean) > _LARGE_SEGMENT_THRESHOLD:
            # Pre-bin with numpy for speed
            counts, edges = np.histogram(clean, bins=n_bins)
            ax.bar(
                edges[:-1],
                counts,
                width=np.diff(edges),
                align="edge",
                edgecolor="black",
                linewidth=0.3,
                alpha=0.75,
            )
        else:
            ax.hist(clean, bins=n_bins, edgecolor="black", linewidth=0.3, alpha=0.75)

        ax.set_xlabel(feature_name)
        ax.set_ylabel("Segments")
        ax.set_title(f"Distribution of {feature_name}  (n={len(clean):,})")
        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)


# ---------------------------------------------------------------------------
# Map tool for inspecting feature values on hover / click
# ---------------------------------------------------------------------------


class FeatureInspectorTool(QgsMapTool):
    """Hover over segments to see the selected feature value.

    Emits ``value_identified(segment_id, value_str)`` on each hit so the
    explorer widget can update its label.  Uses a debounce timer to avoid
    running spatial queries on every pixel of mouse movement.
    """

    value_identified = pyqtSignal(int, str)  # segment_id, formatted value

    def __init__(self, canvas, features_layer, feature_name):
        super().__init__(canvas)
        self.features_layer = features_layer
        self.feature_name = feature_name
        self.setCursor(QCursor(Qt.CrossCursor))

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self._do_identify)

        self._last_point = None

    def canvasMoveEvent(self, event):
        self._last_point = self.toMapCoordinates(event.pos())
        self._debounce.start()

    def canvasReleaseEvent(self, event):
        # Immediate identify on click
        self._last_point = self.toMapCoordinates(event.pos())
        self._do_identify()

    def _do_identify(self):
        if self._last_point is None or self.features_layer is None:
            return

        try:
            if not self.features_layer.isValid():
                return
        except RuntimeError:
            return

        sr = self.searchRadiusMU(self.canvas())
        pt = self._last_point
        rect = QgsRectangle(pt.x() - sr, pt.y() - sr, pt.x() + sr, pt.y() + sr)

        request = QgsFeatureRequest().setFilterRect(rect).setLimit(1)
        for feat in self.features_layer.getFeatures(request):
            seg_id = feat["segment_id"]
            raw = feat[self.feature_name]
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                val_str = "N/A"
            else:
                val_str = f"{raw:.4g}" if isinstance(raw, float) else str(raw)
            self.value_identified.emit(int(seg_id), val_str)
            return

        # No hit — clear
        self.value_identified.emit(0, "")


# ---------------------------------------------------------------------------
# Main explorer widget
# ---------------------------------------------------------------------------


class FeatureExplorerWidget(QWidget):
    """Explore extracted features: choropleth, statistics, histogram, inspect.

    Instantiated by ``FeaturesPanel`` and shown after feature extraction.
    """

    def __init__(self, iface, state, get_features_layer_fn, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.state = state
        self._get_layer = get_features_layer_fn
        self._inspector_tool = None
        self._previous_map_tool = None
        self._saved_renderer = None  # to restore after choropleth
        self._features_df = None
        self._setup_ui()

    # ---- UI construction ---------------------------------------------------

    def _setup_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Explore Features")
        layout = QVBoxLayout()

        # Feature selector
        self._combo = QComboBox()
        self._combo.setToolTip("Select a feature to explore")
        self._combo.currentIndexChanged.connect(self._on_feature_changed)
        layout.addWidget(self._combo)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        self._lbl_min = QLabel("—")
        self._lbl_max = QLabel("—")
        self._lbl_mean = QLabel("—")
        self._lbl_std = QLabel("—")
        self._lbl_median = QLabel("—")
        self._lbl_count = QLabel("—")
        stats_layout.addRow("Min:", self._lbl_min)
        stats_layout.addRow("Max:", self._lbl_max)
        stats_layout.addRow("Mean:", self._lbl_mean)
        stats_layout.addRow("Std:", self._lbl_std)
        stats_layout.addRow("Median:", self._lbl_median)
        stats_layout.addRow("Count:", self._lbl_count)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Action buttons row
        btn_row = QHBoxLayout()

        self._choropleth_btn = QPushButton("Choropleth")
        self._choropleth_btn.setToolTip("Colour segments on the map by this feature's value")
        self._choropleth_btn.clicked.connect(self._on_choropleth)
        btn_row.addWidget(self._choropleth_btn)

        self._histogram_btn = QPushButton("Histogram…")
        self._histogram_btn.setToolTip("Show histogram of feature values")
        self._histogram_btn.clicked.connect(self._on_histogram)
        btn_row.addWidget(self._histogram_btn)

        self._inspect_btn = QPushButton("Inspect")
        self._inspect_btn.setToolTip("Hover over the map to see this feature's value per segment")
        self._inspect_btn.setCheckable(True)
        self._inspect_btn.toggled.connect(self._on_inspect_toggled)
        btn_row.addWidget(self._inspect_btn)

        layout.addLayout(btn_row)

        # Reset style button
        self._reset_btn = QPushButton("Reset to outlines")
        self._reset_btn.setToolTip("Remove choropleth colouring")
        self._reset_btn.clicked.connect(self._on_reset_style)
        self._reset_btn.hide()
        layout.addWidget(self._reset_btn)

        # Classes control for choropleth
        classes_row = QHBoxLayout()
        classes_row.addWidget(QLabel("Classes:"))
        self._classes_spin = QSpinBox()
        self._classes_spin.setRange(3, 20)
        self._classes_spin.setValue(5)
        self._classes_spin.setToolTip("Number of colour classes for choropleth")
        classes_row.addWidget(self._classes_spin)
        classes_row.addStretch()
        layout.addLayout(classes_row)

        # Inspector value display
        self._inspect_label = QLabel("")
        self._inspect_label.setStyleSheet(
            "QLabel { background: #222; color: #0f0; padding: 4px; "
            "font-family: monospace; font-size: 12px; }"
        )
        self._inspect_label.setWordWrap(True)
        self._inspect_label.hide()
        layout.addWidget(self._inspect_label)

        group.setLayout(layout)
        outer.addWidget(group)
        self.setLayout(outer)

    # ---- Public API --------------------------------------------------------

    def update_features(self, features_df):
        """Populate the combo box and statistics from a new features DataFrame."""
        self._features_df = features_df
        self._combo.blockSignals(True)
        self._combo.clear()
        for col in sorted(features_df.columns):
            self._combo.addItem(col)
        self._combo.blockSignals(False)

        if self._combo.count() > 0:
            self._combo.setCurrentIndex(0)
            self._update_stats()

    # ---- Internals ---------------------------------------------------------

    def _selected_feature(self):
        return self._combo.currentText() if self._combo.count() > 0 else None

    def _on_feature_changed(self, _index):
        self._update_stats()
        # If inspector is active, rebind it to the new feature
        if self._inspect_btn.isChecked():
            self._activate_inspector()

    def _update_stats(self):
        name = self._selected_feature()
        if name is None or self._features_df is None:
            return
        vals = self._features_df[name].dropna()
        if len(vals) == 0:
            for lbl in (
                self._lbl_min,
                self._lbl_max,
                self._lbl_mean,
                self._lbl_std,
                self._lbl_median,
                self._lbl_count,
            ):
                lbl.setText("—")
            return

        self._lbl_min.setText(f"{vals.min():.4g}")
        self._lbl_max.setText(f"{vals.max():.4g}")
        self._lbl_mean.setText(f"{vals.mean():.4g}")
        self._lbl_std.setText(f"{vals.std():.4g}")
        self._lbl_median.setText(f"{vals.median():.4g}")
        self._lbl_count.setText(f"{len(vals):,}")

        # Warn if large
        if len(vals) > _LARGE_SEGMENT_THRESHOLD:
            self._histogram_btn.setText("Histogram… (large)")
        else:
            self._histogram_btn.setText("Histogram…")

    # ---- Choropleth --------------------------------------------------------

    def _on_choropleth(self):
        name = self._selected_feature()
        if name is None:
            return

        vlayer = self._get_layer()
        if vlayer is None:
            QMessageBox.warning(self, "GeoOBIA", "No features layer. Extract features first.")
            return

        # Save original renderer for reset
        if self._saved_renderer is None:
            self._saved_renderer = vlayer.renderer().clone()

        n_classes = self._classes_spin.value()

        try:
            ramp = QgsGradientColorRamp(
                QColor(255, 255, 178),  # light yellow
                QColor(189, 0, 38),  # dark red
            )
            # Add a midpoint stop for a YlOrRd-like ramp
            ramp.setStops(
                [
                    QgsGradientColorRamp.StopInfo(0.25, QColor(254, 204, 92)),
                    QgsGradientColorRamp.StopInfo(0.50, QColor(253, 141, 60)),
                    QgsGradientColorRamp.StopInfo(0.75, QColor(227, 74, 51)),
                ]
            )

            renderer = QgsGraduatedSymbolRenderer.createRenderer(
                vlayer,
                attrName=name,
                classes=n_classes,
                mode=QgsGraduatedSymbolRenderer.EqualInterval,
                symbol=QgsSymbol.defaultSymbol(vlayer.geometryType()),
                ramp=ramp,
            )

            # Set semi-transparent fill
            for i in range(renderer.ranges().__len__()):
                sym = renderer.ranges()[i].symbol()
                sym.setOpacity(0.7)

            vlayer.setRenderer(renderer)
            vlayer.triggerRepaint()
            self.iface.mapCanvas().refresh()

            self._reset_btn.show()
            log(f"Choropleth applied: {name}, {n_classes} classes")

        except Exception as exc:
            log(f"Choropleth failed: {exc}", Qgis.Warning)
            QMessageBox.warning(self, "GeoOBIA", f"Failed to create choropleth:\n{exc}")

    def _on_reset_style(self):
        vlayer = self._get_layer()
        if vlayer is None:
            return

        if self._saved_renderer is not None:
            vlayer.setRenderer(self._saved_renderer.clone())
            vlayer.triggerRepaint()
            self.iface.mapCanvas().refresh()
            self._saved_renderer = None

        self._reset_btn.hide()

    # ---- Histogram ---------------------------------------------------------

    def _on_histogram(self):
        name = self._selected_feature()
        if name is None or self._features_df is None:
            return

        vals = self._features_df[name]
        if vals.dropna().empty:
            QMessageBox.information(self, "GeoOBIA", f"No valid values for '{name}'.")
            return

        dlg = HistogramDialog(vals, name, parent=self)
        dlg.exec_()

    # ---- Inspect map tool --------------------------------------------------

    def _on_inspect_toggled(self, checked):
        if checked:
            self._activate_inspector()
            self._inspect_label.show()
        else:
            self._deactivate_inspector()
            self._inspect_label.hide()

    def _activate_inspector(self):
        name = self._selected_feature()
        vlayer = self._get_layer()
        if name is None or vlayer is None:
            self._inspect_btn.setChecked(False)
            return

        canvas = self.iface.mapCanvas()

        # Save current tool so we can restore it
        if self._inspector_tool is None:
            self._previous_map_tool = canvas.mapTool()

        self._inspector_tool = FeatureInspectorTool(canvas, vlayer, name)
        self._inspector_tool.value_identified.connect(self._on_value_identified)
        canvas.setMapTool(self._inspector_tool)

    def _deactivate_inspector(self):
        if self._inspector_tool is not None:
            canvas = self.iface.mapCanvas()
            canvas.unsetMapTool(self._inspector_tool)
            self._inspector_tool = None

            if self._previous_map_tool is not None:
                canvas.setMapTool(self._previous_map_tool)
                self._previous_map_tool = None

        self._inspect_label.setText("")

    def _on_value_identified(self, seg_id, val_str):
        name = self._selected_feature()
        if seg_id > 0 and val_str:
            self._inspect_label.setText(f"Segment {seg_id}  |  {name} = {val_str}")
        else:
            self._inspect_label.setText("")
