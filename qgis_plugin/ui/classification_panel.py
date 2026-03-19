"""Classification workflow panel with supervised and unsupervised tabs."""

import traceback

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from collections import OrderedDict

from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRendererCategory,
    QgsSymbol,
    QgsVectorLayer,
    Qgis,
)
from qgis.PyQt.QtCore import QVariant

from qgis.PyQt.QtWidgets import QFileDialog

from .sample_selector import SampleSelectorTool
from .schema_widgets import build_param_widgets, collect_param_values, create_param_group

TAG = "GeoOBIA"
_SAMPLES_LAYER_NAME = "GeoOBIA Training Samples"

# Default class color palette
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
]


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


class FeatureImportanceDialog(QWidget):
    """Modal dialog showing feature importances with CSV export."""

    def __init__(self, importance, parent=None):
        from qgis.PyQt.QtWidgets import QDialog, QDialogButtonBox
        super().__init__(parent)
        self._importance = importance
        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle("Feature Importance")
        self._dialog.setMinimumSize(400, 450)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        table = QTableWidget(len(self._importance), 2)
        table.setHorizontalHeaderLabels(["Feature", "Importance"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i, (name, val) in enumerate(self._importance.items()):
            table.setItem(i, 0, QTableWidgetItem(str(name)))
            table.setItem(i, 1, QTableWidgetItem(f"{val:.6f}"))
        layout.addWidget(table)

        btn_row = QHBoxLayout()
        export_btn = QPushButton("Export CSV...")
        export_btn.clicked.connect(self._on_export_csv)
        btn_row.addWidget(export_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._dialog.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._dialog.setLayout(layout)

    def _on_export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self._dialog, "Export Feature Importance", "",
            "CSV (*.csv)")
        if not path:
            return
        import pandas as pd
        df = pd.DataFrame({
            "feature": self._importance.index,
            "importance": self._importance.values,
        })
        df.to_csv(path, index=False)
        log(f"Feature importance exported to {path}")

    def exec_(self):
        self._dialog.exec_()


class ClassificationPanel(QWidget):
    def __init__(self, iface, state, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.state = state
        self._sample_tool = None
        self._samples_layer = None
        self._trained_classifier = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Mode tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_unsupervised_tab(), "Unsupervised")
        self._tabs.addTab(self._build_supervised_tab(), "Supervised")
        layout.addWidget(self._tabs)

        # Progress
        from .tasks import TaskProgressWidget
        self._progress = TaskProgressWidget()
        layout.addWidget(self._progress)

        # Status
        self._status = QLabel("")
        layout.addWidget(self._status)

        layout.addStretch()
        self.setLayout(layout)

    # ---- Supervised tab ----

    def _build_supervised_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        # Class management
        class_group = QGroupBox("Classes")
        class_layout = QVBoxLayout()

        self._class_table = QTableWidget(0, 3)
        self._class_table.setHorizontalHeaderLabels(["Class", "Color", "Samples"])
        self._class_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self._class_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._class_table.cellDoubleClicked.connect(self._on_class_color_click)
        self._class_table.cellChanged.connect(self._on_class_name_changed)
        class_layout.addWidget(self._class_table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Class")
        add_btn.clicked.connect(self._add_class)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("Remove Class")
        rm_btn.clicked.connect(self._remove_class)
        btn_row.addWidget(rm_btn)
        class_layout.addLayout(btn_row)

        class_group.setLayout(class_layout)
        layout.addWidget(class_group)

        # Sample selection
        sample_group = QGroupBox("Training Samples")
        sample_layout = QVBoxLayout()

        self._active_class_combo = QComboBox()
        sample_layout.addWidget(QLabel("Active class:"))
        sample_layout.addWidget(self._active_class_combo)

        self._select_btn = QPushButton("Select Samples (map tool)")
        self._select_btn.setCheckable(True)
        self._select_btn.toggled.connect(self._toggle_sample_tool)
        sample_layout.addWidget(self._select_btn)

        self._sample_count_label = QLabel("Total Samples: 0")
        sample_layout.addWidget(self._sample_count_label)

        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)

        # Algorithm
        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout()

        method_row = QFormLayout()
        self._sup_method = QComboBox()
        self._sup_methods = OrderedDict([
            ("Random Forest", "random_forest"),
            ("SVM", "svm"),
            ("Gradient Boosting", "gradient_boosting"),
        ])
        self._sup_method.addItems(self._sup_methods.keys())
        self._sup_method.currentTextChanged.connect(self._on_sup_method_changed)
        method_row.addRow("Method:", self._sup_method)
        algo_layout.addLayout(method_row)

        # Dynamic parameter area (rebuilt when method changes)
        self._sup_params_container = QVBoxLayout()
        algo_layout.addLayout(self._sup_params_container)
        self._sup_param_widgets = OrderedDict()
        self._sup_param_group = None

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # Build initial params
        self._on_sup_method_changed(self._sup_method.currentText())

        # Train / Predict
        action_row = QHBoxLayout()
        train_btn = QPushButton("Train && Predict")
        train_btn.clicked.connect(self._on_train)
        action_row.addWidget(train_btn)
        layout.addLayout(action_row)

        # Feature importance button (shown after training tree-based models)
        self._importance_btn = QPushButton("Feature Importance...")
        self._importance_btn.setToolTip("View feature importance rankings")
        self._importance_btn.clicked.connect(self._on_show_importance)
        self._importance_btn.hide()
        layout.addWidget(self._importance_btn)
        self._cached_importance = None  # pd.Series, set after training

        # Model save/load
        model_row = QHBoxLayout()
        save_btn = QPushButton("Save Model...")
        save_btn.setToolTip("Save the trained model to disk")
        save_btn.clicked.connect(self._on_save_model)
        model_row.addWidget(save_btn)
        load_btn = QPushButton("Load Model...")
        load_btn.setToolTip("Load a previously saved model and predict")
        load_btn.clicked.connect(self._on_load_model)
        model_row.addWidget(load_btn)
        layout.addLayout(model_row)

        tab.setLayout(layout)
        return tab

    # ---- Unsupervised tab ----

    def _build_unsupervised_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout()

        method_row = QFormLayout()
        self._unsup_method = QComboBox()
        self._unsup_methods = OrderedDict([
            ("K-Means", "kmeans"),
            ("GMM", "gmm"),
            ("DBSCAN", "dbscan"),
        ])
        self._unsup_method.addItems(self._unsup_methods.keys())
        self._unsup_method.currentTextChanged.connect(self._on_unsup_method_changed)
        method_row.addRow("Method:", self._unsup_method)
        algo_layout.addLayout(method_row)

        # Dynamic parameter area
        self._unsup_params_container = QVBoxLayout()
        algo_layout.addLayout(self._unsup_params_container)
        self._unsup_param_widgets = OrderedDict()
        self._unsup_param_group = None

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        cluster_btn = QPushButton("Cluster")
        cluster_btn.clicked.connect(self._on_cluster)
        layout.addWidget(cluster_btn)

        # Build initial params
        self._on_unsup_method_changed(self._unsup_method.currentText())

        tab.setLayout(layout)
        return tab

    # ---- Class management ----

    def _add_class(self):
        row = self._class_table.rowCount()
        name = f"Class {row + 1}"
        color = QColor(_PALETTE[row % len(_PALETTE)])

        self._class_table.insertRow(row)

        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
        self._class_table.setItem(row, 0, name_item)

        self._set_color_cell(row, color)

        count_item = QTableWidgetItem("0")
        count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
        self._class_table.setItem(row, 2, count_item)

        self.state.class_colors[name] = color
        self._active_class_combo.addItem(name)

    def _remove_class(self):
        row = self._class_table.currentRow()
        if row < 0:
            return
        name = self._class_table.item(row, 0).text()
        self.state.class_colors.pop(name, None)
        # Remove samples for this class
        self.state.training_samples = {
            k: v for k, v in self.state.training_samples.items() if v != name}
        self._class_table.removeRow(row)
        idx = self._active_class_combo.findText(name)
        if idx >= 0:
            self._active_class_combo.removeItem(idx)
        self._update_sample_counts()

    def _on_class_name_changed(self, row, col):
        """Sync combo box and state when a class name is edited in the table."""
        if col != 0:
            return
        item = self._class_table.item(row, 0)
        if item is None:
            return
        new_name = item.text()
        old_name = self._active_class_combo.itemText(row)
        if new_name == old_name:
            return

        # Update combo box
        self._active_class_combo.setItemText(row, new_name)

        # Update class_colors dict
        color = self.state.class_colors.pop(old_name, None)
        if color is not None:
            self.state.class_colors[new_name] = color

        # Update training samples that used the old name
        for seg_id, cls in list(self.state.training_samples.items()):
            if cls == old_name:
                self.state.training_samples[seg_id] = new_name

    def _set_color_cell(self, row, color):
        """Set a colored QFrame widget in the color column for the given row."""
        swatch = QFrame()
        swatch.setAutoFillBackground(True)
        swatch.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid #888;")
        swatch.setFixedHeight(20)
        self._class_table.setCellWidget(row, 1, swatch)

    def _on_class_color_click(self, row, col):
        if col != 1:
            return
        name = self._class_table.item(row, 0).text()
        current = self.state.class_colors.get(name, QColor(Qt.white))
        color = QColorDialog.getColor(current, self)
        if color.isValid():
            self._set_color_cell(row, color)
            self.state.class_colors[name] = color

    def _on_sup_method_changed(self, method_text):
        """Rebuild parameter widgets from schema for the selected method."""
        algorithm = self._sup_methods.get(method_text, "random_forest")

        self._sup_param_widgets = OrderedDict()
        if self._sup_param_group is not None:
            self._sup_params_container.removeWidget(self._sup_param_group)
            self._sup_param_group.deleteLater()
            self._sup_param_group = None

        schema = self._get_classifier_schema("supervised", algorithm)
        self._sup_param_widgets = build_param_widgets(schema)
        if self._sup_param_widgets:
            self._sup_param_group = create_param_group(
                "Parameters", self._sup_param_widgets)
            self._sup_params_container.addWidget(self._sup_param_group)

    def _on_unsup_method_changed(self, method_text):
        """Rebuild parameter widgets from schema for the selected method."""
        algorithm = self._unsup_methods.get(method_text, "kmeans")

        self._unsup_param_widgets = OrderedDict()
        if self._unsup_param_group is not None:
            self._unsup_params_container.removeWidget(self._unsup_param_group)
            self._unsup_param_group.deleteLater()
            self._unsup_param_group = None

        schema = self._get_classifier_schema("unsupervised", algorithm)
        self._unsup_param_widgets = build_param_widgets(schema)
        if self._unsup_param_widgets:
            self._unsup_param_group = create_param_group(
                "Parameters", self._unsup_param_widgets)
            self._unsup_params_container.addWidget(self._unsup_param_group)

    @staticmethod
    def _get_classifier_schema(kind, algorithm):
        """Get param schema from the classifier class, with fallback."""
        try:
            if kind == "supervised":
                from geobia.classification.supervised import SupervisedClassifier
                return SupervisedClassifier.get_param_schema(algorithm)
            else:
                from geobia.classification.unsupervised import UnsupervisedClassifier
                return UnsupervisedClassifier.get_param_schema(algorithm)
        except AttributeError:
            log("geobia library is outdated — reinstall for parameter tooltips",
                Qgis.Warning)
            return {"type": "object", "properties": {}}

    # ---- Sample selection ----

    def _toggle_sample_tool(self, active):
        if active:
            labels_layer = self.state.labels_layer
            if labels_layer is None:
                QMessageBox.warning(
                    self, "GeoOBIA",
                    "Run segmentation first and click 'Use for Extraction'.")
                self._select_btn.setChecked(False)
                return

            self._sample_tool = SampleSelectorTool(
                self.iface.mapCanvas(),
                labels_layer,
                self.state.training_samples,
                self._active_class_combo.currentText,
            )
            self._sample_tool.sample_added.connect(self._on_sample_added)
            self._sample_tool.sample_removed.connect(self._on_sample_removed)
            self.iface.mapCanvas().setMapTool(self._sample_tool)
        else:
            if self._sample_tool:
                self.iface.mapCanvas().unsetMapTool(self._sample_tool)
                self._sample_tool = None
            # Keep the samples layer visible so user can see selections

    def _on_sample_added(self, seg_id, class_name):
        self._update_sample_counts()
        self._update_samples_layer()

    def _on_sample_removed(self, seg_id):
        self._update_sample_counts()
        self._update_samples_layer()

    def _update_sample_counts(self):
        total = len(self.state.training_samples)
        self._sample_count_label.setText(f"Total Samples: {total}")

        # Update per-class counts in the table
        from collections import Counter
        counts = Counter(self.state.training_samples.values())
        for row in range(self._class_table.rowCount()):
            name = self._class_table.item(row, 0).text()
            self._class_table.item(row, 2).setText(str(counts.get(name, 0)))

    def _update_samples_layer(self):
        """Create/refresh a vector layer showing selected training samples
        colored by class."""
        self._remove_samples_layer()

        seg = self.state.active_seg
        if seg is None or not self.state.training_samples:
            return

        crs = seg.meta.get("crs")
        crs_str = str(crs) if crs else "EPSG:4326"

        vlayer = QgsVectorLayer(
            f"Polygon?crs={crs_str}", _SAMPLES_LAYER_NAME, "memory")
        provider = vlayer.dataProvider()
        provider.addAttributes([
            QgsField("segment_id", QVariant.Int),
            QgsField("class_name", QVariant.String),
        ])
        vlayer.updateFields()

        # Build a lookup from segment_id -> geometry
        gdf = seg.gdf
        geom_lookup = {
            int(row["segment_id"]): row.geometry.wkt
            for _, row in gdf.iterrows()
        }

        features = []
        for seg_id, class_name in self.state.training_samples.items():
            wkt = geom_lookup.get(seg_id)
            if wkt is None:
                continue
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromWkt(wkt))
            feat.setAttributes([seg_id, class_name])
            features.append(feat)

        if not features:
            return

        provider.addFeatures(features)
        vlayer.updateExtents()

        # Categorized renderer: one color per class
        class_names = sorted(set(self.state.training_samples.values()))
        categories = []
        for i, cls_name in enumerate(class_names):
            symbol = QgsSymbol.defaultSymbol(vlayer.geometryType())
            color = self.state.class_colors.get(
                cls_name, QColor(_PALETTE[i % len(_PALETTE)]))
            symbol.setColor(color)
            symbol.setOpacity(0.55)
            categories.append(QgsRendererCategory(cls_name, symbol, cls_name))

        renderer = QgsCategorizedSymbolRenderer("class_name", categories)
        vlayer.setRenderer(renderer)

        QgsProject.instance().addMapLayer(vlayer)
        self._samples_layer = vlayer
        self.iface.mapCanvas().refresh()

    def _remove_samples_layer(self):
        """Remove the training samples layer from the map."""
        if self._samples_layer is not None:
            try:
                self._samples_layer.id()
                QgsProject.instance().removeMapLayer(self._samples_layer.id())
            except (RuntimeError, Exception):
                pass
            self._samples_layer = None
        for lyr in QgsProject.instance().mapLayersByName(_SAMPLES_LAYER_NAME):
            try:
                QgsProject.instance().removeMapLayer(lyr.id())
            except (RuntimeError, Exception):
                pass

    def _show_feature_importance(self, clf):
        """Cache feature importances and show the button if supported."""
        try:
            self._cached_importance = clf.feature_importance()
            self._importance_btn.show()
        except (NotImplementedError, RuntimeError, AttributeError):
            self._cached_importance = None
            self._importance_btn.hide()

    def _on_show_importance(self):
        """Open a dialog showing feature importances with CSV export."""
        if self._cached_importance is None:
            return
        dlg = FeatureImportanceDialog(self._cached_importance, parent=self)
        dlg.exec_()

    # ---- Model save/load ----

    def _on_save_model(self):
        if not hasattr(self, '_trained_classifier') or self._trained_classifier is None:
            QMessageBox.warning(self, "GeoOBIA", "No trained model to save. Run Train & Predict first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "Joblib model (*.joblib)")
        if not path:
            return
        try:
            self._trained_classifier.save(path)
            self._status.setText(f"Model saved to {path}")
            log(f"Model saved: {path}")
        except Exception as e:
            log(f"Save model FAILED: {traceback.format_exc()}", Qgis.Critical)
            QMessageBox.warning(self, "GeoOBIA", f"Failed to save model: {e}")

    def _on_load_model(self):
        if self.state.features_df is None:
            QMessageBox.warning(self, "GeoOBIA", "Extract features first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Joblib model (*.joblib)")
        if not path:
            return
        try:
            from geobia.classification.supervised import SupervisedClassifier
            clf = SupervisedClassifier.load(path)
            self._trained_classifier = clf
            predictions = clf.predict(self.state.features_df)
            self.state.predictions = predictions
            n_classes = predictions.nunique()
            self._status.setText(f"Loaded model: {n_classes} classes predicted.")
            log(f"Model loaded from {path}, {n_classes} classes")
            self._auto_show_results()
        except Exception as e:
            log(f"Load model FAILED: {traceback.format_exc()}", Qgis.Critical)
            QMessageBox.warning(self, "GeoOBIA", f"Failed to load model: {e}")

    def _auto_show_results(self):
        """Auto-show classification results on the map and refresh Results panel."""
        # Find the Results panel sibling in the tab widget
        parent_tabs = self.parent()
        if parent_tabs is None:
            return
        # Look through tabs for the ResultsPanel
        from .results_panel import ResultsPanel
        for i in range(parent_tabs.count()):
            widget = parent_tabs.widget(i)
            if isinstance(widget, ResultsPanel):
                widget._refresh()
                widget._apply_visualization()
                break

    # ---- Training / Prediction ----

    def _get_sup_method_and_params(self):
        """Return (method_name, params_dict) for the selected supervised method."""
        method_text = self._sup_method.currentText()
        algorithm = self._sup_methods.get(method_text, "random_forest")
        params = collect_param_values(self._sup_param_widgets)
        # Filter None values and handle max_depth=0 as None (unlimited)
        cleaned = {}
        for k, v in params.items():
            if v is None:
                continue
            if k == "max_depth" and v == 0:
                continue  # 0 means unlimited, don't pass it
            cleaned[k] = v
        return algorithm, cleaned

    def _on_train(self):
        if self.state.features_df is None:
            QMessageBox.warning(self, "GeoOBIA", "Extract features first.")
            return
        if not self.state.training_samples:
            QMessageBox.warning(self, "GeoOBIA", "Select training samples first.")
            return

        method, params = self._get_sup_method_and_params()
        log(f"Train: {method}, params={params}")

        self._status.setText(f"Training ({method})...")

        # Snapshot data before entering background thread
        import pandas as pd
        features = self.state.features_df.copy()
        training_labels = pd.Series(
            dict(self.state.training_samples), name="class_label")

        def work(set_progress, is_canceled):
            from geobia.classification.supervised import SupervisedClassifier
            set_progress(5)
            clf = SupervisedClassifier(algorithm=method, **params)
            clf.fit(features, training_labels)
            set_progress(50)
            predictions = clf.predict(features)
            set_progress(100)
            return {"predictions": predictions, "classifier": clf}

        def on_success(result):
            predictions = result["predictions"]
            self._trained_classifier = result["classifier"]
            self.state.predictions = predictions
            n_classes = predictions.nunique()
            self._status.setText(f"Classification done: {n_classes} classes.")
            log(f"Train done: {n_classes} classes, {len(predictions)} segments")
            self._show_feature_importance(result["classifier"])
            # Auto-show results on the map
            self._auto_show_results()

        def on_failure(error_msg):
            log(f"Train FAILED:\n{error_msg}", Qgis.Critical)
            self._status.setText("Training failed — see Log Messages.")
            QMessageBox.warning(self, "GeoOBIA",
                                f"Training failed:\n{error_msg}")

        from .tasks import BackgroundTask, run_task
        task = BackgroundTask(
            f"GeoOBIA: {method} classification",
            work, on_success, on_failure,
        )
        run_task(self, task, progress_widget=self._progress)

    def _on_cluster(self):
        if self.state.features_df is None:
            QMessageBox.warning(self, "GeoOBIA", "Extract features first.")
            return

        method_text = self._unsup_method.currentText()
        method = self._unsup_methods.get(method_text, "kmeans")
        params = collect_param_values(self._unsup_param_widgets)
        params = {k: v for k, v in params.items() if v is not None}

        log(f"Cluster: {method}, params={params}")

        self._status.setText(f"Clustering with {method}...")

        features = self.state.features_df.copy()

        def work(set_progress, is_canceled):
            from geobia.classification import classify
            set_progress(5)
            predictions = classify(features, method=method, **params)
            set_progress(100)
            return predictions

        def on_success(predictions):
            self.state.predictions = predictions
            n_classes = predictions.nunique()
            self._status.setText(f"Clustering done: {n_classes} clusters.")
            log(f"Cluster done: {n_classes} clusters, {len(predictions)} segments")
            # Auto-show results on the map
            self._auto_show_results()

        def on_failure(error_msg):
            log(f"Cluster FAILED:\n{error_msg}", Qgis.Critical)
            self._status.setText("Clustering failed — see Log Messages.")
            QMessageBox.warning(self, "GeoOBIA",
                                f"Clustering failed:\n{error_msg}")

        from .tasks import BackgroundTask, run_task
        task = BackgroundTask(
            f"GeoOBIA: {method} clustering",
            work, on_success, on_failure,
        )
        run_task(self, task, progress_widget=self._progress)
