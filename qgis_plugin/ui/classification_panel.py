"""Classification workflow panel with supervised and unsupervised tabs."""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.core import QgsApplication, QgsTask

from .sample_selector import SampleSelectorTool

# Default class color palette
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
]


class ClassificationPanel(QWidget):
    def __init__(self, iface, state, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.state = state
        self._sample_tool = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Mode tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_supervised_tab(), "Supervised")
        self._tabs.addTab(self._build_unsupervised_tab(), "Unsupervised")
        layout.addWidget(self._tabs)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
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

        self._sample_count_label = QLabel("Samples: 0")
        sample_layout.addWidget(self._sample_count_label)

        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)

        # Algorithm
        algo_group = QGroupBox("Algorithm")
        algo_layout = QFormLayout()
        self._sup_n_estimators = QSpinBox()
        self._sup_n_estimators.setRange(1, 10000)
        self._sup_n_estimators.setValue(100)
        algo_layout.addRow("Trees (Random Forest):", self._sup_n_estimators)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # Train / Predict
        action_row = QHBoxLayout()
        train_btn = QPushButton("Train")
        train_btn.clicked.connect(self._on_train)
        action_row.addWidget(train_btn)
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self._on_predict)
        action_row.addWidget(predict_btn)
        layout.addLayout(action_row)

        tab.setLayout(layout)
        return tab

    # ---- Unsupervised tab ----

    def _build_unsupervised_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        algo_group = QGroupBox("Algorithm")
        algo_layout = QFormLayout()

        self._unsup_method = QComboBox()
        self._unsup_method.addItems(["kmeans", "gmm", "dbscan"])
        algo_layout.addRow("Method:", self._unsup_method)

        self._unsup_n_clusters = QSpinBox()
        self._unsup_n_clusters.setRange(2, 1000)
        self._unsup_n_clusters.setValue(8)
        algo_layout.addRow("Clusters:", self._unsup_n_clusters)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        cluster_btn = QPushButton("Cluster")
        cluster_btn.clicked.connect(self._on_cluster)
        layout.addWidget(cluster_btn)

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

        color_item = QTableWidgetItem("")
        color_item.setBackground(color)
        color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
        self._class_table.setItem(row, 1, color_item)

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

    def _on_class_color_click(self, row, col):
        if col != 1:
            return
        item = self._class_table.item(row, 1)
        color = QColorDialog.getColor(item.background().color(), self)
        if color.isValid():
            item.setBackground(color)
            name = self._class_table.item(row, 0).text()
            self.state.class_colors[name] = color

    # ---- Sample selection ----

    def _toggle_sample_tool(self, active):
        if active:
            labels_layer = self.state.labels_layer
            if labels_layer is None:
                QMessageBox.warning(
                    self, "GeoOBIA",
                    "Run segmentation first to create a labels layer.")
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

    def _on_sample_added(self, seg_id, class_name):
        self._update_sample_counts()

    def _on_sample_removed(self, seg_id):
        self._update_sample_counts()

    def _update_sample_counts(self):
        total = len(self.state.training_samples)
        self._sample_count_label.setText(f"Samples: {total}")

        # Update per-class counts in the table
        from collections import Counter
        counts = Counter(self.state.training_samples.values())
        for row in range(self._class_table.rowCount()):
            name = self._class_table.item(row, 0).text()
            self._class_table.item(row, 2).setText(str(counts.get(name, 0)))

    # ---- Training / Prediction ----

    def _on_train(self):
        if self.state.features_df is None:
            QMessageBox.warning(self, "GeoOBIA", "Extract features first.")
            return
        if not self.state.training_samples:
            QMessageBox.warning(self, "GeoOBIA", "Select training samples first.")
            return

        self._progress.setVisible(True)
        self._status.setText("Training...")

        task = _ClassifyTask(
            self.state,
            method="random_forest",
            params={"n_estimators": self._sup_n_estimators.value()},
            callback=self._on_classify_done,
        )
        QgsApplication.taskManager().addTask(task)

    def _on_predict(self):
        # Same as train for now — classify does fit + predict in one call
        self._on_train()

    def _on_cluster(self):
        if self.state.features_df is None:
            QMessageBox.warning(self, "GeoOBIA", "Extract features first.")
            return

        method = self._unsup_method.currentText()
        params = {}
        if method in ("kmeans", "gmm"):
            params["n_clusters"] = self._unsup_n_clusters.value()

        self._progress.setVisible(True)
        self._status.setText(f"Clustering with {method}...")

        task = _ClassifyTask(
            self.state, method=method, params=params,
            callback=self._on_classify_done)
        QgsApplication.taskManager().addTask(task)

    def _on_classify_done(self, success: bool, message: str):
        self._progress.setVisible(False)
        self._status.setText(message)
        if not success:
            QMessageBox.warning(self, "GeoOBIA", message)


class _ClassifyTask(QgsTask):
    def __init__(self, state, method, params, callback):
        super().__init__(f"GeoOBIA: Classify ({method})")
        self.state = state
        self.method = method
        self.params = params
        self.callback = callback
        self.error_msg = ""

    def run(self):
        try:
            import pandas as pd
            from geobia.classification import classify

            features = self.state.features_df
            training_labels = None

            if self.method == "random_forest":
                # Build training labels Series from the samples dict
                training_labels = pd.Series(
                    self.state.training_samples, name="class_label")

            self.setProgress(20)
            predictions = classify(
                features, method=self.method,
                training_labels=training_labels,
                **self.params)

            self.state.predictions = predictions
            self.setProgress(100)
            n_classes = predictions.nunique()
            self.callback(True, f"Classification done: {n_classes} classes.")
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def finished(self, result):
        if not result:
            self.callback(False, f"Classification failed: {self.error_msg}")
