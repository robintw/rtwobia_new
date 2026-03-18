"""Map tool for selecting training samples by clicking on segments."""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor
from qgis.core import (
    QgsPointXY,
    QgsRasterDataProvider,
)
from qgis.gui import QgsMapTool


class SampleSelectorTool(QgsMapTool):
    """Click segments on the map to assign them as training samples.

    Left-click assigns the segment under the cursor to the active class.
    Right-click removes the segment from training samples.
    """

    sample_added = pyqtSignal(int, str)     # segment_id, class_name
    sample_removed = pyqtSignal(int)        # segment_id

    def __init__(self, canvas, labels_layer, training_samples, active_class_fn):
        """
        Args:
            canvas: QgsMapCanvas instance.
            labels_layer: QgsRasterLayer containing segment IDs.
            training_samples: dict[int, str] mapping segment_id -> class_name
                (modified in-place by this tool).
            active_class_fn: callable returning the currently active class name.
        """
        super().__init__(canvas)
        self.labels_layer = labels_layer
        self.training_samples = training_samples
        self.active_class_fn = active_class_fn
        self.setCursor(QCursor(Qt.CrossCursor))

    def canvasReleaseEvent(self, event):
        point = self.toMapCoordinates(event.pos())
        seg_id = self._identify_segment(point)
        if seg_id is None or seg_id <= 0:
            return

        if event.button() == Qt.LeftButton:
            class_name = self.active_class_fn()
            if class_name:
                self.training_samples[seg_id] = class_name
                self.sample_added.emit(seg_id, class_name)
        elif event.button() == Qt.RightButton:
            if seg_id in self.training_samples:
                del self.training_samples[seg_id]
                self.sample_removed.emit(seg_id)

    def _identify_segment(self, point: QgsPointXY):
        """Read the segment ID at the given map coordinate."""
        if self.labels_layer is None or not self.labels_layer.isValid():
            return None

        provider = self.labels_layer.dataProvider()
        result = provider.identify(
            point,
            QgsRasterDataProvider.IdentifyFormatValue,
        )
        if result.isValid():
            values = result.results()
            # Band 1 contains segment IDs
            return int(values.get(1, 0))
        return None
