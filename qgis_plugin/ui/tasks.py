"""Background task runner using QThread for thread-safe computation.

Uses QThread + worker QObject instead of QgsTask, because QgsTask's
internal C++ thread pool causes crashes with numba JIT (pyshepseg)
and other native extensions that need a proper Python thread state.

Usage pattern:
    1. Read all data from QGIS layers into plain Python/numpy objects
    2. Create a BackgroundTask with a work function (no QGIS access)
    3. Submit via run_task() which stores references to prevent GC

Progress is shown via QGIS's message bar.
"""

import traceback

from qgis.PyQt.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from qgis.core import QgsMessageLog, Qgis

TAG = "GeoOBIA"


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


class _Worker(QObject):
    """Executes work_fn in a background thread, emitting signals."""

    progress = pyqtSignal(float)
    finished = pyqtSignal(object)    # result or None
    failed = pyqtSignal(str)         # error traceback

    def __init__(self, work_fn):
        super().__init__()
        self._work_fn = work_fn
        self._canceled = False

    def cancel(self):
        self._canceled = True

    @pyqtSlot()
    def run(self):
        try:
            result = self._work_fn(
                set_progress=self._emit_progress,
                is_canceled=lambda: self._canceled,
            )
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _emit_progress(self, percent):
        self.progress.emit(float(percent))


class BackgroundTask:
    """Run a callable in a background QThread with progress reporting.

    Args:
        description: Human-readable task name for logging.
        work_fn: Callable(set_progress, is_canceled) -> result.
            - set_progress(percent: float) updates progress (0-100).
            - is_canceled() -> bool returns True if canceled.
            Must NOT access any QGIS objects (layers, project, canvas).
        on_success: Callable(result) invoked on the GUI thread.
        on_failure: Callable(error_message: str) invoked on the GUI thread.
    """

    def __init__(self, description, work_fn, on_success=None, on_failure=None):
        self.description = description
        self._thread = QThread()
        self._worker = _Worker(work_fn)
        self._on_success = on_success
        self._on_failure = on_failure
        self._iface = None

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._handle_finished)
        self._worker.failed.connect(self._handle_failed)
        self._worker.progress.connect(self._handle_progress)

        # Clean up thread when done
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)

    def start(self, iface=None):
        """Start the background thread. Optionally pass iface for progress."""
        self._iface = iface
        if iface:
            iface.statusBarIface().showMessage(f"{self.description}...")
        self._thread.start()

    def cancel(self):
        self._worker.cancel()

    def _handle_finished(self, result):
        if self._iface:
            self._iface.statusBarIface().clearMessage()
        if self._on_success:
            self._on_success(result)

    def _handle_failed(self, error_msg):
        if self._iface:
            self._iface.statusBarIface().clearMessage()
        log(f"Task '{self.description}' failed:\n{error_msg}", Qgis.Critical)
        if self._on_failure:
            self._on_failure(error_msg)

    def _handle_progress(self, percent):
        if self._iface:
            self._iface.statusBarIface().showMessage(
                f"{self.description}... {int(percent)}%")


def run_task(owner, task):
    """Start a BackgroundTask, storing references on owner to prevent GC.

    Args:
        owner: Any object with a persistent lifetime (e.g. a panel widget).
            Must have an `iface` attribute for status bar progress.
        task: BackgroundTask instance.
    """
    owner._active_task = task
    iface = getattr(owner, "iface", None)
    task.start(iface)
