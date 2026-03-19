"""QgsTask wrapper with GC prevention and progress reporting.

Usage pattern:
    1. Read all data from QGIS layers into plain Python/numpy objects
    2. Create a BackgroundTask with a work function (no QGIS access)
    3. Connect finished/failed signals
    4. Submit via run_task() which stores a reference to prevent GC
"""

import traceback

from qgis.core import QgsApplication, QgsTask, QgsMessageLog, Qgis

TAG = "GeoOBIA"


def log(msg, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), TAG, level)


class BackgroundTask(QgsTask):
    """Run a callable in the background with progress reporting.

    Args:
        description: Human-readable task name shown in QGIS task manager.
        work_fn: Callable(set_progress, is_canceled) -> result.
            - set_progress(percent: float) updates the progress bar (0-100).
            - is_canceled() -> bool returns True if the user canceled.
            The function must NOT access any QGIS objects (layers, project,
            map canvas). All data should be passed via closure or functools.partial.
        on_success: Callable(result) invoked on the GUI thread with the return
            value of work_fn.
        on_failure: Callable(error_message: str) invoked on the GUI thread.
    """

    def __init__(self, description, work_fn, on_success=None, on_failure=None):
        super().__init__(description, QgsTask.CanCancel)
        self._work_fn = work_fn
        self._on_success = on_success
        self._on_failure = on_failure
        self._result = None
        self._error = None

    def run(self):
        """Executed in a background thread."""
        try:
            self._result = self._work_fn(
                set_progress=self.setProgress,
                is_canceled=self.isCanceled,
            )
            return True
        except Exception:
            self._error = traceback.format_exc()
            return False

    def finished(self, success):
        """Executed on the GUI thread."""
        if success:
            if self._on_success:
                self._on_success(self._result)
        else:
            if self.isCanceled():
                log(f"Task '{self.description()}' was canceled.")
            elif self._on_failure:
                self._on_failure(self._error)
            elif self._error:
                log(f"Task '{self.description()}' failed:\n{self._error}",
                    Qgis.Critical)


def run_task(owner, task):
    """Submit a BackgroundTask, storing a reference on owner to prevent GC.

    Args:
        owner: Any object with a persistent lifetime (e.g. a panel widget).
            The task is stored as owner._active_task.
        task: BackgroundTask instance.
    """
    owner._active_task = task
    QgsApplication.taskManager().addTask(task)
