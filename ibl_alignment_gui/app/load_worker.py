"""Generic background worker for running slow tasks off the GUI thread."""

import logging
import threading
import traceback
from collections.abc import Callable
from typing import Any

from qtpy import QtCore, QtWidgets

logger = logging.getLogger(__name__)


class TaskCancelledError(Exception):
    """Raised inside a background task, at its next progress report, after Cancel is pressed."""


class Worker(QtCore.QObject):
    """
    Run a single callable on a background thread and report the outcome via Qt signals.

    The worker runs ``func(*args, **kwargs)`` when its :meth:`run` slot is invoked (typically by
    a ``QThread.started`` signal). It must not touch any Qt widgets; results are delivered back to
    the main thread through the ``finished`` and ``error`` signals (Qt marshals cross-thread
    signal emissions onto the receiving thread's event loop).

    When ``report_progress`` is True the worker injects its own progress emitter as a
    ``progress_callback`` keyword argument, so ``func`` can report progress by calling
    ``progress_callback(message, current, total)``. After :meth:`cancel`, the next call raises
    :class:`TaskCancelledError`, which stops the task; a task can also call it only to check for
    Cancel.

    Parameters
    ----------
    func : Callable
        The callable to run on the background thread.
    *args : Any
        Positional arguments forwarded to ``func``.
    report_progress : bool
        If True, pass a ``progress_callback`` keyword to ``func`` that re-emits as the
        ``progress`` signal.
    **kwargs : Any
        Keyword arguments forwarded to ``func``.

    Attributes
    ----------
    progress : QtCore.Signal
        Emitted as ``(message, current, total)`` while ``func`` runs (only if it reports progress).
    finished : QtCore.Signal
        Emitted with the return value of ``func`` once it completes successfully.
    error : QtCore.Signal
        Emitted with a formatted traceback string if ``func`` raises.
    cancelled : QtCore.Signal
        Emitted if ``func`` stopped because Cancel was pressed.
    """

    progress = QtCore.Signal(str, int, int)
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, func: Callable, *args: Any, report_progress: bool = False, **kwargs: Any):
        super().__init__()
        self._func: Callable = func
        self._args: tuple = args
        self._kwargs: dict = kwargs
        self._report_progress: bool = report_progress
        self._cancel = threading.Event()  # set from the main thread, read by the task

    def cancel(self) -> None:
        """Ask the task to stop: its next progress report raises :class:`TaskCancelledError`.

        Called from the main thread; the task stops at its next progress report, not part-way
        through a step that does not report (e.g. a download).
        """
        self._cancel.set()

    def run(self) -> None:
        """Run the callable, emitting ``finished`` with its result or ``error`` on failure."""
        try:
            if self._report_progress:
                self._kwargs['progress_callback'] = self._emit
            result = self._func(*self._args, **self._kwargs)
        except TaskCancelledError:
            self.cancelled.emit()
            return
        except Exception:
            logger.exception('Background task failed')
            self.error.emit(traceback.format_exc())
            return

        self.finished.emit(result)

    def _emit(self, message: str, current: int, total: int) -> None:
        """Re-emit a progress update from ``func`` as a Qt signal, or stop it after Cancel."""
        if self._cancel.is_set():
            raise TaskCancelledError()
        self.progress.emit(message, current, total)


class TaskProgressDialog(QtWidgets.QProgressDialog):
    """A modal progress dialog for a background task.

    Cancel (the button, Escape or closing the window) does not close it: it emits
    ``cancel_requested`` and the dialog stays until the task has stopped, as the task can only stop
    at its next progress report. Without a Cancel button, Escape and closing are ignored.

    Parameters
    ----------
    title : str
        The window title.
    message : str
        The initial message.
    cancellable : bool
        Whether the dialog has a Cancel button.
    parent : QtWidgets.QWidget or None
        The parent window.
    """

    cancel_requested = QtCore.Signal()

    def __init__(self, title: str, message: str, cancellable: bool, parent) -> None:
        super().__init__(message, 'Cancel' if cancellable else None, 0, 0, parent)
        self._cancellable = cancellable
        self._finished = False  # set by finish(), once the task has stopped
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.setMinimumDuration(0)
        self.setMinimumWidth(420)
        self.setAutoClose(False)
        self.setAutoReset(False)
        # By default Cancel resets and hides the dialog; keep it open until the task stops.
        self.canceled.disconnect(self.cancel)
        self.canceled.connect(self._request_cancel)
        self.setValue(0)

    def _request_cancel(self) -> None:
        """Ask the task to stop (once), saying so in the dialog."""
        button = self.findChild(QtWidgets.QPushButton)
        if self._finished or not self._cancellable or button is None or not button.isEnabled():
            return
        self.setLabelText('Cancelling… (waiting for the current step to finish)')
        button.setEnabled(False)
        self.cancel_requested.emit()

    def finish(self) -> None:
        """Close the dialog, once the task has stopped."""
        self._finished = True
        self.close()

    def reject(self) -> None:
        """Treat Escape (and closing the window) as Cancel until the task has stopped."""
        if self._finished:
            super().reject()
            return
        self._request_cancel()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Treat closing the window as Cancel; the dialog closes when the task stops."""
        if self._finished:
            super().closeEvent(event)
            return
        event.ignore()
        self._request_cancel()
