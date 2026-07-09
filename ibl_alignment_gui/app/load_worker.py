"""Generic background worker for running slow tasks off the GUI thread."""

import logging
import traceback
from collections.abc import Callable
from typing import Any

from qtpy import QtCore

logger = logging.getLogger(__name__)


class Worker(QtCore.QObject):
    """
    Run a single callable on a background thread and report the outcome via Qt signals.

    The worker runs ``func(*args, **kwargs)`` when its :meth:`run` slot is invoked (typically by
    a ``QThread.started`` signal). It must not touch any Qt widgets; results are delivered back to
    the main thread through the ``finished`` and ``error`` signals (Qt marshals cross-thread
    signal emissions onto the receiving thread's event loop).

    When ``report_progress`` is True the worker injects its own progress emitter as a
    ``progress_callback`` keyword argument, so ``func`` can report progress by calling
    ``progress_callback(message, current, total)``.

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
    """

    progress = QtCore.Signal(str, int, int)
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, func: Callable, *args: Any, report_progress: bool = False, **kwargs: Any):
        super().__init__()
        self._func: Callable = func
        self._args: tuple = args
        self._kwargs: dict = kwargs
        self._report_progress: bool = report_progress

    def run(self) -> None:
        """Run the callable, emitting ``finished`` with its result or ``error`` on failure."""
        try:
            if self._report_progress:
                self._kwargs['progress_callback'] = self._emit
            result = self._func(*self._args, **self._kwargs)
        except Exception:
            logger.exception('Background task failed')
            self.error.emit(traceback.format_exc())
            return

        self.finished.emit(result)

    def _emit(self, message: str, current: int, total: int) -> None:
        """Re-emit a progress update from ``func`` as a Qt signal."""
        self.progress.emit(message, current, total)
