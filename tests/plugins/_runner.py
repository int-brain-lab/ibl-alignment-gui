"""A stand-in for the controller's background-task runner, for the plugin tests."""

import traceback

from ibl_alignment_gui.app.load_worker import TaskCancelledError


class InlineRunner:
    """Stands in for ``controller.run_in_thread``: runs the task at once, on the test's thread.

    It calls the task's callback as the real runner does once the task has stopped. The real
    runner (with a thread) is tested in ``tests/app/test_load_worker.py``.

    Attributes
    ----------
    tasks : list of dict
        Each task started: its ``func``, ``title`` and whether it was ``cancellable``.
    progress : list of str
        Every progress message reported.
    busy : bool
        If True, refuse new tasks, as when another task is running.
    cancel_at : int or None
        Press Cancel when the task reports its ``cancel_at``-th progress message: that report
        raises ``TaskCancelledError``, as after Cancel.
    """

    def __init__(self):
        self.tasks = []
        self.progress = []
        self.busy = False
        self.cancel_at = None

    def __call__(
        self,
        func,
        *args,
        on_finished,
        busy_message='',
        report_progress=False,
        cancellable=False,
        on_error=None,
        on_cancelled=None,
        title='Please wait',
        **kwargs,
    ) -> bool:
        if self.busy:
            return False
        self.tasks.append(dict(func=func, title=title, cancellable=cancellable))
        if report_progress:
            kwargs['progress_callback'] = self._report
        try:
            result = func(*args, **kwargs)
        except TaskCancelledError:
            if on_cancelled is not None:
                on_cancelled()
        except Exception:
            on_error(traceback.format_exc())
        else:
            on_finished(result)
        return True

    def _report(self, message: str, current: int, total: int) -> None:
        self.progress.append(message)
        if self.cancel_at is not None and len(self.progress) >= self.cancel_at:
            raise TaskCancelledError()
