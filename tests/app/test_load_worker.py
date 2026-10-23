"""Tests for running slow tasks on a background thread: the worker, its dialog and the runner.

They run offscreen. The runner is the controller's :meth:`AlignmentGUIController.run_in_thread`,
called on a small stand-in holding only what it uses, so no GUI or data is built.
"""

import os
import threading
import unittest

from qtpy import QtCore, QtGui, QtTest, QtWidgets

from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
from ibl_alignment_gui.app.load_worker import TaskProgressDialog, Worker


def _setup_qt(cls) -> None:
    # Run without a display server unless one has been chosen explicitly.
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Recorder:
    """Records the signals a worker emits."""

    def __init__(self, worker: Worker):
        self.emitted = []
        worker.progress.connect(lambda *args: self.emitted.append(('progress', args)))
        worker.finished.connect(lambda result: self.emitted.append(('finished', result)))
        worker.error.connect(lambda message: self.emitted.append(('error', message)))
        worker.cancelled.connect(lambda: self.emitted.append(('cancelled', None)))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.emitted]


class TestWorker(unittest.TestCase):
    """The worker run directly, on the test's thread."""

    @classmethod
    def setUpClass(cls):
        _setup_qt(cls)

    def test_finished(self):
        """A task's result is emitted with finished"""
        worker = Worker(lambda x: x * 2, 21)
        recorder = _Recorder(worker)
        worker.run()
        self.assertEqual(recorder.emitted, [('finished', 42)])

    def test_error(self):
        """An exception is emitted with error, as its traceback"""

        def fail():
            raise ValueError('boom')

        worker = Worker(fail)
        recorder = _Recorder(worker)
        with self.assertLogs('ibl_alignment_gui.app.load_worker', 'ERROR'):
            worker.run()
        self.assertEqual(recorder.kinds(), ['error'])
        self.assertIn('ValueError: boom', recorder.emitted[0][1])

    def test_cancel_at_next_progress(self):
        """After cancel, the task's next progress report stops it and cancelled is emitted"""
        worker = None
        reached = []

        def task(progress_callback):
            progress_callback('step 1', 1, 2)
            worker.cancel()
            progress_callback('step 2', 2, 2)
            reached.append('after step 2')

        worker = Worker(task, report_progress=True)
        recorder = _Recorder(worker)
        worker.run()
        self.assertEqual(recorder.kinds(), ['progress', 'cancelled'])
        self.assertEqual(reached, [])

    def test_cancel_without_progress_finishes(self):
        """A task that does not report progress after cancel finishes normally"""
        worker = None

        def task(progress_callback):
            worker.cancel()
            return 'done'

        worker = Worker(task, report_progress=True)
        recorder = _Recorder(worker)
        worker.run()
        self.assertEqual(recorder.emitted, [('finished', 'done')])


class TestTaskProgressDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_qt(cls)

    def dialog(self, cancellable: bool = True) -> tuple[TaskProgressDialog, list]:
        dialog = TaskProgressDialog('Title', 'Working…', cancellable, None)
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.finish)
        requests = []
        dialog.cancel_requested.connect(lambda: requests.append(1))
        dialog.show()
        self.app.processEvents()
        return dialog, requests

    def assert_cancelling(self, dialog: TaskProgressDialog, requests: list) -> None:
        self.app.processEvents()
        self.assertEqual(requests, [1])
        self.assertTrue(dialog.isVisible())
        self.assertIn('Cancelling', dialog.labelText())
        self.assertFalse(dialog.findChild(QtWidgets.QPushButton).isEnabled())

    def test_cancel_button(self):
        """Cancel asks the task to stop once and keeps the dialog open"""
        dialog, requests = self.dialog()
        button = dialog.findChild(QtWidgets.QPushButton)
        button.click()
        button.click()  # disabled: no second request
        self.assert_cancelling(dialog, requests)

    def test_escape(self):
        """Escape is the same as Cancel"""
        dialog, requests = self.dialog()
        QtTest.QTest.keyClick(dialog, QtCore.Qt.Key_Escape)
        self.assert_cancelling(dialog, requests)

    def test_close(self):
        """Closing the window is the same as Cancel"""
        dialog, requests = self.dialog()
        self.app.sendEvent(dialog, QtGui.QCloseEvent())
        self.assert_cancelling(dialog, requests)

    def test_not_cancellable(self):
        """Without a Cancel button, Escape and closing do nothing"""
        dialog, requests = self.dialog(cancellable=False)
        self.assertIsNone(dialog.findChild(QtWidgets.QPushButton))
        QtTest.QTest.keyClick(dialog, QtCore.Qt.Key_Escape)
        self.app.sendEvent(dialog, QtGui.QCloseEvent())
        self.app.processEvents()
        self.assertEqual(requests, [])
        self.assertTrue(dialog.isVisible())

    def test_finish_closes(self):
        """finish closes the dialog without asking the task to stop"""
        dialog, requests = self.dialog()
        dialog.finish()
        self.app.processEvents()
        self.assertFalse(dialog.isVisible())
        self.assertEqual(requests, [])


class _Runner:
    """What run_in_thread uses of the controller, with the controller's own methods."""

    run_in_thread = AlignmentGUIController.run_in_thread
    task_running = AlignmentGUIController.task_running
    _set_outcome = AlignmentGUIController._set_outcome
    _on_thread_progress = AlignmentGUIController._on_thread_progress
    _cleanup_thread = AlignmentGUIController._cleanup_thread

    def __init__(self):
        self.view = None
        self._load_thread = None
        self._load_worker = None
        self._load_dialog = None
        self._outcome = None
        self.errors = []

    def _on_thread_error(self, message: str) -> None:
        self.errors.append(message)


class TestRunInThread(unittest.TestCase):
    """The runner with a real background thread."""

    @classmethod
    def setUpClass(cls):
        _setup_qt(cls)

    def setUp(self):
        self.runner = _Runner()
        self.calls = []

    def wait(self) -> None:
        """Wait for the runner to be free again."""
        for _ in range(500):
            if not self.runner.task_running:
                break
            QtTest.QTest.qWait(10)
        self.assertFalse(self.runner.task_running, 'the task did not stop')

    def run_task(self, func, **kwargs) -> bool:
        kwargs.setdefault('on_finished', lambda result: self.calls.append(('finished', result)))
        return self.runner.run_in_thread(func, busy_message='Working…', **kwargs)

    def test_finished(self):
        """on_finished gets the result, on the main thread, and the dialog closes"""
        callback_threads = []

        def on_finished(result):
            callback_threads.append(threading.get_ident())
            self.calls.append(('finished', result))

        self.assertTrue(self.run_task(threading.get_ident, on_finished=on_finished))
        dialog = self.runner._load_dialog
        self.wait()
        # The task ran on another thread; its callback on this (the main) one.
        self.assertEqual(len(self.calls), 1)
        self.assertNotEqual(self.calls[0][1], threading.get_ident())
        self.assertEqual(callback_threads, [threading.get_ident()])
        self.assertFalse(dialog.isVisible())

    def test_cancelled(self):
        """A task cancelled from its dialog calls on_cancelled, not on_finished"""
        started = threading.Event()

        def task(progress_callback):
            started.set()
            while True:
                progress_callback('working', 0, 0)
                threading.Event().wait(0.01)

        self.run_task(
            task,
            report_progress=True,
            cancellable=True,
            on_cancelled=lambda: self.calls.append(('cancelled', None)),
        )
        self.assertTrue(started.wait(5))
        self.runner._load_dialog.findChild(QtWidgets.QPushButton).click()
        self.wait()
        self.assertEqual(self.calls, [('cancelled', None)])

    def test_error(self):
        """A task that raises calls on_error with its traceback"""

        def fail():
            raise ValueError('boom')

        with self.assertLogs('ibl_alignment_gui.app.load_worker', 'ERROR'):
            self.run_task(fail, on_error=lambda message: self.calls.append(('error', message)))
            self.wait()
        self.assertEqual([kind for kind, _ in self.calls], ['error'])
        self.assertIn('ValueError: boom', self.calls[0][1])

    def test_default_error_handler(self):
        """Without on_error, the controller's error handler reports it"""

        def fail():
            raise ValueError('boom')

        with self.assertLogs('ibl_alignment_gui.app.load_worker', 'ERROR'):
            self.run_task(fail)
            self.wait()
        self.assertEqual(len(self.runner.errors), 1)

    def test_one_task_at_a_time(self):
        """A second task is refused while one runs"""
        release = threading.Event()
        self.assertTrue(self.run_task(lambda: release.wait(5)))
        self.assertFalse(self.run_task(lambda: None))
        release.set()
        self.wait()
        self.assertEqual(len(self.calls), 1)

    def test_callback_can_start_task(self):
        """The finished callback runs once the runner is free, so it can start another task"""

        def first_done(result):
            self.calls.append(('first', result))
            self.calls.append(('started second', self.run_task(lambda: 'second')))

        self.run_task(lambda: 'first', on_finished=first_done)
        self.wait()
        self.wait()
        self.assertEqual(
            self.calls, [('first', 'first'), ('started second', True), ('finished', 'second')]
        )


if __name__ == '__main__':
    unittest.main()
