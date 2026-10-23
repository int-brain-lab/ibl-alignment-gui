"""Smoke tests that the GUI can be built.

These construct the main window as the launchers do, without loading any data. They catch
breakages that unit tests of the handlers and loaders miss, such as a dependency upgrade that
makes a plugin (and with it the whole app) fail to import.
"""

import os
import unittest

from qtpy import QtWidgets

from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController


class TestLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Run without a display server unless one has been chosen explicitly.
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _launch(self, **kwargs) -> AlignmentGUIController:
        controller = AlignmentGUIController(offline=True, csv=None, yaml=None, **kwargs)
        self.addCleanup(self.app.processEvents)
        self.addCleanup(controller.view.close)
        controller.view.show()
        self.app.processEvents()
        return controller

    def test_launch_offline(self):
        """Test the offline GUI builds and registers its plugins"""
        controller = self._launch()
        self.assertTrue(controller.view.isVisible())
        self.assertFalse(controller.loaded)
        self.assertCountEqual(
            controller.plugins,
            ['3D features', 'Channel Prediction', 'Cluster Features', 'Range Controller'],
        )

    def test_launch_allen(self):
        """Test the Allen GUI builds and registers the Allen plugins too"""
        controller = self._launch(allen=True)
        self.assertTrue(controller.view.isVisible())
        self.assertCountEqual(
            controller.plugins,
            [
                '3D features',
                'Additional Plots Allen',
                'Channel Prediction',
                'Cluster Features',
                'Custom Filters',
                'Range Controller',
            ],
        )


if __name__ == '__main__':
    unittest.main()
