"""Tests for offering the feature and prediction menus, and the hints of greyed-out menus."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qtpy import QtCore, QtGui, QtWidgets

from ibl_alignment_gui.app.controllers import app_controller
from ibl_alignment_gui.app.controllers.app_controller import (
    NO_FEATURES_HINT,
    NO_SPIKE_SORTING_HINT,
    AlignmentGUIController,
)
from ibl_alignment_gui.app.widgets.custom_widgets import MenuWidget


class TestMenuHint(unittest.TestCase):
    """The menu bar shows why a menu is greyed out when the mouse rests on it."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = QtWidgets.QMainWindow()
        self.addCleanup(self.window.deleteLater)
        self.bar = MenuWidget(self.window)
        self.window.setMenuBar(self.bar)
        self.window.show()
        QtWidgets.QApplication.processEvents()

    def hover(self, tab: str) -> str:
        """The tooltip shown when the mouse rests on a menu, or '' when none is shown."""
        QtWidgets.QToolTip.hideText()
        pos = self.bar.actionGeometry(self.bar.tabs[tab]['menu'].menuAction()).center()
        event = QtGui.QHelpEvent(QtCore.QEvent.ToolTip, pos, self.bar.mapToGlobal(pos))
        QtWidgets.QApplication.sendEvent(self.bar, event)
        QtWidgets.QApplication.processEvents()
        return QtWidgets.QToolTip.text() if QtWidgets.QToolTip.isVisible() else ''

    def test_disabled_menu_shows_its_hint(self):
        menu = self.bar.tabs['prediction']['menu']
        menu.setEnabled(False)
        menu.menuAction().setToolTip('the reason')
        self.assertEqual(self.hover('prediction'), 'the reason')

    def test_enabled_menu_shows_no_hint(self):
        self.bar.tabs['prediction']['menu'].menuAction().setToolTip('the reason')
        self.assertEqual(self.hover('prediction'), '')


class TestUpdateFeatureMenus(unittest.TestCase):
    """The filter menu needs spike sorting; the feature and prediction menus need ephysatlas
    and features."""

    def controller(self, feature_keys=(), prediction_keys=(), has_features=False, spikes=False):
        controller = SimpleNamespace(
            view=MagicMock(),
            prediction_keys=list(prediction_keys),
            model=SimpleNamespace(
                feature_keys=list(feature_keys),
                has_features=has_features,
                has_spike_sorting=spikes,
            ),
        )
        AlignmentGUIController.update_data_menus(controller)
        return controller.view

    def test_filters_need_spike_sorting(self):
        for spikes in (True, False):
            with self.subTest(spikes=spikes):
                view = self.controller(spikes=spikes)
                view.set_menu_enabled.assert_any_call('filter', spikes, NO_SPIKE_SORTING_HINT)

    def test_hidden_without_ephysatlas(self):
        with patch.object(app_controller.plot_loader, 'EPHYS_ATLAS', False):
            view = self.controller()
        view.set_menu_visible.assert_any_call('feature', False)
        view.set_menu_visible.assert_any_call('prediction', False)
        enabled = [c.args[0] for c in view.set_menu_enabled.call_args_list]
        self.assertNotIn('feature', enabled)
        self.assertNotIn('prediction', enabled)

    def test_greyed_out_without_features(self):
        with patch.object(app_controller.plot_loader, 'EPHYS_ATLAS', True):
            view = self.controller(prediction_keys=['Model'])
        view.set_menu_visible.assert_any_call('prediction', True)
        view.set_menu_enabled.assert_any_call('feature', False, NO_FEATURES_HINT)
        view.set_menu_enabled.assert_any_call('prediction', False, NO_FEATURES_HINT)

    def test_enabled_with_features(self):
        with patch.object(app_controller.plot_loader, 'EPHYS_ATLAS', True):
            view = self.controller(['Ephys Atlas'], ['Model'], has_features=True)
        view.set_menu_enabled.assert_any_call('feature', True, NO_FEATURES_HINT)
        view.set_menu_enabled.assert_any_call('prediction', True, NO_FEATURES_HINT)


if __name__ == '__main__':
    unittest.main()
