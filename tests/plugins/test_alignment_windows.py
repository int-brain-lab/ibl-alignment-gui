"""Tests for the auto-alignment result windows and the 'Show auto-alignment results' menu item.

The windows, the menu item and the figure are replaced by stand-ins: no window is opened, no model
is used and ``ephysatlas`` does not need to be installed.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ibl_alignment_gui.plugins import channel_prediction as cp
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    AUTO_ALIGNMENT_CONFIDENCE_KEY,
    AUTO_ALIGNMENT_KEY,
    PLUGIN_KEY,
    alignment_windows,
    show_alignment_windows,
)
from iblutil.util import Bunch
from tests.plugins._stubs import import_spatial_encoder

spatial_encoder = import_spatial_encoder()

RESULT = SimpleNamespace(label='Histology', pid='pid')
PROBE = dict(planned_xyz=None)


def _shank(name: str, config: str = 'default') -> SimpleNamespace:
    """A shank with both auto-alignment views cached."""
    predictions = Bunch({AUTO_ALIGNMENT_KEY: Bunch(), AUTO_ALIGNMENT_CONFIDENCE_KEY: Bunch()})
    return SimpleNamespace(
        name=name, config=config, model=SimpleNamespace(predictions=predictions), view=MagicMock()
    )


class TestAlignmentWindows(unittest.TestCase):
    def setUp(self):
        # The menu item, greyed out as setup leaves it.
        self.action = MagicMock()
        self.controller = SimpleNamespace(
            plugins={PLUGIN_KEY: Bunch(alignment_results_action=self.action)},
            view=None,
            model=SimpleNamespace(brain_atlas=None),
        )
        # Every window made is a mock that records show/close calls.
        self.windows_made = []

        def make_window(*args):
            window = MagicMock()
            window.isVisible.return_value = False
            self.windows_made.append(window)
            return window

        self.enterContext(patch.object(spatial_encoder, 'AlignmentResultWindow', make_window))
        # The figure is drawn by ephysatlas, imported when a window is made.
        plotting = types.ModuleType('ephysatlas.alignment.plotting')
        plotting.plot_alignment_result = MagicMock()
        stubs = {
            'ephysatlas': types.ModuleType('ephysatlas'),
            'ephysatlas.alignment': types.ModuleType('ephysatlas.alignment'),
            'ephysatlas.alignment.plotting': plotting,
        }
        self.enterContext(patch.dict(sys.modules, stubs))

    def windows(self) -> dict:
        return alignment_windows(self.controller)

    def open_window(self, items) -> None:
        spatial_encoder._open_result_window(self.controller, items, RESULT, PROBE)

    def menu_enabled(self) -> bool:
        """Whether the menu item was last enabled (False if never set)."""
        calls = self.action.setEnabled.call_args_list
        return bool(calls and calls[-1].args[0])

    def test_window_opened_once(self):
        """A new window is kept under (shank, config) and shown once"""
        self.open_window(_shank('probe00a'))
        self.assertEqual(list(self.windows()), [('probe00a', 'default')])
        self.windows()['probe00a', 'default'].show.assert_called_once()

    def test_menu_item_enabled_after_first_window(self):
        """The menu item is greyed out until a window exists, then enabled"""
        self.assertFalse(self.menu_enabled())
        self.open_window(_shank('probe00a'))
        self.assertTrue(self.menu_enabled())

    def test_menu_shows_all_windows(self):
        """The menu item shows every kept window again, without making new ones"""
        self.open_window(_shank('probe00a'))
        self.open_window(_shank('probe00b'))
        show_alignment_windows(self.controller)
        self.assertEqual(len(self.windows_made), 2)
        for window in self.windows_made:
            self.assertEqual(window.show.call_count, 2)
            window.raise_.assert_called_once()

    def test_selecting_option_does_not_show_window(self):
        """Selecting either auto-alignment option draws its view but shows no window"""
        items = _shank('probe00a')
        self.open_window(items)
        self.controller.all_shanks = ['probe00a']
        self.controller.shank_items = {'probe00a': {'default': items}}
        self.controller.model.configs = ['default']
        self.controller.model.get_current_shank = lambda s, c: SimpleNamespace(align_exists=True)
        loader = cp.ChannelPrediction(self.controller)
        for option in (AUTO_ALIGNMENT_KEY, AUTO_ALIGNMENT_CONFIDENCE_KEY):
            loader.plot_regions(option)
        self.assertEqual(items.view.plot_histology.call_count, 2)
        self.windows_made[0].show.assert_called_once()  # only when it was opened

    def test_invalidate_closes_shank_window(self):
        """Invalidating a shank closes and forgets its window only; the menu item stays enabled"""
        a, b = _shank('probe00a'), _shank('probe00b')
        self.open_window(a)
        self.open_window(b)
        window_a = self.windows()['probe00a', 'default']
        # The function without the shank_loop decorator, for one shank.
        spatial_encoder.invalidate_predictions.__wrapped__(self.controller, a)
        window_a.close.assert_called_once()
        self.assertEqual(list(self.windows()), [('probe00b', 'default')])
        self.assertEqual(dict(a.model.predictions), {})
        self.assertTrue(self.menu_enabled())

    def test_menu_item_greyed_out_when_no_windows_left(self):
        """Invalidating the last aligned shank greys the menu item out again"""
        items = _shank('probe00a')
        self.open_window(items)
        spatial_encoder.invalidate_predictions.__wrapped__(self.controller, items)
        self.assertFalse(self.menu_enabled())

    def test_new_session_closes_all_windows(self):
        """Loading a new session closes every window and greys the menu item out"""
        for name in ('probe00a', 'probe00b'):
            self.open_window(_shank(name))
        with patch.object(cp, '_refresh_model_options'):
            cp._on_data_loaded(self.controller)
        for window in self.windows_made:
            window.close.assert_called_once()
        self.assertEqual(self.windows(), {})
        self.assertFalse(self.menu_enabled())


if __name__ == '__main__':
    unittest.main()
