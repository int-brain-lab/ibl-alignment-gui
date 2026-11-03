"""Tests for drawing the ephys plot panels of each shank."""

import unittest
from unittest.mock import MagicMock

from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController


class TestPlotPanels(unittest.TestCase):
    def setUp(self):
        self.controller = MagicMock()
        self.controller._show_ephys_view.return_value = False
        self.controller.model.selected_config = 'default'
        self.controller.model.configs = ['default']
        self.controller.all_shanks = ['shank_1']
        # The shank has no alignment, e.g. a data folder without xyz picks
        self.controller.model.get_current_shank.return_value.align_exists = False
        self.items = MagicMock()
        self.controller.shank_items = {'shank_1': {'default': self.items}}
        # Run the real panel plotting and per shank loop rather than the mock's
        for name in ('plot_panels', '_plot_panels'):
            method = getattr(AlignmentGUIController, name)
            setattr(self.controller, name, method.__get__(self.controller))

    def test_data_plots_drawn_without_alignment(self):
        """Image, probe and scatter plots are drawn on shanks that have no alignment"""
        for method, plot_func in (
            (AlignmentGUIController.plot_image_panels, 'plot_image'),
            (AlignmentGUIController.plot_probe_panels, 'plot_probe'),
            (AlignmentGUIController.plot_scatter_panels, 'plot_scatter'),
        ):
            with self.subTest(plot_func=plot_func):
                method(self.controller, 'Correlation')
                getattr(self.items, plot_func).assert_called_once_with('Correlation')

    def test_histology_plots_need_alignment(self):
        """With data_only False, shanks without an alignment are skipped"""
        AlignmentGUIController.plot_image_panels(self.controller, 'Correlation', data_only=False)
        self.items.plot_image.assert_not_called()


if __name__ == '__main__':
    unittest.main()
