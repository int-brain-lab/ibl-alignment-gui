"""Tests for choosing the atlas mapping of the histology from the Set Mapping menu."""

import unittest
from unittest.mock import MagicMock

from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController


class TestSetMapping(unittest.TestCase):
    def test_redraws_the_histology_and_keeps_the_reference_lines(self):
        """The histology is redrawn in the mapping, then the reference lines are added back"""
        controller = MagicMock()
        controller.set_mapping.return_value = [None]
        controller.plot_histology_ref_panels.return_value = [None]

        AlignmentGUIController.plot_region_ref_panels(controller, 'Beryl')

        self.assertEqual(controller.region_init, 'Beryl')
        controller.set_mapping.assert_called_once_with(mapping='Beryl')
        calls = [c[0] for c in controller.method_calls]
        # The lines are added back after the histology figure is cleared and redrawn
        self.assertLess(
            calls.index('plot_histology_panels'), calls.index('add_reference_lines_to_display')
        )
        self.assertLess(
            calls.index('remove_reference_lines_from_display'),
            calls.index('add_reference_lines_to_display'),
        )
        controller.align_reference_lines.assert_called_once()

    def test_setting_up_the_display_leaves_the_reference_lines(self):
        """While the display is set up, the reference lines are added by the setup, not here"""
        controller = MagicMock()
        controller.set_mapping.return_value = [None]
        controller.plot_histology_ref_panels.return_value = [None]

        AlignmentGUIController.plot_region_ref_panels(controller, 'Allen', redraw_lines=False)

        controller.plot_histology_panels.assert_called_once()
        controller.remove_reference_lines_from_display.assert_not_called()
        controller.add_reference_lines_to_display.assert_not_called()
        controller.align_reference_lines.assert_not_called()


if __name__ == '__main__':
    unittest.main()
