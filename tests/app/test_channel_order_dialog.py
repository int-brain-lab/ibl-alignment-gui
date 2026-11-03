"""Tests for telling the user to recompute RMS and spectra that may show the wrong channels."""

import unittest
from unittest.mock import MagicMock, patch

from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController


class TestChannelOrderDialog(unittest.TestCase):
    def setUp(self):
        self.controller = MagicMock()
        self.controller._load_start = 0

    @patch('ibl_alignment_gui.app.controllers.app_controller.QtWidgets.QMessageBox.warning')
    def test_dialog_once_loaded(self, mock_warning):
        """The notice is shown in a dialog once the data are loaded"""
        self.controller.model.get_channel_order_notice.return_value = 'recompute'
        AlignmentGUIController._on_load_finished(self.controller)
        mock_warning.assert_called_once_with(
            self.controller.view, 'Recompute the RMS', 'recompute'
        )

    @patch('ibl_alignment_gui.app.controllers.app_controller.QtWidgets.QMessageBox.warning')
    def test_no_dialog_without_notices(self, mock_warning):
        self.controller.model.get_channel_order_notice.return_value = None
        AlignmentGUIController._on_load_finished(self.controller)
        mock_warning.assert_not_called()


if __name__ == '__main__':
    unittest.main()
