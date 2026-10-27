"""Tests for switching between the ephys, feature and prediction views of the controller.

The view-switching methods run on a small stand-in holding only what they use, so no GUI or data
is built; rebuilding the display is recorded instead.
"""

import unittest
from unittest.mock import MagicMock

from ibl_alignment_gui.app.controllers.app_controller import (
    ALL_PREDICTIONS,
    NO_PREDICTION,
    AlignmentGUIController,
)


class _Controller:
    """The controller's view-switching methods, recording each rebuild of the display."""

    show_prediction_panel = AlignmentGUIController.show_prediction_panel
    show_ephys_plots = AlignmentGUIController.show_ephys_plots
    plot_prediction_panels = AlignmentGUIController.plot_prediction_panels
    _show_ephys_view = AlignmentGUIController._show_ephys_view

    def __init__(self):
        self.prediction_init = NO_PREDICTION
        self.show_feature = False
        self.show_predictions = False
        self.loader = MagicMock()
        self.plugins = {'Channel Prediction': {'loader': self.loader}}
        self.view = MagicMock()
        self.rebuilt = 0

    def on_view_changed(self):
        self.rebuilt += 1


class TestPredictionViews(unittest.TestCase):
    def setUp(self):
        self.controller = _Controller()

    def test_first_prediction_adds_panel(self):
        """The first prediction rebuilds the display with its panel; the next one is drawn"""
        self.controller.plot_prediction_panels('Model')
        self.assertEqual(self.controller.rebuilt, 1)
        self.assertTrue(self.controller.show_prediction_panel)
        self.controller.loader.plot_predictions.assert_not_called()
        # The rebuild selects the option again, which draws it
        self.controller.plot_prediction_panels('Model')
        self.controller.plot_prediction_panels('Other')
        self.assertEqual(self.controller.rebuilt, 1)
        self.assertEqual(
            [c.args for c in self.controller.loader.plot_predictions.call_args_list],
            [('Model',), ('Other',)],
        )

    def test_none_removes_panel(self):
        """'None' rebuilds the display without the panel, once"""
        self.controller.plot_prediction_panels('Model')
        self.controller.plot_prediction_panels(NO_PREDICTION)
        self.controller.plot_prediction_panels(NO_PREDICTION)
        self.assertEqual(self.controller.rebuilt, 2)
        self.assertFalse(self.controller.show_prediction_panel)

    def test_all_predictions(self):
        """'All predictions' replaces the ephys plots, then draws every prediction"""
        self.controller.plot_prediction_panels(ALL_PREDICTIONS)
        self.assertTrue(self.controller.show_predictions)
        self.assertFalse(self.controller.show_prediction_panel)
        self.assertFalse(self.controller.show_ephys_plots)
        self.assertEqual(self.controller.rebuilt, 1)
        self.controller.plot_prediction_panels(ALL_PREDICTIONS)
        self.controller.loader.plot_all_predictions.assert_called_once()
        self.assertEqual(self.controller.rebuilt, 1)

    def test_one_prediction_leaves_all(self):
        """Choosing one prediction from the all predictions view shows it alongside the plots"""
        self.controller.plot_prediction_panels(ALL_PREDICTIONS)
        self.controller.plot_prediction_panels('Model')
        self.assertFalse(self.controller.show_predictions)
        self.assertTrue(self.controller.show_prediction_panel)
        self.assertEqual(self.controller.rebuilt, 2)

    def test_ephys_plot_leaves_all(self):
        """Choosing an ephys or feature plot leaves the all predictions view, with no prediction"""
        for feature in (False, True):
            with self.subTest(feature=feature):
                controller = _Controller()
                controller.plot_prediction_panels(ALL_PREDICTIONS)
                self.assertTrue(controller._show_ephys_view(feature=feature))
                self.assertFalse(controller.show_predictions)
                self.assertEqual(controller.show_feature, feature)
                self.assertEqual(controller.prediction_init, NO_PREDICTION)
                controller.view.check_menu_option.assert_called_once_with(
                    'prediction', NO_PREDICTION
                )
                self.assertEqual(controller.rebuilt, 2)

    def test_ephys_view_unchanged(self):
        """Choosing a plot of the view already shown does not rebuild the display"""
        self.assertFalse(self.controller._show_ephys_view(feature=False))
        self.assertTrue(self.controller._show_ephys_view(feature=True))
        self.assertFalse(self.controller._show_ephys_view(feature=True))
        self.assertEqual(self.controller.rebuilt, 1)
        self.assertFalse(self.controller.show_ephys_plots)


if __name__ == '__main__':
    unittest.main()
