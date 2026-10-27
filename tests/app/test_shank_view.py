"""Tests for the shank view's reference lines and histology panels, built offscreen."""

import os
import unittest

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

from ibl_alignment_gui.app.views.shank_view import ShankView
from ibl_alignment_gui.app.widgets import custom_widgets
from iblutil.util import Bunch


def _setup_qt(cls) -> None:
    # Run without a display server unless one has been chosen explicitly.
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _regions() -> Bunch:
    """Two regions along the probe, as plot_histology takes them."""
    return Bunch(
        region=np.array([[0.0, 100.0], [100.0, 200.0]]),
        axis_label=np.array([[50.0, 'A'], [150.0, 'B']], dtype=object),
        colour=np.array([[255, 0, 0], [0, 0, 255]]),
    )


class TestReferenceLines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_qt(cls)

    def setUp(self):
        self.view = ShankView('a', 0, 'default')
        self.view.create_reference_line_and_point(100)

    def test_one_line_per_ephys_figure(self):
        """A reference line is drawn on every ephys figure"""
        lines = self.view.lines_features[0]
        self.assertEqual(len(lines), len(self.view.feature_figs))
        for fig, line in zip(self.view.feature_figs, lines, strict=True):
            self.assertIn(line, fig.items)

    def test_moved_line_moves_the_others(self):
        """Moving the line on one figure moves it on all the others"""
        moved = self.view.lines_features[0][-1]
        moved.setPos(250)
        line_idx, fig_idx = self.view.match_feature_line(moved)
        self.assertEqual(line_idx, 0)
        self.assertEqual(len(fig_idx), len(self.view.feature_figs) - 1)
        self.view.update_feature_reference_line_and_point(moved, line_idx, fig_idx)
        for line in self.view.lines_features[0]:
            self.assertEqual(line.value(), 250)

    def test_remove_and_add_back(self):
        """The lines are taken off and put back on every ephys figure"""
        self.view.remove_reference_lines_from_display()
        for fig, line in zip(self.view.feature_figs, self.view.lines_features[0], strict=True):
            self.assertNotIn(line, fig.items)
        self.view.add_reference_lines_to_display()
        for fig, line in zip(self.view.feature_figs, self.view.lines_features[0], strict=True):
            self.assertIn(line, fig.items)

    def test_remove_one_line(self):
        """Removing a line takes it off every figure, the histology too"""
        self.view.remove_reference_line(0)
        for fig, line in zip(self.view.feature_figs, self.view.lines_features[0], strict=True):
            self.assertNotIn(line, fig.items)
        self.assertNotIn(self.view.lines_tracks[0], self.view.fig_hist.items)


class TestHistologyRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_qt(cls)

    def test_reference_keeps_histology_regions(self):
        """Drawing the reference histology keeps the hover regions of the main histology"""
        view = ShankView('a', 0, 'default')
        view.plot_histology(view.fig_hist, _regions())
        regions = list(view.hist_regions)
        view.plot_histology(view.fig_hist_ref, _regions(), ax='right')
        self.assertEqual(view.hist_regions, regions)


def _cumulative() -> Bunch:
    """Cumulative probabilities of two classes along the probe, as plot_prediction takes them."""
    return Bunch(
        depths=np.array([0.0, 100.0, 200.0]),
        colours=[(255, 0, 0), (0, 0, 255)],
        probability=np.array([[0.2, 1.0], [0.5, 1.0], [0.9, 1.0]]),
    )


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_qt(cls)

    def setUp(self):
        self.view = ShankView('a', 0, 'default')
        self.view.create_reference_line_and_point(100)
        self.fig = self.view.fig_pred
        self.line = self.view.lines_features[0][self.view.feature_figs.index(self.fig)]

    def test_regions(self):
        """Regions are drawn with their labels and the title, keeping the reference line"""
        self.view.plot_prediction(self.fig, _regions(), 'Model')
        regions = [i for i in self.fig.items if isinstance(i, pg.LinearRegionItem)]
        self.assertEqual([r.region_name for r in regions], ['A', 'B'])
        self.assertEqual(self.view.pred_titles[self.fig], 'Model')
        self.assertIn(self.line, self.fig.items)

    def test_cumulative(self):
        """Cumulative probabilities are drawn as one band per class"""
        self.view.plot_prediction(self.fig, _cumulative(), 'Model (cumulative)')
        bands = [i for i in self.fig.items if isinstance(i, pg.FillBetweenItem)]
        self.assertEqual(len(bands), 2)

    def test_redraw_replaces(self):
        """Drawing another prediction removes the previous one, but not the reference line"""
        self.view.plot_prediction(self.fig, _regions(), 'Model')
        self.view.plot_prediction(self.fig, _cumulative(), 'Other')
        self.assertFalse([i for i in self.fig.items if isinstance(i, pg.LinearRegionItem)])
        self.assertIn(self.line, self.fig.items)

    def test_clear_with_message(self):
        """Clearing leaves only the message, the title and the reference line"""
        self.view.plot_prediction(self.fig, _regions(), 'Model')
        self.view.clear_prediction(self.fig, 'Model', 'Not loaded')
        texts = [i for i in self.fig.items if isinstance(i, pg.TextItem)]
        self.assertEqual([t.toPlainText() for t in texts], ['Not loaded'])
        self.assertFalse([i for i in self.fig.items if isinstance(i, pg.LinearRegionItem)])
        self.assertIn(self.line, self.fig.items)

    def test_hover_title(self):
        """Hovering a region adds its name to the title, until the title is reset"""
        self.view.plot_prediction(self.fig, _regions(), 'Model')
        region = next(i for i in self.fig.items if isinstance(i, pg.LinearRegionItem))
        label = self.view.pred_labels[self.fig].getAxis('top')
        self.view.show_prediction_region(self.fig, region)
        self.assertEqual(label.labelText, 'Model: A')
        self.view.reset_prediction_title()
        self.assertEqual(label.labelText, 'Model')

    def test_long_title_wrapped(self):
        """A long title is wrapped over several lines"""
        self.view.plot_prediction(self.fig, _regions(), 'XGBoost region prediction (cumulative)')
        label = self.view.pred_labels[self.fig].getAxis('top')
        self.assertIn('<br>', label.labelText)


class TestLayouts(unittest.TestCase):
    """The prediction figures are in the ephys layouts only when asked for."""

    @classmethod
    def setUpClass(cls):
        _setup_qt(cls)

    def in_layout(self, widget, fig) -> bool:
        layout = widget.ephys_area.ci.items
        return any(fig in item.items for item in layout if hasattr(item, 'items'))

    def test_single(self):
        for cls in (custom_widgets.SingleConfigWidget, custom_widgets.SingleConfigFeatureWidget):
            for prediction in (False, True):
                with self.subTest(cls=cls.__name__, prediction=prediction):
                    view = ShankView('a', 0, 'default')
                    widget = cls(view, prediction=prediction)
                    self.assertEqual(self.in_layout(widget, view.fig_pred), prediction)

    def test_dual(self):
        for cls in (custom_widgets.DualConfigWidget, custom_widgets.DualConfigFeatureWidget):
            for prediction in (False, True):
                with self.subTest(cls=cls.__name__, prediction=prediction):
                    views = ShankView('a', 0, 'default'), ShankView('a', 0, 'other')
                    widget = cls(*views, prediction=prediction)
                    for view in views:
                        self.assertEqual(self.in_layout(widget, view.fig_pred), prediction)


if __name__ == '__main__':
    unittest.main()
