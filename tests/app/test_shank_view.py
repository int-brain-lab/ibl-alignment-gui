"""Tests for the shank view's reference lines and histology panels, built offscreen."""

import os
import unittest

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.app.views.shank_view import ShankView
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


if __name__ == '__main__':
    unittest.main()
