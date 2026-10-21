"""Tests for the helpers shared by the Channel Prediction plugin and its model backends."""

import unittest

import numpy as np

from ibl_alignment_gui.plugins.ephys_atlas._common import get_region_boundaries
from iblutil.util import Bunch

RED, BLUE = [255, 0, 0], [0, 0, 255]


class TestGetRegionBoundaries(unittest.TestCase):
    def test_regions_along_track(self):
        """One segment per run of the same region, with its label and colour, in microns"""
        regions = Bunch(
            id=np.array([1, 1, 2, 2, 2]),
            acronym=np.array(['A', 'A', 'B', 'B', 'B']),
            rgb=np.array([RED] * 2 + [BLUE] * 3),
        )
        depths_m = np.array([0.0, 10.0, 20.0, 30.0, 40.0]) / 1e6
        view = get_region_boundaries(regions, depths_m)
        np.testing.assert_allclose(view.region, [[0, 10], [20, 40]])
        # Depths go from metres back to microns, so positions are compared with a tolerance.
        np.testing.assert_allclose(view.axis_label[:, 0].astype(float), [5, 30])
        self.assertEqual(view.axis_label[:, 1].tolist(), ['A', 'B'])
        np.testing.assert_array_equal(view.colour, [RED, BLUE])

    def test_single_region(self):
        """A track within one region is one segment"""
        regions = Bunch(
            id=np.array([3, 3, 3]), acronym=np.array(['C'] * 3), rgb=np.array([BLUE] * 3)
        )
        view = get_region_boundaries(regions, np.array([0.0, 10.0, 20.0]) / 1e6)
        np.testing.assert_allclose(view.region, [[0, 20]])
        self.assertEqual(view.axis_label[:, 1].tolist(), ['C'])


if __name__ == '__main__':
    unittest.main()
