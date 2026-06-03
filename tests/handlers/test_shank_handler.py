import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ibl_alignment_gui.handlers.shank_handler import ShankHandler
from iblutil.util import Bunch


class TestShankHandler(unittest.TestCase):
    """Test the ShankHandler class."""

    def setUp(self):
        # Mock alignment loader
        self.mock_align = MagicMock()
        self.mock_align.load_previous_alignments.return_value = None
        self.mock_align.get_starting_alignment.return_value = None
        self.mock_align.feature_prev = np.array([1, 2, 3])
        self.mock_align.track_prev = np.array([10, 20, 30])
        self.mock_align.xyz_picks = np.array([[0, 0, 0], [1, 1, 1]])

        # Mock data loader
        self.mock_data = MagicMock()
        self.mock_data.get_data.return_value = {
            'clusters': {'channels': np.array([1, 0]), 'exists': True},
        }
        self.mock_data.probe_collection = 'test_collection'
        self.mock_data.probe_path = 'test_path'

        self.mock_geom = MagicMock()
        self.mock_geom.get_geometry.return_value = None
        self.mock_geom.get_sites_for_shank.return_value = {
            'sites_coords': np.array([[0, 100], [0, 200]])
        }

        # Mock ephys loader
        self.mock_ephys = MagicMock()
        self.mock_ephys.load_ap_snippets.return_value = np.array([1, 2, 3])

        # Mock plot loader
        self.mock_plots = MagicMock()
        self.mock_plots.chn_min = -50
        self.mock_plots.chn_max = 500
        self.mock_plots.cluster_idx = [0, 1]
        self.mock_plots.slice_plots = Bunch(CCF=Bunch())
        self.mock_plots.image_plots = Bunch(AP=Bunch(), LFP=Bunch())
        self.mock_plots.scatter_plots = Bunch(clusters=Bunch())
        self.mock_plots.probe_plots = Bunch(rfmap=Bunch())
        self.mock_plots.line_plots = Bunch(firing_rate=Bunch())

        # Mock uploader
        self.mock_upload = MagicMock()
        self.mock_upload.brain_atlas = MagicMock()
        self.mock_upload.upload_data.return_value = 'uploaded'

        # Mock histology loader
        self.mock_hist = MagicMock()
        self.mock_hist.get_slices.return_value = Bunch()

        # Bunch with all loaders
        self.loaders = Bunch(
            align=self.mock_align,
            data=self.mock_data,
            geom=self.mock_geom,
            ephys=self.mock_ephys,
            plots=self.mock_plots,
            upload=self.mock_upload,
            hist=self.mock_hist,
        )

        # Instantiate ShankHandler
        self.shank_handler = ShankHandler(self.loaders, shank_idx=0)

        # Make a mock alignment handler
        self.mock_align_handle = MagicMock()
        self.mock_align_handle.offset_hist_data.return_value = None
        self.mock_align_handle.scale_hist_data.return_value = None
        self.mock_align_handle.get_scaled_histology.return_value = ('hist', 'hist_ref', 'scale')
        self.mock_align_handle.ephysalign.feature2track_lin.return_value = np.array([100, 200])
        self.mock_align_handle.track = np.array([10, 20, 30])
        self.mock_align_handle.feature = np.array([1, 2, 3])
        self.mock_align_handle.xyz_channels = np.array([[0, 0, 0], [1, 1, 1]])
        self.mock_align_handle.xyz_track = np.array([[0, 0, 0], [2, 2, 2]])
        self.mock_align_handle.track_lines = [np.array([1, 1, 1])]
        self.shank_handler.align_handle = self.mock_align_handle

    def test_set_init_alignment(self):
        """Test the set_init_alignment method."""
        self.shank_handler.set_init_alignment()
        self.mock_align_handle.set_init_feature_track.assert_called_with(
            self.mock_align.feature_prev, self.mock_align.track_prev
        )
        self.mock_align_handle.set_init_feature_track.assert_called_once()

    def test_feature_prev(self):
        """Test the feature_prev property."""
        np.testing.assert_array_equal(
            self.shank_handler.feature_prev, self.mock_align.feature_prev
        )

    def test_offset_hist_data(self):
        """Test the offset_hist_data method."""
        self.shank_handler.offset_hist_data(10)
        self.mock_align_handle.offset_hist_data.assert_called_with(10)
        self.mock_align_handle.offset_hist_data.assert_called_once()

    def test_scale_hist_data(self):
        """Test the scale_hist_data method."""
        self.shank_handler.scale_hist_data(10)
        self.mock_align_handle.scale_hist_data.assert_called_with(10)
        self.mock_align_handle.scale_hist_data.assert_called_once()

    def test_get_scaled_histology(self):
        """Test the get_scaled_histology method."""
        self.shank_handler.get_scaled_histology()
        self.mock_align_handle.get_scaled_histology.assert_called_once()
        self.assertEqual(self.shank_handler.hist_data, 'hist')
        self.assertEqual(self.shank_handler.hist_data_ref, 'hist_ref')
        self.assertEqual(self.shank_handler.scale_data, 'scale')

    def test_feature2track_lin(self):
        """Test the feature2track_lin method."""
        result = self.shank_handler.feature2track_lin(np.array([10]), np.array([1]), np.array([2]))
        self.mock_align_handle.ephysalign.feature2track_lin.assert_called_with(
            np.array([10]), np.array([1]), np.array([2])
        )
        np.testing.assert_array_equal(result, np.array([100, 200]))

    def test_reset_features_and_tracks(self):
        """Test the reset_features_and_tracks method."""
        self.shank_handler.reset_features_and_tracks()
        self.mock_align_handle.reset_features_and_tracks.assert_called_once()

    def test_get_align_handle_properties(self):
        """Test the property methods accessing align_handle."""
        np.testing.assert_array_equal(self.shank_handler.track, self.mock_align_handle.track)
        np.testing.assert_array_equal(self.shank_handler.feature, self.mock_align_handle.feature)
        np.testing.assert_array_equal(
            self.shank_handler.xyz_channels, self.mock_align_handle.xyz_channels
        )
        np.testing.assert_array_equal(
            self.shank_handler.xyz_track, self.mock_align_handle.xyz_track
        )
        np.testing.assert_array_equal(
            self.shank_handler.track_lines, self.mock_align_handle.track_lines
        )

    def test_get_plot_loader_properties(self):
        """Test the property methods accessing loaders['plots']."""
        self.assertEqual(self.shank_handler.chn_min, self.mock_plots.chn_min)
        self.assertEqual(self.shank_handler.chn_max, self.mock_plots.chn_max)
        self.assertEqual(self.shank_handler.slice_plots, self.mock_plots.slice_plots)
        self.assertEqual(self.shank_handler.image_plots, self.mock_plots.image_plots)
        self.assertEqual(self.shank_handler.scatter_plots, self.mock_plots.scatter_plots)
        self.assertEqual(self.shank_handler.line_plots, self.mock_plots.line_plots)
        self.assertEqual(self.shank_handler.probe_plots, self.mock_plots.probe_plots)

    def test_xyz_clusters(self):
        """Test the xyz_clusters method."""
        self.shank_handler.raw_data = {'clusters': {'channels': np.array([1, 0]), 'exists': True}}
        np.testing.assert_array_equal(
            self.shank_handler.xyz_clusters, np.array([[1, 1, 1], [0, 0, 0]])
        )

    @patch('ibl_alignment_gui.handlers.shank_handler.AlignmentHandler')
    def test_load_data(self, mock_align_handle):
        """Test the load_data method."""
        with self.subTest('Data loaded for the first time'):
            self.shank_handler.data_loaded = False
            self.shank_handler.load_data()
            self.assertTrue(self.shank_handler.data_loaded)
            self.assertTrue(self.shank_handler.align_exists)
            self.mock_geom.get_geometry.assert_called_once()
            mock_align_handle.assert_called_once()
            np.testing.assert_array_equal(self.shank_handler.cluster_chns, np.array([1, 0]))

        with self.subTest('Data already loaded'):
            self.shank_handler.data_loaded = True
            self.mock_geom.get_geometry.reset_mock()
            self.shank_handler.load_data()
            self.assertTrue(self.shank_handler.align_exists)
            self.mock_geom.get_geometry.assert_not_called()

        with self.subTest('No xyz_picks'):
            mock_align_handle.reset_mock()
            self.mock_align.xyz_picks = None
            self.shank_handler.data_loaded = False
            self.shank_handler.load_data()
            self.assertFalse(self.shank_handler.align_exists)
            self.assertEqual(self.mock_plots.slice_plots, Bunch())
            self.assertTrue(self.shank_handler.data_loaded)
            mock_align_handle.assert_not_called()

        with self.subTest('No raw data'):
            self.mock_data.get_data.return_value = {
                'clusters': {'exists': False},
            }
            self.shank_handler.raw_data = {'clusters': {'exists': False}}
            self.shank_handler.data_loaded = False
            self.shank_handler.load_data()
            np.testing.assert_array_equal(self.shank_handler.cluster_chns, np.array([0, 1]))

    def test_filter_units(self):
        """Test the filter_units method."""
        self.mock_plots.reset_mock()
        self.shank_handler.filter_units('ibl_good')
        self.mock_plots.filter_units.assert_called_with('ibl_good')
        self.mock_plots.compute_rasters.assert_called_once()
        self.mock_plots.get_plots.assert_called_once()

    @patch('ibl_alignment_gui.handlers.shank_handler.AlignmentHandler')
    def test_upload_data(self, mock_align_handle):
        """Test the upload_data method."""
        self.shank_handler.load_data()
        result = self.shank_handler.upload_data()
        self.assertEqual(result, 'uploaded')
