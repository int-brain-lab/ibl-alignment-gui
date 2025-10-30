import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from ibl_alignment_gui.handlers.alignment_handler import AlignmentHandler, CircularIndexTracker

class TestCircularIndexTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = CircularIndexTracker(max_idx=5)

    def test_init(self):
        self.assertEqual(self.tracker.max_idx, 5)
        self.assertEqual(self.tracker.total_idx, 0)
        self.assertEqual(self.tracker.last_idx, 0)
        self.assertEqual(self.tracker.diff_idx, 0)
        self.assertEqual(self.tracker.current_idx, 0)
        self.assertEqual(self.tracker.idx, 0)
        self.assertEqual(self.tracker.idx_prev, 0)

    def test_next_idx_to_fill(self):

        self.tracker.next_idx_to_fill()
        self.assertEqual(self.tracker.idx_prev, 0)
        self.assertEqual(self.tracker.idx, 1)
        self.assertEqual(self.tracker.current_idx, 1)
        self.assertEqual(self.tracker.total_idx, 1)

        # Add in two more moves
        for _ in range(2):
            self.tracker.next_idx_to_fill()

        self.assertEqual(self.tracker.idx_prev, 2)
        self.assertEqual(self.tracker.idx, 3)
        self.assertEqual(self.tracker.current_idx, 3)
        self.assertEqual(self.tracker.total_idx, 3)

    def test_next_idx_to_fill_full_buffer(self):

        # Add in 8 moves, so it cycles around the buffer
        for _ in range(8):
            self.tracker.next_idx_to_fill()

        self.assertEqual(self.tracker.idx_prev, 2)
        self.assertEqual(self.tracker.idx, 3)
        self.assertEqual(self.tracker.current_idx, 8)
        self.assertEqual(self.tracker.total_idx, 8)

    def test_prev_idx(self):

        # Fill the buffer with 3 moves
        for _ in range(2):
            self.tracker.next_idx_to_fill()

        # Move back one step
        moved = self.tracker.prev_idx()
        self.assertTrue(moved)
        self.assertEqual(self.tracker.current_idx, 1)
        self.assertEqual(self.tracker.idx, 1)
        self.assertEqual(self.tracker.total_idx, 2)

        # Move back 2 steps
        moved = self.tracker.prev_idx()
        self.assertTrue(moved)
        self.assertEqual(self.tracker.current_idx, 0)
        self.assertEqual(self.tracker.idx, 0)
        self.assertEqual(self.tracker.total_idx, 2)

        # Moving back 3 steps is not allowed
        moved = self.tracker.prev_idx()
        self.assertIsNone(moved)
        self.assertEqual(self.tracker.current_idx, 0)
        self.assertEqual(self.tracker.idx, 0)
        self.assertEqual(self.tracker.total_idx, 2)

        # Once we have moved back make sure the new idx to fill is correct and we reset the total index
        self.tracker.next_idx_to_fill()
        self.assertEqual(self.tracker.current_idx, 1)
        self.assertEqual(self.tracker.idx_prev, 0)
        self.assertEqual(self.tracker.idx, 1)
        self.assertEqual(self.tracker.total_idx, 1)

    def test_prev_idx_full_buffer(self):

        # Add in 7 moves, so it cycles around the buffer
        for _ in range(7):
            self.tracker.next_idx_to_fill()

        # Move back one step
        moved = self.tracker.prev_idx()
        self.assertTrue(moved)
        self.assertEqual(self.tracker.current_idx, 6)
        self.assertEqual(self.tracker.idx, 1)
        self.assertEqual(self.tracker.total_idx, 7)

        # Move back 2 steps
        moved = self.tracker.prev_idx()
        self.assertTrue(moved)
        self.assertEqual(self.tracker.current_idx, 5)
        self.assertEqual(self.tracker.idx, 0)
        self.assertEqual(self.tracker.total_idx, 7)

        # Move back 3 steps cycles around the buffer
        moved = self.tracker.prev_idx()
        self.assertTrue(moved)
        self.assertEqual(self.tracker.current_idx, 4)
        self.assertEqual(self.tracker.idx, 4)
        self.assertEqual(self.tracker.total_idx, 7)

        # Move back 4 steps cycles around the buffer
        moved = self.tracker.prev_idx()
        self.assertTrue(moved)
        self.assertEqual(self.tracker.current_idx, 3)
        self.assertEqual(self.tracker.idx, 3)
        self.assertEqual(self.tracker.total_idx, 7)

        # Moving 5 steps is not allowed as we have looped back to our original location
        moved = self.tracker.prev_idx()
        self.assertIsNone(moved)
        self.assertEqual(self.tracker.current_idx, 3)
        self.assertEqual(self.tracker.idx, 3)
        self.assertEqual(self.tracker.total_idx, 7)


    def test_next_idx(self):
        for _ in range(2):
            self.tracker.next_idx_to_fill()

        # Moving forward is not allowed
        moved = self.tracker.next_idx()
        self.assertIsNone(moved)
        self.assertEqual(self.tracker.current_idx, 2)

        # If we move back two steps and then forward this is allowed
        self.tracker.prev_idx()
        self.tracker.prev_idx()
        moved = self.tracker.next_idx()
        self.assertTrue(moved)
        self.assertEqual(self.tracker.current_idx, 1)

    def test_reset_idx(self):

        # Move three steps
        for _ in range(3):
            self.tracker.next_idx_to_fill()

        # Move back two steps
        self.tracker.prev_idx()
        self.tracker.prev_idx()
        self.assertEqual(self.tracker.idx, 1)
        self.assertEqual(self.tracker.current_idx, 1)

        # Reset the index
        self.tracker.reset_idx()
        self.assertEqual(self.tracker.total_idx, 2)
        self.assertEqual(self.tracker.idx, 2)
        self.assertEqual(self.tracker.current_idx, 2)

        # Make sure we can't move forward as we reset
        moved = self.tracker.next_idx()
        self.assertIsNone(moved)



class TestAlignmentHandler(unittest.TestCase):

    def setUp(self):

        self.mock_atlas = MagicMock()

        # Create a mock ephys align object
        self.mock_ephysalign = MagicMock()
        self.mock_ephysalign.xyz_track = np.array([[1, 2, 3]])
        self.mock_ephysalign.xyz_samples = np.array([[4, 5, 6]])
        self.mock_ephysalign.get_channel_locations.return_value = np.array([7, 8, 9])
        self.mock_ephysalign.get_perp_vector.return_value = np.array([10, 11, 12])
        self.mock_ephysalign.track_init = np.array([1, 2, 3])
        self.mock_ephysalign.feature_init = np.array([4, 5, 6])
        self.mock_ephysalign.track_extent = np.array([0, 1, 2])

        self.mock_ephysalign.get_track_and_feature.side_effect = \
            lambda: (self.mock_ephysalign.feature_init, self.mock_ephysalign.track_init, None)
        self.mock_ephysalign.region_colour = "red"
        self.mock_ephysalign.scale_histology_regions.return_value = (np.array([1]), np.array([2]))
        self.mock_ephysalign.get_scale_factor.return_value = (np.array([1]), np.array([1]))
        self.mock_ephysalign.adjust_extremes_linear.return_value = (np.array([1]), np.array([1]))
        self.mock_ephysalign.adjust_extremes_uniform.return_value = np.array([1])


        ephys_align_patcher =  patch('ibl_alignment_gui.handlers.alignment_handler.EphysAlignment',
                                     return_value=self.mock_ephysalign)
        ephys_align_patcher.start()
        self.addCleanup(ephys_align_patcher.stop)

        # Mock AllenAtlas
        self.mock_AllenAtlas = MagicMock()

        self.align_handler = AlignmentHandler(
            xyz_picks=np.array([[0, 0, 0]]),
            chn_depths=np.array([0, 1, 2]),
            brain_atlas=self.mock_atlas
        )

    def test_init(self):

        self.assertEqual(len(self.align_handler.tracks), self.align_handler.buffer.max_idx + 1)
        self.assertEqual(len(self.align_handler.features), self.align_handler.buffer.max_idx + 1)

    def test_ephys_align_properties(self):

        np.testing.assert_array_equal(self.align_handler.xyz_track, self.mock_ephysalign.xyz_track)
        np.testing.assert_array_equal(self.align_handler.xyz_samples, self.mock_ephysalign.xyz_samples)
        np.testing.assert_array_equal(self.align_handler.xyz_channels, np.array([7, 8, 9]))
        np.testing.assert_array_equal(self.align_handler.track_lines, np.array([10, 11, 12]))

    def test_buffer_properties_and_methods(self):

        self.assertEqual(self.align_handler.current_idx, 0)
        self.assertEqual(self.align_handler.total_idx, 0)
        self.assertEqual(self.align_handler.idx_prev, 0)
        self.assertEqual(self.align_handler.idx, 0)

        self.assertIsNone(self.align_handler.next_idx())
        self.assertIsNone(self.align_handler.prev_idx())


    def test_set_init_feature_track(self):

        with self.subTest('No init feature'):
            self.align_handler.set_init_feature_track()
            idx = self.align_handler.idx
            np.testing.assert_array_equal(self.align_handler.features[idx], self.mock_ephysalign.feature_init)
            np.testing.assert_array_equal(self.align_handler.tracks[idx], self.mock_ephysalign.track_init)
            # Test the feature and track methods work
            np.testing.assert_array_equal(self.align_handler.feature, self.align_handler.features[idx])
            np.testing.assert_array_equal(self.align_handler.track, self.align_handler.tracks[idx])

        with self.subTest('Init feature'):
            feature = np.array([10, 20, 30])
            track = np.array([40, 50, 60])
            self.align_handler.set_init_feature_track(feature, track)
            np.testing.assert_array_equal(self.align_handler.feature, feature)
            np.testing.assert_array_equal(self.align_handler.track, track)

    def test_reset_features_and_tracks(self):
        self.align_handler.reset_features_and_tracks()
        idx = self.align_handler.idx
        np.testing.assert_array_equal(self.align_handler.features[idx], self.mock_ephysalign.feature_init)
        np.testing.assert_array_equal(self.align_handler.tracks[idx], self.mock_ephysalign.track_init)

    def test_get_scaled_histology(self):
        hist_data, hist_data_ref, scale_data = self.align_handler.get_scaled_histology()
        self.assertIn('region', hist_data)
        self.assertIn('axis_label', hist_data)
        self.assertIn('colour', hist_data)
        self.assertIn('region', scale_data)
        self.assertIn('scale', scale_data)
        self.assertIn('region', hist_data_ref)
        self.assertIn('axis_label', hist_data_ref)

    def test_offset_hist_dat(self):
        self.align_handler.tracks[self.align_handler.idx] = np.array([1, 2, 3])
        self.align_handler.features[self.align_handler.idx] = np.array([4, 5, 6])
        self.align_handler.offset_hist_data(offset=5)
        idx = self.align_handler.idx
        prev_idx = self.align_handler.idx_prev
        np.testing.assert_array_equal(self.align_handler.tracks[idx], self.align_handler.tracks[prev_idx] + 5)
        np.testing.assert_array_equal(self.align_handler.features[idx], self.align_handler.features[prev_idx])

    def test_scale_hist_data_calls(self):

        self.align_handler.tracks[self.align_handler.idx] = np.array([1, 2, 3])
        self.align_handler.features[self.align_handler.idx] = np.array([4, 5, 6])
        self.align_handler.tracks[self.align_handler.idx_prev] = np.array([1, 2, 3])
        self.align_handler.features[self.align_handler.idx_prev] = np.array([4, 5, 6])

        with self.subTest('Less than 5 reference lines'):
            self.align_handler.scale_hist_data(
                line_track=np.array([1, 2]),
                line_feature=np.array([3, 4]),
                extend_feature=1,
                lin_fit=True
            )
            self.mock_ephysalign.feature2track.assert_called()
            self.mock_ephysalign.adjust_extremes_uniform.assert_called()
            self.mock_ephysalign.adjust_extremes_linear.assert_not_called()

        with self.subTest('Greater than 4 reference lines'):
            self.mock_ephysalign.reset_mock()
            self.align_handler.scale_hist_data(
                line_track=np.array([1, 2, 7, 8]),
                line_feature=np.array([3, 4, 9, 10]),
                extend_feature=1,
                lin_fit=True
            )
            self.mock_ephysalign.feature2track.assert_called()
            self.mock_ephysalign.adjust_extremes_uniform.assert_not_called()
            self.mock_ephysalign.adjust_extremes_linear.assert_called()
