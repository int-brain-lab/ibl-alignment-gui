import datetime
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ibl_alignment_gui.loaders.alignment_uploader import (
    AlignmentUploaderLocal,
    AlignmentUploaderOne,
)
from iblutil.util import Bunch


class TestAlignmentUploaderOne(unittest.TestCase):
    """ Test the AlignmentUploaderOne class """

    def setUp(self):

        self.mock_one = MagicMock()
        self.mock_one.alyx.rest.return_value = [{'id': uuid.uuid4()}]
        self.insertion = {'id': uuid.uuid4(), 'json': {'extended_qc': {}}, 'name': 'probe00'}
        self.mock_atlas = MagicMock()
        self.uploader = AlignmentUploaderOne(self.insertion, self.mock_one, self.mock_atlas)

    def test_init(self):
        """ Test the init method """
        with self.subTest('Not resolved'):
            insertion = {'id': uuid.uuid4(), 'json': {'extended_qc': {}}, 'name': 'probe00'}
            uploader = AlignmentUploaderOne(insertion, self.mock_one, self.mock_atlas)
            self.assertFalse(uploader.resolved)
            self.assertEqual(uploader.pname, 'probe00')

        with self.subTest('Not resolved'):
            insertion = {'id': uuid.uuid4(), 'json': {'extended_qc': {'alignment_resolved': True}}, 'name': 'probe00'}
            uploader = AlignmentUploaderOne(insertion, self.mock_one, self.mock_atlas)
            self.assertTrue(uploader.resolved)
            self.assertEqual(uploader.pname, 'probe00')

    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderOne.upload_channels')
    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderOne.upload_alignments')
    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderOne.upload_qc')
    def test_upload_data(self, mock_qc, mock_alignments, mock_channels):
        """ Test the upload_data method """
        mock_qc.return_value = False
        mock_channels.return_value = True

        info = self.uploader.upload_data(Bunch())
        mock_channels.assert_called_once()
        mock_alignments.assert_called_once()
        mock_qc.assert_called_once()
        self.assertIn('Alignment not resolved', info)

    def test_get_upload_info(self):
        """ Test the get_upload_info method """
        with self.subTest('Not resolved'):
            self.assertIn('Alignment not resolved',
                          self.uploader.get_upload_info(True, False))

        with self.subTest('Newly resolved'):
            self.assertIn('Alignment resolved',
                          self.uploader.get_upload_info(True, True))

        with self.subTest('Already resolved'):
            self.assertIn('already been resolved',
                          self.uploader.get_upload_info(False, True))

        with self.subTest('No changes'):
            self.assertIn('No changes made',
                          self.uploader.get_upload_info(False, False))

    @patch('ibl_alignment_gui.loaders.alignment_uploader.histology.register_aligned_track')
    def test_upload_channels(self, mock_register):
        """ Test the upload_channels method """
        data = {
            'xyz_channels': np.arange(5),
            'chn_coords': np.arange(5),
        }

        with self.subTest('Upload channels with resolved=False'):
            self.uploader.resolved = False
            is_channels = self.uploader.upload_channels(data)
            self.assertTrue(is_channels)
            mock_register.assert_called_once()

        with self.subTest('Upload channels with resolved=True'):
            mock_register.reset_mock()
            self.uploader.resolved = True
            is_channels = self.uploader.upload_channels(data)
            self.assertFalse(is_channels)
            mock_register.assert_not_called()

    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderOne.save_alignments')
    @patch('ibl_alignment_gui.loaders.alignment_uploader.datetime')
    def test_upload_alignments(self, mock_datetime, mock_save_alignments):
        """ Test the upload_alignments method """
        mock_datetime.now.return_value = datetime.datetime(2025, 8, 4)

        self.uploader.user = 'user1'
        self.uploader.qc_str = 'PASS'
        self.uploader.confidence_str = 'Medium'
        feature = [-0.1, 0, 0.1]
        track = [-0.3, 0, 0.2]
        expected_time = mock_datetime.now.return_value.replace(second=0, microsecond=0).isoformat()
        new_alignment = {f'{expected_time}_{self.uploader.user}': [feature, track, self.uploader.qc_str,
                                                                   self.uploader.confidence_str]}

        with self.subTest('Upload alignments no previous alignment'):

            data = {'feature': feature, 'track': track, 'alignments': {}}
            alignments = self.uploader.upload_alignments(data)
            self.assertEqual(alignments, new_alignment)
            mock_datetime.now.assert_called_once()
            mock_save_alignments.assert_called_once()

        with self.subTest('Upload alignments with previous alignment'):
            data = {'feature': feature, 'track': track, 'alignments': {'2024-04-02_test': [[0], [1], 'WARNING']}}
            alignments = self.uploader.upload_alignments(data)
            self.assertIn('2024-04-02_test', alignments)
            self.assertIn(f'{expected_time}_{self.uploader.user}', alignments)


    def test_remove_duplicate_users(self):
        """ Test the _remove_duplicate_users method """
        self.uploader.user = 'user1'
        alignments = {'2025-02-05_user1': [], '2024-04-02_user2': []}

        with self.subTest('Duplicate user with resolved=False'):
            self.uploader.resolved = False
            updated_alignments = self.uploader._remove_duplicate_users(alignments.copy())
            self.assertNotIn('2025-02-05_user1', updated_alignments)
            self.assertIn('2024-04-02_user2', updated_alignments)

        with self.subTest('Duplicate user with resolved=True'):
            self.uploader.resolved = True
            updated_alignments = self.uploader._remove_duplicate_users(alignments.copy())
            self.assertIn('2025-02-05_user1', updated_alignments)
            self.assertIn('2024-04-02_user2', updated_alignments)

    def test_save_alignments(self):
        """ Test the save_alignments method """
        self.uploader.save_alignments({})
        self.assertEqual(self.mock_one.alyx.rest.call_count, 2)


    @patch('ibl_alignment_gui.loaders.alignment_uploader.critical_note.main_gui')
    def test_get_user_qc(self, mock_note):
        """ Test the get_user_qc method """
        with self.subTest('QC pass no reasons'):
            self.uploader.get_user_qc('High', 'Pass', [], False)
            self.assertEqual(self.uploader.qc_str, 'PASS: None')
            self.assertEqual(self.uploader.confidence_str, 'Confidence: High')
            self.assertFalse(self.uploader.force_resolve)
            mock_note.assert_not_called()

        with self.subTest('QC warning with reasons'):
            self.uploader.get_user_qc('Medium', 'Warning', ['reason1', 'reason2'], False)
            self.assertEqual(self.uploader.qc_str, 'WARNING: reason1, reason2')
            self.assertEqual(self.uploader.confidence_str, 'Confidence: Medium')
            self.assertFalse(self.uploader.force_resolve)
            mock_note.assert_not_called()

        with self.subTest('QC critical with reasons'):
            self.uploader.get_user_qc('Low', 'Critical', ['critical1'], False)
            self.assertEqual(self.uploader.qc_str, 'CRITICAL: critical1')
            self.assertEqual(self.uploader.confidence_str, 'Confidence: Low')
            self.assertFalse(self.uploader.force_resolve)
            mock_note.assert_called_once()

        with self.subTest('Force resolve'):
            self.uploader.get_user_qc('High', 'Pass', [], True)
            self.assertTrue(self.uploader.force_resolve)

    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentQC')
    def test_upload_qc(self, mock_align_qc_class):
        """ Test the upload_qc method """
        mock_align_qc = mock_align_qc_class.return_value

        alignments = {'2025-02-05_user1': [], '2024-04-02_user2': []}
        data = {'probe_collection': 'alf/probe00', 'xyz_picks': [], 'chn_depths': [],
                'cluster_chns': [], 'chn_coords': []}

        with self.subTest('Alignments not resolved, force=False'):
            self.uploader.force_resolve = False
            mock_align_qc.run.return_value = {'alignment_resolved': False}
            resolved = self.uploader.upload_qc(data, alignments)
            self.assertFalse(resolved)
            mock_align_qc.run.assert_called_once()
            mock_align_qc.resolve_manual.assert_not_called()

        with self.subTest('Alignments not resolved, force=True'):
            mock_align_qc.reset_mock()
            self.uploader.force_resolve = True
            mock_align_qc.run.return_value = {'alignment_resolved': False}
            resolved = self.uploader.upload_qc(data, alignments)
            self.assertTrue(resolved)
            mock_align_qc.run.assert_not_called()
            mock_align_qc.resolve_manual.assert_called_once()



class TestAlignmentUploaderLocal(unittest.TestCase):
    """ Test the AlignmentUploaderLocal class """

    def setUp(self):
        self.mock_atlas = MagicMock()
        self.mock_atlas.regions.get.return_value = {
            'id': np.array([1, 2]),
            'acronym': np.array(['VISp', 'MOp'])
        }
        self.mock_atlas.get_lables.return_value = np.array([1, 2])

        #
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.mock_data = {
            'xyz_channels': np.array([[0.001, 0.002, 0.003],
                                      [0.004, 0.005, 0.006]]),
            'chn_coords': np.array([[10, 20],
                                    [30, 40]]),
            'feature': [-0.1, 0, 0.1],
            'track': [-0.3, 0, 0.2],
            'alignments': {}
        }

        self.mock_sites = Bunch({
            'orig_idx': np.array([10, 22])
        })

        self.uploader = AlignmentUploaderLocal(self.temp_path, 0, 1, self.mock_atlas)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderLocal.upload_channels')
    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderLocal.upload_alignments')
    def test_upload_data(self, mock_upload_alignments, mock_upload_channels):
        """ Test the upload_data method """
        info = self.uploader.upload_data(self.mock_data, self.mock_sites)
        mock_upload_channels.assert_called_once()
        mock_upload_alignments.assert_called_once()

        self.assertEqual(info, 'Channels locations saved')

    def test_get_brain_regions(self):
        """ Test the get_brain_regions method """
        regions = self.uploader.get_brain_regions(self.mock_data)
        for key in ['id', 'acronym', 'xyz', 'lateral', 'axial']:
            self.assertIn(key, regions)
            self.assertEqual(len(regions[key]), 2)

    def test_get_channels(self):
        """ Test the get_channels method """
        brain_regions = Bunch({
            'id': np.array([1, 2]),
            'acronym': np.array(['VISp', 'MOp']),
            'xyz': np.array([[0.001, 0.002, 0.003],
                             [0.004, 0.005, 0.006]]),
            'axial': np.array([20, 40]),
            'lateral': np.array([10, 30])
        })
        channels = self.uploader.get_channels(brain_regions)
        self.assertIn('channel_0', channels)
        self.assertIn('channel_1', channels)
        self.assertIn('origin', channels)
        # Test the first channel makes sense
        self.assertEqual(channels['channel_0']['x'], 1000)
        self.assertEqual(channels['channel_0']['y'], 2000)
        self.assertEqual(channels['channel_0']['z'], 3000)
        self.assertEqual(channels['channel_0']['axial'], 20)
        self.assertEqual(channels['channel_0']['lateral'], 10)
        self.assertEqual(channels['channel_0']['brain_region_id'], 1)
        self.assertEqual(channels['channel_0']['brain_region'], 'VISp')
        # Since orig_idx by default is None, make sure this key does not exist
        self.assertNotIn('original_channel_idx', channels['channel_0'])

        # If orig_idx is set then check that the value is set correctly
        self.uploader.orig_idx = self.mock_sites['orig_idx']
        channels = self.uploader.get_channels(brain_regions)
        self.assertEqual(channels['channel_0']['original_channel_idx'], 10)


    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderLocal.save_alignments')
    @patch('ibl_alignment_gui.loaders.alignment_uploader.datetime')
    def test_upload_alignments(self, mock_datetime, mock_save_alignments):
        """ Test the upload_alignments method """
        mock_datetime.now.return_value = datetime.datetime(2025, 8, 4)

        user = 'user1'
        expected_time = mock_datetime.now.return_value.replace(second=0, microsecond=0).isoformat()
        new_alignment = {f'{expected_time}': [self.mock_data['feature'], self.mock_data['track']]}
        new_alignment_with_user = {f'{expected_time}_{user}': [self.mock_data['feature'], self.mock_data['track']]}

        with self.subTest('Upload alignments no previous alignment'):

            self.uploader.upload_alignments(self.mock_data)
            alignments = mock_save_alignments.call_args[0][0]
            self.assertEqual(alignments, new_alignment)
            mock_datetime.now.assert_called_once()
            mock_save_alignments.assert_called_once()

        with self.subTest('Upload alignments with previous alignment'):
            self.mock_data['alignments'] = {'2024-04-02_test': [[0], [1]]}
            self.uploader.upload_alignments(self.mock_data)
            alignments = mock_save_alignments.call_args[0][0]
            self.assertIn('2024-04-02_test', alignments)
            self.assertIn(f'{expected_time}', alignments)

        with self.subTest('Upload alignments with user'):
            self.mock_data['alignments'] = {}
            self.uploader.user = user
            self.uploader.upload_alignments(self.mock_data)
            alignments = mock_save_alignments.call_args[0][0]
            self.assertEqual(alignments, new_alignment_with_user)

    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderLocal.get_brain_regions')
    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderLocal.get_channels')
    @patch('ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderLocal.save_channels')
    def test_upload_channels(self, mock_save, mock_channels, mock_regions):
        """ Test the upload_channels method """
        self.uploader.upload_channels(self.mock_data)
        mock_save.assert_called_once()
        mock_channels.assert_called_once()
        mock_regions.assert_called_once()

    def test_save_alignments(self):
        """ Test the save_alignments method """
        with self.subTest('Single shank data'):
            self.uploader.n_shanks = 1
            self.uploader.shank_idx = 0
            self.uploader.save_alignments({'2024-04-02_test': [[0], [1]]})
            self.assertTrue(self.temp_path.joinpath('prev_alignments.json').exists())

        with self.subTest('Multi shank data'):
            self.uploader.n_shanks = 3
            self.uploader.shank_idx = 1
            self.uploader.save_alignments({'2024-04-02_test': [[0], [1]]})
            self.assertTrue(self.temp_path.joinpath('prev_alignments_shank2.json').exists())

    def test_save_channels(self):
        """ Test the save_channels method """
        with self.subTest('Single shank data'):
            self.uploader.n_shanks = 1
            self.uploader.shank_idx = 0
            self.uploader.save_channels({'channel_0': {'x': 0}})
            self.assertTrue(self.temp_path.joinpath('channel_locations.json').exists())

        with self.subTest('Multi shank data'):
            self.uploader.n_shanks = 4
            self.uploader.shank_idx = 2
            self.uploader.save_channels({'channel_0': {'x': 0}})
            self.assertTrue(self.temp_path.joinpath('channel_locations_shank3.json').exists())


    def test_save_json_file(self):
        """ Test the _save_json_file method """
        file_name = 'test_file.json'
        json_data = {'a': 1, 'b': 2}
        self.uploader._save_json_file(file_name,json_data)
        file_path = self.temp_path.joinpath(file_name)
        self.assertTrue(file_path.exists())
        with open(file_path) as f:
            data = json.load(f)
        self.assertEqual(data, json_data)
