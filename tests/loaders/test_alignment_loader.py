import json
import tempfile
import unittest
import uuid
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Union
from unittest.mock import MagicMock

import numpy as np

from ibl_alignment_gui.loaders.alignment_loader import (
    AlignmentLoaderLocal,
    AlignmentLoaderOne,
)
from iblutil.util import Bunch


class TestAlignmentLoaderOne(unittest.TestCase):
    """Test the AlignmentLoaderOne class."""

    def setUp(self):
        self.one_mock = MagicMock()
        self.user = 'test_user'
        insertion = {
            'id': uuid.uuid4(),
            'json': {'xyz_picks': [[1000, 2000, 3000], [1000, 2500, 4000]]}
        }

        self.loader = AlignmentLoaderOne(insertion, self.one_mock, user=self.user)
        self.alignments = {
            '2025-07-03_user1': [[0.5, 0, 0.5], [0.2, 0, 0.2]],
            '2025-06-10_user2': [[0.2, 0, 0.1], [0.2, 0, 0.1]]
        }

    def test_load_xyz_picks(self):
        """ Test the load_xyz_picks method """
        # Test case where we load xyz picks from insertion json
        with self.subTest('xyz_picks present'):
            expected = np.array([[0.001, 0.002, 0.003], [0.001, 0.0025, 0.004]])
            np.testing.assert_array_equal(self.loader.load_xyz_picks(), expected)

        # Test case where we don't have xyz picks in the insertion json
        with self.subTest('xyz_picks missing'):
            insertion = {
                'id': uuid.uuid4(),
                'json': {'alignment_stored': True}
            }
            loader = AlignmentLoaderOne(insertion, self.one_mock)
            self.assertIsNone(loader.load_xyz_picks())

    def test_load_alignments(self):
        """ Test the load_alignments, load_previous_alignments and get_previous_alignments methods """
        # Test case where we have an ephys aligned trajectory with previous alignments
        with self.subTest('Previous alignments found'):
            self.one_mock.alyx.rest.return_value = [{
                'id': uuid.uuid4(),
                'json': self.alignments
            }]
            self.assertEqual(self.loader.load_alignments(), self.alignments)
            expected_keys = ['2025-07-03_user1', '2025-06-10_user2', 'original']
            self.assertListEqual(self.loader.load_previous_alignments(), expected_keys)
            self.assertListEqual( self.loader.get_previous_alignments(), expected_keys)

        # Test case where we have an ephys aligned trajectory with no alignment
        with self.subTest('No previous alignments in json'):
            # Reset the alignments variable
            self.loader.alignments = Bunch()
            self.one_mock.alyx.rest.return_value = [{
                'id': uuid.uuid4(),
                'json': {}
            }]
            self.assertEqual(self.loader.load_alignments(), dict())
            expected_keys = ['original']
            self.assertListEqual(self.loader.load_previous_alignments(), expected_keys)
            self.assertListEqual(self.loader.get_previous_alignments(), expected_keys)

        # Test case where we don't have an ephys aligned trajectory
        with self.subTest('No ephys aligned trajectory found'):
            # Reset the alignments variable
            self.loader.alignments = Bunch()
            self.one_mock.alyx.rest.return_value = []
            self.assertIsNone(self.loader.load_alignments())
            expected_keys = ['original']
            self.assertListEqual(self.loader.load_previous_alignments(), expected_keys)
            self.assertListEqual(self.loader.get_previous_alignments(), expected_keys)

    def test_starting_alignment(self):
        """ Test the get_starting_alignment method """
        self.loader.alignments = self.alignments
        self.loader.alignment_keys = ['2025-07-03_user1', '2025-06-10_user2', 'original']

        # Load in the alignment at index 0 for the key '2025-07-03_user1'
        with self.subTest('From previous alignment'):
            self.loader.get_starting_alignment(0)
            np.testing.assert_array_equal(self.loader.feature_prev, np.array([0.5, 0, 0.5]))
            np.testing.assert_array_equal(self.loader.track_prev, np.array([0.2, 0, 0.2]))

        # Load in the alignment at index 2 for the key 'original'
        with self.subTest('From original alignment'):
            self.loader.get_starting_alignment(2)
            np.testing.assert_array_equal(self.loader.feature_prev, np.array([-0.006, 0.006]))
            np.testing.assert_array_equal(self.loader.track_prev, np.array([-0.006, 0.006]))

    def test_extra_alignment(self):
        """ Test the add_extra_alignments method """
        self.loader.alignments = self.alignments

        # Test adding extra alignment with a user in the key
        with self.subTest('With user already in key'):
            extra_alignment = {'2025-07-10T15:34:00_user3': [[0.8, 0, 0.6], [0.4, 0, 0.3]]}
            alignment_keys = self.loader.add_extra_alignments(extra_alignment)
            self.assertIn('2025-07-10T15:34:00_user3', alignment_keys)

        # Test adding extra alignment without a user in the key
        with self.subTest('Without user in key'):
            extra_alignment = {'2025-07-10T15:34:00': [[0.8, 0, 0.6], [0.4, 0, 0.3]]}
            alignment_keys = self.loader.add_extra_alignments(extra_alignment)
            self.assertIn('2025-07-10T15:34:00_test_user', alignment_keys)

        # Test adding extra alignment without any original alignments
        with self.subTest('Without original alignment'):
            self.loader.alignments = Bunch()
            extra_alignment = {'2025-07-10T15:34:00': [[0.8, 0, 0.6], [0.4, 0, 0.3]]}
            alignment_keys = self.loader.add_extra_alignments(extra_alignment)
            self.assertListEqual(['2025-07-10T15:34:00_test_user', 'original'], alignment_keys)

    def test_load_trajectory(self):
        """ Test the load_trajectory method """
        # Test case where histology trajectory is found
        with self.subTest('Trajectory found'):
            self.one_mock.alyx.rest.return_value = [{
                'id': 'traj-id-123',
                'x': 123.0
            }]
            self.loader.load_trajectory()
            self.assertEqual(self.loader.traj_id, 'traj-id-123')

        # Test case where histology trajectory is found but has no x value
        with self.subTest('Trajectory missing x'):
            # Reset the traj_id variable
            self.loader.traj_id = None
            self.one_mock.alyx.rest.return_value = [{'id': 'traj-id-456', 'x': None}]
            self.loader.load_trajectory()
            self.assertIsNone(self.loader.traj_id)

        # Test case where no histology trajectory is found
        with self.subTest('No trajectory at all'):
            # Reset the traj_id variable
            self.loader.traj_id = None
            self.one_mock.alyx.rest.return_value = []
            self.loader.load_trajectory()
            self.assertIsNone(self.loader.traj_id)


class TestAlignmentLoaderLocal(unittest.TestCase):
    """ Test the AlignmentLoaderLocal class """

    def setUp(self):
        self.xyz_picks = {'xyz_picks': [[1000, 2000, 3000], [1000, 2500, 4000]]}
        self.alignments = {
            '2025-07-03_user1': [[0.5, 0, 0.5], [0.2, 0, 0.2]],
            '2025-06-10_user2': [[0.2, 0, 0.1], [0.2, 0, 0.1]]
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_temp_file(self, fname: str, data: Union[dict, str]):
        file_path = self.temp_path.joinpath(fname)
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def test_load_xyz_picks_single_shank(self):
        """ Test the load_xyz_picks method for single shank data """
        loader = AlignmentLoaderLocal(self.temp_path, shank_idx=0, n_shanks=1)

        # Test case where we have no xyz_picks data
        with self.subTest('No xyz_picks file present'):
            self.assertIsNone(loader.load_xyz_picks())

        # Test case where we have multi shank xyz_picks data
        with self.subTest('Wrong xyz_picks file (shank2) present'):
            self.create_temp_file('xyz_picks_shank2.json', self.xyz_picks)
            self.assertIsNone(loader.load_xyz_picks())

        # Test case where we have single shank xyz_picks data
        with self.subTest('Correct xyz_picks file present'):
            self.create_temp_file('xyz_picks.json', self.xyz_picks)
            expected = np.array([[0.001, 0.002, 0.003], [0.001, 0.0025, 0.004]])
            np.testing.assert_array_equal(loader.load_xyz_picks(), expected)

    def test_load_xyz_picks_multi_shank(self):
        """ Test the load_xyz_picks method for multi shank data """
        loader = AlignmentLoaderLocal(self.temp_path, shank_idx=2, n_shanks=4)

        # Test case where we have no xyz_picks data
        with self.subTest('No xyz_picks file present'):
            self.assertIsNone(loader.load_xyz_picks())

        # Test case where we have single shank xyz_picks data
        with self.subTest('Wrong xyz_picks file (single-shank) present'):
            self.create_temp_file('xyz_picks.json', self.xyz_picks)
            self.assertIsNone(loader.load_xyz_picks())

        # Test case where we have multi shank xyz_picks data
        with self.subTest('Correct xyz_picks_shank3.json present (shank_idx=2)'):
            self.create_temp_file('xyz_picks_shank3.json', self.xyz_picks)
            expected = np.array([[0.001, 0.002, 0.003], [0.001, 0.0025, 0.004]])
            np.testing.assert_array_equal(loader.load_xyz_picks(), expected)

    def test_load_alignments_single_shank(self):
        """ Test the load_alignments method for single shank data """
        loader = AlignmentLoaderLocal(self.temp_path, shank_idx=0, n_shanks=1)

        # Test case where we have no alignments data
        with self.subTest('No alignments file present'):
            self.assertIsNone(loader.load_alignments())

        # Test case where we have multi shank alignment data
        with self.subTest('Wrong alignments file (shank2) present'):
            self.create_temp_file('prev_alignments_shank2.json', self.alignments)
            self.assertIsNone(loader.load_alignments())

        # Test case where we have single shank alignment data
        with self.subTest('Correct prev_alignments.json present'):
            self.create_temp_file('prev_alignments.json', self.alignments)
            self.assertEqual(loader.load_alignments(), self.alignments)

    def test_load_alignments_multi_shank(self):
        """ Test the load_alignments method for multi shank data """
        loader = AlignmentLoaderLocal(self.temp_path, shank_idx=3, n_shanks=4)

        # Test case where we have no alignments data
        with self.subTest('No alignments file present'):
            self.assertIsNone(loader.load_alignments())

        # Test case where we have single shank alignment data
        with self.subTest('Wrong alignments file (single shank) present'):
            self.create_temp_file('prev_alignments.json', self.alignments)
            self.assertIsNone(loader.load_alignments())

        # Test case where we have multi shank alignment data
        with self.subTest('Correct prev_alignments_shank4.json present (shank_idx=3)'):
            self.create_temp_file('prev_alignments_shank4.json', self.alignments)
            self.assertEqual(loader.load_alignments(), self.alignments)

    def test_load_json_file(self):
        """ Test the _load_json_file method """
        # File exists and contains valid JSON
        with self.subTest('Valid JSON file'):
            self.create_temp_file('valid.json', {'key': 'value'})
            file_path = self.temp_path.joinpath('valid.json')
            data = AlignmentLoaderLocal._load_json_file(file_path)
            self.assertEqual(data, {'key': 'value'})

        # File does not exist
        with self.subTest('File does not exist'):
            file_path = self.temp_path.joinpath('nonexistent.json')
            data = AlignmentLoaderLocal._load_json_file(file_path)
            self.assertIsNone(data)

        # File exists but contains corrupt JSON
        with self.subTest('Corrupt JSON file'):
            file_path = self.temp_path.joinpath('corrupt.json')
            file_path.write_text('{"key": "missing end quote}')
            with self.assertRaises(JSONDecodeError):
                AlignmentLoaderLocal._load_json_file(file_path)
