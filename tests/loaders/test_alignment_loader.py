import json
import tempfile
import unittest
import uuid
from json.decoder import JSONDecodeError
from pathlib import Path
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
            'json': {'xyz_picks': [[1000, 2000, 3000], [1000, 2500, 4000]]},
        }

        self.loader = AlignmentLoaderOne(insertion, self.one_mock, user=self.user)
        self.alignments = {
            '2025-07-03_user1': [[0.5, 0, 0.5], [0.2, 0, 0.2]],
            '2025-06-10_user2': [[0.2, 0, 0.1], [0.2, 0, 0.1]],
        }

    def test_load_xyz_picks(self):
        """Test the load_xyz_picks method"""
        # Test case where we load xyz picks from insertion json
        with self.subTest('xyz_picks present'):
            expected = np.array([[0.001, 0.002, 0.003], [0.001, 0.0025, 0.004]])
            np.testing.assert_array_equal(self.loader.load_xyz_picks(), expected)

        # Test case where we don't have xyz picks in the insertion json
        with self.subTest('xyz_picks missing'):
            insertion = {'id': uuid.uuid4(), 'json': {'alignment_stored': True}}
            loader = AlignmentLoaderOne(insertion, self.one_mock)
            self.assertIsNone(loader.load_xyz_picks())

    def test_load_alignments(self):
        """Test load_alignments, load_previous_alignments and get_previous_alignments methods."""
        # Test case where we have an ephys aligned trajectory with previous alignments
        with self.subTest('Previous alignments found'):
            self.one_mock.alyx.rest.return_value = [{'id': uuid.uuid4(), 'json': self.alignments}]
            self.assertEqual(self.loader.load_alignments(), self.alignments)
            expected_keys = ['2025-07-03_user1', '2025-06-10_user2', 'original']
            self.assertListEqual(self.loader.load_previous_alignments(), expected_keys)
            self.assertListEqual(self.loader.get_previous_alignments(), expected_keys)

        # Test case where we have an ephys aligned trajectory with no alignment
        with self.subTest('No previous alignments in json'):
            # Reset the alignments variable
            self.loader.alignments = Bunch()
            self.one_mock.alyx.rest.return_value = [{'id': uuid.uuid4(), 'json': {}}]
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
        """Test the get_starting_alignment method"""
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

    def test_stored_alignment_idx(self):
        """Test get_stored_alignment_idx returns stored key index or 0 as fallback."""
        self.loader.alignment_keys = ['2025-07-03_user1', '2025-06-10_user2', 'original']
        self.assertEqual(self.loader.get_stored_alignment_idx(), 0)  # no stored key
        self.loader.stored_alignment_key = '2025-06-10_user2'
        self.assertEqual(self.loader.get_stored_alignment_idx(), 1)  # key present
        self.loader.stored_alignment_key = '2025-01-01_other'
        self.assertEqual(self.loader.get_stored_alignment_idx(), 0)  # key missing

    def test_stored_alignment_key_from_insertion(self):
        """Test stored_alignment_key is read from insertion extended_qc on construction."""
        ins = {
            'id': uuid.uuid4(),
            'json': {'extended_qc': {'alignment_stored': '2025-06-10_user2'}},
        }
        loader = AlignmentLoaderOne(ins, self.one_mock)
        self.assertEqual(loader.stored_alignment_key, '2025-06-10_user2')

    def test_extra_alignment(self):
        """Test the add_extra_alignments method"""
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
        """Test the load_trajectory method"""
        # Test case where histology trajectory is found
        with self.subTest('Trajectory found'):
            self.one_mock.alyx.rest.return_value = [{'id': 'traj-id-123', 'x': 123.0}]
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
    """Test the AlignmentLoaderLocal class"""

    def setUp(self):
        self.xyz_picks = {'xyz_picks': [[1000, 2000, 3000], [1000, 2500, 4000]]}
        self.alignments = {
            '2025-07-03_user1': [[0.5, 0, 0.5], [0.2, 0, 0.2]],
            '2025-06-10_user2': [[0.2, 0, 0.1], [0.2, 0, 0.1]],
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_temp_file(self, fname: str, data: dict | str):
        file_path = self.temp_path.joinpath(fname)
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def test_load_xyz_picks_single_shank(self):
        """Test the load_xyz_picks method for single shank data"""
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
        """Test the load_xyz_picks method for multi shank data"""
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
        """Test the load_alignments method for single shank data"""
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
        """Test the load_alignments method for multi shank data"""
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
        """Test the _load_json_file method"""
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


class TestSavedProgress(unittest.TestCase):
    """Test loading a saved work in progress alignment"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.alignments = {
            '2025-07-03_user1': [[0.5, 0, 0.5], [0.2, 0, 0.2]],
            '2025-06-10_user2': [[0.2, 0, 0.1], [0.2, 0, 0.1]],
        }
        self.progress = {
            'feature': [0.1, 0, 0.1],
            'track': [0.3, 0, 0.3],
            'saved': '2026-08-03T14:22:00',
        }
        self.recovered_key = 'recovered (2026-08-03T14:22:00)'

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_progress_file(self, fname: str = 'alignment_progress.json'):
        with open(self.temp_path.joinpath(fname), 'w') as f:
            json.dump(self.progress, f)

    def make_loader(self, n_shanks: int = 1, shank_idx: int = 0):
        insertion = {'id': uuid.uuid4(), 'json': {}}
        loader = AlignmentLoaderOne(
            insertion, MagicMock(), user='test_user', data_path=self.temp_path
        )
        loader.n_shanks = n_shanks
        loader.shank_idx = shank_idx
        loader.load_alignments = MagicMock(return_value=dict(self.alignments))

        return loader

    def test_no_progress_file(self):
        """Test that nothing is recovered when no progress has been saved"""
        loader = self.make_loader()
        keys = loader.load_previous_alignments()

        self.assertIsNone(loader.recovered_key)
        self.assertEqual(keys, sorted(self.alignments, reverse=True) + ['original'])
        # With nothing recovered the stored alignment is used as the starting point
        self.assertEqual(loader.get_start_alignment_idx(), loader.get_stored_alignment_idx())

    def test_progress_is_recovered(self):
        """Test that saved progress is added to the alignments and chosen as the start"""
        self.create_progress_file()
        loader = self.make_loader()
        keys = loader.load_previous_alignments()

        self.assertEqual(loader.recovered_key, self.recovered_key)
        self.assertIn(self.recovered_key, keys)
        self.assertEqual(
            loader.alignments[self.recovered_key],
            [self.progress['feature'], self.progress['track']],
        )

        # The recovered alignment is the one displayed when the data is loaded
        idx = loader.get_start_alignment_idx()
        self.assertEqual(loader.alignment_keys[idx], self.recovered_key)
        loader.get_starting_alignment(idx)
        np.testing.assert_array_equal(loader.feature_prev, self.progress['feature'])
        np.testing.assert_array_equal(loader.track_prev, self.progress['track'])

    def test_recovered_takes_precedence_over_stored(self):
        """Test that a recovered alignment is displayed even when a stored one exists"""
        self.create_progress_file()
        loader = self.make_loader()
        loader.stored_alignment_key = '2025-06-10_user2'
        loader.load_previous_alignments()

        idx = loader.get_start_alignment_idx()
        self.assertEqual(loader.alignment_keys[idx], self.recovered_key)
        # The stored alignment is still reachable from the dropdown
        self.assertIn('2025-06-10_user2', loader.alignment_keys)

    def test_recovered_is_not_uploadable(self):
        """Test that the recovered alignment is never included in an upload"""
        self.create_progress_file()
        loader = self.make_loader()
        loader.load_previous_alignments()

        self.assertIn(self.recovered_key, loader.alignments)
        self.assertNotIn(self.recovered_key, loader.uploadable_alignments)
        self.assertEqual(loader.uploadable_alignments, self.alignments)

    def test_recovered_from_other_loader_is_not_uploadable(self):
        """Test that a recovered alignment copied in from another loader is also excluded"""
        loader = self.make_loader()
        loader.load_previous_alignments()
        # Alignments are copied wholesale between the loaders of different configurations, so a
        # recovered alignment can arrive from a loader that saved it
        self.assertIsNone(loader.recovered_key)
        loader.alignments['recovered (2026-08-01T09:00:00)'] = [[0.1], [0.1]]

        self.assertEqual(loader.uploadable_alignments, self.alignments)

    def test_recovered_key_follows_the_alignments(self):
        """Test that the recovered key is picked up from alignments shared between loaders"""
        self.create_progress_file()
        saved = self.make_loader()
        saved.load_previous_alignments()

        # The loaders of different configurations end up sharing one alignments dict, so a loader
        # with no progress file of its own must still recognise the recovered alignment
        other = self.make_loader()
        other.data_path = None
        other.load_previous_alignments()
        self.assertIsNone(other.recovered_key)
        other.alignments = saved.alignments
        other.get_previous_alignments()

        self.assertEqual(other.recovered_key, self.recovered_key)
        idx = other.get_start_alignment_idx()
        self.assertEqual(other.alignment_keys[idx], self.recovered_key)

    def test_most_recent_recovered_key_is_used(self):
        """Test that the newest recovered alignment is used when two have been merged in"""
        self.create_progress_file()
        loader = self.make_loader()
        loader.load_previous_alignments()
        # An older one merged in from another configuration must not win
        loader.alignments['recovered (2026-08-01T09:00:00)'] = [[0.1], [0.1]]

        self.assertEqual(loader.recovered_key, self.recovered_key)

    def test_progress_replaces_previous_recovery(self):
        """Test that reloading never leaves more than one recovered alignment behind"""
        self.create_progress_file()
        loader = self.make_loader()
        loader.load_previous_alignments()

        # Saving again gives a new key, the stale one must not linger in the dropdown
        self.progress['saved'] = '2026-08-03T15:30:00'
        self.create_progress_file()
        keys = loader.load_previous_alignments()

        recovered = [key for key in keys if key.startswith('recovered')]
        self.assertEqual(recovered, ['recovered (2026-08-03T15:30:00)'])

    def test_progress_recovered_on_refresh(self):
        """Test that the recovered alignment survives the alignments being reloaded"""
        self.create_progress_file()
        loader = self.make_loader()
        loader.load_previous_alignments()
        # Refreshing happens after an upload, so the recovery must not be lost
        keys = loader.load_previous_alignments()
        self.assertIn(self.recovered_key, keys)

        # Once the progress file is deleted the recovered alignment goes away
        self.temp_path.joinpath('alignment_progress.json').unlink()
        keys = loader.load_previous_alignments()
        self.assertIsNone(loader.recovered_key)
        self.assertNotIn(self.recovered_key, keys)

    def test_progress_multi_shank(self):
        """Test that each shank recovers its own saved progress"""
        self.create_progress_file('alignment_progress_shank3.json')

        with self.subTest('Shank with saved progress'):
            loader = self.make_loader(n_shanks=4, shank_idx=2)
            self.assertIn(self.recovered_key, loader.load_previous_alignments())

        with self.subTest('Shank without saved progress'):
            loader = self.make_loader(n_shanks=4, shank_idx=0)
            self.assertNotIn(self.recovered_key, loader.load_previous_alignments())

    def test_no_data_path(self):
        """Test that nothing is recovered when there is nowhere to load progress from"""
        self.create_progress_file()
        loader = self.make_loader()
        loader.data_path = None
        keys = loader.load_previous_alignments()

        self.assertIsNone(loader.recovered_key)
        self.assertNotIn(self.recovered_key, keys)


class TestAlignmentLoaderImport(unittest.TestCase):
    """Test loading previous alignments from local folders alongside the primary source."""

    def setUp(self):
        self.one_mock = MagicMock()
        insertion = {
            'id': uuid.uuid4(),
            'json': {'xyz_picks': [[1000, 2000, 3000], [1000, 2500, 4000]]},
        }
        self.folder = 'HB_011_2024-01-24_probe00'
        self.loader = AlignmentLoaderOne(
            insertion,
            self.one_mock,
            user='test_user',
            import_folder=self.folder,
            import_shank=1,
        )
        # No alignments on the primary source unless a test sets some
        self.one_mock.alyx.rest.return_value = []

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write_alignments(self, user, alignments, shank=1, root=None):
        """Write an alignment file into a user's session folder under a root."""
        root = self.root if root is None else root
        folder = root.joinpath(user, self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        name = 'prev_alignments.json' if shank is None else f'prev_alignments_shank{shank}.json'
        with open(folder.joinpath(name), 'w') as f:
            json.dump(alignments, f)
        return folder

    def test_no_roots(self):
        """Test that nothing is imported until a folder is chosen."""
        self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[1], [2]]})
        self.assertListEqual(self.loader.load_previous_alignments(), ['original'])

    def test_import_from_root(self):
        """Test alignments are read from every user folder under a chosen root."""
        self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[1], [2]]})
        self.write_alignments('alex.roe', {'2026-08-14T09:00:00_alex.roe': [[3], [4]]})
        self.loader.import_roots = [self.root]

        keys = self.loader.load_previous_alignments()
        self.assertListEqual(
            keys,
            [
                '2026-08-14T09:00:00_alex.roe (local)',
                '2026-08-13T15:14:15_jane.doe (local)',
                'original',
            ],
        )
        self.assertEqual(self.loader.alignments[keys[1]], [[1], [2]])

    def test_import_from_multiple_roots(self):
        """Test alignments held under more than one chosen root are merged."""
        other_root = self.root.joinpath('elsewhere')
        self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[1], [2]]})
        self.write_alignments(
            'robin.poe', {'2026-08-15T11:00:00_robin.poe': [[5], [6]]}, root=other_root
        )
        # The nested root is not one of the user folders of the outer root
        self.loader.import_roots = [self.root.joinpath('jane.doe'), other_root]

        keys = self.loader.load_previous_alignments()
        self.assertIn('2026-08-13T15:14:15_jane.doe (local)', keys)
        self.assertIn('2026-08-15T11:00:00_robin.poe (local)', keys)

    def test_import_picked_at_any_level(self):
        """Test the root, a user folder or the session folder itself are all accepted."""
        session = self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[1], [2]]})
        key = '2026-08-13T15:14:15_jane.doe (local)'

        for name, root in (
            ('root', self.root),
            ('user folder', session.parent),
            ('session folder', session),
        ):
            with self.subTest(name):
                self.loader.import_roots = [root]
                self.assertIn(key, self.loader.load_previous_alignments())

    def test_import_ignores_other_sessions(self):
        """Test a session folder belonging to another session is never read."""
        other = self.root.joinpath('jane.doe', 'HB_011_2024-01-25_probe00')
        other.mkdir(parents=True)
        with open(other.joinpath('prev_alignments_shank1.json'), 'w') as f:
            json.dump({'2026-08-13T15:14:15_jane.doe': [[1], [2]]}, f)

        for name, root in (('root', self.root), ('session folder', other)):
            with self.subTest(name):
                self.loader.import_roots = [root]
                self.assertListEqual(self.loader.load_previous_alignments(), ['original'])

    def test_import_shank(self):
        """Test only the file for the shank is read, with no fallback for multi shank data."""
        self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[1], [2]]})
        self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[3], [4]]}, shank=3)
        self.loader.import_roots = [self.root]

        with self.subTest('shank 3'):
            self.loader.import_shank = 3
            self.loader.load_previous_alignments()
            key = '2026-08-13T15:14:15_jane.doe (local)'
            self.assertEqual(self.loader.alignments[key], [[3], [4]])

        with self.subTest('shank without a file'):
            self.loader.import_shank = 4
            self.assertListEqual(self.loader.load_previous_alignments(), ['original'])

    def test_import_single_shank(self):
        """Test single shank data reads a file written with or without a shank number."""
        self.loader.import_shank = None
        self.loader.import_roots = [self.root]
        key = '2026-08-13T15:14:15_jane.doe (local)'

        with self.subTest('with shank in the name'):
            self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[1], [2]]})
            self.assertIn(key, self.loader.load_previous_alignments())

        with self.subTest('without shank in the name'):
            self.write_alignments(
                'jane.doe', {'2026-08-13T15:14:15_jane.doe': [[3], [4]]}, shank=None
            )
            self.loader.load_previous_alignments()
            self.assertEqual(self.loader.alignments[key], [[3], [4]])

    def test_import_tags_key_with_folder_user(self):
        """Test an alignment saved without a user is attributed to the folder's owner."""
        self.write_alignments('jane.doe', {'2026-08-13T15:14:15': [[1], [2]]})
        self.loader.import_roots = [self.root]

        # Tagged with the owner of the folder, not with the user running the gui
        self.assertIn(
            '2026-08-13T15:14:15_jane.doe (local)', self.loader.load_previous_alignments()
        )

    def test_import_skips_alignments_already_known(self):
        """Test an alignment already held by the primary source is not listed twice."""
        alignments = {'2026-08-13T15:14:15_jane.doe': [[1], [2]]}
        self.one_mock.alyx.rest.return_value = [{'id': uuid.uuid4(), 'json': dict(alignments)}]
        self.write_alignments('jane.doe', alignments)
        self.loader.import_roots = [self.root]

        self.assertListEqual(
            self.loader.load_previous_alignments(),
            ['2026-08-13T15:14:15_jane.doe', 'original'],
        )

    def test_import_conflict_prefers_the_attributed_user(self):
        """Test the copy in the folder of the user the key names wins over another copy."""
        key = '2026-08-13T15:14:15_jane.doe'
        # Each folder holds everything the gui that wrote it knew about, so the same alignment
        # turns up in both, and here the two copies disagree
        alex = self.write_alignments('alex.roe', {key: [[9], [9]]}).parent
        jane = self.write_alignments('jane.doe', {key: [[1], [2]]}).parent

        # Whichever order the folders are read in, the copy held by the user the key is
        # attributed to is the one that ends up being used
        for name, roots in (
            ('attributed copy read second', [self.root]),
            ('attributed copy read first', [jane, alex]),
        ):
            with self.subTest(name):
                self.loader.import_roots = roots
                self.loader.load_previous_alignments()
                self.assertEqual(self.loader.alignments[f'{key} (local)'], [[1], [2]])

    def test_import_refresh_drops_stale(self):
        """Test reloading reflects the folders currently chosen rather than adding to them."""
        self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[1], [2]]})
        self.loader.import_roots = [self.root]
        self.assertEqual(len(self.loader.load_previous_alignments()), 2)

        self.loader.import_roots = []
        self.assertListEqual(self.loader.load_previous_alignments(), ['original'])

    def test_imported_are_never_uploadable(self):
        """Test imported alignments are kept out of everything that gets uploaded."""
        alyx = {'2026-08-12T10:00:00_test_user': [[7], [8]]}
        self.one_mock.alyx.rest.return_value = [{'id': uuid.uuid4(), 'json': dict(alyx)}]
        self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[1], [2]]})
        self.loader.import_roots = [self.root]
        self.loader.load_previous_alignments()

        self.assertIn('2026-08-13T15:14:15_jane.doe (local)', self.loader.alignments)
        self.assertDictEqual(self.loader.uploadable_alignments, alyx)

    def test_imported_stay_excluded_once_copied(self):
        """Test imported alignments copied to another loader are still not uploadable."""
        self.write_alignments('jane.doe', {'2026-08-13T15:14:15_jane.doe': [[1], [2]]})
        self.loader.import_roots = [self.root]
        self.loader.load_previous_alignments()

        # Configurations of a shank share one set of alignments, so the loader that receives
        # them has imported nothing itself and must still recognise them
        other = AlignmentLoaderLocal(self.root, 0, 1, user='test_user')
        other.alignments = self.loader.alignments
        self.assertDictEqual(other.uploadable_alignments, {})
