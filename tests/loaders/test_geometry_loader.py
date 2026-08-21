import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import spikeglx

import one.alf.io as alfio
from ibl_alignment_gui.loaders.geometry_loader import (
    ChannelGeometry,
    Geometry,
    GeometryLoader,
    GeometryLoaderLocal,
    GeometryLoaderOne,
    MetaGeometry,
    arrange_channels_into_banks,
    average_chns_at_same_depths,
    find_geometry_mismatches,
    pad_data_to_full_chn_map,
)
from ibl_alignment_gui.utils.parse_yaml import DatasetPaths
from iblutil.util import Bunch
from one.alf.exceptions import ALFObjectNotFound

FIXTURE_PATH = Path(__file__).parents[1].joinpath('fixtures')


def load_channel_fixtures(np_type):
    return alfio.load_object(FIXTURE_PATH.joinpath(np_type), 'channels')


def load_meta_fixtures(np_type):
    file = next(FIXTURE_PATH.joinpath(np_type).glob('*ap.meta'))
    return spikeglx.read_meta_data(file)


def make_channel_geom(x, y):
    """Build a split ChannelGeometry for a synthetic layout"""
    channels = Bunch(
        localCoordinates=np.c_[np.asarray(x), np.asarray(y)], rawInd=np.arange(len(x))
    )
    geom = ChannelGeometry(channels)
    geom.split_sites_per_shank()

    return geom


def make_shank_geom(x, y):
    """Build the shank geometry for a synthetic single shank layout"""
    return make_channel_geom(x, y).shanks[0]


class FixtureGeometryLoader(GeometryLoader):
    """GeometryLoader that reads its metadata and channels from the test fixtures"""

    def __init__(self, np_type):
        self.np_type = np_type
        super().__init__()

    def load_meta_data(self):
        return load_meta_fixtures(self.np_type)

    def load_channels(self, **kwargs):
        return load_channel_fixtures(self.np_type)


class TestGeometry(unittest.TestCase):
    """Test the Geometry class"""

    def _evaluate(self, shank, expected):
        for key in expected:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(shank[key], expected[key])
            else:
                self.assertEqual(expected[key], shank[key])

    def _mock_geometry(self, mock_data):
        """Helper to make a fake Geometry object with abstract methods implemented."""

        class MockGeometry(Geometry):
            def _get_n_shanks(self):
                return mock_data['n_shanks']

            def _get_shank_groups(self):
                return mock_data['shank_groups']

        return MockGeometry(mock_data['x_coords'], mock_data['y_coords'], mock_data['chn_ind'])

    def test_split_sites_per_shank_single_shank(self):
        """Test the split_sites_per_shank method for single shank data"""
        with self.subTest('Single bank, y sorted, even spacing, sequential ind'):
            mock_data = {
                'x_coords': np.array([10, 10, 10, 10, 10, 10]),
                'y_coords': np.array([20, 40, 60, 80, 100, 120]),
                'chn_ind': np.arange(6),
                'n_shanks': 1,
                'shank_groups': {0: np.arange(6)},
            }

            geom = self._mock_geometry(mock_data)
            geom.split_sites_per_shank()

            self.assertEqual(geom.n_shanks, 1)

            expected = {
                'orig_idx': mock_data['shank_groups'][0],
                'sites_coords': np.c_[mock_data['x_coords'], mock_data['y_coords']],
                'raw_ind': mock_data['chn_ind'],
                'spikes_ind': mock_data['chn_ind'],
                'sites_x': mock_data['x_coords'],
                'sites_y': mock_data['y_coords'],
                'sites_min': 20,
                'sites_max': 120,
                'sites_pitch': 20,
                'sites_full': mock_data['y_coords'],
                'idx_full': mock_data['shank_groups'][0],
                'n_banks': 1,
            }

            self._evaluate(geom.shanks[0], expected)

        with self.subTest('Single bank, y unsorted, uneven spacing, sequential ind'):
            mock_data = {
                'x_coords': np.array([10, 10, 10, 10, 10, 10]),
                'y_coords': np.array([20, 40, 60, 160, 80, 180]),
                'chn_ind': np.arange(6),
                'n_shanks': 1,
                'shank_groups': {0: np.arange(6)},
            }

            geom = self._mock_geometry(mock_data)
            geom.split_sites_per_shank()

            self.assertEqual(geom.n_shanks, 1)

            expected = {
                'orig_idx': mock_data['shank_groups'][0],
                'sites_coords': np.c_[mock_data['x_coords'], mock_data['y_coords']],
                'raw_ind': np.array([0, 1, 2, 4, 3, 5]),
                'spikes_ind': np.array([0, 1, 2, 4, 3, 5]),
                'sites_x': mock_data['x_coords'],
                'sites_y': np.array([20, 40, 60, 80, 160, 180]),
                'sites_min': 20,
                'sites_max': 180,
                'sites_pitch': 20,
                'sites_full': np.array([20, 40, 60, 80, 100, 120, 140, 160, 180]),
                'idx_full': np.array([0, 1, 2, 3, 7, 8]),
                'n_banks': 1,
            }

            self._evaluate(geom.shanks[0], expected)

        with self.subTest('Dual bank, y sorted, uneven spacing, non-sequential ind'):
            mock_data = {
                'x_coords': np.array([0, 32, 0, 32, 0, 32, 0, 32]),
                'y_coords': np.array([200, 200, 240, 240, 280, 280, 360, 360]),
                'chn_ind': np.array([0, 5, 6, 2, 4, 3, 8, 1]),
                'n_shanks': 1,
                'shank_groups': {0: np.arange(8)},
            }

            geom = self._mock_geometry(mock_data)
            geom.split_sites_per_shank()

            self.assertEqual(geom.n_shanks, 1)

            expected = {
                'orig_idx': mock_data['shank_groups'][0],
                'sites_coords': np.c_[mock_data['x_coords'], mock_data['y_coords']],
                'raw_ind': mock_data['chn_ind'],
                'spikes_ind': mock_data['chn_ind'],
                'sites_x': mock_data['x_coords'],
                'sites_y': mock_data['y_coords'],
                'sites_min': 200,
                'sites_max': 360,
                'sites_pitch': 40,
                'sites_full': np.array([200, 240, 280, 320, 360]),
                'idx_full': np.array([0, 1, 2, 4]),
                'n_banks': 2,
            }

            self._evaluate(geom.shanks[0], expected)

        with self.subTest('Triple bank, x unsorted'):
            mock_data = {
                'x_coords': np.array([10, 50, 30, 10, 50, 30]),
                'y_coords': np.array([200, 220, 240, 220, 240, 260]),
                'chn_ind': np.arange(6),
                'n_shanks': 1,
                'shank_groups': {0: np.arange(6)},
            }

            geom = self._mock_geometry(mock_data)
            geom.split_sites_per_shank()

            self.assertEqual(geom.n_shanks, 1)

            expected = {
                'orig_idx': mock_data['shank_groups'][0],
                'sites_coords': np.c_[mock_data['x_coords'], mock_data['y_coords']],
                'raw_ind': np.array([0, 1, 3, 2, 4, 5]),
                'spikes_ind': np.array([0, 1, 3, 2, 4, 5]),
                'sites_x': np.array([10, 50, 10, 30, 50, 30]),
                'sites_y': np.array([200, 220, 220, 240, 240, 260]),
                'sites_min': 200,
                'sites_max': 260,
                'sites_pitch': 20,
                'sites_full': np.array([200, 220, 240, 260]),
                'idx_full': np.array([0, 1, 2, 3]),
                'n_banks': 3,
            }

            self._evaluate(geom.shanks[0], expected)

    def test_split_sites_per_shank_multi_shank(self):
        """Test the split_sites_per_shank method for multi shank data"""
        mock_data = {
            'x_coords': np.array(
                [10, 20, 10, 20, 10, 20, 220, 220, 220, 420, 440, 420, 440, 10, 20, 10, 20]
            ),
            'y_coords': np.array(
                [20, 20, 40, 40, 60, 60, 500, 510, 520, 60, 100, 140, 180, 80, 80, 100, 100]
            ),
            'chn_ind': np.arange(18),
            'n_shanks': 3,
            'shank_groups': {
                0: np.array([0, 1, 2, 3, 4, 5, 13, 14, 15, 16]),
                1: np.array([6, 7, 8]),
                2: np.array([9, 10, 11, 12]),
            },
        }

        geom = self._mock_geometry(mock_data)
        geom.split_sites_per_shank()

        self.assertEqual(geom.n_shanks, 3)

        # Content of shank 0
        shank = geom.shanks[0]
        np.testing.assert_array_equal(shank['orig_idx'], mock_data['shank_groups'][0])
        self.assertEqual(shank['sites_min'], 20)
        self.assertEqual(shank['sites_max'], 100)
        self.assertEqual(shank['sites_pitch'], 20)
        self.assertEqual(shank['n_banks'], 2)

        # Content of shank 1
        shank = geom.shanks[1]
        np.testing.assert_array_equal(shank['orig_idx'], mock_data['shank_groups'][1])
        self.assertEqual(shank['sites_min'], 500)
        self.assertEqual(shank['sites_max'], 520)
        self.assertEqual(shank['sites_pitch'], 10)
        self.assertEqual(shank['n_banks'], 1)

        # Content of shank 2
        shank = geom.shanks[2]
        np.testing.assert_array_equal(shank['orig_idx'], mock_data['shank_groups'][2])
        self.assertEqual(shank['sites_min'], 60)
        self.assertEqual(shank['sites_max'], 180)
        self.assertEqual(shank['sites_pitch'], 40)
        self.assertEqual(shank['n_banks'], 2)

    def test_get_sites_for_shank(self):
        """Test the _get_sites_for_shank method for single shank data"""
        mock_data = {
            'x_coords': np.array([0, 0, 200, 200]),
            'y_coords': np.array([200, 210, 240, 250]),
            'chn_ind': np.arange(4),
            'n_shanks': 2,
            'shank_groups': {0: np.array([0, 1]), 1: np.array([2, 3])},
        }

        geom = self._mock_geometry(mock_data)
        geom.split_sites_per_shank()

        shank = geom._get_sites_for_shank(0)
        np.testing.assert_array_equal(shank['orig_idx'], mock_data['shank_groups'][0])

        shank = geom._get_sites_for_shank(1)
        np.testing.assert_array_equal(shank['orig_idx'], mock_data['shank_groups'][1])


class TestChannelGeometry(unittest.TestCase):
    """Test the ChannelGeometry class"""

    def test_get_n_shanks_and_get_shank_groups(self):
        """Test the _get_n_shanks and _get_shank_groups methods"""
        with self.subTest('Single shank'):
            channels = Bunch()
            x = np.array([10, 10, 10, 10, 10, 10])
            y = np.array([20, 40, 60, 80, 100, 120])
            channels['localCoordinates'] = np.c_[x, y]
            channels['rawInd'] = np.arange(6)
            geom = ChannelGeometry(channels)
            geom.n_shanks = geom._get_n_shanks()
            self.assertEqual(geom.n_shanks, 1)
            groups = geom._get_shank_groups()
            np.testing.assert_array_equal(groups[0], np.arange(6))

        with self.subTest('Dual shank - shank_diff=default'):
            channels = Bunch()
            x = np.array([10, 130, 10, 130, 10, 130])
            y = np.array([20, 40, 60, 80, 100, 120])
            channels['localCoordinates'] = np.c_[x, y]
            channels['rawInd'] = np.arange(6)
            geom = ChannelGeometry(channels)
            geom.n_shanks = geom._get_n_shanks()
            self.assertEqual(geom.n_shanks, 2)
            groups = geom._get_shank_groups()
            np.testing.assert_array_equal(groups[0], np.array([0, 2, 4]))
            np.testing.assert_array_equal(groups[1], np.array([1, 3, 5]))

        with self.subTest('Dual shank - shank_diff=150'):
            channels = Bunch()
            x = np.array([10, 130, 10, 130, 10, 130])
            y = np.array([20, 40, 60, 80, 100, 120])
            channels['localCoordinates'] = np.c_[x, y]
            channels['rawInd'] = np.arange(6)
            geom = ChannelGeometry(channels, shank_diff=150)
            geom.n_shanks = geom._get_n_shanks()
            self.assertEqual(geom.n_shanks, 1)
            groups = geom._get_shank_groups()
            np.testing.assert_array_equal(groups[0], np.arange(6))

        with self.subTest('Quarter shank'):
            channels = Bunch()
            x = np.array([10, 30, 140, 160, 280, 300, 450, 500])
            y = np.array([20, 40, 20, 40, 20, 40, 20, 40])
            channels['localCoordinates'] = np.c_[x, y]
            channels['rawInd'] = np.arange(8)
            geom = ChannelGeometry(channels)
            geom.split_sites_per_shank()
            self.assertEqual(geom.n_shanks, 4)
            groups = geom._get_shank_groups()
            np.testing.assert_array_equal(groups[0], np.array([0, 1]))
            np.testing.assert_array_equal(groups[1], np.array([2, 3]))
            np.testing.assert_array_equal(groups[2], np.array([4, 5]))
            np.testing.assert_array_equal(groups[3], np.array([6, 7]))

    def test_split_sites_per_shank(self):
        """Test the split_sites_per_shank method with channels data"""
        channels = Bunch()
        x = np.array([10, 20, 10, 20, 10, 20, 220, 220, 220, 420, 440, 420, 440, 10, 20, 10, 20])
        y = np.array([20, 20, 40, 40, 60, 60, 500, 510, 520, 60, 100, 140, 180, 80, 80, 100, 100])
        ind = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        channels['localCoordinates'] = np.c_[x, y]
        channels['rawInd'] = ind
        geom = ChannelGeometry(channels)
        geom.split_sites_per_shank()

        self.assertEqual(geom.n_shanks, 3)

        # Content of shank 0
        shank = geom.shanks[0]
        np.testing.assert_array_equal(
            shank['orig_idx'], np.array([0, 1, 2, 3, 4, 5, 13, 14, 15, 16])
        )
        self.assertEqual(shank['sites_min'], 20)
        self.assertEqual(shank['sites_max'], 100)
        self.assertEqual(shank['sites_pitch'], 20)
        self.assertEqual(shank['n_banks'], 2)

        # Content of shank 1
        shank = geom.shanks[1]
        np.testing.assert_array_equal(shank['orig_idx'], np.array([6, 7, 8]))
        self.assertEqual(shank['sites_min'], 500)
        self.assertEqual(shank['sites_max'], 520)
        self.assertEqual(shank['sites_pitch'], 10)
        self.assertEqual(shank['n_banks'], 1)

        # Content of shank 2
        shank = geom.shanks[2]
        np.testing.assert_array_equal(shank['orig_idx'], np.array([9, 10, 11, 12]))
        self.assertEqual(shank['sites_min'], 60)
        self.assertEqual(shank['sites_max'], 180)
        self.assertEqual(shank['sites_pitch'], 40)
        self.assertEqual(shank['n_banks'], 2)


class TestMetaGeometry(unittest.TestCase):
    """Test the MetaGeometry class"""

    def setUp(self):
        # Create mock metadata with 4 electrodes on 2 shanks
        self.mock_meta = Bunch(
            x=np.array([0, 0, 100, 100, 0, 0, 100, 100]),
            y=np.array([10, 20, 10, 20, 40, 50, 40, 50]),
            shank=np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        )

    @patch('ibl_alignment_gui.loaders.geometry_loader.spikeglx.geometry_from_meta')
    def test_init(self, mock_geom):
        """Test that the coordinates and channel indices are taken from the metadata"""
        mock_geom.return_value = self.mock_meta

        geom = MetaGeometry(self.mock_meta)
        mock_geom.assert_called_once_with(self.mock_meta, sort=False)
        np.testing.assert_array_equal(geom.x_coords, self.mock_meta['x'])
        np.testing.assert_array_equal(geom.y_coords, self.mock_meta['y'])
        np.testing.assert_array_equal(geom.chn_ind, np.arange(8))

    @patch('ibl_alignment_gui.loaders.geometry_loader.spikeglx.geometry_from_meta')
    def test_get_n_shanks(self, mock_geom):
        """Test the _get_n_shanks method"""
        mock_geom.return_value = self.mock_meta

        geom = MetaGeometry(self.mock_meta)
        n_shanks = geom._get_n_shanks()
        self.assertEqual(n_shanks, 2)

    @patch('ibl_alignment_gui.loaders.geometry_loader.spikeglx.geometry_from_meta')
    def test_get_shank_groups(self, mock_geom):
        """Test the _get_shank_groups method"""
        mock_geom.return_value = self.mock_meta

        geom = MetaGeometry(self.mock_meta)
        groups = geom._get_shank_groups()
        np.testing.assert_array_equal(groups[0], np.array([0, 1, 4, 5]))
        np.testing.assert_array_equal(groups[1], np.array([2, 3, 6, 7]))


class TestRealGeometries(unittest.TestCase):
    """Test with real geometries for different probe types stored in fixtures"""

    def test_np1_3a(self):
        """Test reading geometry from channel data of a 3A NP1 probe"""
        channels = load_channel_fixtures('NP1_3A')
        geom = ChannelGeometry(channels)
        geom.split_sites_per_shank()

        self.assertEqual(geom.n_shanks, 1)
        self.assertEqual(len(geom.shanks), 1)
        np.testing.assert_array_equal(geom.shanks[0]['sites_coords'], channels['localCoordinates'])
        self.assertEqual(geom.shanks[0]['sites_min'], 20)
        self.assertEqual(geom.shanks[0]['sites_max'], 3840)
        self.assertEqual(geom.shanks[0]['sites_pitch'], 20)
        self.assertEqual(geom.shanks[0]['n_banks'], 4)

    def test_np1_3b(self):
        """Test reading geometry from channel data of a NP1 probe"""
        channels = load_channel_fixtures('NP1_3B')
        geom = ChannelGeometry(channels)
        geom.split_sites_per_shank()

        self.assertEqual(geom.n_shanks, 1)
        self.assertEqual(len(geom.shanks), 1)
        np.testing.assert_array_equal(geom.shanks[0]['sites_coords'], channels['localCoordinates'])
        self.assertEqual(geom.shanks[0]['sites_min'], 20)
        self.assertEqual(geom.shanks[0]['sites_max'], 3840)
        self.assertEqual(geom.shanks[0]['sites_pitch'], 20)
        self.assertEqual(geom.shanks[0]['n_banks'], 4)

    def test_np21(self):
        """Test reading geometry from channel data of a NP2 single shank probe"""
        channels = load_channel_fixtures('NP21')
        geom = ChannelGeometry(channels)
        geom.split_sites_per_shank()

        self.assertEqual(geom.n_shanks, 1)
        self.assertEqual(len(geom.shanks), 1)
        # Note these channels are unordered in y so we expect rawInd and orig_idx to differ
        self.assertFalse(np.array_equal(geom.shanks[0]['orig_idx'], geom.shanks[0]['raw_ind']))
        self.assertEqual(geom.shanks[0]['sites_min'], 2880)
        self.assertEqual(geom.shanks[0]['sites_max'], 5745)
        self.assertEqual(geom.shanks[0]['sites_pitch'], 15)
        self.assertEqual(geom.shanks[0]['n_banks'], 2)

    def test_np24(self):
        """Test reading geometry of a NP2 four shank probe with channels on 3 shanks"""
        channels = load_channel_fixtures('NP24')
        geom = ChannelGeometry(channels)
        geom.split_sites_per_shank()

        self.assertEqual(geom.n_shanks, 3)
        self.assertEqual(len(geom.shanks), 3)

        # Shank 0
        shank = geom.shanks[0]
        # Note these channels are unordered in y so we expect rawInd and orig_idx to differ
        self.assertFalse(np.array_equal(shank['orig_idx'], shank['raw_ind']))
        np.testing.assert_array_equal(np.unique(shank['sites_x']), np.array([27, 59]))
        self.assertTrue(all(np.isin(np.array([0, 2, 382, 383]), shank['orig_idx'])))
        self.assertEqual(shank['sites_min'], 1620)
        self.assertEqual(shank['sites_max'], 3750)
        self.assertEqual(shank['sites_pitch'], 15)
        self.assertEqual(shank['n_banks'], 2)

        # Shank 1
        shank = geom.shanks[1]
        # Note these channels are unordered in y so we expect rawInd and orig_idx to differ
        self.assertFalse(np.array_equal(shank['orig_idx'], shank['raw_ind']))
        np.testing.assert_array_equal(np.unique(shank['sites_x']), np.array([277, 309]))
        self.assertTrue(all(np.isin(np.array([79, 145, 310, 334]), shank['orig_idx'])))
        self.assertEqual(shank['sites_min'], 720)
        self.assertEqual(shank['sites_max'], 3765)
        self.assertEqual(shank['sites_pitch'], 15)
        self.assertEqual(shank['n_banks'], 2)

        # Shank 2
        shank = geom.shanks[2]
        # Note these channels are unordered in y so we expect rawInd and orig_idx to differ
        self.assertFalse(np.array_equal(shank['orig_idx'], shank['raw_ind']))
        np.testing.assert_array_equal(np.unique(shank['sites_x']), np.array([559]))
        self.assertTrue(all(np.isin(np.array([1, 3, 309, 311]), shank['orig_idx'])))
        self.assertEqual(shank['sites_min'], 0)
        self.assertEqual(shank['sites_max'], 1650)
        self.assertEqual(shank['sites_pitch'], 15)
        self.assertEqual(shank['n_banks'], 1)

    def test_np21_meta(self):
        """Test reading geometry from the ap.meta file of a NP2 single shank probe"""
        meta = load_meta_fixtures('NP21')
        geom = MetaGeometry(meta)
        geom.split_sites_per_shank()

        self.assertEqual(geom.n_shanks, 1)
        self.assertEqual(len(geom.shanks), 1)
        # Electrode indices are positional, unlike the rawInd used by the channel geometry
        np.testing.assert_array_equal(geom.chn_ind, np.arange(384))

        shank = geom.shanks[0]
        self.assertEqual(shank['sites_y'].size, 384)
        np.testing.assert_array_equal(np.unique(shank['sites_x']), np.array([27, 59]))
        self.assertEqual(shank['sites_min'], 20)
        self.assertEqual(shank['sites_max'], 2885)
        self.assertEqual(shank['sites_pitch'], 15)
        self.assertEqual(shank['n_banks'], 2)

    def test_np24_meta(self):
        """Test reading geometry from the ap.meta file of a NP2 four shank probe"""
        meta = load_meta_fixtures('NP24')
        geom = MetaGeometry(meta)
        geom.split_sites_per_shank()

        # Only three of the four shanks have recorded electrodes
        self.assertEqual(geom.n_shanks, 3)
        self.assertEqual(len(geom.shanks), 3)
        np.testing.assert_array_equal(geom.chn_ind, np.arange(384))

        # In the metadata the x coordinates are relative to each shank, so unlike the channel
        # geometry the same values repeat across shanks.
        expected = [
            {'n_sites': 168, 'x': np.array([27, 59]), 'min': 1640, 'max': 3770, 'n_banks': 2},
            {'n_sites': 105, 'x': np.array([27, 59]), 'min': 740, 'max': 3785, 'n_banks': 2},
            {'n_sites': 111, 'x': np.array([59]), 'min': 20, 'max': 1670, 'n_banks': 1},
        ]

        for i, exp in enumerate(expected):
            with self.subTest(f'Shank {i}'):
                shank = geom.shanks[i]
                self.assertEqual(shank['sites_y'].size, exp['n_sites'])
                np.testing.assert_array_equal(np.unique(shank['sites_x']), exp['x'])
                self.assertEqual(shank['sites_min'], exp['min'])
                self.assertEqual(shank['sites_max'], exp['max'])
                self.assertEqual(shank['sites_pitch'], 15)
                self.assertEqual(shank['n_banks'], exp['n_banks'])

    def test_geometry_mismatch_is_logged(self):
        """Test that loading a real probe warns that the metadata and channels disagree.

        The two sources describe the same sites but reference the depths to different origins,
        so the channel sites fall outside the range the metadata reports and the channels are
        used instead.
        """
        for np_type in ['NP21', 'NP24']:
            with self.subTest(np_type):
                loader = FixtureGeometryLoader(np_type)

                with self.assertLogs(
                    'ibl_alignment_gui.loaders.geometry_loader', level='WARNING'
                ) as log:
                    loader.get_geometry()

                self.assertEqual(len(log.output), 1)
                message = log.output[0]
                self.assertIn('does not match the spike sorting channels', message)
                self.assertIn('using the channels instead', message)

                # One entry per shank, reporting the channel sites out of range
                differences = message.split('Differences: ')[1].split('; ')
                self.assertEqual(len(differences), loader.channels.n_shanks)
                self.assertTrue(all('fall outside' in diff for diff in differences))
                # The banks agree, only the depth reference does not
                self.assertNotIn('number of banks', message)

                # The electrodes are dropped, so the channels are used from here on
                self.assertIsNone(loader.electrodes)
                self.assertIsNotNone(loader.channels)
                np.testing.assert_array_equal(
                    loader.get_sites_for_shank(0)['sites_y'],
                    loader.channels.shanks[0]['sites_y'],
                )


class TestGeometryLoader(unittest.TestCase):
    """Test the GeometryLoader class"""

    def setUp(self):

        self.mock_meta = Bunch(x=np.array([0, 10]), y=np.array([0, 10]), shank=np.array([0, 0]))
        self.mock_channels = Bunch(
            localCoordinates=np.array([[0, 0], [10, 10]]), rawInd=np.array([0, 1])
        )

    def _mock_loaders(self, mock_channels, mock_meta):
        """Helper to make a fake Geometry object with abstract methods implemented."""

        class MockGeometryLoader(GeometryLoader):
            def load_channels(self):
                return mock_channels

            def load_meta_data(self):
                return mock_meta

        return MockGeometryLoader()

    @patch('ibl_alignment_gui.loaders.geometry_loader.find_geometry_mismatches', return_value=[])
    @patch('ibl_alignment_gui.loaders.geometry_loader.ChannelGeometry')
    @patch('ibl_alignment_gui.loaders.geometry_loader.MetaGeometry')
    def test_get_geometry(self, mock_meta, mock_channels, _mock_mismatch):
        """Test the get_geometry method

        The geometry comparison is stubbed out here so that this only covers which sources are
        loaded; the comparison itself is covered by TestFindGeometryMismatches and the mismatch
        handling by test_get_geometry_mismatch_uses_channels.
        """
        mock_meta.return_value = MagicMock()
        mock_meta.split_sites_per_shank.return_value = Bunch()
        mock_meta._get_sites_per_shank.return_value = np.array([0])
        mock_channels.return_value = MagicMock()
        mock_channels.split_sites_per_shank.return_value = Bunch()
        mock_meta._get_sites_per_shank.return_value = np.array([1])

        with self.subTest('Channels and meta exist'):
            loader = self._mock_loaders(self.mock_channels, self.mock_meta)
            loader.get_geometry()
            self.assertIsNotNone(loader.channels)
            self.assertIsNotNone(loader.electrodes)
            mock_meta.assert_called_once_with(self.mock_meta)
            mock_channels.assert_called_once_with(self.mock_channels)

        with self.subTest('Channels exists and meta does not'):
            loader = self._mock_loaders(self.mock_channels, None)
            loader.get_geometry()
            self.assertIsNotNone(loader.channels)
            self.assertIsNone(loader.electrodes)

        with self.subTest('Meta exists and channels does not'):
            loader = self._mock_loaders(None, self.mock_channels)
            loader.get_geometry()
            self.assertIsNone(loader.channels)
            self.assertIsNotNone(loader.electrodes)

        with self.subTest('Channels and meta do not exist'):
            loader = self._mock_loaders(None, None)
            with self.assertRaises(ValueError):
                loader.get_geometry()

    @patch('ibl_alignment_gui.loaders.geometry_loader.ChannelGeometry')
    @patch('ibl_alignment_gui.loaders.geometry_loader.MetaGeometry')
    @patch('ibl_alignment_gui.loaders.geometry_loader.find_geometry_mismatches')
    def test_get_geometry_mismatch_uses_channels(self, mock_mismatch, mock_meta, mock_channels):
        """Test that a mismatch is warned about and the electrodes are dropped"""
        mock_mismatch.return_value = ['shank 0: the deepest site differs: 10 vs 20']

        loader = self._mock_loaders(self.mock_channels, self.mock_meta)
        with self.assertLogs('ibl_alignment_gui.loaders.geometry_loader', level='WARNING') as log:
            loader.get_geometry()

        # The channels are what the data is indexed against, so they win
        self.assertIsNone(loader.electrodes)
        self.assertIsNotNone(loader.channels)
        self.assertTrue(any('deepest site differs' in msg for msg in log.output))

    @patch('ibl_alignment_gui.loaders.geometry_loader.ChannelGeometry')
    @patch('ibl_alignment_gui.loaders.geometry_loader.MetaGeometry')
    @patch('ibl_alignment_gui.loaders.geometry_loader.find_geometry_mismatches')
    def test_get_geometry_match_keeps_electrodes(self, mock_mismatch, mock_meta, mock_channels):
        """Test that nothing is warned about and both are kept when they agree"""
        mock_mismatch.return_value = []

        loader = self._mock_loaders(self.mock_channels, self.mock_meta)
        with self.assertNoLogs('ibl_alignment_gui.loaders.geometry_loader', level='WARNING'):
            loader.get_geometry()

        self.assertIsNotNone(loader.electrodes)
        self.assertIsNotNone(loader.channels)

    @patch('ibl_alignment_gui.loaders.geometry_loader.find_geometry_mismatches')
    def test_get_geometry_single_source_is_not_compared(self, mock_mismatch):
        """Test that the comparison is skipped when only one source is available"""
        loader = self._mock_loaders(self.mock_channels, None)
        loader.get_geometry()

        mock_mismatch.assert_not_called()

    def test_get_sites_for_shank(self):
        """Test the get_sites_for_shank method"""
        electrode_sites = np.array([0])
        channel_sites = np.array([1])

        loader = self._mock_loaders(self.mock_channels, self.mock_meta)

        with self.subTest('Channels and meta exist'):
            loader.electrodes = MagicMock()
            loader.electrodes._get_sites_for_shank.return_value = electrode_sites
            loader.channels = MagicMock()
            loader.channels._get_sites_for_shank.return_value = channel_sites

            # Default returns electrodes
            sites = loader.get_sites_for_shank(0)
            np.testing.assert_array_equal(sites, electrode_sites)
            # If sites=channels return channels
            sites = loader.get_sites_for_shank(0, sites='channels')
            np.testing.assert_array_equal(sites, channel_sites)

        with self.subTest('Channels exists and meta does not'):
            loader.electrodes = None
            sites = loader.get_sites_for_shank(0)
            np.testing.assert_array_equal(sites, channel_sites)

            sites = loader.get_sites_for_shank(0, sites='channels')
            np.testing.assert_array_equal(sites, channel_sites)

        with self.subTest('Meta exists and channels does not'):
            loader.electrodes = MagicMock()
            loader.electrodes._get_sites_for_shank.return_value = electrode_sites
            loader.channels = None
            sites = loader.get_sites_for_shank(0)
            np.testing.assert_array_equal(sites, electrode_sites)

            sites = loader.get_sites_for_shank(0, sites='channels')
            np.testing.assert_array_equal(sites, electrode_sites)


class TestGeometryLoaderOne(unittest.TestCase):
    """Test the GeometryLoaderOne class"""

    def setUp(self):
        self.mock_one = MagicMock()
        self.session_path = Path('/mnt/s0/Data/Subjects/steinmetzlab/KM002/2024-09-16/001')
        self.mock_one.eid2path.return_value = self.session_path

        self.mock_data = Bunch({'x': np.arange(10), 'y': np.arange(10)})

        self.insertion = {'id': uuid.uuid4(), 'session': uuid.uuid4(), 'name': 'probe00'}

        self.loader = GeometryLoaderOne(self.insertion, self.mock_one)

    @patch('ibl_alignment_gui.loaders.data_loader.spikeglx.read_meta_data')
    def test_load_meta_data(self, mock_meta):
        """Test the load_meta_data method"""
        mock_meta.return_value = self.mock_data

        with self.subTest('Meta data exists'):
            self.mock_one.load_dataset.return_value = self.session_path.joinpath(
                '_spikeglx.ap.meta'
            )
            data = self.loader.load_meta_data()
            self.assertTrue(data, self.mock_data)

        with self.subTest('Meta data does not exist'):
            self.mock_one.load_dataset.side_effect = ALFObjectNotFound
            data = self.loader.load_meta_data()
            self.assertIsNone(data)

    def test_load_channels(self):
        """Test the load_channels method"""
        with self.subTest('Channels data exists'):
            self.mock_one.load_object.return_value = self.mock_data
            data = self.loader.load_channels()
            self.assertTrue(data.pop('exists'))
            self.assertTrue(data, self.mock_data)

        with self.subTest('Channels data does not exist'):
            self.mock_one.load_object.side_effect = ALFObjectNotFound
            data = self.loader.load_channels()
            self.assertIsNone(data)


class TestGeometryLoaderLocal(unittest.TestCase):
    """Test the GeometryLoaderLocal class"""

    def setUp(self):
        self.probe_path = Path(
            '/mnt/s0/Data/Subjects/steinmetzlab/KM002/2024-09-16/001/alf/probe00'
        )
        self.collections = DatasetPaths(
            spike_sorting=self.probe_path.joinpath('spikes'),
            raw_ephys=self.probe_path.joinpath('meta'),
        )
        self.mock_data = Bunch({'x': np.arange(10), 'y': np.arange(10)})
        self.loader = GeometryLoaderLocal(self.collections)

    @patch('ibl_alignment_gui.loaders.geometry_loader.spikeglx.read_meta_data')
    def test_load_meta_data(self, mock_meta):
        """Test the load_meta_data method"""

        mock_meta.return_value = Bunch({'x': np.arange(10), 'y': np.arange(10)})

        meta_file = self.probe_path.joinpath('spikeglx.ap.meta')

        with (
            self.subTest('Meta data exists'),
            patch.object(Path, 'glob', return_value=iter([meta_file])),
        ):
            data = self.loader.load_meta_data()
            self.assertEqual(data, mock_meta.return_value)

        with (
            self.subTest('Meta data does not exist'),
            patch.object(Path, 'glob', return_value=iter([])),
        ):
            data = self.loader.load_meta_data()
            self.assertIsNone(data)

        with self.subTest('No raw ephys path in the yaml'):
            # A yaml that does not give a raw_ephys dataset leaves nothing to glob
            loader = GeometryLoaderLocal(
                DatasetPaths(spike_sorting=self.probe_path.joinpath('spikes'))
            )
            self.assertIsNone(loader.meta_path)
            self.assertIsNone(loader.load_meta_data())

    @patch('ibl_alignment_gui.loaders.geometry_loader.alfio.load_object')
    def test_load_channels(self, mock_data):
        """Test the load_channels method"""

        with self.subTest('Channels data exists'):
            mock_data.return_value = self.mock_data
            data = self.loader.load_channels()
            self.assertTrue(data.pop('exists'))
            self.assertTrue(data, self.mock_data)

        with self.subTest('Channels data does not exist'):
            mock_data.side_effect = ALFObjectNotFound
            data = self.loader.load_channels()
            self.assertIsNone(data)


class TestArrangeChannelsIntoBanks(unittest.TestCase):
    """Test the arrange_channels_into_banks function"""

    def test_single_bank(self):
        """Test that a single bank with even spacing is returned in depth order"""
        shank = make_shank_geom([10] * 4, [0, 20, 40, 60])
        img, scale, offset = arrange_channels_into_banks(shank, np.array([1.0, 2.0, 3.0, 4.0]))

        np.testing.assert_array_equal(img, np.array([[1.0, 2.0, 3.0, 4.0]]))
        # x scale is the bank width, y scale spans the shank over the rows of the image
        np.testing.assert_array_equal(scale, np.array([10, (60 - 0) / img.shape[1]]))
        np.testing.assert_array_equal(offset, np.array([0, 0]))

    def test_two_banks(self):
        """Test a NP2 style layout where both banks have a site at every depth"""
        # Both banks sit at every depth, so the within bank spacing equals the site pitch
        shank = make_shank_geom([0, 32, 0, 32], [0, 0, 15, 15])
        img, scale, offset = arrange_channels_into_banks(shank, np.array([1.0, 2.0, 3.0, 4.0]))

        # One row per bank, ordered by x coordinate, one column per depth
        self.assertEqual(img.shape, (2, 2))
        np.testing.assert_array_equal(img, np.array([[1.0, 3.0], [2.0, 4.0]]))
        np.testing.assert_array_equal(offset, np.array([0, 0]))

    def test_checkerboard_layout(self):
        """Test a NP1 style checkerboard where the within bank spacing is twice the pitch"""
        # Rows alternate between the outer (11, 59) and inner (27, 43) columns
        x = [11, 59, 27, 43, 11, 59, 27, 43]
        y = [0, 0, 20, 20, 40, 40, 60, 60]
        shank = make_shank_geom(x, y)
        img, scale, offset = arrange_channels_into_banks(shank, np.arange(1.0, 9.0))

        # An extra row is added and each value is repeated into it so the sites stay square
        self.assertEqual(shank['sites_full'].size, 4)
        self.assertEqual(img.shape, (4, 5))
        expected = np.array(
            [
                [1.0, 1.0, 5.0, 5.0, np.nan],
                [np.nan, 3.0, 3.0, 7.0, 7.0],
                [np.nan, 4.0, 4.0, 8.0, 8.0],
                [2.0, 2.0, 6.0, 6.0, np.nan],
            ]
        )
        np.testing.assert_array_equal(img, expected)
        np.testing.assert_array_equal(scale, np.array([10, (60 - 0) / 5]))

    def test_gaps_are_nan(self):
        """Test that depths with no site are filled with nan"""
        shank = make_shank_geom([10] * 3, [0, 20, 60])

        # The full site map covers the missing depth at y=40
        np.testing.assert_array_equal(shank['sites_full'], np.array([0, 20, 40, 60]))
        img, _, _ = arrange_channels_into_banks(shank, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(img, np.array([[1.0, 2.0, np.nan, 3.0]]))

    def test_bank_width(self):
        """Test that bnk_width sets the scaling along x"""
        shank = make_shank_geom([10] * 4, [0, 20, 40, 60])

        _, scale, _ = arrange_channels_into_banks(shank, np.arange(4.0), bnk_width=25)
        self.assertEqual(scale[0], 25)

    def test_real_probes(self):
        """Test the image dimensions and offsets for real probe geometries"""
        # NP1 is a checkerboard so gets the extra row, the NP2 probes do not
        expected = [
            ('NP1_3B', 0, (4, 193), 20),
            ('NP21', 0, (2, 192), 2880),
            ('NP24', 0, (2, 143), 1620),
        ]
        for np_type, shank_idx, img_shape, sites_min in expected:
            with self.subTest(np_type):
                geom = ChannelGeometry(load_channel_fixtures(np_type))
                geom.split_sites_per_shank()
                shank = geom.shanks[shank_idx]

                data = np.arange(shank['sites_y'].size, dtype=float)
                img, scale, offset = arrange_channels_into_banks(shank, data)

                self.assertEqual(img.shape, img_shape)
                self.assertEqual(img.shape[0], shank['n_banks'])
                np.testing.assert_array_equal(offset, np.array([0, sites_min]))
                self.assertEqual(scale[0], 10)
                # Every value is placed somewhere in the image
                self.assertTrue(np.all(np.isin(data, img[~np.isnan(img)])))


class TestAverageChnsAtSameDepths(unittest.TestCase):
    """Test the average_chns_at_same_depths function"""

    def test_pairs_are_averaged(self):
        """Test that the two channels at each depth are averaged into one column"""
        shank = make_shank_geom([0, 32, 0, 32], [0, 0, 15, 15])
        data = np.array([[1.0, 3.0, 10.0, 30.0], [2.0, 4.0, 20.0, 40.0]])

        averaged = average_chns_at_same_depths(shank, data)

        # One column per depth, each the mean of the pair at that depth
        np.testing.assert_array_equal(averaged, np.array([[2.0, 20.0], [3.0, 30.0]]))

    def test_unpaired_depth_and_nan(self):
        """Test a depth with a single channel, and a pair where one channel is nan"""
        shank = make_shank_geom([0, 32, 0], [0, 0, 15])
        data = np.array([[1.0, np.nan, 7.0]])

        averaged = average_chns_at_same_depths(shank, data)

        # nan is ignored in the pair at y=0, and the lone channel at y=15 is passed through
        np.testing.assert_array_equal(averaged, np.array([[1.0, 7.0]]))

    def test_real_probe(self):
        """Test that the real probe data is reduced to one column per depth"""
        geom = ChannelGeometry(load_channel_fixtures('NP21'))
        geom.split_sites_per_shank()
        shank = geom.shanks[0]

        data = np.tile(np.arange(shank['sites_y'].size, dtype=float), (3, 1))
        averaged = average_chns_at_same_depths(shank, data)

        self.assertEqual(averaged.shape, (3, np.unique(shank['sites_y']).size))


class TestPadDataToFullChnMap(unittest.TestCase):
    """Test the pad_data_to_full_chn_map function"""

    def test_gap_filled_with_nan(self):
        """Test that a missing depth becomes a nan column"""
        shank = make_shank_geom([10] * 3, [0, 20, 60])

        padded = pad_data_to_full_chn_map(shank, np.array([[1.0, 2.0, 3.0]]))

        np.testing.assert_array_equal(padded, np.array([[1.0, 2.0, np.nan, 3.0]]))

    def test_no_gaps(self):
        """Test that data covering every depth is returned unchanged"""
        shank = make_shank_geom([10] * 4, [0, 20, 40, 60])
        data = np.array([[1.0, 2.0, 3.0, 4.0]])

        padded = pad_data_to_full_chn_map(shank, data)

        np.testing.assert_array_equal(padded, data)

    def test_real_probe_after_averaging(self):
        """Test padding the depth averaged data, which is how the plots use it"""
        geom = ChannelGeometry(load_channel_fixtures('NP24'))
        geom.split_sites_per_shank()
        shank = geom.shanks[0]

        data = np.tile(np.arange(shank['sites_y'].size, dtype=float), (3, 1))
        padded = pad_data_to_full_chn_map(shank, average_chns_at_same_depths(shank, data))

        # Padded out to the full site map, with the depths that have no site left as nan
        self.assertEqual(padded.shape, (3, shank['sites_full'].size))
        np.testing.assert_array_equal(np.where(~np.isnan(padded[0]))[0], shank['idx_full'])


class TestFindGeometryMismatches(unittest.TestCase):
    """Test the find_geometry_mismatches function"""

    def _real_geometries(self, np_type):
        electrodes = MetaGeometry(load_meta_fixtures(np_type))
        electrodes.split_sites_per_shank()
        channels = ChannelGeometry(load_channel_fixtures(np_type))
        channels.split_sites_per_shank()

        return electrodes, channels

    def test_identical_geometries(self):
        """Test that nothing is reported for two identical geometries"""
        geom = make_channel_geom([10] * 4, [0, 20, 40, 60])

        self.assertEqual(find_geometry_mismatches(geom, geom), [])

    def test_shank_count(self):
        """Test that a differing shank count is reported on its own"""
        electrodes, _ = self._real_geometries('NP21')  # one shank
        _, channels = self._real_geometries('NP24')  # three shanks

        mismatches = find_geometry_mismatches(electrodes, channels)

        # Nothing else can be compared once the shanks do not line up
        self.assertEqual(len(mismatches), 1)
        self.assertIn('number of shanks differs', mismatches[0])

    def test_number_of_banks(self):
        """Test that a differing number of banks is reported"""
        electrodes = make_channel_geom([10] * 4, [0, 20, 40, 60])
        channels = make_channel_geom([0, 32, 0, 32], [0, 0, 20, 20])

        mismatches = find_geometry_mismatches(electrodes, channels)

        self.assertEqual(len(mismatches), 1)
        self.assertIn('number of banks differs: 1 (metadata) vs 2 (channels)', mismatches[0])

    def test_channels_are_a_subset(self):
        """Test that channels missing from the metadata are not treated as a mismatch.

        Channels that recorded no spikes are dropped from the spike sorting output, so fewer
        sites and a narrower depth range are both expected.
        """
        electrodes = make_channel_geom([10] * 5, [0, 20, 40, 60, 80])

        with self.subTest('Sites missing from the middle'):
            channels = make_channel_geom([10] * 3, [0, 40, 80])
            self.assertEqual(find_geometry_mismatches(electrodes, channels), [])

        with self.subTest('Sites missing from the ends'):
            channels = make_channel_geom([10] * 3, [20, 40, 60])
            self.assertEqual(find_geometry_mismatches(electrodes, channels), [])

    def test_sites_outside_the_metadata_range(self):
        """Test that channel sites beyond the depths given by the metadata are reported"""
        electrodes = make_channel_geom([10] * 4, [0, 20, 40, 60])

        with self.subTest('Deeper than the metadata'):
            channels = make_channel_geom([10] * 4, [0, 20, 40, 100])
            mismatches = find_geometry_mismatches(electrodes, channels)
            self.assertEqual(len(mismatches), 1)
            self.assertIn(
                '1 of 4 channel sites fall outside the 0 to 60 depth range', mismatches[0]
            )

        with self.subTest('Shallower than the metadata'):
            channels = make_channel_geom([10] * 4, [-40, -20, 0, 20])
            mismatches = find_geometry_mismatches(electrodes, channels)
            self.assertEqual(len(mismatches), 1)
            self.assertIn('2 of 4 channel sites fall outside', mismatches[0])
