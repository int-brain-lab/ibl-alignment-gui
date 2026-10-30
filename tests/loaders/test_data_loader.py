import logging
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import one.alf.io as alfio
from ibl_alignment_gui.loaders import data_loader
from ibl_alignment_gui.loaders.data_loader import (
    DataLoader,
    DataLoaderLocal,
    DataLoaderOne,
    FeatureLoaderOne,
    SpikeGLXLoader,
    SpikeGLXLoaderLocal,
    SpikeGLXLoaderOne,
)
from ibl_alignment_gui.utils.optional import IBL_EXTRA_HINT, has_ibllib
from ibl_alignment_gui.utils.parse_yaml import DatasetPaths
from iblutil.util import Bunch
from one.alf.exceptions import ALFObjectNotFound
from tests.loaders._alf_session import write_session


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.mock_geometry = Bunch(
            {
                'spikes_ind': np.array([0, 1]),
                'raw_ind': np.array([0, 1]),
                'unsort': np.array([0, 1]),
            }
        )

        self.sample_data = Bunch(
            {
                'amps': np.ones((5, 2)),
                'timestamps': np.arange(5),
            }
        )

        self.eid_or_path = str(uuid.uuid4())

    def _mock_loaders(self, mock_data):
        """Helper to make a fake DataLoader with abstract methods implemented."""

        class MockDataLoader(DataLoader):
            def load_passive_data(self, *a, **kw):
                return mock_data

            def load_raw_passive_data(self, *a, **kw):
                return mock_data

            def load_ephys_data(self, *a, **kw):
                return mock_data

            def load_spikes_data(self, *a, **kw):
                return mock_data

        loader = MockDataLoader()
        loader.shank_sites = self.mock_geometry

        return loader

    def test_load_data(self):
        """Test the load_data method"""
        with self.subTest('Data exists'):
            func = MagicMock(return_value=self.sample_data)
            data = DataLoader.load_data(func, self.eid_or_path, 'ephysTimeRmsAP')
            self.assertTrue(data['exists'])

        with self.subTest('Data does not exist - catches error'):
            func = MagicMock(side_effect=ALFObjectNotFound)
            with self.assertLogs('ibl_alignment_gui.loaders.data_loader', level='WARNING') as log:
                data = DataLoader.load_data(func, self.eid_or_path, 'ephysTimeRmsAP')
            self.assertFalse(data['exists'])
            self.assertTrue(any('ephysTimeRmsAP' in msg for msg in log.output))

        with self.subTest('Data does not exist - custom error message'):
            func = MagicMock(side_effect=ALFObjectNotFound)
            with self.assertLogs(
                logging.getLogger('ibl_alignment_gui.loaders.data_loader'), logging.WARNING
            ) as log:
                data = DataLoader.load_data(
                    func, self.eid_or_path, 'ephysTimeRmsAP', raise_message='custom message'
                )

            self.assertFalse(data['exists'])
            self.assertTrue(any('custom message' in msg for msg in log.output))

        with self.subTest('Data does not exist - raise error'):
            func = MagicMock(side_effect=ALFObjectNotFound)
            with self.assertRaises(ALFObjectNotFound):
                DataLoader.load_data(func, self.eid_or_path, 'ephysTimeRmsAP', raise_error=True)

    @patch('ibl_alignment_gui.loaders.data_loader.DataLoader.get_spikes_data')
    @patch('ibl_alignment_gui.loaders.data_loader.DataLoader.get_rms_data')
    @patch('ibl_alignment_gui.loaders.data_loader.DataLoader.get_psd_data')
    @patch('ibl_alignment_gui.loaders.data_loader.DataLoader.get_passive_data')
    def test_get_data(self, mock_passive, mock_psd, mock_rms, mock_spikes):
        """Test the get_data method"""
        mock_passive.return_value = (Bunch(exists=True), Bunch(exists=True), Bunch(exists=True))
        mock_rms.return_value = Bunch(exists=True)
        mock_psd.return_value = Bunch(exists=True)
        mock_spikes.return_value = (Bunch(exists=True), Bunch(exists=True), Bunch(exists=True))

        loader = self._mock_loaders(Bunch(exists=True))
        data = loader.get_data(Bunch({'unsort': np.array([0, 1])}))
        self.assertIn('spikes', data)
        self.assertIn('clusters', data)
        self.assertIn('channels', data)
        self.assertIn('rms_LF', data)
        self.assertIn('psd_LF', data)
        self.assertIn('rf_map', data)
        self.assertIn('pass_stim', data)
        self.assertIn('gabor', data)

    def test_get_passive_data(self):
        """Test the get_passive_data method"""
        with self.subTest('All passive data exists'):
            loader = self._mock_loaders(Bunch(exists=True))
            loader.load_passive_data = MagicMock(
                return_value=Bunch(
                    {
                        'position': np.array([35, -35]),
                        'contrast': np.array([0.5, 0.5]),
                        'start': np.array([1, 2]),
                        'exists': True,
                    }
                )
            )
            loader.load_raw_passive_data = MagicMock(return_value={'raw': '/tmp/rfmap.bin'})

            with patch('numpy.fromfile', return_value=np.arange(15 * 15 * 2, dtype=np.uint8)):
                rf_data, stim_data, gabor_data = loader.get_passive_data()
            self.assertTrue(rf_data['exists'])
            self.assertTrue(stim_data['exists'])
            self.assertTrue(gabor_data['exists'])

        with self.subTest('Passive RFM data does not exist'):
            loader = self._mock_loaders(Bunch(exists=True))
            loader.load_passive_data = MagicMock(
                side_effect=[
                    Bunch(exists=False),
                    Bunch(exists=True),
                    Bunch(
                        {
                            'position': np.array([35, -35]),
                            'contrast': np.array([0.5, 0.5]),
                            'start': np.array([1, 2]),
                            'exists': True,
                        }
                    ),
                ]
            )
            loader.load_raw_passive_data = MagicMock(side_effect=ALFObjectNotFound)
            with self.assertLogs('ibl_alignment_gui.loaders.data_loader', level='WARNING') as log:
                rf_data, stim_data, gabor_data = loader.get_passive_data()

            self.assertFalse(rf_data['exists'])
            self.assertTrue(stim_data['exists'])
            self.assertTrue(gabor_data['exists'])
            self.assertTrue(any('passiveRFM' in msg for msg in log.output))

        with self.subTest('Passive Gabor data does not exists'):
            loader = self._mock_loaders(Bunch(exists=True))
            loader.load_passive_data = MagicMock(
                side_effect=[Bunch(exists=True), Bunch(exists=True), Bunch(exists=False)]
            )
            loader.load_raw_passive_data = MagicMock(return_value={'raw': '/tmp/fake.bin'})
            with patch('numpy.fromfile', return_value=np.arange(15 * 15 * 2, dtype=np.uint8)):
                rf_data, stim_data, gabor_data = loader.get_passive_data()
            self.assertTrue(rf_data['exists'])
            self.assertTrue(stim_data['exists'])
            self.assertFalse(gabor_data['exists'])

        with self.subTest('Passive Gabor data missing keys'):
            loader = self._mock_loaders(Bunch(exists=True))
            loader.load_passive_data = MagicMock(
                side_effect=[Bunch(exists=True), Bunch(exists=True), ALFObjectNotFound]
            )
            loader.load_raw_passive_data = MagicMock(return_value={'raw': '/tmp/fake.bin'})
            with (
                self.assertLogs('ibl_alignment_gui.loaders.data_loader', level='WARNING') as log,
                patch('numpy.fromfile', return_value=np.arange(15 * 15 * 2, dtype=np.uint8)),
            ):
                rf_data, stim_data, gabor_data = loader.get_passive_data()
            self.assertTrue(rf_data['exists'])
            self.assertTrue(stim_data['exists'])
            self.assertFalse(gabor_data['exists'])
            self.assertTrue(any('Failed to process' in msg for msg in log.output))

    def test_get_rms_data(self):
        """Test the get_rms_data method"""

        def _test_rms(test_data, xlabel, band='AP'):
            loader = self._mock_loaders(test_data)
            data = loader.get_rms_data(band)
            self.assertIn('rms', data)
            self.assertIn('timestamps', data)
            self.assertEqual(data['xaxis'], xlabel)

        with self.subTest('AP RMS data with rms attribute'):
            mock_data = Bunch({'rms': np.ones((5, 2)), 'timestamps': np.arange(5), 'exists': True})
            _test_rms(mock_data, 'Time (s)')

        with self.subTest('LF RMS data with rms attribute'):
            mock_data = Bunch({'rms': np.ones((5, 2)), 'timestamps': np.arange(5), 'exists': True})
            _test_rms(mock_data, 'Time (s)', band='LF')

        with self.subTest('AP RMS data with amps attribute'):
            mock_data = Bunch(
                {'amps': np.ones((5, 2)), 'timestamps': np.arange(5), 'exists': True}
            )
            _test_rms(mock_data, 'Time (s)')

        with self.subTest('AP RMS data without timestamps attribute'):
            mock_data = Bunch({'rms': np.ones((5, 2)), 'exists': True})
            _test_rms(mock_data, 'Time samples')

        with self.subTest('No AP RMS data'):
            mock_data = Bunch(
                {
                    'exists': False,
                }
            )
            loader = self._mock_loaders(mock_data)
            data = loader.get_rms_data('AP')
            self.assertNotIn('rms', data)
            self.assertFalse(data['exists'])

    def test_get_psd_data(self):
        """Test the get_psd_data method"""

        def _test_psd(test_data, band='AP'):
            loader = self._mock_loaders(test_data)
            data = loader.get_psd_data(band)
            self.assertIn('power', data)
            self.assertIn('freqs', data)

        with self.subTest('LF PSD data with power attribute'):
            mock_data = Bunch({'power': np.ones((5, 2)), 'freqs': np.arange(5), 'exists': True})
            _test_psd(mock_data)

        with self.subTest('AP PSD data with power attribute'):
            mock_data = Bunch({'power': np.ones((5, 2)), 'freqs': np.arange(5), 'exists': True})
            _test_psd(mock_data, band='AP')

        with self.subTest('LF PSD data with amps attribute'):
            mock_data = Bunch({'amps': np.ones((5, 2)), 'freqs': np.arange(5), 'exists': True})
            _test_psd(mock_data, band='AP')

        with self.subTest('No LF PSD data'):
            mock_data = Bunch(
                {
                    'exists': False,
                }
            )
            loader = self._mock_loaders(mock_data)
            data = loader.get_psd_data('LF')
            self.assertNotIn('psd', data)
            self.assertFalse(data['exists'])

    @patch('ibl_alignment_gui.loaders.data_loader.DataLoader.filter_spikes_by_fr')
    def test_get_spikes_data(self, mock_fr):
        """Test the get_spikes_data method"""
        mock_spikes = Bunch({'clusters': np.array([0, 1, 2]), 'exists': True})

        mock_clusters = Bunch(
            {
                'channels': np.array([0, 1, 2]),
                'metrics': Bunch(firing_rate=np.array([1.0, 0.0, 2.0])),
                'exists': True,
            }
        )

        mock_channels = Bunch({'rawInd': np.array([0, 1, 2]), 'exists': True})

        mock_fr.return_value = (mock_spikes, mock_clusters)

        with self.subTest('Spike sorting data exists'):
            loader = self._mock_loaders(Bunch(exists=True))
            loader.load_spikes_data = MagicMock(
                side_effect=[mock_spikes, mock_clusters, mock_channels]
            )
            spikes, clusters, channels = loader.get_spikes_data()
            self.assertTrue(spikes['exists'])
            self.assertTrue(clusters['exists'])
            self.assertTrue(channels['exists'])

        with self.subTest('Spike sorting data exists and filter'):
            loader = self._mock_loaders(Bunch(exists=True))
            loader.filter = True
            loader.load_spikes_data = MagicMock(
                side_effect=[mock_spikes, mock_clusters, mock_channels]
            )
            _ = loader.get_spikes_data()
            mock_fr.assert_called_once()

        with self.subTest('Spike sorting data does not exists'):
            loader = self._mock_loaders(Bunch(exists=False))
            spikes, clusters, channels = loader.get_spikes_data()
            self.assertFalse(spikes['exists'])
            self.assertFalse(clusters['exists'])
            self.assertFalse(channels['exists'])

    def test_filter_spikes_by_chns(self):
        """Test the filter_spikes_by_chns method"""
        mock_spikes = Bunch({'clusters': np.array([0, 1, 2, 5, 1, 3, 2]), 'exists': True})
        mock_clusters = Bunch({'channels': np.array([3, 0, 2, 1, 5, 17]), 'exists': True})
        mock_channels = Bunch({'rawInd': np.arange(18), 'exists': True})

        loader = self._mock_loaders(Bunch(exists=True))
        spikes, clusters, channels = loader.filter_spikes_by_chns(
            mock_spikes, mock_clusters, mock_channels
        )
        np.testing.assert_array_equal(spikes['clusters'], np.array([1, 1, 3]))
        self.assertTrue(spikes['exists'])
        # Clusters and channels don't change for now
        self.assertEqual(mock_clusters, clusters)
        self.assertEqual(mock_channels, channels)

    def test_filter_raw_by_chns(self):
        """Test the filter_raw_by_chns method"""
        mock_data = Bunch({'rms': np.ones((5, 6)), 'timestamps': np.arange(5), 'exists': True})

        loader = self._mock_loaders(Bunch(exists=True))
        data = loader.filter_raw_by_chns(mock_data)
        self.assertEqual(data['rms'].shape[1], 2)
        np.testing.assert_array_equal(data['rms'], np.ones((5, 2)))
        np.testing.assert_array_equal(data['timestamps'], np.arange(5))
        self.assertTrue(data['exists'])

    def test_filter_spikes_by_fr(self):
        """Test the filter_spikes_by_fr method"""
        mock_spikes = Bunch(
            {
                'clusters': np.array([0, 1, 3, 2]),
                'amps': np.array([30, 60, 20, 40]),
                'exists': True,
            }
        )
        metrics = pd.DataFrame()
        metrics['firing_rate'] = np.array([1.0, 0.0, 2.0, 2.0])
        mock_clusters = Bunch(
            {'metrics': metrics, 'channels': np.array([3, 0, 2, 1]), 'exists': True}
        )

        spikes, clusters = DataLoader.filter_spikes_by_fr(mock_spikes, mock_clusters, min_fr=0.5)
        # Low FR clusters are removed (i.e cluster 1) and the spikes re-indexed
        # (i.e > 1 shifted down by 1)
        np.testing.assert_array_equal(spikes['clusters'], np.array([0, 2, 1]))
        np.testing.assert_array_equal(spikes['amps'], np.array([30, 20, 40]))
        np.testing.assert_array_equal(
            clusters.metrics.firing_rate.values, np.array([1.0, 2.0, 2.0])
        )
        np.testing.assert_array_equal(clusters.metrics.index, np.array([0, 1, 2]))
        np.testing.assert_array_equal(clusters['channels'], np.array([3, 2, 1]))
        self.assertTrue(clusters['exists'])
        self.assertTrue(spikes['exists'])


class TestDataLoaderOne(unittest.TestCase):
    def setUp(self):
        self.session_path = Path('/mnt/s0/Data/Subjects/steinmetzlab/KM002/2024-09-16/001')
        self.mock_one = MagicMock()
        self.mock_one.eid2path.return_value = self.session_path

        self.insertion = {'id': uuid.uuid4(), 'session': uuid.uuid4(), 'name': 'probe00'}

    def test_get_spike_sorting_path(self):
        """Test the get_spike_sorting_path method"""
        loader = DataLoaderOne(self.insertion, self.mock_one, 0)

        with self.subTest('Spike sorting collection iblsorter from one'):
            self.mock_one.list_collections.return_value = [
                'alf/probe00/iblsorter',
                'alf/probe00/pykilosort',
            ]
            probe_path = loader.get_spike_sorting_path()
            self.assertEqual(probe_path, self.session_path.joinpath('alf/probe00/iblsorter'))

        with self.subTest('Spike sorting collection other from one'):
            self.mock_one.list_collections.return_value = ['alf/probe00/test_sorter']
            probe_path = loader.get_spike_sorting_path()
            self.assertEqual(probe_path, self.session_path.joinpath('alf/probe00'))

        with self.subTest('Spike sorting collection specified'):
            loader.spike_collection = 'pykilosort'
            probe_path = loader.get_spike_sorting_path()
            self.assertEqual(probe_path, self.session_path.joinpath('alf/probe00/pykilosort'))

            loader.spike_collection = ''
            probe_path = loader.get_spike_sorting_path()
            self.assertEqual(probe_path, self.session_path.joinpath('alf/probe00'))

    def test_loaders(self):
        """Test the load methods"""
        mock_data = Bunch({'vals1': np.arange(5), 'vals2': np.ones(5)})

        loader = DataLoaderOne(self.insertion, self.mock_one, 0)
        self.mock_one.load_object.return_value = mock_data

        data = loader.load_ephys_data('ephysTimeRmsAP')
        self.assertTrue(data.pop('exists'))
        self.assertIs(data, mock_data)

        data = loader.load_passive_data('passiveRFM')
        self.assertTrue(data.pop('exists'))
        self.assertIs(data, mock_data)

        data = loader.load_raw_passive_data('RFMapStim')
        self.assertTrue(data.pop('exists'))
        self.assertIs(data, mock_data)

        data = loader.load_spikes_data('spikes', attributes=['vals1', 'vals2'])
        self.assertTrue(data.pop('exists'))
        self.assertIs(data, mock_data)


class TestDataLoaderLocal(unittest.TestCase):
    def setUp(self):
        self.probe_path = Path(
            '/mnt/s0/Data/Subjects/steinmetzlab/KM002/2024-09-16/001/alf/probe00'
        )
        self.collections = DatasetPaths(
            spike_sorting=self.probe_path.joinpath('spikes'),
            processed_ephys=self.probe_path.joinpath('ephys'),
            task=self.probe_path.joinpath('task'),
            raw_task=self.probe_path.joinpath('raw_task'),
            raw_ephys=self.probe_path.joinpath('raw_ephys'),
        )

    def test_local_paths(self):
        """Test the init method to ensure local paths are set correctly"""
        loader = DataLoaderLocal(self.collections)
        self.assertEqual(loader.spike_path, self.collections.spike_sorting)
        self.assertEqual(loader.ephys_path, self.collections.processed_ephys)
        self.assertEqual(loader.task_path, self.collections.task)
        self.assertEqual(loader.raw_task_path, self.collections.raw_task)
        self.assertEqual(loader.meta_path, self.collections.raw_ephys)
        self.assertEqual(loader.probe_collection, 'spikes')

    @patch('ibl_alignment_gui.loaders.data_loader.alfio.load_object')
    def test_loaders(self, mock_data):
        """Test the load methods"""
        mock_data.return_value = Bunch({'vals1': np.arange(5), 'vals2': np.ones(5)})

        loader = DataLoaderLocal(self.collections)

        data = loader.load_ephys_data('ephysTimeRmsAP')
        self.assertTrue(data.pop('exists'))
        self.assertIs(data, mock_data.return_value)

        data = loader.load_passive_data('passiveRFM')
        self.assertTrue(data.pop('exists'))
        self.assertIs(data, mock_data.return_value)

        data = loader.load_raw_passive_data('RFMapStim')
        self.assertTrue(data.pop('exists'))
        self.assertIs(data, mock_data.return_value)

        data = loader.load_spikes_data('spikes', attributes=['vals1', 'vals2'])
        self.assertTrue(data.pop('exists'))
        self.assertIs(data, mock_data.return_value)


class TestLoadersWithoutDatasets(unittest.TestCase):
    """A session can have no spike sorting, and no raw data."""

    def test_no_spike_sorting(self):
        loader = DataLoaderLocal(DatasetPaths(processed_ephys=Path('/proc')))
        self.assertIsNone(loader.probe_collection)
        self.assertEqual(loader.load_spikes_data('spikes', attributes=['amps']), {'exists': False})

    def test_no_raw_data(self):
        loader = SpikeGLXLoaderLocal(None)
        self.assertIsNone(loader.load_meta_data())
        self.assertIsNone(loader.load_ap_data())
        self.assertIsNone(loader.load_lf_data())


class TestSpikeGLXLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.mock_spikeglx = MagicMock()
        self.mock_spikeglx.geometry_from_meta.return_value = Bunch(
            {'x': np.arange(10), 'y': np.arange(10)}
        )
        self.mock_spikeglx.meta = Bunch({'fileTimeSecs': 2000})
        self.mock_spikeglx.fs = 30000
        self.mock_spikeglx.n_sync = 1
        self.mock_spikeglx.__getitem__.return_value = np.ones((30000, 385))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _mock_loaders(self, mock_data):
        """Helper to make a fake DataLoader with abstract methods implemented."""

        class MockDataLoader(SpikeGLXLoader):
            def load_meta_data(self):
                return mock_data

            def load_ap_data(self):
                return mock_data

            def load_lf_data(self):
                return mock_data

        return MockDataLoader(self.temp_path)

    @patch('ibl_alignment_gui.loaders.data_loader.spikeglx.geometry_from_meta')
    def test_get_meta_data(self, mock_geom):
        """Test the get_meta_data method."""
        with self.subTest('Meta data exists'):
            mock_geom.return_value = Bunch({'x': np.arange(10), 'y': np.arange(10)})
            loader = self._mock_loaders(Bunch(adc=np.ones(10)))
            data = loader.get_meta_data()
            self.assertTrue(data.pop('exists'))
            np.testing.assert_equal(dict(data), dict(mock_geom.return_value))

        with self.subTest('Meta data does not exist'):
            loader = self._mock_loaders(None)
            data = loader.get_meta_data()
            self.assertFalse(data['exists'])

    def test_get_time_snippets(self):
        """Test the get_time_snippets method."""
        with self.subTest('2000s'):
            self.mock_spikeglx.meta = Bunch({'fileTimeSecs': 2000})
            loader = self._mock_loaders(None)
            snippets = loader.get_time_snippets(self.mock_spikeglx)
            np.testing.assert_array_equal(snippets, np.array([533, 1066, 1599]))
            snippets = loader.get_time_snippets(self.mock_spikeglx, n=4)
            np.testing.assert_array_equal(snippets, np.array([400, 800, 1200, 1600]))

        with self.subTest('400s'):
            self.mock_spikeglx.meta = Bunch({'fileTimeSecs': 400})
            loader = self._mock_loaders(None)
            snippets = loader.get_time_snippets(self.mock_spikeglx)
            np.testing.assert_array_equal(snippets, np.array([133, 266, 399]))
            snippets = loader.get_time_snippets(self.mock_spikeglx, n=2)
            np.testing.assert_array_equal(snippets, np.array([200, 400]))

    @patch('ibl_alignment_gui.loaders.data_loader.ibldsp.voltage.detect_bad_channels')
    @patch('ibl_alignment_gui.loaders.data_loader.ibldsp.voltage.destripe')
    def test_get_ap_snippet(self, mock_destripe, mock_bad_channels):
        """Test the _get_ap_snippet method"""
        mock_bad_channels.return_value = (np.zeros(384), None)
        mock_destripe.return_value = np.ones((384, 30000))

        loader = self._mock_loaders(None)
        snippet, *_ = loader._get_ap_snippet(self.mock_spikeglx, t=0, twin=1)
        self.assertEqual(snippet.shape, (1500, 384))

    @patch('ibl_alignment_gui.loaders.data_loader.SpikeGLXLoader.get_time_snippets')
    @patch('ibl_alignment_gui.loaders.data_loader.SpikeGLXLoader._get_ap_snippet')
    def test_load_ap_snippets(self, mock_snippet, mock_times):
        """Test the load_ap_snippets method"""
        mock_raw = np.random.randn(1500, 384)
        mock_labels = np.zeros(384)
        mock_features = {
            'xcor_hf': np.random.rand(384),
            'xcor_lf': np.random.rand(384),
            'psd_hf': np.random.rand(384),
        }
        mock_snippet.return_value = (mock_raw, mock_labels, mock_features)
        mock_times.return_value = np.array([200, 400, 600])

        with self.subTest('No raw data snippets'):
            loader = self._mock_loaders(None)
            data = loader.load_ap_snippets()
            self.assertFalse(data['exists'])

        # First time read in data and save to disk
        with self.subTest('Generate raw data snippets'):
            loader = self._mock_loaders(Bunch({'exists': True, 'fs': 30000}))
            data = loader.load_ap_snippets()
            self.assertTrue(data.pop('exists'))
            for t in mock_times.return_value:
                np.testing.assert_array_equal(data['images'][t], mock_snippet.return_value[0])
            self.assertEqual(data['fs'], 30000)
            # Check channel quality metrics
            self.assertIn('dead_channels', data)
            self.assertIn('noisy_channels_coherence', data)
            self.assertIn('noisy_channels_psd', data)
            self.assertIn('outside_channels', data)

            # Verify structure of quality metrics
            for key in [
                'dead_channels',
                'noisy_channels_coherence',
                'noisy_channels_psd',
                'outside_channels',
            ]:
                self.assertIn('values', data[key])
                self.assertIn('lines', data[key])
                self.assertIn('points', data[key])
            mock_times.assert_called_once()

        # Second time data is loaded from disk so we don't expect the get_time_snippets
        # to be called
        with self.subTest('Load existing raw data snippets'):
            mock_times.reset_mock()
            mock_snippet.reset_mock()
            _ = loader.load_ap_snippets()
            mock_times.assert_not_called()


class TestSpikeGLXLoaderOne(unittest.TestCase):
    def setUp(self):
        self.mock_one = MagicMock()
        self.session_path = Path('/mnt/s0/Data/Subjects/steinmetzlab/KM002/2024-09-16/001')
        self.mock_one.eid2path.return_value = self.session_path

        self.insertion = {'id': uuid.uuid4(), 'session': uuid.uuid4(), 'name': 'probe00'}

        self.loader = SpikeGLXLoaderOne(self.insertion, self.mock_one)

    @patch('ibl_alignment_gui.loaders.data_loader.spikeglx.read_meta_data')
    def test_load_meta_data(self, mock_meta):
        """Test the load_meta_data method"""
        mock_meta.return_value = Bunch({'x': np.arange(10), 'y': np.arange(10)})

        with self.subTest('Meta data exists'):
            self.mock_one.load_dataset.return_value = self.session_path.joinpath(
                '_spikeglx.ap.meta'
            )
            data = self.loader.load_meta_data()
            self.assertIs(data, mock_meta.return_value)

        with self.subTest('Meta data does not exist'):
            self.mock_one.load_dataset.side_effect = ALFObjectNotFound
            data = self.loader.load_meta_data()
            self.assertIsNone(data)

    @unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)
    @patch('brainbox.io.spikeglx.Streamer')
    def test_load_ap_data(self, mock_streamer):
        """Test the load_ap_data method"""
        self.loader.load_ap_data()
        mock_streamer.assert_called_with(
            pid=self.loader.pid, one=self.mock_one, remove_cached=self.loader.force, typ='ap'
        )


class TestSpikeGLXLoaderLocal(unittest.TestCase):
    def setUp(self):
        self.probe_path = Path(
            '/mnt/s0/Data/Subjects/steinmetzlab/KM002/2024-09-16/001/alf/probe00'
        )
        self.loader = SpikeGLXLoaderLocal(self.probe_path.joinpath('meta'))

    @patch('ibl_alignment_gui.loaders.data_loader.spikeglx.read_meta_data')
    def test_load_meta_data(self, mock_meta):
        """Test the load_meta_data method"""
        mock_meta.return_value = Bunch({'x': np.arange(10), 'y': np.arange(10)})

        with (
            self.subTest('Meta data exists'),
            patch.object(
                Path, 'glob', return_value=iter([self.probe_path.joinpath('spikeglx.ap.meta')])
            ),
        ):
            data = self.loader.load_meta_data()
            self.assertEqual(data, mock_meta.return_value)

        with (
            self.subTest('Meta data does not exist'),
            patch.object(Path, 'glob', return_value=iter([])),
        ):
            data = self.loader.load_meta_data()
            self.assertIsNone(data)

    @patch('ibl_alignment_gui.loaders.data_loader.spikeglx.Reader')
    def test_load_ap_data(self, mock_reader):
        """Test the load_ap_data method"""
        mock_reader.return_value = Bunch({'fs': 30000})

        with (
            self.subTest('AP data exists'),
            patch.object(
                Path, 'glob', return_value=iter([self.probe_path.joinpath('spikeglx.ap.cbin')])
            ),
        ):
            data = self.loader.load_ap_data()
            self.assertEqual(data, mock_reader.return_value)

        with (
            self.subTest('AP data does not exist'),
            patch.object(Path, 'glob', return_value=iter([])),
        ):
            data = self.loader.load_ap_data()
            self.assertIsNone(data)


class TestFeatureLoaderOneTables(unittest.TestCase):
    """Reading a feature table, downloading it when it is not complete on disk."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.loader = FeatureLoaderOne({'id': 'pid'}, MagicMock(), '2026_W39')
        self.table = self.root.joinpath(
            'ea_active', '2026_W39', 'agg_full', 'df_all_cols_merged.pqt'
        )
        self.enterContext(patch.object(data_loader, 'EPHYS_ATLAS', True))

    def write_table(self, *args) -> None:
        """Stand-in for a download that writes the table."""
        self.table.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'pid': ['pid'], 'channel': [0], 'rms_lf': [1.0]})
        df.set_index(['pid', 'channel']).to_parquet(self.table)

    def make_folder(self, *args) -> None:
        """Stand-in for a download that finds nothing: the folder is made, no table."""
        self.table.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> pd.DataFrame:
        return self.loader.read_full_dataframe('ea_active', '2026_W39', self.root)

    def test_table_present_not_downloaded(self):
        """A table on disk is read without downloading"""
        self.write_table()
        with patch.object(self.loader, 'download_features') as download:
            df = self.read()
        download.assert_not_called()
        self.assertEqual(df['rms_lf'].tolist(), [1.0])

    def test_empty_folder_downloads_again(self):
        """An empty folder left by a failed download is downloaded again"""
        self.make_folder()
        with patch.object(self.loader, 'download_features', side_effect=self.write_table) as dl:
            df = self.read()
        dl.assert_called_once()
        self.assertEqual(df['rms_lf'].tolist(), [1.0])

    def test_unavailable_table_is_empty(self):
        """When the download finds nothing, the table is empty and a warning is logged"""
        with (
            patch.object(self.loader, 'download_features', side_effect=self.make_folder),
            self.assertLogs('ibl_alignment_gui.loaders.data_loader', 'WARNING') as logs,
        ):
            df = self.read()
        self.assertTrue(df.empty)
        self.assertIn('ea_active 2026_W39 feature table is not available', logs.output[0])

    def test_table_not_on_s3_is_empty(self):
        """A table that is not on S3 (download_tables' assertion) gives no rows, and is retried"""
        missing = AssertionError(
            'aggregates/atlas/features/multi_area_comm/2026_W42 not found on AWS'
        )
        with (
            patch.object(self.loader, 'download_features', side_effect=missing) as download,
            self.assertLogs('ibl_alignment_gui.loaders.data_loader', 'WARNING') as logs,
        ):
            self.assertTrue(self.read().empty)
            self.assertTrue(self.read().empty)
        self.assertEqual(download.call_count, 2)  # nothing kept as a broken download
        self.assertIn('ea_active 2026_W39 feature table is not available', logs.output[0])

    def test_load_features_table_not_on_s3(self):
        """A multi-area insertion whose vintage has no table on S3 gets no features, no error"""
        loader = FeatureLoaderOne({'id': 'pid'}, MagicMock(), '2026_W42', multi_area=True)
        loader.one.cache_dir = self.root
        missing = AssertionError('not found on AWS')
        with (
            patch.object(loader, 'download_features', side_effect=missing) as download,
            self.assertLogs('ibl_alignment_gui.loaders.data_loader', 'WARNING'),
        ):
            features = loader.load_features()
        self.assertFalse(features['exists'])
        # Both projects were asked for: ea_active, then multi_area_comm.
        self.assertEqual(
            [c.args[0] for c in download.call_args_list], ['ea_active', 'multi_area_comm']
        )

    def test_insertion_rows_carry_vintage(self):
        """The rows read for an insertion say which vintage they are"""
        self.write_table()
        df = self.loader.load_dataframe('ea_active', '2026_W39', self.root)
        self.assertEqual(df.attrs['features_version'], '2026_W39')

    def test_other_download_error_raises(self):
        """Any other download failure (e.g. the network) still raises"""
        with (
            patch.object(self.loader, 'download_features', side_effect=ConnectionError('down')),
            self.assertRaises(ConnectionError),
        ):
            self.read()


class TestGetData(unittest.TestCase):
    """The data get_data returns for a session on disk, read locally or through ONE.

    This is what the plots receive; a refactor of the loaders must keep it.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.paths = write_session(self.root)
        self.shank_sites = Bunch(spikes_ind=np.array([0, 1]), raw_ind=np.array([0, 1]))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _expected(self):
        """Return the get_data result for the shank on channels 0 and 1"""
        # Spikes of cluster 1 (channel 2) are off the shank; no cluster is rate filtered
        keep = np.array([0, 2, 3, 5])
        frames = np.arange(15 * 15 * 2, dtype=np.uint8)
        return {
            'spikes': Bunch(
                clusters=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)[keep],
                times=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])[keep],
                depths=np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)[keep],
                amps=(np.arange(1, 7, dtype=np.float32) * 1e-6)[keep],
                exists=True,
            ),
            'clusters': Bunch(
                channels=np.array([0, 2, 1]),
                peakToTrough=np.array([0.1, 0.2, 0.3]),
                waveforms=np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4),
                metrics=pd.DataFrame({'firing_rate': [1.0, 1.0, 0.001], 'label': [1.0, 0.0, 1.0]}),
                exists=True,
            ),
            'channels': Bunch(
                rawInd=np.arange(4),
                localCoordinates=np.c_[np.zeros(4), np.arange(4) * 20.0],
                exists=True,
            ),
            'rms_AP': Bunch(
                rms=np.arange(12.0).reshape(3, 4)[:, [0, 1]],
                timestamps=np.array([1.5, 4.5, 7.5]),
                xaxis='Time (s)',
                exists=True,
            ),
            # Without timestamps the amps are renamed and the x axis is in samples
            'rms_LF': Bunch(
                rms=np.arange(12.0, 24.0).reshape(3, 4)[:, [0, 1]],
                timestamps=np.array([0, 3]),
                xaxis='Time samples',
                exists=True,
            ),
            'psd_LF': Bunch(
                power=np.arange(20.0).reshape(5, 4)[:, [0, 1]],
                freqs=np.arange(5.0),
                exists=True,
            ),
            'rf_map': Bunch(
                times=np.array([0.0, 1.0]),
                frames=np.transpose(np.reshape(frames, [15, 15, -1], order='F'), [2, 1, 0]),
                exists=True,
            ),
            'pass_stim': Bunch(valveOn=np.array([2.0, 3.0]), exists=True),
            'gabor': Bunch(leftGabor=np.array([1.0]), rightGabor=np.array([2.0]), exists=True),
        }

    def assert_data_equal(self, actual, expected):
        """Assert both hold the same objects, keys, values and dtypes"""
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for name, obj in expected.items():
            self.assertEqual(set(actual[name].keys()), set(obj.keys()), name)
            for key, value in obj.items():
                with self.subTest(name=name, key=key):
                    if isinstance(value, pd.DataFrame):
                        pd.testing.assert_frame_equal(actual[name][key], value)
                    elif isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(actual[name][key], value)
                        self.assertEqual(actual[name][key].dtype, value.dtype)
                    else:
                        self.assertEqual(actual[name][key], value)

    def _load_object(self, eid, obj, collection=None, attribute=None, **kwargs):
        """Serve an object from the session on disk, as ONE would"""
        folders = {
            'alf/probe00': self.paths.spike_sorting,
            'raw_ephys_data/probe00': self.paths.processed_ephys,
        }
        if collection is not None:
            folder = folders[Path(collection).as_posix()]
        elif obj == 'RFMapStim':
            folder = self.paths.raw_task
        else:
            folder = self.paths.task
        # ONE ignores attribute=None, while alfio would look for an attribute called None
        kwargs = {'attribute': attribute} if attribute is not None else {}
        return alfio.load_object(folder, obj, **kwargs)

    def test_local(self):
        """DataLoaderLocal returns the session's data for the shank"""
        data = DataLoaderLocal(self.paths).get_data(self.shank_sites)
        self.assert_data_equal(data, self._expected())

    def test_one(self):
        """DataLoaderOne returns the same data as DataLoaderLocal"""
        # DataLoaderOne sets filter=True before DataLoader.__init__ resets it to False, so ONE
        # sessions are not rate filtered either
        one = MagicMock()
        one.load_object.side_effect = self._load_object
        insertion = {'id': uuid.uuid4(), 'session': uuid.uuid4(), 'name': 'probe00'}
        loader = DataLoaderOne(insertion, one, session_path=self.root, spike_collection='')
        self.assert_data_equal(loader.get_data(self.shank_sites), self._expected())

    def test_missing_datasets(self):
        """Without QC or passive datasets those objects are missing and the rest still loads"""
        empty = self.root.joinpath('empty')
        empty.mkdir()
        paths = DatasetPaths(spike_sorting=self.paths.spike_sorting, processed_ephys=empty)
        with self.assertLogs('ibl_alignment_gui.loaders.data_loader', level='WARNING'):
            data = DataLoaderLocal(paths).get_data(self.shank_sites)
        self.assertTrue(data['spikes']['exists'])
        for key in ('rms_AP', 'rms_LF', 'psd_LF', 'rf_map', 'pass_stim', 'gabor'):
            self.assertEqual(data[key], Bunch(exists=False), key)
