import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import one.alf.io as alfio
from ibl_alignment_gui.loaders.sources import SortingSourceSI, raw_channel_indices
from ibl_alignment_gui.utils.optional import SPIKEINTERFACE_EXTRA_HINT, has_spikeinterface
from tests.loaders._si_analyzer import write_analyzer

LOGGER = 'ibl_alignment_gui.loaders.sources'


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestSortingSourceSI(unittest.TestCase):
    """Test the SortingSourceSI class."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = write_analyzer(Path(cls.temp_dir.name).joinpath('analyzer'))

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        self.source = SortingSourceSI(self.path)
        self.analyzer = self.source.analyzer

    def test_spikes_from_spike_vector(self):
        """Spike times are in seconds, amps positive in V and depths of the unit's main channel"""
        spikes = self.source.load_spikes()
        spike_vector = self.analyzer.sorting.to_spike_vector()
        main_channels = self.analyzer.get_main_channels()
        amps = self.analyzer.get_extension('spike_amplitudes').get_data()

        self.assertTrue(spikes['exists'])
        # The plots need the to_df method of the ALF objects
        self.assertIsInstance(spikes, alfio.AlfBunch)
        np.testing.assert_array_equal(spikes['clusters'], spike_vector['unit_index'])
        np.testing.assert_allclose(
            spikes['times'], spike_vector['sample_index'] / self.analyzer.sampling_frequency
        )
        np.testing.assert_allclose(spikes['amps'], np.abs(amps) * 1e-6)
        np.testing.assert_array_equal(
            spikes['depths'],
            self.analyzer.get_channel_locations()[main_channels[spike_vector['unit_index']], 1],
        )

    def test_clusters_on_main_channel(self):
        """Cluster channels are the main channels and waveforms are their templates in V"""
        clusters = self.source.load_clusters()
        main_channels = self.analyzer.get_main_channels()
        templates = self.analyzer.get_extension('templates').get_data()

        self.assertTrue(clusters['exists'])
        np.testing.assert_array_equal(clusters['channels'], main_channels)
        self.assertEqual(clusters['waveforms'].shape, (*templates.shape[:2], 1))
        for unit, channel in enumerate(main_channels):
            with self.subTest(unit=unit):
                np.testing.assert_allclose(
                    clusters['waveforms'][unit, :, 0], templates[unit, :, channel] * 1e-6
                )

    def test_peak_to_trough_from_waveforms(self):
        """Without template metrics the peak to trough is measured on the waveforms in ms"""
        clusters = self.source.load_clusters()
        waveforms = clusters['waveforms'][:, :, 0]
        expected = (
            (np.argmax(waveforms, axis=1) - np.argmin(waveforms, axis=1))
            / self.analyzer.sampling_frequency
            * 1e3
        )
        np.testing.assert_allclose(clusters['peakToTrough'], expected)

    def test_metrics_without_quality_metrics(self):
        """Metrics hold the cluster ids and firing rates, and units are not labelled"""
        with self.assertLogs(LOGGER, level='WARNING') as log:
            metrics = self.source.load_clusters()['metrics']

        n_spikes = np.bincount(self.analyzer.sorting.to_spike_vector()['unit_index'])
        np.testing.assert_array_equal(metrics['cluster_id'], self.analyzer.unit_ids)
        np.testing.assert_allclose(
            metrics['firing_rate'], n_spikes / self.analyzer.get_total_duration()
        )
        self.assertNotIn('label', metrics.columns)
        self.assertIn('Units are not labelled as IBL good', log.output[0])

    def test_no_label_query(self):
        """No query leaves the units unlabelled without a warning"""
        source = SortingSourceSI(self.path, good_units_query=None)
        with self.assertNoLogs(LOGGER, level='WARNING'):
            metrics = source.load_clusters()['metrics']
        self.assertNotIn('label', metrics.columns)

    def test_kilosort_label(self):
        """The Kilosort label of the units is kept as ks2_label"""
        self.analyzer.sorting.set_property('KSLabel', np.array(['good', 'mua', 'good', 'mua']))
        metrics = self.source.load_clusters()['metrics']
        np.testing.assert_array_equal(metrics['ks2_label'], ['good', 'mua', 'good', 'mua'])

    def test_channels(self):
        """Channels are the analyzer's channel locations in channel order"""
        channels = self.source.load_channels()
        self.assertTrue(channels['exists'])
        np.testing.assert_array_equal(channels['rawInd'], np.arange(8))
        np.testing.assert_array_equal(
            channels['localCoordinates'], self.analyzer.get_channel_locations()
        )
        # The generated probe sets no shanks
        self.assertNotIn('shank', channels)

    def test_no_spike_sorting(self):
        """Without an analyzer every object is returned with exists=False"""
        source = SortingSourceSI(None)
        for load in (source.load_spikes, source.load_clusters, source.load_channels):
            with self.subTest(load=load.__name__):
                self.assertEqual(load(), {'exists': False})


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestSortingSourceSIExtensions(unittest.TestCase):
    """The optional and missing extensions of a SortingAnalyzer."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_optional_extensions_used(self):
        """Depths, peak to trough and labels come from the extensions when computed"""
        path = write_analyzer(
            self.folder.joinpath('analyzer'),
            extensions={
                'random_spikes': {},
                'templates': {},
                'spike_amplitudes': {},
                'noise_levels': {},
                'spike_locations': {},
                'template_metrics': {},
                'quality_metrics': {'metric_names': ['amplitude_median', 'firing_rate']},
            },
        )
        source = SortingSourceSI(path, good_units_query='amplitude_median < -50')
        analyzer = source.analyzer
        spikes = source.load_spikes()
        clusters = source.load_clusters()

        locations = analyzer.get_extension('spike_locations').get_data()
        template_metrics = analyzer.get_extension('template_metrics').get_data()
        quality_metrics = analyzer.get_extension('quality_metrics').get_data()

        np.testing.assert_array_equal(spikes['depths'], locations['y'])
        np.testing.assert_allclose(
            clusters['peakToTrough'], template_metrics['peak_to_trough_duration'] * 1e3
        )
        np.testing.assert_allclose(
            clusters['metrics']['firing_rate'], quality_metrics['firing_rate']
        )
        np.testing.assert_array_equal(
            clusters['metrics']['label'], (quality_metrics['amplitude_median'] < -50).astype(int)
        )

    def test_missing_extensions(self):
        """Without the required extensions the spikes and clusters do not exist"""
        path = write_analyzer(self.folder.joinpath('analyzer'), extensions=['random_spikes'])
        source = SortingSourceSI(path)
        with self.assertLogs(LOGGER, level='WARNING') as log:
            spikes = source.load_spikes()
        self.assertIn('templates, spike_amplitudes', log.output[0])

        self.assertEqual(spikes, {'exists': False})
        self.assertEqual(source.load_clusters(), {'exists': False})
        self.assertTrue(source.load_channels()['exists'])

    def test_multi_segment_refused(self):
        """An analyzer with several segments raises an error"""
        path = write_analyzer(self.folder.joinpath('analyzer'), durations=[2.0, 2.0])
        with self.assertRaises(ValueError) as ctx:
            SortingSourceSI(path).load_spikes()
        self.assertIn('2 segments', str(ctx.exception))


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestSortingSourceSIShanks(unittest.TestCase):
    """The shanks of a multi-shank probe."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = write_analyzer(Path(self.temp_dir.name).joinpath('analyzer'), n_shanks=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_channels_carry_probe_shanks(self):
        """The shank of each channel is read from the probe"""
        channels = SortingSourceSI(self.path).load_channels()
        np.testing.assert_array_equal(channels['shank'], ['0'] * 4 + ['1'] * 4)


class TestSortingSourceSIWithoutSpikeInterface(unittest.TestCase):
    """An analyzer cannot be read without SpikeInterface."""

    def test_install_hint(self):
        """Giving an analyzer without SpikeInterface installed says how to install it"""
        with (
            patch('ibl_alignment_gui.loaders.sources.has_spikeinterface', return_value=False),
            self.assertRaises(ImportError) as ctx,
        ):
            SortingSourceSI(Path('analyzer'))
        self.assertIn(SPIKEINTERFACE_EXTRA_HINT, str(ctx.exception))

    def test_no_analyzer(self):
        """Without an analyzer SpikeInterface is not needed"""
        with patch('ibl_alignment_gui.loaders.sources.has_spikeinterface', return_value=False):
            self.assertEqual(SortingSourceSI(None).load_channels(), {'exists': False})


class TestRawChannelIndices(unittest.TestCase):
    """Test the raw_channel_indices function."""

    def test_spikeglx_ids(self):
        """SpikeGLX ids give the channel in the raw file, also when channels were removed"""
        ids = np.array(['imec0.ap#AP0', 'imec0.ap#AP2', 'imec0.ap#AP7'])
        np.testing.assert_array_equal(raw_channel_indices(ids), [0, 2, 7])

    def test_other_ids(self):
        """Other ids give the channel order"""
        np.testing.assert_array_equal(raw_channel_indices(np.array(['3', '5', 'a'])), [0, 1, 2])
