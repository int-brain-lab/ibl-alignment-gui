import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import spikeglx

import one.alf.io as alfio
from ibl_alignment_gui.loaders.data_loader import RawDataLoader
from ibl_alignment_gui.loaders.sources import (
    RAW_CONTACT_IDS_FILE,
    RawSourceSI,
    RawSourceSpikeGLX,
    SortingSourceSI,
    channel_contact_ids,
    raw_channel_indices,
    shank_numbers,
)
from ibl_alignment_gui.utils.optional import SPIKEINTERFACE_EXTRA_HINT, has_spikeinterface
from tests.loaders._si_analyzer import write_analyzer
from tests.loaders._spikeglx_file import write_spikeglx

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

    def test_channels_matched_by_contact(self):
        """The raw channels are found from the contact ids saved with the raw data QC"""
        contact_ids = channel_contact_ids(self.analyzer)
        with tempfile.TemporaryDirectory() as qc_dir:
            qc_path = Path(qc_dir)
            source = SortingSourceSI(self.path, raw_qc_path=qc_path)
            # Without the saved ids, the channels are in the order of the analyzer
            np.testing.assert_array_equal(source.load_channels()['rawInd'], np.arange(8))
            # A raw recording with two more channels, in another order
            raw_contact_ids = np.r_[['x0', 'x1'], contact_ids[::-1]]
            np.save(qc_path.joinpath(RAW_CONTACT_IDS_FILE), raw_contact_ids)
            np.testing.assert_array_equal(source.load_channels()['rawInd'], np.arange(9, 1, -1))

    def test_release(self):
        """A released source frees its analyzer, and loads it again if it is needed"""
        source = SortingSourceSI(self.path)
        source.load_channels()
        self.assertIsNotNone(source._analyzer)
        source.release()
        self.assertIsNone(source._analyzer)
        self.assertTrue(source.load_spikes()['exists'])

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

    def test_contact_ids(self):
        """The contact ids give the channel in the raw file, over the channel ids"""
        ids = np.array(['imec0.ap#AP0', 'imec0.ap#AP1', 'imec0.ap#AP2'])
        raw_ind = raw_channel_indices(
            ids, np.array(['e5', 'e1', 'e3']), np.array(['e0', 'e1', 'e2', 'e3', 'e4', 'e5'])
        )
        np.testing.assert_array_equal(raw_ind, [5, 1, 3])

    def test_contact_not_in_raw(self):
        """A channel whose contact is not in the raw recording is refused"""
        with self.assertRaises(ValueError):
            raw_channel_indices(np.array(['CH1', 'CH2']), np.array(['e0', 'e9']), np.array(['e0']))

    def test_raw_contact_ids_only(self):
        """Without the contact ids of the sorting, the channel ids are used"""
        ids = np.array(['imec0.ap#AP3', 'imec0.ap#AP4'])
        with self.assertLogs(LOGGER, 'WARNING'):
            raw_ind = raw_channel_indices(ids, None, np.array(['e0', 'e1']))
        np.testing.assert_array_equal(raw_ind, [3, 4])


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestChannelContactIds(unittest.TestCase):
    """Test the channel_contact_ids function."""

    def setUp(self):
        import spikeinterface as si  # noqa: PLC0415
        from probeinterface import generate_linear_probe  # noqa: PLC0415

        recording = si.generate_recording(num_channels=4, durations=[0.1])
        probe = generate_linear_probe(num_elec=4)
        probe.set_contact_ids(['e0', 'e1', 'e2', 'e3'])
        # The contacts are wired to the channels in reverse order
        probe.set_device_channel_indices([3, 2, 1, 0])
        recording.set_probe(probe)
        self.recording = recording

    def test_channel_order(self):
        """The ids are in the order of the channels"""
        np.testing.assert_array_equal(
            channel_contact_ids(self.recording), ['e3', 'e2', 'e1', 'e0']
        )

    def test_kept_when_channels_removed(self):
        """The ids of the remaining channels are kept when channels are removed"""
        recording = self.recording.remove_channels([self.recording.channel_ids[1]])
        np.testing.assert_array_equal(channel_contact_ids(recording), ['e3', 'e1', 'e0'])

    def test_no_probe(self):
        """A recording without a probe has no contact ids"""
        import spikeinterface as si  # noqa: PLC0415

        recording = si.NumpyRecording(np.zeros((10, 2), dtype='float32'), sampling_frequency=1)
        self.assertIsNone(channel_contact_ids(recording))


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestRawSourceSI(unittest.TestCase):
    """Test the RawSourceSI class."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def numpy_recording(n_segments=1, shank_ids=None):
        """A recording of 4 channels in 2 columns, the left column on shank 'b' when given"""
        import probeinterface  # noqa: PLC0415
        import spikeinterface as si  # noqa: PLC0415

        traces = [np.arange(400, dtype=np.int16).reshape(100, 4)] * n_segments
        recording = si.NumpyRecording(traces, sampling_frequency=1000)
        recording.set_channel_gains(2.0)
        recording.set_channel_offsets(0.0)
        probe = probeinterface.Probe()
        probe.set_contacts(positions=[[300, 0], [0, 0], [300, 20], [0, 20]])
        if shank_ids is not None:
            probe.set_shank_ids(shank_ids)
        probe.set_device_channel_indices(np.arange(4))
        recording.set_probe(probe)
        return recording

    def test_matches_spikeglx_source(self):
        """A SpikeGLX file read with SpikeInterface gives the same data and snippets"""
        import spikeinterface.extractors as se  # noqa: PLC0415

        for np_type in ('NP21', 'NP24'):
            with self.subTest(np_type=np_type):
                bin_file = write_spikeglx(self.folder.joinpath(np_type), np_type)
                sglx = RawSourceSpikeGLX(lambda bin_file=bin_file: spikeglx.Reader(bin_file))
                source = RawSourceSI(
                    lambda bin_file=bin_file: se.read_spikeglx(
                        bin_file.parents[1], stream_id='imec0.ap'
                    )
                )

                self.assertEqual(source.fs, sglx.fs)
                self.assertAlmostEqual(source.duration, sglx.duration)
                np.testing.assert_allclose(
                    source.get_traces(0, 3000), sglx.get_traces(0, 3000), rtol=1e-6
                )
                np.testing.assert_array_equal(source.geometry['shank'], sglx.geometry['shank'])
                np.testing.assert_allclose(
                    source.geometry['sample_shift'], sglx.geometry['sample_shift']
                )

                kwargs = {'similarity_threshold': (-0.5, 1), 'psd_hf_threshold': 0.02}
                raw, labels, _ = RawDataLoader._get_ap_snippet(source, 0, 1, **kwargs)
                raw_sglx, labels_sglx, _ = RawDataLoader._get_ap_snippet(sglx, 0, 1, **kwargs)
                np.testing.assert_allclose(raw, raw_sglx, atol=1e-12)
                np.testing.assert_array_equal(labels, labels_sglx)

    def test_recording_opened_once_when_read(self):
        """The recording is not opened until it is needed, then reused"""
        open_recording = MagicMock(return_value=self.numpy_recording())
        source = RawSourceSI(open_recording)
        open_recording.assert_not_called()
        self.assertEqual(source.fs, 1000)
        self.assertAlmostEqual(source.duration, 0.1)
        open_recording.assert_called_once()

    def test_traces_in_volts(self):
        """Traces are scaled to V, and a block past the end stops at the last sample"""
        source = RawSourceSI(self.numpy_recording)
        traces = source.get_traces(98, 10)
        self.assertEqual(traces.shape, (2, 4))
        np.testing.assert_allclose(traces[0], np.arange(392, 396) * 2e-6)

    def test_geometry_from_probe(self):
        """Positions come from the probe, shanks are numbered by position, no shift is zero"""
        source = RawSourceSI(lambda: self.numpy_recording(shank_ids=['a', 'b', 'a', 'b']))
        geometry = source.geometry
        np.testing.assert_array_equal(geometry['x'], [300, 0, 300, 0])
        np.testing.assert_array_equal(geometry['y'], [0, 0, 20, 20])
        # Shank 'b' is on the left
        np.testing.assert_array_equal(geometry['shank'], [1, 0, 1, 0])
        np.testing.assert_array_equal(geometry['sample_shift'], np.zeros(4))

    def test_multi_segment_refused(self):
        """Recordings with several segments are refused"""
        source = RawSourceSI(lambda: self.numpy_recording(n_segments=2))
        with self.assertRaises(ValueError):
            _ = source.fs


class TestShankNumbers(unittest.TestCase):
    """Test the shank_numbers function."""

    def test_numbered_by_position(self):
        """Shank ids are numbered from the leftmost shank"""
        x = np.array([500, 0, 250, 0])
        ids = np.array(['10', '2', '1', '2'])
        np.testing.assert_array_equal(shank_numbers(ids, x), [2, 0, 1, 0])

    def test_no_shank_ids(self):
        """Without shank ids every channel is on shank 0"""
        x = np.arange(3.0)
        np.testing.assert_array_equal(shank_numbers(None, x), np.zeros(3))
        np.testing.assert_array_equal(shank_numbers(np.array(['', '', '']), x), np.zeros(3))
