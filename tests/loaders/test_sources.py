import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ibl_alignment_gui.loaders.sources import (
    AlfReader,
    AlfReaderLocal,
    AlfReaderOne,
    EphysQCSourceAlf,
    PassiveSourceAlf,
    RawSourceSpikeGLX,
    SortingSourceAlf,
    load_alf,
)
from iblutil.util import Bunch
from one.alf.exceptions import ALFMultipleCollectionsFound, ALFObjectNotFound
from tests.loaders._alf_session import write_session

LOGGER = 'ibl_alignment_gui.loaders.sources'


class _SessionTest(unittest.TestCase):
    """Write the test session to a temporary folder for each test."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paths = write_session(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()


class TestLoadAlf(unittest.TestCase):
    """Test the load_alf function."""

    def test_found_object_gets_exists_flag(self):
        """The loaded object is returned with exists=True"""
        data = Bunch(rms=np.ones(3))
        result = load_alf(MagicMock(return_value=data), 'eid', 'ephysTimeRmsAP')
        self.assertIs(result, data)
        self.assertTrue(result['exists'])

    def test_missing_object_logs_and_returns_not_exists(self):
        """A missing object logs a warning naming it and returns exists=False"""
        func = MagicMock(side_effect=ALFObjectNotFound)
        with self.assertLogs(LOGGER, level='WARNING') as log:
            result = load_alf(func, 'eid', 'ephysTimeRmsAP')
        self.assertEqual(result, Bunch(exists=False))
        self.assertIn('ephysTimeRmsAP data was not found', log.output[0])

    def test_multiple_collections_logged(self):
        """An object in several collections logs that it cannot be disambiguated"""
        func = MagicMock(side_effect=ALFMultipleCollectionsFound)
        with self.assertLogs(LOGGER, level='WARNING') as log:
            result = load_alf(func, 'eid', 'spikes')
        self.assertFalse(result['exists'])
        self.assertIn('matches multiple collections', log.output[0])

    def test_custom_message_logged(self):
        """A given message replaces the default warning"""
        func = MagicMock(side_effect=ALFObjectNotFound)
        with self.assertLogs(LOGGER, level='WARNING') as log:
            load_alf(func, 'eid', 'spikes', raise_message='custom message')
        self.assertIn('custom message', log.output[0])

    def test_raise_error_reraises(self):
        """With raise_error the exception is raised after logging"""
        func = MagicMock(side_effect=ALFObjectNotFound)
        with self.assertLogs(LOGGER, level='WARNING'), self.assertRaises(ALFObjectNotFound):
            load_alf(func, 'eid', 'spikes', raise_error=True)


class TestAlfReaderLocal(_SessionTest):
    """Test the AlfReaderLocal class."""

    def test_load_selected_attributes(self):
        """Only the requested attributes are loaded"""
        data = AlfReaderLocal(self.paths.spike_sorting).load('spikes', attribute=['times'])
        self.assertEqual(set(data.keys()), {'times', 'exists'})
        np.testing.assert_array_equal(data['times'], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_load_all_attributes(self):
        """Without attributes the whole object is loaded, whatever its namespace"""
        data = AlfReaderLocal(self.paths.processed_ephys).load('ephysTimeRmsAP')
        self.assertEqual(set(data.keys()), {'rms', 'timestamps', 'exists'})

    def test_missing_object(self):
        """A missing object returns exists=False"""
        with self.assertLogs(LOGGER, level='WARNING'):
            data = AlfReaderLocal(self.paths.task).load('spikes')
        self.assertEqual(data, Bunch(exists=False))


class TestAlfReaderOne(unittest.TestCase):
    """Test the AlfReaderOne class."""

    def setUp(self):
        self.one = MagicMock()
        self.one.load_object.return_value = Bunch(times=np.arange(3))

    def test_collection_and_attributes_passed(self):
        """The collection and attributes are passed to ONE"""
        reader = AlfReaderOne(self.one, 'eid', collection='alf/probe00')
        data = reader.load('spikes', attribute=['times'])
        self.one.load_object.assert_called_once_with(
            'eid', 'spikes', collection='alf/probe00', attribute=['times']
        )
        self.assertTrue(data['exists'])

    def test_unset_arguments_left_out(self):
        """Without a collection or attributes ONE searches the session for the whole object"""
        AlfReaderOne(self.one, 'eid').load('passiveRFM')
        self.one.load_object.assert_called_once_with('eid', 'passiveRFM')

    def test_missing_object(self):
        """A missing object returns exists=False"""
        self.one.load_object.side_effect = ALFObjectNotFound
        with self.assertLogs(LOGGER, level='WARNING'):
            data = AlfReaderOne(self.one, 'eid').load('passiveRFM')
        self.assertEqual(data, Bunch(exists=False))


class TestSortingSourceAlf(_SessionTest):
    """Test the SortingSourceAlf class."""

    def test_objects_loaded_with_their_attributes(self):
        """Spikes, clusters and channels load the attributes the plots use"""
        source = SortingSourceAlf(AlfReaderLocal(self.paths.spike_sorting))
        expected = {
            source.load_spikes: {'depths', 'amps', 'times', 'clusters'},
            source.load_clusters: {'metrics', 'peakToTrough', 'waveforms', 'channels'},
            source.load_channels: {'rawInd', 'localCoordinates'},
        }
        for load, attributes in expected.items():
            with self.subTest(load=load.__name__):
                self.assertEqual(set(load().keys()), attributes | {'exists'})

    def test_no_spike_sorting(self):
        """Without a reader every object has exists=False"""
        source = SortingSourceAlf(None)
        for load in (source.load_spikes, source.load_clusters, source.load_channels):
            with self.subTest(load=load.__name__):
                self.assertEqual(load(), Bunch(exists=False))


class TestEphysQCSourceAlf(unittest.TestCase):
    """Test the EphysQCSourceAlf class."""

    def _source(self, data):
        reader = MagicMock(spec=AlfReader)
        reader.load.return_value = data
        return EphysQCSourceAlf(reader)

    def test_rms_with_timestamps(self):
        """RMS with timestamps keeps them and labels the x axis in seconds"""
        data = self._source(
            Bunch(rms=np.ones((3, 2)), timestamps=np.arange(3.0), exists=True)
        ).load_rms('ephysTimeRmsAP')
        np.testing.assert_array_equal(data['timestamps'], np.arange(3.0))
        self.assertEqual(data['xaxis'], 'Time (s)')

    def test_rms_amps_without_timestamps(self):
        """Older RMS files have amps renamed to rms and an x axis in samples"""
        data = self._source(Bunch(amps=np.ones((3, 2)), exists=True)).load_rms('ephysTimeRmsLF')
        self.assertNotIn('amps', data)
        np.testing.assert_array_equal(data['rms'], np.ones((3, 2)))
        np.testing.assert_array_equal(data['timestamps'], [0, 3])
        self.assertEqual(data['xaxis'], 'Time samples')

    def test_rms_amps_with_timestamps(self):
        """Older RMS files with timestamps have amps renamed and keep the seconds axis"""
        data = self._source(
            Bunch(amps=np.ones((3, 2)), timestamps=np.arange(3.0), exists=True)
        ).load_rms('ephysTimeRmsAP')
        self.assertEqual(set(data.keys()), {'rms', 'timestamps', 'xaxis', 'exists'})
        self.assertEqual(data['xaxis'], 'Time (s)')

    def test_psd_amps_renamed(self):
        """Older power files have amps renamed to power"""
        data = self._source(
            Bunch(amps=np.ones((5, 2)), freqs=np.arange(5.0), exists=True)
        ).load_psd('ephysSpectralDensityLF')
        self.assertEqual(set(data.keys()), {'power', 'freqs', 'exists'})

    def test_missing_objects_unchanged(self):
        """Missing objects return exists=False without renaming"""
        source = self._source(Bunch(exists=False))
        self.assertEqual(source.load_rms('ephysTimeRmsAP'), Bunch(exists=False))
        self.assertEqual(source.load_psd('ephysSpectralDensityLF'), Bunch(exists=False))

    def test_any_object_name(self):
        """The object name is passed to the reader, so other RMS objects can be loaded"""
        source = self._source(Bunch(exists=False))
        source.load_rms('ephysTimeRmsAPMain')
        source.reader.load.assert_called_once_with('ephysTimeRmsAPMain')

    def test_no_processed_ephys(self):
        """Without a reader both objects have exists=False"""
        source = EphysQCSourceAlf(None)
        self.assertEqual(source.load_rms('ephysTimeRmsAP'), Bunch(exists=False))
        self.assertEqual(source.load_psd('ephysSpectralDensityLF'), Bunch(exists=False))


class TestPassiveSourceAlf(_SessionTest):
    """Test the PassiveSourceAlf class."""

    def setUp(self):
        super().setUp()
        self.source = PassiveSourceAlf(
            AlfReaderLocal(self.paths.task), AlfReaderLocal(self.paths.raw_task)
        )

    def test_rf_map_frames_reshaped(self):
        """The RF map stimulus is read as (n_frames, 15, 15) frames"""
        data = self.source.load_rf_map()
        self.assertTrue(data['exists'])
        self.assertEqual(data['frames'].shape, (2, 15, 15))
        np.testing.assert_array_equal(data['times'], [0.0, 1.0])

    def test_rf_map_without_stimulus(self):
        """Without the raw RF map stimulus the RF map is missing"""
        source = PassiveSourceAlf(AlfReaderLocal(self.paths.task), None)
        with self.assertLogs(LOGGER, level='WARNING'):
            self.assertEqual(source.load_rf_map(), Bunch(exists=False))

    def test_stims(self):
        """The passive stimulus times are loaded"""
        np.testing.assert_array_equal(self.source.load_stims()['valveOn'], [2.0, 3.0])

    def test_gabor_split_by_side(self):
        """Gabors are split by position, keeping those with a contrast above 0.1"""
        data = self.source.load_gabor()
        np.testing.assert_array_equal(data['leftGabor'], [1.0])
        np.testing.assert_array_equal(data['rightGabor'], [2.0])

    def test_gabor_unreadable(self):
        """A gabor object without the expected attributes logs and is missing"""
        with (
            patch.object(AlfReaderLocal, 'load', return_value=Bunch(start=[1.0], exists=True)),
            self.assertLogs(LOGGER, level='WARNING') as log,
        ):
            data = self.source.load_gabor()
        self.assertEqual(data, Bunch(exists=False))
        self.assertIn('Failed to process passiveGabor', log.output[0])

    def test_no_passive_data(self):
        """Without readers every object is missing, without warnings"""
        source = PassiveSourceAlf(None, None)
        with self.assertNoLogs(LOGGER, level='WARNING'):
            for load in (source.load_rf_map, source.load_stims, source.load_gabor):
                with self.subTest(load=load.__name__):
                    self.assertEqual(load(), Bunch(exists=False))


class TestRawSourceSpikeGLX(unittest.TestCase):
    """Test the RawSourceSpikeGLX class."""

    def setUp(self):
        self.reader = MagicMock()
        self.reader.fs = 30000
        self.reader.nc = 5
        self.reader.nsync = 1
        self.reader.meta = {'fileTimeSecs': 600}
        # The reader sorts the channels by position: its column k is file channel order[k]
        self.order = np.array([1, 0, 3, 2])
        self.reader.raw_channel_order = np.r_[self.order, 4]
        self.reader.geometry = {'x': self.order * 10.0, 'shank': np.zeros(4)}
        self.reader.__getitem__.return_value = np.tile(self.order, (50, 1))
        self.open_reader = MagicMock(return_value=self.reader)
        self.source = RawSourceSpikeGLX(self.open_reader)

    def test_reader_opened_once_when_read(self):
        """The reader is not opened until it is needed, then reused"""
        self.open_reader.assert_not_called()
        self.assertEqual(self.source.fs, 30000)
        self.assertEqual(self.source.duration, 600)
        _ = self.source.geometry
        self.source.get_traces(0, 50)
        self.open_reader.assert_called_once()

    def test_traces_in_file_order_without_sync(self):
        """A block of samples is read on every channel but the sync channel, in file order"""
        traces = self.source.get_traces(100, 50)
        self.reader.__getitem__.assert_called_once_with((slice(100, 150), slice(None, -1)))
        np.testing.assert_array_equal(traces[0], [0, 1, 2, 3])

    def test_geometry_in_file_order(self):
        """The reader's sorted geometry is put back in file order"""
        np.testing.assert_array_equal(self.source.geometry['x'], [0, 10, 20, 30])
        np.testing.assert_array_equal(self.source.geometry['shank'], np.zeros(4))

    def test_reader_without_order(self):
        """A reader that does not sort its channels is read as it is"""
        del self.reader.raw_channel_order
        self.reader.geometry = {'x': np.arange(4.0)}
        np.testing.assert_array_equal(self.source.geometry['x'], np.arange(4))
        np.testing.assert_array_equal(self.source.get_traces(0, 50)[0], self.order)
