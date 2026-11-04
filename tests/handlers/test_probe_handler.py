import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import spikeglx

from ibl_alignment_gui.handlers.probe_handler import (
    ProbeHandler,
    ProbeHandlerLocal,
    ProbeHandlerLocalYaml,
    ProbeHandlerONE,
    find_features_file,
    find_sorting_analyzer,
    make_data_loader,
    make_geometry_loader,
)
from ibl_alignment_gui.loaders.data_loader import DataLoaderLocal, DataLoaderSI
from ibl_alignment_gui.loaders.geometry_loader import (
    ChannelGeometry,
    ElectrodeGeometry,
    GeometryLoaderLocal,
    GeometryLoaderSI,
)
from ibl_alignment_gui.loaders.sources import RawSourceSpikeGLX
from ibl_alignment_gui.utils.optional import SPIKEINTERFACE_EXTRA_HINT, has_spikeinterface
from ibl_alignment_gui.utils.parse_yaml import DatasetPaths
from iblutil.util import Bunch
from tests.loaders._si_analyzer import write_analyzer

FIXTURE_PATH = Path(__file__).parents[1].joinpath('fixtures')


def make_insertion(subject: str, name: str, pid: str) -> dict:
    """Build a minimal insertion dict matching the Alyx REST response shape."""
    return {
        'id': pid,
        'name': name,
        'session_info': {
            'subject': subject,
            'start_time': '2025-01-15T12:34:56',
            'number': 1,
            'lab': 'testlab',
        },
    }


class TestResolvePid(unittest.TestCase):
    """Test the method ProbeHandlerONE.resolve_pid."""

    def setUp(self):
        # Build a handler without running the heavy __init__ (ONE/atlas/loaders).
        self.model = ProbeHandlerONE.__new__(ProbeHandlerONE)
        self.model.one = MagicMock()

        # Two subjects; subj2 has a multi-shank probe so the shank index is non-trivial.
        self.insertions = [
            make_insertion('subj1', 'probe00', 'pid-subj1'),
            make_insertion('subj2', 'probe00a', 'pid-subj2-a'),
            make_insertion('subj2', 'probe00b', 'pid-subj2-b'),
        ]
        self.model.sess_ins = self.insertions
        self.model.subj_ins = [ins['session_info']['subject'] for ins in self.insertions]
        self.model.subjects = np.unique(self.model.subj_ins)

    def test_resolve_pid_returns_indices(self):
        # Arrange
        target = self.insertions[2]  # subj2, probe00b -> second shank
        self.model.one.alyx.rest.return_value = [target]
        # Act
        with patch.object(ProbeHandlerONE, 'initialise_shanks'):
            subj_idx, sess_idx, shank_idx = self.model.resolve_pid('pid-subj2-b')
        # Assert
        self.assertEqual(self.model.subjects[subj_idx], 'subj2')
        self.assertEqual(self.model.sessions[sess_idx], '2025-01-15 001 probe00')
        self.assertEqual(shank_idx, 1)

    def test_resolve_pid_first_shank(self):
        # Arrange
        target = self.insertions[1]  # subj2, probe00a -> first shank
        self.model.one.alyx.rest.return_value = [target]
        # Act
        with patch.object(ProbeHandlerONE, 'initialise_shanks'):
            subj_idx, sess_idx, shank_idx = self.model.resolve_pid('pid-subj2-a')
        # Assert
        self.assertEqual(shank_idx, 0)

    def test_resolve_pid_unknown_pid_raises(self):
        # Arrange
        self.model.one.alyx.rest.return_value = []
        # Act / Assert
        with self.assertRaises(ValueError):
            self.model.resolve_pid('does-not-exist')

    def test_resolve_pid_subject_without_spikesorting_raises(self):
        # Arrange: insertion exists but its subject is not in the spikesorted subject list.
        target = make_insertion('subj_no_spikes', 'probe00', 'pid-other')
        self.model.one.alyx.rest.return_value = [target]
        # Act / Assert
        with self.assertRaises(ValueError):
            self.model.resolve_pid('pid-other')


class TestHasFeatures(unittest.TestCase):
    """The session has features when any loaded shank has them."""

    def handler(self, loaded=True, exists=True):
        return MagicMock(data_loaded=loaded, raw_data={'features': {'exists': exists}})

    def has_features(self, *handlers):
        shanks = {f'shank_{i}': {'default': h} for i, h in enumerate(handlers)}
        return ProbeHandler.has_features.fget(SimpleNamespace(shanks=shanks))

    def test_has_features(self):
        self.assertTrue(self.has_features(self.handler(exists=False), self.handler()))
        self.assertFalse(self.has_features(self.handler(exists=False)))
        self.assertFalse(self.has_features(self.handler(loaded=False)))
        self.assertFalse(self.has_features())


class TestHasSpikeSorting(unittest.TestCase):
    """The session has spike sorting when any loaded shank has spikes."""

    def has_spike_sorting(self, *exists):
        shanks = {
            f'shank_{i}': {
                'default': MagicMock(data_loaded=True, raw_data={'spikes': {'exists': e}})
            }
            for i, e in enumerate(exists)
        }
        return ProbeHandler.has_spike_sorting.fget(SimpleNamespace(shanks=shanks))

    def test_has_spike_sorting(self):
        self.assertTrue(self.has_spike_sorting(False, True))
        self.assertFalse(self.has_spike_sorting(False))
        self.assertFalse(self.has_spike_sorting())


class TestChannelOrderNotices(unittest.TestCase):
    """RMS and spectra without rawInd are reported when the file is not in depth order."""

    @staticmethod
    def channels(x, y, raw_ind=None):
        """Spike sorting channels with the given coordinates, in raw index order by default"""
        raw_ind = np.arange(len(x)) if raw_ind is None else np.asarray(raw_ind)
        geom = ChannelGeometry(Bunch(localCoordinates=np.c_[x, y], rawInd=raw_ind))
        geom.split_sites_per_shank()
        return geom

    @staticmethod
    def electrodes(np_type):
        """The electrodes of an ap.meta fixture"""
        meta_file = next(FIXTURE_PATH.joinpath(np_type).glob('*.ap.meta'))
        geom = ElectrodeGeometry(RawSourceSpikeGLX(lambda: spikeglx.Reader(meta_file)).geometry)
        geom.split_sites_per_shank()
        return geom

    def shank(self, sites, folder='/ephys', missing=('ephysTimeRmsAP',), from_meta=False):
        geom = SimpleNamespace(
            electrodes=sites if from_meta else None, channels=None if from_meta else sites
        )
        data = SimpleNamespace(ephys_path=Path(folder), objects_without_raw_ind=list(missing))
        return MagicMock(loaders={'geom': geom, 'data': data})

    def notice(self, shanks):
        model = SimpleNamespace(
            shanks={label: {'default': shank} for label, shank in shanks.items()},
            configs=['default'],
        )
        model.is_out_of_depth_order = lambda *args: ProbeHandler.is_out_of_depth_order(
            model, *args
        )
        return ProbeHandler.get_channel_order_notice(model)

    def test_depth_order_not_reported(self):
        """Files in shank and depth order are correct, also with the pairs of a row swapped"""
        cases = {
            'pairs of a row': self.channels([27, 59, 27, 59], [0, 0, 15, 15]),
            'swapped pairs of a row': self.channels([59, 27, 59, 27], [0, 0, 15, 15]),
            'one shank after the other': self.channels([0, 0, 250, 250], [0, 15, 0, 15]),
        }
        for case, sites in cases.items():
            with (
                self.subTest(case=case),
                self.assertNoLogs('ibl_alignment_gui.handlers.probe_handler', 'WARNING'),
            ):
                self.assertIsNone(self.notice({'probe00': self.shank(sites)}))

    def test_out_of_depth_order_reported(self):
        """Files whose channels jump in depth or between shanks are logged and shown"""
        cases = {
            # Blocks of channels out of depth order on one shank, as in an NP2.4 imro table
            'blocks out of order': self.shank(self.channels([27] * 4, [0, 15, 2160, 30])),
            # The channels of the shanks interleaved, from the ap.meta
            'shanks interleaved': self.shank(self.electrodes('NP24'), from_meta=True),
            # An NP2.4 recorded on one shank whose imro table wires the banks out of order
            'one shank, banks out of order': self.shank(self.electrodes('NP21'), from_meta=True),
            # The raw index, not the order of the channels object, gives the file order
            'raw index': self.shank(self.channels([27] * 3, [0, 15, 30], raw_ind=[2, 0, 1])),
        }
        for case, shank in cases.items():
            with (
                self.subTest(case=case),
                self.assertLogs('ibl_alignment_gui.handlers.probe_handler', 'WARNING') as logs,
            ):
                notice = self.notice({'probe00': shank})
                self.assertIn('may be in the wrong order', notice)
                self.assertIn('extract_ephys', logs.output[0])

    def test_one_log_and_notice_for_shared_folder(self):
        """Shanks that share a folder are logged once, listing every object, in one notice"""
        sites = self.electrodes('NP24')
        shanks = {
            'shank_1': self.shank(sites, missing=['ephysTimeRmsAP'], from_meta=True),
            'shank_2': self.shank(
                sites, missing=['ephysTimeRmsAP', 'ephysTimeRmsLF'], from_meta=True
            ),
        }
        with self.assertLogs('ibl_alignment_gui.handlers.probe_handler', 'WARNING') as logs:
            notice = self.notice(shanks)
        self.assertEqual(len(logs.output), 1)
        self.assertIn('ephysTimeRmsAP, ephysTimeRmsLF', logs.output[0])
        self.assertEqual(notice.count(str(Path('/ephys'))), 1)

    def test_files_with_raw_ind_not_reported(self):
        shank = self.shank(self.electrodes('NP24'), missing=(), from_meta=True)
        with self.assertNoLogs('ibl_alignment_gui.handlers.probe_handler', 'WARNING'):
            self.assertIsNone(self.notice({'shank_1': shank}))

    def test_unknown_geometry_not_reported(self):
        with self.assertNoLogs('ibl_alignment_gui.handlers.probe_handler', 'WARNING'):
            self.assertIsNone(self.notice({'probe00': self.shank(None)}))


class TestFindFeaturesFile(unittest.TestCase):
    """A data folder's features file is the one parquet file with 'features' in its name."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)
        # Other parquet files in the folder are not features files
        self.folder.joinpath('clusters.metrics.pqt').touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_none(self):
        self.assertIsNone(find_features_file(self.folder))

    def test_one(self):
        for name in ('ephys_atlas_features_2026_W42.pqt', 'features.parquet'):
            with self.subTest(name=name):
                path = self.folder.joinpath(name)
                path.touch()
                self.assertEqual(find_features_file(self.folder), path)
                path.unlink()

    def test_several(self):
        self.folder.joinpath('features_a.pqt').touch()
        self.folder.joinpath('features_b.pqt').touch()
        with self.assertLogs('ibl_alignment_gui.handlers.probe_handler', 'WARNING') as logs:
            self.assertIsNone(find_features_file(self.folder))
        self.assertIn('features_a.pqt', logs.output[0])


class TestFindSortingAnalyzer(unittest.TestCase):
    """A data folder's SortingAnalyzer is the folder itself or the one analyzer in it."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_analyzer(self, path: Path) -> Path:
        path.mkdir(exist_ok=True)
        path.joinpath('spikeinterface_info.json').touch()
        return path

    def test_none(self):
        self.folder.joinpath('alf').mkdir()
        self.assertIsNone(find_sorting_analyzer(self.folder))

    def test_folder_is_analyzer(self):
        self.make_analyzer(self.folder)
        self.assertEqual(find_sorting_analyzer(self.folder), self.folder)

    def test_one_in_folder(self):
        analyzer = self.make_analyzer(self.folder.joinpath('sorting_analyzer'))
        self.assertEqual(find_sorting_analyzer(self.folder), analyzer)

    def test_several(self):
        self.make_analyzer(self.folder.joinpath('analyzer_a'))
        self.make_analyzer(self.folder.joinpath('analyzer_b'))
        with self.assertLogs('ibl_alignment_gui.handlers.probe_handler', 'WARNING') as logs:
            self.assertIsNone(find_sorting_analyzer(self.folder))
        self.assertIn('analyzer_a', logs.output[0])


class TestMakeLoaders(unittest.TestCase):
    """The local loaders are chosen by the format of the spike sorting."""

    def test_alf(self):
        paths = DatasetPaths(spike_sorting=Path('ks'))
        self.assertIs(type(make_data_loader(paths)), DataLoaderLocal)
        self.assertIs(type(make_geometry_loader(paths)), GeometryLoaderLocal)

    @unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
    def test_spikeinterface(self):
        paths = DatasetPaths(spike_sorting=Path('analyzer'), spike_sorting_format='spikeinterface')
        data = make_data_loader(paths)
        geom = make_geometry_loader(paths)
        self.assertIsInstance(data, DataLoaderSI)
        self.assertIsInstance(geom, GeometryLoaderSI)
        self.assertEqual(data.sorting.path, paths.spike_sorting)
        self.assertEqual(geom.sorting.path, paths.spike_sorting)


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestProbeHandlerLocalSpikeInterface(unittest.TestCase):
    """A data folder with a SortingAnalyzer in it is shown without converting it to ALF."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)
        write_analyzer(self.folder.joinpath('sorting_analyzer'), n_shanks=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_shanks_loaded_and_plotted(self):
        """Each shank of the probe gets its own spikes and plots"""
        model = ProbeHandlerLocal()
        self.assertEqual(model.get_shanks(self.folder), ['shank_1', 'shank_2'])

        for label in model.shank_labels:
            with self.subTest(shank=label):
                shank = model.get_current_shank(label, model.default_config)
                shank.load_data()
                plots = shank.loaders['plots']
                plots.get_plots()

                spikes = shank.raw_data['spikes']
                self.assertTrue(spikes['exists'])
                self.assertGreater(spikes['clusters'].size, 0)
                # Only the spikes of units on this shank are kept
                channels = shank.raw_data['clusters']['channels'][spikes['clusters']]
                np.testing.assert_array_equal(
                    np.isin(channels, shank.chn_sites['spikes_ind']), True
                )
                self.assertTrue(plots.scatter_plots)
                self.assertTrue(plots.image_plots)


class TestProbeHandlerLocalYaml(unittest.TestCase):
    """The configurations of a session yaml are offered by the handler."""

    PROBE = (
        '      probe_00:\n'
        '        datasets:\n'
        '          spike_sorting:\n'
        '            path: {name}/spike_sorting\n'
        '          output:\n'
        '            path: {name}/alignment\n'
    )

    def setUp(self):
        # The yaml's root is the temporary folder, so its paths are absolute on every platform
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def write_yaml(self, body: str) -> Path:
        path = self.root.joinpath('session.yaml')
        path.write_text(f'path: {self.root.as_posix()}\n\n{body}')
        return path

    def test_single_config(self):
        yaml_file = self.write_yaml('probes:\n' + self.PROBE.format(name='single')[2:])
        model = ProbeHandlerLocalYaml(yaml_file)
        self.assertEqual(model.configs, ['default'])
        self.assertEqual(model.possible_configs, ['default'])
        self.assertEqual(model.selected_config, 'default')

    @patch('ibl_alignment_gui.handlers.probe_handler.make_slice_loader')
    def test_single_probe_with_several_shanks(self, mock_slice_loader):
        """The shanks of a single probe read the histology given for that probe"""
        # The metadata of a 4-shank probe recorded on 3 shanks
        meta = next(Path(__file__).parents[1].joinpath('fixtures', 'NP24').glob('*.ap.meta'))
        raw = self.root.joinpath('raw')
        raw.mkdir()
        raw.joinpath(meta.name).write_text(meta.read_text())
        yaml_file = self.write_yaml(
            'probes:\n'
            '  probe00:\n'
            '    datasets:\n'
            '      raw_ephys:\n        path: raw\n'
            '      picks:\n        path: picks\n'
            '      histology:\n        path: hist\n'
            '      output:\n        path: out\n'
        )
        model = ProbeHandlerLocalYaml(yaml_file)

        self.assertEqual(model.get_shanks(None), ['shank_1', 'shank_2', 'shank_3'])
        model.download_histology()
        # The yaml paths are resolved, which on Windows expands short (8.3) folder names
        self.assertEqual(mock_slice_loader.call_args.args[0], self.root.resolve().joinpath('hist'))

    def test_two_configs(self):
        yaml_file = self.write_yaml(
            'configurations:\n'
            '  dense:\n    probes:\n'
            + self.PROBE.format(name='dense')
            + '  sparse:\n    probes:\n'
            + self.PROBE.format(name='sparse')
        )
        model = ProbeHandlerLocalYaml(yaml_file)
        self.assertEqual(model.configs, ['dense', 'sparse'])
        self.assertEqual(model.possible_configs, ['dense', 'sparse', 'both'])
        self.assertEqual(model.default_config, 'dense')
        self.assertEqual(model.non_default_config, 'sparse')
        self.assertEqual(model.selected_config, 'dense')
        self.assertEqual(set(model.data_paths), {'dense', 'sparse'})


if __name__ == '__main__':
    unittest.main()
