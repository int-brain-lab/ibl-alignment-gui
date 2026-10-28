import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from ibl_alignment_gui.handlers.probe_handler import (
    ProbeHandler,
    ProbeHandlerONE,
    find_features_file,
)


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


if __name__ == '__main__':
    unittest.main()
