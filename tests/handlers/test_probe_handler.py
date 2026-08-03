import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ibl_alignment_gui.handlers.probe_handler import ProbeHandlerONE


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


if __name__ == '__main__':
    unittest.main()
