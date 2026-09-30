import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ibl_alignment_gui.handlers.probe_handler import ProbeHandler, ProbeHandlerONE
from ibl_alignment_gui.loaders.alignment_loader import AlignmentLoaderLocal
from iblutil.util import Bunch


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


class TestLocalAlignmentLayout(unittest.TestCase):
    """Test mapping an insertion onto the folder layout local alignments are held in."""

    def test_local_alignment_folder(self):
        """Test the session folder name is built from the subject, date and probe."""
        folder = ProbeHandler.local_alignment_folder('HB_011', '2024-01-24', 'probe00')
        self.assertEqual(folder, 'HB_011_2024-01-24_probe00')

        # One folder holds every shank, so the shank identifier is stripped from the probe name
        folder = ProbeHandler.local_alignment_folder('HB_011', '2024-01-24', 'probe00c')
        self.assertEqual(folder, 'HB_011_2024-01-24_probe00')

    def test_local_alignment_shank(self):
        """Test the shank number is taken from the letter the probe name ends in."""
        for label, expected in (
            ('probe00', None),
            ('probe00a', 1),
            ('probe00d', 4),
            ('probe01b', 2),
        ):
            with self.subTest(label):
                self.assertEqual(ProbeHandler.local_alignment_shank(label), expected)


class TestSetLocalAlignmentRoots(unittest.TestCase):
    """Test pushing the chosen local alignment folders down to the shanks."""

    def setUp(self):
        self.handler = ProbeHandlerONE.__new__(ProbeHandlerONE)
        self.handler.default_config = 'default'
        self.handler.non_default_config = None
        self.handler.local_alignment_roots = []
        self.handler.shanks = defaultdict(Bunch)
        self.loaders = {}
        for shank in ('probe00a', 'probe00b'):
            loader = MagicMock()
            loader.import_roots = []
            self.loaders[shank] = loader
            self.handler.shanks[shank]['default'] = MagicMock(loaders={'align': loader})

    def test_roots_reach_every_shank(self):
        """Test the folders are set on every shank and the alignments reloaded."""
        roots = [Path('/data/alignments')]
        self.handler.set_local_alignment_roots(roots)

        for loader in self.loaders.values():
            self.assertListEqual(loader.import_roots, roots)
            loader.load_previous_alignments.assert_called_once()

    def test_roots_are_copied(self):
        """Test the handler keeps its own list, so later edits don't leak into the loaders."""
        roots = [Path('/data/alignments')]
        self.handler.set_local_alignment_roots(roots)
        roots.append(Path('/data/other'))

        self.assertListEqual(self.handler.local_alignment_roots, [Path('/data/alignments')])


class TestLocalAlignmentFileShanks(unittest.TestCase):
    """Test finding the shanks that a chosen local alignment file belongs to."""

    def setUp(self):
        self.handler = ProbeHandlerONE.__new__(ProbeHandlerONE)
        self.handler.shanks = defaultdict(Bunch)
        folder = 'HB_011_2024-01-24_probe00'
        for i, shank in enumerate(('probe00a', 'probe00b', 'probe00c')):
            loader = AlignmentLoaderLocal(
                Path('/data'), i, 3, import_folder=folder, import_shank=i + 1
            )
            self.handler.shanks[shank]['default'] = MagicMock(loaders={'align': loader})
        self.folder = Path('/root/jane.doe', folder)

    def test_file_matches_its_own_shank(self):
        """Test a file is matched to the shank it was written for, whichever is selected."""
        for n, shank in ((1, 'probe00a'), (3, 'probe00c')):
            with self.subTest(shank=shank):
                file = self.folder.joinpath(f'prev_alignments_shank{n}.json')
                self.assertListEqual(self.handler.get_local_alignment_file_shanks(file), [shank])

    def test_file_matches_no_shank(self):
        """Test a file for another session, or a shank the probe lacks, matches nothing."""
        for name, file in (
            (
                'other session',
                Path('/root/jane.doe/HB_011_2024-01-25_probe00/prev_alignments_shank1.json'),
            ),
            ('missing shank', self.folder.joinpath('prev_alignments_shank4.json')),
            ('no shank number', self.folder.joinpath('prev_alignments.json')),
        ):
            with self.subTest(name):
                self.assertListEqual(self.handler.get_local_alignment_file_shanks(file), [])


class TestSelectAlignments(unittest.TestCase):
    """Test choosing the alignment of every shank once the local alignments have changed."""

    def setUp(self):
        self.handler = ProbeHandlerONE.__new__(ProbeHandlerONE)
        self.handler.shanks = defaultdict(Bunch)
        self.loaders = {}
        alignments = {
            'probe00a': {
                '2026-08-20T10:00:00_alyx': [[1], [2]],
                '2026-08-13T15:14:15_jane (local)': [[3], [4]],
            },
            'probe00b': {'2026-08-20T10:00:00_alyx': [[5], [6]]},
        }
        for i, (shank, aligns) in enumerate(alignments.items()):
            loader = AlignmentLoaderLocal(Path('/data'), i, 2)
            loader.alignments = dict(aligns)
            loader.get_previous_alignments()
            loader.get_starting_alignment(0)
            self.loaders[shank] = loader
            self.handler.shanks[shank]['default'] = MagicMock(loaders={'align': loader})

    def test_local_shanks_switch_to_local(self):
        """Test shanks with a local alignment switch to it, and the rest are left alone."""
        changed = self.handler.select_alignments(['probe00a', 'probe00b'])

        self.assertListEqual(changed, ['probe00a'])
        self.assertEqual(self.loaders['probe00a'].selected_key, '2026-08-13T15:14:15_jane (local)')
        np.testing.assert_array_equal(self.loaders['probe00a'].feature_prev, [3])
        self.assertEqual(self.loaders['probe00b'].selected_key, '2026-08-20T10:00:00_alyx')

    def test_other_shanks_keep_their_alignment(self):
        """Test shanks not asked to switch keep the alignment they already had."""
        self.assertListEqual(self.handler.select_alignments(['probe00b']), [])
        self.assertEqual(self.loaders['probe00a'].selected_key, '2026-08-20T10:00:00_alyx')

    def test_dropped_alignment_falls_back(self):
        """Test a shank whose chosen alignment is no longer on offer goes back to its start."""
        loader = self.loaders['probe00a']
        self.handler.select_alignments(['probe00a'])
        loader.alignments.pop('2026-08-13T15:14:15_jane (local)')

        self.assertListEqual(self.handler.select_alignments(), ['probe00a'])
        self.assertEqual(loader.selected_key, '2026-08-20T10:00:00_alyx')
