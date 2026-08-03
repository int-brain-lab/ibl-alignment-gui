from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from iblutil.util import Bunch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from ibl_alignment_gui.backends.allen.docdb_api import DocDB
    from one.api import ONE


class AlignmentLoader(ABC):
    """
    Abstract base class for loading xyz picks and previous alignments.

    Subclasses must implement the abstract `load_alignments` and `load_xyz_picks` methods.

    Parameters
    ----------
    user : str or None
        Username string used for tagging alignments.
    xyz_picks : np.ndarray or None
        Pre-loaded xyz_picks. If None, it will be loaded using `load_xyz_picks`.
    data_path : Path or None
        The path to the folder that work in progress alignments are saved to. If None, no saved
        progress is loaded.
    shank_idx : int
        Index of the shank (0-based).
    n_shanks : int
        Total number of shanks.
    """

    def __init__(
        self,
        user: str | None = None,
        xyz_picks: np.ndarray | None = None,
        data_path: Path | None = None,
        shank_idx: int = 0,
        n_shanks: int = 1,
    ) -> None:
        # Set before the xyz picks are loaded, as they may be read from the data path
        self.data_path: Path | None = data_path
        self.shank_idx: int = shank_idx
        self.n_shanks: int = n_shanks

        self.user: str | None = user
        self.xyz_picks: np.ndarray | None = (
            self.load_xyz_picks() if xyz_picks is None else xyz_picks
        )

        self.alignments: Bunch | dict = Bunch()
        self.alignment_keys: list = ['original']
        self.feature_prev: np.ndarray | None = None
        self.track_prev: np.ndarray | None = None
        self.stored_alignment_key: str | None = None

    @abstractmethod
    def load_alignments(self) -> dict[str, Any] | None:
        """Load previously saved alignments."""

    @abstractmethod
    def load_xyz_picks(self) -> np.ndarray | None:
        """Load xyz picks."""

    def load_previous_alignments(self) -> list[str]:
        """
        Load previous alignments into memory.

        Any work in progress that has been saved is loaded alongside them, so that it is
        recovered whenever the previous alignments are refreshed.

        Returns
        -------
        list of str
            Sorted alignment keys including 'original'.
        """
        data = self.load_alignments()
        if data:
            self.alignments = data

        self.load_progress()

        return self.get_previous_alignments()

    def load_progress(self) -> None:
        """
        Load a saved work in progress alignment and add it to the available alignments.

        The alignment is added under a key that marks it as recovered, so that it can be chosen
        from the alignment dropdown but can't be mistaken for an alignment that has been
        uploaded. Nothing is added if there is no saved progress.
        """
        # Drop any alignment recovered previously, so that what is loaded always reflects what
        # is currently saved to file, even if the alignments haven't been reloaded
        for key in [key for key in self.alignments if str(key).startswith('recovered')]:
            self.alignments.pop(key)

        if self.data_path is None:
            return

        progress_name = (
            'alignment_progress.json'
            if self.n_shanks == 1
            else f'alignment_progress_shank{self.shank_idx + 1}.json'
        )

        progress = self._load_json_file(self.data_path.joinpath(progress_name))

        if not progress:
            return

        key = f'recovered ({progress["saved"]})'
        self.alignments[key] = [progress['feature'], progress['track']]

    @property
    def recovered_key(self) -> str | None:
        """
        Return the key that a recovered work in progress alignment is stored under.

        Derived from the alignments rather than remembered, so that it stays correct when the
        alignments are copied between the loaders of different configurations. The most recently
        saved one is used if there is more than one.

        Returns
        -------
        str or None
            The key of the recovered alignment, or None if there isn't one.
        """
        keys = [key for key in self.alignments if str(key).startswith('recovered')]

        return max(keys) if keys else None

    @property
    def uploadable_alignments(self) -> dict[str, Any]:
        """
        Return the alignments that can be uploaded.

        Recovered alignments are left out, as they are a local record of work in progress and
        must never be saved alongside the alignments that have been uploaded. Any recovered
        alignment is excluded, not just this loader's own, as alignments are copied between the
        loaders of different configurations.

        Returns
        -------
        dict
            The alignments, excluding any recovered alignment.
        """
        return {
            key: val
            for key, val in self.alignments.items()
            if not str(key).startswith('recovered')
        }

    def get_previous_alignments(self) -> list[str]:
        """
        Return all available alignment keys sorted in reverse order.

        Returns
        -------
        self.alignments: list of str
            Alignment keys including 'original'.
        """
        self.alignment_keys = [*self.alignments.keys()]
        self.alignment_keys = sorted(self.alignment_keys, reverse=True)
        self.alignment_keys.append('original')

        return self.alignment_keys

    def get_starting_alignment(self, idx: int) -> None:
        """
        Set the starting alignment based on the selected index.

        Parameters
        ----------
        idx : int
            Index in alignment_keys.
        """
        start_lims = 6000 / 1e6
        if self.alignment_keys[idx] == 'original':
            self.feature_prev = np.array([-1 * start_lims, start_lims])
            self.track_prev = np.array([-1 * start_lims, start_lims])
        else:
            self.feature_prev = np.array(self.alignments[self.alignment_keys[idx]][0])
            self.track_prev = np.array(self.alignments[self.alignment_keys[idx]][1])

    def get_stored_alignment_idx(self) -> int:
        """
        Return the index of the stored (resolved) alignment in the alignment keys list.

        If no stored alignment is set or the stored key is not present in the current
        alignment keys, returns 0 (i.e. the most recent alignment).

        Returns
        -------
        int
            Index of the stored alignment in ``self.alignment_keys``, or 0 if not found.
        """
        if (
            self.stored_alignment_key is None
            or self.stored_alignment_key not in self.alignment_keys
        ):
            return 0
        return self.alignment_keys.index(self.stored_alignment_key)

    def get_start_alignment_idx(self) -> int:
        """
        Return the index of the alignment to display when the data is first loaded.

        A recovered alignment takes precedence, so that work saved before a crash is shown,
        otherwise the stored alignment is used.

        Returns
        -------
        int
            Index of the alignment in ``self.alignment_keys``.
        """
        if self.recovered_key is not None and self.recovered_key in self.alignment_keys:
            return self.alignment_keys.index(self.recovered_key)

        return self.get_stored_alignment_idx()

    @staticmethod
    def _load_json_file(file: Path) -> dict[str, Any] | None:
        """
        Load JSON content from a file.

        Parameters
        ----------
        file : Path
            The path to the JSON file.

        Returns
        -------
        dict or None
            Parsed JSON content, or None if file does not exist.
        """
        if file.exists():
            with open(file) as f:
                return json.load(f)

        return None

    def add_extra_alignments(self, extra_alignments: dict[str, Any]) -> list[str]:
        """
        Add additional alignment data.

        Parameters
        ----------
        extra_alignments : dict
            Dictionary of new alignments to add.

        Returns
        -------
        list of str
            Updated alignment keys.
        """
        extra_align = Bunch()
        for key, val in extra_alignments.items():
            if len(key) == 19 and self.user:
                extra_align[f'{key}_{self.user}'] = val
            else:
                extra_align[key] = val

        if self.alignments:
            self.alignments.update(extra_align)
        else:
            self.alignments = extra_align

        return self.get_previous_alignments()


class AlignmentLoaderOne(AlignmentLoader):
    """
    Alignment loader using ONE.

    xyz picks and previous alignments are loaded from the Alyx database.

    Parameters
    ----------
    insertion : dict
        Dictionary representing a probe insertion, must contain a 'json' key.
    one : ONE
        An ONE instance used to query the Alyx database.
    user : str or None
        Username for tagging alignments.
    data_path : Path or None
        The path to the folder that work in progress alignments are saved to, normally the folder
        containing the spike sorting data.
    """

    def __init__(
        self,
        insertion: dict,
        one: ONE,
        user: str | None = None,
        data_path: Path | None = None,
    ):
        self.insertion: dict[str, Any] = insertion
        self.one: ONE = one
        self.traj_id: str | None = None

        super().__init__(user=user, data_path=data_path)

        self.stored_alignment_key: str | None = (
            insertion['json'].get('extended_qc', {}).get('alignment_stored')
        )

    def load_xyz_picks(self) -> np.ndarray | None:
        """
        Load xyz picks from the insertion JSON field.

        Returns
        -------
        np.ndarray or None
            The xyz picks as a (N, 3) array in m, or None if not available.
        """
        xyz_picks = self.insertion['json'].get('xyz_picks', None)
        return np.array(xyz_picks) / 1e6 if xyz_picks is not None else None

    def load_alignments(self) -> dict[str, Any] | None:
        """
        Load previous alignments from the Alyx database.

        Returns
        -------
        dict or None
            Dictionary of alignments, or None if not found.
        """
        traj = self.one.alyx.rest(
            'trajectories',
            'list',
            probe_insertion=self.insertion['id'],
            provenance='Ephys aligned histology track',
            no_cache=True,
        )
        if traj:
            return traj[0]['json']

    def load_trajectory(self) -> None:
        """Load the histology track trajectory and stores the trajectory id."""
        hist = self.one.alyx.rest(
            'trajectories',
            'list',
            probe_insertion=self.insertion['id'],
            provenance='Histology track',
        )

        if hist and hist[0]['x'] is not None:
            self.traj_id = hist[0]['id']


class AlignmentLoaderLocal(AlignmentLoader):
    """
    Alignment loader using local file system.

    xyz picks and previous alignments are loaded from files on disk.

    For single-shank data, expected filenames:
        - *xyz_picks.json
        - prev_alignments.json

    For multi-shank data, expected filenames:
        - *xyz_picks_shank<N>.json
        - prev_alignments_shank<N>.json

    Parameters
    ----------
    data_path : Path
        The path to the local data folder.
    shank_idx : int
        Index of the shank (0-based).
    n_shanks : int
        Total number of shanks.
    user : str or None
        Username for tagging alignments.
    xyz_picks : np.ndarray or None
        Preloaded xyz picks. If not provided, it will attempt to load from file.
    """

    def __init__(
        self,
        data_path: Path,
        shank_idx: int,
        n_shanks: int,
        user: str | None = None,
        xyz_picks: np.ndarray | None = None,
        histology_space: str = 'ccf',
    ):
        self.histology_space: str = histology_space

        super().__init__(
            user=user,
            xyz_picks=xyz_picks,
            data_path=data_path,
            shank_idx=shank_idx,
            n_shanks=n_shanks,
        )

    def load_xyz_picks(self) -> np.ndarray | None:
        """
        Load xyz picks from local file.

        Returns
        -------
        np.ndarray or None
            The xyz picks as a (N, 3) array in m, or None if not found.
        """
        space = '_image_space' if self.histology_space != 'ccf' else ''
        xyz_name = (
            f'*xyz_picks{space}.json'
            if self.n_shanks == 1
            else f'*xyz_picks{space}_shank{self.shank_idx + 1}.json'
        )

        xyz_file = sorted(self.data_path.glob(xyz_name))

        if len(xyz_file) == 0:
            return

        user_picks = self._load_json_file(xyz_file[0])
        return np.array(user_picks['xyz_picks']) / 1e6

    def load_alignments(self) -> dict[str, Any] | None:
        """
        Load previous alignment data from local file.

        Returns
        -------
        dict or None
            Dictionary of alignment data or None if file not found.
        """
        prev_align_name = (
            'prev_alignments.json'
            if self.n_shanks == 1
            else f'prev_alignments_shank{self.shank_idx + 1}.json'
        )

        prev_align_file = self.data_path.joinpath(prev_align_name)

        return self._load_json_file(prev_align_file)


class AlignmentLoaderDocDB(AlignmentLoaderLocal):
    """
    Alignment loader using the Allen Neural Dynamics DocDB.

    Used by the Allen/Code Ocean (anatomical) workflow when the DocDB option is enabled.
    xyz picks are always read from the local file system (inherited from
    :class:`AlignmentLoaderLocal`); previous alignments are read from the DocDB QC evaluation
    for this session/probe/shank, falling back to the local ``prev_alignments.json`` when DocDB
    has no matching record or is unreachable.

    The session and probe names are derived from ``data_path`` to match how they are written by
    :class:`~ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderDocDB`:
    ``session = data_path.parent.stem`` and ``probe = data_path.stem``.

    Parameters
    ----------
    data_path : Path
        The path to the local data folder.
    shank_idx : int
        Index of the shank (0-based).
    n_shanks : int
        Total number of shanks.
    docdb : DocDB
        The DocDB client used to read previous alignments (injected, analogous to ``one``).
    user : str or None
        Username for tagging alignments.
    xyz_picks : np.ndarray or None
        Preloaded xyz picks. If not provided, it will attempt to load from file.
    """

    def __init__(
        self,
        data_path: Path,
        shank_idx: int,
        n_shanks: int,
        docdb: DocDB,
        user: str | None = None,
        xyz_picks: np.ndarray | None = None,
        use_db: bool = True,
        histology_space: str = 'ccf',
    ):
        self.docdb: DocDB = docdb
        self.use_db = use_db
        super().__init__(
            data_path,
            shank_idx,
            n_shanks,
            user=user,
            xyz_picks=xyz_picks,
            histology_space=histology_space,
        )

    def load_alignments(self) -> dict[str, Any] | None:
        """
        Load previous alignment data from DocDB, falling back to the local file.

        Returns
        -------
        dict or None
            Dictionary of alignment data from DocDB, the local file if DocDB has no matching
            record, or None if neither is available.
        """
        if self.use_db:
            session_name = self.data_path.parent.stem
            probe = self.data_path.stem
            try:
                alignments = self.docdb.load_alignments(session_name, probe, self.shank_idx)
            except ValueError as err:
                logger.warning(
                    f'Failed to load previous alignments from docdb ({err}). '
                    'Falling back to local file.'
                )
                alignments = None

            if alignments is None:
                alignments = super().load_alignments()
        else:
            alignments = super().load_alignments()

        return alignments
