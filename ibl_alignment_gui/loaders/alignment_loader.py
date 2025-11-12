import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from iblutil.util import Bunch
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
    """

    def __init__(self, user: str | None = None, xyz_picks: np.ndarray | None = None) -> None:

        self.user: str | None = user
        self.xyz_picks: np.ndarray | None = self.load_xyz_picks() \
            if xyz_picks is None else xyz_picks

        self.alignments: Bunch | dict = Bunch()
        self.alignment_keys: list = ['original']
        self.feature_prev: np.ndarray | None = None
        self.track_prev: np.ndarray | None = None

    @abstractmethod
    def load_alignments(self) -> dict[str, Any] | None:
        """Load previously saved alignments."""

    @abstractmethod
    def load_xyz_picks(self) -> np.ndarray | None:
        """Load xyz picks."""

    def load_previous_alignments(self) -> list[str]:
        """
        Load previous alignments into memory.

        Returns
        -------
        list of str
            Sorted alignment keys including 'original'.
        """
        data = self.load_alignments()
        if data:
            self.alignments = data

        return self.get_previous_alignments()

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
    """

    def __init__(self, insertion: dict, one: ONE, user: str | None = None):

        self.insertion: dict[str, Any] = insertion
        self.one: ONE = one
        self.traj_id: str | None = None

        super().__init__(user=user)

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
        traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.insertion['id'],
                                  provenance='Ephys aligned histology track', no_cache=True)
        if traj:
            return traj[0]['json']

    def load_trajectory(self) -> None:
        """Load the histology track trajectory and stores the trajectory id."""
        hist = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.insertion['id'],
                                  provenance='Histology track')

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

    def __init__(self, data_path: Path, shank_idx: int, n_shanks: int, user: str | None = None,
                 xyz_picks: np.ndarray | None = None):

        self.data_path: Path = data_path
        self.shank_idx: int = shank_idx
        self.n_shanks: int = n_shanks

        super().__init__(user=user, xyz_picks=xyz_picks)

    def load_xyz_picks(self) -> np.ndarray | None:
        """
        Load xyz picks from local file.

        Returns
        -------
        np.ndarray or None
            The xyz picks as a (N, 3) array in m, or None if not found.
        """
        xyz_name = '*xyz_picks.json' if self.n_shanks == 1 else \
            f'*xyz_picks_shank{self.shank_idx + 1}.json'

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
        prev_align_name = 'prev_alignments.json' if self.n_shanks == 1 else \
            f'prev_alignments_shank{self.shank_idx + 1}.json'

        prev_align_file = self.data_path.joinpath(prev_align_name)

        return self._load_json_file(prev_align_file)

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
