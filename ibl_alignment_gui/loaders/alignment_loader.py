from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from iblutil.util import Bunch

logger = logging.getLogger(__name__)

# Marks an alignment that was read from a local folder or file rather than from the primary source.
# Kept as a suffix so that the alignment keys, which are timestamps, still sort
# chronologically in the dropdown.
LOCAL_SUFFIX = ' (local)'

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
    import_folder : str or None
        The name of the session folder that previous alignments are additionally read from, of
        the form ``<subject>_<date>_<probe>``. If None, no local folders are read.
    import_shank : int or None
        The 1 based shank number used to pick the alignment file out of an imported folder.
        None for single shank data, where the file carries no shank number.
    """

    def __init__(
        self,
        user: str | None = None,
        xyz_picks: np.ndarray | None = None,
        data_path: Path | None = None,
        shank_idx: int = 0,
        n_shanks: int = 1,
        *,
        import_folder: str | None = None,
        import_shank: int | None = None,
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

        # Previous alignments held on the local file system, merged in alongside those of the
        # primary source. The folders are chosen at runtime, so they are set by the probe handler
        # rather than passed in here.
        self.import_folder: str | None = import_folder
        self.import_shank: int | None = import_shank
        self.import_roots: list[Path] = []
        self.imported_keys: set[str] = set()
        # The imported alignments whose key is attributed to the user of the folder they were
        # found in, and which therefore win over a copy found in another user's folder
        self._authoritative_keys: set[str] = set()

    @abstractmethod
    def load_alignments(self) -> dict[str, Any] | None:
        """Load previously saved alignments."""

    @abstractmethod
    def load_xyz_picks(self) -> np.ndarray | None:
        """Load xyz picks."""

    def load_previous_alignments(self) -> list[str]:
        """
        Load previous alignments into memory.

        Any alignments held in the chosen local folders are merged in, and any work in
        progress that has been saved is loaded alongside them, so that both are recovered
        whenever the previous alignments are refreshed.

        Returns
        -------
        list of str
            Sorted alignment keys including 'original'.
        """
        data = self.load_alignments()
        if data:
            self.alignments = data

        self.load_imported()
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

    def load_imported(self) -> None:
        """
        Merge previous alignments held in local folders or files into the available alignments.

        The folders are chosen at runtime and hold what the offline gui writes out, laid out as
        ``<root>/<alyx user>/<subject>_<date>_<probe>/``. Anything found is added under a key
        marked as local, so that it can be chosen from the alignment dropdown but is never
        mistaken for, or uploaded alongside, the alignments of the primary source.

        A single alignment file can be chosen in place of a folder, so that the alignments of
        only one shank are read. The file is only read by the shank it was written for.

        Nothing is added if nothing has been chosen, or if the shank has no session folder name
        to look for.
        """
        # Drop anything imported previously, so that what is loaded always reflects the folders
        # that are currently chosen and what they hold
        for key in self.imported_keys:
            self.alignments.pop(key, None)

        self.imported_keys = set()
        self._authoritative_keys = set()

        if not self.import_roots or self.import_folder is None:
            return

        for root in self.import_roots:
            if root.is_file():
                if self.matches_import_file(root):
                    data = self._load_json_file(root)
                    if data:
                        # The user is taken from the folder the session folder is held in
                        self._merge_imported(data, root.parent.parent.name)
                continue

            for user, folder in self._find_import_folders(root):
                for name in self._import_names():
                    data = self._load_json_file(folder.joinpath(name))
                    if data:
                        self._merge_imported(data, user)
                        break

        if self.imported_keys:
            logger.info(
                f'Loaded {len(self.imported_keys)} previous alignment(s) for '
                f'{self.import_folder} from local folders'
            )

    def _import_names(self) -> list[str]:
        """
        Return the names of the alignment files to look for in an imported folder.

        Multi-shank data only ever matches the file for its own shank, as a file without a shank
        number belongs to a different layout and would be the wrong alignment for this shank.

        Returns
        -------
        list of str
            The filenames to look for, in the order they should be tried.
        """
        if self.import_shank is None:
            # Single shank data may have been written with or without the shank in the name
            return ['prev_alignments.json', 'prev_alignments_shank1.json']

        return [f'prev_alignments_shank{self.import_shank}.json']

    def matches_import_file(self, file: Path) -> bool:
        """
        Return whether a chosen alignment file belongs to this shank.

        The file must be one of the names this shank reads, so that the file of one shank is
        never loaded into another, and must sit in this shank's session folder, so that a file
        belonging to another session is never read for this one.

        Parameters
        ----------
        file : Path
            The chosen file.

        Returns
        -------
        bool
            True if the file holds the alignments of this shank.
        """
        return file.name in self._import_names() and file.parent.name == self.import_folder

    def _find_import_folders(self, root: Path) -> list[tuple[str, Path]]:
        """
        Find the folders under a chosen root that hold alignments for this shank's session.

        The root is whichever folder was chosen, so all three levels of the layout are accepted:
        the session folder itself, a single user's folder, or a root holding a folder per user.

        Parameters
        ----------
        root : Path
            The chosen folder.

        Returns
        -------
        list of (str, Path)
            The alyx user and session folder of each match.
        """
        # The session folder itself. Its name is checked so that a folder belonging to another
        # session is never read for this one
        if root.name == self.import_folder and any(
            root.joinpath(name).exists() for name in self._import_names()
        ):
            return [(root.parent.name, root)]

        # A single user's folder
        if root.joinpath(self.import_folder).is_dir():
            return [(root.name, root.joinpath(self.import_folder))]

        # A root holding a folder per user
        return [
            (folder.parent.name, folder)
            for folder in sorted(root.glob(f'*/{self.import_folder}'))
            if folder.is_dir()
        ]

    def _merge_imported(self, data: dict[str, Any], user: str) -> None:
        """
        Merge the alignments read from one local folder into the available alignments.

        The same alignment is often held in more than one folder, as each folder holds
        everything the gui that wrote it knew about. Where two copies disagree, the one found in
        the folder of the user the alignment is attributed to is taken, and otherwise the first
        folder read wins. Alignments already known to the primary source are left out, so that
        the ones these folders share with it are not listed twice.

        Parameters
        ----------
        data : dict
            The alignments read from the folder.
        user : str
            The alyx user whose folder they were read from.
        """
        for key, val in data.items():
            # An alignment saved without a user is attributed to the owner of the folder it was
            # found in, rather than to whoever is running the gui
            base = f'{key}_{user}' if len(key) == 19 else key

            if base in self.alignments:
                continue

            import_key = f'{base}{LOCAL_SUFFIX}'
            authoritative = base.endswith(f'_{user}')

            if import_key in self.imported_keys:
                if import_key in self._authoritative_keys or not authoritative:
                    continue
                if self.alignments[import_key] != val:
                    logger.warning(
                        f'Copies of alignment {base} differ between folders, using the one '
                        f'found in the folder of {user}'
                    )

            self.alignments[import_key] = val
            self.imported_keys.add(import_key)
            if authoritative:
                self._authoritative_keys.add(import_key)

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
        must never be saved alongside the alignments that have been uploaded. Alignments
        imported from local folders are left out for the same reason: they are someone else's
        record of work done elsewhere, and uploading them would attribute them to this session
        and change how many users the primary source believes have aligned this insertion.

        Both are recognised by their key rather than by what this loader imported, so that they
        stay excluded once the alignments are copied between the loaders of different
        configurations.

        Returns
        -------
        dict
            The alignments, excluding any recovered or imported alignment.
        """
        return {
            key: val
            for key, val in self.alignments.items()
            if not str(key).startswith('recovered') and not str(key).endswith(LOCAL_SUFFIX)
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
    import_folder : str or None
        The name of the session folder that previous alignments are additionally read from.
    import_shank : int or None
        The 1 based shank number used to pick the alignment file out of an imported folder.
    """

    def __init__(
        self,
        insertion: dict,
        one: ONE,
        user: str | None = None,
        data_path: Path | None = None,
        *,
        import_folder: str | None = None,
        import_shank: int | None = None,
    ):
        self.insertion: dict[str, Any] = insertion
        self.one: ONE = one
        self.traj_id: str | None = None

        super().__init__(
            user=user,
            data_path=data_path,
            import_folder=import_folder,
            import_shank=import_shank,
        )

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

    xyz picks and previous alignments are loaded from files on disk. The previous alignments are
    read from the folder that the uploader writes them to, which is not necessarily the folder
    that the xyz picks are read from.

    For single-shank data, expected filenames:
        - ``*xyz_picks.json``
        - ``prev_alignments.json``

    For multi-shank data, expected filenames:
        - ``*xyz_picks_shank<N>.json``
        - ``prev_alignments_shank<N>.json``

    Parameters
    ----------
    data_path : Path
        The path to the folder that the alignment results are written to, and so the folder that
        previous alignments and saved progress are read from.
    shank_idx : int
        Index of the shank (0-based).
    n_shanks : int
        Total number of shanks.
    user : str or None
        Username for tagging alignments.
    xyz_picks : np.ndarray or None
        Preloaded xyz picks. If not provided, it will attempt to load from file.
    picks_path : Path or None
        The path to the folder that the xyz picks are read from. Defaults to `data_path` when the
        picks sit alongside the alignment results.
    import_folder : str or None
        The name of the session folder that previous alignments are additionally read from.
    import_shank : int or None
        The 1 based shank number used to pick the alignment file out of an imported folder.
    """

    def __init__(
        self,
        data_path: Path,
        shank_idx: int,
        n_shanks: int,
        user: str | None = None,
        xyz_picks: np.ndarray | None = None,
        histology_space: str = 'ccf',
        picks_path: Path | None = None,
        *,
        import_folder: str | None = None,
        import_shank: int | None = None,
    ):
        self.histology_space: str = histology_space
        # Set before the xyz picks are loaded in the base class
        self.picks_path: Path = picks_path if picks_path is not None else data_path

        super().__init__(
            user=user,
            xyz_picks=xyz_picks,
            data_path=data_path,
            shank_idx=shank_idx,
            n_shanks=n_shanks,
            import_folder=import_folder,
            import_shank=import_shank,
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

        xyz_file = sorted(self.picks_path.glob(xyz_name))

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
    ``session = data_path.parent.stem`` and ``probe = data_path.stem``. Both are therefore derived
    from the folder that the alignment results are written to, so that the record written on upload
    is the one read back.

    Parameters
    ----------
    data_path : Path
        The path to the folder that the alignment results are written to.
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
    picks_path : Path or None
        The path to the folder that the xyz picks are read from. Defaults to `data_path`.
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
        picks_path: Path | None = None,
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
            picks_path=picks_path,
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
