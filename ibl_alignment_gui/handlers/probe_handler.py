import re
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ibl_alignment_gui.handlers.shank_handler import ShankHandler
from ibl_alignment_gui.loaders.alignment_loader import (
    AlignmentLoaderLocal,
    AlignmentLoaderOne,
)
from ibl_alignment_gui.loaders.alignment_uploader import (
    AlignmentUploaderLocal,
    AlignmentUploaderOne,
)
from ibl_alignment_gui.loaders.data_loader import (
    CollectionData,
    DataLoaderLocal,
    DataLoaderOne,
    SpikeGLXLoaderLocal,
    SpikeGLXLoaderOne,
)
from ibl_alignment_gui.loaders.geometry_loader import (
    GeometryLoaderLocal,
    GeometryLoaderOne,
)
from ibl_alignment_gui.loaders.histology_loader import (
    NrrdSliceLoader,
    download_histology_data,
)
from ibl_alignment_gui.loaders.plot_loader import PlotLoader
from iblatlas.atlas import AllenAtlas
from iblutil.util import Bunch
from one import params
from one.api import ONE


class ProbeHandler(ABC):
    """
    Abstract base class for handling alignment and data loading for a probe.

    This class provides access to loader methods that handle different aspects of the
    alignment process. Where applicable the probe is split into shanks and each shank is handled
    separately.

    It can also handle multiple configurations for each shank, for example if two different channel
    maps are used to record data.

    Parameters
    ----------
    brain_atlas: AllenAtlas
        An AllenAtlas instance.
    """

    def __init__(self, brain_atlas: AllenAtlas):

        self.brain_atlas: AllenAtlas = brain_atlas or AllenAtlas()
        self.shanks: dict[str, Bunch] = defaultdict(Bunch)

        # Configuration state
        self.default_config: str = 'default'
        self.non_default_config: str | None = None
        self.configs: list[str] = ['default']
        self.possible_configs: list[str] = ['default']
        self.selected_config: str = 'default'

        # Active shank indices
        self.selected_shank: str | None = None
        self.selected_idx: int | None = None

    # -------------------------------------------------------------------------
    # Shank access methods
    # -------------------------------------------------------------------------
    def get_current_shank(self, shank, config) -> ShankHandler:
        """Return the currently active shank."""
        return self.shanks[shank][config]

    def get_selected_shank(self) -> Bunch:
        """Return the currently selected shank."""
        return self.shanks[self.selected_shank]

    def get_config(self, idx: int) -> None:
        """
        Select a configuration by index.

        Parameters
        ----------
        idx : int
            Index in the list of possible configurations.
        """
        self.selected_config = self.possible_configs[idx]

    # -------------------------------------------------------------------------
    # Alignment methods - methods in loaders['align']
    # -------------------------------------------------------------------------
    def load_previous_alignments(self) -> dict:
        """
        Load previous alignments for the selected shank.

        Always returns the alignments from the default configuration.

        Returns
        -------
        dict
            Previous alignments for the selected shank.
        """
        # Load the previous alignment from the default configuration
        (self.get_selected_shank()[self.default_config].loaders['align']
         .load_previous_alignments())
        # Set the previous alignments from the non default configuration to the default
        # one (if it exists)
        if self.non_default_config is not None:
            self.get_selected_shank()[self.non_default_config].loaders['align'].alignments = (
                self.get_selected_shank()[self.default_config].loaders['align'].alignments(
                    self.get_selected_shank()[
                        self.non_default_config].loaders['align'].get_previous_alignments()))

        return (self.get_selected_shank()[self.default_config].loaders['align']
                .get_previous_alignments())

    def get_previous_alignments(self) -> dict:
        """
        Get previous alignments for the selected shank.

        Always returns the alignments from the default configuration.

        Returns
        -------
        dict
            Previous alignments for the selected shank
        """
        return (self.get_selected_shank()[self.default_config].loaders['align']
                .get_previous_alignments())

    def get_starting_alignment(self, idx: int) -> None:
        """
        Set the index of the starting alignment for the selected shank and each configuration.

        Parameters
        ----------
        idx: int
            The index of the previous alignment to load.
        """
        for config in self.configs:
            self.get_selected_shank()[config].loaders['align'].get_starting_alignment(idx)

    def set_init_alignment(self) -> None:
        """Initialise the alignment for the selected shank and each configuration."""
        for config in self.configs:
            self.get_selected_shank()[config].set_init_alignment()

    # -------------------------------------------------------------------------
    # Alignment handler methods - methods in align_handle
    # -------------------------------------------------------------------------
    def next_idx(self) -> int:
        """
        Return the index of the next available alignment for the selected shank.

        Returns
        -------
        int
            The index of the next available alignment stored in th circular buffer.
        """
        if self.non_default_config is not None:
            self.get_selected_shank()[self.non_default_config].align_handle.next_idx()
        return self.get_selected_shank()[self.default_config].align_handle.next_idx()

    def prev_idx(self) -> int:
        """
        Return the index of the previously available alignment for the selected shank.

        Returns
        -------
        int
            The index of the previous available alignment stored in th circular buffer.
        """
        if self.non_default_config is not None:
            self.get_selected_shank()[self.non_default_config].align_handle.prev_idx()
        return self.get_selected_shank()[self.default_config].align_handle.prev_idx()

    @property
    def current_idx(self) -> int:
        """
        Return the current index of the alignment stored in the buffer for the selected shank.

        Returns
        -------
        int
            The index of the current alignment
        """
        return self.get_selected_shank()[self.default_config].align_handle.current_idx

    @property
    def total_idx(self) -> int:
        """
        Return the total index of the alignments stored in the buffer for the selected shank.

        Returns
        -------
        int
            The total number of alignments stored in the circular buffer
        """
        return self.get_selected_shank()[self.default_config].align_handle.total_idx

    @property
    def y_min(self) -> float:
        """
        Minimum y channel value for the selected shank.

        If config is 'both' returns the minimum across both configurations.

        Returns
        -------
        float:
            The minimum channel value
        """
        if self.selected_config == 'both':
            y_min = [0]
            for config in self.configs:
                y_min.append(self.get_selected_shank()[config].loaders['plots'].chn_min)
            return np.nanmin(y_min)
        else:
            return np.min(
                [0, self.get_selected_shank()[self.selected_config].loaders['plots'].chn_min])

    @property
    def y_max(self) -> float:
        """
        Maximum y channel value for the selected shank.

        If config is 'both' returns the maximum across both configurations.

        Returns
        -------
        float:
            The maximum channel value
        """
        if self.selected_config == 'both':
            y_max = []
            for config in self.configs:
                y_max.append(self.get_selected_shank()[config].loaders['plots'].chn_max)
            return np.nanmax(y_max)
        else:
            return self.get_selected_shank()[self.selected_config].loaders['plots'].chn_max

    def get_plot(self, shank: str, plot: str, key: str, config: str | None = None) -> Any:
        """
        Access a specific plot for a specific shank and configuration.

        Parameters
        ----------
        shank: str
            The shank label to access.
        plot: str
            The plot type to access. One of 'image', 'scatter', 'line', 'probe' or 'slice'
        key: str
            The plot key to access.
        config: str
            The configuration to access. If None, uses the default configuration.

        Returns
        -------
        Any
            The requested plot, or None if not found.
        """
        config = config or self.default_config
        return getattr(self.shanks[shank][config].loaders['plots'], plot).get(key, None)

    def get_plot_keys(self, plot: str) -> list[str]:
        """
        Find a list of available keys across all shanks and configurations for a given plot type.

        Parameters
        ----------
        plot
            The plot type to get the keys for. One of 'image', 'scatter', 'line', 'probe'
            or 'slice'

        Returns
        -------
        list
            A list of unique plot keys.
        """
        keys = []
        for shank in self.shanks:
            for config in self.configs:
                keys += getattr(self.shanks[shank][config].loaders['plots'], plot).keys()

        return sorted(set(keys))

    @property
    def image_keys(self) -> list[str]:
        """
        Find the list of available image plot keys across all shanks and configurations.

        Returns
        -------
        list:
            A list of unique image plot keys.
        """
        return self.get_plot_keys('image_plots')

    @property
    def scatter_keys(self) -> list[str]:
        """
        Find the list of available scatter plot keys across all shanks and configurations.

        Returns
        -------
        list:
            A list of unique scatter plot keys.
        """
        return self.get_plot_keys('scatter_plots')

    @property
    def line_keys(self) -> list[str]:
        """
        Find the list of available line plot keys across all shanks and configurations.

        Returns
        -------
        list:
            A list of unique line plot keys.
        """
        return self.get_plot_keys('line_plots')

    @property
    def probe_keys(self) -> list[str]:
        """
        Find the list of available probe plot keys across all shanks and configurations.

        Returns
        -------
        tuple:
            A tuple of unique probe plot keys.
        """
        return self.get_plot_keys('probe_plots')

    @property
    def slice_keys(self) -> list[str]:
        """
        Find the list of available slice plot keys across all shanks and configurations.

        Returns
        -------
        list:
            A list of unique slice plot keys.
        """
        return self.get_plot_keys('slice_plots')

    # -------------------------------------------------------------------------
    # Data loading & upload
    # -------------------------------------------------------------------------
    def load_data(self) -> None:
        """Download and load data for all configs and shanks."""
        slice_loader = self.download_histology()
        for probe in self.shanks:
            for config in self.configs:
                self.shanks[probe][config].loaders['hist'] = slice_loader
                self.shanks[probe][config].load_data()

    def upload_data(self) -> str:
        """
        Upload data for the selected shank for each configuration.

        Always returns the upload result from the default configuration.

        Returns
        -------
        str
            Upload result from the default config.
        """
        info = Bunch()
        for config in self.configs:
            info[config] = self.get_selected_shank()[config].upload_data()
        return info[self.default_config]

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_shank_label(shank_label: str) -> str:
        """
        Normalize a shank label to the form 'probe0X'.

        Parameters
        ----------
        shank_label : str
            Input shank label.

        Returns
        -------
        str
            Normalized label.
        """
        match = re.match(r'(probe\d+)', shank_label)
        return match.group(1) if match else shank_label

    # -------------------------------------------------------------------------
    # Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def set_info(self, *args):
        """Set probe information."""

    @abstractmethod
    def download_histology(self):
        """Load histology data."""

    @abstractmethod
    def get_shanks(self, *args):
        """Return shank information."""

    @abstractmethod
    def initialise_shanks(self):
        """Initialize shank data."""


class ProbeHandlerONE(ProbeHandler):
    """
    ONE implementation of ProbeHandler.

    For this ProbeHandler all ephys and alignment data is downloaded and accessed via
    ONE and Alyx.

    The data for all shanks on a probe will be loaded at once.

    Parameters
    ----------
    one : ONE
        An ONE instance used to upload results to Alyx
    brain_atlas : AllenAtlas
        An AllenAtlas object.
    spike_collection : str, optional
        Spike sorting algorithm to load (e.g. 'pykilosort', 'iblsorter').
    """

    def __init__(
            self,
            one: ONE = None,
            brain_atlas: AllenAtlas | None = None,
            spike_collection: str | None = None):

        self.one = one or ONE()
        self.spike_collection = spike_collection
        super().__init__(brain_atlas)

    def get_subjects(self) -> np.ndarray:
        """
        Find all subjects that have probe insertions with spikesorting data.

        Returns
        -------
        np.ndarray
            An array of subject names
        """
        self.sess_ins = self.one.alyx.rest('insertions', 'list', dataset_type='spikes.times',
                                           expires=timedelta(days=1))
        self.subj_ins = [sess['session_info']['subject'] for sess in self.sess_ins]
        self.subjects = np.unique(self.subj_ins)

        return self.subjects

    def get_sessions(self, idx: int) -> np.ndarray:
        """
        Find all probes for a given subject.

        Note if multi-shank data it will return probe00 rather than probe00a, the individual shank
        is chosen using the shank dropdown.

        Parameters
        ----------
        idx : idx
            The index of the chosen subject

        Returns
        -------
        np.ndarray
            All probes with spikesorting data for the chosen subject
        """
        subj = self.subjects[idx]
        sess_idx = [i for i, e in enumerate(self.subj_ins) if e == subj]
        self.sess = [self.sess_ins[idx] for idx in sess_idx]
        self.sessions = [self.get_session_probe_name(sess) for sess in self.sess]
        self.sessions = np.unique(self.sessions)

        return self.sessions

    def get_shanks(self, idx: int) -> list:
        """
        Find all shanks for a given probe and initialise the loaders.

        Parameters
        ----------
        idx : idx
            The index of the chosen probe

        Returns
        -------
        np.ndarray
            All shanks for the chosen probe
        """
        sess = self.sessions[idx]

        sess_idx = [i for i, e in enumerate(self.sess) if self.get_session_probe_name(e) == sess]
        self.shank_labels = [self.sess[idx] for idx in sess_idx]
        shanks = [s['name'] for s in self.shank_labels]
        idx = np.argsort(shanks)
        self.shank_labels = np.array(self.shank_labels)[idx]
        shanks = np.array(shanks)[idx]

        self.initialise_shanks()

        return list(shanks)

    def get_session_probe_name(self, ins: dict) -> str:
        """
        Make a string containing the combination of session information and probe name.

        Removes the shank identifiers from the probe names.

        Parameters
        ----------
        ins: dict
            A dict containing insertion data

        Returns
        -------
        str:
            A string with the session info and probe name
        """
        return (ins['session_info']['start_time'][:10] + ' ' +
                self.normalize_shank_label(ins['name']))

    def set_info(self, idx):
        """
        Set the information about the selected shank.

        Parameters
        ----------
        idx: int
            The index of the selected shank
        """
        self.selected_shank = self.shank_labels[idx]['name']
        self.selected_idx = idx
        self.subj = self.shank_labels[idx]['session_info']['subject']
        self.lab = self.shank_labels[idx]['session_info']['lab']

    def download_histology(self) -> NrrdSliceLoader:
        """Download and load in the histology slice data."""
        _, hist_path = download_histology_data(self.subj, self.lab)
        return NrrdSliceLoader(hist_path, self.brain_atlas)

    def initialise_shanks(self):
        """Initialise each shank with the loaders."""
        self.shanks = defaultdict(Bunch)

        for ins in self.shank_labels:
            loaders = Bunch()
            loaders['data'] = DataLoaderOne(ins, self.one, spike_collection=self.spike_collection)
            loaders['geom'] = GeometryLoaderOne(ins, self.one,
                                                probe_collection=loaders['data'].probe_collection)
            loaders['align'] = AlignmentLoaderOne(ins, self.one)
            loaders['upload'] = AlignmentUploaderOne(ins, self.one, self.brain_atlas)
            loaders['ephys'] = SpikeGLXLoaderOne(ins, self.one)
            loaders['plots'] = PlotLoader()
            self.shanks[ins['name']][self.default_config] = ShankHandler(loaders, 0)


class ProbeHandlerCSV(ProbeHandler):
    """
    ProbeHandler where data from two channel maps has been recorded on the shanks.

    The data for the dense configuration is available via ONE whereas the data for the quarter
    configuration is only available on the local file system. Reads in a csv file that contains
    information about where to read the relevant data from.
    """

    def __init__(
            self,
            csv_file: str | Path,
            one: ONE = None,
            brain_atlas: AllenAtlas | None = None):
        super().__init__(brain_atlas)

        csv_file = Path(csv_file)
        assert csv_file.exists()

        self.root_path = csv_file.parent
        self.df = pd.read_csv(csv_file, keep_default_na=False)
        self.df['session_strip'] = self.df['session'].str.rsplit('/', n=1).str[0]
        self.one = one or ONE()

        self.possible_configs = ['quarter', 'dense', 'both']
        self.configs = ['quarter', 'dense']
        self.default_config = 'dense'
        self.non_default_config = 'quarter'
        self.selected_config = 'quarter'

    def get_subjects(self) -> np.ndarray:
        """
        Find all sessions with spike sorting data.

        Returns
        -------
        np.ndarray
            All sessions with spikesorting data.
        """
        # Returns sessions
        self.subjects = self.df['session_strip'].unique()
        return self.subjects

    def get_sessions(self, idx) -> np.ndarray:
        """
        Find all probes for a given session.

        Note if multi-shank data it will return probe00 rather than probe00a, the individual shank
        is chosen using the shank dropdown.

        Parameters
        ----------
        idx : idx
            The index of the chosen subject

        Returns
        -------
        np.ndarray
            All probes with spikesorting data for the chosen session
        """
        self.session_df = self.df.loc[self.df['session_strip'] == self.subjects[idx]]
        self.sessions = np.unique([self.normalize_shank_label(pr)
                                   for pr in self.session_df['probe'].values])
        return self.sessions

    def get_shanks(self, idx: int) -> np.ndarray:
        """
        Find all shanks for a given probe and initialise the loaders.

        Parameters
        ----------
        idx : idx
            The index of the chosen probe

        Returns
        -------
        np.ndarray
            All shanks for the chosen probe
        """
        shank = self.sessions[idx]
        self.shank_df = self.session_df.loc[
            self.session_df['probe'].str.contains(shank)].sort_values('probe')
        self.initialise_shanks()
        self.shank_labels = self.shank_df['probe'].unique()
        return self.shank_labels

    def set_info(self, idx: int) -> None:
        """
        Set the information about the selected shank.

        Parameters
        ----------
        idx: int
            The index of the selected shank
        """
        self.selected_shank = self.shank_labels[idx]
        self.selected_idx = idx

    def download_histology(self) -> NrrdSliceLoader:
        """Download and load in the histology slice data."""
        _, hist_path = download_histology_data(self.subj, self.lab)
        return NrrdSliceLoader(hist_path, self.brain_atlas)

    def initialise_shanks(self) -> None:
        """Initialise each shank and config with the selected loaders."""
        self.shanks = defaultdict(Bunch)
        user = params.get().ALYX_LOGIN

        for _, shank in self.shank_df.iterrows():
            loaders = Bunch()
            collections = CollectionData(
                spike_collection=shank.spike_collection or '',
                ephys_collection=shank.ephys_collection or '',
                task_collection=shank.task_collection or '',
                raw_task_collection=shank.raw_task_collection or '',
                meta_collection=shank.meta_collection or '')

            local_path = self.root_path.joinpath(shank.local_path)

            ins = self.get_insertion(shank)
            xyz_picks = ins['json'].get('xyz_picks', None)
            xyz_picks = np.array(xyz_picks) / 1e6 if xyz_picks is not None else None

            if shank.is_quarter:  # Quarter is offline
                loaders['data'] = DataLoaderLocal(local_path, collections)
                loaders['geom'] = GeometryLoaderLocal(local_path, collections)
                loaders['align'] = AlignmentLoaderLocal(
                    local_path.joinpath(collections.spike_collection), 0, 1,
                    user=user, xyz_picks=xyz_picks)
                loaders['upload'] = AlignmentUploaderLocal(
                    local_path.joinpath(collections.spike_collection), 0, loaders['geom'],
                    self.brain_atlas, user=user)
                loaders['ephys'] = SpikeGLXLoaderLocal(local_path, collections.meta_collection)
                loaders['plots'] = PlotLoader()
                self.shanks[shank.probe]['quarter'] = ShankHandler(loaders, 0)
            else:  # Dense is online
                # If we don't have the data locally we download it
                if collections.spike_collection == '':
                    loaders['data'] = DataLoaderOne(ins, self.one)
                    loaders['geom'] = GeometryLoaderOne(
                        ins, self.one, probe_collection=loaders['data'].probe_collection)
                # Otherwise we load from local
                else:
                    loaders['data'] = DataLoaderLocal(local_path, collections)
                    loaders['geom'] = GeometryLoaderLocal(local_path, collections)

                loaders['align'] = AlignmentLoaderOne(ins, self.one, user=user)
                loaders['upload'] = AlignmentUploaderOne(ins, self.one, self.brain_atlas)
                loaders['ephys'] = SpikeGLXLoaderOne(ins, self.one)
                loaders['plots'] = PlotLoader()
                self.shanks[shank.probe]['dense'] = ShankHandler(loaders, 0)

        self._sync_alignments()
        self.subj = shank['subject']
        self.lab = shank['lab']

    def _sync_alignments(self) -> None:
        """Synchronize alignments between dense and quarter loaders."""
        for _, shank_group in self.shanks.items():
            dense_align = shank_group['dense'].loaders['align']
            quarter_align = shank_group['quarter'].loaders['align']

            if dense_align.alignment_keys != ['original']:
                # Alyx alignment exists: overwrite local
                quarter_align.alignments = dense_align.alignments
                quarter_align.get_previous_alignments()
                quarter_align.get_starting_alignment(0)

            elif quarter_align.alignment_keys != ['original']:
                # Local alignment exists: add to online
                dense_align.add_extra_alignments(quarter_align.alignments)
                dense_align.get_previous_alignments()
                dense_align.get_starting_alignment(0)

                # Ensure consistency by syncing quarter with updated dense
                quarter_align.alignments = dense_align.alignments
                quarter_align.get_previous_alignments()
                quarter_align.get_starting_alignment(0)

    def get_insertion(self, shank: pd.Series) -> dict:
        """Get the alyx probe insertion for the shank."""
        ins = self.one.alyx.rest('insertions', 'list', id=shank.pid, expires=timedelta(days=1))
        return ins[0]


class ProbeHandlerLocal(ProbeHandler):
    """
    Local file system implementation of ProbeHandler.

    For this ProbeHandler, all ephys and alignment data must be stored in a single folder on disk.
    """

    def __init__(self, brain_atlas: AllenAtlas | None = None):
        super().__init__(brain_atlas)

    def get_shanks(self, folder_path: Path) -> list[str]:
        """
        Find the number of shanks on the probes.

        Loads the channels or ap meta data from the folder path and initialises the loaders
        for each shank.

        Parameters
        ----------
        folder_path : Path
            A path to the folder on the local disk that contains the data
        """
        self.folder_path = folder_path
        collections = CollectionData()

        # Load in the geometry and find the number of shnaks
        self.geom = GeometryLoaderLocal(self.folder_path, collections)
        self.geom.get_geometry()

        self.n_shanks = self.geom.channels.n_shanks
        if self.n_shanks == 1:
            self.shank_labels = ['1/1']
        else:
            self.shank_labels = [f'{iShank + 1}/{self.n_shanks}'
                                 for iShank in range(self.n_shanks)]

        self.initialise_shanks()

        return self.shank_labels

    def set_info(self, idx: int) -> None:
        """
        Set the information about the selected shank.

        Parameters
        ----------
        idx: int
            The index of the selected shank
        """
        self.selected_shank = f'shank_{self.shank_labels[idx]}'
        self.selected_idx = idx

    def download_histology(self) -> NrrdSliceLoader:
        """Load in the histology slice data."""
        return NrrdSliceLoader(self.folder_path, self.brain_atlas)

    def initialise_shanks(self) -> None:
        """Initialise each shank with the loaders."""
        self.shanks = defaultdict(Bunch)

        for ish, ishank in enumerate(self.shank_labels):
            loaders = Bunch()
            loaders['geom'] = self.geom
            loaders['data'] = DataLoaderLocal(self.folder_path, CollectionData())
            loaders['align'] = AlignmentLoaderLocal(self.folder_path, ish, self.n_shanks)
            loaders['upload'] = AlignmentUploaderLocal(self.folder_path, ish, self.n_shanks,
                                                       self.brain_atlas)
            loaders['ephys'] = SpikeGLXLoaderLocal(self.folder_path, '')
            loaders['plots'] = PlotLoader()
            self.shanks[f'shank_{ishank}'][self.default_config] = ShankHandler(loaders, ish)
