import logging
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spikeglx

import ibldsp.voltage
import one.alf.io as alfio
from brainbox.io.spikeglx import Streamer
from iblutil.numerical import ismember
from iblutil.util import Bunch
from one.alf.exceptions import ALFObjectNotFound
from one.api import ONE

logger = logging.getLogger(__name__)


@dataclass
class CollectionData:
    """
    Container for dataset collection names used in an experiment.

    Attributes
    ----------
    spike_collection : str or None
        Collection name for spike-sorted data (e.g., 'alf/probe00/iblsorter'),
        default is empty string.
    ephys_collection : str or None
        Collection name for raw electrophysiology data (e.g., 'raw_ephys_data').
    task_collection : str or None
        Collection name for task data (e.g., 'alf/task01').
    raw_task_collection : str or None
        Collection name for raw task data (e.g., 'raw_task01').
    meta_collection : str or None
        Collection name for raw electrophysiology metadata with probe info
        (e.g., 'raw_ephys_data').
    """

    spike_collection: str | None = ''
    ephys_collection: str | None = ''
    task_collection: str | None = ''
    raw_task_collection: str | None = ''
    meta_collection: str | None = ''


class DataLoader(ABC):
    """
    Abstract base class for loading processed data  for the ibl_alignment_gui.

    Loads spikesorted and raw ephys data.

    Subclasses must implement the follow abstract methods:
    - `load_passive_data`
    - `load_raw_passive_data`
    - `load_ephys_data`
    - `load_spikes_data`
    """

    def __init__(self):

        self.filter: bool = False
        self.shank_sites: Bunch | None = None

    def get_data(self, shank_sites: Bunch[str, Any]) -> Bunch[str, Any]:
        """
        Load all relevant data associated with the probe.

        Parameters
        ----------
        shank_sites : Bunch
            A Bunch object containing the channels that correspond to the shank

        Returns
        -------
        data: Bunch
            A Bunch object containing spikes, clusters, channels, RMS and PSD data, and
            passive stimulus data.
        """
        self.shank_sites = shank_sites

        data = Bunch()
        # Load in spike sorting data
        data['spikes'], data['clusters'], data['channels'] = self.get_spikes_data()
        # Load in rms AP data
        data['rms_AP'] = self.get_rms_data(band='AP')
        # Load in rms LF data
        data['rms_LF'] = self.get_rms_data(band='LF')
        # Load in psd LF data
        data['psd_LF'] = self.get_psd_data(band='LF')
        # Load in passive data
        # TODO this data should be shared across probes
        data['rf_map'], data['pass_stim'], data['gabor'] = self.get_passive_data()

        return data

    @staticmethod
    def load_data(
            load_function: Callable,
            *args: Any,
            raise_message: str | None = None,
            raise_exception: Exception = ALFObjectNotFound,
            raise_error: bool = False,
            **kwargs
    ) -> Bunch[str, Any]:
        """
        Safely load data using a provided function.

        Parameters
        ----------
        load_function : Callable
            Function to load the data (e.g., ONE or alfio).
        args : tuple
            Arguments for the loading function.
        raise_message : str or None
            Message to log if an exception is raised.
        raise_exception : Exception or ALFObjectNotFound
            Type of exception to catch.
        raise_error : bool
            Whether to raise the exception after logging.

        Returns
        -------
        Bunch
            The loaded data with an 'exists' flag.
        """
        alf_object = args[1]
        try:
            data = load_function(*args, **kwargs)
            if isinstance(data, dict | Bunch):
                data['exists'] = True
            return data
        except raise_exception as e:
            raise_message = raise_message or (f'{alf_object} data was not found, '
                                              f'some plots will not display')
            logger.warning(raise_message)
            if raise_error:
                logger.error(raise_message)
                logger.error(traceback.format_exc())
                raise e
            return Bunch(exists=False)

    @abstractmethod
    def load_passive_data(self, alf_object: str, **kwargs) -> Bunch[str, Any]:
        """Abstract method to load passive data."""

    @abstractmethod
    def load_raw_passive_data(self, alf_object: str, **kwargs) -> Bunch[str, Any]:
        """Abstract method to load raw passive data."""

    def get_passive_data(self) -> tuple[Bunch[str, Any], Bunch[str, Any], Bunch[str, Any]]:
        """
        Load passive visual stimulus data including RF map, visual stimuli, and gabor events.

        Returns
        -------
        rf_data : Bunch
            RF map data including frames.
        stim_data : Bunch
            Visual stimulus data.
        vis_stim : Bunch
            Gabor event data for left and right gabors.
        """
        try:
            rf_data = self.load_passive_data('passiveRFM')
            frame_path = self.load_raw_passive_data('RFMapStim')
            frames = np.fromfile(frame_path['raw'], dtype="uint8")
            rf_data['frames'] = np.transpose(np.reshape(frames, [15, 15, -1],
                                                        order="F"), [2, 1, 0])
        except Exception:
            logger.warning('passiveRFM data was not found, some plots will not display')
            rf_data = Bunch(exists=False)

        # Load in passive stim data
        stim_data = self.load_passive_data('passiveStims')

        # Load in passive gabor data
        try:
            gabor = self.load_passive_data('passiveGabor')
            if not gabor['exists']:
                vis_stim = Bunch(exists=False)
            else:
                vis_stim = Bunch()
                vis_stim['leftGabor'] = gabor['start'][
                    (gabor['position'] == 35) & (gabor['contrast'] > 0.1)]
                vis_stim['rightGabor'] = gabor['start'][
                    (gabor['position'] == -35) & (gabor['contrast'] > 0.1)]
                vis_stim['exists'] = True
        except Exception:
            logger.warning('Failed to process passiveGabor data, some plots will not display')
            vis_stim = Bunch(exists=False)

        return rf_data, stim_data, vis_stim

    @abstractmethod
    def load_ephys_data(self, alf_object: str, **kwargs) -> Bunch[str, Any]:
        """Abstract method to load ephys data."""

    def get_rms_data(self, band: str = 'AP') -> Bunch[str, Any]:
        """
        Load RMS data for specified band.

        Only returns data on channels present on selected shank.

        Parameters
        ----------
        band : str
            Band type ('AP' or 'LF').

        Returns
        -------
        rms_data : Bunch
            RMS data
        """
        rms_data = self.load_ephys_data(f'ephysTimeRms{band}')
        rms_data = self.filter_raw_by_chns(rms_data)

        if rms_data['exists']:
            if 'amps' in rms_data:
                rms_data['rms'] = rms_data.pop('amps')
            if 'timestamps' not in rms_data:
                rms_data['timestamps'] = np.array([0, rms_data['rms'].shape[0]])
                rms_data['xaxis'] = 'Time samples'
            else:
                rms_data['xaxis'] = 'Time (s)'

        return rms_data

    def get_psd_data(self, band: str = 'LF') -> Bunch[str, Any]:
        """
        Load power spectral density data for specified band.

        Only returns data on channels present on selected shank.

        Parameters
        ----------
        band : str
            Band type ('AP' or 'LF').

        Returns
        -------
        psd_data: Bunch
            PSD data
        """
        psd_data = self.load_ephys_data(f'ephysSpectralDensity{band}')
        psd_data = self.filter_raw_by_chns(psd_data)

        if psd_data['exists'] and 'amps' in psd_data:
            psd_data['power'] = psd_data.pop('amps')

        return psd_data

    @abstractmethod
    def load_spikes_data(
            self,
            alf_object: str,
            attributes: list[str],
            **kwargs
    ) -> Bunch[str, Any]:
        """Abstract method to load spike sorting data."""

    def get_spikes_data(self) -> tuple[Bunch[str, Any], Bunch[str, Any], Bunch[str, Any]]:
        """
        Load spike sorting data and optionally filter by minimum firing rate threshold.

        Only returns data on channels present on selected shank.

        Returns
        -------
        spikes: Bunch
            spikes data
        clusters: Bunch
            clusters data
        channels: Bunch
            channels data
        """
        spikes = self.load_spikes_data('spikes',
                                       ['depths', 'amps', 'times', 'clusters'])

        clusters = self.load_spikes_data('clusters',
                                         ['metrics', 'peakToTrough', 'waveforms', 'channels'])

        channels = self.load_spikes_data('channels',
                                         ['rawInd', 'localCoordinates'])

        if self.filter and spikes['exists']:
            # Remove low firing rate clusters
            spikes, clusters = self.filter_spikes_by_fr(spikes, clusters)

        if spikes['exists']:
            spikes, *_ = self.filter_spikes_by_chns(spikes, clusters, channels)

        return spikes, clusters, channels

    def filter_spikes_by_chns(
            self,
            spikes: Bunch[str, Any],
            clusters: Bunch[str, Any],
            channels: Bunch[str, Any]
    ) -> tuple[Bunch[str, Any], Bunch[str, Any], Bunch[str, Any]]:
        """
        Filter spikes to only include data relevant to channels present on selected shank.

        Returns
        -------
        spikes: Bunch
            Filtered spikes data
        clusters: Bunch
            Filtered clusters data
        channels: Bunch
            Filtered channels data
        """
        spikes_idx = np.isin(channels['rawInd'][clusters['channels'][spikes['clusters']]],
                             self.shank_sites['spikes_ind'])

        for key in spikes:
            if key == 'exists':
                continue
            spikes[key] = spikes[key][spikes_idx]

        return spikes, clusters, channels

    def filter_raw_by_chns(self, data: Bunch[str, Any]) -> Bunch[str, Any]:
        """
        Filter ephys data to only include data relevant to channels present on selected shank.

        Parameters
        ----------
        data : dict
            Raw ephys data

        Returns
        -------
        data: Bunch
            Filtered ephys data
        """
        for key in data:
            if key == 'exists':
                continue
            if data[key].ndim == 1:
                continue

            data[key] = data[key][:, self.shank_sites['raw_ind']]

        return data

    @staticmethod
    def filter_spikes_by_fr(
            spikes: Bunch[str, Any],
            clusters: Bunch[str, Any],
            min_fr: float = 50 / 3600
    ) -> tuple[Bunch[str, Any], Bunch[str, Any]]:
        """
        Remove low-firing clusters and filter spikes accordingly.

        Parameters
        ----------
        spikes : Bunch
            Spike data.
        clusters : Bunch
            Cluster data.
        min_fr : float
            Minimum firing rate in Hz.

        Returns
        -------
        spikes: Bunch
            Spikes data above fr threshold
        clusters: Bunch
            Clusters data above fr threshold
        """
        clu_idx = clusters['metrics'].firing_rate > min_fr
        exists = clusters.pop('exists')
        clusters = alfio.AlfBunch({k: v[clu_idx] for k, v in clusters.items()})
        clusters['exists'] = exists

        spike_idx, ib = ismember(spikes['clusters'], clusters['metrics'].index)
        clusters['metrics'].reset_index(drop=True, inplace=True)
        exists = spikes.pop('exists')
        spikes = alfio.AlfBunch({k: v[spike_idx] for k, v in spikes.items()})
        spikes['exists'] = exists
        spikes.clusters = clusters['metrics'].index[ib].astype(np.int32)

        return spikes, clusters


class DataLoaderOne(DataLoader):
    """
    Data loader using ONE.

    Data are downloaded using Alyx/ ONE.

    Parameters
    ----------
    insertion : dict
        Dictionary representing a probe insertion (must include 'session' and 'name').
    one : ONE
        An ONE instance used to access data.
    session_path : Path or None
        Path to the session folder. If None, it is resolved using the eid via `one.eid2path`.
    spike_collection : str or None
        Spike sorting algorithm to load (e.g. 'pykilosort', 'iblsorter').
    """

    def __init__(self, insertion: dict, one: ONE,
                 session_path: Path | None = None, spike_collection: str | None = None):

        self.one: ONE = one
        self.eid: str = insertion['session']
        self.session_path: Path = session_path or one.eid2path(self.eid)
        self.probe_label: str = insertion['name']
        self.spike_collection: str | None = spike_collection
        self.probe_path: Path = self.get_spike_sorting_path()
        self.probe_collection: str = str(self.probe_path.relative_to(self.session_path))
        self.filter: bool = True

        super().__init__()

    def get_spike_sorting_path(self) -> Path:
        """
        Determine the path to the spike sorting output.

        Returns
        -------
        probe_path: Path
            A Path to the spike sorting folder for the probe.
        """
        probe_path = self.session_path.joinpath('alf', self.probe_label)

        if self.spike_collection == '':
            return probe_path
        elif self.spike_collection:
            return probe_path.joinpath(self.spike_collection)

        # Find all spike sorting collections
        all_collections = self.one.list_collections(self.eid)
        # iblsorter is default, then pykilosort
        for sorter in ['iblsorter', 'pykilosort']:
            if f'alf/{self.probe_label}/{sorter}' in all_collections:
                return probe_path.joinpath(sorter)
        # If neither exist return ks2 path
        return probe_path

    def load_passive_data(self, alf_object: str, **kwargs) -> Bunch[str, Any]:
        """
        Load passive data using ONE.

        Returns
        -------
        Bunch
        """
        return self.load_data(self.one.load_object, self.eid, alf_object)

    def load_raw_passive_data(self, alf_object: str, **kwargs) -> Bunch[str, Any]:
        """
        Load raw passive data using ONE.

        Returns
        -------
        Bunch
        """
        return self.load_data(self.one.load_object, self.eid, alf_object)

    def load_ephys_data(self, alf_object: str, **kwargs) -> Bunch[str, Any]:
        """
        Load ephys data using ONE.

        Returns
        -------
        Bunch
        """
        return self.load_data(self.one.load_object, self.eid, alf_object,
                              collection=f'raw_ephys_data/{self.probe_label}', **kwargs)

    def load_spikes_data(
            self,
            alf_object: str,
            attributes: list[str],
            **kwargs
    ) -> Bunch[str, Any]:
        """
        Load spike sorting data using ONE.

        Returns
        -------
        Bunch
        """
        return self.load_data(
            self.one.load_object, self.eid, alf_object, collection=self.probe_collection,
            attribute=attributes, **kwargs)


class DataLoaderLocal(DataLoader):
    """
    Data loader using local file system.

    Data are loaded from files on disk. Uses a CollectionData object to resolve the paths for
    the different data directories.

    Parameters
    ----------
    probe_path : Path
        Root directory of probe data
    collections : CollectionData
        Object containing subcollection paths for spike, ephys, task, raw_task, and metadata.
    """

    def __init__(self, probe_path: Path, collections: CollectionData):

        self.probe_path: Path = probe_path
        self.spike_path: Path = probe_path.joinpath(collections.spike_collection)
        self.ephys_path: Path = probe_path.joinpath(collections.ephys_collection)
        self.task_path: Path = probe_path.joinpath(collections.task_collection)
        self.raw_task_path: Path = probe_path.joinpath(collections.raw_task_collection)
        self.meta_path: Path = probe_path.joinpath(collections.meta_collection)
        self.probe_collection: str = collections.spike_collection

        super().__init__()

    def load_passive_data(self, alf_object: str, **kwargs) -> Bunch[str, Any]:
        """
        Load passive data from local path.

        Returns
        -------
        Bunch
        """
        return self.load_data(alfio.load_object, self.task_path, alf_object, **kwargs)

    def load_raw_passive_data(self, alf_object: str, **kwargs) -> Bunch[str, Any]:
        """
        Load raw passive data from local path.

        Returns
        -------
        Bunch
        """
        return self.load_data(alfio.load_object, self.raw_task_path, alf_object)

    def load_ephys_data(self, alf_object: str, **kwargs) -> Bunch[str, Any]:
        """
        Load ephys data from local path.

        Returns
        -------
        Bunch
        """
        return self.load_data(alfio.load_object, self.ephys_path, alf_object, **kwargs)

    def load_spikes_data(
            self,
            alf_object: str,
            attributes: list[str],
            **kwargs
    ) -> Bunch[str, Any]:
        """
        Load spike sorting data from local path.

        Returns
        -------
        Bunch
        """
        return self.load_data(
            alfio.load_object, self.spike_path, alf_object, attribute=attributes, **kwargs)


class SpikeGLXLoader(ABC):
    """
    Abstract base class for loading SpikeGLX metadata and AP band snippets.

    Subclasses must implement the follow abstract methods:
    - `load_meta_data`
    - `load_ap_data`

    Parameters
    ----------
    save_path : Path or None
        Directory where cached snippet data will be saved.
    """

    def __init__(self, save_path: Path | None = None):

        self.meta: Bunch | None = None
        self.save_path: Path | None = save_path
        self.cached_path: Path | None = save_path.joinpath('alignment_gui_raw_data_snippets.npy') \
            if save_path else None

    def get_meta_data(self) -> Bunch[str, Any]:
        """
        Load and parse metadata for the AP band.

        Returns
        -------
        Bunch
            A Bunch containing geometry info and an 'exists' flag.
        """
        self.meta = self.load_meta_data()
        if not self.meta:
            return Bunch({'exists': False})

        geometry = spikeglx.geometry_from_meta(self.meta, sort=True)
        return Bunch(geometry, exists=True)

    @abstractmethod
    def load_meta_data(self) -> dict | None:
        """Abstract method to load AP metadata."""

    @abstractmethod
    def load_ap_data(self) -> spikeglx.Reader | Streamer | None:
        """Abstract method to return a SpikeGLX reader or Streamer object."""

    def load_ap_snippets(self, twin: float = 1) -> Bunch[str, Any] | defaultdict[str, Any]:
        """
        Load AP snippets centered around selected time points.

        Parameters
        ----------
        twin : float
            Time window in seconds for each snippet.

        Returns
        -------
        data: Bunch
            Snippets of raw data for three timepoints in addition to metadata (exists, fs).
        """
        if self.cached_path and self.cached_path.exists():
            return np.load(self.cached_path, allow_pickle=True).item()

        sr = self.load_ap_data()
        if not sr:
            return Bunch(exists=False)

        data = defaultdict(Bunch)
        for t in self.get_time_snippets(sr):
            data['images'][t] = self._get_snippet(sr, t, twin=twin)

        data['exists'] = True
        data['fs'] = sr.fs

        if self.cached_path:
            np.save(self.cached_path, data)

        return data

    def _get_snippet(
            self,
            sr: spikeglx.Reader | Streamer,
            t: float, twin: float = 1
    ) -> np.ndarray:
        """
        Extract a snippet of AP data centered at time t.

        Parameters
        ----------
        sr : spikeglx.Reader or Streamer
            The raw data reader
        t : float
            Time in seconds for center of snippet.
        twin : float
            Time window (in seconds) to extract.

        Returns
        -------
        np.ndarray
            Snippet of raw data (time, channels)
        """
        start_sample = int(t * sr.fs)
        end_sample = start_sample + int(twin * sr.fs)
        raw = sr[start_sample:end_sample, :-sr.nsync].T

        # Detect bad channels and destripe
        channel_labels, _ = ibldsp.voltage.detect_bad_channels(raw, sr.fs)
        raw = ibldsp.voltage.destripe(raw, fs=sr.fs, h=sr.geometry,
                                      channel_labels=channel_labels)

        # Extract a window in time (450–500 ms)
        window = slice(int(0.450 * sr.fs), int(0.500 * sr.fs))
        return raw[:, window].T

    @staticmethod
    def get_time_snippets(sr: spikeglx.Reader, n: int = 3, pad: int = 200) -> np.ndarray:
        """
        Return n time points across the file duration, excluding pad seconds from start and end.

        Parameters
        ----------
        sr : spikeglx.Reader or Streamer
            The raw data reader
        n : int
            Number of time points to extract.
        pad : int
            Time (seconds) to exclude from beginning and end.

        Returns
        -------
        np.ndarray
            Array of time points in seconds.
        """
        file_duration = sr.meta['fileTimeSecs']
        pad = pad if file_duration > 500 else 0
        usable_time = file_duration - 2 * pad
        intervals = usable_time // n
        return intervals * (np.arange(n) + 1)


class SpikeGLXLoaderOne(SpikeGLXLoader):
    """
    SpikeGLX loader using ONE.

    Raw data is streamed via ONE

    Parameters
    ----------
    insertion : dict
        Dictionary representing a probe insertion (must include 'session' and 'name').
    one : ONE
        An ONE instance used to access data.
    session_path : Path or None
        Path to the session folder. If None, it is resolved using the eid via `one.eid2path`.
    force : bool
        If True, forces removal of cached data before streaming.
    """

    def __init__(
            self,
            insertion: dict,
            one: ONE,
            session_path: Path | None = None,
            force: bool = False):

        self.one: ONE = one
        self.eid: str = insertion['session']
        self.session_path: Path = session_path or self.one.eid2path(self.eid)
        self.pid: str = insertion['id']
        self.probe_label: str = insertion['name']
        self.force: bool = force
        save_path = self.session_path.joinpath(f'raw_ephys_data/{self.probe_label}')

        super().__init__(save_path)

    def load_meta_data(self) -> dict | None:
        """
        Load AP metadata using ONE.

        Returns
        -------
        dict or None
            A dict containing the spikeglx AP band metadata, or None if not found.
        """
        try:
            meta_file = self.one.load_dataset(
                self.eid, '*.ap.meta', collection=f'raw_ephys_data/{self.probe_label}',
                download_only=True)
            return spikeglx.read_meta_data(meta_file)
        except ALFObjectNotFound:
            return None

    def load_ap_data(self) -> Streamer:
        """
        Load AP data using ONE.

        Returns
        -------
        Streamer
            A streamer object for AP band.
        """
        return Streamer(pid=self.pid, one=self.one, remove_cached=self.force, typ='ap')


class SpikeGLXLoaderLocal(SpikeGLXLoader):
    """
    SpikeGLX loader using local file system.

    Parameters
    ----------
    probe_path : Path
        Root directory of probe data
    meta_collection : str
        Name of subfolder containing meta and binary files.
    """

    def __init__(self, probe_path: Path, meta_collection: str):

        self.meta_path: Path = probe_path.joinpath(meta_collection)

        super().__init__(self.meta_path)

    def load_meta_data(self) -> dict | None:
        """
        Load AP metadata from local path.

        Returns
        -------
        dict or None
            A dict containing the spikeglx AP band metadata, or None if not found.
        """
        meta_file = next(self.meta_path.glob('*.ap.*meta'), None)
        return spikeglx.read_meta_data(meta_file) if meta_file else None

    def load_ap_data(self) -> spikeglx.Reader | None:
        """
        Load binary AP data from local path.

        Returns
        -------
        spikeglx.Reader or None
            A spikeglx.Reader instance to load the raw data, or None if not found.
        """
        ap_file = next(self.meta_path.glob('*.ap.*bin'), None)
        return spikeglx.Reader(ap_file) if ap_file else None
