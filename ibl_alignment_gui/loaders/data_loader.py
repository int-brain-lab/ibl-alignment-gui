from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy
import spikeglx

import ibldsp.voltage
import one.alf.io as alfio
from ibl_alignment_gui.loaders.sources import (
    AlfReaderLocal,
    AlfReaderOne,
    EphysQCSourceAlf,
    PassiveSourceAlf,
    SortingSourceAlf,
    SortingSourceSI,
)
from iblutil.numerical import ismember
from iblutil.util import Bunch
from one.alf.exceptions import ALFObjectNotFound

try:
    import ephysatlas.data

    EPHYS_ATLAS = True
except ImportError:
    EPHYS_ATLAS = False

if TYPE_CHECKING:
    from brainbox.io.spikeglx import Streamer
    from ibl_alignment_gui.loaders.sources import EphysQCSource, PassiveSource, SortingSource
    from ibl_alignment_gui.utils.parse_yaml import DatasetPaths
    from one.api import ONE

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load the spike sorting, ephys QC and passive data of a probe for the ibl_alignment_gui.

    The data are read from one source of each kind, and only the data on the channels of the
    selected shank are returned.

    Parameters
    ----------
    sorting : SortingSource
        The source of the spikes, clusters and channels.
    ephys_qc : EphysQCSource
        The source of the RMS and power spectral density.
    passive : PassiveSource
        The source of the passive stimulus data.
    """

    def __init__(self, sorting: SortingSource, ephys_qc: EphysQCSource, passive: PassiveSource):
        self.sorting: SortingSource = sorting
        self.ephys_qc: EphysQCSource = ephys_qc
        self.passive: PassiveSource = passive
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
        data['rms_AP'] = self.get_rms_data('ephysTimeRmsAP')
        # Load in rms LF data
        data['rms_LF'] = self.get_rms_data('ephysTimeRmsLF')
        # Load in psd LF data
        data['psd_LF'] = self.get_psd_data('ephysSpectralDensityLF')
        # Load in passive data
        # TODO this data should be shared across probes
        data['rf_map'], data['pass_stim'], data['gabor'] = self.get_passive_data()

        return data

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
        return self.passive.load_rf_map(), self.passive.load_stims(), self.passive.load_gabor()

    def get_rms_data(self, alf_object: str) -> Bunch[str, Any]:
        """
        Load RMS data for specified band.

        Only returns data on channels present on selected shank.

        Parameters
        ----------
        alf_object : str
            The alf object to load

        Returns
        -------
        rms_data : Bunch
            RMS data
        """
        return self.filter_raw_by_chns(self.ephys_qc.load_rms(alf_object))

    def get_psd_data(self, alf_object: str) -> Bunch[str, Any]:
        """
        Load power spectral density data for specified band.

        Only returns data on channels present on selected shank.

        Parameters
        ----------
        alf_object : str
            The alf object to load

        Returns
        -------
        psd_data: Bunch
            PSD data
        """
        return self.filter_raw_by_chns(self.ephys_qc.load_psd(alf_object))

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
        spikes = self.sorting.load_spikes()
        clusters = self.sorting.load_clusters()
        channels = self.sorting.load_channels()

        if self.filter and spikes['exists']:
            # Remove low firing rate clusters
            spikes, clusters = self.filter_spikes_by_fr(spikes, clusters)

        if spikes['exists']:
            spikes, *_ = self.filter_spikes_by_chns(spikes, clusters, channels)

        return spikes, clusters, channels

    def filter_spikes_by_chns(
        self, spikes: Bunch[str, Any], clusters: Bunch[str, Any], channels: Bunch[str, Any]
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
        spikes_idx = np.isin(
            channels['rawInd'][clusters['channels'][spikes['clusters']]],
            self.shank_sites['spikes_ind'],
        )

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
            # Skip the exists flag, labels and 1D arrays such as timestamps
            if not isinstance(data[key], np.ndarray) or data[key].ndim == 1:
                continue
            # data[key] = data[key][:, self.shank_sites['raw_ind']]
            data[key] = self._safe_take(data[key], self.shank_sites['raw_ind'])

        return data

    @staticmethod
    def _safe_take(arr, indices, axis=1):
        """np.take along ``axis`` that fills out-of-bounds positions with NaN.

        Channel indices can exceed the data array (e.g. main-block RMS has fewer
        channels than the combined channel set). In-bounds indices are taken
        normally; out-of-bounds positions are returned as NaN.
        """
        indices = np.asarray(indices)
        max_idx = arr.shape[axis] - 1
        oob = indices > max_idx

        if not np.any(oob):
            return np.take(arr, indices, axis=axis)

        logger.warning(
            f'Channel indices exceed data size (max_idx={max_idx}, '
            f'max_chn_ind={indices.max()}). Filling {oob.sum()} channels with NaN.'
        )
        result = np.take(arr, np.clip(indices, 0, max_idx), axis=axis).astype(float)
        # Build a slicer that targets the OOB positions along `axis`.
        oob_slice = [slice(None)] * result.ndim
        oob_slice[axis] = oob
        result[tuple(oob_slice)] = np.nan
        return result

    @staticmethod
    def filter_spikes_by_fr(
        spikes: Bunch[str, Any], clusters: Bunch[str, Any], min_fr: float = 50 / 3600
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
        clusters['metrics'] = clusters['metrics'].reset_index(drop=True)
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

    def __init__(
        self,
        insertion: dict,
        one: ONE,
        session_path: Path | None = None,
        spike_collection: str | None = None,
    ):
        self.one: ONE = one
        self.eid: str = insertion['session']
        self.session_path: Path = session_path or one.eid2path(self.eid)
        self.probe_label: str = insertion['name']
        self.spike_collection: str | None = spike_collection
        self.probe_path: Path = self.get_spike_sorting_path()
        self.probe_collection: str = str(self.probe_path.relative_to(self.session_path))
        # TODO DataLoader.__init__ resets this to False, so ONE sessions are not rate filtered
        self.filter: bool = True

        super().__init__(
            SortingSourceAlf(AlfReaderOne(one, self.eid, collection=self.probe_collection)),
            EphysQCSourceAlf(
                AlfReaderOne(one, self.eid, collection=f'raw_ephys_data/{self.probe_label}')
            ),
            PassiveSourceAlf(AlfReaderOne(one, self.eid), AlfReaderOne(one, self.eid)),
        )

    @property
    def spike_sorting_path(self) -> Path:
        """
        Return the path to the folder holding the spike sorting data.

        Returns
        -------
        Path
            The path to the spike sorting folder.
        """
        return self.probe_path

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


class DataLoaderLocal(DataLoader):
    """
    Data loader using local file system.

    Data are loaded from files on disk, from the dataset paths resolved from the session yaml.

    Parameters
    ----------
    data_paths : DatasetPaths
        The resolved dataset paths for the probe, holding the spike sorting, processed ephys,
        task, raw task and raw ephys directories.
    """

    def __init__(self, data_paths: DatasetPaths):
        self.spike_path: Path = data_paths.spike_sorting
        self.ephys_path: Path = data_paths.processed_ephys
        self.task_path: Path = data_paths.task
        self.raw_task_path: Path = data_paths.raw_task
        self.meta_path: Path = data_paths.raw_ephys
        # A session can have no spike sorting, only the RMS and power spectra
        self.probe_collection: str | None = self.spike_path.name if self.spike_path else None

        super().__init__(
            SortingSourceAlf(_local_reader(self.spike_path)),
            EphysQCSourceAlf(_local_reader(self.ephys_path)),
            PassiveSourceAlf(_local_reader(self.task_path), _local_reader(self.raw_task_path)),
        )

    @property
    def spike_sorting_path(self) -> Path:
        """
        Return the path to the folder holding the spike sorting data.

        Returns
        -------
        Path
            The path to the spike sorting folder.
        """
        return self.spike_path


class DataLoaderSI(DataLoaderLocal):
    """
    Data loader using a SpikeInterface SortingAnalyzer on the local file system.

    The spike sorting is read from the SortingAnalyzer, and the RMS, power spectra and passive
    data from their ALF folders as in :class:`DataLoaderLocal`.

    Parameters
    ----------
    data_paths : DatasetPaths
        The resolved dataset paths for the probe, with the SortingAnalyzer as the
        ``spike_sorting``.
    """

    def __init__(self, data_paths: DatasetPaths):
        super().__init__(data_paths)
        self.sorting = SortingSourceSI(self.spike_path)


def _local_reader(path: Path | None) -> AlfReaderLocal | None:
    """Return a reader for a folder, or None when the dataset has no folder."""
    return AlfReaderLocal(path) if path is not None else None


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
        self.cached_ap_path: Path | None = (
            save_path.joinpath('alignment_gui_raw_data_snippets.npy') if save_path else None
        )
        self.cached_lf_path: Path | None = (
            save_path.joinpath('alignment_gui_raw_lf_data_snippets.npy') if save_path else None
        )

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

    @abstractmethod
    def load_lf_data(self) -> spikeglx.Reader | Streamer | None:
        """Abstract method to return a SpikeGLX reader or Streamer object."""

    def load_ap_snippets(self, twin: float = 1) -> Bunch[str, Any] | defaultdict[str, Any]:
        """
        Load AP snippets centered around selected time points.

        Also computes channel quality metrics across snippets to detect dead, noisy
        and outside channels.

        Parameters
        ----------
        twin : float
            Time window in seconds for each snippet.

        Returns
        -------
        data: Bunch
            Snippets of raw data for three timepoints in addition to metadata (exists, fs).
        """
        if self.cached_ap_path and self.cached_ap_path.exists():
            data = np.load(self.cached_ap_path, allow_pickle=True).item()
            if 'dead_channels' in data:
                return data

        sr = self.load_ap_data()
        if not sr:
            return Bunch(exists=False)

        times = self.get_time_snippets(sr)

        # Thresholds for channel quality detection
        detection_thresholds = {'similarity_threshold': (-0.5, 1), 'psd_hf_threshold': 0.02}

        data = defaultdict(Bunch)

        for i, t in enumerate(times):
            raw, labels, features = self._get_ap_snippet(sr, t, twin=twin, **detection_thresholds)

            if i == 0:
                chn_labels = np.zeros((raw.shape[1], len(times)))
                chn_features = {k: np.zeros((raw.shape[1], len(times))) for k in features}

            for k in features:
                chn_features[k][:, i] = features[k]

            chn_labels[:, i] = labels
            data['images'][t] = raw

        chn_features_med = {k: np.median(chn_features[k], axis=-1) for k in chn_features}
        channel_flags, _ = scipy.stats.mode(chn_labels, axis=1)

        data['dead_channels'] = Bunch(
            values=chn_features_med['xcor_hf'],
            lines=[detection_thresholds['similarity_threshold'][0]],
            points=channel_flags == 1,
        )

        data['noisy_channels_coherence'] = Bunch(
            values=chn_features_med['xcor_hf'],
            lines=[detection_thresholds['similarity_threshold'][1]],
            points=chn_features_med['xcor_hf'] > detection_thresholds['similarity_threshold'][1],
        )

        data['noisy_channels_psd'] = Bunch(
            values=chn_features_med['psd_hf'],
            lines=[detection_thresholds['psd_hf_threshold']],
            points=channel_flags == 2,
        )

        data['outside_channels'] = Bunch(
            values=chn_features_med['xcor_lf'], lines=[-0.75, 0.75], points=channel_flags == 3
        )

        data['exists'] = True
        data['fs'] = sr.fs

        if self.cached_ap_path:
            np.save(self.cached_ap_path, data)

        return data

    def load_lf_snippets(self, twin: float = 5) -> Bunch[str, Any] | defaultdict[str, Any]:
        """
        Load LF snippets centered around selected time points.

        Also computes channel quality metrics across snippets to detect dead, noisy
        and outside channels.

        Parameters
        ----------
        twin : float
            Time window in seconds for each snippet.

        Returns
        -------
        data: Bunch
            Snippets of raw data for three timepoints in addition to metadata (exists, fs).
        """
        if self.cached_lf_path and self.cached_lf_path.exists():
            data = np.load(self.cached_lf_path, allow_pickle=True).item()
            return data

        sr = self.load_lf_data()
        if not sr:
            return Bunch(exists=False)

        times = self.get_time_snippets(sr)

        data = defaultdict(Bunch)

        for t in times:
            raw = self._get_lf_snippet(sr, t, twin=twin)

            data['images'][t] = raw

        data['exists'] = True
        data['fs'] = sr.fs

        # TODO uncomment once we are happy with the snippet length
        # if self.cached_lf_path:
        #     np.save(self.cached_lf_path, data)

        return data

    @staticmethod
    def _get_ap_snippet(
        sr: spikeglx.Reader | Streamer, t: float, twin: float = 1, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        raw = sr[start_sample:end_sample, : -sr.nsync].T

        # Detect bad channels and destripe
        channel_labels, channel_features = ibldsp.voltage.detect_bad_channels(raw, sr.fs, **kwargs)
        raw = ibldsp.voltage.destripe(
            raw, fs=sr.fs, h=sr.geometry, channel_labels=channel_labels, k_filter=False
        )

        # Extract a window in time (450–500 ms)
        window = slice(int(0.45 * sr.fs), int(0.5 * sr.fs))
        return raw[:, window].T, channel_labels, channel_features

    @staticmethod
    def _get_lf_snippet(
        sr: spikeglx.Reader | Streamer, t: float, twin: float = 3, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a snippet of LF data centered at time t.

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
        raw = sr[start_sample:end_sample, : -sr.nsync].T

        butter_kwargs = {'N': 3, 'Wn': 2 / sr.fs * 2, 'btype': 'highpass'}
        # Detect bad channels and destripe
        raw = ibldsp.voltage.destripe(
            raw,
            fs=sr.fs,
            channel_labels=True,
            butter_kwargs=butter_kwargs,
            h=sr.geometry,
            k_filter=None,
        )

        # Extract a window in time (1–2 seconds)
        window = slice(int(0.5 * sr.fs), int(2.5 * sr.fs))
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
        self, insertion: dict, one: ONE, session_path: Path | None = None, force: bool = False
    ):
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
                self.eid,
                '*.ap.meta',
                collection=f'raw_ephys_data/{self.probe_label}',
                download_only=True,
            )
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
        import brainbox.io.spikeglx as spikeglx_io  # noqa: PLC0415

        return spikeglx_io.Streamer(pid=self.pid, one=self.one, remove_cached=self.force, typ='ap')

    def load_lf_data(self):
        """
        Load LF data using ONE.

        Returns
        -------
        Streamer
            A streamer object for LF band.
        """
        import brainbox.io.spikeglx as spikeglx_io  # noqa: PLC0415

        return spikeglx_io.Streamer(pid=self.pid, one=self.one, remove_cached=self.force, typ='lf')


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

    def __init__(self, meta_path: Path):
        self.meta_path: Path = meta_path

        super().__init__(self.meta_path)

    def load_meta_data(self) -> dict | None:
        """
        Load AP metadata from local path.

        Returns
        -------
        dict or None
            A dict containing the spikeglx AP band metadata, or None if not found.
        """
        if self.meta_path is None:  # no raw_ephys dataset
            return None
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
        if self.meta_path is None:
            return None
        ap_file = next(self.meta_path.glob('*.ap.*bin'), None)
        return spikeglx.Reader(ap_file) if ap_file else None

    def load_lf_data(self) -> spikeglx.Reader | None:
        """
        Load binary LF data from local path.

        Returns
        -------
        spikeglx.Reader or None
            A spikeglx.Reader instance to load the raw data, or None if not found.
        """
        if self.meta_path is None:
            return None
        lf_file = next(self.meta_path.glob('*.lf.*bin'), None)
        return spikeglx.Reader(lf_file) if lf_file else None


class FeatureLoader(ABC):
    """Abstract base class for loading ephys atlas features."""

    @abstractmethod
    def load_features(self, shank_sites: Bunch | None = None) -> Bunch[str, Any]:
        """Abstract method to load ephys atlas features.

        Parameters
        ----------
        shank_sites : Bunch or None
            The sites for the shank being loaded (see
            :meth:`GeometryLoader.get_sites_for_shank`). When given, implementations that read a
            combined multi-shank file use it to keep only this shank's channels.
        """


class FeatureLoaderOne(FeatureLoader):
    """
    Feature loader using ONE.

    Loads feature from ephys feature table.
    """

    def __init__(self, insertion: dict, one: ONE, feature_version: str, multi_area: bool = False):
        self.one: ONE = one
        self.pid: str = insertion['id']
        self.feature_version: str = feature_version
        self.multi_area: bool = multi_area

        if self.multi_area:
            self.feature_version = '2026_W26'

    def load_features(self, shank_sites: Bunch | None = None) -> Bunch[str, Any]:
        """
        Load ephys atlas features from ONE.

        Parameters
        ----------
        shank_sites : Bunch or None
            Unused here — the ONE table is already filtered to this insertion's ``pid``. Accepted
            to keep the :class:`FeatureLoader` interface uniform across loaders.

        Returns
        -------
        feature_data: Bunch
            A Bunch containing a dataFrame containing ephys atlas features for the probe.
        """
        table_path = self.one.cache_dir.joinpath('ephys_atlas_features')
        table_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.load_dataframe('ea_active', self.feature_version, table_path)

        if len(data) == 0 and self.multi_area:
            data = self.load_dataframe('multi_area_comm', self.feature_version, table_path)

        feature_data = Bunch(exists=False) if len(data) == 0 else Bunch(df=data, exists=True)

        return feature_data

    def download_features(self, project, feature, save_path: Path) -> None:
        """
        Download the latest ephys atlas features from S3.

        Parameters
        ----------
        project: str
            The project name
        feature: str
            The feature version
        save_path: Path
            A path to save the downloaded file
        """
        ephysatlas.data.download_tables(save_path, label=feature, project=project, one=self.one)

    def load_dataframe(self, project: str, feature: str, save_path: Path) -> pd.DataFrame:
        """
        Load the ephys-atlas feature table filtered to this insertion's ``pid``.

        Parameters
        ----------
        project : str
            The project name.
        feature : str
            The feature version label.
        save_path : Path
            Directory under which the ``<project>/<feature>`` tables are stored.

        Returns
        -------
        pd.DataFrame
            The feature table rows for this insertion's ``pid``.
        """
        data = self.read_full_dataframe(project, feature, save_path)
        if data.empty:  # the table is not available (it has no columns to filter on)
            return data
        # Record the table's vintage, so a model can tell which feature set these are (the key
        # ibleatools writes on the features it computes).
        data.attrs['features_version'] = feature

        return data[data['pid'] == self.pid]

    def read_full_dataframe(self, project: str, feature: str, save_path: Path) -> pd.DataFrame:
        """
        Read the full ephys-atlas feature table for all insertions, downloading if needed.

        Parameters
        ----------
        project : str
            The project name.
        feature : str
            The feature version label.
        save_path : Path
            Directory under which the ``<project>/<feature>`` tables are stored.

        Returns
        -------
        pd.DataFrame
            The full feature table across all insertions, or an empty frame if the table is not
            cached and cannot be downloaded (ephysatlas is not installed, or the release has no
            such table).
        """
        # Check for the table itself: a download that found nothing (or was interrupted) leaves
        # the folder behind, which must not be mistaken for a downloaded table.
        table = save_path.joinpath(project, feature, 'agg_full', 'df_all_cols_merged.pqt')

        if not table.exists():
            if not EPHYS_ATLAS:
                # Downloading the feature tables needs ephysatlas; without it (and no cached
                # tables) there are simply no features for this insertion.
                logger.warning(
                    'ephysatlas is not installed; cannot download ephys-atlas feature tables. '
                    'No features will be available for this insertion.'
                )
                return pd.DataFrame()
            try:
                self.download_features(project, feature, save_path)
            except AssertionError as exc:
                # ephysatlas.data.download_tables asserts that the table is on S3: a release
                # without this table (e.g. a vintage with no multi-area table) is not a failure,
                # just no features. Other errors (network, credentials) still raise.
                logger.debug('Feature table download found nothing: %s', exc)
            if not table.exists():
                logger.warning(
                    'The %s %s feature table is not available; no features from it.',
                    project,
                    feature,
                )
                return pd.DataFrame()

        return pd.read_parquet(table).reset_index()


class FeatureLoaderLocal(FeatureLoader):
    """
    Feature loader using the local file system.

    Mirrors :class:`FeatureLoaderOne` but reads a single per-channel feature parquet from disk
    instead of downloading the ephys-atlas feature table via ONE. Used in offline/yaml mode where
    the features file is supplied directly (e.g. by the local channel-prediction plugin dialog or
    the launcher script).

    Parameters
    ----------
    features_path : Path
        Path to a per-channel features parquet file (one row per channel).
    """

    def __init__(self, features_path: Path):
        self.features_path: Path = Path(features_path)

    def load_features(self, shank_sites: Bunch | None = None) -> Bunch[str, Any]:
        """
        Load the per-channel features parquet from disk.

        A single features file may hold every channel of a multi-shank probe. When ``shank_sites``
        is given, the rows are split down to this shank by matching the file's ``channel`` column
        against ``shank_sites['raw_ind']`` (the raw-data channel indices for the shank). For a
        single-shank file (or a file already restricted to this shank) this is a no-op.

        Parameters
        ----------
        shank_sites : Bunch or None
            The sites for the shank being loaded (see
            :meth:`GeometryLoader.get_sites_for_shank`). When None the full file is returned.

        Returns
        -------
        feature_data: Bunch
            A Bunch with ``df`` (the features DataFrame) and ``exists=True`` if the file was
            found and non-empty, otherwise ``Bunch(exists=False)``.
        """
        # The local file is flat (unlike the S3 multi-index table), so we neither filter by pid nor
        # reset the index here. ``infer_regions`` selects the columns it needs via the model's
        # FEATURES list, so the (per-shank) DataFrame is passed through as-is.
        if not self.features_path.is_file():
            logger.warning('Local features file not found: %s', self.features_path)
            return Bunch(exists=False)

        data = pd.read_parquet(self.features_path)

        if shank_sites is not None and 'channel' in data.columns:
            data = data[data['channel'].isin(shank_sites['raw_ind'])]

        return Bunch(exists=False) if len(data) == 0 else Bunch(df=data, exists=True)
