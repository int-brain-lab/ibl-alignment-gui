"""
Sources of the spike sorting, ephys QC and passive data shown in the GUI.

A source reads one kind of data from one place and returns it as a Bunch in the format the
plots expect, with an ``exists`` flag that is False when the data cannot be found. The
``DataLoader`` classes combine one source of each kind. A ``RawSource`` reads one band of the
raw recording for the ``RawDataLoader``.

The ALF sources read ALF objects through an ``AlfReader``, which hides whether the objects
come from a folder on disk or through ONE. The SpikeInterface sources read a saved
``SortingAnalyzer``; SpikeInterface is an optional dependency and is only imported when one is
loaded.
"""

from __future__ import annotations

import logging
import re
import traceback
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import one.alf.io as alfio
from ibl_alignment_gui.utils.optional import SPIKEINTERFACE_EXTRA_HINT, has_spikeinterface
from iblutil.util import Bunch
from one.alf.exceptions import ALFMultipleCollectionsFound, ALFObjectNotFound

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import spikeglx
    from spikeinterface import BaseRecording, SortingAnalyzer

    from one.api import ONE

logger = logging.getLogger(__name__)

SPIKES_ATTRIBUTES = ['depths', 'amps', 'times', 'clusters']
CLUSTERS_ATTRIBUTES = ['metrics', 'peakToTrough', 'waveforms', 'channels']
CHANNELS_ATTRIBUTES = ['rawInd', 'localCoordinates']
# The probe contact id of each channel of the raw recording, in file order, written with the RMS
# and spectra computed from a SpikeInterface recording
RAW_CONTACT_IDS_FILE = '_iblqc_ephysChannels.contactIds.npy'

# The SortingAnalyzer extensions needed to show the spikes and clusters
SI_REQUIRED_EXTENSIONS = ['templates', 'spike_amplitudes']
# The quality metrics query that labels a unit as IBL good, as in SpikeInterface's
# export_to_ibl_gui
SI_GOOD_UNITS_QUERY = (
    'amplitude_median < -40 and isi_violations_ratio < 0.5 and amplitude_cutoff < 0.2'
)


def load_alf(
    load_function: Callable,
    *args: Any,
    raise_message: str | None = None,
    raise_exception: type[Exception] | tuple[type[Exception], ...] = (
        ALFObjectNotFound,
        ALFMultipleCollectionsFound,
    ),
    raise_error: bool = False,
    **kwargs,
) -> Bunch[str, Any]:
    """
    Safely load data using a provided function.

    Parameters
    ----------
    load_function : Callable
        Function to load the data (e.g., ONE or alfio).
    args : tuple
        Arguments for the loading function; the second is the ALF object name.
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
        if raise_message is None:
            if isinstance(e, ALFMultipleCollectionsFound):
                raise_message = (
                    f'{alf_object} data matches multiple collections ({e}); '
                    f'cannot disambiguate, some plots will not display'
                )
            else:
                raise_message = f'{alf_object} data was not found, some plots will not display'
        logger.warning(raise_message)
        if raise_error:
            logger.error(raise_message)
            logger.error(traceback.format_exc())
            raise e
        return Bunch(exists=False)


# -------------------------------------------------------------------------------------------------
# ALF readers
# -------------------------------------------------------------------------------------------------


class AlfReader(ABC):
    """Abstract base class for reading ALF objects from one location."""

    @abstractmethod
    def load(self, alf_object: str, attribute: list[str] | None = None) -> Bunch[str, Any]:
        """
        Load an ALF object.

        Parameters
        ----------
        alf_object : str
            The ALF object name, e.g. 'spikes'.
        attribute : list of str or None
            The attributes to load, or None for all of them.

        Returns
        -------
        Bunch
            The object's attributes with an 'exists' flag, False when it is not found.
        """


class AlfReaderLocal(AlfReader):
    """
    Read ALF objects from a folder on disk.

    Parameters
    ----------
    path : Path
        The folder holding the ALF files.
    """

    def __init__(self, path: Path):
        self.path: Path = path

    def load(self, alf_object: str, attribute: list[str] | None = None) -> Bunch[str, Any]:
        """
        Load an ALF object from the folder.

        Parameters
        ----------
        alf_object : str
            The ALF object name, e.g. 'spikes'.
        attribute : list of str or None
            The attributes to load, or None for all of them.

        Returns
        -------
        Bunch
            The object's attributes with an 'exists' flag, False when it is not found.
        """
        kwargs = {'attribute': attribute} if attribute is not None else {}
        return load_alf(alfio.load_object, self.path, alf_object, **kwargs)


class AlfReaderOne(AlfReader):
    """
    Read ALF objects of a session through ONE.

    Parameters
    ----------
    one : ONE
        An ONE instance used to access data.
    eid : str
        The session id.
    collection : str or None
        The collection to load from, or None to let ONE search the whole session.
    """

    def __init__(self, one: ONE, eid: str, collection: str | None = None):
        self.one: ONE = one
        self.eid: str = eid
        self.collection: str | None = collection

    def load(self, alf_object: str, attribute: list[str] | None = None) -> Bunch[str, Any]:
        """
        Load an ALF object through ONE.

        Parameters
        ----------
        alf_object : str
            The ALF object name, e.g. 'spikes'.
        attribute : list of str or None
            The attributes to load, or None for all of them.

        Returns
        -------
        Bunch
            The object's attributes with an 'exists' flag, False when it is not found.
        """
        kwargs = {}
        if self.collection is not None:
            kwargs['collection'] = self.collection
        if attribute is not None:
            kwargs['attribute'] = attribute
        return load_alf(self.one.load_object, self.eid, alf_object, **kwargs)


# -------------------------------------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------------------------------------


class SortingSource(ABC):
    """Abstract base class for the spike sorting data of a probe."""

    @abstractmethod
    def load_spikes(self) -> Bunch[str, Any]:
        """Load the spikes: depths, amps, times and clusters."""

    @abstractmethod
    def load_clusters(self) -> Bunch[str, Any]:
        """Load the clusters: metrics, peakToTrough, waveforms and channels."""

    @abstractmethod
    def load_channels(self) -> Bunch[str, Any]:
        """Load the channels: rawInd and localCoordinates."""

    def release(self) -> None:  # noqa: B027
        """Free what was kept to load the objects, once they have been loaded."""


class EphysQCSource(ABC):
    """Abstract base class for the RMS and power spectral density of a probe's raw data."""

    @abstractmethod
    def load_rms(self, alf_object: str) -> Bunch[str, Any]:
        """
        Load the RMS of one band over time.

        Parameters
        ----------
        alf_object : str
            The ALF object holding the RMS, e.g. 'ephysTimeRmsAP'.

        Returns
        -------
        Bunch
            rms (time, channels), timestamps, the xaxis label and an 'exists' flag.
        """

    @abstractmethod
    def load_psd(self, alf_object: str) -> Bunch[str, Any]:
        """
        Load the power spectral density of one band.

        Parameters
        ----------
        alf_object : str
            The ALF object holding the power, e.g. 'ephysSpectralDensityLF'.

        Returns
        -------
        Bunch
            power (frequencies, channels), freqs and an 'exists' flag.
        """


class PassiveSource(ABC):
    """Abstract base class for the passive stimulus data of a session."""

    @abstractmethod
    def load_rf_map(self) -> Bunch[str, Any]:
        """Load the receptive field mapping times and stimulus frames."""

    @abstractmethod
    def load_stims(self) -> Bunch[str, Any]:
        """Load the passive visual stimulus times."""

    @abstractmethod
    def load_gabor(self) -> Bunch[str, Any]:
        """Load the start times of the left and right gabor stimuli."""


class RawSource(ABC):
    """
    Abstract base class for one band (AP or LF) of a probe's raw recording.

    The channels are given in file order: column i of the traces and entry i of the geometry
    are channel i of the raw file.
    """

    @property
    @abstractmethod
    def fs(self) -> float:
        """The sampling frequency (Hz)."""

    @property
    @abstractmethod
    def duration(self) -> float:
        """The duration of the recording (s)."""

    @property
    def n_samples(self) -> int:
        """The number of samples in the recording."""
        return round(self.duration * self.fs)

    @property
    @abstractmethod
    def geometry(self) -> dict[str, np.ndarray]:
        """The channel header in file order: at least x, y, shank and sample_shift."""

    @abstractmethod
    def get_traces(self, start: int, n_samples: int) -> np.ndarray:
        """
        Read a block of the recording.

        Parameters
        ----------
        start : int
            The first sample.
        n_samples : int
            The number of samples.

        Returns
        -------
        np.ndarray
            The traces (time, channels) in V, in file order and without the sync channels.
        """


class SortingSourceAlf(SortingSource):
    """
    Spike sorting data from ALF objects.

    Parameters
    ----------
    reader : AlfReader or None
        Reader for the spike sorting collection, or None when the session has no spike sorting.
    """

    def __init__(self, reader: AlfReader | None):
        self.reader: AlfReader | None = reader

    def _load(self, alf_object: str, attribute: list[str]) -> Bunch[str, Any]:
        if self.reader is None:
            return Bunch(exists=False)
        return self.reader.load(alf_object, attribute=attribute)

    def load_spikes(self) -> Bunch[str, Any]:
        """
        Load the spikes object.

        Returns
        -------
        Bunch
            depths, amps, times and clusters with an 'exists' flag.
        """
        return self._load('spikes', SPIKES_ATTRIBUTES)

    def load_clusters(self) -> Bunch[str, Any]:
        """
        Load the clusters object.

        Returns
        -------
        Bunch
            metrics, peakToTrough, waveforms and channels with an 'exists' flag.
        """
        return self._load('clusters', CLUSTERS_ATTRIBUTES)

    def load_channels(self) -> Bunch[str, Any]:
        """
        Load the channels object.

        Returns
        -------
        Bunch
            rawInd and localCoordinates with an 'exists' flag.
        """
        return self._load('channels', CHANNELS_ATTRIBUTES)


class EphysQCSourceAlf(EphysQCSource):
    """
    RMS and power spectral density from the ``_iblqc_`` ALF objects.

    Parameters
    ----------
    reader : AlfReader or None
        Reader for the processed ephys collection, or None when there is none.
    """

    def __init__(self, reader: AlfReader | None):
        self.reader: AlfReader | None = reader

    def _load(self, alf_object: str) -> Bunch[str, Any]:
        if self.reader is None:
            return Bunch(exists=False)
        return self.reader.load(alf_object)

    def load_rms(self, alf_object: str) -> Bunch[str, Any]:
        """
        Load the RMS of one band over time.

        Older files name the RMS 'amps', and some have no timestamps; the x axis is then in
        time samples.

        Parameters
        ----------
        alf_object : str
            The ALF object holding the RMS, e.g. 'ephysTimeRmsAP'.

        Returns
        -------
        Bunch
            rms (time, channels), timestamps, the xaxis label and an 'exists' flag.
        """
        rms_data = self._load(alf_object)
        if rms_data['exists']:
            if 'amps' in rms_data:
                rms_data['rms'] = rms_data.pop('amps')
            if 'timestamps' not in rms_data:
                rms_data['timestamps'] = np.array([0, rms_data['rms'].shape[0]])
                rms_data['xaxis'] = 'Time samples'
            else:
                rms_data['xaxis'] = 'Time (s)'
        return rms_data

    def load_psd(self, alf_object: str) -> Bunch[str, Any]:
        """
        Load the power spectral density of one band.

        Older files name the power 'amps'.

        Parameters
        ----------
        alf_object : str
            The ALF object holding the power, e.g. 'ephysSpectralDensityLF'.

        Returns
        -------
        Bunch
            power (frequencies, channels), freqs and an 'exists' flag.
        """
        psd_data = self._load(alf_object)
        if psd_data['exists'] and 'amps' in psd_data:
            psd_data['power'] = psd_data.pop('amps')
        return psd_data


class PassiveSourceAlf(PassiveSource):
    """
    Passive stimulus data from ALF objects.

    Parameters
    ----------
    reader : AlfReader or None
        Reader for the passive task collection, or None when there is none.
    raw_reader : AlfReader or None
        Reader for the raw passive collection holding the RF map stimulus, or None.
    """

    def __init__(self, reader: AlfReader | None, raw_reader: AlfReader | None):
        self.reader: AlfReader | None = reader
        self.raw_reader: AlfReader | None = raw_reader

    @staticmethod
    def _load(reader: AlfReader | None, alf_object: str) -> Bunch[str, Any]:
        if reader is None:
            return Bunch(exists=False)
        return reader.load(alf_object)

    def load_rf_map(self) -> Bunch[str, Any]:
        """
        Load the receptive field mapping times and stimulus frames.

        Returns
        -------
        Bunch
            The passiveRFM attributes and the frames (n_frames, 15, 15), with an 'exists' flag.
        """
        if self.reader is None and self.raw_reader is None:
            return Bunch(exists=False)
        try:
            rf_data = self._load(self.reader, 'passiveRFM')
            frame_path = self._load(self.raw_reader, 'RFMapStim')
            frames = np.fromfile(frame_path['raw'], dtype='uint8')
            rf_data['frames'] = np.transpose(
                np.reshape(frames, [15, 15, -1], order='F'), [2, 1, 0]
            )
        except Exception:
            logger.warning('passiveRFM data was not found, some plots will not display')
            rf_data = Bunch(exists=False)
        return rf_data

    def load_stims(self) -> Bunch[str, Any]:
        """
        Load the passive visual stimulus times.

        Returns
        -------
        Bunch
            The passiveStims attributes with an 'exists' flag.
        """
        return self._load(self.reader, 'passiveStims')

    def load_gabor(self) -> Bunch[str, Any]:
        """
        Load the start times of the left and right gabor stimuli.

        Only gabors with a contrast above 0.1 are kept.

        Returns
        -------
        Bunch
            leftGabor and rightGabor start times with an 'exists' flag.
        """
        try:
            gabor = self._load(self.reader, 'passiveGabor')
            if not gabor['exists']:
                return Bunch(exists=False)
            vis_stim = Bunch()
            vis_stim['leftGabor'] = np.array(
                gabor['start'][(gabor['position'] == 35) & (gabor['contrast'] > 0.1)],
                dtype=np.float64,
            )
            vis_stim['rightGabor'] = np.array(
                gabor['start'][(gabor['position'] == -35) & (gabor['contrast'] > 0.1)],
                dtype=np.float64,
            )
            vis_stim['exists'] = True
        except Exception:
            logger.warning('Failed to process passiveGabor data, some plots will not display')
            vis_stim = Bunch(exists=False)
        return vis_stim


class SortingSourceSI(SortingSource):
    """
    Spike sorting data from a saved SpikeInterface SortingAnalyzer.

    The objects are returned as AlfBunch, the same as the ALF objects, as the plots use its
    to_df method.

    The analyzer must have a single segment and the templates and spike_amplitudes extensions.
    The spike_locations, template_metrics and quality_metrics extensions are used when they have
    been computed.

    Parameters
    ----------
    path : Path or None
        Path to the SortingAnalyzer folder or zarr, or None when the session has no spike sorting.
    good_units_query : str or None
        Query on the quality metrics that labels a unit as IBL good. When None, the units are not
        labelled.
    raw_qc_path : Path or None
        The folder of the datasets computed from the raw recording, whose probe contact ids
        (``_iblqc_ephysChannels.contactIds``) give the raw channel of each analyzer channel.
    """

    def __init__(
        self,
        path: Path | None,
        good_units_query: str | None = SI_GOOD_UNITS_QUERY,
        raw_qc_path: Path | None = None,
    ):
        if path is not None and not has_spikeinterface():
            raise ImportError(
                f'{path} is a SpikeInterface SortingAnalyzer. {SPIKEINTERFACE_EXTRA_HINT}'
            )
        self.path: Path | None = path
        self.good_units_query: str | None = good_units_query
        self.raw_qc_path: Path | None = raw_qc_path
        self._analyzer: SortingAnalyzer | None = None
        self._missing_extensions: list[str] = []

    @property
    def analyzer(self) -> SortingAnalyzer:
        """The SortingAnalyzer, loaded on first use."""
        if self._analyzer is None:
            import spikeinterface as si  # noqa: PLC0415

            analyzer = si.load(self.path)
            if analyzer.get_num_segments() != 1:
                raise ValueError(
                    f'The SortingAnalyzer at {self.path} has {analyzer.get_num_segments()} '
                    f'segments, only single segment analyzers are supported'
                )
            self._missing_extensions = [
                ext for ext in SI_REQUIRED_EXTENSIONS if not analyzer.has_extension(ext)
            ]
            if self._missing_extensions:
                logger.warning(
                    'The SortingAnalyzer at %s is missing the %s extensions, compute them with '
                    'analyzer.compute() to show the spikes and clusters',
                    self.path,
                    ', '.join(self._missing_extensions),
                )
            self._analyzer = analyzer
        return self._analyzer

    def release(self) -> None:
        """
        Free the SortingAnalyzer once the spikes, clusters and channels have been loaded.

        Each shank loads its own source, so an analyzer kept by each would hold a copy of the
        sorting per shank. It is loaded again if an object is loaded after this.
        """
        self._analyzer = None

    def _has_sorting(self) -> bool:
        """Whether there is an analyzer with the extensions needed for the spikes and clusters."""
        if self.path is None:
            return False
        return self.analyzer is not None and not self._missing_extensions

    def load_spikes(self) -> Bunch[str, Any]:
        """
        Load the spikes.

        The depths come from the spike_locations extension when computed, otherwise from the
        depth of the main channel of each spike's unit.

        Returns
        -------
        Bunch
            depths (um), amps (V), times (s) and clusters with an 'exists' flag.
        """
        if not self._has_sorting():
            return Bunch(exists=False)
        analyzer = self.analyzer
        spikes = analyzer.sorting.to_spike_vector()

        # Both take the start time, and a recording's time vector, into account
        if analyzer.has_recording():
            times = analyzer.recording.sample_index_to_time(spikes['sample_index'])
        else:
            times = analyzer.sorting.sample_index_to_time(spikes['sample_index'])

        if analyzer.has_extension('spike_locations'):
            depths = analyzer.get_extension('spike_locations').get_data()['y']
        else:
            main_channels = analyzer.get_main_channels()
            depths = analyzer.get_channel_locations()[main_channels[spikes['unit_index']], 1]

        # The amplitudes are signed in uV, the plots expect positive amplitudes in V
        amps = np.abs(analyzer.get_extension('spike_amplitudes').get_data()) * 1e-6

        return alfio.AlfBunch(
            depths=depths,
            amps=amps,
            times=times,
            clusters=spikes['unit_index'].astype(np.int32),
            exists=True,
        )

    def load_clusters(self) -> Bunch[str, Any]:
        """
        Load the clusters.

        Returns
        -------
        Bunch
            metrics, peakToTrough (ms), waveforms (units, samples, 1) on the main channel in V and
            channels with an 'exists' flag.
        """
        if not self._has_sorting():
            return Bunch(exists=False)
        analyzer = self.analyzer
        main_channels = analyzer.get_main_channels()

        templates = analyzer.get_extension('templates').get_data()
        waveforms = templates[np.arange(analyzer.get_num_units()), :, main_channels] * 1e-6

        peak_to_trough = None
        if analyzer.has_extension('template_metrics'):
            template_metrics = analyzer.get_extension('template_metrics').get_data()
            # peak_to_valley is the name used by older SpikeInterface versions
            for column in ('peak_to_trough_duration', 'peak_to_valley'):
                if column in template_metrics.columns:
                    peak_to_trough = template_metrics[column].to_numpy() * 1e3
                    break
        if peak_to_trough is None:
            peak_to_trough = (
                (np.argmax(waveforms, axis=1) - np.argmin(waveforms, axis=1))
                / analyzer.sampling_frequency
                * 1e3
            )

        return alfio.AlfBunch(
            metrics=self._load_metrics(),
            peakToTrough=peak_to_trough,
            waveforms=waveforms[:, :, np.newaxis],
            channels=main_channels.astype(np.int64),
            exists=True,
        )

    def _load_metrics(self) -> pd.DataFrame:
        """
        Build the cluster metrics table from the quality metrics and the sorting properties.

        Returns
        -------
        pd.DataFrame
            One row per unit with cluster_id, firing_rate and the quality metrics, plus label
            (IBL good) and ks2_label (Kilosort label) when they can be set.
        """
        analyzer = self.analyzer
        metrics = pd.DataFrame({'cluster_id': analyzer.unit_ids})
        if analyzer.has_extension('quality_metrics'):
            quality_metrics = analyzer.get_extension('quality_metrics').get_data()
            metrics = pd.concat([metrics, quality_metrics.reset_index(drop=True)], axis=1)

        if 'firing_rate' not in metrics.columns:
            n_spikes = np.bincount(
                analyzer.sorting.to_spike_vector()['unit_index'],
                minlength=analyzer.get_num_units(),
            )
            metrics['firing_rate'] = n_spikes / analyzer.get_total_duration()

        if self.good_units_query is not None:
            try:
                metrics['label'] = metrics.eval(self.good_units_query).astype(int)
            except pd.errors.UndefinedVariableError as err:
                logger.warning(
                    'Units are not labelled as IBL good, the quality metrics needed by the '
                    'query %r have not been computed: %s',
                    self.good_units_query,
                    err,
                )

        if 'KSLabel' in analyzer.sorting.get_property_keys():
            metrics['ks2_label'] = analyzer.sorting.get_property('KSLabel')

        return metrics

    def load_channels(self) -> Bunch[str, Any]:
        """
        Load the channels.

        The raw indices are found from the probe contact ids, matched to those of the raw
        recording when they were saved with the raw data QC, or read from SpikeGLX channel ids
        (e.g. 'imec0.ap#AP12'). Both give the channel in the raw file even when channels were
        removed before sorting. Otherwise they are the channel order of the analyzer.

        Returns
        -------
        Bunch
            rawInd, localCoordinates (um) and, when the probe sets them, the shank of each
            channel, with an 'exists' flag.
        """
        if self.path is None:
            return Bunch(exists=False)
        analyzer = self.analyzer
        channels = alfio.AlfBunch(
            rawInd=raw_channel_indices(
                analyzer.channel_ids, channel_contact_ids(analyzer), self.load_raw_contact_ids()
            ),
            localCoordinates=analyzer.get_channel_locations(),
            exists=True,
        )
        shank_ids = analyzer.get_probe().shank_ids
        if shank_ids is not None and np.all(shank_ids != ''):
            channels['shank'] = np.asarray(shank_ids)
        return channels

    def load_raw_contact_ids(self) -> np.ndarray | None:
        """
        Load the probe contact ids of the raw recording, saved with the raw data QC.

        Returns
        -------
        np.ndarray or None
            The contact id of each channel of the raw recording in file order, or None when they
            were not saved.
        """
        if self.raw_qc_path is None:
            return None
        file = self.raw_qc_path.joinpath(RAW_CONTACT_IDS_FILE)
        return np.load(file) if file.exists() else None


def channel_contact_ids(extractor: BaseRecording | SortingAnalyzer) -> np.ndarray | None:
    """
    Get the probe contact id of each channel of a recording or analyzer.

    probeinterface gives Neuropixels contacts the same ids whichever reader opened the recording
    (e.g. 'e12' or 's1e288' on multi-shank probes), and SpikeInterface keeps them when channels
    are removed, so they match the channels of a sorting to those of its raw recording.

    Parameters
    ----------
    extractor : BaseRecording or SortingAnalyzer
        The recording or analyzer.

    Returns
    -------
    np.ndarray or None
        The contact id of each channel, in channel order, or None when there is no probe or it
        does not set the ids of all its contacts.
    """
    try:
        probe = extractor.get_probe()
    except ValueError:  # no probe, or more than one
        return None
    if probe is None or probe.device_channel_indices is None:
        return None
    contact_ids = np.asarray(probe.contact_ids).astype(str)
    if contact_ids.size != extractor.get_num_channels() or np.any(contact_ids == ''):
        return None
    ids = np.empty(contact_ids.size, dtype=contact_ids.dtype)
    ids[probe.device_channel_indices] = contact_ids
    return ids


def raw_channel_indices(
    channel_ids: np.ndarray,
    contact_ids: np.ndarray | None = None,
    raw_contact_ids: np.ndarray | None = None,
) -> np.ndarray:
    """
    Get the index of each channel in the raw recording file.

    Parameters
    ----------
    channel_ids : np.ndarray
        The SpikeInterface channel ids.
    contact_ids : np.ndarray or None
        The probe contact id of each channel.
    raw_contact_ids : np.ndarray or None
        The probe contact id of each channel of the raw recording, in file order.

    Returns
    -------
    np.ndarray
        The position of each channel's contact in the raw recording when both contact ids are
        given, otherwise the channel numbers of SpikeGLX ids (e.g. 'imec0.ap#AP12' gives 12),
        otherwise the position of each channel.

    Raises
    ------
    ValueError
        If a channel's contact is not in the raw recording.
    """
    if contact_ids is not None and raw_contact_ids is not None:
        raw_position = {contact_id: i for i, contact_id in enumerate(raw_contact_ids)}
        missing = [contact_id for contact_id in contact_ids if contact_id not in raw_position]
        if missing:
            raise ValueError(
                f'{len(missing)} channels of the spike sorting are not in the raw recording the '
                f'QC was computed from (contacts {", ".join(missing[:5])}), was it computed from '
                f'another recording?'
            )
        return np.array([raw_position[contact_id] for contact_id in contact_ids])
    if raw_contact_ids is not None:
        logger.warning(
            'The raw recording saved its probe contact ids but the spike sorting has none, so '
            'its channels are matched by their ids'
        )
    matches = [re.search(r'#(?:AP|LF)(\d+)$', str(channel_id)) for channel_id in channel_ids]
    if all(matches):
        return np.array([int(match.group(1)) for match in matches])
    return np.arange(len(channel_ids))


class RawSourceSpikeGLX(RawSource):
    """
    One band of a SpikeGLX recording, read with a spikeglx.Reader or a ONE Streamer.

    The reader is opened when the data are first needed, so a stream is only started when the
    raw data are read.

    Parameters
    ----------
    open_reader : Callable
        Returns the spikeglx.Reader or Streamer of the band.
    """

    def __init__(self, open_reader: Callable[[], spikeglx.Reader]):
        self._open_reader: Callable[[], spikeglx.Reader] = open_reader
        self._reader: spikeglx.Reader | None = None

    @property
    def reader(self) -> spikeglx.Reader:
        """The spikeglx.Reader or Streamer, opened on first use."""
        if self._reader is None:
            self._reader = self._open_reader()
        return self._reader

    @property
    def fs(self) -> float:
        """The sampling frequency (Hz)."""
        return self.reader.fs

    @property
    def duration(self) -> float:
        """The duration of the recording (s)."""
        return self.reader.meta['fileTimeSecs']

    @property
    def n_samples(self) -> int:
        """The number of samples in the recording."""
        return self.reader.ns

    @cached_property
    def _file_order(self) -> np.ndarray:
        """The reader's columns in file order; the reader sorts them by position on the probe."""
        n_channels = self.reader.nc - self.reader.nsync
        order = getattr(self.reader, 'raw_channel_order', None)
        if order is None:
            return np.arange(n_channels)
        return np.argsort(order[:n_channels])

    @property
    def geometry(self) -> dict[str, np.ndarray]:
        """The channel header in file order, read from the metadata."""
        return {k: np.asarray(v)[self._file_order] for k, v in self.reader.geometry.items()}

    def get_traces(self, start: int, n_samples: int) -> np.ndarray:
        """
        Read a block of the recording.

        Parameters
        ----------
        start : int
            The first sample.
        n_samples : int
            The number of samples.

        Returns
        -------
        np.ndarray
            The traces (time, channels) in V, in file order and without the sync channels.
        """
        traces = self.reader[start : start + n_samples, : -self.reader.nsync]
        return traces[:, self._file_order]


class RawSourceSI(RawSource):
    """
    One band of a raw recording read with SpikeInterface.

    The recording must be raw, as read from disk: the snippets and channel flags are computed by
    destriping it, which is wrong on data that are already filtered or referenced. The channels
    are in the order of the recording, which for a recording read from disk is file order.

    The recording is opened when the data are first needed.

    Parameters
    ----------
    open_recording : Callable
        Returns the SpikeInterface recording of the band.
    """

    def __init__(self, open_recording: Callable[[], BaseRecording]):
        self._open_recording: Callable[[], BaseRecording] = open_recording
        self._recording: BaseRecording | None = None

    @property
    def recording(self) -> BaseRecording:
        """The SpikeInterface recording, opened on first use."""
        if self._recording is None:
            recording = self._open_recording()
            if recording.get_num_segments() != 1:
                raise ValueError(
                    f'The recording has {recording.get_num_segments()} segments, only single '
                    f'segment recordings are supported'
                )
            self._recording = recording
        return self._recording

    @property
    def fs(self) -> float:
        """The sampling frequency (Hz)."""
        return self.recording.get_sampling_frequency()

    @property
    def duration(self) -> float:
        """
        The duration of the recording (s).

        Computed from the number of samples, as for SpikeGLX: SpikeInterface computes it from its
        times, which can fall a float error short and move a snippet by a sample.
        """
        return self.n_samples / self.fs

    @property
    def n_samples(self) -> int:
        """The number of samples in the recording."""
        return self.recording.get_num_samples()

    @cached_property
    def geometry(self) -> dict[str, np.ndarray]:
        """
        The channel header in file order, read from the probe.

        The shanks are numbered by their position on the probe. The sample shift is the
        probe's inter_sample_shift, zero when the recording does not give it.
        """
        recording = self.recording
        n_channels = recording.get_num_channels()
        locations = recording.get_channel_locations()
        x, y = locations[:, 0], locations[:, 1]

        sample_shift = recording.get_property('inter_sample_shift')
        if sample_shift is None:
            sample_shift = np.zeros(n_channels)

        return {
            'x': x,
            'y': y,
            'shank': shank_numbers(recording.get_probe().shank_ids, x),
            'sample_shift': np.asarray(sample_shift, dtype=float),
        }

    def get_traces(self, start: int, n_samples: int) -> np.ndarray:
        """
        Read a block of the recording.

        Parameters
        ----------
        start : int
            The first sample.
        n_samples : int
            The number of samples.

        Returns
        -------
        np.ndarray
            The traces (time, channels) in V, in file order.
        """
        end = min(start + n_samples, self.recording.get_num_samples())
        traces = self.recording.get_traces(start_frame=start, end_frame=end, return_in_uV=True)
        return traces * 1e-6


def shank_numbers(shank_ids: np.ndarray | None, x: np.ndarray) -> np.ndarray:
    """
    Get the shank of each channel, numbering the shanks by their position on the probe.

    SpikeInterface shank ids can be strings, or empty when the probe does not set them, so they
    are not used as numbers.

    Parameters
    ----------
    shank_ids : np.ndarray or None
        The SpikeInterface shank id of each channel.
    x : np.ndarray
        The x coordinate of each channel.

    Returns
    -------
    np.ndarray
        The shank of each channel, 0 for the leftmost shank.
    """
    if shank_ids is None or np.any(np.asarray(shank_ids) == ''):
        return np.zeros(x.size, dtype=int)
    shank_ids = np.asarray(shank_ids)
    shanks = sorted(np.unique(shank_ids), key=lambda sh: np.min(x[shank_ids == sh]))
    numbers = np.zeros(x.size, dtype=int)
    for i, sh in enumerate(shanks):
        numbers[shank_ids == sh] = i
    return numbers
