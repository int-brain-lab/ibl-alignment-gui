"""
Sources of the spike sorting, ephys QC and passive data shown in the GUI.

A source reads one kind of data from one place and returns it as a Bunch in the format the
plots expect, with an ``exists`` flag that is False when the data cannot be found. The
``DataLoader`` classes combine one source of each kind.

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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import one.alf.io as alfio
from iblutil.util import Bunch
from one.alf.exceptions import ALFMultipleCollectionsFound, ALFObjectNotFound

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spikeinterface import SortingAnalyzer

    from one.api import ONE

logger = logging.getLogger(__name__)

SPIKES_ATTRIBUTES = ['depths', 'amps', 'times', 'clusters']
CLUSTERS_ATTRIBUTES = ['metrics', 'peakToTrough', 'waveforms', 'channels']
CHANNELS_ATTRIBUTES = ['rawInd', 'localCoordinates']

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
    """

    def __init__(self, path: Path | None, good_units_query: str | None = SI_GOOD_UNITS_QUERY):
        self.path: Path | None = path
        self.good_units_query: str | None = good_units_query
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

        samples = spikes['sample_index']
        if analyzer.has_recording() and analyzer.recording.has_time_vector():
            times = analyzer.recording.get_times()[samples]
        else:
            times = samples / analyzer.sampling_frequency

        if analyzer.has_extension('spike_locations'):
            depths = analyzer.get_extension('spike_locations').get_data()['y']
        else:
            main_channels = analyzer.get_main_channels()
            depths = analyzer.get_channel_locations()[main_channels[spikes['unit_index']], 1]

        # The amplitudes are signed in uV, the plots expect positive amplitudes in V
        amps = np.abs(analyzer.get_extension('spike_amplitudes').get_data()) * 1e-6

        return Bunch(
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

        return Bunch(
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

        The raw indices are read from SpikeGLX channel ids (e.g. 'imec0.ap#AP12'), which give
        the channel in the raw file even when channels were removed before sorting. For other
        recordings they are the channel order of the analyzer.

        Returns
        -------
        Bunch
            rawInd and localCoordinates (um) with an 'exists' flag.
        """
        if self.path is None:
            return Bunch(exists=False)
        analyzer = self.analyzer
        return Bunch(
            rawInd=raw_channel_indices(analyzer.channel_ids),
            localCoordinates=analyzer.get_channel_locations(),
            exists=True,
        )


def raw_channel_indices(channel_ids: np.ndarray) -> np.ndarray:
    """
    Get the index of each channel in the raw recording file.

    Parameters
    ----------
    channel_ids : np.ndarray
        The SpikeInterface channel ids.

    Returns
    -------
    np.ndarray
        The channel numbers of SpikeGLX ids (e.g. 'imec0.ap#AP12' gives 12), otherwise the
        position of each channel.
    """
    matches = [re.search(r'#(?:AP|LF)(\d+)$', str(channel_id)) for channel_id in channel_ids]
    if all(matches):
        return np.array([int(match.group(1)) for match in matches])
    return np.arange(len(channel_ids))
