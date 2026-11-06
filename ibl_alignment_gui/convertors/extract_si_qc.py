"""
Write the datasets the alignment GUI computes from the raw data of a SpikeInterface recording.

For one probe this writes, to the output folder:

- the RMS of the AP and LF bands and the spectral density of the LF band, as the
  ``_iblqc_ephysTimeRms*`` and ``_iblqc_ephysSpectralDensity*`` ALF files;
- snippets of the AP and LF bands, with the channels flagged as dead, noisy or outside the
  brain, as the files the GUI caches them in.

The recordings are opened with the SpikeInterface reader of the acquisition system, and must be
raw, as read from disk. The data are processed as by
:mod:`ibl_alignment_gui.convertors.extract_ephys`, SpikeInterface only reads them.

Examples
--------
A SpikeGLX recording, with an LF file:

>>> import spikeinterface.extractors as se
>>> ap = se.read_spikeglx(folder, stream_id='imec0.ap')
>>> lf = se.read_spikeglx(folder, stream_id='imec0.lf')
>>> extract_si_qc(ap, out_folder, lf_recording=lf)

A compressed SpikeGLX recording of an NP2 probe, which records wideband, so that the LF band is
computed from the AP band:

>>> ap = se.read_cbin_ibl(cbin_file_path=folder / 'recording.ap.cbin')
>>> extract_si_qc(ap, out_folder, lfp_from_ap=True)

An Open Ephys recording of an NP1 probe:

>>> ap = se.read_openephys(folder, stream_name='Record Node 101#Neuropix-PXI-100.ProbeA')
>>> lf = se.read_openephys(folder, stream_name='Record Node 101#Neuropix-PXI-100.ProbeA-LFP')
>>> extract_si_qc(ap, out_folder, lf_recording=lf)
"""

import logging
from pathlib import Path

import numpy as np

from ibl_alignment_gui.convertors.extract_ephys import (
    LF_DECIMATION,
    N_WORKERS,
    SAMPLE_SPACING_SECS,
    compute_rmsmap,
    save_rmsmap,
)
from ibl_alignment_gui.loaders.data_loader import RawDataLoader
from ibl_alignment_gui.loaders.sources import RawSourceSI

_logger = logging.getLogger(__name__)

# The sampling frequency of the LF band, and of the AP band decimated to it
LF_FS = 2500


def extract_si_qc(
    ap_recording,
    out_folder,
    lf_recording=None,
    lfp_from_ap=False,
    sample_spacing_secs=SAMPLE_SPACING_SECS,
    n_workers=N_WORKERS,
):
    """
    Write the RMS, spectral density and snippets of a probe from its SpikeInterface recordings.

    Parameters
    ----------
    ap_recording : spikeinterface.BaseRecording
        The raw AP band of the probe.
    out_folder : str or pathlib.Path
        The folder to write the datasets to, created if it does not exist.
    lf_recording : spikeinterface.BaseRecording or None
        The raw LF band of the probe. When None, the LF datasets are only written if
        ``lfp_from_ap`` is set.
    lfp_from_ap : bool
        Whether to compute the LF band from the AP band, for probes that record wideband, such
        as NP2 probes. The AP band is decimated to 2.5 kHz.
    sample_spacing_secs : float or None
        The RMS and spectra are computed on a sample taken every ``sample_spacing_secs``
        seconds. None reads the whole recording.
    n_workers : int
        The number of threads the RMS windows are shared between, each reading from a clone of
        the recording.

    Raises
    ------
    ValueError
        If a recording is filtered, if both ``lf_recording`` and ``lfp_from_ap`` are given, or
        if ``lfp_from_ap`` is set for an AP band that does not decimate to 2.5 kHz.
    """
    for recording in (ap_recording, lf_recording):
        if recording is not None and recording.is_filtered():
            raise ValueError(
                f'{recording} is filtered, the raw recording, as read from disk, is needed'
            )
    if lf_recording is not None and lfp_from_ap:
        raise ValueError('Give either an LF recording or lfp_from_ap, not both')
    if lfp_from_ap and abs(ap_recording.get_sampling_frequency() / LF_DECIMATION - LF_FS) > 10:
        raise ValueError(
            f'The AP band is sampled at {ap_recording.get_sampling_frequency()} Hz, so it does '
            f'not decimate to the {LF_FS} Hz of the LF band'
        )

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    kwargs = {'sample_spacing_secs': sample_spacing_secs, 'n_workers': n_workers}

    _logger.info('Computing the AP RMS')
    rms = compute_rmsmap(_source_opener(ap_recording), 'ap', spectra=False, **kwargs)
    save_rmsmap(rms, out_folder, 'ap', spectra=False)
    if lf_recording is not None:
        _logger.info('Computing the LF RMS and spectral density')
        rms = compute_rmsmap(_source_opener(lf_recording), 'lf', **kwargs)
        save_rmsmap(rms, out_folder, 'lf')
    elif lfp_from_ap:
        _logger.info('Computing the LF RMS and spectral density from the AP band')
        rms = compute_rmsmap(_source_opener(ap_recording), 'ap', lfp_from_ap=True, **kwargs)
        save_rmsmap(rms, out_folder, 'lf')

    # The snippets are computed without a save path, so that cached snippets are not reused
    _logger.info('Computing the snippets')
    loader = RawDataLoader(
        RawSourceSI(lambda: ap_recording),
        RawSourceSI(lambda: lf_recording) if lf_recording is not None else None,
    )
    np.save(out_folder.joinpath(RawDataLoader.AP_SNIPPETS_FILE), loader.load_ap_snippets())
    if lf_recording is not None:
        np.save(out_folder.joinpath(RawDataLoader.LF_SNIPPETS_FILE), loader.load_lf_snippets())


def _source_opener(recording):
    """
    Get a function that opens a source of a recording, for each thread computing the RMS.

    The first source reads from the recording, the next ones from clones of it, which open the
    files again, so that the threads do not share a reader.

    Parameters
    ----------
    recording : spikeinterface.BaseRecording
        The recording.

    Returns
    -------
    Callable[[], RawSourceSI]
        Opens a source of the recording.
    """
    opened = []

    def open_source():
        current = recording.clone() if opened else recording
        opened.append(current)
        return RawSourceSI(lambda: current)

    return open_source
