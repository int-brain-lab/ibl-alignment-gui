import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import spikeglx
from scipy import signal
from tqdm import tqdm

import one.alf.io as alfio
from ibl_alignment_gui.loaders.sources import RawSourceSpikeGLX
from ibldsp import fourier, utils, voltage

_logger = logging.getLogger(__name__)


RMS_WIN_LENGTH_SECS = 3
WELCH_WIN_LENGTH_SAMPLES = 1024
# The RMS and spectra are computed on a sample of the recording taken every
# SAMPLE_SPACING_SECS, of AP_SAMPLE_LENGTH_SECS for the AP band and LF_SAMPLE_LENGTH_SECS for
# the LF band, which is enough for the plots and reads a fraction of the file
SAMPLE_SPACING_SECS = 10
AP_SAMPLE_LENGTH_SECS = 1
LF_SAMPLE_LENGTH_SECS = 3
# Data read either side of each sample so that the filters settle before it. The 300 Hz
# high-pass of the AP band settles within milliseconds, the 1 Hz high-pass of the LF band needs
# much longer.
AP_PAD_SECS = 0.1
LF_PAD_SECS = 0.5
# The LF band of NP2 probes, which record wideband, is the AP band decimated from 30 to 2.5 kHz
LF_DECIMATION = 12
# The windows are processed in parallel, by up to this many threads
N_WORKERS = min(8, os.cpu_count() or 1)


def rmsmap(fbin, spectra=True, sample_spacing_secs=None, lfp_from_ap=False, n_workers=N_WORKERS):
    """
    Compute the RMS map in the time domain and the spectra for each channel of a SpikeGLX file.

    Parameters
    ----------
    fbin : str, pathlib.Path or spikeglx.Reader
        Binary file in SpikeGLX format; the attached metadata is looked up alongside it.
    spectra : bool
        Whether to compute the power spectrum, which is only needed for LF data.
    sample_spacing_secs : float or None
        When given, the RMS and spectra are computed on a sample taken every
        ``sample_spacing_secs`` seconds, of ``AP_SAMPLE_LENGTH_SECS`` for the AP band and
        ``LF_SAMPLE_LENGTH_SECS`` for the LF band. When None, the whole recording is read in
        windows of about ``RMS_WIN_LENGTH_SECS``.
    lfp_from_ap : bool
        Whether to compute the LF band from an AP file, for NP2 probes that record wideband
        and have no LF file. The AP data are decimated to the LF sampling rate.
    n_workers : int
        The number of threads the windows are shared between, each with its own reader. The
        filtering and FFTs release the GIL, so the threads run in parallel.

    Returns
    -------
    dict
        See :func:`compute_rmsmap`.
    """
    given = isinstance(fbin, spikeglx.Reader)
    file_bin = fbin.file_bin if given else fbin
    if given:
        band = fbin.type
    else:
        with spikeglx.Reader(file_bin) as reader:
            band = reader.type
    readers = []

    def open_source():
        # A reader passed in is used when there is one thread, each thread opens its own
        # otherwise, as reading a compressed file is not thread safe
        if given and n_workers == 1:
            return RawSourceSpikeGLX(lambda: fbin)
        reader = spikeglx.Reader(file_bin, sort=False)
        readers.append(reader)
        return RawSourceSpikeGLX(lambda: reader)

    try:
        return compute_rmsmap(
            open_source,
            band=band,
            spectra=spectra,
            sample_spacing_secs=sample_spacing_secs,
            lfp_from_ap=lfp_from_ap,
            n_workers=n_workers,
        )
    finally:
        for reader in readers:
            reader.close()


def compute_rmsmap(
    open_source,
    band,
    spectra=True,
    sample_spacing_secs=None,
    lfp_from_ap=False,
    n_workers=N_WORKERS,
):
    """
    Compute the RMS map in the time domain and the spectra for each channel of a raw recording.

    Parameters
    ----------
    open_source : Callable[[], RawSource]
        Returns the band of the recording to read. Each thread calls it once, so that threads
        do not share a reader.
    band : str
        The band of the recording, 'ap' or 'lf'.
    spectra : bool
        Whether to compute the power spectrum, which is only needed for LF data.
    sample_spacing_secs : float or None
        When given, the RMS and spectra are computed on a sample taken every
        ``sample_spacing_secs`` seconds, of ``AP_SAMPLE_LENGTH_SECS`` for the AP band and
        ``LF_SAMPLE_LENGTH_SECS`` for the LF band. When None, the whole recording is read in
        windows of about ``RMS_WIN_LENGTH_SECS``.
    lfp_from_ap : bool
        Whether to compute the LF band from the AP band, for NP2 probes that record wideband.
        The AP data are decimated to the LF sampling rate.
    n_workers : int
        The number of threads the windows are shared between. The filtering and FFTs release
        the GIL, so the threads run in parallel.

    Returns
    -------
    dict
        The RMS amplitudes in channel-time space, the spectral density in channel-frequency
        space averaged over the windows, the time and frequency scales, and the raw file index
        of each channel.

    Notes
    -----
    The channels are in file order, as a RawSource gives them, so that they can be indexed with
    the raw channel indices (``channels.rawInd``) used by the GUI. The sync channels are left
    out.

    The AP band is destriped before computing the RMS: high-passed at 300 Hz, corrected for the
    sample shift of the ADCs and re-referenced to the median of the channels of each shank. The
    LF band is high-passed at 1 Hz.
    """
    source = open_source()
    n_channels = source.geometry['x'].size
    lf_band = lfp_from_ap or band == 'lf'
    fs = source.fs / (LF_DECIMATION if lfp_from_ap else 1)
    length_secs = LF_SAMPLE_LENGTH_SECS if lf_band else AP_SAMPLE_LENGTH_SECS
    windows = _rms_windows(source.n_samples, source.fs, sample_spacing_secs, length_secs)

    # pre-allocate output dictionary of numpy arrays
    win = {
        'TRMS': np.zeros((len(windows), n_channels)),
        'nsamples': np.zeros((len(windows),)),
        'fscale': fourier.fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / fs, one_sided=True),
        # the middle of each window
        'tscale': np.array([first + (last - first - 1) / 2 for first, last in windows])
        / source.fs,
        'raw_ind': np.arange(n_channels),
    }
    win['spectral_density'] = np.zeros((len(win['fscale']), n_channels))
    n_spectra = 0

    # A few batches per thread, so that the threads finish together
    batches = [b for b in np.array_split(np.arange(len(windows)), n_workers * 4) if b.size]
    kwargs = {'spectra': spectra, 'lfp_from_ap': lfp_from_ap, 'lf_band': lf_band}
    with tqdm(total=len(windows)) as pbar:
        if n_workers == 1:
            results = (
                (batch, _process_windows(source, [windows[i] for i in batch], **kwargs))
                for batch in batches
            )
        else:
            pool = ThreadPoolExecutor(n_workers)
            # Each thread reads from its own source
            local = threading.local()

            def process(windows_batch):
                if not hasattr(local, 'source'):
                    local.source = open_source()
                return _process_windows(local.source, windows_batch, **kwargs)

            futures = {
                pool.submit(process, [windows[i] for i in batch]): batch for batch in batches
            }
            results = ((futures[future], future.result()) for future in as_completed(futures))
        for batch, (rms, nsamples, spectral_density, n) in results:
            win['TRMS'][batch, :] = rms
            win['nsamples'][batch] = nsamples
            win['spectral_density'] += spectral_density
            n_spectra += n
            pbar.update(batch.size)
        if n_workers > 1:
            pool.shutdown()

    # The spectral density is averaged, so it does not depend on the number of windows
    win['spectral_density'] /= max(n_spectra, 1)
    return win


def _process_windows(source, windows, spectra=True, lfp_from_ap=False, lf_band=False):
    """
    Compute the RMS and the summed spectral density of windows of a recording.

    Parameters
    ----------
    source : RawSource
        The band of the recording.
    windows : list of tuple of int
        The first and last (exclusive) sample of each window.
    spectra : bool
        Whether to compute the power spectrum.
    lfp_from_ap : bool
        Whether to compute the LF band from the AP band.
    lf_band : bool
        Whether the data are of the LF band, read or computed from the AP band.

    Returns
    -------
    rms : np.ndarray
        The RMS of each channel in each window, (windows, channels).
    nsamples : np.ndarray
        The number of samples of each window.
    spectral_density : np.ndarray
        The spectral density summed over the windows, (frequencies, channels).
    n_spectra : int
        The number of windows summed into the spectral density.
    """
    # The channel geometry in file order, for the destriping
    geometry = source.geometry
    n_channels = geometry['x'].size
    n_samples = source.n_samples

    decimation = LF_DECIMATION if lfp_from_ap else 1
    fs = source.fs / decimation
    # The padding is a whole number of decimated samples
    pad_secs = LF_PAD_SECS if lf_band else AP_PAD_SECS
    pad = int(pad_secs * source.fs) // decimation * decimation

    rms = np.zeros((len(windows), n_channels))
    nsamples = np.zeros(len(windows))
    n_freqs = fourier.fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / fs, one_sided=True).size
    spectral_density = np.zeros((n_freqs, n_channels))
    n_spectra = 0
    for iw, (first, last) in enumerate(windows):
        start, stop = max(first - pad, 0), min(last + pad, n_samples)
        data = source.get_traces(start, stop - start).transpose()
        if lfp_from_ap:
            # The FIR filter of decimate keeps the AP band out of the LF band
            data = signal.decimate(data, decimation, ftype='fir', zero_phase=True, axis=-1)
        if lf_band:
            # remove low frequency noise below 1 Hz
            data = fourier.hp(data.astype(np.float32), 1 / fs, [0, 1])
        else:
            data = voltage.destripe(data, fs=source.fs, h=geometry, k_filter=False)
        # Drop the padding, now that the filters have settled
        data = data[:, (first - start) // decimation : (last - start) // decimation]
        rms[iw, :] = utils.rms(data)
        nsamples[iw] = data.shape[1]
        # the last window may be smaller than what is needed for welch
        if spectra and data.shape[1] >= WELCH_WIN_LENGTH_SAMPLES:
            # compute a smoothed spectrum using welch method
            _, w = signal.welch(
                data,
                fs=fs,
                window='hann',
                nperseg=WELCH_WIN_LENGTH_SAMPLES,
                detrend='constant',
                return_onesided=True,
                scaling='density',
                axis=-1,
            )
            spectral_density += w.T
            n_spectra += 1

    return rms, nsamples, spectral_density, n_spectra


def _rms_windows(ns, fs, sample_spacing_secs=None, length_secs=AP_SAMPLE_LENGTH_SECS):
    """
    Get the first and last sample of each window the RMS is computed on.

    Parameters
    ----------
    ns : int
        The number of samples in the recording.
    fs : float
        The sampling frequency (Hz).
    sample_spacing_secs : float or None
        The time between the starts of the samples. When None, the recording is split into
        consecutive windows of about ``RMS_WIN_LENGTH_SECS``.
    length_secs : float
        The length of each sample.

    Returns
    -------
    list of tuple of int
        The first and last (exclusive) sample of each window.
    """
    if sample_spacing_secs is None:
        nswin = 2 ** np.ceil(np.log2(fs * RMS_WIN_LENGTH_SECS))
        return list(utils.WindowGenerator(ns=ns, nswin=nswin, overlap=0).firstlast)
    length = min(int(length_secs * fs), ns)
    starts = np.arange(0, ns - length + 1, int(sample_spacing_secs * fs))
    return [(int(first), int(first) + length) for first in starts]


def _is_np2(bin_file):
    """
    Whether a SpikeGLX file was recorded with a Neuropixels 2.0 probe, read from its metadata.

    Parameters
    ----------
    bin_file : str or pathlib.Path
        The binary file, with its metadata alongside it.

    Returns
    -------
    bool
        True for NP2 probes, which record the AP and LF bands together in the AP file.
    """
    meta = spikeglx.read_meta_data(Path(bin_file).with_suffix('.meta'))
    return str(spikeglx._get_neuropixel_version_from_meta(meta)).startswith('NP2')


def extract_rmsmap(
    fbin,
    out_folder=None,
    spectra=True,
    sample_spacing_secs=SAMPLE_SPACING_SECS,
    lfp_from_ap=False,
    n_workers=N_WORKERS,
):
    """
    Write the _ibl_ephysRmsMap and _ibl_ephysSpectra ALF files for a binary file.

    Parameters
    ----------
    fbin : str or pathlib.Path
        Binary file in SpikeGLX format; the attached metadata is looked up alongside it.
    out_folder : str, pathlib.Path or None
        Folder to write the ALF files to. Defaults to the folder the ``fbin`` file lives in.
    spectra : bool
        Whether to compute the power spectrum, which is only needed for LF data.
    sample_spacing_secs : float or None
        The RMS and spectra are computed on a sample taken every ``sample_spacing_secs``
        seconds, which is enough for the plots and reads a fraction of the file. None reads the
        whole recording.
    lfp_from_ap : bool
        Whether to compute the LF band from an AP file, for NP2 probes that have no LF file.
    n_workers : int
        The number of threads the windows are shared between.

    Raises
    ------
    ValueError
        If ``lfp_from_ap`` is set for a file that is not from an NP2 probe, whose AP band holds
        no LFP.
    """
    if lfp_from_ap and not _is_np2(fbin):
        raise ValueError(f'{fbin} is not from an NP2 probe, so its AP band holds no LFP')
    _logger.info(f'Computing QC for {fbin}')
    with spikeglx.Reader(fbin) as reader:
        band = 'lf' if lfp_from_ap else reader.type
    out_folder = Path(fbin).parent if out_folder is None else Path(out_folder)

    rms = rmsmap(
        fbin,
        spectra=spectra,
        sample_spacing_secs=sample_spacing_secs,
        lfp_from_ap=lfp_from_ap,
        n_workers=n_workers,
    )
    save_rmsmap(rms, out_folder, band, spectra=spectra)


def save_rmsmap(rms, out_folder, band, spectra=True):
    """
    Write the _iblqc_ephysTimeRms and _iblqc_ephysSpectralDensity ALF files of a band.

    Parameters
    ----------
    rms : dict
        The RMS and spectra, as returned by :func:`compute_rmsmap`.
    out_folder : str or pathlib.Path
        Folder to write the ALF files to, created if it does not exist.
    band : str
        The band the files are saved as, 'ap' or 'lf'.
    spectra : bool
        Whether to write the spectral density.
    """
    out_folder = Path(out_folder)
    alf_object_time = f'ephysTimeRms{band.upper()}'
    alf_object_freq = f'ephysSpectralDensity{band.upper()}'
    # output ALF files, single precision
    out_folder.mkdir(parents=True, exist_ok=True)
    # The raw index of each channel is saved with the data, (1, channels) so that it broadcasts
    # over the windows and frequencies. It records that the channels are in file order, which
    # older versions of this function did not write them in.
    raw_ind = rms['raw_ind'][np.newaxis, :].astype(np.int32)
    tdict = {
        'rms': rms['TRMS'].astype(np.single),
        'timestamps': rms['tscale'].astype(np.single),
        'rawInd': raw_ind,
    }
    alfio.save_object_npy(out_folder, object=alf_object_time, dico=tdict, namespace='iblqc')
    if spectra:
        fdict = {
            'power': rms['spectral_density'].astype(np.single),
            'freqs': rms['fscale'].astype(np.single),
            'rawInd': raw_ind,
        }
        alfio.save_object_npy(out_folder, object=alf_object_freq, dico=fdict, namespace='iblqc')


def _sample2v(ap_file):
    """
    Return the factor that converts the raw AP data of a file to Volts.

    Parameters
    ----------
    ap_file : pathlib.Path
        Path to the AP binary file, whose metadata holds the conversion factors.

    Returns
    -------
    float
        The sample to Volt conversion factor for the AP band.
    """
    md = spikeglx.read_meta_data(ap_file.with_suffix('.meta'))
    s2v = spikeglx._conversion_sample2v_from_meta(md)
    return s2v['ap'][0]


def ks2_to_alf(ks_path, bin_path, out_path, bin_file=None, ampfactor=1, label=None, force=True):
    """
    Convert Kilosort 2 output to an ALF dataset for single probe data.

    Parameters
    ----------
    ks_path : pathlib.Path
        Path of the Kilosort output.
    bin_path : pathlib.Path
        Path of the raw data.
    out_path : pathlib.Path
        Path to write the ALF dataset to.
    bin_file : pathlib.Path or None
        The specific binary file to read, when the path holds more than one.
    ampfactor : float
        Factor converting the spike amplitudes to Volts.
    label : str or None
        Optional label added to the dataset names.
    force : bool
        Whether to overwrite an existing dataset.
    """
    from phylib.io import alf  # noqa: PLC0415

    from ibllib.ephys import ephysqc  # noqa: PLC0415

    m = ephysqc.phy_model_from_ks2_path(ks2_path=ks_path, bin_path=bin_path, bin_file=bin_file)
    ac = alf.EphysAlfCreator(m)
    ac.convert(out_path, label=label, force=force, ampfactor=ampfactor)

    # set depths to spike_depths to catch cases where it can't be computed from the pc
    # features (e.g. in the case of KS3)
    m.depths = np.load(out_path.joinpath('spikes.depths.npy'))
    ephysqc.spike_sorting_metrics_ks2(ks_path, m, save=True, save_path=out_path)


def extract_data(ks_path, ephys_path, out_path):
    """
    Convert a Kilosort output and its raw data into the datasets the alignment GUI reads.

    Writes the ALF spike sorting datasets, and the RMS and spectral density files for each AP and
    LF binary file found alongside them. For NP2 recordings without an LF file, as told by their
    metadata, the LF RMS and spectral density are computed from the AP file.

    Parameters
    ----------
    ks_path : pathlib.Path
        Path of the Kilosort output.
    ephys_path : pathlib.Path
        Path of the raw ephys data.
    out_path : pathlib.Path
        Path to write the datasets to. Must differ from ``ks_path`` so that the Kilosort output
        is not overwritten.
    """
    efiles = spikeglx.glob_ephys_files(ephys_path)

    for efile in efiles:
        if efile.get('ap') and efile.ap.exists():
            ks2_to_alf(
                ks_path,
                ephys_path,
                out_path,
                bin_file=efile.ap,
                ampfactor=_sample2v(efile.ap),
                label=None,
                force=True,
            )

            extract_rmsmap(efile.ap, out_folder=out_path, spectra=False)
            # NP2 probes record wideband, without an LF file, so the LF band is the AP band
            if not (efile.get('lf') and efile.lf.exists()) and _is_np2(efile.ap):
                extract_rmsmap(efile.ap, out_folder=out_path, lfp_from_ap=True)
        if efile.get('lf') and efile.lf.exists():
            extract_rmsmap(efile.lf, out_folder=out_path)


# if __name__ == '__main__':
#
#    ephys_path = Path('C:/Users/Mayo/Downloads/raw_ephys_data')
#    ks_path = Path('C:/Users/Mayo/Downloads/KS2')
#    out_path = Path('C:/Users/Mayo/Downloads/alf')
#    extract_data(ks_path, ephys_path, out_path)
