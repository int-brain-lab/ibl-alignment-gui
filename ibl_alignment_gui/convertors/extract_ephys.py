import logging
from pathlib import Path

import numpy as np
import spikeglx
from phylib.io import alf
from scipy import signal
from tqdm import tqdm

import one.alf.io as alfio
from ibldsp import fourier, utils
from ibllib.ephys import ephysqc

_logger = logging.getLogger(__name__)


RMS_WIN_LENGTH_SECS = 3
WELCH_WIN_LENGTH_SAMPLES = 1024


def rmsmap(fbin, spectra=True):
    """
    Compute the RMS map in the time domain and the spectra for each channel of the probe.

    Parameters
    ----------
    fbin : str, pathlib.Path or spikeglx.Reader
        Binary file in SpikeGLX format; the attached metadata is looked up alongside it.
    spectra : bool
        Whether to compute the power spectrum, which is only needed for LF data.

    Returns
    -------
    dict
        The RMS amplitudes in channel-time space, the spectral density in channel-frequency
        space, and the time and frequency scales.
    """
    if not isinstance(fbin, spikeglx.Reader):
        sglx = spikeglx.Reader(fbin)
        sglx.open()
    rms_win_length_samples = 2 ** np.ceil(np.log2(sglx.fs * RMS_WIN_LENGTH_SECS))
    # the window generator will generates window indices
    wingen = utils.WindowGenerator(ns=sglx.ns, nswin=rms_win_length_samples, overlap=0)
    # pre-allocate output dictionary of numpy arrays
    win = {'TRMS': np.zeros((wingen.nwin, sglx.nc)),
           'nsamples': np.zeros((wingen.nwin,)),
           'fscale': fourier.fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / sglx.fs, one_sided=True),
           'tscale': wingen.tscale(fs=sglx.fs)}
    win['spectral_density'] = np.zeros((len(win['fscale']), sglx.nc))
    # loop through the whole session
    with tqdm(total=wingen.nwin) as pbar:
        for first, last in wingen.firstlast:
            data = sglx.read_samples(first_sample=first, last_sample=last)[0].transpose()
            # remove low frequency noise below 1 Hz
            data = fourier.hp(data, 1 / sglx.fs, [0, 1])
            iw = wingen.iw
            win['TRMS'][iw, :] = utils.rms(data)
            win['nsamples'][iw] = data.shape[1]
            if spectra:
                # the last window may be smaller than what is needed for welch
                if last - first < WELCH_WIN_LENGTH_SAMPLES:
                    continue
                # compute a smoothed spectrum using welch method
                _, w = signal.welch(
                    data, fs=sglx.fs, window='hann', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                    detrend='constant', return_onesided=True, scaling='density', axis=-1
                )
                win['spectral_density'] += w.T
            # print at least every 20 windows
            if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
                pbar.update(iw)

    sglx.close()
    return win


def extract_rmsmap(fbin, out_folder=None, spectra=True):
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
    """
    _logger.info(f"Computing QC for {fbin}")
    sglx = spikeglx.Reader(fbin)
    # check if output ALF files exist already:
    out_folder = Path(fbin).parent if out_folder is None else Path(out_folder)
    alf_object_time = f'ephysTimeRms{sglx.type.upper()}'
    alf_object_freq = f'ephysSpectralDensity{sglx.type.upper()}'

    # crunch numbers
    rms = rmsmap(fbin, spectra=spectra)
    # output ALF files, single precision with the optional label as suffix before extension
    if not out_folder.exists():
        out_folder.mkdir()
    tdict = {'rms': rms['TRMS'].astype(np.single), 'timestamps': rms['tscale'].astype(np.single)}
    alfio.save_object_npy(out_folder, object=alf_object_time, dico=tdict, namespace='iblqc')
    if spectra:
        fdict = {'power': rms['spectral_density'].astype(np.single),
                 'freqs': rms['fscale'].astype(np.single)}
        alfio.save_object_npy(
            out_folder, object=alf_object_freq, dico=fdict, namespace='iblqc')


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
    LF binary file found alongside them.

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
            ks2_to_alf(ks_path, ephys_path, out_path, bin_file=efile.ap,
                       ampfactor=_sample2v(efile.ap), label=None, force=True)

            extract_rmsmap(efile.ap, out_folder=out_path, spectra=False)
        if efile.get('lf') and efile.lf.exists():
            extract_rmsmap(efile.lf, out_folder=out_path)


# if __name__ == '__main__':
#
#    ephys_path = Path('C:/Users/Mayo/Downloads/raw_ephys_data')
#    ks_path = Path('C:/Users/Mayo/Downloads/KS2')
#    out_path = Path('C:/Users/Mayo/Downloads/alf')
#    extract_data(ks_path, ephys_path, out_path)
