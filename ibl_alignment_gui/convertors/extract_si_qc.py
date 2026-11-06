"""
Write the datasets the alignment GUI computes from the raw data, reading it with SpikeInterface.

For a probe recorded with SpikeGLX this writes, to the output folder:

- the RMS of the AP and LF bands and the spectral density of the LF band, as the
  ``_iblqc_ephysTimeRms*`` and ``_iblqc_ephysSpectralDensity*`` ALF files;
- snippets of the AP and LF bands, with the channels flagged as dead, noisy or outside the
  brain, as the files the GUI caches them in.

The data are processed as by :mod:`ibl_alignment_gui.convertors.extract_ephys`, SpikeInterface
only reads them. For NP2 probes, which record wideband without an LF file, the LF RMS and
spectral density are computed from the AP band.

Usage::

    python -m ibl_alignment_gui.convertors.extract_si_qc PROBE_FOLDER OUT_FOLDER
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from ibl_alignment_gui.convertors.extract_ephys import (
    N_WORKERS,
    SAMPLE_SPACING_SECS,
    _is_np2,
    compute_rmsmap,
    save_rmsmap,
)
from ibl_alignment_gui.loaders.data_loader import RawDataLoader
from ibl_alignment_gui.loaders.sources import RawSourceSI

_logger = logging.getLogger(__name__)


def find_raw_files(probe_folder):
    """
    Find the AP and LF files of a probe recorded with SpikeGLX.

    Parameters
    ----------
    probe_folder : str or pathlib.Path
        The folder holding the binary files of one probe, compressed (``.cbin``) or not.

    Returns
    -------
    pathlib.Path
        The AP file.
    pathlib.Path or None
        The LF file, or None when there is none, as for NP2 probes.

    Raises
    ------
    FileNotFoundError
        If the folder holds no AP file.
    ValueError
        If the folder holds the files of more than one probe.
    """
    probe_folder = Path(probe_folder)
    files = {}
    for band in ('ap', 'lf'):
        found = sorted(probe_folder.glob(f'*.{band}.bin')) + sorted(
            probe_folder.glob(f'*.{band}.cbin')
        )
        if len(found) > 1:
            raise ValueError(
                f'{probe_folder} holds more than one {band.upper()} file, give the folder of '
                f'one probe: {[f.name for f in found]}'
            )
        files[band] = found[0] if found else None
    if files['ap'] is None:
        raise FileNotFoundError(f'No AP file (*.ap.bin or *.ap.cbin) found in {probe_folder}')
    return files['ap'], files['lf']


def open_recording(bin_file):
    """
    Open a band of a SpikeGLX recording with SpikeInterface.

    Parameters
    ----------
    bin_file : pathlib.Path
        The binary file of the band, compressed (``.cbin``) or not.

    Returns
    -------
    spikeinterface.BaseRecording
        The band, without the sync channel.
    """
    import spikeinterface.extractors as se  # noqa: PLC0415

    bin_file = Path(bin_file)
    if bin_file.suffix == '.cbin':
        return se.read_cbin_ibl(cbin_file_path=bin_file)
    # SpikeGLX names its files <run>_g<gate>_t<trigger>.<probe>.<band>.bin, and the stream after
    # the probe and band
    probe, band = bin_file.name.split('.')[-3:-1]
    return se.read_spikeglx(bin_file.parent, stream_id=f'{probe}.{band}')


def extract_si_qc(
    probe_folder,
    out_folder,
    sample_spacing_secs=SAMPLE_SPACING_SECS,
    n_workers=N_WORKERS,
):
    """
    Write the RMS, spectral density and snippets of a probe, reading it with SpikeInterface.

    Parameters
    ----------
    probe_folder : str or pathlib.Path
        The folder holding the SpikeGLX files of one probe.
    out_folder : str or pathlib.Path
        The folder to write the datasets to, created if it does not exist.
    sample_spacing_secs : float or None
        The RMS and spectra are computed on a sample taken every ``sample_spacing_secs``
        seconds. None reads the whole recording.
    n_workers : int
        The number of threads the RMS windows are shared between, each with its own reader.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    ap_file, lf_file = find_raw_files(probe_folder)

    def source(bin_file):
        return RawSourceSI(lambda: open_recording(bin_file))

    kwargs = {'sample_spacing_secs': sample_spacing_secs, 'n_workers': n_workers}
    _logger.info(f'Computing the AP RMS of {ap_file}')
    rms = compute_rmsmap(lambda: source(ap_file), 'ap', spectra=False, **kwargs)
    save_rmsmap(rms, out_folder, 'ap', spectra=False)
    if lf_file is not None:
        _logger.info(f'Computing the LF RMS and spectral density of {lf_file}')
        rms = compute_rmsmap(lambda: source(lf_file), 'lf', **kwargs)
        save_rmsmap(rms, out_folder, 'lf')
    elif _is_np2(ap_file):
        # NP2 probes record wideband, so the LF band is the AP band
        _logger.info(f'Computing the LF RMS and spectral density from {ap_file}')
        rms = compute_rmsmap(lambda: source(ap_file), 'ap', lfp_from_ap=True, **kwargs)
        save_rmsmap(rms, out_folder, 'lf')

    # The snippets are computed without a save path, so that cached snippets are not reused
    _logger.info('Computing the snippets')
    loader = RawDataLoader(source(ap_file), source(lf_file) if lf_file else None)
    np.save(out_folder.joinpath(RawDataLoader.AP_SNIPPETS_FILE), loader.load_ap_snippets())
    if lf_file is not None:
        np.save(out_folder.joinpath(RawDataLoader.LF_SNIPPETS_FILE), loader.load_lf_snippets())


def main(argv=None):
    """Run :func:`extract_si_qc` from the command line."""
    parser = argparse.ArgumentParser(
        description='Write the RMS, spectral density and snippets the alignment GUI shows for '
        'a probe recorded with SpikeGLX, reading the raw data with SpikeInterface.'
    )
    parser.add_argument('probe_folder', type=Path, help='The folder of the SpikeGLX files')
    parser.add_argument('out_folder', type=Path, help='The folder to write the datasets to')
    parser.add_argument(
        '--n-workers', type=int, default=N_WORKERS, help='The number of threads to read with'
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    extract_si_qc(args.probe_folder, args.out_folder, n_workers=args.n_workers)


if __name__ == '__main__':
    main()
