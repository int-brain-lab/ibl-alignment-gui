"""A short SpikeGLX recording written from the metadata of a test fixture."""

from pathlib import Path

import numpy as np

FIXTURE_PATH = Path(__file__).parents[1].joinpath('fixtures')


def write_spikeglx(folder: Path, np_type: str, seconds: float = 2) -> Path:
    """
    Write a SpikeGLX AP recording of random samples with the probe of a fixture's ap.meta.

    The files are named as SpikeGLX writes them, so SpikeInterface can read the run folder.

    Parameters
    ----------
    folder : Path
        The folder to write the run folder to.
    np_type : str
        The fixture folder holding the ap.meta, e.g. 'NP24'.
    seconds : float
        The duration of the recording.

    Returns
    -------
    Path
        The ap.bin file, in folder/rec_g0/rec_g0_imec0.
    """
    probe_folder = folder.joinpath('rec_g0', 'rec_g0_imec0')
    probe_folder.mkdir(parents=True)
    meta_file = next(FIXTURE_PATH.joinpath(np_type).glob('*.ap.meta'))
    meta = dict(line.split('=', 1) for line in meta_file.read_text().splitlines() if '=' in line)

    fs = float(meta['imSampRate'])
    n_samples = int(seconds * fs)
    data = np.random.default_rng(0).integers(
        -200, 200, (n_samples, int(meta['nSavedChans'])), dtype=np.int16
    )
    bin_file = probe_folder.joinpath('rec_g0_t0.imec0.ap.bin')
    data.tofile(bin_file)

    meta['fileSizeBytes'] = str(data.nbytes)
    meta['fileTimeSecs'] = str(n_samples / fs)
    bin_file.with_suffix('.meta').write_text(''.join(f'{k}={v}\n' for k, v in meta.items()))
    return bin_file
