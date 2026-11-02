import tempfile
import unittest
from pathlib import Path

import numpy as np
import spikeglx

import one.alf.io as alfio
from ibl_alignment_gui.utils.optional import IBL_EXTRA_HINT, has_ibllib

FIXTURE_PATH = Path(__file__).parents[1].joinpath('fixtures')


def write_spikeglx(folder: Path, np_type: str, band: str, duration: float) -> Path:
    """
    Write a SpikeGLX file whose channels have an amplitude set by their index in the file.

    Parameters
    ----------
    folder : Path
        The folder to write the binary and meta files to.
    np_type : str
        The fixture folder holding the ap.meta file to use.
    band : str
        'ap' or 'lf'.
    duration : float
        The duration of the recording (s).

    Returns
    -------
    Path
        The binary file.
    """
    fs = 30000 if band == 'ap' else 2500
    nc = 385
    ns = int(duration * fs)
    meta = next(FIXTURE_PATH.joinpath(np_type).glob('*.ap.meta')).read_text()
    replace = {
        'imSampRate': fs,
        'fileTimeSecs': ns / fs,
        'fileSizeBytes': ns * nc * 2,
        # The band is read from the number of AP and LF channels
        'snsApLfSy': '384,0,1' if band == 'ap' else '0,384,1',
    }
    lines = [
        f'{key}={replace[key]}' if (key := line.split('=')[0]) in replace else line
        for line in meta.splitlines()
    ]
    bin_file = folder.joinpath(f'{np_type}_g0_t0.imec0.{band}.bin')
    bin_file.with_suffix('.meta').write_text('\n'.join(lines) + '\n')

    # A 1 kHz (AP) or 100 Hz (LF) sine with an amplitude of 20 * (file index + 1)
    freq = 1000 if band == 'ap' else 100
    sine = np.sin(2 * np.pi * freq * np.arange(ns) / fs)
    data = sine[:, np.newaxis] * (np.arange(nc) + 1)[np.newaxis, :] * 20
    data.astype(np.int16).tofile(bin_file)

    return bin_file


@unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)
class TestRmsmap(unittest.TestCase):
    """The RMS and spectra are computed for the channels in the order they are in the file."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def assert_file_order(self, rms):
        """The RMS grows with the file index, so it only increases along the channels"""
        mean_rms = rms.mean(axis=0)[:-1]  # without the sync channel
        self.assertTrue(np.all(np.diff(mean_rms) > 0))

    def test_file_order(self):
        """Channels are in file order, also for probes whose sorted order differs (NP2)"""
        from ibl_alignment_gui.convertors.extract_ephys import rmsmap  # noqa: PLC0415

        for np_type in ('NP21', 'NP24'):
            with self.subTest(np_type=np_type):
                bin_file = write_spikeglx(self.folder, np_type, 'ap', 0.5)
                win = rmsmap(bin_file, spectra=False)
                self.assert_file_order(win['TRMS'])
                np.testing.assert_array_equal(win['raw_ind'], np.arange(385))

    def test_open_sorted_reader(self):
        """An open reader that sorts the channels still gives them in file order"""
        from ibl_alignment_gui.convertors.extract_ephys import rmsmap  # noqa: PLC0415

        bin_file = write_spikeglx(self.folder, 'NP24', 'ap', 0.5)
        with spikeglx.Reader(bin_file, sort=True) as sglx:
            win = rmsmap(sglx, spectra=False)
            # The reader passed in is left open for the caller
            self.assertTrue(sglx.is_open)
        self.assert_file_order(win['TRMS'])


@unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)
class TestExtractRmsmap(unittest.TestCase):
    """The RMS and spectra are saved with the raw index of each channel."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_raw_ind_saved(self):
        """Each object holds rawInd, (1, channels), and loads as a consistent ALF object"""
        from ibl_alignment_gui.convertors.extract_ephys import extract_rmsmap  # noqa: PLC0415

        bin_file = write_spikeglx(self.folder, 'NP24', 'lf', 1.0)
        out = self.folder.joinpath('alf')
        extract_rmsmap(bin_file, out_folder=out, spectra=True)

        for alf_object, attribute in (
            ('ephysTimeRmsLF', 'rms'),
            ('ephysSpectralDensityLF', 'power'),
        ):
            with self.subTest(alf_object=alf_object):
                data = alfio.load_object(out, alf_object, namespace='iblqc')
                np.testing.assert_array_equal(data['rawInd'], np.arange(385)[np.newaxis, :])
                self.assertEqual(data[attribute].shape[1], 385)
                self.assertEqual(alfio.check_dimensions(data), 0)
        self.assert_file_order_saved(out)

    def assert_file_order_saved(self, out):
        rms = alfio.load_object(out, 'ephysTimeRmsLF', namespace='iblqc')['rms']
        self.assertTrue(np.all(np.diff(rms.mean(axis=0)[:-1]) > 0))
