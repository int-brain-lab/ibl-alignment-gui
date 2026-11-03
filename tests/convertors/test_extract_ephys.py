import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import spikeglx

import one.alf.io as alfio
from ibl_alignment_gui.utils.optional import IBL_EXTRA_HINT, has_ibllib

FIXTURE_PATH = Path(__file__).parents[1].joinpath('fixtures')


def write_spikeglx(
    folder: Path,
    np_type: str,
    band: str,
    duration: float,
    components: list[tuple[float, float]] | None = None,
    per_channel: bool = True,
    adc_shift: bool = False,
) -> Path:
    """
    Write a SpikeGLX file holding sines, by default with an amplitude set by the file index.

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
    components : list of (float, float) or None
        The frequency (Hz) and amplitude (counts) of each sine; by default a sine of 20 counts at
        1 kHz for AP and 100 Hz for LF.
    per_channel : bool
        Whether the amplitude of each channel is multiplied by its file index + 1, otherwise the
        channels all hold the same signal.
    adc_shift : bool
        Whether each channel is sampled with the delay of its ADC, as the probe records it.

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

    components = components or [(1000 if band == 'ap' else 100, 20)]
    # The time at which each channel is sampled, (samples, channels)
    shift = np.zeros(nc)
    if adc_shift:
        shift[:-1] = spikeglx.geometry_from_meta(
            spikeglx.read_meta_data(bin_file.with_suffix('.meta')), sort=False
        )['sample_shift']
    t = (np.arange(ns)[:, np.newaxis] + shift[np.newaxis, :]) / fs
    data = sum(amplitude * np.sin(2 * np.pi * freq * t) for freq, amplitude in components)
    scale = np.arange(nc) + 1 if per_channel else np.ones(nc)
    data = data * scale[np.newaxis, :]
    data.astype(np.int16).tofile(bin_file)

    return bin_file


class _ConverterTest(unittest.TestCase):
    """Write the test recordings to a temporary folder."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()


@unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)
class TestRmsmap(_ConverterTest):
    """The RMS and spectra are computed for the channels in the order they are in the file."""

    def assert_file_order(self, rms):
        """The RMS grows with the file index, so it only increases along the channels"""
        mean_rms = rms.mean(axis=0)[:-1]  # without the sync channel
        self.assertTrue(np.all(np.diff(mean_rms) > 0))

    def test_file_order(self):
        """Channels are in file order, also for probes whose sorted order differs (NP2)"""
        from ibl_alignment_gui.convertors.extract_ephys import rmsmap  # noqa: PLC0415

        for np_type in ('NP21', 'NP24'):
            with self.subTest(np_type=np_type):
                bin_file = write_spikeglx(self.folder, np_type, 'lf', 1.0)
                win = rmsmap(bin_file, spectra=False, n_workers=1)
                self.assert_file_order(win['TRMS'])
                np.testing.assert_array_equal(win['raw_ind'], np.arange(385))

    def test_open_sorted_reader(self):
        """An open reader that sorts the channels still gives them in file order"""
        from ibl_alignment_gui.convertors.extract_ephys import rmsmap  # noqa: PLC0415

        bin_file = write_spikeglx(self.folder, 'NP24', 'lf', 1.0)
        with spikeglx.Reader(bin_file, sort=True) as sglx:
            win = rmsmap(sglx, spectra=False, n_workers=1)
            # The reader passed in is left open for the caller
            self.assertTrue(sglx.is_open)
        self.assert_file_order(win['TRMS'])

    def test_lf_keeps_lfp(self):
        """The LF RMS keeps the LFP band"""
        from ibl_alignment_gui.convertors.extract_ephys import rmsmap  # noqa: PLC0415

        bin_file = write_spikeglx(self.folder, 'NP24', 'lf', 2.0, components=[(50, 20)])
        rms = rmsmap(bin_file, spectra=False, n_workers=1)['TRMS'].mean(axis=0)[:-1]
        # A sine of amplitude A has an RMS of A / sqrt(2), here in V for channel 0
        s2v = spikeglx.Reader(bin_file).sample2volts[0]
        self.assertGreater(rms[0], 0.5 * 20 / np.sqrt(2) * s2v)


@unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)
class TestApDestriped(_ConverterTest):
    """The AP RMS is computed on destriped data, re-referenced to the median of each shank."""

    def test_destripe_with_car(self):
        """The probe channels are destriped by ibldsp with the file order geometry and CAR"""
        from ibl_alignment_gui.convertors import extract_ephys  # noqa: PLC0415

        bin_file = write_spikeglx(self.folder, 'NP24', 'ap', 0.5)
        with patch.object(
            extract_ephys.voltage, 'destripe', wraps=extract_ephys.voltage.destripe
        ) as mock_destripe:
            extract_ephys.rmsmap(bin_file, spectra=False, n_workers=1)

        args, kwargs = mock_destripe.call_args
        self.assertEqual(args[0].shape[0], 384)  # without the sync channel
        self.assertIs(kwargs['k_filter'], False)
        meta = spikeglx.read_meta_data(bin_file.with_suffix('.meta'))
        geometry = spikeglx.geometry_from_meta(meta, sort=False)
        np.testing.assert_array_equal(kwargs['h']['shank'], geometry['shank'])

    def test_parallel_as_in_process(self):
        """Sharing the windows between threads gives the same result"""
        from ibl_alignment_gui.convertors.extract_ephys import rmsmap  # noqa: PLC0415

        lf_file = write_spikeglx(self.folder, 'NP24', 'lf', 7.0)
        in_process = rmsmap(lf_file, sample_spacing_secs=1, n_workers=1)
        parallel = rmsmap(lf_file, sample_spacing_secs=1, n_workers=2)
        for key in ('TRMS', 'spectral_density', 'tscale'):
            with self.subTest(key=key):
                np.testing.assert_allclose(parallel[key], in_process[key], rtol=1e-6)

    def test_common_signal_removed(self):
        """A signal the same on every channel of a shank is referenced away"""
        from ibl_alignment_gui.convertors.extract_ephys import rmsmap  # noqa: PLC0415

        bin_file = write_spikeglx(
            self.folder,
            'NP24',
            'ap',
            0.5,
            components=[(1000, 500)],
            per_channel=False,
            adc_shift=True,
        )
        rms = rmsmap(bin_file, spectra=False, n_workers=1)['TRMS'].mean(axis=0)[:-1]
        sine_rms = 500 / np.sqrt(2) * spikeglx.Reader(bin_file).sample2volts[0]
        self.assertTrue(np.all(rms < 0.01 * sine_rms))


@unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)
class TestExtractRmsmap(_ConverterTest):
    """The RMS and spectra are saved with the raw index of each channel."""

    def test_raw_ind_saved(self):
        """Each object holds rawInd, (1, channels), and loads as a consistent ALF object"""
        from ibl_alignment_gui.convertors.extract_ephys import extract_rmsmap  # noqa: PLC0415

        bin_file = write_spikeglx(self.folder, 'NP24', 'lf', 1.0)
        out = self.folder.joinpath('alf')
        extract_rmsmap(
            bin_file, out_folder=out, spectra=True, sample_spacing_secs=None, n_workers=1
        )

        for alf_object, attribute in (
            ('ephysTimeRmsLF', 'rms'),
            ('ephysSpectralDensityLF', 'power'),
        ):
            with self.subTest(alf_object=alf_object):
                data = alfio.load_object(out, alf_object, namespace='iblqc')
                np.testing.assert_array_equal(data['rawInd'], np.arange(385)[np.newaxis, :])
                self.assertEqual(data[attribute].shape[1], 385)
                self.assertEqual(alfio.check_dimensions(data), 0)
        rms = alfio.load_object(out, 'ephysTimeRmsLF', namespace='iblqc')['rms']
        self.assertTrue(np.all(np.diff(rms.mean(axis=0)[:-1]) > 0))


@unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)
class TestSampledRms(_ConverterTest):
    """The RMS and spectra are computed on samples of the recording rather than on all of it."""

    def test_windows(self):
        """A sample every spacing, or consecutive windows when there is no spacing"""
        from ibl_alignment_gui.convertors.extract_ephys import _rms_windows  # noqa: PLC0415

        with self.subTest('AP samples of 1 s'):
            windows = _rms_windows(ns=35 * 100, fs=100, sample_spacing_secs=10)
            self.assertEqual(windows, [(0, 100), (1000, 1100), (2000, 2100), (3000, 3100)])

        with self.subTest('LF samples of 3 s'):
            windows = _rms_windows(ns=35 * 100, fs=100, sample_spacing_secs=10, length_secs=3)
            self.assertEqual(windows, [(0, 300), (1000, 1300), (2000, 2300), (3000, 3300)])

        with self.subTest('whole recording'):
            windows = _rms_windows(ns=1000, fs=100)
            # windows of 2 ** ceil(log2(3 s * 100 Hz)) = 512 samples
            self.assertEqual(windows, [(0, 512), (512, 1000)])

    def test_sample_length_per_band(self):
        """AP samples are 1 s long and LF samples 3 s, timed at their middle"""
        from ibl_alignment_gui.convertors.extract_ephys import extract_rmsmap  # noqa: PLC0415

        out = self.folder.joinpath('alf')
        ap_file = write_spikeglx(self.folder, 'NP24', 'ap', 2.5)
        extract_rmsmap(ap_file, out_folder=out, spectra=False, sample_spacing_secs=1, n_workers=1)
        lf_file = write_spikeglx(self.folder, 'NP24', 'lf', 7.0)
        extract_rmsmap(lf_file, out_folder=out, sample_spacing_secs=2, n_workers=1)

        ap = alfio.load_object(out, 'ephysTimeRmsAP', namespace='iblqc')
        lf = alfio.load_object(out, 'ephysTimeRmsLF', namespace='iblqc')
        np.testing.assert_allclose(ap['timestamps'], [0.5, 1.5], atol=1e-3)
        np.testing.assert_allclose(lf['timestamps'], [1.5, 3.5, 5.5], atol=1e-3)

    def test_sampled_matches_whole(self):
        """On a steady signal, the sampled RMS and spectra are those of the whole recording"""
        from ibl_alignment_gui.convertors.extract_ephys import rmsmap  # noqa: PLC0415

        lf_file = write_spikeglx(self.folder, 'NP24', 'lf', 7.0)
        whole = rmsmap(lf_file, n_workers=1)
        sampled = rmsmap(lf_file, sample_spacing_secs=3, n_workers=1)
        # Different numbers of windows, but the spectral density is averaged over them
        self.assertNotEqual(len(whole['tscale']), len(sampled['tscale']))
        np.testing.assert_allclose(
            sampled['TRMS'].mean(axis=0)[:-1], whole['TRMS'].mean(axis=0)[:-1], rtol=0.01
        )
        # The power of the sine, at its peak and in total, rather than the leakage around it
        for psd in (np.max, np.sum):
            np.testing.assert_allclose(
                psd(sampled['spectral_density'][:, :-1], axis=0),
                psd(whole['spectral_density'][:, :-1], axis=0),
                rtol=0.05,
            )


@unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)
class TestLfpFromAp(_ConverterTest):
    """NP2 probes have no LF file, so the LF band is computed from the AP file."""

    def setUp(self):
        super().setUp()
        # An LFP at 50 Hz and spiking power at 2 kHz, which would alias to 500 Hz if the
        # AP band were not filtered out before decimating to 2.5 kHz
        self.ap_file = write_spikeglx(
            self.folder, 'NP24', 'ap', 1.5, components=[(50, 200), (2000, 200)], per_channel=False
        )
        self.s2v = spikeglx.Reader(self.ap_file).sample2volts[0]

    def test_lfp_without_aliasing(self):
        """The LF band keeps the LFP and leaves out the AP band"""
        from ibl_alignment_gui.convertors.extract_ephys import rmsmap  # noqa: PLC0415

        win = rmsmap(self.ap_file, lfp_from_ap=True, n_workers=1)
        freqs, power = win['fscale'], win['spectral_density'][:, 0]

        self.assertAlmostEqual(freqs[-1], 1250, delta=1)
        np.testing.assert_allclose(win['TRMS'][0, :-1], 200 / np.sqrt(2) * self.s2v, rtol=0.05)
        # The LFP peak, and nothing where the 2 kHz sine would alias to
        peak = power[np.argmin(np.abs(freqs - 50))]
        self.assertLess(power[np.argmin(np.abs(freqs - 500))], 1e-4 * peak)

    def test_saved_as_lf(self):
        """The LF band computed from the AP file is saved as the LF objects"""
        from ibl_alignment_gui.convertors.extract_ephys import extract_rmsmap  # noqa: PLC0415

        out = self.folder.joinpath('alf')
        extract_rmsmap(
            self.ap_file, out_folder=out, sample_spacing_secs=None, lfp_from_ap=True, n_workers=1
        )
        for alf_object in ('ephysTimeRmsLF', 'ephysSpectralDensityLF'):
            with self.subTest(alf_object=alf_object):
                data = alfio.load_object(out, alf_object, namespace='iblqc')
                self.assertIn('rawInd', data)

    def test_only_np2(self):
        """The LF band is only computed from the AP file of an NP2 probe"""
        from ibl_alignment_gui.convertors.extract_ephys import extract_rmsmap  # noqa: PLC0415

        # The same file, with the metadata of an NP1 probe
        meta = self.ap_file.with_suffix('.meta')
        meta.write_text(meta.read_text().replace('imDatPrb_type=2013', 'imDatPrb_type=0'))
        with self.assertRaises(ValueError):
            extract_rmsmap(self.ap_file, out_folder=self.folder, lfp_from_ap=True, n_workers=1)


@unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)
class TestExtractData(_ConverterTest):
    """The LF band is computed from the AP file only for NP2 probes without an LF file."""

    def run_extract(self, probe_type, lf_file):
        from ibl_alignment_gui.convertors import extract_ephys  # noqa: PLC0415

        ap_file = write_spikeglx(self.folder, 'NP24', 'ap', 0.1)
        meta = ap_file.with_suffix('.meta')
        meta.write_text(
            meta.read_text().replace('imDatPrb_type=2013', f'imDatPrb_type={probe_type}')
        )
        if lf_file:
            write_spikeglx(self.folder, 'NP24', 'lf', 0.1)
        with (
            patch.object(extract_ephys, 'ks2_to_alf'),
            patch.object(extract_ephys, 'extract_rmsmap') as mock_rmsmap,
        ):
            extract_ephys.extract_data(self.folder, self.folder, self.folder.joinpath('out'))
        return [call.kwargs.get('lfp_from_ap', False) for call in mock_rmsmap.call_args_list]

    def test_np2_without_lf_file(self):
        self.assertEqual(self.run_extract(2013, lf_file=False), [False, True])

    def test_np1_without_lf_file(self):
        self.assertEqual(self.run_extract(0, lf_file=False), [False])

    def test_lf_file(self):
        self.assertEqual(self.run_extract(2013, lf_file=True), [False, False])
