import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import spikeglx

import one.alf.io as alfio
from ibl_alignment_gui.utils.optional import SPIKEINTERFACE_EXTRA_HINT, has_spikeinterface
from tests.convertors.test_extract_ephys import write_spikeglx

ALF_OBJECTS = ('ephysTimeRmsAP', 'ephysTimeRmsLF', 'ephysSpectralDensityLF')
SAMPLE_SPACING_SECS = 0.5


def write_probe(folder, lf_file=True):
    """Write the AP file of an NP2 probe, and its LF file when asked, to the raw folder."""
    raw = folder.joinpath('raw')
    raw.mkdir()
    ap_file = write_spikeglx(raw, 'NP24', 'ap', 1.2, adc_shift=True)
    lf_file = write_spikeglx(raw, 'NP24', 'lf', 6.0) if lf_file else None
    return raw, ap_file, lf_file


def extract(raw, out, **kwargs):
    """Read the probe with SpikeInterface and extract its datasets to the out folder."""
    import spikeinterface.extractors as se  # noqa: PLC0415

    from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc  # noqa: PLC0415

    # The metadata of the test files do not give the saturation level SpikeInterface reads
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Unable to compute saturation threshold')
        if 'lfp_from_ap' not in kwargs and next(raw.glob('*.lf.bin'), None):
            kwargs['lf_recording'] = se.read_spikeglx(raw, stream_id='imec0.lf')
        ap = se.read_spikeglx(raw, stream_id='imec0.ap')
        extract_si_qc(ap, out, sample_spacing_secs=SAMPLE_SPACING_SECS, n_workers=2, **kwargs)
    return out


def extract_spikeglx(ap_file, out, lf_file=None, lfp_from_ap=False):
    """Extract the RMS and spectra of the probe with extract_ephys, reading it with spikeglx."""
    from ibl_alignment_gui.convertors.extract_ephys import extract_rmsmap  # noqa: PLC0415

    kwargs = {'out_folder': out, 'sample_spacing_secs': SAMPLE_SPACING_SECS, 'n_workers': 1}
    extract_rmsmap(ap_file, spectra=False, **kwargs)
    if lf_file is not None:
        extract_rmsmap(lf_file, **kwargs)
    elif lfp_from_ap:
        extract_rmsmap(ap_file, lfp_from_ap=True, **kwargs)
    return out


class _RmsTest(unittest.TestCase):
    def assert_rms_equal(self, out, expected):
        for alf_object in ALF_OBJECTS:
            with self.subTest(alf_object=alf_object):
                got = alfio.load_object(out, alf_object, namespace='iblqc')
                want = alfio.load_object(expected, alf_object, namespace='iblqc')
                self.assertEqual(set(got), set(want))
                for key in want:
                    np.testing.assert_allclose(got[key], want[key], rtol=1e-5, atol=1e-12)


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestExtractSIQC(_RmsTest):
    """The datasets of a probe with an LF file read with SpikeInterface match those of spikeglx."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        folder = Path(cls.temp_dir.name)
        cls.raw, cls.ap_file, cls.lf_file = write_probe(folder)
        cls.out = extract(cls.raw, folder.joinpath('si'))

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_rms_as_spikeglx(self):
        """The RMS and spectra match those computed by extract_ephys from the files"""
        expected = Path(self.temp_dir.name).joinpath('sglx')
        extract_spikeglx(self.ap_file, expected, lf_file=self.lf_file)
        self.assert_rms_equal(self.out, expected)

    def test_snippets_as_spikeglx(self):
        """The snippets and channel flags match those of the GUI's spikeglx loader"""
        from ibl_alignment_gui.loaders.data_loader import RawDataLoader  # noqa: PLC0415
        from ibl_alignment_gui.loaders.sources import RawSourceSpikeGLX  # noqa: PLC0415

        expected = RawDataLoader(
            RawSourceSpikeGLX(lambda: spikeglx.Reader(self.ap_file)),
            RawSourceSpikeGLX(lambda: spikeglx.Reader(self.lf_file)),
        )

        for file, load in (
            (RawDataLoader.AP_SNIPPETS_FILE, expected.load_ap_snippets),
            (RawDataLoader.LF_SNIPPETS_FILE, expected.load_lf_snippets),
        ):
            with self.subTest(file=file):
                got = np.load(self.out.joinpath(file), allow_pickle=True).item()
                want = load()
                self.assertEqual(got['channel_order'], 'file')
                self.assertEqual(got['fs'], want['fs'])
                # The durations, which the times are clipped to, differ by float error
                np.testing.assert_allclose(list(got['images']), list(want['images']))
                for image, expected_image in zip(
                    got['images'].values(), want['images'].values(), strict=True
                ):
                    np.testing.assert_allclose(image, expected_image, rtol=1e-5, atol=1e-9)
                for key in ('dead_channels', 'noisy_channels_psd', 'outside_channels'):
                    if key in want:
                        np.testing.assert_allclose(got[key].values, want[key].values, atol=1e-6)
                        np.testing.assert_array_equal(got[key].points, want[key].points)

    def test_contact_ids(self):
        """The contact ids of the AP recording are saved in file order"""
        import spikeinterface.extractors as se  # noqa: PLC0415

        from ibl_alignment_gui.loaders.sources import (  # noqa: PLC0415
            RAW_CONTACT_IDS_FILE,
            channel_contact_ids,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Unable to compute saturation threshold')
            ap = se.read_spikeglx(self.raw, stream_id='imec0.ap')
        expected = channel_contact_ids(ap)
        self.assertEqual(expected.size, 384)
        np.testing.assert_array_equal(np.load(self.out.joinpath(RAW_CONTACT_IDS_FILE)), expected)


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestExtractSIQCWithoutLF(_RmsTest):
    """The datasets of a probe without an LF file."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.folder = Path(self.temp_dir.name)
        self.raw, self.ap_file, _ = write_probe(self.folder, lf_file=False)

    def test_lf_from_ap_of_np1_metadata(self):
        """The metadata of a SpikeGLX recording say whether the LF band is in the AP band"""
        import spikeinterface.extractors as se  # noqa: PLC0415

        from ibl_alignment_gui.convertors.extract_si_qc import _check_wideband  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Unable to compute saturation threshold')
            ap = se.read_spikeglx(self.raw, stream_id='imec0.ap')
        _check_wideband(ap)
        # The metadata of an NP1 probe, which the probe read before does not know of
        meta = self.ap_file.with_suffix('.meta')
        meta.write_text(meta.read_text().replace('imDatPrb_type=2013', 'imDatPrb_type=0'))
        with self.assertRaises(ValueError):
            _check_wideband(ap)

    def test_lf_from_ap(self):
        """The LF band of an NP2 probe is computed from the AP band, without LF snippets"""
        from ibl_alignment_gui.loaders.data_loader import RawDataLoader  # noqa: PLC0415

        out = extract(self.raw, self.folder.joinpath('si'), lfp_from_ap=True)
        expected = extract_spikeglx(self.ap_file, self.folder.joinpath('sglx'), lfp_from_ap=True)
        self.assert_rms_equal(out, expected)
        self.assertTrue(out.joinpath(RawDataLoader.AP_SNIPPETS_FILE).exists())
        self.assertFalse(out.joinpath(RawDataLoader.LF_SNIPPETS_FILE).exists())

    def test_without_lf(self):
        """Without an LF recording or lfp_from_ap, only the AP datasets are written"""
        from ibl_alignment_gui.loaders.data_loader import RawDataLoader  # noqa: PLC0415

        out = extract(self.raw, self.folder.joinpath('si'))
        self.assertEqual(
            sorted(f.name for f in out.iterdir()),
            sorted(
                [
                    '_iblqc_ephysChannels.contactIds.npy',
                    '_iblqc_ephysTimeRmsAP.rawInd.npy',
                    '_iblqc_ephysTimeRmsAP.rms.npy',
                    '_iblqc_ephysTimeRmsAP.timestamps.npy',
                    RawDataLoader.AP_SNIPPETS_FILE,
                ]
            ),
        )


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestRecordingChecks(unittest.TestCase):
    """Recordings that cannot give the datasets are refused."""

    def setUp(self):
        import spikeinterface as si  # noqa: PLC0415

        self.ap = si.generate_recording(num_channels=4, sampling_frequency=30000, durations=[1])
        self.lf = si.generate_recording(num_channels=4, sampling_frequency=2500, durations=[1])

    def test_filtered(self):
        import spikeinterface.preprocessing as spre  # noqa: PLC0415

        from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc  # noqa: PLC0415

        with self.assertRaises(ValueError):
            extract_si_qc(spre.highpass_filter(self.ap), 'unused')

    def test_lf_twice(self):
        from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc  # noqa: PLC0415

        with self.assertRaises(ValueError):
            extract_si_qc(self.ap, 'unused', lf_recording=self.lf, lfp_from_ap=True)

    def test_lf_channels_differ(self):
        """The LF band must have the channels of the AP band"""
        from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc  # noqa: PLC0415

        lf = self.lf.remove_channels([self.lf.channel_ids[0]])
        with self.assertRaises(ValueError):
            extract_si_qc(self.ap, 'unused', lf_recording=lf)

    def test_lfp_from_lf_band(self):
        """The LF band can only be computed from a band sampled at 30 kHz"""
        from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc  # noqa: PLC0415

        with self.assertRaises(ValueError):
            extract_si_qc(self.lf, 'unused', lfp_from_ap=True)

    def test_lfp_from_ap_of_np1(self):
        """The LF band is not computed from the AP band of a probe with a separate LF band"""
        from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc  # noqa: PLC0415

        # probeinterface gives Neuropixels 1.0 probes an LF band at 2.5 kHz
        probe = self.ap.get_probe()
        probe.annotate(lf_sample_frequency_hz=2500.0)
        self.ap.set_probe(probe)
        with self.assertRaises(ValueError):
            extract_si_qc(self.ap, 'unused', lfp_from_ap=True)

    def test_lfp_from_ap_of_unknown_probe(self):
        """A probe that does not give its LF band is taken to record wideband, with a warning"""
        from ibl_alignment_gui.convertors.extract_si_qc import _check_wideband  # noqa: PLC0415

        with self.assertLogs('ibl_alignment_gui.convertors.extract_si_qc', 'WARNING'):
            _check_wideband(self.ap)

    def test_clone_per_thread(self):
        """Each thread after the first reads from a clone of the recording"""
        from ibl_alignment_gui.convertors.extract_si_qc import _source_opener  # noqa: PLC0415

        open_source = _source_opener(self.ap)
        first, second = open_source().recording, open_source().recording
        self.assertIs(first, self.ap)
        self.assertIsNot(second, self.ap)
        np.testing.assert_array_equal(second.get_traces(), self.ap.get_traces())
