import unittest
import warnings

import numpy as np
import spikeglx

import one.alf.io as alfio
from ibl_alignment_gui.utils.optional import SPIKEINTERFACE_EXTRA_HINT, has_spikeinterface
from tests.convertors.test_extract_ephys import _ConverterTest, write_spikeglx

ALF_OBJECTS = ('ephysTimeRmsAP', 'ephysTimeRmsLF', 'ephysSpectralDensityLF')


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestExtractSIQC(_ConverterTest):
    """The datasets read with SpikeInterface match those read with spikeglx."""

    def setUp(self):
        super().setUp()
        # The metadata of the test files do not give the saturation level SpikeInterface reads
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings('ignore', 'Unable to compute saturation threshold')

    def write_probe(self, lf_file=True):
        """Write the AP file of an NP2 probe, and its LF file when asked, to the raw folder."""
        raw = self.folder.joinpath('raw')
        raw.mkdir()
        ap_file = write_spikeglx(raw, 'NP24', 'ap', 1.2, adc_shift=True)
        lf_file = write_spikeglx(raw, 'NP24', 'lf', 6.0) if lf_file else None
        return raw, ap_file, lf_file

    def extract(self, raw, name='si', compressed=False, **kwargs):
        """Read the probe with SpikeInterface and extract its datasets."""
        import spikeinterface.extractors as se  # noqa: PLC0415

        from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc  # noqa: PLC0415

        def read(band):
            if compressed:
                cbin_file = next(raw.glob(f'*.{band}.cbin'), None)
                return se.read_cbin_ibl(cbin_file_path=cbin_file) if cbin_file else None
            if next(raw.glob(f'*.{band}.bin'), None) is None:
                return None
            return se.read_spikeglx(raw, stream_id=f'imec0.{band}')

        if 'lfp_from_ap' not in kwargs:
            kwargs['lf_recording'] = read('lf')
        out = self.folder.joinpath(name)
        extract_si_qc(read('ap'), out, sample_spacing_secs=0.5, n_workers=2, **kwargs)
        return out

    def assert_rms_equal(self, out, expected):
        for alf_object in ALF_OBJECTS:
            with self.subTest(alf_object=alf_object):
                got = alfio.load_object(out, alf_object, namespace='iblqc')
                want = alfio.load_object(expected, alf_object, namespace='iblqc')
                self.assertEqual(set(got), set(want))
                for key in want:
                    np.testing.assert_allclose(got[key], want[key], rtol=1e-5, atol=1e-12)

    def test_rms_as_spikeglx(self):
        """The RMS and spectra match those computed by extract_ephys from the files"""
        from ibl_alignment_gui.convertors.extract_ephys import extract_rmsmap  # noqa: PLC0415

        raw, ap_file, lf_file = self.write_probe()
        out = self.extract(raw)
        expected = self.folder.joinpath('sglx')
        kwargs = {'out_folder': expected, 'sample_spacing_secs': 0.5, 'n_workers': 1}
        extract_rmsmap(ap_file, spectra=False, **kwargs)
        extract_rmsmap(lf_file, **kwargs)
        self.assert_rms_equal(out, expected)

    def test_lf_from_ap(self):
        """Without an LF file, the LF band of an NP2 probe is computed from the AP band"""
        from ibl_alignment_gui.convertors.extract_ephys import extract_rmsmap  # noqa: PLC0415
        from ibl_alignment_gui.loaders.data_loader import RawDataLoader  # noqa: PLC0415

        raw, ap_file, _ = self.write_probe(lf_file=False)
        out = self.extract(raw, lfp_from_ap=True)
        expected = self.folder.joinpath('sglx')
        kwargs = {'out_folder': expected, 'sample_spacing_secs': 0.5, 'n_workers': 1}
        extract_rmsmap(ap_file, spectra=False, **kwargs)
        extract_rmsmap(ap_file, lfp_from_ap=True, **kwargs)
        self.assert_rms_equal(out, expected)
        self.assertTrue(out.joinpath(RawDataLoader.AP_SNIPPETS_FILE).exists())
        self.assertFalse(out.joinpath(RawDataLoader.LF_SNIPPETS_FILE).exists())

    def test_snippets_as_spikeglx(self):
        """The snippets and channel flags match those of the GUI's spikeglx loader"""
        from ibl_alignment_gui.loaders.data_loader import RawDataLoader  # noqa: PLC0415
        from ibl_alignment_gui.loaders.sources import RawSourceSpikeGLX  # noqa: PLC0415

        raw, ap_file, lf_file = self.write_probe()
        out = self.extract(raw)
        expected = RawDataLoader(
            RawSourceSpikeGLX(lambda: spikeglx.Reader(ap_file)),
            RawSourceSpikeGLX(lambda: spikeglx.Reader(lf_file)),
        )

        for file, load in (
            (RawDataLoader.AP_SNIPPETS_FILE, expected.load_ap_snippets),
            (RawDataLoader.LF_SNIPPETS_FILE, expected.load_lf_snippets),
        ):
            with self.subTest(file=file):
                got = np.load(out.joinpath(file), allow_pickle=True).item()
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

    def test_compressed(self):
        """Compressed files give the same datasets as uncompressed ones"""
        raw, ap_file, lf_file = self.write_probe()
        expected = self.extract(raw, 'bin')
        for bin_file in (ap_file, lf_file):
            with spikeglx.Reader(bin_file) as reader:
                reader.compress_file(keep_original=False)
        self.assertEqual(len(list(raw.glob('*.cbin'))), 2)
        self.assert_rms_equal(self.extract(raw, 'cbin', compressed=True), expected)

    def test_without_lf(self):
        """Without an LF recording or lfp_from_ap, only the AP datasets are written"""
        from ibl_alignment_gui.loaders.data_loader import RawDataLoader  # noqa: PLC0415

        raw, _, _ = self.write_probe(lf_file=False)
        out = self.extract(raw)
        self.assertEqual(
            sorted(f.name for f in out.iterdir()),
            sorted(
                [
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

    def test_lfp_from_lf_band(self):
        """The LF band can only be computed from a band sampled at 30 kHz"""
        from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc  # noqa: PLC0415

        with self.assertRaises(ValueError):
            extract_si_qc(self.lf, 'unused', lfp_from_ap=True)

    def test_clone_per_thread(self):
        """Each thread after the first reads from a clone of the recording"""
        from ibl_alignment_gui.convertors.extract_si_qc import _source_opener  # noqa: PLC0415

        open_source = _source_opener(self.ap)
        first, second = open_source().recording, open_source().recording
        self.assertIs(first, self.ap)
        self.assertIsNot(second, self.ap)
        np.testing.assert_array_equal(second.get_traces(), self.ap.get_traces())
