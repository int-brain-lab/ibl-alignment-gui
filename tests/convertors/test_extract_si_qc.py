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

    def extract(self, raw, name='si'):
        from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc  # noqa: PLC0415

        out = self.folder.joinpath(name)
        extract_si_qc(raw, out, sample_spacing_secs=0.5, n_workers=2)
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
        out = self.extract(raw)
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
        self.assert_rms_equal(self.extract(raw, 'cbin'), expected)


@unittest.skipUnless(has_spikeinterface(), SPIKEINTERFACE_EXTRA_HINT)
class TestFindRawFiles(_ConverterTest):
    """The files of one probe are found in its folder."""

    def test_files(self):
        from ibl_alignment_gui.convertors.extract_si_qc import find_raw_files  # noqa: PLC0415

        ap_file = write_spikeglx(self.folder, 'NP24', 'ap', 0.01)
        self.assertEqual(find_raw_files(self.folder), (ap_file, None))
        lf_file = write_spikeglx(self.folder, 'NP24', 'lf', 0.01)
        self.assertEqual(find_raw_files(self.folder), (ap_file, lf_file))

    def test_no_ap_file(self):
        from ibl_alignment_gui.convertors.extract_si_qc import find_raw_files  # noqa: PLC0415

        with self.assertRaises(FileNotFoundError):
            find_raw_files(self.folder)

    def test_two_probes(self):
        from ibl_alignment_gui.convertors.extract_si_qc import find_raw_files  # noqa: PLC0415

        ap_file = write_spikeglx(self.folder, 'NP24', 'ap', 0.01)
        ap_file.rename(ap_file.with_name(ap_file.name.replace('imec0', 'imec1')))
        ap_file.with_suffix('.meta').rename(
            ap_file.with_name(ap_file.name.replace('imec0', 'imec1')).with_suffix('.meta')
        )
        write_spikeglx(self.folder, 'NP24', 'ap', 0.01)
        with self.assertRaises(ValueError):
            find_raw_files(self.folder)
