import unittest
from pathlib import Path

import numpy as np

from ibl_alignment_gui.loaders.geometry_loader import GeometryLoaderLocal
from ibl_alignment_gui.loaders.plot_loader import PlotLoader
from ibl_alignment_gui.utils.parse_yaml import DatasetPaths
from iblutil.util import Bunch

FIXTURE_PATH = Path(__file__).parents[1].joinpath('fixtures')


class TestRawSnippetsPerShank(unittest.TestCase):
    """Each shank shows the raw snippets and channel flags of its own channels only."""

    def setUp(self):
        # A 4-shank probe recorded on 3 shanks
        self.geom = GeometryLoaderLocal(DatasetPaths(raw_ephys=FIXTURE_PATH.joinpath('NP24')))
        self.geom.get_geometry()

        # The value of each channel is its index in the file
        channels = np.arange(384)
        flags = Bunch(values=channels.astype(float), points=channels % 2 == 0, lines=[0.5])
        snippets = Bunch(
            images={10.0: np.tile(channels.astype(float), (1500, 1))},
            fs=30000,
            dead_channels=flags,
            noisy_channels_coherence=flags,
            noisy_channels_psd=flags,
            outside_channels=flags,
            channel_order='file',
            exists=True,
        )
        self.data = Bunch(raw_ap_snippets=snippets, raw_lf_snippets=Bunch(exists=False))

    def plot_loader(self, shank):
        loader = PlotLoader()
        loader.data = self.data
        loader.shank_sites = self.geom.get_sites_for_shank(shank)
        loader.chn_min = loader.shank_sites['sites_min']
        loader.chn_max = loader.shank_sites['sites_max']
        return loader

    def test_image_holds_the_shank_channels(self):
        """The image holds exactly the channels of the shank"""
        for shank in range(self.geom.electrodes.n_shanks):
            with self.subTest(shank=shank):
                loader = self.plot_loader(shank)
                img = loader.image_raw_ap_data()['Raw ap snippet 0'].img
                shown = np.unique(img[~np.isnan(img)])
                np.testing.assert_array_equal(shown, np.sort(loader.shank_sites['raw_ind']))

    def test_flags_on_the_shank_sites(self):
        """The channel flags are the shank's channels, one per site in depth order"""
        for shank in range(self.geom.electrodes.n_shanks):
            with self.subTest(shank=shank):
                loader = self.plot_loader(shank)
                raw_ind = loader.shank_sites['raw_ind']
                for plot in (
                    loader.line_dead_channels()['Dead Channels'],
                    *loader.line_noisy_channels_coherence().values(),
                    *loader.line_noisy_channels_psd().values(),
                    *loader.line_outside_channels().values(),
                ):
                    np.testing.assert_array_equal(plot.x, raw_ind)
                    np.testing.assert_array_equal(plot.y, loader.shank_sites['sites_y'])
                    np.testing.assert_array_equal(plot.mask, raw_ind % 2 == 0)


if __name__ == '__main__':
    unittest.main()
