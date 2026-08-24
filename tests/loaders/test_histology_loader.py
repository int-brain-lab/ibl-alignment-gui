import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ibl_alignment_gui.loaders.histology_loader import (
    NrrdSliceLoader,
    SliceLoader,
    TiffSliceLoader,
    make_slice_loader,
)


class TestMakeSliceLoader(unittest.TestCase):
    """Test the make_slice_loader factory."""

    def setUp(self):
        self.atlas = MagicMock()

    def slice_keys(self, loader):
        """The slice options the loader offers, with the volume reading stubbed out."""
        with patch.object(SliceLoader, '_make_slice_bunch', return_value={'stub': True}):
            return sorted(loader.get_slices(np.zeros((3, 3))).keys())

    def test_no_histology_path(self):
        # A session may have no histology at all, so the loader must not go looking for files
        for space in ['ccf', 'anatomical']:
            with self.subTest(space):
                loader = make_slice_loader(None, self.atlas, space)
                self.assertIsInstance(loader, NrrdSliceLoader)
                self.assertEqual(loader.hist_paths, {})
                # The atlas slices come from the atlas rather than from files, so they remain
                self.assertEqual(self.slice_keys(loader), ['Annotation', 'CCF'])

    @staticmethod
    def globbing(files):
        """Patch Path.glob so that each call gets its own iterator over the matching files."""

        def glob(self, pattern):
            suffix = pattern.lstrip('*')
            return iter([f for f in files if f.suffix == suffix])

        return patch.object(Path, 'glob', glob)

    def test_nrrd_and_tiff_folders(self):
        with self.subTest('nrrd present'), self.globbing([Path('a_RD.nrrd')]):
            self.assertIsInstance(make_slice_loader(Path('/hist'), self.atlas), NrrdSliceLoader)

        with self.subTest('only tiffs present'), self.globbing([Path('a_RD.tif')]):
            self.assertIsInstance(make_slice_loader(Path('/hist'), self.atlas), TiffSliceLoader)

    def test_empty_histology_folder(self):
        # A folder with no volumes gives the same result as no folder at all
        with self.globbing([]):
            loader = make_slice_loader(Path('/hist'), self.atlas)

        self.assertEqual(loader.hist_paths, {})
        self.assertEqual(self.slice_keys(loader), ['Annotation', 'CCF'])

    def test_histology_channels_are_offered(self):
        # The red and green channels are matched on RD and GR anywhere in the filename
        files = [Path('/hist/histology_image_RD.nrrd'), Path('/hist/histology_image_GR.nrrd')]
        with self.globbing(files):
            loader = make_slice_loader(Path('/hist'), self.atlas)

        self.assertEqual(sorted(loader.hist_paths), ['Histology green', 'Histology red'])
        self.assertEqual(
            self.slice_keys(loader), ['Annotation', 'CCF', 'Histology green', 'Histology red']
        )


if __name__ == '__main__':
    unittest.main()
