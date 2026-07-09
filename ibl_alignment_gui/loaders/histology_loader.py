import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import requests
import SimpleITK as sitk  # noqa: N813

from iblatlas.atlas import AllenAtlas, BrainAtlas
from ibl_alignment_gui.utils.allen.anatomical_atlas import BrainAtlasAnatomical, _BLESSED_DIRECTION
from iblutil.util import Bunch
from one import params
from one.webclient import http_download_file

logger = logging.getLogger(__name__)


class LazySliceDict(dict):
    """
    Dict of histology slice Bunches that loads channels from disk on first access.

    Eager entries (CCF, Annotation) are populated immediately. Lazy entries
    (histology channels) are stored as ``None`` placeholders until a key is
    accessed, at which point the registered callback loads and caches the slice.
    """

    def __init__(
        self,
        eager_data: dict,
        lazy_callbacks: dict[str, Callable[[], Bunch]],
    ):
        super().__init__(eager_data)
        self._callbacks: dict[str, Callable] = {}
        for key, cb in lazy_callbacks.items():
            self._callbacks[key] = cb
            super().__setitem__(key, None)  # placeholder so key appears in .keys()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if value is None and key in self._callbacks:
            value = self._callbacks[key]()
            super().__setitem__(key, value)  # cache for subsequent accesses
        return value

    def get(self, key, default=None):
        # CPython's dict.get() bypasses __getitem__, so override to trigger lazy load.
        if key in self:
            return self[key]
        return default


class SliceLoader(ABC):
    """
    Abstract base class for loading histology slices.

    Subclasses must implement the `get_paths` and `load_volume` methods.

    Parameters
    ----------
    file_path : Path
        Directory containing histology files.
    brain_atlas : AllenAtlas
        Reference brain atlas.
    """

    def __init__(self, file_path: Path, brain_atlas: BrainAtlas):
        self.file_path: Path = file_path
        self.brain_atlas: BrainAtlas = brain_atlas
        self.hist_paths: dict[str, Path] = {}
        self.get_paths()

    @abstractmethod
    def get_paths(self) -> None:
        """Locate and store relevant histology file paths in self.hist_paths."""

    @abstractmethod
    def load_volume(self, vol_path: Path) -> np.ndarray:
        """
        Load a 3D volume from a file.

        Parameters
        ----------
        vol_path : Path
            A path to the volume file.

        Returns
        -------
        np.ndarray
            Loaded 3D image volume.
        """

    def get_slices(self, xyz: np.ndarray) -> LazySliceDict:
        """
        Generate slice images for CCF, annotation, and histology channels.

        CCF and Annotation are computed immediately (atlas arrays are already in
        memory).  Histology channel volumes are loaded from disk only when their
        key is first accessed in the returned dict.

        Parameters
        ----------
        xyz : np.ndarray
            n x 3 array of xyz coordinates along the probe track.

        Returns
        -------
        LazySliceDict
            Keys: 'CCF', 'Annotation', and one key per entry in hist_paths.
            Each value is a Bunch with 'slice' (2D array), 'scale', and 'offset'.
        """
        index = self.brain_atlas.bc.xyz2i(xyz)[:, self.brain_atlas.xyz2dims]
        width = [self.brain_atlas.bc.i2x(0), self.brain_atlas.bc.i2x(self.brain_atlas.bc.nx - 1)]
        height = [self.brain_atlas.bc.i2z(index[0, 2]), self.brain_atlas.bc.i2z(index[-1, 2])]
        scale = np.array(
            [
                (width[1] - width[0]) / self.brain_atlas.bc.nx,
                (height[1] - height[0]) / len(xyz),
            ]
        )
        offset = np.array([width[0], height[0]])

        ann = self._make_slice_bunch(self.brain_atlas.label, index, scale, offset, annotation=True)
        ann['label'] = True

        eager = {
            'CCF': self._make_slice_bunch(self.brain_atlas.image, index, scale, offset),
            'Annotation': ann,
        }

        def _make_callback(vol_path):
            def _load():
                try:
                    vol = self.load_volume(vol_path)
                    return self._make_slice_bunch(vol, index, scale, offset)
                except Exception as e:
                    logger.error(f'Failed to load {vol_path}: {e}')
                    return None

            return _load

        lazy = {key: _make_callback(path) for key, path in self.hist_paths.items()}
        return LazySliceDict(eager, lazy)

    def _make_slice_bunch(
        self,
        vol: np.ndarray,
        index: np.ndarray,
        scale: np.ndarray,
        offset: np.ndarray,
        annotation: bool = False,
    ) -> Bunch:
        """Extract a 2D wavy slice from *vol* and package it with shared metadata."""
        hist_slice = vol[index[:, 0], :, index[:, 2]]
        if annotation:
            hist_slice = self.brain_atlas._label2rgb(hist_slice)
        hist_slice = np.swapaxes(hist_slice, 0, 1)
        return Bunch({'slice': hist_slice, 'scale': scale, 'offset': offset})


class NrrdSliceLoader(SliceLoader):
    """
    SliceLoader for histology in the NRRD format.

    Parameters
    ----------
    file_path : Path
        Directory containing .nrrd files.
    brain_atlas : AllenAtlas
        Brain atlas for alignment.
    """

    def __init__(self, file_path: Path, brain_atlas: AllenAtlas):
        super().__init__(file_path, brain_atlas)

    def get_paths(self) -> None:
        """Load histology file paths with predefined color channel suffixes."""
        col_map = {'red': 'RD', 'green': 'GR'}
        files = list(self.file_path.glob('*.nrrd'))

        for color, abbrev in col_map.items():
            match = next((f for f in files if abbrev in f.name), None)
            if match:
                self.hist_paths[f'Histology {color}'] = match

    def load_volume(self, vol_path: Path) -> np.ndarray:
        """
        Load a volume using AllenAtlas.

        Parameters
        ----------
        vol_path : Path
            A path to histology volume.

        Returns
        -------
        np.ndarray
            Loaded image volume.
        """
        return AllenAtlas._read_volume(vol_path)


@dataclass(frozen=True)
class ImageSpacePaths:
    """
    Paths to the NRRD files produced by the histology registration pipeline,
    all living in a single folder.

    atlas_image_path : Path
        CCF template warped into anatomical space (``ccf_in_*.nrrd``).
    atlas_labels_path : Path
        CCF labels warped into anatomical space (``labels_in_*.nrrd``).
    pipeline_image_path : Path
        Pipeline reference image used by the registration (``histology_registration_pipeline.nrrd``).
    histology_image_path : Path
        Main registered histology channel (``histology_registration.nrrd``).
    other_channel_paths : list[Path]
        Any additional fluorescence channels matching ``Ex_*_Em_*.nrrd``.
    """

    atlas_image_path: Path
    atlas_labels_path: Path
    pipeline_image_path: Path
    histology_image_path: Path
    other_channel_paths: list[Path] = field(default_factory=list)

    @classmethod
    def from_folder(cls, input_path: Path) -> 'ImageSpacePaths':
        """
        Discover all required files in *input_path* and return an ImageSpacePaths.

        Raises StopIteration if any required file is missing.
        """

        def _glob_first(pattern: str) -> Path:
            return next(input_path.glob(pattern))

        other_channel_paths: list[Path] = []
        pattern = re.compile(r'^Ex_\d+_Em_\d+\.nrrd$')
        for p in input_path.iterdir():
            if pattern.match(p.name):
                other_channel_paths.append(p)

        return cls(
            atlas_image_path=_glob_first('ccf_in_*.nrrd'),
            atlas_labels_path=_glob_first('labels_in_*.nrrd'),
            pipeline_image_path=_glob_first('histology_registration_pipeline.nrrd'),
            histology_image_path=_glob_first('histology_registration.nrrd'),
            other_channel_paths=other_channel_paths,
        )


class AnatomicalSliceLoader(SliceLoader):
    """
    SliceLoader for histology registered in original anatomical (non-CCF) space.

    Expects a folder produced by the histology registration pipeline containing:
    ``ccf_in_*.nrrd``, ``labels_in_*.nrrd``,
    ``histology_registration_pipeline.nrrd``, ``histology_registration.nrrd``,
    and optionally ``Ex_*_Em_*.nrrd`` channel files.

    The BrainAtlasAnatomical built from these files works in the physical space
    of the anatomical images (mm, RAS).  Coordinates passed to ``get_slices``
    must therefore be in that same anatomical physical space, not in Allen CCF
    space.

    Parameters
    ----------
    file_path : Path
        Folder containing the registration pipeline NRRD outputs.
    brain_atlas : BrainAtlas
        Unused; accepted to satisfy the SliceLoader interface and the
        ``make_slice_loader`` factory signature.
    """

    def __init__(self, file_path: Path, brain_atlas: BrainAtlas):
        super().__init__(file_path, brain_atlas)
        if not isinstance(self.brain_atlas, BrainAtlasAnatomical):
            self.brain_atlas = self._build_anatomical_atlas()

    def get_paths(self) -> None:
        self.image_space_paths = ImageSpacePaths.from_folder(self.file_path)
        self.hist_paths: dict[str, Path] = {
            'Histology registration': self.image_space_paths.histology_image_path,
        }
        for p in self.image_space_paths.other_channel_paths:
            self.hist_paths[p.stem] = p

    def load_volume(self, vol_path: Path) -> np.ndarray:
        """Read a channel NRRD, reorient to IRP, and return as a numpy array."""
        img = sitk.ReadImage(str(vol_path))
        img = sitk.DICOMOrient(img, _BLESSED_DIRECTION)
        return sitk.GetArrayFromImage(img)

    def _build_anatomical_atlas(self) -> BrainAtlasAnatomical:
        return build_anatomical_atlas(self.file_path)


def build_anatomical_atlas(histology_path: Path) -> BrainAtlasAnatomical:
    """
    Build a BrainAtlasAnatomical from the registration pipeline NRRD files in *histology_path*.

    Parameters
    ----------
    histology_path : Path
        Folder containing ``ccf_in_*.nrrd``, ``labels_in_*.nrrd``, and
        ``histology_registration_pipeline.nrrd``.

    Returns
    -------
    BrainAtlasAnatomical
    """
    paths = ImageSpacePaths.from_folder(histology_path)
    return BrainAtlasAnatomical(
        intensity_img=sitk.ReadImage(str(paths.atlas_image_path)),
        label_img=sitk.ReadImage(str(paths.atlas_labels_path)),
        pipeline_img=sitk.ReadImage(str(paths.pipeline_image_path)),
    )


def make_slice_loader(file_path: Path, brain_atlas: BrainAtlas, space: str = 'ccf') -> SliceLoader:
    """
    Return the appropriate SliceLoader for the given folder.

    Parameters
    ----------
    file_path : Path
        Folder containing histology files.
    brain_atlas : BrainAtlas
        Brain atlas passed to the loader (used directly by NrrdSliceLoader;
        ignored by AnatomicalSliceLoader which builds its own atlas from the
        folder files).
    space : {'ccf', 'anatomical'}
        Which loader to use.  'ccf' returns a NrrdSliceLoader operating in
        Allen CCF space; 'anatomical' returns an AnatomicalSliceLoader
        operating in the original image space.  Matches the ``histology.space``
        field in the alignment YAML.

    Returns
    -------
    SliceLoader
        NrrdSliceLoader for 'ccf', AnatomicalSliceLoader for 'anatomical'.
    """
    if space == 'anatomical':
        return AnatomicalSliceLoader(file_path, brain_atlas)
    return _build_slice_loader(file_path, brain_atlas)


def _build_slice_loader(hist_path: Path, brain_atlas: AllenAtlas) -> SliceLoader:
    """
    Pick the right SliceLoader by inspecting the histology directory.

    Used by the offline ProbeHandlers (:class:`ProbeHandlerLocal` and
    :class:`ProbeHandlerLocalYaml`). If the directory contains any ``.tif`` / ``.tiff`` files
    (e.g. brainreg outputs), return a :class:`TiffSliceLoader`. Otherwise default to the existing
    :class:`NrrdSliceLoader` so all current NRRD workflows keep working.

    Parameters
    ----------
    hist_path : Path
        Directory containing the histology volumes.
    brain_atlas : AllenAtlas
        Brain atlas for alignment.

    Returns
    -------
    SliceLoader
        A :class:`TiffSliceLoader` if TIFFs are present, otherwise a :class:`NrrdSliceLoader`.
    """
    if any(hist_path.glob('*.tif')) or any(hist_path.glob('*.tiff')):
        return TiffSliceLoader(hist_path, brain_atlas)
    return NrrdSliceLoader(hist_path, brain_atlas)


class TiffSliceLoader(SliceLoader):
    """
    SliceLoader for histology in TIFF format (e.g. brainreg outputs).

    Detects brainreg's standard ``C0`` (green) / ``C1`` (red) channel
    suffixes first, then falls back to the NRRD loader's ``GR`` / ``RD``
    substring rules so manually-named TIFFs also load.

    Parameters
    ----------
    file_path : Path
        Directory containing ``.tif`` / ``.tiff`` files.
    brain_atlas : AllenAtlas
        Brain atlas for alignment.
    """

    def __init__(self, file_path: Path, brain_atlas: AllenAtlas):
        super().__init__(file_path, brain_atlas)

    def get_paths(self) -> None:
        """Locate histology TIFFs and store paths keyed by display label."""
        # Brainreg writes both `.tif` and `.tiff` depending on version.
        files = list(self.file_path.glob('*.tif')) + list(self.file_path.glob('*.tiff'))

        brainreg_map = {'green': 'C0', 'red': 'C1'}

        # Preferred: brainreg files in Allen CCF space — filename contains 'standard'.
        # (Subject-space brainreg outputs share the same C0/C1 suffix but are not in
        # atlas coordinates, so we must not match them when standard ones exist.)
        standard_files = [f for f in files if 'standard' in f.stem]
        for color, abbrev in brainreg_map.items():
            match = next((f for f in standard_files if abbrev in f.stem), None)
            if match:
                self.hist_paths[f'Histology {color}'] = match

        # Fallback 1: any brainreg-style C0/C1 file (e.g. user only kept subject-space).
        for color, abbrev in brainreg_map.items():
            label = f'Histology {color}'
            if label in self.hist_paths:
                continue
            match = next((f for f in files if abbrev in f.stem), None)
            if match:
                self.hist_paths[label] = match

        # Fallback 2: generic GR/RD substring (mirrors NrrdSliceLoader convention)
        # so users with manually-renamed TIFFs do not need brainreg-style names.
        generic_map = {'green': 'GR', 'red': 'RD'}
        for color, abbrev in generic_map.items():
            label = f'Histology {color}'
            if label in self.hist_paths:
                continue
            match = next((f for f in files if abbrev in f.stem), None)
            if match:
                self.hist_paths[label] = match

    def load_volume(self, vol_path: Path) -> np.ndarray:
        """
        Load a TIFF and reorient to AllenAtlas (AP, ML, DV) convention.

        Parameters
        ----------
        vol_path : Path
            A path to a histology TIFF volume.

        Returns
        -------
        np.ndarray
            Loaded volume with shape ``(AP, ML, DV)`` ready for slicing by
            :meth:`SliceLoader.get_slice`.

        Notes
        -----
        Brainreg's ``downsampled_standard_brain_C*.tiff`` are 25 µm
        isotropic in Allen CCF space, with ``sitk.GetArrayFromImage`` axis
        order ``(AP, DV, ML)``. The AllenAtlas convention is ``(AP, ML, DV)``,
        so a single axis swap suffices — no flips required.
        """
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(vol_path)))
        return np.transpose(arr, (0, 2, 1))


def download_histology_data(
    subject: str, laboratory: str
) -> tuple[list[Path], Path] | tuple[None, Path]:
    """
    Download histology data from flatiron server if not already cached locally.

    Parameters
    ----------
    subject: str
        Subject name
    laboratory: str
        Laboratory name

    Returns
    -------
    path_to_files: list[Path] or None
        List of paths to downloaded or cached nrrd files, or None if not found.
    cache_dir: Path
        Directory where files are cached.
    """
    # If we detect >= 2 nrrd file we assume the histology data already exists
    cache_dir = params.get_cache_dir().joinpath(laboratory, 'Subjects', subject, 'histology')
    expected_files = list(cache_dir.glob('*.nrrd'))

    if len(expected_files) >= 2:
        return expected_files, cache_dir

    # Otherwise we attempt to download files
    lab_hist = 'mrsicflogellab' if laboratory == 'hoferlab' else laboratory

    par = params.get()

    def _find_histology_folder(subj: str, lab: str):
        flatiron_path = Path('histology', lab, subj, 'downsampledStacks_25', 'sample2ARA')
        url = f'{par.HTTP_DATA_SERVER}/{"/".join(flatiron_path.parts)}/'
        try:
            response = requests.get(
                url, auth=(par.HTTP_DATA_SERVER_LOGIN, par.HTTP_DATA_SERVER_PWD)
            )
            response.raise_for_status()
            return flatiron_path, response.text
        except Exception as e:
            logger.warning(f'Failed to find path for lab={lab}, subject={subj}: {e}')
            return None

    attempts = [(subject, lab_hist), (subject.replace('_', ''), lab_hist)]

    if lab_hist == 'churchlandlab_ucla':
        attempts.append((subject, 'churchlandlab'))

    histology_folder = None
    for subj, lab in attempts:
        histology_folder = _find_histology_folder(subj, lab)
        if histology_folder:
            break

    if not histology_folder:
        logger.error(f'Could not find histology folder for subject={subject}, lab={laboratory}')
        return None, cache_dir

    rel_path, html_text = histology_folder
    base_url = f'{par.HTTP_DATA_SERVER}/{"/".join(rel_path.parts)}'

    tif_files = [match + '.tif' for match in re.findall(r'href="(.*).tif"', html_text)]

    cache_dir.mkdir(exist_ok=True, parents=True)
    path_to_files = []
    for file in tif_files:
        img_path = Path(cache_dir, file)
        if not img_path.exists():
            file_url = f'{base_url}/{file}'
            http_download_file(
                file_url,
                target_dir=cache_dir,
                username=par.HTTP_DATA_SERVER_LOGIN,
                password=par.HTTP_DATA_SERVER_PWD,
            )
        path_to_files.append(tif2nrrd(img_path))

    if len(path_to_files) > 3:
        path_to_files = path_to_files[1:3]

    return path_to_files, cache_dir


def tif2nrrd(path_to_image: str | Path) -> Path:
    """
    Convert a tif image to nrrd format if the nrrd does not already exist.

    Parameters
    ----------
    path_to_image: str or Path
        The path to the tif file.

    Returns
    -------
    path_to_nrrd: Path
        The path to the nrrd file.
    """
    path_to_nrrd = Path(path_to_image).with_suffix('.nrrd')
    if not path_to_nrrd.exists():
        reader = sitk.ImageFileReader()
        reader.SetImageIO('TIFFImageIO')
        reader.SetFileName(str(path_to_image))
        img = reader.Execute()

        new_img = sitk.PermuteAxes(img, [2, 1, 0])
        new_img = sitk.Flip(new_img, [True, False, False])
        new_img.SetSpacing([1, 1, 1])
        writer = sitk.ImageFileWriter()
        writer.SetImageIO('NrrdImageIO')
        writer.SetFileName(str(path_to_nrrd))
        writer.Execute(new_img)

    return path_to_nrrd
