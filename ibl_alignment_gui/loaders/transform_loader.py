import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ibl_alignment_gui.utils.allen.anatomical_atlas import BrainAtlasAnatomical
from iblatlas.atlas import BrainAtlas

try:
    import ants

    ANTS = True
except ImportError:
    ANTS = False

logger = logging.getLogger(__name__)

# ants.apply_transforms_to_points operates on 3D points.
ANTS_DIMENSION = 3


@dataclass(frozen=True)
class AntsTransformChainFiles:
    """
    The four ANTs transform files warping a point from SmartSPIM space to the Allen CCF.

    The point starts in the SmartSPIM light-sheet anatomical space and ends in the Allen
    CCF, via the SmartSPIM template.
    The chain is applied as: light-sheet -> template (affine + warp), then
    template -> CCF (affine + warp). The affines are inverted when transforming
    points (points move opposite to images), hence ``which_to_invert``.

    smartspim_template_affine_transform : Path
        ``ls_to_template_SyN_0GenericAffine.mat``
    smartspim_template_warp_transform : Path
        ``ls_to_template_SyN_1InverseWarp.nii.gz``
    template_to_ccf_affine_transform : Path
        ``spim_template_to_ccf/syn_0GenericAffine.mat``
    template_to_ccf_warp_transform : Path
        ``spim_template_to_ccf/syn_1InverseWarp.nii.gz``
    """

    smartspim_template_affine_transform: Path
    smartspim_template_warp_transform: Path
    template_to_ccf_affine_transform: Path
    template_to_ccf_warp_transform: Path

    def as_list(self) -> list[str]:
        """Return the transforms as a list of paths in the order ANTs applies them."""
        return [
            self.smartspim_template_affine_transform.as_posix(),
            self.smartspim_template_warp_transform.as_posix(),
            self.template_to_ccf_affine_transform.as_posix(),
            self.template_to_ccf_warp_transform.as_posix(),
        ]

    def which_to_invert(self) -> list[bool]:
        """Return the per-transform invert flags matching the order of :meth:`as_list`."""
        return [True, False, True, False]

    @classmethod
    def from_folder(cls, transforms_path: Path) -> 'AntsTransformChainFiles | None':
        """
        Discover the four transform files beneath *transforms_path*.

        The light-sheet -> template transforms are located recursively (they live under
        ``SmartSPIM_*/image_atlas_alignment/*/`` or the legacy ``SmartSPIM_*/registration/``),
        while the template -> CCF transforms live in a fixed ``spim_template_to_ccf/`` folder.

        Parameters
        ----------
        transforms_path : Path
            Folder holding the SmartSPIM registration assets.

        Returns
        -------
        AntsTransformChainFiles or None
            The resolved transform chain, or None if any of the four files is missing.
        """
        transforms_path = Path(transforms_path)

        def _first(pattern: str) -> Path | None:
            return next(iter(sorted(transforms_path.glob(pattern))), None)

        ls_affine = _first('**/ls_to_template_SyN_0GenericAffine.mat')
        ls_warp = _first('**/ls_to_template_SyN_1InverseWarp.nii.gz')
        ccf_affine = _first('spim_template_to_ccf/syn_0GenericAffine.mat')
        ccf_warp = _first('spim_template_to_ccf/syn_1InverseWarp.nii.gz')

        if not all([ls_affine, ls_warp, ccf_affine, ccf_warp]):
            logger.warning(
                'Incomplete ANTs transform chain in %s; CCF channel locations will not be '
                'written. Found ls_affine=%s, ls_warp=%s, ccf_affine=%s, ccf_warp=%s',
                transforms_path,
                ls_affine,
                ls_warp,
                ccf_affine,
                ccf_warp,
            )
            return None

        return cls(
            smartspim_template_affine_transform=ls_affine,
            smartspim_template_warp_transform=ls_warp,
            template_to_ccf_affine_transform=ccf_affine,
            template_to_ccf_warp_transform=ccf_warp,
        )


class TransformLoader(ABC):
    """
    Abstract base class for warping channel locations into the Allen CCF.

    Loads a registration transform and warps points from an atlas' own physical space
    into the Allen CCF.
    Subclasses must implement `get_transforms` (locate/validate the transform artifacts)
    and `transform_to_ccf` (apply the transform to a set of points).

    Parameters
    ----------
    transforms_path : Path
        Folder containing the transform artifacts.
    """

    def __init__(self, transforms_path: Path):
        self.transforms_path: Path = Path(transforms_path)
        self.transforms = None
        self.get_transforms()

    @abstractmethod
    def get_transforms(self) -> None:
        """Locate and validate the transform artifacts, storing them in `self.transforms`.

        Implementations should leave `self.transforms` as None when the transforms are
        unavailable, so that `exists` reports False and downstream code can skip cleanly.
        """

    @property
    def exists(self) -> bool:
        """Whether a usable transform was loaded."""
        return self.transforms is not None

    @abstractmethod
    def transform_to_ccf(self, points: np.ndarray, atlas: BrainAtlas) -> np.ndarray:
        """
        Warp points from the atlas' physical space into the Allen CCF.

        Parameters
        ----------
        points : np.ndarray
            An (N, 3) array of points in the atlas' physical space (RAS, metres).
        atlas : BrainAtlas
            The atlas the points are defined in.

        Returns
        -------
        np.ndarray
            An (N, 3) array of points in the Allen CCF.
        """


class TransformLoaderAllen(TransformLoader):
    """
    TransformLoader for the SmartSPIM -> Allen CCF ANTs registration pipeline.

    Loads the four-file ANTs transform chain (see :class:`AntsTransformChainFiles`) and
    warps channel locations from a :class:`BrainAtlasAnatomical` physical space into the
    Allen CCF.

    Parameters
    ----------
    transforms_path : Path
        Folder holding the SmartSPIM registration assets.
    """

    def get_transforms(self) -> None:
        """Load the ANTs transform chain, requiring antspyx to be installed."""
        if not ANTS:
            logger.warning(
                'antspyx is not installed; CCF channel locations will not be written. '
                'Install antspyx to enable the SmartSPIM -> CCF transform.'
            )
            return

        self.transforms = AntsTransformChainFiles.from_folder(self.transforms_path)

    def transform_to_ccf(self, points: np.ndarray, atlas: BrainAtlas) -> np.ndarray:
        """
        Warp anatomical points into the Allen CCF using the ANTs transform chain.

        Parameters
        ----------
        points : np.ndarray
            An (N, 3) array of points in the anatomical atlas space (RAS, metres).
        atlas : BrainAtlasAnatomical
            The anatomical atlas the points are defined in. Must carry the SimpleITK
            intensity and pipeline images used to map into the registration space.

        Returns
        -------
        np.ndarray
            An (N, 3) array of points in the Allen CCF, in the native units of the
            registration output.
        """
        if not self.exists:
            raise RuntimeError('No ANTs transform chain loaded; cannot transform to CCF')
        if not isinstance(atlas, BrainAtlasAnatomical):
            raise TypeError(
                f'TransformLoaderAllen requires a BrainAtlasAnatomical, got {type(atlas).__name__}'
            )

        intensity_img = atlas.intensity_sitk_image
        pipeline_img = atlas.pipeline_sitk_image

        # Convert IBL app world coordinates (RAS, m) to ITK world coordinates (LPS, mm).
        ras_to_lps = np.array([-1, -1, 1])
        points_lps_mm = 1e3 * ras_to_lps * points

        # The transforms were computed in the physical space of the pipeline image, so move
        # each point from the intensity image's physical space into the pipeline image's
        # physical space via the shared voxel index.
        pipeline_points: list[list[float]] = []
        for point in points_lps_mm:
            index = intensity_img.TransformPhysicalPointToContinuousIndex(point.tolist())
            pipeline_points.append(
                list(pipeline_img.TransformContinuousIndexToPhysicalPoint(index))
            )

        points_df = pd.DataFrame(np.array(pipeline_points), columns=list('xyz'))

        logger.info('Warping channel locations to CCF')
        ccf_df = ants.apply_transforms_to_points(
            ANTS_DIMENSION,
            points_df,
            self.transforms.as_list(),
            whichtoinvert=self.transforms.which_to_invert(),
        )

        # apply_transforms_to_points preserves row order, so row i is point i.
        return ccf_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
