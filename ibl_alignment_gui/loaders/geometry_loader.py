import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import spikeglx

import ibl_alignment_gui.loaders.data_loader as dloader
import one.alf.io as alfio
from ibl_alignment_gui.utils.parse_yaml import DatasetPaths
from iblutil.util import Bunch
from one.alf.exceptions import ALFObjectNotFound
from one.api import ONE

logger = logging.getLogger(__name__)


class Geometry(ABC):
    """
    Abstract base class for splitting the sites on a probe per shank.

    Sites can either correspond to spike sorting channels or raw data electrodes.

    Subclasses must implement the `_get_n_shanks` and `_get_shank_groups` methods.

    Parameters
    ----------
    x_coords: np.ndarray
        x coordinates of the sites
    y_coords: np.ndarray
        y coordinates of the sites
    chn_ind
        Map of spike-sorting channels to raw data
    """

    def __init__(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        chn_ind: np.ndarray,
    ) -> None:
        self.x_coords: np.ndarray = x_coords
        self.y_coords: np.ndarray = y_coords
        self.chn_ind: np.ndarray = chn_ind
        self.n_shanks: int | None = None
        self.shank_groups: Bunch | None = None
        self.shanks: Bunch | None = None

    @abstractmethod
    def _get_n_shanks(self):
        """Get the number of shanks on the probe."""

    @abstractmethod
    def _get_shank_groups(self):
        """Group the sites per shank."""

    def split_sites_per_shank(self) -> None:
        """
        Split the sites into shanks and store the results in `self.shanks`.

        Channels are sorted according to the y coordinates.

        Each entry in self.shanks is a Bunch containing:
            spikes_ind: np.ndarray
              Indices of the sites that correspond to the spike sorting data.
            raw_ind: np.ndarray
              Indices of sites relative to raw data ordering.
            site_coords: np.ndarray
              (x, y) coordinates of the shank sites.
            sites_min/ sites_max: float
              Minimum and maximum y coordinate values.
            sites_pitch: float
              Minimum difference between y coordinates (sites spacing).
            sites_full: np.ndarray
              Full set of y coordinates covering the whole shank.
            idx_full: np.ndarray
              Indices of the actual y coordinates within sites_full.
            n_banks: int
              Number of banks in the shank.
        """
        self.n_shanks = self._get_n_shanks()
        self.shank_groups = self._get_shank_groups()
        self.shanks = Bunch()

        for i in range(self.n_shanks):
            info = Bunch()

            orig_idx = self.shank_groups[i]
            x_coords = self.x_coords[orig_idx]
            y_coords = self.y_coords[orig_idx]
            # These are unsorted and are used to save the channels for the alignment
            # in the original order
            info['orig_idx'] = orig_idx
            info['sites_coords'] = np.c_[x_coords, y_coords]
            # These are sorted by depth and are used for plotting
            y_sort = np.argsort(y_coords)
            idx_sort = orig_idx[y_sort]
            # These are the sites that match into the spike sorting per shank
            info['spikes_ind'] = self.chn_ind[idx_sort]
            # These are the sites that match into the raw_data per shank
            info['raw_ind'] = self.chn_ind[idx_sort]
            info['sites_x'] = x_coords[y_sort]
            info['sites_y'] = y_coords[y_sort]
            info['sites_min'] = np.nanmin(info['sites_y'])
            info['sites_max'] = np.nanmax(info['sites_y'])
            info['sites_pitch'] = np.min(np.abs(np.diff(np.unique(info['sites_y']))))
            info['sites_full'] = np.arange(
                info['sites_min'], info['sites_max'] + info['sites_pitch'], info['sites_pitch']
            )
            info['idx_full'] = np.where(np.isin(info['sites_full'], info['sites_y']))[0]
            info['n_banks'] = np.unique(info['sites_x']).size
            self.shanks[i] = info

    def _get_sites_for_shank(self, shank_idx: int) -> Bunch[str, Any]:
        """
        Get the sites information for a given shank.

        Parameters
        ----------
        shank_idx : int
            Index of the shank.

        Returns
        -------
        Bunch
            Site information for the given shank.
        """
        return self.shanks[shank_idx]


class ChannelGeometry(Geometry):
    """
    Geometry class using spike sorting channel data.

    Here the sites correspond to the spike sorting channels.

    Parameters
    ----------
    channels: Bunch
        A Bunch object containing spike sorting channels data.
    shank_diff: int
        The minimum difference in x coordinates to separate shanks.
    """

    def __init__(self, channels: Bunch[str, np.ndarray], shank_diff: int = 100) -> None:
        self.shank_diff: int = shank_diff
        chn_x = channels['localCoordinates'][:, 0]
        chn_y = channels['localCoordinates'][:, 1]
        chn_ind = channels['rawInd']

        super().__init__(chn_x, chn_y, chn_ind)

    def _get_n_shanks(self) -> int:
        """
        Detect the number of shanks on the probe using the difference in spacing in x coordinates.

        Returns
        -------
        int
            The number of shanks detected on the probe
        """
        x_coords = np.unique(self.x_coords)
        x_coords_diff = np.diff(x_coords)
        n_shanks = np.sum(x_coords_diff > self.shank_diff) + 1
        return n_shanks

    def _get_shank_groups(self) -> Bunch[int, np.ndarray]:
        """
        Get the channel index per shank using the difference in spacing in x coordinates.

        Returns
        -------
        Bunch
            A bunch containing the sites indices for each shank.
        """
        x_coords = np.unique(self.x_coords)
        shank_groups = np.split(x_coords, np.where(np.diff(x_coords) > self.shank_diff)[0] + 1)

        assert len(shank_groups) == self.n_shanks

        groups = Bunch()
        for i, grp in enumerate(shank_groups):
            grp_sort = np.sort(grp)
            if len(grp_sort) == 1:
                grp_sort = np.array([grp_sort[0], grp_sort[0]])
            grp_shank = [grp_sort[0], grp_sort[-1]]

            shank_chns = np.bitwise_and(
                self.x_coords >= grp_shank[0], self.x_coords <= grp_shank[-1]
            )
            groups[i] = np.where(shank_chns)[0]

        return groups


class MetaGeometry(Geometry):
    """
    Geometry class using spikeglx ap metadata. Here the sites correspond to the electrode sites.

    Parameters
    ----------
    meta: Bunch
        A Bunch object containing spikeglx metadata
    """

    def __init__(self, meta: Bunch[str, Any]) -> None:
        self.meta = spikeglx.geometry_from_meta(meta, sort=False)
        elec_x = self.meta['x']
        elec_y = self.meta['y']

        chn_ind = np.arange(elec_x.size)

        super().__init__(elec_x, elec_y, chn_ind)

    def _get_n_shanks(self) -> int:
        """
        Detect the number of shanks on the probe using the spikeglx metadata.

        Returns
        -------
        int
            The number of shanks detected on the probe
        """
        n_shanks = np.unique(self.meta['shank']).size

        return n_shanks

    def _get_shank_groups(self) -> Bunch[int, np.ndarray]:
        """
        Get the channel index per shank using the spikeglx metadata.

        Returns
        -------
        Bunch
            A bunch containing the channel indices for each shank.
        """
        groups = Bunch()
        shanks = np.unique(self.meta['shank'])

        for i, sh in enumerate(shanks):
            groups[i] = np.where(self.meta['shank'] == sh)[0]

        return groups


def find_geometry_mismatches(electrodes: Geometry, channels: Geometry) -> list[str]:
    """
    Describe the ways the electrode and channel geometries disagree about the probe layout.

    The spike sorting channels are expected to describe a subset of the electrodes, so the
    number of sites and their exact depths are not required to match - channels that recorded
    no spikes may be missing entirely. What is checked is that each shank has the same number
    of banks, and that every channel site falls within the depth range the metadata reports for
    that shank.

    Note that the x coordinates are deliberately not compared: the metadata x is relative to
    each shank, so the same values repeat across shanks, whereas the spike sorting
    localCoordinates are absolute across the whole probe.

    Parameters
    ----------
    electrodes : Geometry
        Geometry built from the ap.meta metadata, split per shank.
    channels : Geometry
        Geometry built from the spike sorting channels, split per shank.

    Returns
    -------
    list of str
        One entry per difference found, empty when the two agree.
    """
    if electrodes.n_shanks != channels.n_shanks:
        return [
            f'number of shanks differs: {electrodes.n_shanks} (metadata) '
            f'vs {channels.n_shanks} (channels)'
        ]

    mismatches = []
    for i in range(electrodes.n_shanks):
        elec = electrodes.shanks[i]
        chn = channels.shanks[i]

        if elec['n_banks'] != chn['n_banks']:
            mismatches.append(
                f'shank {i}: number of banks differs: {elec["n_banks"]} (metadata) '
                f'vs {chn["n_banks"]} (channels)'
            )

        outside = (chn['sites_y'] < elec['sites_min']) | (chn['sites_y'] > elec['sites_max'])
        if np.any(outside):
            mismatches.append(
                f'shank {i}: {np.sum(outside)} of {chn["sites_y"].size} channel sites fall '
                f'outside the {elec["sites_min"]} to {elec["sites_max"]} depth range of the '
                f'metadata'
            )

    return mismatches


class GeometryLoader(ABC):
    """
    Abstract base class for loading probe geometry from metadata or channels.

    Subclasses must implement the `load_meta_data` and `load_channels` methods.
    """

    def __init__(self):
        self.electrodes: Geometry | None = None
        self.channels: Geometry | None = None

    def get_geometry(self):
        """Load probe geometry from both the metadata and the channels."""
        meta = self.load_meta_data()
        if meta is not None:
            self.electrodes = MetaGeometry(meta)
            self.electrodes.split_sites_per_shank()

        chns = self.load_channels()
        if chns is not None:
            self.channels = ChannelGeometry(chns)
            self.channels.split_sites_per_shank()

        if self.electrodes is None and self.channels is None:
            raise ValueError('Could not load geometry: metadata and channels both missing')

        # When both sources are available they should describe the same probe. If they do not,
        # the spike sorting channels are the ones the data is indexed against, so drop the
        # electrodes and fall back to the channels everywhere (see get_sites_for_shank).
        if self.electrodes is not None and self.channels is not None:
            mismatches = find_geometry_mismatches(self.electrodes, self.channels)
            if mismatches:
                logger.warning(
                    'The probe geometry read from the ap.meta metadata does not match the '
                    'spike sorting channels, using the channels instead. Differences: %s',
                    '; '.join(mismatches),
                )
                self.electrodes = None

    @abstractmethod
    def load_meta_data(self) -> Bunch[str, Any] | None:
        """Load probe metadata from spikeglx ap.meta file."""

    @abstractmethod
    def load_channels(self, **kwargs) -> Bunch[str, np.ndarray] | None:
        """Load spike sorting channels data."""

    def get_sites_for_shank(self, shank_idx: int, sites=None) -> Bunch[str, Any]:
        """
        Get the sites information for a given shank.

        By default, the site information from the electrodes (ap.metadata) is returned.
        If sites='channels', the site information for the channels is returned

        Parameters
        ----------
        shank_idx: int
            Index of the shank.
        sites: str
            The origin of the site information

        Returns
        -------
        Bunch
            Site information for the given shank.
        """
        if sites == 'channels':
            # TODO add a logger if channels don't exist to say we are using electrodes
            shank_sites = (
                self.channels._get_sites_for_shank(shank_idx)
                if self.channels is not None
                else self.electrodes._get_sites_for_shank(shank_idx)
            )
        else:
            shank_sites = (
                self.electrodes._get_sites_for_shank(shank_idx)
                if self.electrodes is not None
                else self.channels._get_sites_for_shank(shank_idx)
            )

        return shank_sites


class GeometryLoaderOne(GeometryLoader):
    """
    Geometry loader using the ONE API.

    Parameters
    ----------
    insertion : dict
        Dictionary representing a probe insertion (must include 'session' and 'name').
    one : ONE
        An ONE instance used to access data.
    session_path : Path or None
        Path to the session folder. If None, it is resolved using the eid via `one.eid2path`.
    probe_collection : str or None
        The collection to the spike sorting data to load
    """

    def __init__(
        self,
        insertion: dict[str, Any],
        one: ONE,
        session_path: Path | None = None,
        probe_collection: str | None = None,
    ):
        self.one: ONE = one
        self.eid: str = insertion['session']
        self.session_path: Path = session_path or one.eid2path(self.eid)
        self.probe_label: str = insertion['name']
        self.probe_collection: str = probe_collection

        super().__init__()

    def load_meta_data(self) -> dict | None:
        """
        Load probe metadata from spikeglx ap.meta file.

        Returns
        -------
        dict or None
            A dict containing the spikeglx AP band metadata, or None if not found.
        """
        try:
            meta_file = self.one.load_dataset(
                self.eid,
                '*.ap.meta',
                collection=f'raw_ephys_data/{self.probe_label}',
                download_only=True,
            )
            return spikeglx.read_meta_data(meta_file)
        except ALFObjectNotFound:
            return None

    def load_channels(self, **kwargs) -> Bunch[str, Any] | None:
        """
        Load spike sorting channels data.

        Returns
        -------
        Bunch or None
            A Bunch containing the channels data, or None if not found.
        """
        chns = dloader.DataLoader.load_data(
            self.one.load_object,
            self.eid,
            'channels',
            collection=self.probe_collection,
            attribute=['localCoordinates', 'rawInd'],
            **kwargs,
        )

        return chns if chns['exists'] else None


class GeometryLoaderLocal(GeometryLoader):
    """
    Geometry loader using local file system.

    Parameters
    ----------
    data_paths : DatasetPaths
        The resolved dataset paths for the probe. The channels are read from the
        ``spike_sorting`` folder and the metadata from the ``raw_ephys`` folder.
    """

    def __init__(self, data_paths: DatasetPaths):
        self.spike_path: Path = data_paths.spike_sorting
        self.meta_path: Path = data_paths.raw_ephys

        super().__init__()

    def load_meta_data(self) -> dict | None:
        """
        Load probe metadata from spikeglx ap.meta file.

        Returns
        -------
        dict or None
            A dict containing the spikeglx AP band metadata, or None if not found.
        """
        if self.meta_path is None:
            return None
        meta_file = next(self.meta_path.glob('*.ap.*meta'), None)
        return spikeglx.read_meta_data(meta_file) if meta_file else None

    def load_channels(self, **kwargs) -> Bunch[str, Any] | None:
        """
        Load spike sorting channels data.

        Returns
        -------
        Bunch or None
            A Bunch containing the channels data, or None if not found.
        """
        chns = dloader.DataLoader.load_data(
            alfio.load_object,
            self.spike_path,
            'channels',
            attribute=['localCoordinates', 'rawInd'],
            **kwargs,
        )
        return chns if chns['exists'] else None


def arrange_channels_into_banks(
    shank_geom: Bunch[str, Any], data: np.ndarray, bnk_width: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Arrange channel data into probe banks for visualization.

    Channels are grouped by bank (x-coordinate). For each bank, channel values
    are aligned along the y-axis. If the spacing between channels does not
    match the expected spacing (`self.chn_diff`), gaps are filled with NaNs

    Parameters
    ----------
    shank_geom: Bunch
        A bunch containing channel geometry information for a shank,
        returned from `Geometry.get_chns_for_shank()`.
    data : np.ndarray
        1D array of values per channel
    bnk_width: int
        The width of each bank in the x dimension for visualization.

    Returns
    -------
    bnk_data : np.ndarray
        A 2D array with data organised into individual banks on the shank.
    bnk_scale : np.ndarray
        Scale factor to apply along x and y axes
    bnk_offset : list of np.ndarray
        Offset to apply along x and y axes.
    """
    # Find the minimum spacing between channels in each bank
    x_coords = np.unique(shank_geom['sites_x'])
    bnk_diff = []
    for x in x_coords:
        bnk_chns = np.where(shank_geom['sites_x'] == x)[0]
        bnk_ycoords = shank_geom['sites_y'][bnk_chns]
        bnk_diff.append(np.min(np.abs(np.diff(bnk_ycoords))))
    bnk_diff = np.min(bnk_diff)

    if bnk_diff != shank_geom['sites_pitch']:
        bnk_data = np.full((shank_geom['sites_full'].shape[0] + 1, shank_geom['n_banks']), np.nan)
    else:
        bnk_data = np.full((shank_geom['sites_full'].shape[0], shank_geom['n_banks']), np.nan)

    for ibank, bank in enumerate(np.unique(shank_geom['sites_x'])):
        # Find the channels in the current bank
        bnk_chns = np.where(shank_geom['sites_x'] == bank)[0]
        bnk_ycoords = shank_geom['sites_y'][bnk_chns]

        # NP1.0 checkerboard
        if bnk_diff != shank_geom['sites_pitch']:
            idx_full = np.where(np.isin(shank_geom['sites_full'], bnk_ycoords))[0]
            bnk_data[idx_full, ibank] = data[bnk_chns]
            # Fill in the extra row for checkerboard display
            bnk_data[[idx_full + 1], ibank] = data[bnk_chns]
        else:  # NP2.0
            idx_full = np.where(np.isin(shank_geom['sites_full'], bnk_ycoords))[0]
            # Fill in the data for the channels in the current bank
            bnk_data[idx_full, ibank] = data[bnk_chns]

    # Get the scaling and offset for the shank
    bnk_yscale = (shank_geom['sites_max'] - shank_geom['sites_min']) / bnk_data.shape[0]
    bnk_xscale = bnk_width
    bnk_offset = np.array([0, shank_geom['sites_min']])

    return bnk_data.T, np.array([bnk_xscale, bnk_yscale]), bnk_offset


def average_chns_at_same_depths(shank_geom: Bunch[str, Any], data: np.ndarray) -> np.ndarray:
    """
    Average data across channels at the same depth.

    Parameters
    ----------
    shank_geom: Bunch
        A bunch containing channel geometry information for a shank, returned
        from `Geometry.get_chns_for_shank()`.
    data : np.ndarray
        2D array of data with shape (time or frequency x channels).

    Returns
    -------
    np.ndarray
        2D array with averaged data across equivalent depths.
    """
    # Identify channels at the same depth
    _, chn_depth, chn_count = np.unique(
        shank_geom['sites_y'], return_index=True, return_counts=True
    )
    chn_depth_eq = np.copy(chn_depth)
    chn_depth_eq[np.where(chn_count == 2)] += 1

    # Average pairs of channels at the same depth
    averaged_data = np.nanmean(
        np.stack([data[:, chn_depth], data[:, chn_depth_eq]], axis=-1),
        axis=-1,
    )

    return averaged_data


def pad_data_to_full_chn_map(shank_geom: Bunch[str, Any], data: np.ndarray) -> np.ndarray:
    """
    Pad data to align with the full channel map, filling gaps with NaNs.

    Parameters
    ----------
    shank_geom: Bunch
        A bunch containing channel geometry information for a shank, returned
        from `Geometry.get_chns_for_shank()`.
    data : np.ndarray
        2D array of data with shape (time or frequency x channels).

    Returns
    -------
    np.ndarray
        2D array padded to the full channel map.
    """
    padded_data = np.full((data.shape[0], shank_geom['sites_full'].shape[0]), np.nan)
    padded_data[:, shank_geom['idx_full']] = data

    return padded_data
