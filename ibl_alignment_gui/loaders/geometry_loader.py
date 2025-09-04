from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import spikeglx

import ibl_alignment_gui.loaders.data_loader as dloader
import one.alf.io as alfio
from iblutil.util import Bunch
from one.alf.exceptions import ALFObjectNotFound
from one.api import ONE


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
    sort_ind
        Map of sorted raw ephys data to unsorted index
    """

    def __init__(
            self,
            x_coords: np.ndarray,
            y_coords: np.ndarray,
            chn_ind: np.ndarray,
            sort_ind: np.ndarray | None
    ) -> None:

        self.x_coords: np.ndarray = x_coords
        self.y_coords: np.ndarray = y_coords
        self.chn_ind: np.ndarray = chn_ind
        self.sort_ind: np.ndarray | None = sort_ind
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

            # TODO we want this to be somewhere else as it isn't strictly per shank
            info['unsort'] = self.sort_ind
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
            # TODO should apply unsort here
            info['raw_ind'] = self.chn_ind[idx_sort]
            info['sites_x'] = x_coords[y_sort]
            info['sites_y'] = y_coords[y_sort]
            info['sites_min'] = np.nanmin(info['sites_y'])
            info['sites_max'] = np.nanmax(info['sites_y'])
            info['sites_pitch'] = np.min(np.abs(np.diff(np.unique(info['sites_y']))))
            info['sites_full'] = np.arange(info['sites_min'],
                                           info['sites_max'] + info['sites_pitch'],
                                           info['sites_pitch'])
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
        sort_ind = None

        super().__init__(chn_x, chn_y, chn_ind, sort_ind)

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
                self.x_coords >= grp_shank[0], self.x_coords <= grp_shank[-1])
            groups[i] = np.where(shank_chns)[0]

        return groups


class MetaGeometry(Geometry):
    """
    Geometry class using spikeglx ap metadata. Here the sites correspond to the electrode sites.

    Parameters
    ----------
    meta: Bunch
        A Bunch object containing spikeglx metadata
    sorted: bool
        Whether the ephys data has already been sorted or not.
    """

    def __init__(self, meta: Bunch[str, Any], sort: bool = False) -> None:

        self.meta = spikeglx.geometry_from_meta(meta, sort=False)
        elec_x = self.meta['x']
        elec_y = self.meta['y']
        if sort:
            _, sort_vals = spikeglx.geometry_from_meta(meta, sort=True, return_index=True)
            sort_ind = np.empty_like(sort_vals)
            sort_ind[sort_vals] = np.arange(sort_vals.size)
        else:
            sort_ind = np.arange(elec_x.size)

        chn_ind = np.arange(elec_x.size)

        super().__init__(elec_x, elec_y, chn_ind, sort_ind)

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

class GeometryLoader(ABC):
    """
    Abstract base class for loading probe geometry from metadata or channels.

    Subclasses must implement the `load_meta_data` and `load_channels` methods.
    """

    def __init__(self):

        self.electrodes: Geometry | None = None
        self.channels: Geometry | None = None

    def get_geometry(self, sort=False):
        """Load probe geometry from both the metadata and the channels."""
        meta = self.load_meta_data()
        if meta is not None:
            self.electrodes = MetaGeometry(meta, sort=sort)
            self.electrodes.split_sites_per_shank()

        chns = self.load_channels()
        if chns is not None:
            self.channels = ChannelGeometry(chns)
            self.channels.split_sites_per_shank()

        if self.electrodes is None and self.channels is None:
            raise ValueError("Could not load geometry: metadata and channels both missing")

        # TODO we need to check that metadata and channels are equivalent.
        #  If they are not then we use the channels and put out a warning

    @abstractmethod
    def load_meta_data(self) -> Bunch[str, Any] | None:
        """Load probe metadata from spikeglx ap.meta file."""

    @abstractmethod
    def load_channels(self, **kwargs) -> Bunch[str, np.ndarray] | None:
        """Load spike sorting channels data."""

    def get_sites_for_shank(self, shank_idx, sites=None) -> Bunch[str, Any]:
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
            shank_sites = self.channels._get_sites_for_shank(shank_idx) \
                if self.channels is not None else self.electrodes._get_sites_for_shank(shank_idx)
        else:
            shank_sites = self.electrodes._get_sites_for_shank(shank_idx) \
                if self.electrodes is not None else self.channels._get_sites_for_shank(shank_idx)

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

    def __init__(self, insertion: dict[str, Any], one: ONE, session_path: Path | None = None,
                 probe_collection: str | None = None):

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
                self.eid, '*.ap.meta', collection=f'raw_ephys_data/{self.probe_label}',
                download_only=True)
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
            self.one.load_object, self.eid, 'channels', collection=self.probe_collection,
            attribute=['localCoordinates', 'rawInd'], **kwargs)

        return chns if chns['exists'] else None


class GeometryLoaderLocal(GeometryLoader):
    """
    Geometry loader using local file system.

    Parameters
    ----------
    probe_path: Path
        A path to root folder containing the spike sorting and metadata collections.
    collections: dloader.CollectionData
        A CollectionData instance specifying the folders relative to the rootpath that
        contain the spikesorting
         and metadata data.
    """

    def __init__(self, probe_path: Path, collections: dloader.CollectionData):

        self.probe_path: Path = probe_path
        self.spike_path: Path = probe_path.joinpath(collections.spike_collection)
        self.meta_path: Path = probe_path.joinpath(collections.meta_collection)

        super().__init__()

    def load_meta_data(self) -> dict | None:
        """
        Load probe metadata from spikeglx ap.meta file.

        Returns
        -------
        dict or None
            A dict containing the spikeglx AP band metadata, or None if not found.
        """
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
        chns = dloader.DataLoader.load_data(alfio.load_object, self.spike_path, 'channels',
                              attribute=['localCoordinates', 'rawInd'], **kwargs)
        return chns if chns['exists'] else None



def arrange_channels_into_banks(
        shank_geom: Bunch[str, Any],
        data: np.ndarray,
        bnk_width: int = 10
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
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
    bnk_data : list of np.ndarray
        List of data values for each bank, shape of each array is (1, n_channels_per_bank)
        for each bank
    bnk_scale : list of np.ndarray
        List of scaling factors for each bank along x and y axes
    bnk_offset : list of np.ndarray
        List of offset values for each bank along x and y axes.
    bnk_index : list of np.ndarray
        List of original indices for data in each bank.

    """
    bnk_data = list()
    bnk_scale = list()
    bnk_offset = list()
    bnk_index = list()

    for ibank, bank in enumerate(np.unique(shank_geom['sites_x'])):

        # Find the channels in the current bank
        bnk_chns = np.where(shank_geom['sites_x'] == bank)[0]
        bnk_ycoords = shank_geom['sites_y'][bnk_chns]
        # Find the spacing between the channels in the current bank
        bnk_diff = np.min(np.abs(np.diff(bnk_ycoords)))

        # NP1.0 checkerboard
        if bnk_diff != shank_geom['sites_pitch']:
            bnk_full = np.arange(np.min(bnk_ycoords), np.max(bnk_ycoords) + bnk_diff, bnk_diff)
            idx_full = np.where(np.isin(bnk_full, bnk_ycoords))
            bnk_vals = np.full((bnk_full.shape[0]), np.nan)
            bnk_yoffset = np.min(bnk_ycoords)

        else:  # NP2.0
            idx_full = np.where(np.isin(shank_geom['sites_full'], bnk_ycoords))
            bnk_vals = np.full((shank_geom['sites_full'].shape[0]), np.nan)
            bnk_yoffset = shank_geom['sites_min']

        # Fill in the data for the channels in the current bank
        bnk_vals[idx_full] = data[bnk_chns]
        bnk_vals = bnk_vals[np.newaxis, :]

        # Get the scaling and offset for the current bank
        bnk_yscale = ((shank_geom['sites_max'] - shank_geom['sites_min']) / bnk_vals.shape[1])
        bnk_xscale = bnk_width / bnk_vals.shape[0]
        bnk_xoffset = bnk_width * ibank

        bnk_index.append(bnk_chns)
        bnk_data.append(bnk_vals)
        bnk_scale.append(np.array([bnk_xscale, bnk_yscale]))
        bnk_offset.append(np.array([bnk_xoffset, bnk_yoffset]))

    return bnk_data, bnk_scale, bnk_offset, bnk_index


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
    _, chn_depth, chn_count = np.unique(shank_geom['sites_y'], return_index=True,
                                        return_counts=True)
    chn_depth_eq = np.copy(chn_depth)
    chn_depth_eq[np.where(chn_count == 2)] += 1

    # Average pairs of channels at the same depth
    averaged_data = np.mean(
        np.stack([data[:, chn_depth], data[:, chn_depth_eq]], axis=-1), axis=-1,)

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
