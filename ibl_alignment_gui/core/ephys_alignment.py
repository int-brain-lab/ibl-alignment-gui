"""Geometric alignment between electrophysiology features and a histology track.

An :class:`EphysAlignment` maps positions along a probe (the electrophysiology
"feature" space the user sees) onto 3D brain coordinates, warping between the two
using a set of user-placed reference lines.

Two 1D spaces are used throughout:

- **feature space**: depth along the ephys plots the user annotates (metres).
- **track space**: depth along the reconstructed histology track (metres).

:meth:`EphysAlignment.feature2track` and :meth:`EphysAlignment.track2feature`
interpolate between these spaces using the reference lines, and points in track
space are turned into 3D coordinates by interpolating along ``xyz_track``.

Coordinate convention
----------------------
All ``xyz`` coordinates are in the atlas RAS frame -- (x, y, z) = (Right, Anterior,
Superior) -- expressed in metres.
"""

import logging

import numpy as np
import pandas as pd
import scipy

from iblatlas import atlas
from iblatlas.atlas import BrainAtlas
from iblutil.util import Bunch

logger = logging.getLogger(__name__)

TIP_SIZE_UM = 200


def _cumulative_distance(xyz: np.ndarray) -> np.ndarray:
    """
    Cumulative Euclidean distance along a sequence of points.

    Parameters
    ----------
    xyz : np.ndarray
        (n_points, 3) xyz coordinates.

    Returns
    -------
    np.ndarray
        (n_points,) cumulative distance from the first point, starting at 0.
    """
    return np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))])


def interpolate_along_track(xyz_track: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """
    Get the xyz coordinates of points a given distance along a track.

    Walks along the piecewise-linear track and linearly interpolates a 3D position for
    each requested distance, measured from the first (usually deepest) point.

    Parameters
    ----------
    xyz_track : np.ndarray
        xyz coordinates defining the track; the first point is usually
        the deepest.
    depths : np.ndarray
        (n_depths,) distances from the first point of the track, by convention 0 at the
        deepest point and increasing towards the surface.

    Returns
    -------
    np.ndarray
        Interpolated xyz coordinates.
    """
    distance = _cumulative_distance(xyz_track)
    xyz_channels = np.zeros((depths.shape[0], 3))
    for m in np.arange(3):
        xyz_channels[:, m] = np.interp(depths, distance, xyz_track[:, m])

    return xyz_channels


class EphysAlignment:
    """
    Align electrophysiology features to a reconstructed histology track.

    See the module docstring for the feature-space / track-space convention. All
    ``xyz`` coordinates refer to RAS convention (metres).
    """

    def __init__(
        self,
        xyz_picks: np.ndarray,
        chn_depths: np.ndarray | None = None,
        track_prev: np.ndarray | None = None,
        feature_prev: np.ndarray | None = None,
        brain_atlas: BrainAtlas | None = None,
        speedy: bool = False,
        track_margin_m: float = 6e-3,
    ) -> None:
        """
        Set up the alignment from the user-picked trajectory.

        Builds the full insertion track from the picks, initialises the feature/track
        reference points (either from a previous alignment or as an identity mapping
        sized to the probe), samples the track through the atlas, and precomputes the
        histology regions the track passes through.

        Parameters
        ----------
        xyz_picks : np.ndarray
             User-picked xyz coordinates defining the probe trajectory.
        chn_depths : np.ndarray or None
            Channel depths along the probe (um). Used to size the initial track range.
        track_prev : np.ndarray or None
            Track reference points from a previous alignment, if resuming one.
        feature_prev : np.ndarray or None
            Feature reference points from a previous alignment, if resuming one.
        brain_atlas : BrainAtlas or None
            Atlas used for coordinate and region lookups. Defaults to ``AllenAtlas(25)``.
        speedy : bool
            If True, estimate the brain exit from the atlas z-limits instead of the
            (slower) brain-surface intersection.
        track_margin_m : float
            Default half-range (m) for the initial track when no channel or previous
            alignment information is available.
        """
        if not brain_atlas:
            self.brain_atlas = atlas.AllenAtlas(25)
        else:
            self.brain_atlas = brain_atlas

        # xyz coordinates are RAS (Right, Anterior, Superior), in metres
        self.xyz_track, self.track_extent, self.cumulative_dist = self.get_insertion_track(
            xyz_picks,
            speedy=speedy,
        )

        self.chn_depths = chn_depths
        if np.any(track_prev):
            # Resume from a previous alignment
            self.track_init = track_prev
            self.feature_init = feature_prev
        else:
            # Start from an identity feature->track mapping spanning the probe.
            # Determine required range based on probe geometry
            tip_track_m = -1 * track_margin_m
            if chn_depths is not None and len(chn_depths) > 0:
                probe_span = 1e-6 * np.max(chn_depths)  # meters
                # Add 50% margin for alignment flexibility
                margin_factor = 1.5
                top_track_m = max(track_margin_m, probe_span * margin_factor)
            else:
                # Default to 6mm if no channel information available
                top_track_m = track_margin_m

            self.track_init = np.array([tip_track_m, top_track_m])
            self.feature_init = np.copy(self.track_init)

        # Sample the track through the atlas to find the regions it crosses.
        if isinstance(self.brain_atlas, atlas.AllenAtlas):
            # Allen atlas: fixed 10 um spacing along the track
            self.sampling_trk = np.arange(
                self.track_extent[0],
                self.track_extent[-1] - 10 * 1e-6,
                10 * 1e-6,
            )
            self.xyz_samples = interpolate_along_track(
                self.xyz_track,
                self.sampling_trk - self.sampling_trk[0],
            )
        else:
            # Other atlases: sample once per voxel in the DV (z) direction, so the
            # sampling resolution follows the atlas rather than a hard-coded 10 um.
            # xyz_track is sorted by z, so its first/last points bound the z range.
            i_min, i_max = np.sort(
                self.brain_atlas.bc.z2i(self.xyz_track[[0, -1], 2], mode='clip'),
            )
            z_samples = np.sort(self.brain_atlas.bc.i2z(np.arange(int(i_min), int(i_max) + 1)))
            depths_at_z_samples = np.interp(z_samples, self.xyz_track[:, 2], self.cumulative_dist)
            self.xyz_samples = interpolate_along_track(self.xyz_track, depths_at_z_samples)
            # track_extent[0] == -first_electrode_dist, so this matches the depth
            # convention used by the Allen branch above
            self.sampling_trk = depths_at_z_samples + self.track_extent[0]

        # Drop any samples that fall outside the atlas x/y (ML/AP) bounds
        xlim = np.sort(self.brain_atlas.bc.xlim)
        ylim = np.sort(self.brain_atlas.bc.ylim)
        x_in_range = np.bitwise_and(
            self.xyz_samples[:, 0] >= xlim[0],
            self.xyz_samples[:, 0] <= xlim[1],
        )
        y_in_range = np.bitwise_and(
            self.xyz_samples[:, 1] >= ylim[0],
            self.xyz_samples[:, 1] <= ylim[1],
        )
        rem = np.bitwise_and(x_in_range, y_in_range)
        self.xyz_samples = self.xyz_samples[rem]
        self.sampling_trk = self.sampling_trk[rem]

        self.region, self.region_label, self.region_colour, self.region_id = (
            self.get_histology_regions(self.xyz_samples, self.sampling_trk, self.brain_atlas)
        )

    def get_insertion_track(
        self,
        xyz_picks: np.ndarray,
        speedy: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extend the probe trajectory from the bottom of the brain to the top of the atlas.

        Fits a line to the first and last portions of the picks and extrapolates to the
        brain exit (bottom) and the top of the atlas (entry), so the track spans the
        full atlas even when channels sit above the brain surface.

        Parameters
        ----------
        xyz_picks : np.ndarray
            xyz coordinates defining the probe trajectory.
        speedy : bool
            If True, estimate the brain exit from the atlas z-limits instead of the
            (slower) brain-surface intersection.

        Returns
        -------
        xyz_track : np.ndarray
            xyz coordinates of the extended trajectory, sorted so the most ventral
            (deepest) point is first.
        track_extent : np.ndarray
            Distance to the deepest and shallowest points of the track, offset so
            that 0 is at the first electrode.
        cumulative_dist : np.ndarray
            Cumulative distance along xyz_track from the deepest point.
        """
        # Use the first and last quarter of xyz_picks to estimate the trajectory beyond xyz_picks
        n_picks = np.max([4, round(xyz_picks.shape[0] / 4)])
        traj_entry = atlas.Trajectory.fit(xyz_picks[:n_picks, :])
        traj_exit = atlas.Trajectory.fit(xyz_picks[-1 * n_picks :, :])

        # Force the entry to be on the upper z lim of the atlas to account for cases where channels
        # may be located above the surface of the brain
        entry_lims = traj_entry.eval_z(self.brain_atlas.bc.zlim)
        entry_top_lim = np.argmax(entry_lims[:, 2])
        entry = entry_lims[entry_top_lim, :]
        if speedy:
            exit_lims = traj_exit.eval_z(self.brain_atlas.bc.zlim)
            exit_bottom_lim = np.argmin(exit_lims[:, 2])
            brain_exit = exit_lims[exit_bottom_lim, :]
        else:
            brain_exit = atlas.Insertion.get_brain_exit(traj_exit, self.brain_atlas)
            # The exit is just below the bottom surface of the brain
            brain_exit[2] = brain_exit[2] - 200 / 1e6

        # Fall back to the atlas z-limit if the surface intersection failed
        if any(np.isnan(brain_exit)):
            brain_exit = (traj_exit.eval_z(self.brain_atlas.bc.zlim))[1, :]
        xyz_track = np.r_[brain_exit[np.newaxis, :], xyz_picks, entry[np.newaxis, :]]
        # Sort so that most ventral coordinate is first
        xyz_track = xyz_track[np.argsort(xyz_track[:, 2]), :]

        # Offset distances so that 0 sits at the first electrode (TIP_SIZE_UM above the tip)
        cumulative_dist = _cumulative_distance(xyz_track)
        tip_distance = cumulative_dist[1] + TIP_SIZE_UM / 1e6
        track_length = cumulative_dist[-1]
        track_extent = np.array([0, track_length]) - tip_distance

        return xyz_track, track_extent, cumulative_dist

    def get_track_and_feature(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the current feature, track and xyz_track arrays.

        Returns
        -------
        feature_init : np.ndarray
            Initial feature-space reference points.
        track_init : np.ndarray
            Initial track-space reference points.
        xyz_track : np.ndarray
            xyz coordinates of the extended trajectory.
        """
        return self.feature_init, self.track_init, self.xyz_track

    @staticmethod
    def feature2track(
        feature_new: np.ndarray,
        feature_ref: np.ndarray,
        track_ref: np.ndarray,
    ) -> np.ndarray:
        """
        Convert feature-space points to track space via the reference-line fit.

        Builds a linear interpolant from the (feature_ref, track_ref) reference pairs
        and evaluates it, extrapolating beyond the outermost reference lines.

        Parameters
        ----------
        feature_new : np.ndarray
            Points in feature space to convert to track space.
        feature_ref : np.ndarray
            Reference coordinates in feature space (ephys plots).
        track_ref : np.ndarray
            Reference coordinates in track space (histology track).

        Returns
        -------
        np.ndarray
            Corresponding values in track space.
        """
        fcn = scipy.interpolate.interp1d(feature_ref, track_ref, fill_value='extrapolate')
        return fcn(feature_new)

    @staticmethod
    def track2feature(
        track_new: np.ndarray,
        feature_ref: np.ndarray,
        track_ref: np.ndarray,
    ) -> np.ndarray:
        """
        Convert track-space points to feature space via the reference-line fit.

        Builds a linear interpolant from the (track_ref, feature_ref) reference pairs
        and evaluates it, extrapolating beyond the outermost reference lines.

        Parameters
        ----------
        track_new : np.ndarray
            Points in track space to convert to feature space.
        feature_ref : np.ndarray
            Reference coordinates in feature space (ephys plots).
        track_ref : np.ndarray
            Reference coordinates in track space (histology track).

        Returns
        -------
        np.ndarray
            Corresponding values in feature space.
        """
        fcn = scipy.interpolate.interp1d(track_ref, feature_ref, fill_value='extrapolate')
        return fcn(track_new)

    @staticmethod
    def feature2track_lin(
        feature_new: np.ndarray,
        feature_ref: np.ndarray,
        track_ref: np.ndarray,
    ) -> np.ndarray | int:
        """
        Linear-fit version of feature2track, used for the extreme reference points.

        Fits a straight line to the interior (user-chosen) reference points and
        evaluates it. Only applied when there are at least three user reference lines
        (``feature_ref.size >= 5``, i.e. 3 lines plus the 2 extreme points); otherwise
        returns 0.

        Parameters
        ----------
        feature_new : np.ndarray
            Points in feature space to convert to track space.
        feature_ref : np.ndarray
            Reference coordinates in feature space (ephys plots).
        track_ref : np.ndarray
            Reference coordinates in track space (histology track).

        Returns
        -------
        np.ndarray or int
            Linear-fit values of `feature_new`, or 0 if there are too few reference points.
        """
        if feature_ref.size >= 5:
            fcn_lin = np.poly1d(np.polyfit(feature_ref[1:-1], track_ref[1:-1], 1))
            lin_fit = fcn_lin(feature_new)
        else:
            lin_fit = 0
        return lin_fit

    @staticmethod
    def adjust_extremes_uniform(feature: np.ndarray, track: np.ndarray) -> np.ndarray:
        """
        Adjust the outermost (non user-chosen) track points with a uniform shift.

        Shifts the first and last track points so that coordinates outside the
        user-picked span keep the same feature-to-track offset (no scaling).

        Parameters
        ----------
        feature : np.ndarray
            Reference coordinates in feature space (ephys plots).
        track : np.ndarray
            Reference coordinates in track space (histology track).

        Returns
        -------
        np.ndarray
            Track reference points with the first and last values adjusted.
        """
        diff = np.diff(feature - track)
        track[0] -= diff[0]
        track[-1] += diff[-1]
        return track

    def adjust_extremes_linear(
        self,
        feature: np.ndarray,
        track: np.ndarray,
        extend_feature: float = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Adjust the outermost reference points with a linear fit.

        Extends the first and last feature points by ``extend_feature`` and sets the
        matching track points from a linear fit, so coordinates outside the user-picked
        span are scaled linearly rather than left unchanged.

        Parameters
        ----------
        feature : np.ndarray
            Reference coordinates in feature space (ephys plots).
        track : np.ndarray
            Reference coordinates in track space (histology track).
        extend_feature : float
            Amount to extend the extreme feature coordinates before fitting.

        Returns
        -------
        feature : np.ndarray
            Feature reference points with the first and last values adjusted.
        track : np.ndarray
            Track reference points with the first and last values adjusted.
        """
        feature[0] = self.track_init[0] - extend_feature
        feature[-1] = self.track_init[-1] + extend_feature
        extend_track = self.feature2track_lin(feature[[0, -1]], feature, track)
        track[0] = extend_track[0]
        track[-1] = extend_track[-1]
        return feature, track

    def scale_histology_regions(
        self,
        feature: np.ndarray,
        track: np.ndarray,
        region: np.ndarray | None = None,
        region_label: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recompute histology region boundaries in feature space from the reference fit.

        Maps the stored track-space region boundaries (and their label positions) into
        feature space so they can be displayed against the ephys plots.

        Parameters
        ----------
        feature : np.ndarray
            Reference coordinates in feature space (ephys plots).
        track : np.ndarray
            Reference coordinates in track space (histology track).
        region : np.ndarray or None
            Region boundaries to scale. Defaults to ``self.region``.
        region_label : np.ndarray or None
            Label positions and acronyms to scale. Defaults to
            ``self.region_label``.

        Returns
        -------
        region : np.ndarray
            Region boundaries in feature space (um).
        region_label : np.ndarray
            Label positions (um) and acronyms.
        """
        region = np.copy(region) if region is not None else np.copy(self.region)
        region_label = (
            np.copy(region_label) if region_label is not None else np.copy(self.region_label)
        )
        region = self.track2feature(region, feature, track) * 1e6
        region_label[:, 0] = (
            self.track2feature(np.float64(region_label[:, 0]), feature, track) * 1e6
        )
        return region, region_label

    @staticmethod
    def get_histology_regions(
        xyz_coords: np.ndarray,
        depth_coords: np.ndarray,
        brain_atlas: BrainAtlas | None = None,
        mapping: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the brain regions and their boundaries along the depth of a probe or track.

        Looks up the atlas label at each sampled coordinate and groups consecutive
        samples with the same region id into contiguous boundaries.

        Parameters
        ----------
        xyz_coords : np.ndarray
            xyz coordinates of points along the probe or track.
        depth_coords : np.ndarray
            Depth along the probe or track for each xyz coordinate.
        brain_atlas : BrainAtlas or None
            Atlas used for the label lookup. Defaults to ``AllenAtlas(25)``.
        mapping : str or None
            Optional region mapping passed to the atlas label lookup.

        Returns
        -------
        region : np.ndarray
            Depth coordinates bounding each brain region.
        region_label : np.ndarray
            Label position (depth) and acronym for each region.
        region_colour : np.ndarray
            Allen atlas RGB colour for each region.
        region_id : np.ndarray
            Allen atlas id for each region.

        Raises
        ------
        ValueError
            If xyz_coords is empty, not an (n, 3) array, or contains non-finite values.
        """
        if not brain_atlas:
            brain_atlas = atlas.AllenAtlas(25)

        # Sanity-check the sampled coordinates before the atlas lookup
        if len(xyz_coords) == 0:
            raise ValueError('Empty coordinate array provided')
        if xyz_coords.ndim != 2 or xyz_coords.shape[1] != 3:
            raise ValueError(f'Coordinates must be Nx3 array, got shape {xyz_coords.shape}')
        if np.any(~np.isfinite(xyz_coords)) or np.any(~np.isfinite(depth_coords)):
            raise ValueError('Coordinates contain NaN or infinite values')

        region_ids = brain_atlas.get_labels(xyz_coords, mapping=mapping)
        region_info = brain_atlas.regions.get(region_ids)
        # Region changes occur where consecutive samples have different ids
        boundaries = np.where(np.diff(region_info.id))[0]
        region = np.empty((boundaries.size + 1, 2))
        region_label = np.empty((boundaries.size + 1, 2), dtype=object)
        region_id = np.empty((boundaries.size + 1, 1), dtype=int)
        region_colour = np.empty((boundaries.size + 1, 3), dtype=int)
        for bound in np.arange(boundaries.size + 1):
            if bound == 0:
                _region = np.array([0, boundaries[bound]])
            elif bound == boundaries.size:
                _region = np.array([boundaries[bound - 1], region_info.id.size - 1])
            else:
                _region = np.array([boundaries[bound - 1], boundaries[bound]])
            _region_colour = region_info.rgb[_region[1]]
            _region_label = region_info.acronym[_region[1]]
            _region_id = region_info.id[_region[1]]
            _region = depth_coords[_region]
            _region_mean = np.mean(_region)
            region[bound, :] = _region
            region_colour[bound, :] = _region_colour
            region_id[bound, :] = _region_id
            region_label[bound, :] = (_region_mean, _region_label)

        return region, region_label, region_colour, region_id

    @staticmethod
    def get_nearest_boundary(
        xyz_coords: np.ndarray,
        allen: pd.DataFrame,
        extent: float = 100,
        steps: int = 8,
        parent: bool = True,
        brain_atlas: BrainAtlas | None = None,
    ) -> dict:
        """
        Find the distance to the closest neighbouring brain region along the trajectory.

        For each point in xyz_coords, samples a plane through the point perpendicular to
        the trajectory and finds the nearest sampled point that lies in a different
        region, giving the distance to the nearest region boundary. Optionally repeats
        the calculation for the parent regions.

        Parameters
        ----------
        xyz_coords : np.ndarray
            xyz coordinates of points along the probe or track.
        allen : pd.DataFrame
            Allen structure tree, loaded from ``allen_structure_tree`` in iblatlas.
        extent : float
            Half-extent of the sampling plane in each direction from the point (um).
        steps : int
            Number of steps used to discretise the plane.
        parent : bool
            If True, also compute the nearest-boundary distance between parent regions.
        brain_atlas : BrainAtlas or None
            Atlas used for the label lookup. Defaults to ``AllenAtlas(25)``.

        Returns
        -------
        dict
            Nearest-boundary results, with keys ``dist``, ``id`` and ``col`` (and the
            ``parent_*`` equivalents when ``parent`` is True).
        """
        if not brain_atlas:
            brain_atlas = atlas.AllenAtlas(25)

        vector = atlas.Insertion.from_track(xyz_coords, brain_atlas=brain_atlas).trajectory.vector
        nearest_bound = dict()
        nearest_bound['dist'] = np.zeros(xyz_coords.shape[0])
        nearest_bound['id'] = np.zeros(xyz_coords.shape[0])
        # nearest_bound['adj_id'] = np.zeros((xyz_coords.shape[0]))
        nearest_bound['col'] = []

        if parent:
            nearest_bound['parent_dist'] = np.zeros(xyz_coords.shape[0])
            nearest_bound['parent_id'] = np.zeros(xyz_coords.shape[0])
            # nearest_bound['parent_adj_id'] = np.zeros((xyz_coords.shape[0]))
            nearest_bound['parent_col'] = []

        for iP, point in enumerate(xyz_coords):
            d = np.dot(vector, point)
            x_vals = np.r_[
                np.linspace(point[0] - extent / 1e6, point[0] + extent / 1e6, steps), point[0]
            ]
            y_vals = np.r_[
                np.linspace(point[1] - extent / 1e6, point[1] + extent / 1e6, steps), point[1]
            ]

            X, Y = np.meshgrid(x_vals, y_vals)
            Z = (d - vector[0] * X - vector[1] * Y) / vector[2]
            XYZ = np.c_[np.reshape(X, X.size), np.reshape(Y, Y.size), np.reshape(Z, Z.size)]
            dist = np.sqrt(np.sum((XYZ - point) ** 2, axis=1))

            try:
                brain_id = brain_atlas.regions.get(brain_atlas.get_labels(XYZ))['id']
            except Exception as err:
                logger.error(f'Failed to get brain region for boundary: {err}')
                continue

            dist_sorted = np.argsort(dist)
            brain_id_sorted = brain_id[dist_sorted]
            nearest_bound['id'][iP] = brain_id_sorted[0]
            nearest_bound['col'].append(
                allen['color_hex_triplet'][np.where(allen['id'] == brain_id_sorted[0])[0][0]]
            )
            bound_idx = np.where(brain_id_sorted != brain_id_sorted[0])[0]
            if np.any(bound_idx):
                nearest_bound['dist'][iP] = dist[dist_sorted[bound_idx[0]]] * 1e6
                # nearest_bound['adj_id'][iP] = brain_id_sorted[bound_idx[0]]
            else:
                nearest_bound['dist'][iP] = np.max(dist) * 1e6
                # nearest_bound['adj_id'][iP] = brain_id_sorted[0]

            if parent:
                # Now compute for the parents
                brain_parent = np.array(
                    [
                        allen['parent_structure_id'][np.where(allen['id'] == br)[0][0]]
                        for br in brain_id_sorted
                    ]
                )
                brain_parent[np.isnan(brain_parent)] = 0

                nearest_bound['parent_id'][iP] = brain_parent[0]
                nearest_bound['parent_col'].append(
                    allen['color_hex_triplet'][np.where(allen['id'] == brain_parent[0])[0][0]]
                )

                parent_idx = np.where(brain_parent != brain_parent[0])[0]
                if np.any(parent_idx):
                    nearest_bound['parent_dist'][iP] = dist[dist_sorted[parent_idx[0]]] * 1e6
                else:
                    nearest_bound['parent_dist'][iP] = np.max(dist) * 1e6

        return nearest_bound

    @staticmethod
    def arrange_into_regions(
        depth_coords: np.ndarray,
        region_ids: np.ndarray,
        distance: np.ndarray,
        region_colours: list[str],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
        """
        Reshape get_nearest_boundary output for plotting with pyqtgraph or matplotlib.

        Groups consecutive samples of the same region into per-region polylines of
        depth vs distance, padded so each region draws as a closed shape.

        Parameters
        ----------
        depth_coords : np.ndarray
            Depth along the probe or track for each point.
        region_ids : np.ndarray
            Brain region id at each depth.
        distance : np.ndarray
            Distance to the nearest boundary at each point.
        region_colours : list of str
            Allen atlas hex colour for each point's region.

        Returns
        -------
        all_x : list of np.ndarray
            Distance values for each region along the track.
        all_y : list of np.ndarray
            Depth values for each region along the track.
        all_colour : list of str
            Colour assigned to each region along the track.
        """
        boundaries = np.where(np.diff(region_ids))[0]
        bound = np.r_[0, boundaries + 1, region_ids.shape[0]]
        all_y = []
        all_x = []
        all_colour = []
        for iB in np.arange(len(bound) - 1):
            y = depth_coords[bound[iB] : (bound[iB + 1])]
            y = np.r_[y[0], y, y[-1]]
            x = distance[bound[iB] : (bound[iB + 1])]
            x = np.r_[0, x, 0]
            all_y.append(y)
            all_x.append(x)
            col = region_colours[bound[iB]]
            col = '#' + col if isinstance(col, str) else '#FFFFFF'
            all_colour.append(col)

        return all_x, all_y, all_colour

    def get_scale_factor(
        self,
        region: np.ndarray,
        region_orig: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find how much each brain region has been scaled by the alignment.

        Compares the scaled region boundaries against the original ones and groups
        consecutive regions that share the same scale factor.

        Parameters
        ----------
        region : np.ndarray
            Scaled histology boundaries.
        region_orig : np.ndarray or None
            Original histology boundaries. Defaults to ``self.region``.

        Returns
        -------
        scaled_region : np.ndarray
           Regions that share a common scale factor.
        scale_factor : np.ndarray
            Scale factor applied to each region.
        """
        region_orig = region_orig if region_orig is not None else self.region
        scale = []
        for reg, reg_orig in zip(region, region_orig * 1e6, strict=False):
            scale = np.r_[scale, (reg[1] - reg[0]) / (reg_orig[1] - reg_orig[0])]
        boundaries = np.where(np.diff(np.around(scale, 3)))[0]
        if boundaries.size == 0:
            scaled_region = np.array([[region[0][0], region[-1][1]]])
            scale_factor = np.unique(scale)
        else:
            scaled_region = np.empty((boundaries.size + 1, 2))
            scale_factor = []
            for bound in np.arange(boundaries.size + 1):
                if bound == 0:
                    _scaled_region = np.array([region[0][0], region[boundaries[bound]][1]])
                    _scale_factor = scale[0]
                elif bound == boundaries.size:
                    _scaled_region = np.array([region[boundaries[bound - 1]][1], region[-1][1]])
                    _scale_factor = scale[-1]
                else:
                    _scaled_region = np.array(
                        [region[boundaries[bound - 1]][1], region[boundaries[bound]][1]]
                    )
                    _scale_factor = scale[boundaries[bound]]
                scaled_region[bound, :] = _scaled_region
                scale_factor = np.r_[scale_factor, _scale_factor]
        return scaled_region, scale_factor

    def get_channel_locations(
        self,
        feature: np.ndarray,
        track: np.ndarray,
        depths: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Get the 3D xyz coordinates of points along the ephys feature axis.

        Two interpolation steps:

        1. feature space -> track space, using the reference-line fit.
        2. track space -> 3D xyz (RAS) coordinates, by interpolating along xyz_track.

        Parameters
        ----------
        feature : np.ndarray
            Reference coordinates in feature space (ephys plots).
        track : np.ndarray
            Reference coordinates in track space (histology track).
        depths : np.ndarray or None
            Feature-space depths to locate (m). Defaults to the channel depths.

        Returns
        -------
        np.ndarray
            xyz coordinates for each depth.
        """
        if depths is None:
            depths = self.chn_depths / 1e6
        # nb using scipy here so we can change to cubic spline if needed
        channel_depths_track = self.feature2track(depths, feature, track) - self.track_extent[0]
        xyz_channels = interpolate_along_track(self.xyz_track, channel_depths_track)
        return xyz_channels

    def get_tip_location(self, feature: np.ndarray, track: np.ndarray) -> np.ndarray:
        """
        Get the 3D xyz coordinates of the probe tip.

        The tip is ``TIP_SIZE_UM`` below the first electrode. Uses the same
        feature-to-track transform as the channels, so the tip position updates
        dynamically as the alignment is adjusted.

        Parameters
        ----------
        feature : np.ndarray
            Reference coordinates in feature space (ephys plots).
        track : np.ndarray
            Reference coordinates in track space (histology track).

        Returns
        -------
        np.ndarray
            xyz coordinates of the tip.
        """
        tip_depth = np.array([-TIP_SIZE_UM / 1e6])
        tip_depth_track = self.feature2track(tip_depth, feature, track) - self.track_extent[0]
        xyz_tip = interpolate_along_track(self.xyz_track, tip_depth_track)
        return xyz_tip[0]

    def get_brain_locations(self, xyz_channels: np.ndarray) -> Bunch:
        """
        Find the brain regions at a set of 3D electrode locations.

        Parameters
        ----------
        xyz_channels : np.ndarray
            xyz coordinates of the electrodes.

        Returns
        -------
        Bunch
            Brain region information for each electrode.
        """
        brain_regions = self.brain_atlas.regions.get(self.brain_atlas.get_labels(xyz_channels))
        return brain_regions

    def get_perp_vector(self, feature: np.ndarray, track: np.ndarray) -> list[np.ndarray]:
        """
        Find the lines perpendicular to the trajectory at each reference line.

        For each user reference line, computes a short segment perpendicular to the
        local trajectory direction, used to draw the slice location.

        Parameters
        ----------
        feature : np.ndarray
            Reference coordinates in feature space (ephys plots).
        track : np.ndarray
            Reference coordinates in track space (histology track).

        Returns
        -------
        list of np.ndarray
            Array of endpoint xyz coordinates per reference line.
        """
        slice_lines = []
        for line in feature[1:-1]:
            depths = np.array([line, line + 10 / 1e6])
            xyz = self.get_channel_locations(feature, track, depths)

            extent = 500e-6
            vector = np.diff(xyz, axis=0)[0]
            point = xyz[0, :]
            vector_perp = np.array([1, 0, -1 * vector[0] / vector[2]])
            xyz_per = np.r_[
                [point + (-1 * extent * vector_perp)], [point + (extent * vector_perp)]
            ]
            slice_lines.append(xyz_per)

        return slice_lines
