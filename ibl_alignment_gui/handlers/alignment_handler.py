import numpy as np

from iblatlas.atlas import AllenAtlas
from ibllib.pipes.ephys_alignment import EphysAlignment
from iblutil.util import Bunch


class CircularIndexTracker:
    """
    A class to manage circular buffer indexing.

    Parameters
    ----------
    max_idx: int
        Size of the circular buffer.

    Attributes
    ----------
    max_idx: int
        Size of the circular buffer.
    current_idx : int
        The current index in logical (non-wrapped) space.
    total_idx : int
        The highest index filled so far in logical space.
    last_idx : int
        The last recorded total index.
    diff_idx : int
        Offset between current index and last index used in reset logic.
    idx : int
        The wrapped index used in the circular buffer.
    idx_prev : int
        The previous wrapped index.
    """

    def __init__(self, max_idx: int):

        self.max_idx: int = max_idx
        self.current_idx: int = 0
        self.total_idx: int = 0
        self.last_idx: int = 0
        self.diff_idx: int = 0
        self.idx: int = 0
        self.idx_prev: int = 0

    def _update_diff_idx(self) -> None:
        """Update the diff_idx value."""
        if self.current_idx < self.last_idx:
            self.total_idx = self.current_idx
            delta = np.mod(self.last_idx, self.max_idx) - np.mod(self.total_idx, self.max_idx)
            self.diff_idx = self.max_idx - delta if delta >= 0 else np.abs(delta)
        else:
            self.diff_idx = self.max_idx - 1

    def next_idx_to_fill(self) -> None:
        """
        Advance to the next index.

        If the next index doesn't exist create it.
        """
        self._update_diff_idx()
        self.total_idx += 1
        self.current_idx += 1
        self.idx_prev = self.idx
        self.idx = np.mod(self.current_idx, self.max_idx)

    def prev_idx(self) -> bool:
        """
        Move to the previous index in the buffer.

        If no previous index is available, remain at current index.

        Returns: bool
            Whether the tracker moved to the previous index.
        """
        self.last_idx = max(self.last_idx, self.total_idx)

        if self.current_idx > np.max([0, self.total_idx - self.diff_idx]):
            self.current_idx -= 1
            self.idx = np.mod(self.current_idx, self.max_idx)

            return True

    def next_idx(self) -> bool:
        """
        Move to the next index in the buffer.

        If no further index is available, remain at current index.

        Returns: bool
            Whether the tracker moved to the next index.
        """
        if ((self.current_idx < self.total_idx) &
                (self.current_idx > self.total_idx - self.max_idx)):
            self.current_idx += 1
            self.idx = np.mod(self.current_idx, self.max_idx)

            return True

    def reset_idx(self) -> None:
        """Reset the index."""
        self._update_diff_idx()
        self.total_idx += 1
        self.current_idx += 1
        self.idx = np.mod(self.current_idx, self.max_idx)


class AlignmentHandler:
    """
    Handles the alignment of electrophysiology data to histology data.

    The location of the electrodes along the probe trajectory are adjusted according user defined
    reference lines that are placed on the electrophysiology (feature) and histology (track) data.
    Uses a circular buffer to keep a history of the alignment steps the user performs to adjust
    the locations of the electrodes in the brain to match the observed features in th ephys data.

    Parameters
    ----------
    xyz_picks : np.ndarray
        An array of xyz coordinates in 3D space that define the trajectory of the probe through the
        brain. The most ventral point defines the initial estimate of the probe tip.
    chn_depths : np.ndarray
        An array containing the depths of the recording channels along the probe.
    brain_atlas : AllenAtlas
        An AllenAtlas object containing a volume to do a lookup between xyz coordinates and
        brain region.

    Attributes
    ----------
    buffer : CircularIndexTracker
        Circular buffer to store and manage multiple alignments steps.
    brain_atlas : AllenAtlas
        An AllenAtlas object
    ephysalign : EphysAlignment
        An EphysAlignment object used to perform alignment logic
    hist_mapping : str
        Defines histology mapping mode, e.g. 'Allen'
    tracks : list
        A list of arrays, each containing the position of the track reference lines along the
        probe at a specific alignment step
    features : list
        A list of arrays, each containing the position of the feature reference lines along the
        probe at a specific alignment step
    """

    def __init__(self, xyz_picks: np.ndarray, chn_depths: np.ndarray, brain_atlas: AllenAtlas):

        self.buffer: CircularIndexTracker = CircularIndexTracker(10)
        self.brain_atlas: AllenAtlas = brain_atlas
        self.ephysalign: EphysAlignment = EphysAlignment(xyz_picks, chn_depths,
                                                         brain_atlas=self.brain_atlas)
        self.tracks: list = [0] * (self.buffer.max_idx + 1)
        self.features: list = [0] * (self.buffer.max_idx + 1)
        self.hist_mapping: str = 'Allen'

    @property
    def xyz_track(self) -> np.ndarray:
        """
        Return the xyz coordinates along the probe trajectory.

        The coordinates are extended to the top and bottom of the brain surface.

        Returns
        -------
        np.ndarray
            xyz positions of trajctory in 3D space
        """
        return self.ephysalign.xyz_track

    @property
    def xyz_samples(self) -> np.ndarray:
        """
        Return the xyz coordinates along the probe trajectory.

        The coordinates are extended to the full extent of the Atlas volume sampled at
        10 um intervals.

        Returns
        -------
        np.ndarray
            xyz positions of samples in 3D space
        """
        return self.ephysalign.xyz_samples

    @property
    def xyz_channels(self) -> np.ndarray:
        """
        Return xyz channel locations estimated using the fit from the track and feature arrays.

        Estimates using the values stored at the current index of the circular buffer.

        Returns
        -------
        np.ndarray
            xyz positions of channels in 3D space
        """
        return self.ephysalign.get_channel_locations(self.features[self.idx],
                                                     self.tracks[self.idx])

    @property
    def track_lines(self) -> list[np.ndarray]:
        """
        Return the perpendicular vectors (lines) at the position of each track reference line.

        Estimates using the values stored at the current index of the circular buffer.

        Returns
        -------
        list of np.ndarray
            List of arrays containing points defining perpendicular vector at each track
            reference line
        """
        return self.ephysalign.get_perp_vector(self.features[self.idx], self.tracks[self.idx])

    @property
    def track(self):
        """
        Track array at the current index of the circular buffer.

        Returns
        -------
        np.ndarray
            An array of positions of the track reference lines for the current index
        """
        return self.tracks[self.idx]

    @property
    def feature(self) -> np.ndarray:
        """
        Feature array at the current index of the circular buffer.

        Returns
        -------
        np.ndarray
            An array of positions of the feature reference lines for the current index
        """
        return self.features[self.idx]

    @property
    def idx(self) -> int:
        """
        The current index in the circular buffer.

        Returns
        -------
        int
            The current index
        """
        return self.buffer.idx

    @property
    def idx_prev(self) -> int:
        """
        The previous index in the circular buffer.

        Returns
        -------
        int
            The previous index
        """
        return self.buffer.idx_prev

    @property
    def current_idx(self):
        """See :meth:`CircularIndexTracker.current_idx` for details."""
        return self.buffer.current_idx

    @property
    def total_idx(self) -> int:
        """See :meth:`CircularIndexTracker.total_idx` for details."""
        return self.buffer.total_idx

    def next_idx(self) -> bool:
        """See :meth:`CircularIndexTracker.next_idx` for details."""
        return self.buffer.next_idx()

    def prev_idx(self) -> bool:
        """See :meth:`CircularIndexTracker.prev_idx` for details."""
        return self.buffer.prev_idx()

    def set_init_feature_track(
            self,
            feature: np.ndarray | None = None,
            track: np.ndarray | None = None
    ) -> None:
        """
        Set the initial feature and track values for the current buffer index.

        Parameters
        ----------
        feature : np.ndarray, optional
            Initial feature alignment.
        track : np.ndarray, optional
            Initial track alignment.
        """
        if feature is not None:
            self.ephysalign.feature_init = feature
        if track is not None:
            self.ephysalign.track_init = track
        self.features[self.idx], self.tracks[self.idx], _ = (
            self.ephysalign.get_track_and_feature())

    def reset_features_and_tracks(self) -> None:
        """Reset features and tracks to their initial alignment state."""
        self.buffer.reset_idx()
        self.tracks[self.idx] = self.ephysalign.track_init
        self.features[self.idx] = self.ephysalign.feature_init

    def get_scaled_histology(self) -> tuple[Bunch, Bunch, Bunch]:
        """
        Compute the brain regions along the probe track using the current alignment.

        Returns
        -------
        hist_data : Bunch
            Scaled histology regions and axis labels for the current track.
        hist_data_ref : Bunch
            Reference histology data for comparison.
        scale_data : Bunch
            Scaling factors applied to the histology regions.
        """
        hist_data = Bunch()
        scale_data = Bunch()
        hist_data_ref = Bunch()

        region_label = None
        region = None
        colour = self.ephysalign.region_colour

        hist_data['region'], hist_data['axis_label'] = (
            self.ephysalign.scale_histology_regions(
                self.features[self.idx], self.tracks[self.idx],
                region=region, region_label=region_label))
        hist_data['colour'] = colour

        scale_data['region'], scale_data['scale'] = (
            self.ephysalign.get_scale_factor(hist_data['region'], region_orig=region))
        hist_data_ref['region'], hist_data_ref['axis_label'] = (
            self.ephysalign.scale_histology_regions(self.ephysalign.track_extent,
                                                    self.ephysalign.track_extent,
                                                    region=region, region_label=region_label))

        hist_data_ref['colour'] = colour

        return hist_data, hist_data_ref, scale_data

    def offset_hist_data(self, offset: float) -> None:
        """
        Apply an offset to the brain regions along the probe track.

        Adds the new alignment state into next buffer index of the feature and track arrays.

        Parameters
        ----------
        offset : float
            Offset value to apply to the track alignment.
        """
        self.buffer.next_idx_to_fill()
        self.tracks[self.idx] = (self.tracks[self.idx_prev] + offset)
        self.features[self.idx] = (self.features[self.idx_prev])

    def scale_hist_data(
            self,
            line_track: np.ndarray,
            line_feature: np.ndarray,
            extend_feature: int = 1,
            lin_fit: bool = True
    ) -> None:
        """
        Scale brain regions along the probe track.

        Scales based on location of the user chosen track and feature reference lines.
        Adds the new alignment state into next buffer index of the feature and track arrays.

        Parameters
        ----------
        line_track : np.ndarray
            An array of positions of the track reference lines
        line_feature : np.ndarray
            An array of positions of the feature reference lines
        extend_feature : int, optional
            Factor for extending the feature alignment beyond original extremes.
        lin_fit : bool, optional
            Whether to apply linear fitting to adjust extremes. Only applied when number of
            fit lines >= 5
        """
        self.buffer.next_idx_to_fill()

        depths_track = np.sort(np.r_[self.tracks[self.idx_prev][[0, -1]], line_track])

        self.tracks[self.idx] = self.ephysalign.feature2track(depths_track,
                                                              self.features[self.idx_prev],
                                                              self.tracks[self.idx_prev])
        self.features[self.idx] = np.sort(
            np.r_[self.features[self.idx_prev][[0, -1]], line_feature])

        if (self.features[self.idx].size >= 5) & lin_fit:
            self.features[self.idx], self.tracks[self.idx] = (
                self.ephysalign.adjust_extremes_linear(
                    self.features[self.idx], self.tracks[self.idx], extend_feature))
        else:
            self.tracks[self.idx] = self.ephysalign.adjust_extremes_uniform(
                self.features[self.idx], self.tracks[self.idx])
