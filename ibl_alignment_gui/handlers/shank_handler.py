import numpy as np

from ibl_alignment_gui.handlers.alignment_handler import AlignmentHandler
from ibl_alignment_gui.loaders.plot_loader import (
    ImageData,
    LineData,
    ProbeData,
    ScatterData,
)
from iblutil.util import Bunch


class ShankHandler:
    """
    Model for handling data on a shank of a probe for a given recording configuration.

    Parameters
    ----------
    loaders: Bunch
        A Bunch object containing all the relevant loaders and uploaders to read and write data.
    shank_idx: int
        The index of the shank in the probe

    """

    def __init__(self, loaders: Bunch, shank_idx: int):

        self.shank_idx: int = shank_idx
        self.loaders: Bunch = loaders
        self.loaders['align'].load_previous_alignments()
        self.loaders['align'].get_starting_alignment(0)
        self.align_exists: bool = True
        self.data_loaded: bool = False

    # -------------------------------------------------------------------------
    # Alignment loader - attributes and methods in loaders['align']
    # -------------------------------------------------------------------------
    def set_init_alignment(self) -> None:
        """Set the initial alignment based on previous features and tracks."""
        self.align_handle.set_init_feature_track(self.loaders['align'].feature_prev,
                                                 self.loaders['align'].track_prev)

    @property
    def feature_prev(self) -> np.ndarray:
        """
        Return the previous feature from the alignment loader for the currently active shank.

        Returns
        -------
        np.ndarray
            Previous feature array.
        """
        return self.loaders['align'].feature_prev

    # -------------------------------------------------------------------------
    # Alignment handler - attributes and methods in align_handle
    # -------------------------------------------------------------------------
    def offset_hist_data(self, *args) -> None:
        """See :meth:`AlignmentHandler.offset_hist_data` for details."""
        self.align_handle.offset_hist_data(*args)

    def scale_hist_data(self, *args, **kwargs) -> None:
        """See :meth:`AlignmentHandler.scale_hist_data` for details."""
        self.align_handle.scale_hist_data(*args, **kwargs)

    def get_scaled_histology(self) -> None:
        """See :meth:`AlignmentHandler.get_scaled_histology` for details."""
        self.hist_data, self.hist_data_ref, self.scale_data = (
            self.align_handle.get_scaled_histology())

    def feature2track_lin(
            self,
            depths: np.ndarray,
            feature: np.ndarray,
            track: np.ndarray
    ) -> np.ndarray:
        """
        Estimate values of depth according to linear fit between feature and track reference lines.

        Parameters
        ----------
        depths: np.ndarray
            The depths to estimate the new depths for
        feature: np.ndarray
            The feature line positions
        track: np.ndarray
            The track line positions

        Returns
        -------
        np.ndarray
            The new estimated depths
        """
        return self.align_handle.ephysalign.feature2track_lin(depths, feature, track)

    def reset_features_and_tracks(self) -> None:
        """See :meth:`AlignmentHandler.reset_features_and_tracks` for details."""
        self.align_handle.reset_features_and_tracks()

    @property
    def track(self) -> np.ndarray:
        """See :meth:`AlignmentHandler.track` for details."""
        return self.align_handle.track

    @property
    def feature(self) -> np.ndarray:
        """See :meth:`AlignmentHandler.feature` for details."""
        return self.align_handle.feature

    @property
    def xyz_channels(self) -> np.ndarray:
        """See :meth:`AlignmentHandler.xyz_channels` for details."""
        return self.align_handle.xyz_channels

    @property
    def xyz_track(self) -> np.ndarray:
        """See :meth:`AlignmentHandler.xyz_track` for details."""
        return self.align_handle.xyz_track

    @property
    def track_lines(self) -> list[np.ndarray]:
        """See :meth:`AlignmentHandler.track_lines` for details."""
        return self.align_handle.track_lines

    # -------------------------------------------------------------------------
    # Plot properties - methods and attributes in loaders['plots']
    # -------------------------------------------------------------------------
    @property
    def chn_min(self) -> float:
        """
        Return the minimum y channel value for the currently active shank.

        Returns
        -------
        float:
            The minimum channel value, or 0 if the minimum is positive.
        """
        return np.min([0, self.loaders['plots'].chn_min])

    @property
    def chn_max(self) -> float:
        """
        Return the maximum y channel value for the currently active shank.

        Returns
        -------
        float:
            The maximum channel value
        """
        return self.loaders['plots'].chn_max

    @property
    def y_min(self) -> float:
        """
        Return the minimum y channel value.

        Returns
        -------
        float:
            The minimum channel value
        """
        return self.loaders['plots'].chn_min

    @property
    def y_max(self) -> float:
        """
        Return the maximum y channel value.

        Returns
        -------
        float:
            The maximum channel value
        """
        return self.loaders['plots'].chn_max

    @property
    def image_plots(self) -> Bunch[str, ImageData]:
        """
        Access the image plots for the currently active shank.

        Returns
        -------
        Bunch:
            A bunch of available slice plots.
        """
        return self.loaders['plots'].image_plots

    @property
    def scatter_plots(self) -> Bunch[str, ScatterData]:
        """
        Access the scatter plots for the currently active shank.

        Returns
        -------
        Bunch:
            A bunch of available slice plots.
        """
        return self.loaders['plots'].scatter_plots

    @property
    def line_plots(self) -> Bunch[str, LineData]:
        """
        Access the slice plots for the currently active shank.

        Returns
        -------
        Bunch:
            A bunch of available slice plots.
        """
        return self.loaders['plots'].line_plots

    @property
    def probe_plots(self) -> Bunch[str, ProbeData]:
        """
        Access the probe plots for the currently active shank.

        Returns
        -------
        Bunch:
            A bunch of available probe plots.
        """
        return self.loaders['plots'].probe_plots

    @property
    def slice_plots(self) -> Bunch[str, Bunch]:
        """
        Access the slice plots from the current shank's plot loader.

        Returns
        -------
        Bunch:
            A bunch of available slice plots.
        """
        return self.loaders['plots'].slice_plots

    @property
    def feature_plots(self) -> Bunch[str, Bunch]:
        """
        Access the feature plots from the current shank's plot loader.

        Returns
        -------
        Bunch:
            A bunch of available feature plots.
        """
        return self.loaders['plots'].feature_plots

    def reset_levels(self) -> None:
        """Reset the levels for all image, scatter, line and probe plots."""
        for plot in [self.image_plots, self.scatter_plots, self.line_plots, self.probe_plots]:
            for _, data in plot.items():
                data.levels = np.copy(data.default_levels)

    # -------------------------------------------------------------------------
    # Methods of current class
    # -------------------------------------------------------------------------

    @property
    def xyz_clusters(self) -> np.ndarray:
        """
        Return the xyz cluster locations estimated using the fit from the track and feature lines.

        The values in the current index in the circular buffer for the currently active shank
        are used.

        Returns
        -------
        np.ndarray
            xyz positions of clusters in 3D space
        """
        clust = self.raw_data['clusters']['channels'][self.loaders['plots'].cluster_idx]
        return self.xyz_channels[clust]

    def load_data(self) -> None:
        """Load the geometry, ephys and alignment data."""
        if self.data_loaded:
            return

        # Load the geometry data
        self.loaders['geom'].get_geometry()
        shank_sites = self.loaders['geom'].get_sites_for_shank(self.shank_idx)
        self.chn_sites = self.loaders['geom'].get_sites_for_shank(self.shank_idx, sites='channels')

        # Load in the spike sorting and ephys data
        self.raw_data = self.loaders['data'].get_data(shank_sites)

        # Load in the raw data snippets
        self.raw_data['raw_snippets'] = self.loaders['ephys'].load_ap_snippets()

        # Load in the features data
        if self.loaders.get('features', None) is not None:
            self.raw_data['features'] = self.loaders['features'].load_features()
        else:
            self.raw_data['features'] = Bunch(exists=False)

        # Create the plot data using the raw data
        self.loaders['plots'].get_data(self.raw_data, shank_sites)

        # These are the locations of the channels and clusters from spikesorting on the probe
        self.chn_coords = self.chn_sites['sites_coords']
        self.chn_depths = self.chn_coords[:, 1]
        if self.raw_data['clusters']['exists']:
            self.cluster_chns = self.raw_data['clusters']['channels']
        elif self.chn_depths is not None:
            self.cluster_chns = np.arange(self.chn_depths.size)

        if self.chn_coords is not None and self.loaders['align'].xyz_picks is not None:
            # Load the alignment handler
            self.align_handle = AlignmentHandler(
                self.loaders['align'].xyz_picks,
                self.chn_depths,
                self.loaders['upload'].brain_atlas)

            self.set_init_alignment()
            # Load in the histology data
            self.loaders['plots'].slice_plots = (
                self.loaders['hist'].get_slices(self.align_handle.xyz_samples))
        else:
            self.align_exists = False
            self.loaders['plots'].slice_plots = Bunch()

        self.data_loaded = True

    def load_plots(self):
        """Load all the plot data for the current shank."""
        self.loaders['plots'].get_plots()

    def filter_units(self, filter_type: str) -> None:
        """
        Filter the spikesorting data by selected unit type and recompute plot data.

        Parameters
        ----------
        filter_type: str
            The type of unit to filter by
        """
        self.loaders['plots'].filter_units(filter_type)
        self.loaders['plots'].compute_rasters()
        self.loaders['plots'].get_plots()

    def upload_data(self) -> str:
        """Upload the data, save the channels and the alignments."""
        data = {'chn_coords': self.chn_coords,
                'xyz_channels': self.align_handle.xyz_channels,
                'feature': self.align_handle.feature.tolist(),
                'track': self.align_handle.track.tolist(),
                'alignments': self.loaders['align'].alignments,
                'cluster_chns': self.cluster_chns,
                'probe_collection': self.loaders['data'].probe_collection,
                'probe_path': self.loaders['data'].probe_path,
                'chn_depths': self.chn_depths,
                'xyz_picks': self.loaders['align'].xyz_picks,
                }
        return self.loaders['upload'].upload_data(data, shank_sites=self.chn_sites)
