import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
from matplotlib import cm, colors

from brainbox.task import passive
from ibl_alignment_gui.loaders.geometry_loader import (
    arrange_channels_into_banks,
    average_chns_at_same_depths,
    pad_data_to_full_chn_map,
)
from iblutil.numerical import bincount2D
from iblutil.util import Bunch

logger = logging.getLogger(__name__)

np.seterr(divide='ignore', invalid='ignore')


def skip_missing(required_keys):
    """Skip method execution if required data keys are missing or false."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for key in required_keys:
                val = self.data[key]['exists']
                if not val:
                    return {}
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


@dataclass
class ScatterData:
    """
    Data structure for 2D scatter plots.

    Attributes
    ----------
    x : np.ndarray
        x-coordinates of points.
    y : np.ndarray
        y-coordinates of points.
    levels : list or np.ndarray
        Levels for colormap scaling. These can be updated by the user
    default_levels : list or np.ndarray
        Default levels for colormap scaling.
    colours : np.ndarray
        Hex colour or data values for each point.
    pen : string or None
        Colour for the outline marker of each point
    size : np.ndarray
        Size of each point.
    symbol : str or np.ndarray
        Marker symbol(s) for each point.
    xrange : np.ndarray
        Range of the x-axis.
    xaxis : str
        Label for the x-axis.
    title : str
        Plot title.
    cmap : str
        Colormap name for coloring points.
    cluster : bool
        Whether data is cluster data.
    """

    x: np.ndarray
    y: np.ndarray
    levels: list | np.ndarray
    default_levels: list | np.ndarray
    colours: np.ndarray
    pen: str | None
    size: np.ndarray
    symbol: str | np.ndarray
    xrange: np.ndarray
    xaxis: str
    title: str
    cmap: str
    cluster: bool


@dataclass
class ImageData:
    """
    Data structure for 2D image plots.

    Attributes
    ----------
    img : np.ndarray
        2D array representing image values.
    scale : np.ndarray
        Scaling factors for axes (x and y).
    levels : list or np.ndarray
        Levels for colormap scaling. These can be updated by the user
    default_levels : list or np.ndarray
        Default levels for colormap scaling.
    offset : np.ndarray
        Offset for axes (x and y).
    xrange : np.ndarray
        Range of the x-axis.
    xaxis : str
        Label for the x-axis.
    cmap : str
        Colormap name.
    title : str
        Plot title.
    """

    img: np.ndarray
    scale: np.ndarray
    levels: np.ndarray
    default_levels: list | np.ndarray
    offset: np.ndarray
    xrange: np.ndarray
    xaxis: str
    cmap: str
    title: str


@dataclass
class LineData:
    """
    Data structure for line plots.

    Attributes
    ----------
    x : np.ndarray
        x-coordinates of the line.
    y : np.ndarray
        y-coordinates of the line.
    levels : list or np.ndarray
        Levels for colormap scaling. These can be updated by the user
    default_levels : list or np.ndarray
        Default levels for colormap scaling.
    xrange : np.ndarray
        Range of the x-axis.
    xaxis : str
        Label for the x-axis.
    """

    x: np.ndarray
    y: np.ndarray
    levels: np.ndarray
    default_levels: list | np.ndarray
    xrange: np.ndarray
    xaxis: str


@dataclass
class ProbeData:
    """
    Data structure for probe plots.

    Attributes
    ----------
    img : list of np.ndarray
        List of 2D arrays representing images for each probe bank.
    idx : list of np.ndarray
        List of channel indices for each bank.
    scale : list of np.ndarray
        List of scaling factors for each bank (x, y).
    levels : list or np.ndarray
        Levels for colormap scaling. These can be updated by the user
    default_levels : list or np.ndarray
        Default levels for colormap scaling.
    offset : list of np.ndarray
        List of offsets (x, y) for each bank.
    xrange : np.ndarray
        Range of the x-axis.
    cmap : str
        Colormap name.
    title : str
        Plot title.
    boundaries : np.ndarray or None
        Array of boundaries for banks or regions.
    """

    img: list[np.ndarray]
    idx: list[np.ndarray]
    scale: list[np.ndarray]
    levels: list | np.ndarray
    default_levels: list | np.ndarray
    offset: list[np.ndarray]
    xrange: np.ndarray
    cmap: str
    title: str
    boundaries: np.ndarray | None = None


FILTER_MATCH = {
    'IBL good': ('label', 1),
    'KS good': ('ks2_label', 'good'),
    'KS mua': ('ks2_label', 'mua'),
}

TBIN = 0.05
DBIN = 5
BNK_SIZE = 10


def compute_spike_average(
        spikes: Bunch[str, Any],
        clusters: Bunch[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute average spike amplitudes, depths, and firing rates for each cluster.

    Parameters
    ----------
    spikes : Bunch
        Spike data containing 'amps', 'depths', 'times', 'clusters'.
    clusters : Bunch
        Cluster data containing 'channels' and 'metrics'.

    Returns
    -------
    clust_idx : np.ndarray
        Array of cluster indices.
    avg_amps : np.ndarray
        Average spike amplitude per cluster (uV).
    avg_depths : np.ndarray
        Average depth per cluster.
    avg_fr : np.ndarray
        Average firing rate per cluster (spikes/sec).

    Notes
    -----
    - Clusters with no spikes are returned as NaN.
    """
    # Remove exists key for pandas operation
    exists = spikes.pop('exists')
    spike_df = spikes.to_df().groupby('clusters')
    avgs = spike_df.agg(['mean', 'count'])
    # Add back in for use elsewhere
    spikes['exists'] = exists

    # Some clusters don't have any spikes so we need to reindex into the original clusters data
    idx = avgs.index.values
    clust_idx = np.arange(clusters['channels'].size)

    avg_amps = np.full(clust_idx.size, np.nan)
    avg_amps[idx] = avgs['amps']['mean'].values * 1e6

    avg_fr = np.full(clust_idx.size, np.nan)
    avg_fr[idx] = avgs['depths']['count'].values / spikes['times'].max()

    avg_depths = np.full(clust_idx.size, np.nan)
    avg_depths[idx] = avgs['depths']['mean'].values

    return clust_idx, avg_amps, avg_depths, avg_fr


def compute_bincount(
        spike_times: np.ndarray,
        spike_depths: np.ndarray,
        spike_amps: np.ndarray,
        xbin: float = TBIN,
        ybin: float = DBIN,
        **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D binned spike count and amplitude over time and depth.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times.
    spike_depths : np.ndarray
        Depths of spikes.
    spike_amps : np.ndarray
        Amplitudes of spikes.
    xbin : float
        Bin width along the x-axis (time).
    ybin : float
        Bin width along the y-axis (depth).
    **kwargs :
        Additional arguments for `bincount2D`.

    Returns
    -------
    count : np.ndarray
        2D binned spike counts.
    amp : np.ndarray
        2D binned spike amplitudes.
    times : np.ndarray
        Bin edges for x-axis (time).
    depths : np.ndarray
        Bin edges for y-axis (depth).
    """
    count, times, depths = bincount2D(spike_times, spike_depths, xbin=xbin, ybin=ybin, **kwargs)

    amp, times, depths = bincount2D(spike_times, spike_depths, xbin=xbin, ybin=ybin,
                                    weights=spike_amps, **kwargs)

    return count, amp, times, depths


def group_bincount(arr: np.ndarray, group_size: int, axis: int = 1) -> np.ndarray:
    """
    Average over chunks of `group_size` along the given axis.

    If leftover elements exist, sum them and append as the final group.

    Parameters
    ----------
    arr : np.ndarray
        2D array to process.
    group_size : int
        Number of elements per group to average.
    axis : int
        Axis to operate on: 0 (rows) or 1 (columns). Default is 1.

    Returns
    -------
    np.ndarray
        Array with grouped means and a final summed group if leftovers exist.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 or 1.")

    # Transpose if operating on axis 0 to reuse logic
    if axis == 0:
        arr = arr.T

    num_elements = arr.shape[1]
    num_full = num_elements // group_size
    full_cols = num_full * group_size

    arr_full = arr[:, :full_cols]
    arr_extra = arr[:, full_cols:]

    # Compute mean over full groups
    arr_grouped = arr_full.reshape(arr.shape[0], num_full, group_size)
    arr_avg = arr_grouped.sum(axis=2)

    # Sum the leftover group (if any)
    if arr_extra.shape[1] > 0:
        arr_sum = arr_extra.sum(axis=1, keepdims=True)
        result = np.concatenate([arr_avg, arr_sum], axis=1)
    else:
        result = arr_avg

    return result.T if axis == 0 else result


class PlotLoader:
    """Class for handling plot data generation."""

    def __init__(self):

        self.data: Bunch | None = None
        self.shank_sites: Bunch | None = None
        self.chn_min: float | None = None
        self.chn_max: float | None = None
        self.image_plots: Bunch | None = None
        self.probe_plots: Bunch | None = None
        self.line_plots: Bunch | None = None
        self.scatter_plots: Bunch | None = None

    # --------------------------------------------------------------------------------------------
    # Main entry point to get all plots
    # --------------------------------------------------------------------------------------------

    def get_data(self, data: Bunch[str, Any], shank_sites: Bunch[str, Any]):
        """
        Get all plot data.

        Parameters
        ----------
        data: Bunch
            A bunch containing all the spikes and ephys data required to generate plots
        shank_sites: Bunch
            A bunch containing electrode geometry information for given shank
        """
        self.data = data
        self.shank_sites = shank_sites

        self.chn_min = self.shank_sites['sites_min']
        self.chn_max = self.shank_sites['sites_max']

        self.filter_units('All')
        self.compute_avg_cluster_activity()
        self.compute_rasters()
        self.get_plots()

    def get_plots(self):
        """
        Get all plot data for the different plot types.

        Notes
        -----
        This method sets the following attributes:

        self.image_plots : Bunch
            All plots of type image
        self.scatter_plots : Bunch
            All plots of type scatter
        self.line_plots : Bunch
            All plots of type line
        self.probe_plots : Bunch
            All plots of type probe
        """
        self.image_plots = self._get_plots('image')
        self.scatter_plots = self._get_plots('scatter')
        self.line_plots = self._get_plots('line')
        self.probe_plots = self._get_plots('probe')

    def _get_plots(self, plot_prefix: str) -> Bunch[str, Any]:
        """
        Find and call all methods that begin with given `plot_prefix`.

        Parameters
        ----------
        plot_prefix : str
            Prefix for plot methods (e.g., 'scatter', 'image').

        Returns
        -------
        Bunch
            A bunch object containing the plot data for all methods with the specified prefix.

        """
        results = Bunch()
        for attr_name in dir(self):
            if attr_name.startswith(plot_prefix):
                method = getattr(self, attr_name)
                if callable(method):
                    results.update(method())

        return results

    # --------------------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------------------

    @property
    def spike_amps(self) -> np.ndarray:
        """Get spike amplitudes for the selected spikes and non-NaN depths and amplitudes."""
        return self.data['spikes']['amps'][self.spike_idx][self.kp_idx]

    @property
    def spike_depths(self) -> np.ndarray:
        """Get spike depths for the selected spikes and non-NaN depths and amplitudes."""
        return self.data['spikes']['depths'][self.spike_idx][self.kp_idx]

    @property
    def spike_clusters(self) -> np.ndarray:
        """Get spike clusters for the selected spikes and non-NaN depths and amplitudes."""
        return self.data['spikes']['clusters'][self.spike_idx][self.kp_idx]

    @property
    def spike_times(self) -> np.ndarray:
        """Get spike times for the selected spikes and non-NaN depths and amplitudes."""
        return self.data['spikes']['times'][self.spike_idx][self.kp_idx]

    # --------------------------------------------------------------------------------------------
    # Data handling
    # --------------------------------------------------------------------------------------------
    @skip_missing(['spikes'])
    def compute_avg_cluster_activity(self) -> None:
        """
        Compute average amplitude, depth and firing rate for each cluster.

        Notes
        -----
        This method sets the following attributes:

        self.clust_id : np.ndarray
            Cluster identifiers.
        self.avg_amp : np.ndarray
            Average spike amplitude per cluster.
        self.avg_depth : np.ndarray
            Average spike depth per cluster.
        self.avg_fr : np.ndarray
            Average firing rate per cluster.
        """
        self.clust_id, self.avg_amp, self.avg_depth, self.avg_fr = (
            compute_spike_average(self.data['spikes'], self.data['clusters']))

    @skip_missing(['spikes'])
    def compute_rasters(self) -> None:
        """
        Compute binned firing rate, amplitude, spike times, and depths.

        Notes
        -----
        This method sets the following attributes:

        self.chn_min_bc : float
            Minimum depth boundary including spike depths.
        self.chn_max_bc : float
            Maximum depth boundary including spike depths.
        self.fr : np.ndarray
            Binned firing rate array.
        self.amp : np.ndarray
            Binned spike amplitude array.
        self.times : np.ndarray
            Binned spike time array.
        self.depths : np.ndarray
            Depth values corresponding to bins.
        """
        self.chn_min_bc = np.min(np.r_[self.chn_min, self.spike_depths])
        self.chn_max_bc = np.max(np.r_[self.chn_max, self.spike_depths])

        self.fr, self.amp, self.times, self.depths = (
            compute_bincount(self.spike_times, self.spike_depths, self.spike_amps,
                             ylim=[self.chn_min_bc, self.chn_max_bc]))

    @skip_missing(['spikes'])
    def filter_units(self, filter_type) -> None:
        """
        Filter spikes according to cluster metrics.

        Parameters
        ----------
        filter_type: str
            The filter criterion. Options are 'All', 'IBL good', 'KS good', 'KS mua'.

        Notes
        -----
        This method sets the following attributes:

        self.cluster_idx : np.ndarray
            The index of clusters that match the filter criteria
        self.spike_idx : np.ndarray
            The index of spikes contained in the filtered clusters (cluster_idx)
        self.kp_idx : np.ndarray
            The index of spikes that do not have NaN values for depth and amplitude
        """
        try:
            if filter_type == "All":
                self.cluster_idx = np.arange(self.data['clusters'].channels.size)
                self.spike_idx = np.arange(self.data['spikes']['clusters'].size)
            else:
                column, condition = FILTER_MATCH[filter_type]
                self.cluster_idx = np.where(self.data['clusters'].metrics[column] == condition)[0]
                self.spike_idx = np.where(
                    np.isin(self.data['spikes']['clusters'], self.cluster_idx))[0]

            self.kp_idx = np.where(~np.isnan(self.data['spikes']['depths'][self.spike_idx]) &
                                   ~np.isnan(self.data['spikes']['amps'][self.spike_idx]))[0]
        except Exception:
            logger.warning(f'{filter_type} metrics not found will return all units instead')
            self.filter_units('All')

    # --------------------------------------------------------------------------------------------
    # Scatter plots
    # --------------------------------------------------------------------------------------------
    @skip_missing(['spikes'])
    def scatter_firing_rate(self) -> dict[str, Any]:
        """
        Generate data for a scatter plot of spike depths vs spike times, coloured by amplitude.

        Returns
        -------
        Dict
            A dict containing a ScatterData object with key 'Amplitude'.

        Notes
        -----
        - Spikes data is subsampled for performance.
        - Amplitudes are split into a_bin bins and colours set accordingly.
        - Saturated amplitudes, those above the 90th percentile, are coloured dark purple.
        """
        a_bin = 10
        subsample = 500

        # Subsample data
        times = self.spike_times[::subsample]
        depths = self.spike_depths[::subsample]
        amps = self.spike_amps[::subsample]

        # Amplitude bins (ignore top 10% outliers)
        amp_range = np.quantile(amps, [0, 0.9])
        amp_bins = np.linspace(amp_range[0], amp_range[1], a_bin)

        # Map amplitudes to bin indices
        bin_idx = np.digitize(amps, amp_bins, right=True)

        # Build colormap
        colour_bin = np.linspace(0.0, 1.0, a_bin + 1)
        colormap = cm.get_cmap('BuPu')(colour_bin)[..., :3]

        # Initialize colours and sizes
        spikes_colours = np.array(["#000000"] * amps.size)
        spikes_size = np.zeros(amps.size)

        # Assign colour and sizes according to bin index
        valid = bin_idx < a_bin
        spikes_colours[valid] = [colors.to_hex(c) for c in colormap[bin_idx[valid]]]
        spikes_size[valid] = bin_idx[valid] / (a_bin / 4)

        # For saturated amplitudes, set to dark purple and larger size
        saturated = bin_idx >= a_bin
        spikes_colours[saturated] = "#400080"
        spikes_size[saturated] = (a_bin - 1) / (a_bin / 4)

        xrange = np.array([np.min(times), np.max(times)])

        scatter = ScatterData(
            x=times,
            y=depths,
            levels=amp_range * 1e6,
            default_levels=amp_range * 1e6,
            colours=spikes_colours,
            pen=None,
            size=spikes_size,
            symbol=np.array('o'),
            xrange=xrange,
            xaxis='Time (s)',
            title='Amplitude (uV)',
            cmap='BuPu',
            cluster=False
        )

        return {'Amplitude': scatter}

    @skip_missing(['spikes'])
    def scatter_amp_depth_fr(self) -> dict[str, Any]:
        """
        Generate data for a scatter plot of cluster depth vs. cluster amplitude.

        Scatter points are coloured by cluster firing rate.

        Returns
        -------
        Dict
            A dict containing a ScatterData object with key 'Cluster Amp vs Depth vs FR'.
        """
        levels = np.nanquantile(self.avg_fr[self.cluster_idx], [0, 1])

        scatter = ScatterData(
            x=self.avg_amp[self.cluster_idx],
            y=self.avg_depth[self.cluster_idx],
            levels=levels,
            default_levels=np.copy(levels),
            colours=self.avg_fr[self.cluster_idx],
            pen='k',
            size=np.array(8),
            symbol=np.array('o'),
            xrange=np.array([0.9 * np.nanmin(self.avg_amp[self.cluster_idx]),
                             1.1 * np.nanmax(self.avg_amp[self.cluster_idx])]),
            xaxis='Amplitude (uV)',
            title='Firing Rate (Sp/s)',
            cmap='hot',
            cluster=True
        )

        return {'Cluster Amp vs Depth vs FR': scatter}

    @skip_missing(['spikes'])
    def scatter_amp_depth_duration(self) -> dict[str, Any]:
        """
        Generate data for a scatter plot of cluster depth vs. cluster amplitude.

        Scatter points are coloured by cluster peak to trough duration.

        Returns
        -------
        Dict
            A dict containing a ScatterData object with key 'Cluster Amp vs Depth vs Duration'.
        """
        levels = np.array([-1.5, 1.5])

        scatter = ScatterData(
            x=self.avg_amp[self.cluster_idx],
            y=self.avg_depth[self.cluster_idx],
            levels=levels,
            default_levels=np.copy(levels),
            colours=self.data['clusters']['peakToTrough'][self.cluster_idx],
            pen='k',
            size=np.array(8),
            symbol=np.array('o'),
            xrange=np.array([0.9 * np.nanmin(self.avg_amp[self.cluster_idx]),
                             1.1 * np.nanmax(self.avg_amp[self.cluster_idx])]),
            xaxis='Amplitude (uV)',
            title='Peak to Trough duration (ms)',
            cmap='RdYlGn',
            cluster=True
        )

        return {'Cluster Amp vs Depth vs Duration': scatter}

    @skip_missing(['spikes'])
    def scatter_fr_depth_amp(self) -> dict[str, Any]:
        """
        Generate data for a scatter plot of cluster depth vs. cluster firing rate.

        Scatter points are coloured by cluster amplitude.

        Returns
        -------
        Dict
            A dict containing a ScatterData object with key 'Cluster FR vs Depth vs Amp'.
        """
        levels = np.nanquantile(self.avg_amp[self.cluster_idx], [0, 1])

        scatter = ScatterData(
            x=self.avg_fr[self.cluster_idx],
            y=self.avg_depth[self.cluster_idx],
            levels=levels,
            default_levels=np.copy(levels),
            colours=self.avg_amp[self.cluster_idx],
            pen='k',
            size=np.array(8),
            symbol=np.array('o'),
            xrange=np.array([0.9 * np.nanmin(self.avg_fr[self.cluster_idx]),
                             1.1 * np.nanmax(self.avg_fr[self.cluster_idx])]),
            xaxis='Firing Rate (Sp/s)',
            title='Amplitude (uV)',
            cmap='magma',
            cluster=True
        )

        return {'Cluster FR vs Depth vs Amp': scatter}

    # --------------------------------------------------------------------------------------------
    # Image plots
    # --------------------------------------------------------------------------------------------
    @skip_missing(['spikes'])
    def image_firing_rate(self) -> dict[str, Any]:
        """
        Generate data for an image plot of binned firing rates across time.

        Returns
        -------
        Dict
            A dict containing a ImageData object with key 'Firing Rate'.
        """
        xscale = (self.times[-1] - self.times[0]) / self.fr.shape[1]
        yscale = (self.depths[-1] - self.depths[0]) / self.fr.shape[0]
        levels = np.quantile(np.mean(self.fr.T, axis=0), [0, 1])

        img = ImageData(
            img=self.fr.T,
            scale=np.array([xscale, yscale]),
            levels=levels,
            default_levels=np.copy(levels),
            offset=np.array([0, self.chn_min]),
            xrange=np.array([self.times[0], self.times[-1]]),
            xaxis='Time (s)',
            cmap='binary',
            title='Firing Rate'
        )

        return {'Firing Rate': img}

    @skip_missing(['spikes'])
    def image_correlation(self) -> dict[str, Any]:
        """
        Generate data for an image plot of the correlation of binned firing rates across depth.

        Returns
        -------
        Dict
            A dict containing a ImageData object with key 'Correlation'.
        """
        # Resample to 40um depth bins for correlation calculation
        dbin = 40
        factor = int(dbin / DBIN)
        bincount = group_bincount(self.fr, factor, axis=0)
        depths = self.depths[::factor]

        corr = np.corrcoef(bincount)
        corr[np.isnan(corr)] = 0
        scale = (np.max(depths) - np.min(depths)) / corr.shape[0]
        levels = np.array([np.min(corr), np.max(corr)])

        img = ImageData(
            img=corr,
            scale=np.array([scale, scale]),
            levels=levels,
            default_levels=np.copy(levels),
            offset=np.array([self.chn_min, self.chn_min]),
            xrange=np.array([self.chn_min, self.chn_max]),
            cmap='viridis',
            title='Correlation',
            xaxis='Distance from probe tip (um)'
        )
        return {'Correlation': img}

    @skip_missing(['rms_AP'])
    def image_rms_ap(self) -> dict[str, Any]:
        """
        Generate data for an image plot of the RMS of the AP band across time.

        Returns
        -------
        Dict
            A dict containing a ImageData object with key 'rms_AP'.
        """
        return self._image_rms('AP')

    @skip_missing(['rms_LF'])
    def image_rms_lf(self) -> dict[str, Any]:
        """
        Generate data for an image plot of the RMS of the LFP band across time.

        Returns
        -------
        Dict
            A bunch containing a ImageData object with key 'rms_LF'.
        """
        return self._image_rms('LF')

    def _image_rms(self, band: str) -> dict[str, Any]:
        """
        Generate data for an image plot of the RMS for the specified frequency band (AP or LF).

        Parameters
        ----------
        band: str
            The frequency band to process (AP or LF).

        Returns
        -------
        Dict
            A dict containing a ImageData object with key 'rms_{band}'.

        Notes
        -----
        - Channels with the same depth are averaged together
        - The median across depths is subtracted per time point to remove striping,
          but the global median is added back for interpretability.
        - If the probe has non-contiguous channels, the output is padded with NaNs
          to align with the full channel map.
        """
        # Identify channels at the same depth
        img = average_chns_at_same_depths(self.shank_sites,
                                          self.data[f"rms_{band}"]["rms"]) * 1e6  # convert to µV

        # Median subtract across depths (remove horizontal bands)
        depth_medians = np.median(img, axis=1, keepdims=True)
        global_median = np.mean(depth_medians)
        img = img - depth_medians + global_median

        # Reconstruct full channel map (handles gaps in channel geometry)
        img_full = pad_data_to_full_chn_map(self.shank_sites, img)

        # Scaling for plotting
        timestamps = self.data[f"rms_{band}"]["timestamps"]
        xscale = (timestamps[-1] - timestamps[0]) / img_full.shape[0]
        yscale = (self.chn_max - self.chn_min) / img_full.shape[1]
        levels = np.quantile(img, [0.1, 0.9])

        cmap = "plasma" if band == "AP" else "inferno"

        img = ImageData(
            img=img_full,
            scale=np.array([xscale, yscale]),
            levels=levels,
            default_levels=np.copy(levels),
            offset=np.array([0, self.chn_min]),
            cmap=cmap,
            xrange=np.array([timestamps[0], timestamps[-1]]),
            xaxis=self.data[f"rms_{band}"]["xaxis"],
            title=f"{band} RMS (uV)",
        )

        return {f"rms {band}": img}

    @skip_missing(['psd_LF'])
    def image_lfp_spectrum(self) -> dict[str, Any]:
        """
        Generate data for an image plot of the LFP power spectrum across frequency.

        Returns
        -------
        Dict
            A dict containing a ImageData object with key 'LF spectrum'.

        Notes
        -----
        - Channels with the same depth are averaged together
        - The power spectrum is limited to the range 0-300 Hz
        - The power is converted to dB scale
        """
        # Find frequency range
        freq_range = [0, 300]
        freq_idx = np.where((self.data['psd_LF']['freqs'] >= freq_range[0]) &
                            (self.data['psd_LF']['freqs'] < freq_range[1]))[0]

        # Extract PSD data for the selected frequency range and selected channels
        lfp_power = self.data['psd_LF']['power'][freq_idx, :]
        lfp_power = 10 * np.log10(lfp_power)

        # Average data across channels at the same depth
        img = average_chns_at_same_depths(self.shank_sites, lfp_power)

        # Reconstruct full channel map (handles gaps in channel geometry)
        img_full = pad_data_to_full_chn_map(self.shank_sites, img)

        # Scaling for plotting
        xscale = (freq_range[-1] - freq_range[0]) / img_full.shape[0]
        yscale = (self.chn_max - self.chn_min) / img_full.shape[1]
        levels = np.quantile(img, [0.1, 0.9])

        img = ImageData(
            img=img_full,
            scale=np.array([xscale, yscale]),
            levels=levels,
            default_levels=np.copy(levels),
            offset=np.array([0, self.chn_min]),
            cmap='viridis',
            xrange=np.array([freq_range[0], freq_range[-1]]),
            xaxis='Frequency (Hz)',
            title='PSD (dB)'
        )

        return {'LF spectrum': img}

    @skip_missing(['spikes'])
    def image_passive_events(self) -> dict[str, Any]:
        """
        Generate data for image plots of the passive event aligned PSTHs.

        Returns
        -------
        Dict
            A dict containing multiple ImageData objects with keys according to stimulus type.

        Notes
        -----
        - Will only return data for passive events that are present in the data
        """
        # Find the list of passive events that are present in the data
        if not self.data['pass_stim']['exists'] and not self.data['gabor']['exists']:
            return dict()
        elif not self.data['pass_stim']['exists'] and self.data['gabor']['exists']:
            stim_types = ['leftGabor', 'rightGabor']
            stims = {stim_type: self.data['gabor'][stim_type] for stim_type in stim_types}
        elif self.data['pass_stim']['exists'] and not self.data['gabor']['exists']:
            stim_types = ['valveOn', 'toneOn', 'noiseOn']
            stims = {stim_type: self.data['pass_stim'][stim_type] for stim_type in stim_types}
        else:
            stim_types = ['valveOn', 'toneOn', 'noiseOn', 'leftGabor', 'rightGabor']
            stims = {stim_type: self.data['pass_stim'][stim_type]
                     for stim_type in stim_types[0:3]}
            stims.update({stim_type: self.data['gabor'][stim_type]
                          for stim_type in stim_types[3:]})

        # Compute normalised event aligned psths
        base_stim = 1
        pre_stim = 0.4
        post_stim = 1
        stim_events = passive.get_stim_aligned_activity(
            stims, self.spike_times, self.spike_depths, pre_stim=pre_stim, post_stim=post_stim,
            base_stim=base_stim, y_lim=[self.chn_min, self.chn_max])

        # Loop over each stimulus type and create ImageData objects
        passive_imgs = dict()
        for stim_type, aligned_img in stim_events.items():
            xscale = (post_stim + pre_stim) / aligned_img.shape[1]
            yscale = ((self.chn_max - self.chn_min) / aligned_img.shape[0])
            levels = np.array([-10, 10])

            img = ImageData(
                img=aligned_img.T,
                scale=np.array([xscale, yscale]),
                levels=levels,
                default_levels=np.copy(levels),
                offset=np.array([-1 * pre_stim, self.chn_min]),
                cmap='bwr',
                xrange=np.array([-1 * pre_stim, post_stim]),
                xaxis='Time from Stim Onset (s)',
                title='Firing rate (z score)'
            )

            passive_imgs.update({stim_type: img})

        return passive_imgs

    @skip_missing(['raw_snippets'])
    def image_raw_data(self) -> dict[str, Any]:
        """
        Generate data for image plots of raw ephys data snippets.

        Returns
        -------
        Dict
            A dict containing multiple ImageData objects with keys according to the time of the
            snippet during the recording.
        """
        raw_imgs = dict()

        for t, raw_img in self.data['raw_snippets']['images'].items():
            x_range = np.array([0, raw_img.shape[0] - 1]) / self.data['raw_snippets']['fs'] * 1e3
            xscale = (x_range[1] - x_range[0]) / raw_img.shape[0]
            yscale = (self.chn_max - self.chn_min) / raw_img.shape[1]
            levels = 10 ** (-90 / 20) * 4 * np.array([-1, 1])

            img = ImageData(
                img=raw_img,
                scale=np.array([xscale, yscale]),
                levels=levels,
                default_levels=np.copy(levels),
                offset=np.array([0, self.chn_min]),
                cmap='bone',
                xrange=x_range,
                xaxis='Time (ms)',
                title='Power (uV)'
            )
            raw_imgs[f'Raw ap t={t}'] = img

        return raw_imgs

    # --------------------------------------------------------------------------------------------
    # Line plots
    # --------------------------------------------------------------------------------------------
    @skip_missing(['spikes'])
    def line_firing_rate(self) -> dict[str, Any]:
        """
        Generate data for a line plot of depth vs firing rate averaged across time.

        Returns
        -------
        Dict
            A dict containing a LineData object with key 'Firing Rate'.
        """
        # Resample to 10um depth bins for smoother depth profile
        dbin = 10
        factor = int(dbin / DBIN)
        bincount = group_bincount(self.fr, factor, axis=0)
        depths = self.depths[::factor]

        mean_fr = np.mean(bincount, axis=1)

        line = LineData(
            x=mean_fr,
            y=depths,
            xrange=np.array([0, np.max(mean_fr)]),
            levels=np.array([0, np.max(mean_fr)]),
            default_levels=np.array([0, np.max(mean_fr)]),
            xaxis='Firing Rate (Sp/s)'
        )

        return {'Firing Rate': line}

    @skip_missing(['spikes'])
    def line_amplitude(self) -> dict[str, Any]:
        """
        Generate data for a line plot of depth vs amplitude averaged across time.

        Returns
        -------
        Dict
            A dict containing a LineData object with key 'Amplitude'.
        """
        # Resample to 10um depth bins for smoother depth profile
        dbin = 10
        factor = int(dbin / DBIN)
        bincount = group_bincount(self.amp, factor, axis=0)
        depths = self.depths[::factor]

        mean_amp = np.mean(bincount, axis=1) * 1e6

        line = LineData(
            x=mean_amp,
            y=depths,
            xrange=np.array([0, np.max(mean_amp)]),
            levels=np.array([0, np.max(mean_amp)]),
            default_levels=np.array([0, np.max(mean_amp)]),
            xaxis='Amplitude (uV)'
        )

        return {'Amplitude': line}

    # --------------------------------------------------------------------------------------------
    # Probe plots
    # --------------------------------------------------------------------------------------------
    @skip_missing(['rms_AP'])
    def probe_rms_ap(self) -> dict[str, Any]:
        """
        Generate data for a probe plot of the RMS of the AP band averaged across time.

        Returns
        -------
        Dict
            A dict containing a ProbeData object with key 'rms_AP'.
        """
        return self._probe_rms('AP')

    @skip_missing(['rms_LF'])
    def probe_rms_lf(self) -> dict[str, Any]:
        """
        Generate data for a probe plot of the RMS of the LFP band averaged across time.

        Returns
        -------
        Dict
            A dict containing a ProbeData object with key 'rms_LF'.
        """
        return self._probe_rms('LF')

    def _probe_rms(self, band: str) -> dict[str, Any]:
        """
        Generate data for a probe plot of the RMS for the specified frequency band (AP or LF).

        Parameters
        ----------
        band: str
            The frequency band to process (AP or LF).

        Returns
        -------
        Dict
            A dict containing a ProbeData object with key 'rms_{band}'.
        """
        # Average data across time
        rms_avg = np.mean(self.data[f'rms_{band}']['rms'], axis=0) * 1e6
        levels = np.quantile(rms_avg, [0.1, 0.9])
        # Split the data into banks of channels according to the probe geometry
        probe_img, probe_scale, probe_offset, probe_idx = (
            arrange_channels_into_banks(self.shank_sites, rms_avg, bnk_width=BNK_SIZE))

        cmap = 'plasma' if band == 'AP' else 'inferno'

        probe = ProbeData(
            img=probe_img,
            idx=probe_idx,
            scale=probe_scale,
            offset=probe_offset,
            levels=levels,
            default_levels=np.copy(levels),
            cmap=cmap,
            xrange=np.array([0 * BNK_SIZE, (self.shank_sites['n_banks']) * BNK_SIZE]),
            title=band + ' RMS (uV)'
        )

        return {f'rms {band}': probe}

    @skip_missing(['psd_LF'])
    def probe_lfp_spectrum(self) -> dict[str, Any]:
        """
        Generate data for probe plots of the LFP power averaged across different frequency bands.

        Returns
        -------
        Dict
            A dict containing multiple ProbeData objects with keys according to frequency bands.
        """
        # Define frequency bands
        freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))

        data_probe = dict()
        for freq in freq_bands:
            freq_idx = np.where((self.data['psd_LF']['freqs'] >= freq[0])
                                & (self.data['psd_LF']['freqs'] < freq[1]))[0]
            lfp_power = np.mean(self.data['psd_LF']['power'][freq_idx], axis=0)
            lfp_power = 10 * np.log10(lfp_power)
            probe_img, probe_scale, probe_offset, probe_idx = (
                arrange_channels_into_banks(self.shank_sites, lfp_power, bnk_width=BNK_SIZE))
            levels = np.quantile(lfp_power, [0.1, 0.9])

            probe = ProbeData(
                img=probe_img,
                idx=probe_idx,
                scale=probe_scale,
                offset=probe_offset,
                levels=levels,
                default_levels=np.copy(levels),
                cmap='viridis',
                xrange=np.array([0 * BNK_SIZE, (self.shank_sites['n_banks']) * BNK_SIZE]),
                title=f'{freq[0]}-{freq[1]} Hz (dB)'
            )
            data_probe.update({f'{freq[0]} - {freq[1]} Hz': probe})

        return data_probe

    @skip_missing(['spikes', 'rf_map'])
    def probe_rfmap(self) -> dict[str, Any]:
        """
        Generate data for probe plots of the Receptive Field map (on and off) across depth.

        Returns
        -------
        Dict
            A dict containing ProbeData objects with for keys 'RF Map - on' and 'RF Map - off'.

        Notes
        -----
        - Although this is a probe plot the data is not split into banks as for the case of other
         probe plots.
        """
        # Extract stimulus times and positions
        rf_map_times, rf_map_pos, rf_stim_frames = (
            passive.get_on_off_times_and_positions(self.data['rf_map']))

        # Compute receptive field map over depth
        rf_map, _ = \
            passive.get_rf_map_over_depth(rf_map_times, rf_map_pos, rf_stim_frames,
                                          self.spike_times,
                                          self.spike_depths,
                                          d_bin=160, y_lim=[self.chn_min_bc, self.chn_max_bc])

        # Apply SVD decomposition to obtain ON and OFF maps
        rfs_svd = passive.get_svd_map(rf_map)
        img = {}
        img['on'] = np.vstack(rfs_svd['on'])
        img['off'] = np.vstack(rfs_svd['off'])

        # Scaling
        yscale = ((self.chn_max - self.chn_min) / img['on'].shape[0])
        xscale = 1
        depths = np.linspace(self.chn_min, self.chn_max, len(rfs_svd['on']) + 1)
        levels = np.quantile(np.c_[img['on'], img['off']], [0, 1])

        data_img = dict()
        sub_type = ['on', 'off']
        for sub in sub_type:
            sub_data = {
                f'RF Map - {sub}':
                    ProbeData(
                        img=[img[sub].T],
                        idx=[np.arange(img[sub].shape[0])],
                        scale=[np.array([xscale, yscale])],
                        levels=levels,
                        default_levels=np.copy(levels),
                        offset=[np.array([0, self.chn_min])],
                        cmap='viridis',
                        xrange=np.array([0, 15]),
                        title='rfmap (dB)',
                        boundaries=depths
                    )
            }
            data_img.update(sub_data)

        return data_img
