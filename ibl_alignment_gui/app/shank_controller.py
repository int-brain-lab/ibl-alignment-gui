import numpy as np
import pyqtgraph as pg

from ibl_alignment_gui.app.shank_view import ShankView
from ibl_alignment_gui.handlers.shank_handler import ShankHandler
from ibl_alignment_gui.utils.qt.custom_widgets import ColorBar
from iblutil.util import Bunch


class ShankController:
    """
    Controller for handling data and plots on a shank of a probe for a given configuration.

    Parameters
    ----------
    model: ShankHandler
        The model containing all the data for the shank and config combination
    name: str
        The name of shank
    index: int
        The index of the shank
    config:
        The config of the shank

    Attributes
    ----------
    name: str
        The name of shank
    index: int
        The index of the shank
    config:
        The config of the shank
    model: ShankHandler
        The model containing all the data for the shank and config combination
    view: ShankView
        The view containing all the plots for the shank and config combination
    cluster: bool
        Whether the chosen scatter plot contains cluster data
    cluster_data: np.ndarray | None
        The cluster data for the chosen scatter plot if it exists
    """

    def __init__(self, model: ShankHandler, name: str, index: int, config: str):

        self.name: str = name
        self.index: int = index
        self.config: str = config
        self.model: ShankHandler = model
        self.view: ShankView = ShankView(self.name, self.index, self.config)
        self.cluster: bool = False
        self.cluster_data: np.ndarray | None = None

    def init_reference_line_arrays(self):
        """" See :meth:`ShankView.init_reference_line_arrays` for details."""
        self.view.init_reference_line_arrays()

    def init_plot_items(self):
        """" See :meth:`ShankView.init_plot_items` for details."""
        self.view.init_plot_items()

    # --------------------------------------------------------------------------------------------
    # Plot functions
    # --------------------------------------------------------------------------------------------
    def plot_histology(self) -> pg.LinearRegionItem:
        """
        Plot the histology plot.

        Returns
        -------
        pg.LinearRegionItem
            A pg.LinearRegionItem corresponding to the second last brain region added to the
            histology plot
        """
        self.view.plot_histology(self.view.fig_hist, self.model.hist_data)

        return self.view.hist_regions[-2]

    def plot_histology_ref(self) -> None:
        """Plot the histology reference plot."""
        self.view.plot_histology(self.view.fig_hist_ref, self.model.hist_data_ref, ax='right')

    def plot_scale_factor(self) -> ColorBar:
        """
        Plot the scale factor along the probe.

        Returns
        -------
        cbar: ColorBar
            The ColorBar object added to the plot
        """
        data = self.model.scale_data
        data['scale_factor'] = data['scale'] - 0.5
        cbar = self.view.plot_scale_factor(data)

        return cbar

    def plot_slice(self, plot_key: str) -> tuple[pg.ViewBox, pg.ImageItem, ColorBar]:
        """
        Plot a slice plot showing a coronal slice through the brain.

        Parameters
        ----------
        plot_key: str
            The key of the plot to display

        Returns
        -------
        fig: pg.ViewBox
            The viewbox that contains the image
        img: pg.ImageItem
            The image item that was added to the plot
        cbar: ColorBar
            The colobar object that was added to the plot
        """
        data = self.model.slice_plots.get(plot_key, None)
        data_traj = Bunch()
        data_traj['x'] = self.model.xyz_track[:, 0]
        data_traj['y'] = self.model.xyz_track[:, 2]
        img, cbar = self.view.plot_slice(data, data_traj)
        fig = self.view.fig_slice

        return fig, img, cbar

    def plot_fit(self) -> None:
        """Plot fit lines on the fit figure."""
        data = Bunch()
        data['x'] = self.model.feature * 1e6
        data['y'] = self.model.track * 1e6
        data['depth_lin'] = self.model.feature2track_lin(self.view.depth, data['x'], data['y'])
        data['depth'] = self.view.depth
        self.view.plot_fit(data)

    def plot_channels(self, fig_slice: pg.ViewBox, colour: str | None = None) -> None:
        """
        Plot channels on a slice plot.

        Parameters
        ----------
        fig_slice: pg.ViewBox
            The slice fig to add the channel items to
        colour: str
            The colour of the scatter points used to plot the channels

        Notes
        -----
        - fig_slice is passed in as a parameter as for the dual config display the channels
        plotted on a different slice figure than the one stored in the view.
        """
        data = Bunch()
        data['xyz_channels'] = self.model.xyz_channels
        data['track_lines'] = self.model.track_lines
        self.view.plot_channels(fig_slice, data, colour=colour)

    def plot_scatter(self, plot_key: str, levels: list | None = None) -> ColorBar | None:
        """
        Plot a scatter plot.

        Parameters
        ----------
        plot_key: str
            The key of the plot to display
        levels:
            The levels used to scale the colorbar on the plot

        Returns
        -------
        cbar: ColorBar
            The colobar abject that was added to the plot
        """
        data = self.model.scatter_plots.get(plot_key, None)
        cbar = self.view.plot_scatter(data, levels=levels)

        if data and data.cluster:
            self.cluster_data = data.x
            self.cluster = True
        else:
            self.cluster = False

        return cbar

    def plot_line(self, plot_key: str) -> None:
        """
        Plot a line plot.

        Parameters
        ----------
        plot_key: str
            The key of the plot to display
        """
        data = self.model.line_plots.get(plot_key, None)
        self.view.plot_line(data)

    def plot_probe(self, plot_key: str, levels: list | None = None) -> ColorBar | None:
        """
        Plot a probe plot.

        Parameters
        ----------
        plot_key: str
            The key of the plot to display
        levels:
            The levels used to scale the colorbar on the plot

        Returns
        -------
        cbar: ColorBar
            The colobar abject that was added to the plot
        """
        data = self.model.probe_plots.get(plot_key, None)
        cbar = self.view.plot_probe(data, levels=levels)

        return cbar

    def plot_image(self, plot_key: str, levels: list | None = None) -> ColorBar | None:
        """
        Plot an image plot.

        Parameters
        ----------
        plot_key: str
            The key of the plot to display
        levels:
            The levels used to scale the colorbar on the plot

        Returns
        -------
        cbar: ColorBar
            The colobar abject that was added to the plot
        """
        data = self.model.image_plots.get(plot_key, None)
        cbar = self.view.plot_image(data, levels=levels)

        return cbar

    # --------------------------------------------------------------------------------------------
    # Update displays
    # --------------------------------------------------------------------------------------------
    def toggle_labels(self, *args) -> None:
        """See :meth:`ShankView.toggle_labels` for details."""
        self.view.toggle_labels(*args)

    def toggle_channels(self, *args) -> None:
        """See :meth:`ShankView.toggle_channels` for details."""
        self.view.toggle_channels(*args)

    def set_header_style(self, *args) -> None:
        """See :meth:`ShankView.set_header_style` for details."""
        self.view.set_header_style(*args)

    def set_xaxis_range(self, fig: str) -> None:
        """
        Set the x-axis range of the specified figure.

        Parameters
        ----------
        fig: str
            The attribute name of the figure to set the x-axis for
        """
        self.view.set_xaxis_range(getattr(self.view, fig))

    def set_yaxis_range(self, fig: str) -> None:
        """
        Set the y-axis range of the specified figure.

        Parameters
        ----------
        fig: str
            The attribute name of the figure to set the y-axis for
        """
        self.view.set_yaxis_range(getattr(self.view, fig))

    def set_probe_lims(self) -> None:
        """Set the limits for the probe tip and probe top based on values stored in model."""
        self.view.set_probe_lims(self.model.chn_min, self.model.chn_max)

    def set_yaxis_lims(self, *args) -> None:
        """See :meth:`ShankView.set_yaxis_lims` for details."""
        self.view.set_yaxis_lims(*args)

    def set_scale_title(self, hover_item: pg.LinearRegionItem):
        """
        Update the title of the scale plot color bar based on the hovered region.

        Parameters
        ----------
        hover_item: pg.LinearRegionItem
            The region item currently hovered over.
        """
        idx = self.view.match_linear_region(hover_item)
        self.view.set_fig_scale_title(self.model.scale_data['scale'][idx])

    def reset_slice_axis(self) -> None:
        """See :meth:`ShankView.reset_slice_axis` for details."""
        self.view.reset_slice_axis()

    def filter_units(self, filter_type: str) -> None:
        """See :meth:`ShankHandler.filter_units` for details."""
        self.model.filter_units(filter_type)

    def reset_levels(self) -> None:
        """See :meth:`ShankHandler.reset_levels` for details."""
        self.model.reset_levels()

    # --------------------------------------------------------------------------------------------
    # Fitting functions
    # --------------------------------------------------------------------------------------------
    def offset_hist_data(self, *args) -> None:
        """See :meth:`ShankHandler.offset_hist_data` for details."""
        self.model.offset_hist_data(*args)

    def scale_hist_data(self, extend_feature: float, lin_fit: bool) -> None:
        """
        Scale brain regions along the probe track based on reference lines.

        Parameters
        ----------
        extend_feature: float
            Amount to extend for linear fit
        lin_fit: bool
            Whether to use a linear fit or not
        """
        line_feature, line_track = self.view.get_feature_and_track_coords()
        self.model.scale_hist_data(line_track, line_feature,
                                   extend_feature=extend_feature, lin_fit=lin_fit)

    def get_scaled_histology(self) -> None:
        """See :meth:`ShankHandler.get_scaled_histology` for details."""
        self.model.get_scaled_histology()

    # --------------------------------------------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------------------------------------------
    @property
    def points(self) -> list[pg.ScatterPlotItem]:
        """Return the points stored in the view."""
        return self.view.points

    def match_feature_line(self, *args) -> tuple[int | None, list | np.ndarray | None]:
        """See :meth:`ShankView.match_feature_line` for details."""
        return self.view.match_feature_line(*args)

    def match_track_line(self, *args) -> int | None:
        """See :meth:`ShankView.match_track_line` for details."""
        return self.view.match_track_line(*args)

    def create_reference_line_and_point(self, *args, **kwargs) -> tuple:
        """See :meth:`ShankView.create_reference_line_and_point` for details."""
        return self.view.create_reference_line_and_point(*args, **kwargs)

    def remove_reference_line(self, *args) -> None:
        """See :meth:`ShankView.remove_reference_line` for details."""
        self.view.remove_reference_line(*args)

    def delete_reference_line_and_point(self, *args) -> None:
        """See :meth:`ShankView.delete_reference_line_and_point` for details."""
        self.view.delete_reference_line_and_point(*args)

    def update_feature_reference_line_and_point(self, *args) -> None:
        """See :meth:`ShankView.update_feature_reference_line_and_point` for details."""
        self.view.update_feature_reference_line_and_point(*args)

    def update_track_reference_line_and_point(self, *args) -> None:
        """See :meth:`ShankView.update_track_reference_line_and_point` for details."""
        self.view.update_track_reference_line_and_point(*args)

    def align_reference_lines_and_points(self) -> None:
        """See :meth:`ShankView.align_reference_lines` for details."""
        self.view.align_reference_lines_and_points()

    def remove_reference_lines_from_display(self) -> None:
        """" See :meth:`ShankView.remove_reference_lines_to_display` for details."""
        self.view.remove_reference_lines_from_display()

    def add_reference_lines_to_display(self) -> None:
        """See :meth:`ShankView.add_reference_lines_to_display` for details."""
        self.view.add_reference_lines_to_display()
