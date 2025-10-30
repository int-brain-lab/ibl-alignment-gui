import random

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from ibl_alignment_gui.loaders.plot_loader import (
    ImageData,
    LineData,
    ProbeData,
    ScatterData,
)
from ibl_alignment_gui.utils.qt.adapted_axis import replace_axis
from ibl_alignment_gui.utils.qt.custom_widgets import ColorBar, set_axis
from iblutil.util import Bunch


class ShankView:
    """
    View for displaying plots for a shank on a probe for a given recording configuration.

    Parameters
    ----------
    name: str
        The name of the shank
    index: int
        The index of the shank
    config:
        The config of the shank
    """

    COLOURS = ['#cc0000', '#6aa84f', '#1155cc', '#a64d79']
    HEADER_STYLE = {
        'selected': """QLabel {
                    background-color: #c92d0e;
                    border: 1px solid lightgrey;
                    color: white;
                    padding: 6px;
                    font-weight: bold;
                }
                """,
        'deselected': """
                QLabel {
                    background-color: rgb(240, 240, 240);
                    border: 1px solid lightgrey;
                    color: black;
                    padding: 6px;
                    font-weight: bold;
                }
                """
    }

    def __init__(self, name: str, index: int, config: str):

        self.name: str = name
        self.index: int = index
        self.config: str = config

        # Set colour for reference lines and points
        colour: str = self.COLOURS[self.index]
        self.pen: QtGui.QPen = pg.mkPen(color=colour, style=QtCore.Qt.SolidLine, width=3)
        self.pen_dot: QtGui.QPe = pg.mkPen(color=colour, style=QtCore.Qt.DotLine, width=2)
        self.brush: QtGui.QBrush = pg.mkBrush(color=colour)
        self.colour: QtGui.QColor = QtGui.QColor(colour)

        # Set some pens for plotting
        self.kpen_dot: QtGui.QPen = pg.mkPen(color='k', style=QtCore.Qt.DotLine, width=2)
        self.kpen_solid: QtGui.QPen = pg.mkPen(color='k', style=QtCore.Qt.SolidLine, width=2)

        # Probe geometry
        self.probe_tip: int | float = 0
        self.probe_top: int | float = 3840
        self.view_total = [-2000, 6000]
        self.depth = np.arange(self.view_total[0], self.view_total[1], 20)

        # Axis limits and settings
        self.ylim_extra: int = 100
        self.yaxis_pad: float = 0.05
        self.yrange: list | np.ndarray = [self.probe_tip, self.probe_top]
        self.xrange: list | np.ndarray = [0, 1]
        self.yscale: float = 1
        self.ephys_plot: pg.PlotItem | None = None

        # Initialize plot items and reference arrays
        self.init_plot_items()
        self.init_reference_line_arrays()

    def init_plot_items(self) -> None:
        """Initialise all plot items and attributes used to keep track of plots."""
        # Horizontal lines indicating the top and tip of the electrodes / channels
        self.probe_top_lines: list[pg.InfiniteLine] = []
        self.probe_tip_lines: list[pg.InfiniteLine] = []

        # Plot items for the image and scatter plots
        self.fig_img: pg.PlotItem | None = None
        self.fig_img_cb: pg.PlotItem | None = None
        self.img_item: pg.ScatterPlotItem | pg.ImageItem | None = None
        self.img_cbar: ColorBar | None = None
        self.fig_data_ax: pg.AxisItem | None = None

        # Plot items for line plot
        self.fig_line: pg.PlotItem | None = None
        self.line_item: pg.PlotCurveItem | None = None

        # Plot items for the probe plot
        self.fig_probe: pg.PlotItem | None = None
        self.fig_probe_cb: pg.PlotItem | None = None
        self.probe_items: list[pg.ImageItem] = []
        self.probe_cbar: ColorBar | None = None
        self.probe_bounds: list[pg.InfiniteLine] = []

        # Plot items for the slice plot
        self.slice_lines: list[pg.PlotCurveItem] = []
        self.slice_plot: pg.ImageItem = None
        self.traj_line: pg.PlotCurveItem | None = None
        self.slice_chns: pg.ScatterPlotItem | None = None

        # Plot items for the fit plot
        self.fit_plot: pg.PlotCurveItem | None = None
        self.fit_scatter: pg.ScatterPlotItem | None = None
        self.fit_plot_line: pg.PlotCurveItem | None = None

        # Plot items for the scale factor plot
        self.fig_scale: pg.PlotItem | None = None
        self.fig_scale_cb: pg.PlotItem | None = None
        self.fig_scale_ax: pg.AxisItem | None = None
        self.scale_regions: list[pg.LinearRegionItem] = []

        # Plot items for the histology plot
        self.fig_hist: pg.PlotItem | None = None
        self.fig_hist_extra_yaxis: pg.AxisItem | None = None
        self.ax_hist: pg.AxisItem | None = None
        self.ax_hist2: pg.AxisItem | None = None
        self.hist_regions: list[pg.LinearRegionItem] = []

        # Plot items for the histology reference plot
        self.fig_hist_ref: pg.PlotItem | None = None
        self.ax_hist_ref: pg.AxisItem | None = None

        self.header: QtWidgets.QLabel = QtWidgets.QLabel(self.name)
        self.header.setAlignment(QtCore.Qt.AlignCenter)

        self.create_ephys_plots()
        self.create_histology_plots()
        self.create_slice_items()
        self.create_fit_items()

    def init_reference_line_arrays(self) -> None:
        """Initialise arrays used to keep track of reference lines and points."""
        self.lines_features: list[list] = []
        self.lines_tracks: list = []
        self.points: list = []

    # --------------------------------------------------------------------------------------------
    # Plot creation
    # --------------------------------------------------------------------------------------------
    @staticmethod
    def _create_plot_item(
            mouse_enabled: tuple[bool, bool] = (False, False),
            max_width: int | None = None,
            max_height: int | None = None,
            pen: str = 'k'
    ) -> pg.PlotItem:
        """
        Create and configure a pg.PlotItem used for a plot panel.

        Parameters
        ----------
        mouse_enabled : list of bools, default=False
            Whether mouse interaction (panning/zooming) is enabled for the x-axis and y-axis.
        max_width : int, optional
            Maximum width of the plot widget in pixels.
        max_height : int, optional
            Maximum height of the plot widget in pixels.
        pen: str, default='k'
            The colour pen to use for the x-axis

        Returns
        -------
        pg.PlotItem
            A configured plot item
        """
        plot = pg.PlotItem()
        plot.setMouseEnabled(*mouse_enabled)
        if max_width:
            plot.setMaximumWidth(max_width)
        if max_height:
            plot.setMaximumHeight(max_height)

        set_axis(plot, 'bottom', pen=pen)
        set_axis(plot, 'left', show=False)

        return plot

    @staticmethod
    def _create_plot_cb_item(
            max_width: int | None = None,
            max_height: int | None = None,
    ) -> pg.PlotItem:
        """
        Create and configure a pg.PlotItem used for a colorbar panel.

        Parameters
        ----------
        max_width : int, optional
            Maximum width of the plot widget in pixels.
        max_height : int, optional
            Maximum height of the plot widget in pixels.

        Returns
        -------
        pg.PlotItem
            A configured plot item
        """
        plot = pg.PlotItem()
        plot.setMouseEnabled(x=False, y=False)
        if max_width:
            plot.setMaximumWidth(max_width)
        if max_height:
            plot.setMaximumHeight(max_height)

        set_axis(plot, 'bottom', show=False)
        set_axis(plot, 'left', pen='w')
        set_axis(plot, 'top', pen='w')

        return plot

    @staticmethod
    def remove_items(fig, item, delete=True):
        """
        Remove all items from a plot item and optionally delete them.

        Parameters
        ----------
        fig: pg.PlotItem
            The plot item from which to remove items
        item: list[pg.GraphicsItem] or pg.GraphicsItem
            A list of items or a single item to remove
        delete: bool, default=True
            Whether the item should be deleted
        """
        if isinstance(item, list):
            for it in item:
                fig.removeItem(it)
                if delete:
                    del it
            return []
        elif item:
            fig.removeItem(item)
            if delete:
                del item

    def create_ephys_plots(self) -> None:
        """Create plots for the electrophysiology panels."""
        # 2D image / scatter plots
        self.fig_img = self._create_plot_item(mouse_enabled=(True, True))
        self.probe_tip_lines.append(
            self.fig_img.addLine(y=self.probe_tip, pen=self.kpen_dot, z=50))
        self.probe_top_lines.append(
            self.fig_img.addLine(y=self.probe_top, pen=self.kpen_dot, z=50))
        self.fig_data_ax = set_axis(self.fig_img, 'left', label='Distance from probe tip (uV)')
        self.fig_img_cb = self._create_plot_cb_item(max_height=70)

        # 1D line plot
        self.fig_line = self._create_plot_item(mouse_enabled=(False, True))
        self.probe_tip_lines.append(
            self.fig_line.addLine(y=self.probe_tip, pen=self.kpen_dot, z=50))
        self.probe_top_lines.append(
            self.fig_line.addLine(y=self.probe_top, pen=self.kpen_dot, z=50))
        self.fig_line.setYLink(self.fig_img)

        # 2D probe plot
        self.fig_probe = self._create_plot_item(mouse_enabled=(False, True), max_width=50, pen='w')
        self.probe_tip_lines.append(
            self.fig_probe.addLine(y=self.probe_tip, pen=self.kpen_dot, z=50))
        self.probe_top_lines.append(
            self.fig_probe.addLine(y=self.probe_top, pen=self.kpen_dot, z=50))
        self.fig_probe_cb = self._create_plot_cb_item(max_height=70)
        self.fig_probe.setYLink(self.fig_img)

    def create_histology_plots(self) -> None:
        """Create the plots the histology panels."""
        # Histology plot that updates with alignment
        self.fig_hist = self._create_plot_item(mouse_enabled=(False, True), pen='w')
        replace_axis(self.fig_hist)
        self.ax_hist = set_axis(self.fig_hist, 'left', pen=None)
        self.ax_hist.setWidth(0)
        self.ax_hist.setStyle(tickTextOffset=-60)

        # Scale factor plot
        self.fig_scale = self._create_plot_item(mouse_enabled=(False, True), max_width=50,
                                                pen='w')
        self.fig_scale.setYLink(self.fig_hist)
        self.fig_scale_cb = self._create_plot_cb_item(max_height=70)
        set_axis(self.fig_scale_cb, 'left', show=False)
        set_axis(self.fig_scale_cb, 'right', show=False)
        self.fig_scale_ax = set_axis(self.fig_scale_cb, 'top', pen='w')

        # Histology plot used as a reference
        self.fig_hist_ref = self._create_plot_item(mouse_enabled=(False, True), pen='w')
        replace_axis(self.fig_hist_ref, orientation='right', pos=(2, 2))
        self.ax_hist_ref = set_axis(self.fig_hist_ref, 'right', pen=None)
        self.ax_hist_ref.setWidth(0)
        self.ax_hist_ref.setStyle(tickTextOffset=-60)

        # Additional axis for exporting to png
        self.fig_hist_extra_yaxis = self._create_plot_item(max_width=2, pen='w')
        self.ax_hist2 = set_axis(self.fig_hist_extra_yaxis, 'left', pen=None)
        self.ax_hist2.setWidth(10)

    def create_slice_items(self) -> None:
        """Create the slice figure area to show the coronal slices and channels."""
        self.fig_slice_area = pg.GraphicsLayoutWidget(border=None)
        self.fig_slice_area.setContentsMargins(0, 0, 0, 0)
        self.fig_slice_area.ci.setContentsMargins(0, 0, 0, 0)
        self.fig_slice_area.ci.layout.setSpacing(0)
        self.fig_slice = pg.ViewBox(enableMenu=False)
        self.fig_slice.setContentsMargins(0, 0, 0, 0)
        self.fig_slice_area.addItem(self.fig_slice)

    def create_fit_items(self) -> None:
        """
        Create plot items to put on the fit figure.

        The actual fit PlotItem is stored in the app_view as it is shared across the
        different shanks.
        """
        self.fit_plot = pg.PlotCurveItem(pen=self.pen)
        self.fit_scatter = pg.ScatterPlotItem(size=7, symbol='o', brush='w', pen=self.pen)
        self.fit_plot_lin = pg.PlotCurveItem(pen=self.pen_dot)

    # --------------------------------------------------------------------------------------------
    # Plot functions
    # --------------------------------------------------------------------------------------------
    def clear_fit(self) -> None :
        """Clear the data from fit lines."""
        self.fit_plot.setData()
        self.fit_scatter.setData()
        self.fit_plot_lin.setData()

    def plot_fit(self, data: Bunch) -> None:
        """
        Plot data onto fit lines.

        Parameters
        ----------
        data: Bunch
            A Bunch object containing the fit data
        """
        self.clear_fit()

        if len(data.x) > 2:
            self.fit_plot.setData(x=data.x, y=data.y)
            self.fit_scatter.setData(x=data.x, y=data.y)

            if np.any(data.depth_lin):
                self.fit_plot_lin.setData(x=data.depth, y=data.depth_lin)
            else:
                self.fit_plot_lin.setData()

    def clear_histology(self, fig: pg.PlotItem):
        """Clear items from the histology plot."""
        fig.clear()
        self.hist_regions = []

    def plot_histology(self, fig: pg.PlotItem, data: Bunch, ax: str ='left') -> None:
        """
        Plot histology regions on the given figure.

        Shows brain regions intersecting with the probe track.

        Parameters
        ----------
        fig : pg.PlotItem
            The figure on which to plot the histology regions.
        data : Bunch
            A Bunch object containing the histology data.
        ax : str, default='left'
            Orientation of the axis on which to add labels. 'left' for the main histology
            figure (fig_hist), and 'right' for the reference figure (fig_hist_ref).
        """
        self.clear_histology(fig)

        # Axis configuration
        axis = fig.getAxis(ax)
        axis.setTicks([data.axis_label])
        axis.setZValue(10)
        set_axis(fig, 'bottom', pen='w', label='blank')

        # Plot regions and boundaries
        for colour, region in zip(data.colour, data.region, strict=False):
            item = pg.LinearRegionItem(
                values=region,
                orientation=pg.LinearRegionItem.Horizontal,
                brush=QtGui.QColor(*colour),
                movable=False)
            fig.addItem(item)
            fig.addItem(pg.InfiniteLine(pos=region[0], angle=0, pen='w'))
            # Keep track of each histology LinearRegionItem for hover interaction
            self.hist_regions.append(item)

        # Add additional boundary for final region
        fig.addItem(pg.InfiniteLine(pos=data.region[-1][1], angle=0, pen='w'))

        # Add probe limits as dotted lines
        fig.addItem(pg.InfiniteLine(pos=self.probe_tip, angle=0, pen=self.kpen_dot))
        fig.addItem(pg.InfiniteLine(pos=self.probe_top, angle=0, pen=self.kpen_dot))

        self.set_yaxis_range(fig)

    def clear_scale_factor(self):
        """Clear items from the scale factor plot."""
        self.fig_scale.clear()
        self.scale_regions = []

    def plot_scale_factor(self, data) -> ColorBar:
        """
        Plot the scale factor applied to brain regions alongside the histology figure.

        Parameters
        ----------
        data : Bunch
            A Bunch object containing the scaling data

        Returns
        -------
        cbar: ColorBar
            The created colorbar
        """
        self.clear_scale_factor()

        cbar = ColorBar('seismic', plot_item=self.fig_scale_cb)
        colours = cbar.cmap.mapToQColor(data.scale_factor)
        cbar.set_levels((0, 1.5), label='Scale')

        for ir, region in enumerate(data.region):
            item = pg.LinearRegionItem(
                values=region,
                orientation=pg.LinearRegionItem.Horizontal,
                brush=colours[ir],
                movable=False)
            self.fig_scale.addItem(item)
            self.fig_scale.addItem(pg.InfiniteLine(pos=region[0], angle=0, pen=colours[ir]))
            self.scale_regions.append(item)

        # Add additional boundary for final region
        self.fig_scale.addItem(pg.InfiniteLine(pos=data.region[-1][1], angle=0, pen=colours[-1]))

        self.set_yaxis_range(self.fig_scale)
        set_axis(self.fig_scale, 'bottom', pen='w', label='blank')

        return cbar

    def clear_slice(self):
        """Clear items from the slice plot."""
        self.slice_plot = self.remove_items(self.fig_slice, self.slice_plot)
        self.traj_line = self.remove_items(self.fig_slice, self.traj_line)

    def plot_slice(
            self,
            data: Bunch | None,
            data_traj: Bunch
    ) -> tuple[pg.ImageItem, ColorBar | None]:
        """
        Plot a slice image showing a coronal histology slice.

        Add a trajectory line showing the probe location through the slice.

        Parameters
        ----------
        data : Bunch
            A Bunch object containing the slice data
        data_traj: Bunch
            A Bunch object containing the trajectory data

        Returns
        -------
        pg.ImageItem:
            The created image item.
        ColorBar
            The created colorbar.
        """
        self.clear_slice()

        self.slice_plot = pg.ImageItem()
        if data is None:
            return self.slice_plot, None

        self.slice_plot.setImage(data.slice)
        self.slice_plot.setTransform(self.make_transform(data.scale, data.offset))

        label_img = data.get('label', False)
        if not label_img:
            color_bar = ColorBar('cividis')
            lut = color_bar.get_colour_map()
            self.slice_plot.setLookupTable(lut)
        else:
            color_bar = None

        self.fig_slice.addItem(self.slice_plot)
        self.fig_slice.autoRange()

        # Create a line showing the trajectory
        self.traj_line = pg.PlotCurveItem(x=data_traj.x, y=data_traj.y, pen=self.kpen_solid)
        self.fig_slice.addItem(self.traj_line)

        return self.slice_plot, color_bar

    def clear_channels(self, fig_slice: pg.ViewBox) -> None:
        """
        Clear channels and reference lines from the slice plot.

        Parameters
        ----------
        fig_slice: pg.ViewBox
            The fig slice to remove the channels and reference lines from
        """
        self.slice_lines = self.remove_items(fig_slice, self.slice_lines)
        self.slice_chns = self.remove_items(fig_slice, self.slice_chns)

    def plot_channels(self, fig_slice: pg.ViewBox, data: Bunch, colour: str = 'r') -> None:
        """
        Plot the locations of electrode channels and track reference lines on the histology slice.

        Note special case as the fig_slice may not come from the current item so it is passed in.

        Parameters
        ----------
        fig_slice : pg.ViewBox
            The fig slice to plot the channels and reference lines on
        data : Bunch
            A Bunch object containing the channels data
        colour : str
            The colour to use to plot the channels
        """
        self.clear_channels(fig_slice)

        self.slice_chns = pg.ScatterPlotItem(x=data['xyz_channels'][:, 0],
                                             y=data['xyz_channels'][:, 2],
                                             pen=colour, brush=colour)
        fig_slice.addItem(self.slice_chns)

        self.slice_lines = []
        for ref_line in data['track_lines']:
            line = pg.PlotCurveItem(x=ref_line[:, 0], y=ref_line[:, 2], pen=self.kpen_dot)
            fig_slice.addItem(line)
            self.slice_lines.append(line)

    def clear_scatter(self) -> None:
        """Clear items from the scatter/ image plot."""
        self.img_item = self.remove_items(self.fig_img, self.img_item)
        self.img_cbar = self.remove_items(self.fig_img_cb, self.img_cbar)

    def plot_scatter(
            self,
            data: ScatterData | None,
            levels: list | np.ndarray | None = None
    ) -> ColorBar | None:
        """
        Plot a 2D scatter plot of electrophysiology data.

        Parameters
        ----------
        data : ScatterData
            A ScatterData object containing the data to plot
        levels : list or np.ndarray, optional
            A list or array containing the levels to set for the colorbar.
            Defaults to data.levels

        Returns
        -------
        ColorBar
            The created colorbar
        """
        self.clear_scatter()

        if data is None:
            return self.plot_empty(self.fig_img, self.fig_img_cb, img=True)

        levels = data.levels if levels is None else levels

        self.img_cbar = ColorBar(data.cmap, plot_item=self.fig_img_cb)
        self.img_cbar.set_levels(levels, label=data.title)

        brush = data.colours if isinstance(data.colours[0], str) \
            else self.img_cbar.get_brush(data.colours, levels=list(levels))

        # Create scatter plot and add to figure
        self.img_item = pg.ScatterPlotItem(
            x=data.x,
            y=data.y,
            symbol=data.symbol.tolist(),
            size=data.size.tolist(),
            brush=brush,
            pen=data.pen)
        self.fig_img.addItem(self.img_item)

        set_axis(self.fig_img, 'bottom', pen='k', label=data.xaxis)

        self.set_xaxis_range(self.fig_img, data.xrange)
        self.set_yaxis_range(self.fig_img)

        self.ephys_plot = self.img_item
        self.y_scale = 1
        self.xrange = data.xrange

        return self.img_cbar

    def clear_line(self) -> None:
        """Clear items from the scatter plot."""
        self.line_item = self.remove_items(self.fig_line, self.line_item)

    def plot_line(self, data: LineData | None) -> None:
        """
        Plot a 1D line plot of electrophysiology data.

        Parameters
        ----------
        data : LineData
            A LineData object containing data to plot
        """
        self.clear_line()

        if data is None:
            return self.plot_empty(self.fig_line)

        self.line_item = pg.PlotCurveItem(x=data.x, y=data.y, pen=self.kpen_solid)
        self.fig_line.addItem(self.line_item)

        set_axis(self.fig_line, 'bottom', pen='k')
        set_axis(self.fig_line, 'bottom', label=data.xaxis)

        self.set_xaxis_range(self.fig_line, data.xrange)
        self.set_yaxis_range(self.fig_line)

    def clear_probe(self) -> None:
        """Clear items from the probe plot."""
        self.probe_items = self.remove_items(self.fig_probe, self.probe_items)
        self.probe_cbar = self.remove_items(self.fig_probe_cb, self.probe_cbar)
        self.probe_bounds = self.remove_items(self.fig_probe, self.probe_bounds)


    def plot_probe(
            self,
            data: ProbeData | None,
            levels: list | np.ndarray | None = None
    ) -> ColorBar | None:
        """
        Plot a 2D probe plot of electrophysiology data.

        Parameters
        ----------
        data : ProbeData
            A ProbeData object containing data to plot
        levels : list or np.ndarray, optional
            A list or array containing the levels to set for the colorbar.
            Defaults to data.levels

        Returns
        -------
        ColorBar
            The created colorbar
        """
        self.clear_probe()

        if data is None:
            return self.plot_empty(self.fig_probe, self.fig_probe_cb)

        levels = data.levels if levels is None else levels

        self.plot_cbar = ColorBar(data.cmap, plot_item=self.fig_probe_cb)
        self.plot_cbar.set_levels(levels, label=data.title)

        # Create image plots per shank and add to figure
        self.probe_items = []
        for img, scale, offset in zip(data.img, data.scale, data.offset, strict=False):
            image = pg.ImageItem()
            image.setImage(img)
            image.setTransform(self.make_transform(scale, offset))
            image.setLookupTable(self.plot_cbar.get_colour_map())
            image.setLevels((levels[0], levels[1]))
            self.fig_probe.addItem(image)
            self.probe_items.append(image)

        # Add in a fake label so that the appearance is the same as other plots
        set_axis(self.fig_probe, 'bottom', pen='w', label='blank')

        self.set_xaxis_range(self.fig_probe, data.xrange)
        self.set_yaxis_range(self.fig_probe)

        # Optionally plot horizontal boundary lines
        self.probe_bounds = []
        if data.boundaries is not None:
            for bound in data.boundaries:
                line = pg.InfiniteLine(pos=bound, angle=0, pen='w')
                self.fig_probe.addItem(line)
                self.probe_bounds.append(line)

        return self.plot_cbar

    def plot_image(
            self,
            data: ImageData | None,
            levels: list | np.ndarray | None = None
    ) -> ColorBar | None:
        """
        Plot a 2D image plot of electrophysiology data.

        Parameters
        ----------
        data : ImageData
            An ImageData object containing data to plot
        levels : list or np.ndarray, optional
            A list or array containing the levels to set for the colorbar.
            Defaults to data.levels

        Returns
        -------
        ColorBar
            The created colorbar
        """
        self.clear_scatter()

        if data is None:
            return self.plot_empty(self.fig_img, self.fig_img_cb, img=True)

        levels = data.levels if levels is None else levels

        self.img_item = pg.ImageItem()
        self.img_item.setImage(data.img)
        self.img_item.setTransform(self.make_transform(data.scale, data.offset))
        self.fig_img.addItem(self.img_item)

        if data.cmap:
            self.img_cbar = ColorBar(data.cmap, plot_item=self.fig_img_cb)
            self.img_item.setLookupTable(self.img_cbar.get_colour_map())
            self.img_item.setLevels((levels[0], levels[1]))
            self.img_cbar.set_levels(levels, label=data.title)
        else:
            self.img_item.setLevels((1, 0))
            self.img_cbar = None

        set_axis(self.fig_img, 'bottom', pen='k', label=data.xaxis)

        self.set_xaxis_range(self.fig_img, data.xrange)
        self.set_yaxis_range(self.fig_img)

        self.ephys_plot = self.img_item
        self.y_scale = data.scale[1]
        self.xrange = data.xrange

        return self.img_cbar

    # --------------------------------------------------------------------------------------------
    # Plot utils
    # --------------------------------------------------------------------------------------------
    def plot_empty(
            self,
            fig: pg.PlotItem,
            fig_cb: pg.PlotItem | None = None,
            img: bool = False
    ) -> None:
        """
        Create an empty placeholder plot when no data is available.

        Parameters
        ----------
        fig: pg.PlotItem
            The figure to display empty data
        fig_cb: pg.PlotItem
            An optional colourbar to reset
        img: bool
            Whether the figure is an image plot or not
        """
        self.set_xaxis_range(fig, [0, 1])
        self.set_yaxis_range(fig)
        set_axis(fig, 'bottom', pen='w', label='blank')
        if fig_cb:
            set_axis(fig_cb, 'top', pen='w')
        if img:
            self.ephys_plot = None
            self.y_scale = 1
            self.xrange = [0, 1]

    def set_yaxis_range(self, fig: pg.PlotItem) -> None:
        """
        Set the y-axis range of a given figure.

        Parameters
        ----------
        fig: pg.PlotItem
            The figure whose y-axis range will be updated
        """
        fig.setYRange(min=self.yrange[0] - self.ylim_extra, max=self.yrange[1] + self.ylim_extra,
                      padding=self.yaxis_pad)

    def set_xaxis_range(self, fig: pg.PlotItem, xrange: np.ndarray | list | None = None) -> None:
        """
        Set the x-axis range of a given figure.

        Parameters
        ----------
        fig: pg.PlotItem
            The figure whose x-axis range will be updated
        xrange: list, optional
            The xrange values to use. If None, the default values are used.
        """
        xrange = xrange if xrange is not None else self.xrange
        fig.setXRange(*xrange, padding=0)

    @staticmethod
    def make_transform(scale: list | np.ndarray, offset: list | np.ndarray) -> QtGui.QTransform:
        """
        Create a Qt transform matrix based on scaling and offset values.

        Parameters
        ----------
        scale : list or np.ndarray
            The x-scale and y-scale factors to apply.
        offset : list or np.ndarray
            The x-offset and y-offset translations to apply.

        Returns
        -------
        QtGui.QTransform
            The constructed transformation matrix.
        """
        return QtGui.QTransform(scale[0], 0., 0., 0., scale[1], 0., offset[0], offset[1], 1.)

    def reset_slice_axis(self) -> None:
        """Reset the axis range of the slice image."""
        self.fig_slice.autoRange()

    # --------------------------------------------------------------------------------------------
    # Update displays
    # --------------------------------------------------------------------------------------------

    def toggle_labels(self, show: bool) -> None:
        """
        Show/hide the brain region axis labels on the histology plot.

        Parameters
        ----------
        show: bool
            Whether to show the labels or not.
        """
        pen = 'k' if show else None
        for ax in [self.ax_hist, self.ax_hist_ref]:
            ax.setPen(pen)
            ax.setTextPen(pen)
        for fig in [self.fig_hist, self.fig_hist_ref]:
            fig.update()

    def toggle_channels(self, fig_slice: pg.ViewBox, show: bool) -> None:
        """
        Show/hide the channels and traj line on the slice plot.

        Parameters
        ----------
        fig_slice: pg.ViewBox
            The fig slice to add or remove the items
        show: bool
            Whether to show the channels and traj line or not
        """
        func = fig_slice.addItem if show else fig_slice.removeItem
        if self.traj_line:
            func(self.traj_line)
        func(self.slice_chns)
        for line in self.slice_lines:
            func(line)

    def set_header_style(self, selected : bool) -> None:
        """
        Set the stylesheet of the header item.

        Update style based on whether the current shank is selected or not.

        Parameters
        ----------
        selected: bool
            Whether the current shank is selected or not
        """
        if selected:
            self.header.setStyleSheet(self.HEADER_STYLE['selected'])
        else:
            self.header.setStyleSheet(self.HEADER_STYLE['deselected'])

    def set_probe_lims(self, min_val: float, max_val: float) -> None:
        """
        Set the values of the probe tip and probe top.

        Update all the associated lines showing the new probe extent.

        Parameters
        ----------
        min_val: float
            The value for probe tip
        max_val: float
            The value for the probe top
        """
        self.probe_tip = min_val
        self.probe_top = max_val
        for top_line in self.probe_top_lines:
            top_line.setY(self.probe_top)
        for tip_line in self.probe_tip_lines:
            tip_line.setY(self.probe_tip)

    def set_yaxis_lims(self, min_val: float, max_val: float) -> None:
        """
        Set the yrange values that are used to set the y-axis limits used to display plots.

        Parameters
        ----------
        min_val: float
            The minimum y-axis value
        max_val: float
            The maximum y-axis value
        """
        self.yrange = [min_val, max_val]

    def set_fig_scale_title(self, value: float) -> None:
        """
        Update the label of the scale plot axis to display the current scale value.

        Parameters
        ----------
        value : float
            The scale factor to display in the axis label. The value is rounded
            to two decimal places before updating the label.
        """
        self.fig_scale_ax.setLabel('Scale = ' + str(np.around(value, 2)))


    def match_linear_region(self, hover_item: pg.LinearRegionItem) -> int | None:
        """
        Find the index of a hovered linear region within the list of scale regions.

        Parameters
        ----------
        hover_item : pg.LinearRegionItem
            The region item currently hovered over.

        Returns
        -------
        region_idx : int
            The index of the hovered region in the scale regions list.
        """
        try:
            region_idx = self.scale_regions.index(hover_item)
        except ValueError:
            region_idx = None
        return region_idx

    # --------------------------------------------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------------------------------------------
    def get_feature_and_track_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the values of the track and feature reference lines.

        Returns
        -------
        line_track: np.ndarray
            An array containing the positions of the track reference lines
        line_feature: np.ndarray
            An array containing the positions of the track reference lines
        """
        line_feature = np.array([line[0].pos().y() for line in self.lines_features]) / 1e6
        line_track = np.array([line.pos().y() for line in self.lines_tracks]) / 1e6
        return line_feature, line_track

    def match_feature_line(
            self,
            feature_line: pg.InfiniteLine
    ) -> tuple[int | None, list | np.ndarray | None]:
        """
        Find the index of the feature reference line matching the given line.

        Also find the indices of the feature plots that this line does not belong to.

        Parameters
        ----------
        feature_line: pg.InfiniteLine
            The feature line to match

        Returns
        -------
        line_idx: int or None
            The index of the matching feature line or None if not found
        fig_idx: np.ndarray or None
            An array containing the indices of the other plots, or None if not found.
        """
        idx = np.where(np.array(self.lines_features) == feature_line)
        if idx[0].size == 0:
            return None, None
        line_idx = idx[0][0]
        fig_idx = np.setdiff1d(np.arange(0, 3), idx[1][0]) # indices of two other plots
        return line_idx, fig_idx

    def match_track_line(self, track_line: pg.InfiniteLine) -> int | None:
        """
        Find the index of the track reference line matching the given line.

        Parameters
        ----------
        track_line: pg.InfiniteLine
            The track line to match

        Returns
        -------
        line_idx: int or None
            The index of the matching track line or None if not found
        """
        try:
            line_idx = self.lines_tracks.index(track_line)
        except ValueError:
            line_idx = None

        return line_idx

    def create_reference_line_and_point(
            self,
            pos,
            fix_colour=False
    ) -> tuple[pg.InfiniteLine, list[pg.InfiniteLine], pg.PlotDataItem]:
        """
        Create a new reference line.

        Creates a feature reference line on the line, image and probe figures, a track reference
        line on the histology figure and a scatter point to be added to the fit figure

        Parameters
        ----------
        pos : float
            Y-axis position at which to draw the horizontal line.
        fix_colour: bool
            Whether to use a fixed colour for the reference line or choose a random color

        Returns
        -------
        line_track: pg.InfiniteLine
            The track reference line
        line_feature: list[pg.InfiniteLine]
            The feature reference lines
        point: pg.PlotDataItem
            The scatter point
        """
        colour = self.colour if fix_colour else None
        pen, brush = self.create_line_style(colour=colour)

        # Reference line on histology figure (track)
        line_track = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
        line_track.setZValue(100)
        self.fig_hist.addItem(line_track)
        self.lines_tracks.append(line_track)

        # Reference lines on image, line and probe figures (feature)
        line_features = []
        for fig in [self.fig_img, self.fig_line, self.fig_probe]:
            line_feature = pg.InfiniteLine(pos=pos, angle=0, pen=pen, movable=True)
            line_feature.setZValue(100)
            fig.addItem(line_feature)
            line_features.append(line_feature)
        self.lines_features.append(line_features)

        # Scatter point to be added to fit figure
        point= pg.PlotDataItem(x=[line_track.pos().y()], y=[line_features[0].pos().y()],
                               symbolBrush=brush, symbol='o', symbolSize=10)
        self.points.append(point)

        return line_track, line_features, point

    @staticmethod
    def create_line_style(
            colour: QtGui.QColor | None =None
    ) -> tuple[QtGui.QPen, QtGui.QBrush]:
        """
        Generate a random line style (color and dash style) for reference lines.

        If the colour is given this is used.

        Parameters
        ----------
        colour: QtGui.QColor, optional
            The colour to use for the line. If None, a random colour is chosen.

        Returns
        -------
        pen : QtGui.QPen
            A pen object defining the line color, dash style, and width.
        brush : QtGui.QBrush
            A brush object with the same color as the pen for use with filled items.
        """
        colours = ['#000000', '#cc0000', '#6aa84f', '#1155cc', '#a64d79']
        styles = [QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DashDotLine]

        colour = colour or QtGui.QColor(random.choice(colours))
        style = random.choice(styles)

        pen = pg.mkPen(color=colour, style=style, width=3)
        brush = pg.mkBrush(color=colour)

        return pen, brush

    def remove_reference_line(self, line_idx: int) -> None:
        """
        Remove a reference line (track and feature) from the displays.

        Parameters
        ----------
        line_idx: int
            The index of the reference line to remove
        """
        self.fig_img.removeItem(self.lines_features[line_idx][0])
        self.fig_line.removeItem(self.lines_features[line_idx][1])
        self.fig_probe.removeItem(self.lines_features[line_idx][2])
        self.fig_hist.removeItem(self.lines_tracks[line_idx])

    def delete_reference_line_and_point(self, line_idx: int) -> None:
        """
        Delete a reference line (track, feature and point) from the tracking arrays.

        Parameters
        ----------
        line_idx: int
            The index of the reference line to remove.
        """
        _ = self.lines_features.pop(line_idx)
        _ = self.lines_tracks.pop(line_idx)
        _ = self.points.pop(line_idx)

    def update_feature_reference_line_and_point(
            self,
            feature_line: pg.InfiniteLine,
            line_idx: int,
            fig_idx: list | np.ndarray
    ) -> None:
        """
        Update the feature lines to match the coordinate of the moved feature line.

        Also update the scatter point location.

        Parameters
        ----------
        feature_line: pyqtgraph.InfiniteLine
            The feature line instance that was moved by the user.
        line_idx:
            The index of the reference line in the tracking arrays.
        fig_idx: list
            The index of the figures where the feature line position needs to be updated.
        """
        self.lines_features[line_idx][fig_idx[0]].setPos(feature_line.value())
        self.lines_features[line_idx][fig_idx[1]].setPos(feature_line.value())
        self.points[line_idx].setData(x=[self.lines_features[line_idx][0].pos().y()],
                                      y=[self.lines_tracks[line_idx].pos().y()])

    def update_track_reference_line_and_point(
            self,
            track_line: pg.InfiniteLine,
            line_idx: int
    ) -> None:
        """
        Update the scatter point location to match the coordinate of the moved track line.

        Parameters
        ----------
        track_line : pg.InfiniteLine
            The track line instance that was moved by the user.
        line_idx: int
            The index of the reference line in the tracking arrays.
        """
        self.lines_tracks[line_idx].setPos(track_line.value())
        self.points[line_idx].setData(x=[self.lines_features[line_idx][0].pos().y()],
                                      y=[self.lines_tracks[line_idx].pos().y()])

    def align_reference_lines_and_points(self) -> None:
        """
        Align the position of the track reference lines and scatter points.

        The position is updated based on the new positions of their corresponding
        feature reference lines.
        """
        for line_feature, line_track, point in (
                zip(self.lines_features, self.lines_tracks, self.points, strict=False)):
            line_track.setPos(line_feature[0].getYPos())
            point.setData(x=[line_feature[0].pos().y()], y=[line_feature[0].pos().y()])

    def remove_reference_lines_from_display(self) -> None:
        """Remove all reference lines from the respective plots."""
        for line_feature, line_track in zip(self.lines_features, self.lines_tracks, strict=False):
            self.fig_img.removeItem(line_feature[0])
            self.fig_line.removeItem(line_feature[1])
            self.fig_probe.removeItem(line_feature[2])
            self.fig_hist.removeItem(line_track)

    def add_reference_lines_to_display(self) -> None:
        """Add all reference lines to the respective plots."""
        for line_feature, line_track in zip(self.lines_features, self.lines_tracks, strict=False):
            self.fig_img.addItem(line_feature[0])
            self.fig_line.addItem(line_feature[1])
            self.fig_probe.addItem(line_feature[2])
            self.fig_hist.addItem(line_track)
