import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from ibl_alignment_gui.app.shank_view import ShankView
from ibl_alignment_gui.loaders.geometry_loader import ChannelGeometry, arrange_channels_into_banks
from ibl_alignment_gui.utils.qt.adapted_axis import replace_axis
from ibl_alignment_gui.utils.qt.custom_widgets import (
    ColorBar,
    PopupWindow,
    SelectionWidget,
    SliderWidget,
    set_axis,
)
from ibllib.pipes.ephys_alignment import EphysAlignment
from iblutil.util import Bunch
from one.remote import aws

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController

PLUGIN_NAME = 'Ephys Features'


def setup(controller: 'AlignmentGUIController') -> None:
    """
    Set up the ephys features plugin in the controller.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = Bunch()
    controller.plugins[PLUGIN_NAME]['loader'] = EphysFeatures('Ephys features', controller)

    action = QtWidgets.QAction(PLUGIN_NAME, controller.view)
    action.triggered.connect(lambda: callback(controller))
    controller.plugin_options.addAction(action)


def callback(controller: 'AlignmentGUIController') -> None:
    """
    Open the ephys features viewer.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME]['activated'] = True
    controller.plugins[PLUGIN_NAME]['loader'].setup()


CMAPS = [
    'viridis',
    'magma',
    'YlGn',
    'Reds',
    'Purples',
    'plasma',
    'hot',
    'inferno',
    'coolwarm',
]


class EphysFeatureView(PopupWindow):
    """
    A view for displaying ephys features within regions.

    Parameters
    ----------
    title: str
        The title of the popup window
    controller: AlignmentGUIController
        The main application controller.
    step: int
        The number of probes to show per page.

    Attributes
    ----------
    plots_hist: list
        A list of plot items for the histology regions.
    plots_feat: list
        A list of plot items for the ephys features.
    plots_cbar: list
        A list of plot items for the colorbars.
    fig_areas: list
        A list of graphics layout widgets for each probe.
    step: int
        The number of probes to show per page.
    controller: AlignmentGUIController
        The main application controller.
    """

    def __init__(self, title: str, controller: 'AlignmentGUIController', step: int = 10):
        # Initialise plot variables
        self.plots_hist: list = list()
        self.plots_feat: list = list()
        self.plots_cbar: list = list()
        self.fig_areas: list = list()
        self.step: int = step
        self.controller: AlignmentGUIController = controller

        super().__init__(title, controller.view, size=(1600, 800), graphics=False)

    def setup(self) -> None:
        """Add widgets to the main layout."""
        # Widget for dropdowns and pagination
        selection_widget = QtWidgets.QWidget()
        selection_layout = QtWidgets.QHBoxLayout()
        selection_widget.setLayout(selection_layout)

        # Dropdowns for selecting region, plot and colormap
        self.region_list, self.region_combobox, *_ = SelectionWidget.create_combobox()
        self.plot_list, self.plot_combobox, *_ = SelectionWidget.create_combobox()
        self.cmap_list, self.cmap_combobox, *_ = SelectionWidget.create_combobox()

        # Buttons and label for pagination
        self.next_button = QtWidgets.QPushButton('Next')
        self.prev_button = QtWidgets.QPushButton('Previous')
        self.page_label = QtWidgets.QLabel('1/1')

        # Add widgets to layout
        selection_layout.addWidget(self.region_combobox)
        selection_layout.addWidget(self.plot_combobox)
        selection_layout.addWidget(self.cmap_combobox)
        selection_layout.addItem(QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding))
        selection_layout.addWidget(self.prev_button)
        selection_layout.addWidget(self.next_button)
        selection_layout.addWidget(self.page_label)

        # Widget for plots
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QHBoxLayout()
        plot_widget.setLayout(plot_layout)

        for _ in range(self.step):
            fig_area, fig_hist, fig_feat, fig_cbar = self.create_figures()
            self.fig_areas.append(fig_area)
            self.plots_hist.append(fig_hist)
            self.plots_feat.append(fig_feat)
            self.plots_cbar.append(fig_cbar)
            plot_layout.addWidget(fig_area)

        # Widget for colorbar and adjusting plots
        scale_widget = QtWidgets.QWidget()
        scale_layout = QtWidgets.QGridLayout()
        scale_widget.setLayout(scale_layout)

        self.align_button = QtWidgets.QCheckBox('Align Plots')
        self.align_button.setChecked(True)
        self.pid_label = QtWidgets.QLabel()
        self.pid_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.normalise_button = QtWidgets.QCheckBox('Normalise Plots')
        self.normalise_button.setChecked(False)
        self.slider = SliderWidget()

        scale_layout.addWidget(self.align_button, 0, 0, 1, 1)
        scale_layout.addWidget(self.normalise_button, 1, 0, 1, 1)
        scale_layout.addWidget(self.pid_label, 2, 0, 1, 1)
        scale_layout.addItem(
            QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding), 0, 1, 1, 3
        )
        scale_layout.addWidget(self.slider, 0, 2, 3, 10)

        # Add widgets to main layout
        self.layout.addWidget(selection_widget)
        self.layout.addWidget(plot_widget)
        self.layout.addWidget(scale_widget)

    @staticmethod
    def create_figures() -> tuple[pg.GraphicsLayoutWidget, pg.PlotItem, pg.PlotItem, pg.PlotItem]:
        """Create the plot items for each probe."""
        fig_area = pg.GraphicsLayoutWidget()
        fig_area.setMouseTracking(True)

        fig_layout = pg.GraphicsLayout()
        fig_area.addItem(fig_layout)

        fig_hist = ShankView._create_plot_item(mouse_enabled=(False, True), pen='w')
        replace_axis(fig_hist)
        set_axis(fig_hist, 'bottom', pen=None, label='')
        ax_hist = set_axis(fig_hist, 'left', pen='k')
        ax_hist.setWidth(0)
        ax_hist.setStyle(tickTextOffset=-30)

        fig_feat = ShankView._create_plot_item(mouse_enabled=(False, True), pen='w')

        fig_cbar = ShankView._create_plot_cb_item(max_height=70)

        fig_layout.addItem(fig_cbar, 0, 0, 1, 2)
        fig_layout.addItem(fig_hist, 1, 0, 1, 1)
        fig_layout.addItem(fig_feat, 1, 1, 1, 1)
        fig_layout.layout.setRowStretchFactor(0, 1)
        fig_layout.layout.setRowStretchFactor(1, 10)

        return fig_area, fig_hist, fig_feat, fig_cbar

    @property
    def normalised(self) -> bool:
        """Whether the plots are normalised."""
        return self.normalised_button.isChecked()

    @property
    def aligned(self) -> bool:
        """Whether the plots are aligned."""
        return self.align_button.isChecked()

    def clear_plots(self) -> None:
        """Clear all plots."""
        for fig_hist, fig_feat, fig_cbar in zip(
            self.plots_hist, self.plots_feat, self.plots_cbar, strict=False
        ):
            fig_hist.clear()
            fig_feat.clear()
            fig_cbar.clear()
            set_axis(fig_cbar, 'top', pen='w')

    def plot_probe(self, idx: int, data: dict, offset: float, levels: list | None = None) -> None:
        """
        Plot the feature data for a single probe.

        Parameters
        ----------
        idx: int
            The index of the probe to plot.
        data: dict
            The feature data for the probe.
        offset: float
            The offset to apply to the y-axis.
        levels: list, optional
            The color levels to use for the plot.
        """
        fig_probe = self.plots_feat[idx]
        fig_cbar = self.plots_cbar[idx]

        levels = levels if levels is not None else data['levels']

        color_bar = ColorBar(self.cmap, plot_item=fig_cbar)
        color_bar.set_levels(levels)
        image = pg.ImageItem()
        image.setImage(data['img'])
        transform = [
            data['scale'][0],
            0.0,
            0.0,
            0.0,
            data['scale'][1],
            0.0,
            data['offset'][0],
            data['offset'][1] - offset,
            1.0,
        ]
        image.setTransform(QtGui.QTransform(*transform))
        image.setLookupTable(color_bar.get_colour_map())
        image.setLevels((levels[0], levels[1]))
        fig_probe.addItem(image)

        fig_probe.setXRange(min=data['xrange'][0], max=data['xrange'][1], padding=0)
        if self.aligned:
            fig_probe.setYRange(min=-1000, max=1000)
        else:
            fig_probe.setYRange(min=0, max=3840)

        set_axis(fig_probe, 'bottom', pen=None, label='')

    def plot_region(self, idx: int, data: dict, offset: float, selected_region: int) -> None:
        """
        Plot the histology regions for a single probe.

        Parameters
        ----------
        idx: int
            The index of the probe to plot.
        data: dict
            The region data for the probe.
        offset: float
            The offset to apply to the y-axis.
        selected_region: int
            The id of the currently selected region.
        """
        fig = self.plots_hist[idx]
        fig.clear()
        axis = fig.getAxis('left')
        axis.setTicks([])

        labels = copy.copy(data['labels'])
        labels[:, 0] = labels[:, 0] - offset

        axis = fig.getAxis('left')
        axis.setTicks([labels])
        axis.setZValue(10)
        axis.setPen('k')

        # Plot each histology region
        for reg, col, reg_id in zip(
            data['regions'], data['colors'], data['region_id'], strict=False
        ):
            colour = QtGui.QColor(*col)
            if reg_id == selected_region:
                colour.setAlpha(255)
            else:
                colour.setAlpha(60)
            region = pg.LinearRegionItem(
                values=(reg[0] - offset, reg[1] - offset),
                orientation=pg.LinearRegionItem.Horizontal,
                brush=colour,
                movable=False,
            )
            # Add a white line at the boundary between regions
            bound = pg.InfiniteLine(pos=reg[0] - offset, angle=0, pen='w')
            fig.addItem(region)
            fig.addItem(bound)

        if self.aligned:
            fig.setYRange(min=-1000, max=1000)
        else:
            fig.setYRange(min=0, max=3840)


class EphysFeatures:
    """
    A plugin for visualising ephys features within brain regions.

    Parameters
    ----------
    title: str
        The title of the popup window.
    controller: AlignmentGUIController
        The main application controller.

    Attributes
    ----------
    page_num: int
        The total number of pages.
    page_idx: int
        The current page index.
    step: int
        The number of probes to show per page.
    max_idx: int
        The maximum index of probes.
    levels: list
        The current color levels.
    max_levels: list
        The maximum color levels across all probes within the region.
    plot_name: str
        The name of the feature to plot.
    plot_idx: int
        The index of the feature to plot.
    cmap: str
        The name of the colormap to use.
    chosen_region: int
        The id of the currently selected region.
    pids: list
        The list of probe ids that pass through the chosen region.
    current_pids: list
        The list of probe ids for the current page.
    selected_pid: str
        The currently selected probe id.
    selected_idx: int
        The index of the currently selected probe within the current page.
    feature_data: Bunch
        A bunch containing the feature data for each pid.
    region_data: Bunch
        A bunch containing the region data for each pid.
    offset_data: Bunch
        A bunch containing the offset data for each pid.
    """

    def __init__(self, title: str, controller: 'AlignmentGUIController'):
        self.controller = controller
        self.title = title
        # Initialise pagination variables
        self.page_num: int = 0
        self.page_idx: int = 0
        self.step: int = 10
        self.max_idx: int = 0

        # Initialize slider variables
        self.levels: list = [0, 100]
        self.max_levels: list = [0, 100]

        # Initialise selection variables
        # Plot
        self.plot_name: str = 'psd_delta'
        self.plot_idx: int = 0
        # Colormap
        self.cmap = 'viridis'
        # Region
        self.chosen_region = None
        # Probes
        self.pids = []
        self.current_pids = []
        self.selected_pid = None
        self.selected_idx = 0

        # Initialise data variables
        self.feature_data = Bunch()
        self.region_data = Bunch()
        self.offset_data = Bunch()

    def setup(self) -> None:
        """Set up the ephys features viewer and connect signals."""
        self.view = EphysFeatureView._get_or_create(self.title, self.controller, step=self.step)
        self.view.closed.connect(self.on_close)

        self.view.region_combobox.activated.connect(self.on_region_chosen)
        self.view.plot_combobox.activated.connect(self.on_plot_chosen)
        self.view.cmap_combobox.activated.connect(self.on_cmap_chosen)
        self.view.next_button.clicked.connect(self.on_next_page_pressed)
        self.view.prev_button.clicked.connect(self.on_prev_page_pressed)

        for i, fig_area in enumerate(self.view.fig_areas):
            fig_area.scene().sigMouseClicked.connect(
                lambda event, idx=i: self.on_area_clicked(event, idx)
            )

        self.view.align_button.clicked.connect(self.on_align_plots)
        self.view.slider.reset.connect(self.on_reset_levels)
        self.view.normalise_button.clicked.connect(self.on_normalise_plots)
        self.view.slider.released.connect(self.on_slider_released)

        self.one = self.controller.model.one
        self.ba = self.controller.model.brain_atlas

        self.region_ids = self.get_regions(self.controller)
        self.data = self.get_features()

        # Populate region combobox
        acronyms = self.ba.regions.id2acronym(self.region_ids)
        SelectionWidget.populate_combobox(
            acronyms, self.view.region_list, self.view.region_combobox
        )

        # Populate plot combobox
        ignore_cols = [
            'pid',
            'axial_um',
            'lateral_um',
            'x',
            'y',
            'z',
            'acronym',
            'atlas_id',
            'x_target',
            'y_target',
            'z_target',
            'outside',
            'Allen_id',
            'Cosmos_id',
            'Beryl_id',
            'alpha_mean',
            'alpha_std',
        ]
        self.features = [k for k in self.data if k not in ignore_cols]
        self.features.sort()
        SelectionWidget.populate_combobox(
            self.features, self.view.plot_list, self.view.plot_combobox
        )

        # Populate colormap combobox
        SelectionWidget.populate_combobox(CMAPS, self.view.cmap_list, self.view.cmap_combobox)
        self.view.cmap = self.cmap

    def on_close(self) -> None:
        """
        Triggered when the plugin window is closed.

        Deactivate the plugin and callbacks.
        """
        self.controller.plugins[PLUGIN_NAME]['activated'] = False

    @property
    def normalised(self) -> bool:
        """Whether the plots are normalised."""
        return self.view.normalise_button.isChecked()

    @property
    def aligned(self) -> bool:
        """Whether the plots are aligned."""
        return self.view.align_button.isChecked()

    # -------------------------------------------------------------------------
    # Download and prepare data
    # -------------------------------------------------------------------------
    def get_features(self) -> pd.DataFrame:
        """
        Download ephys atlas features table from S3.

        Returns
        -------
        pd.DataFrame
            The ephys atlas features table.
        """
        # Create folder to store the features table
        table_path = self.one.cache_dir.joinpath('ephys_atlas_features')
        table_path.mkdir(parents=True, exist_ok=True)
        s3, bucket_name = aws.get_s3_from_alyx(alyx=self.one.alyx)
        # Download file
        base_path = Path('aggregates/atlas/features/ea_active/2025_W43/agg_full/')
        fname = 'df_all_cols_merged.pqt'
        aws.s3_download_file(
            base_path.joinpath(fname), table_path.joinpath(fname), s3=s3, bucket_name=bucket_name
        )

        data = pd.read_parquet(table_path.joinpath('df_all_cols_merged.pqt')).reset_index()
        return data

    def get_regions(self, controller: 'AlignmentGUIController') -> np.ndarray:
        """
        Get all unique Allen region ids for the shanks in the controller.

        Parameters
        ----------
        controller: AlignmentGUIController
            The main application controller.

        Returns
        -------
        np.ndarray
            An array of unique Allen region ids.
        """
        all_regions = np.array([])
        for shank in controller.all_shanks:
            regions = controller.model.shanks[shank][
                controller.model.default_config
            ].align_handle.ephysalign.region_id
            all_regions = np.r_[all_regions, np.array(regions.ravel())]

        return np.unique(all_regions)

    def load_data(self, pids: list | np.ndarray) -> None:
        """
        Prepare feature, region and offset data for a list of probe ids.

        Will only prepare data for pids that are not already in the respective data dictionaries.

        Parameters
        ----------
        pids: list
            The list of pids to prepare data for.
        """
        feature_data = self.feature_data.get(self.plot_name, None)
        if feature_data is None:
            self.feature_data[self.plot_name] = {}

        missing_feature_pids = [p for p in pids if p not in self.feature_data[self.plot_name]]
        if len(missing_feature_pids) > 0:
            self.feature_data[self.plot_name].update(
                self.prepare_feature_data(missing_feature_pids)
            )

        missing_region_pids = [p for p in pids if p not in self.region_data]
        if len(missing_region_pids) > 0:
            self.region_data.update(self.prepare_region_data(missing_region_pids))

        missing_offset_pids = [p for p in pids if p not in self.offset_data]
        if len(missing_offset_pids) > 0:
            self.offset_data.update(self.prepare_offset_data(missing_offset_pids))

    def prepare_feature_data(self, pids: list | np.ndarray) -> Bunch:
        """
        Prepare feature data for a list of probe ids.

        Parameters
        ----------
        pids: list
            The list of pids to prepare data for.

        Returns
        -------
        feature_data: Bunch
            A bunch containing the feature data for each pid.
        """
        feature_data = Bunch()

        for pid in pids:
            df = self.data[self.data['pid'] == pid]
            data = df[self.plot_name].values
            chn_coords = Bunch()
            chn_coords['localCoordinates'] = np.c_[df['lateral_um'].values, df['axial_um'].values]
            chn_coords['rawInd'] = np.arange(chn_coords['localCoordinates'].shape[0])
            chn_geom = ChannelGeometry(chn_coords)
            chn_geom.split_sites_per_shank()
            sites = chn_geom._get_sites_for_shank(0)

            bnk_width = 10
            probe_img, probe_scale, probe_offset = arrange_channels_into_banks(sites, data)
            probe_levels = np.nanquantile(data, [0.1, 0.9])

            data_dict = {
                'img': probe_img,
                'scale': probe_scale,
                'offset': probe_offset,
                'default_levels': np.copy(probe_levels),
                'levels': np.copy(probe_levels),
                'xrange': np.array([0 * bnk_width, sites['n_banks'] * bnk_width]),
            }

            feature_data[pid] = data_dict

        return feature_data

    def prepare_region_data(self, pids: list | np.ndarray) -> Bunch:
        """
        Prepare region data for a list of probe ids.

        Parameters
        ----------
        pids: list
            The list of pids to prepare data for.

        Returns
        -------
        region_data: Bunch
            A bunch containing the region data for each pid.
        """
        region_data = Bunch()

        for pid in pids:
            df = self.data[self.data['pid'] == pid]
            mlapdv = np.c_[df['x'].values, df['y'].values, df['z'].values]

            region, region_label, region_colour, region_id = EphysAlignment.get_histology_regions(
                mlapdv, df['axial_um'].values, brain_atlas=self.ba
            )

            data_dict = {
                'regions': region,
                'labels': region_label,
                'colors': region_colour,
                'region_id': region_id,
            }

            region_data[pid] = data_dict

        return region_data

    def prepare_offset_data(self, pids: list | np.ndarray) -> Bunch:
        """
        Prepare offset data for a list of probe ids.

        Parameters
        ----------
        pids: list
            The list of pids to prepare data for.

        Returns
        -------
        offset_data: Bunch
            A bunch containing the offset data for each pid.

        """
        offset_data = Bunch()

        for pid in pids:
            data = self.region_data[pid]

            idx_reg = np.where(data['region_id'] == self.chosen_region)[0]
            regs = np.array([])
            for idx in idx_reg:
                regs = np.r_[regs, data['regions'][idx]]
            offset = np.min(regs) + (np.max(regs) - np.min(regs)) / 2
            offset_data[pid] = offset

        return offset_data

    # -------------------------------------------------------------------------
    # User selection
    # -------------------------------------------------------------------------
    def get_pids(self) -> np.ndarray:
        """
        Get the list of probe ids for the current page.

        Returns
        -------
        np.ndarray
            An array of probe ids for the current page.
        """
        if self.page_idx == self.page_num:
            pid_idx = np.arange(self.page_idx * self.step, self.max_idx)
        else:
            pid_idx = np.arange(self.page_idx * self.step, (self.page_idx * self.step) + self.step)

        return self.pids[pid_idx]

    def on_cmap_chosen(self, idx: int) -> None:
        """
        Select colormap to use.

        Set new default colormap and re-plot all features.

        Parameters
        ----------
        idx: int
            The index of the chosen colormap.
        """
        item = self.view.cmap_list.item(idx)
        self.cmap = item.text()
        self.view.cmap = self.cmap
        self.plot_all(self.current_pids)

    def on_region_chosen(self, idx: int) -> None:
        """
        Select region to display.

        Set new chosen region and get the probes that pass through this region.

        Parameters
        ----------
        idx: int
            The index of the chosen region.
        """
        # Find the chosen region
        self.chosen_region = self.region_ids[idx]

        # Find the pids that pass through this region and order them according
        # to number of channels in region
        pids = self.data[self.data['atlas_id'] == self.chosen_region]
        pids = pids.groupby('pid').atlas_id.count().sort_values()[::-1]
        self.pids = pids.index.values

        # Reset the offset data
        self.offset_data = Bunch()
        # Compute the number of pages required
        self.page_num = np.ceil(self.pids.size / self.step) - 1
        self.max_idx = self.pids.size
        self.page_idx = 0
        self.update_page_label()
        # Get the pids for the first page
        self.current_pids = self.get_pids()
        # Plot the data for the first page
        self.on_plot_chosen(self.plot_idx)

    def on_plot_chosen(self, idx: int) -> None:
        """
        Select feature to plot.

        Set new plot name and prepare data for current pids and plot.

        Parameters
        ----------
        idx: int
        """
        self.plot_idx = idx
        item = self.view.plot_list.item(idx)
        self.plot_name = item.text()

        # Find the min max levels for the chosen feature across all probes
        df = self.data[self.data['pid'].isin(self.pids)]
        self.max_levels = np.nanquantile(df[self.plot_name].values, [0, 1])
        self.view.slider.set_slider_intervals(self.max_levels)
        self.levels = np.copy(self.max_levels)

        self.load_data(self.current_pids)
        self.view.slider.set_slider_values(self.max_levels)
        self.plot_all(self.current_pids)

        if not self.normalised:
            self.on_area_clicked(None, self.selected_idx)

    def on_normalise_plots(self) -> None:
        """Normalise all plots to the same color scale."""
        if self.normalised:
            for area in self.view.fig_areas:
                area.setBackground('white')
        else:
            area = self.view.fig_areas[self.selected_idx]
            area.setBackground('lightblue')

        self.plot_features(self.current_pids)

    def on_align_plots(self):
        """
        Align plots based on chosen region.

        If align is True then plots are centered on the chosen region.
        """
        self.plot_all(self.current_pids)

    def on_reset_levels(self) -> None:
        """Reset colour levels to default for the selected probe."""
        if not self.normalised:
            selected = self.feature_data[self.plot_name][self.selected_pid]
            selected['levels'] = np.copy(selected['default_levels'])
            self.view.slider.set_slider_values(selected['levels'])
            self.plot_single_feature()

    def on_area_clicked(self, _, idx: int) -> None:
        """
        Highlight selected probe and update slider and pid label.

        Only works in non-normalised mode.

        Parameters
        ----------
        idx: int
            The index of the clicked area.
        """
        if not self.normalised:
            # Highlight the background of the selected area
            full_idx = (self.page_idx * self.step) + idx
            if full_idx < len(self.pids):
                for i, area in enumerate(self.view.fig_areas):
                    if i == idx:
                        area.setBackground('lightblue')
                    else:
                        area.setBackground('white')

                self.selected_idx = idx
                self.selected_pid = self.current_pids[idx]

                # Update the colorbar slider to the values of the selected probe
                data = self.feature_data[self.plot_name][self.selected_pid]
                self.view.slider.set_slider_values(data['levels'])
                # Update the pid label
                self.view.pid_label.setText(f'PID: {self.selected_pid}')

    # -------------------------------------------------------------------------
    # Slider
    # -------------------------------------------------------------------------
    def on_slider_released(self) -> None:
        """Update colorbar levels based on slider values."""
        if self.normalised:
            self.levels = self.view.slider.get_slider_values()
            self.view.slider.set_slider_values(self.levels)
            self.plot_features(self.current_pids)
        else:
            data = self.feature_data[self.plot_name][self.selected_pid]
            data['levels'] = self.view.slider.get_slider_values()
            self.view.slider.set_slider_values(data['levels'])
            self.plot_single_feature()

    # -------------------------------------------------------------------------
    # Pagination
    # -------------------------------------------------------------------------
    def on_next_page_pressed(self) -> None:
        """
        Go to the next page of probes.

        If data needs to be loaded for the new page, it will first be loaded before plotting.
        """
        if self.page_idx < self.page_num:
            self.page_idx += 1
            self.update_page_label()
            # Get the pids for the current page
            self.current_pids = self.get_pids()
            self.load_data(self.current_pids)
            self.plot_all(self.current_pids)

    def on_prev_page_pressed(self) -> None:
        """
        Go to the previous page of probes.

        If data needs to be loaded for the new page, it will first be loaded before plotting.
        """
        if self.page_idx > 0:
            self.page_idx -= 1
            self.update_page_label()
            # Get the pids for the current page
            self.current_pids = self.get_pids()
            self.load_data(self.current_pids)
            self.plot_all(self.current_pids)

    def update_page_label(self) -> None:
        """Update the label showing the current page index."""
        self.view.page_label.setText(f'{int(self.page_idx) + 1}/{int(self.page_num) + 1}')

    # -------------------------------------------------------------------------
    # Plot data
    # -------------------------------------------------------------------------
    def plot_all(self, pids: list | np.ndarray) -> None:
        """
        Update all plots for a list of probe ids.

        Parameters
        ----------
        pids: list or np.ndarray
            The list of probe ids to plot.
        """
        self.view.clear_plots()
        self.plot_features(pids)
        self.plot_regions(pids)

    def plot_regions(self, pids: list | np.ndarray) -> None:
        """
        Plot the region data for a list of probe ids.

        Parameters
        ----------
        pids: list or np.ndarray
            The list of probe ids to plot.
        """
        for i, pid in enumerate(pids):
            data = self.region_data[pid]
            offset = self.offset_data[pid] if self.aligned else 0
            self.view.plot_region(i, data, offset, self.chosen_region)

        if i < self.step - 1:
            for fig in self.view.plots_hist[i + 1 :]:
                axis = fig.getAxis('left')
                axis.setTicks([])
                axis.setPen(None)

    def plot_features(self, pids: list | np.ndarray) -> None:
        """
        Plot the feature data for a list of probe ids.

        Parameters
        ----------
        pids: list or np.ndarray
            The list of probe ids to plot.
        """
        for i, pid in enumerate(pids):
            data = self.feature_data[self.plot_name][pid]
            offset = self.offset_data[pid] if self.aligned else 0
            levels = self.levels if self.normalised else None
            self.view.plot_probe(i, data, offset, levels=levels)

        if i < self.step - 1:
            for fig in self.view.plots_feat[i + 1 :]:
                set_axis(fig, 'bottom', pen=None)

    def plot_single_feature(self) -> None:
        """Plot the feature data for the currently selected probe."""
        data = self.feature_data[self.plot_name][self.selected_pid]
        offset = self.offset_data[self.selected_pid] if self.aligned else 0
        self.view.plot_probe(self.selected_idx, data, offset)
