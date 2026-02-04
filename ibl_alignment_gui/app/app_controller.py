import time
from collections import defaultdict
from collections.abc import Callable

import matplotlib.pyplot as mpl  # noqa  # This is needed to make qt show properly :/
import numpy as np
import pyqtgraph as pg

from ibl_alignment_gui.app.app_view import AlignmentGUIView
from ibl_alignment_gui.app.shank_controller import ShankController
from ibl_alignment_gui.handlers.probe_handler import (
    ProbeHandlerCSV,
    ProbeHandlerLocal,
    ProbeHandlerLocalYaml,
    ProbeHandlerONE,
)
from ibl_alignment_gui.plugins.add_plugins import Plugins
from ibl_alignment_gui.plugins.qc_dialog import display as display_qc_dialog
from ibl_alignment_gui.plugins.upload_dialog import display as display_upload_dialog
from ibl_alignment_gui.utils.qt.custom_widgets import ColorBar
from ibl_alignment_gui.utils.utils import shank_loop
from iblutil.util import Bunch


class AlignmentGUIController:
    """
    The main controller class for the alignment GUI application.

    Parameters
    ----------
    offline: bool
        Whether to run in offline mode (local files) or online mode (ONE/Alyx)
    csv: Path or str or None
        Path to a CSV file containing local sessions on the filesystem.

    Attributes
    ----------
    csv : str or bool or None
        The CSV path used for session loading. Defaults to `False` if no CSV is provided.
    offline : bool
        Indicates whether the controller is in offline mode.
    view : AlignmentGUIView
        The GUI view for the alignment application.
    model : ProbeHandlerLocal or ProbeHandlerCSV or ProbeHandlerONE
        The data model used for session management and probe handling.
    loaded : bool
        Indicates whether session data has been loaded.
    extend_feature : int
        Parameter controlling feature extension when applying alignments.
    lin_fit : bool
        Whether to use a linear fit when applying alignments.
    show_lines : bool
        Whether to display reference lines in the GUI.
    show_labels : bool
        Whether to display labels in the GUI.
    show_channels : bool
        Whether to display channels in the GUI.
    hover_line : pg.InfiniteLine or None
        The currently hovered line item, if any.
    hover_shank : str or None
        The shank name of the currently hovered item, if any.
    hover_idx : int or None
        The shank index of the currently hovered item, if any.
    hover_config : str or None
        The configuration of the currently hovered item, if any.
    all_shanks : list
        A list of all shanks.
    shank_items : defaultdict of Bunch
        A dictionary containing the ShankController instances for each shank and configuration.
    slice_figs : Bunch
        A container for slice figures currently displayed.
    blockPlugins : bool
        Whether plugins are temporarily blocked from running.
    img_init, probe_init, line_init, slice_init, filter_init : str or None
        Track the initially loaded image, probe, line, slice, and filter
        states, respectively.
    plugins : dict
        A mapping of plugin names to plugin instances.
    """

    def __init__(self, offline: bool = False, csv: str | None = None, yaml: str | None = None):
        self.offline = offline
        self.csv: str | None = csv
        self.yaml: str | None = yaml

        if offline:
            if self.yaml is None:
                self.model = ProbeHandlerLocal()
            else:
                self.model = ProbeHandlerLocalYaml(self.yaml)
        elif self.csv is None:
            self.model = ProbeHandlerONE()
        else:
            self.model = ProbeHandlerCSV(self.csv)

        self.view: AlignmentGUIView = AlignmentGUIView(
            offline=self.offline, config=len(self.model.configs) > 1
        )

        if not offline:
            self.view.populate_selection_dropdown('subject', self.model.get_subjects())

        # Keep track of whether data has been loaded or not
        self.loaded: bool = False

        # Parameters for applying alignments
        self.extend_feature: int = 1
        self.lin_fit: bool = True

        # Keep track of display options
        self.show_lines: bool = True
        self.show_labels: bool = True
        self.show_channels: bool = True

        # Mouse hover and interactions
        self.hover_line: pg.InfiniteLine | None = None
        self.hover_shank: str | None = None
        self.hover_idx: int | None = None
        self.hover_config: str | None = None

        # Available shanks and their controllers
        self.all_shanks: list = list()
        self.shank_items: dict[str, Bunch] = defaultdict(Bunch)

        # Initial images
        self.img_init: str | None = None
        self.probe_init: str | None = None
        self.line_init: str | None = None
        self.feature_init: str | None = None
        self.slice_init: str | None = None
        self.filter_init: str | None = None

        # The ephys view mode
        self.show_feature = False

        # Store the slice figures
        self.slice_figs: Bunch = Bunch()

        # Plugin management
        self.blockPlugins: bool = False

        # Setup all callbacks
        self.setup_connections()

        # Setup plugins
        Plugins(self)

        if self.yaml is not None:
            self.on_folder_selected(self.yaml)
            self.data_button_pressed()

    def setup_connections(self):
        """Set up all the connections between the view and controller methods."""
        # Setup connections for selection dropdowns and buttons
        if not self.offline:
            self.view.connect_selection_dropdown('subject', self.on_subject_selected)
            self.view.connect_selection_dropdown('session', self.on_session_selected)
        elif self.yaml is None:
            self.view.connect_selection_button('folder', self.on_folder_selected)

        self.view.connect_selection_dropdown('shank', self.on_shank_selected)
        self.view.connect_selection_dropdown('align', self.on_alignment_selected)
        self.view.connect_selection_dropdown('config', self.on_config_selected)
        self.view.connect_selection_button('data', self.data_button_pressed)

        # Setup connections for alignment buttons
        self.view.connect_button('fit', self.fit_button_pressed)
        self.view.connect_button('offset', self.offset_button_pressed)
        self.view.connect_button('reset', self.reset_button_pressed)
        self.view.connect_button('upload', self.complete_button_pressed)
        self.view.connect_button('next', self.next_button_pressed)
        self.view.connect_button('previous', self.prev_button_pressed)

        # Setup connections for tab / grid views
        self.view.connect_tabs('slice', self.slice_tab_changed)
        self.view.connect_tabs(
            'shank', self.shank_tab_changed, layout_callback=self.tab_layout_changed
        )

        # Setup connections for fit figures
        self.view.connect_lin_fit(self.lin_fit_option_changed)

        # Setup shortcuts and add the options to tabs in the menubar
        fit_options = {
            # Shortcuts to apply fit
            'Fit': {'shortcut': 'Return', 'callback': self.fit_button_pressed},
            # Shortcuts to apply offset
            'Offset': {'shortcut': 'O', 'callback': self.offset_button_pressed},
            'Offset + 100um': {'shortcut': 'Shift+Up',
                               'callback': self.moveup_button_pressed},
            'Offset - 100um': {'shortcut': 'Shift+Down',
                               'callback': self.movedown_button_pressed},
            # Shortcut to remove a reference line
            'Remove Line': {'shortcut': 'Shift+D', 'callback': self.delete_reference_line},
            # Shortcut to move between previous/next moves
            'Next': {'shortcut': 'Shift+Right', 'callback': self.next_button_pressed},
            'Previous': {'shortcut': 'Shift+Left', 'callback': self.prev_button_pressed},
            # Shortcut to reset GUI to initial state
            'Reset': {'shortcut': 'Shift+R', 'callback': self.reset_button_pressed},
            # Shortcut to upload final state to Alyx/to local file
            'Upload': {'shortcut': 'Shift+U', 'callback': self.complete_button_pressed},
        }
        display_options = {
                # Shortcuts to toggle between plots options
                'Toggle Image Plots ->': {'shortcut': 'Alt+1',
                                          'callback': lambda: self.toggle_plots('image', 1)},
                'Toggle Line Plots ->': {'shortcut': 'Alt+2',
                                         'callback': lambda: self.toggle_plots('line', 1)},
                'Toggle Probe Plots ->': {'shortcut': 'Alt+3',
                                          'callback': lambda: self.toggle_plots('probe', 1)},
                'Toggle Slice Plots ->': {'shortcut': 'Alt+4',
                                          'callback': lambda: self.toggle_plots('slice', 1)},
                'Toggle Image Plots <-': {'shortcut': 'Shift+Alt+1',
                                          'callback': lambda: self.toggle_plots('image', -1)},
                'Toggle Line Plots <-': {'shortcut': 'Shift+Alt+2',
                                         'callback': lambda: self.toggle_plots('line', -1)},
                'Toggle Probe Plots <-': {'shortcut': 'Shift+Alt+3',
                                          'callback': lambda: self.toggle_plots('probe', -1)},
                'Toggle Slice Plots <-': {'shortcut': 'Shift+Alt+4',
                                          'callback': lambda: self.toggle_plots('slice', -1)},
                # Shortcut to reset axis on figures
                'Reset Axis': {'shortcut': 'Shift+A', 'callback': self.reset_axis_button_pressed},
                # Shortcut to hide/show region labels
                'Hide/Show Labels': {'shortcut': 'Shift+L', 'callback': self.toggle_labels},
                # Shortcut to hide/show reference lines
                'Hide/Show Lines': {'shortcut': 'Shift+H',
                                    'callback': self.toggle_reference_lines},
                # Shortcut to hide/show reference lines and channels on slice image
                'Hide/Show Channels': {'shortcut': 'Shift+C',
                                       'callback': self.toggle_channels},
                # Shortcut to change toggle between grid and tab view
                'Toggle layout': {'shortcut': 'T', 'callback': self.toggle_layout},
                # Shortcuts to move between shanks
                'Next shank': {'shortcut': 'Right',
                               'callback': lambda: self.loop_through_tabs(1)},
                'Previous shank': {'shortcut': 'Left',
                                   'callback': lambda: self.loop_through_tabs(-1)},
                # Shortcut to reset all plots to their default range
                'Reset Range': {'shortcut': 'R', 'callback': self.on_reset_levels},
            }

        self.view.add_shortcuts_to_menu('fit', fit_options)
        self.view.add_shortcuts_to_menu('display', display_options)

    def load_data(self):
        self.model.load_data()
        self.loaded = True
        self.create_shanks()
        self.execute_plugins('load_data', self)

    def load_plots(self):
        self.model.load_plots()

    def populate_menubar(self):
        """Populate menu bar tabs based on avaialble plots."""
        self.img_init = self.view.populate_menu_tab(
            'image', self.plot_image_panels, self.model.image_keys)
        self.view.populate_menu_tab(
            'image', self.plot_scatter_panels, self.model.scatter_keys, set_checked=False)
        self.probe_init = self.view.populate_menu_tab(
            'probe', self.plot_probe_panels, self.model.probe_keys)
        self.line_init = self.view.populate_menu_tab(
            'line', self.plot_line_panels, self.model.line_keys)
        self.feature_init = self.view.populate_menu_tab(
            'feature', self.plot_feature_panels, self.model.feature_keys)
        self.slice_init = self.view.populate_menu_tab(
            'slice', self.plot_slice_panels, self.model.slice_keys)
        filter_keys = ['All', 'KS good', 'KS mua', 'IBL good']
        self.filter_init = self.view.populate_menu_tab(
            'filter', self.filter_unit_pressed, filter_keys)

    # --------------------------------------------------------------------------------------------
    # Plugins
    # --------------------------------------------------------------------------------------------
    def execute_plugins(self, func: str, *args, **kwargs):
        """
        Execute plugin methods that are linked to specific methods within the controller.

        Parameters
        ----------
        func : str
            The key to the function to execute
        """
        if self.blockPlugins:
            return
        for _, plug in self.plugins.items():
            if plug.get('activated', False):
                plug_func = plug.get(func, None)
                if plug_func is not None:
                    plug_func(*args, **kwargs)

    def connect_cluster_plugin(self, items: ShankController) -> None:
        """Connect the cluster feature plugin to the scatter plot."""
        if 'Cluster Features' in self.plugins and items.cluster:
            scatter = items.view.ephys_plot
            scatter.sigClicked.connect(
                lambda plot, points:
                self.plugins['Cluster Features']['callback'](self, items, plot, points)
            )

    # --------------------------------------------------------------------------------------------
    # Shank controllers
    # --------------------------------------------------------------------------------------------
    def create_shanks(self) -> None:
        """Create ShankController instance for each shank and config combination."""
        self.shank_items = defaultdict(Bunch)
        for i, shank in enumerate(self.all_shanks):
            for c in self.model.configs:
                self.shank_items[shank][c] = (
                    ShankController(self.model.shanks[shank][c], shank, i, c))

    def init_shanks(self) -> None:
        """Initialise the plots for each ShankController and add callbacks to plot scenes."""
        shank_tabs = self.view.init_tabs(self.shank_items,
                                         self.model.selected_config,
                                         self.model.default_config,
                                         self.model.non_default_config,
                                         feature_view=self.show_feature)
        for tab in shank_tabs:
            tab.setup_double_click(self.on_mouse_double_clicked)
            tab.setup_mouse_hover(self.on_mouse_hover)

    @shank_loop
    def reset_reference_line_arrays(self, items: ShankController, **kwargs) -> None:
        """See :meth:`ShankController.init_reference_line_arrays` for details."""
        items.init_reference_line_arrays()

    @shank_loop
    def reset_shanks(self, items: ShankController, **kwargs) -> None:
        """See :meth:`ShankController.init_plot_items` for details."""
        items.init_plot_items()

    def get_config(self) -> str:
        """
        Get the current config or default if both are selected.

        Returns
        -------
        config: str
            The config to use
        """
        return (self.model.default_config if self.model.selected_config == 'both'
                else self.model.selected_config)

    # --------------------------------------------------------------------------------------------
    # Plotting functions
    # --------------------------------------------------------------------------------------------
    @shank_loop
    def plot_histology_panels(self, items: ShankController, **kwargs):
        """Plot histology panel per shank and config."""
        self.hover_region = items.plot_histology()

    @shank_loop
    def plot_histology_ref_panels(self, items: ShankController, **kwargs) -> None:
        """Plot histology reference panel per shank and config."""
        items.plot_histology_ref()

    def plot_scale_factor_panels(self, shanks: list | tuple | None = None) -> None:
        """
        Plot scale factor panel for list of shanks.

        If the selected_config is both adjusts the display of the colorbar.

        Parameters
        ----------
        shanks: list or tuple
            List of shanks to plot figure for
        """
        if self.model.selected_config == 'both':
            results = self._plot_scale_factor_panels(shanks=shanks)
            for res in results:
                cbar = res['cbar']
                cbar.set_axis(cbar.ticks, cbar.label, loc='top', extent=20)
                cbar.set_axis([], loc='bottom', extent=20)
        else:
            self._plot_scale_factor_panels(shanks=shanks)

    @shank_loop
    def _plot_scale_factor_panels(self, items: ShankController, **kwargs) -> Bunch:
        """
        Plot scale factor panels per shank and config.

        Returns
        -------
        Bunch
            A bunch containing the cbar object as well as the shank and config it belongs to
        """
        cbar = items.plot_scale_factor()
        return Bunch(shank=kwargs.get('shank'), config=kwargs.get('config'), cbar=cbar)

    @shank_loop
    def plot_fit_panels(self,  items: ShankController, **kwargs) -> None:
        """Plot fit panel per shank and config."""
        items.plot_fit()

    @shank_loop
    def remove_fit_panels(self, items: ShankController, **kwargs) -> None:
        """Remove lines on fit plot for shanks other than the selected shanks."""
        if kwargs.get('shank') != self.model.selected_shank:
            items.view.clear_fit()

    @shank_loop
    def _plot_slice_panels(self, items: ShankController, plot_key: str, **kwargs) -> Bunch:
        """
        Plot slice panel per shank and config.

        Parameters
        ----------
        plot_key: str
            The key of the slice plot to display
        """
        self.slice_init = plot_key
        fig, img, cbar = items.plot_slice(plot_key)

        return Bunch(shank=kwargs.get('shank'), fig=fig, img=img, cbar=cbar)

    def plot_slice_panels(self, plot_key: str, data_only: bool = True) -> None:
        """
        Plot slice panels and configure the LUT.

        Parameters
        ----------
        plot_key: str
            The key of the slice plot to display
        data_only: bool
            Whether the plot can be generated without histology data

        Notes
        -----
        - If the plot type is 'Annotation', the LUT is removed.
        """
        if plot_key != self.slice_init:
            if not self.show_channels:
                self.toggle_channels()
            # If plot key changes reset the lut levels
            self.view.set_levels(None)

        self.slice_figs = Bunch()
        if self.model.selected_config == 'both':
            results = self._plot_slice_panels(plot_key, data_only=data_only,
                                              configs=[self.model.default_config])
            self.slice_figs = {res['shank']: res['fig'] for res in results}
            self.plot_channel_panels()
        else:
            results = self._plot_slice_panels(plot_key, data_only=data_only,
                                              configs=[self.model.selected_config])
            self.slice_figs = {res['shank']: res['fig'] for res in results}
            self.plot_channel_panels(configs=[self.model.selected_config])

        imgs = [res['img'] for res in results]
        cb = [res['cbar'] for res in results][0]

        if self.slice_init != 'Annotation':
            self.view.set_lut(imgs, cb)
        else:
            self.view.remove_lut()

    @shank_loop
    def plot_channel_panels(self, items: ShankController, **kwargs) -> None:
        """Plot channels on slice plots."""
        self.show_channels = True
        c = 'g' if items.config == self.model.default_config else 'r'
        items.plot_channels(self.slice_figs[kwargs.get('shank')], colour=c)

    def plot_line_panels(self, plot_key: str, data_only: bool = True, **kwargs) -> None:
        """
        Plot line panels per shank and config.

        Parameters
        ----------
        plot_key: str
            The key of the line plot to display
        data_only: bool
            Whether the plot can be generated without histology data
        """
        self.line_init = plot_key

        if self.show_feature:
            self.show_feature = False
            self.on_view_changed()
            return

        self._plot_line_panels(plot_key, data_only=data_only, **kwargs)

        self.execute_plugins('plot_line_panels', plot_key, 'line')

    @shank_loop
    def _plot_line_panels(
            self,
            items: ShankController,
            plot_key: str,
            data_only: bool = True,
            **kwargs
    ) -> None:
        """
        Plot line panels per shank and config.

        Parameters
        ----------
        plot_key: str
            The key of the line plot to display
        data_only: bool
            Whether the plot can be generated without histology data
        """
        items.plot_line(plot_key)

    def plot_feature_panels(self, plot_key: str, data_only: bool = True, **kwargs) -> None:
        """
        Plot feature panels per shank and config.

        Parameters
        ----------
        plot_key: str
            The key of the feature plot to display
        data_only: bool
            Whether the plot can be generated without histology data
        """
        self.feature_init = plot_key

        if not self.show_feature:
            self.show_feature = True
            self.on_view_changed()
            return

        self._plot_feature_panels(plot_key, data_only=data_only, **kwargs)

        self.execute_plugins('plot_feature_panels', plot_key, 'feature')

    @shank_loop
    def _plot_feature_panels(
            self,
            items: ShankController,
            plot_key: str,
            data_only: bool = True,
            **kwargs
    ) -> None:
        """
        Plot feature panels per shank and config.

        Parameters
        ----------
        plot_key: str
            The key of the feature plot to display
        data_only: bool
            Whether the plot can be generated without histology data
        """
        items.plot_feature(plot_key)

    def plot_panels(
            self,
            plot_key: str,
            plot_type: str,
            plot_func: str,
            init_attr: str,
            dual_cb_name: str | None = None,
            plugin_event: str | None = None,
            data_only: bool = True,
            **kwargs
    ) -> None:
        """
        Plot a generic panel per shank and config.

        Parameters
        ----------
        plot_key: str
            The key of the plot to display
        plot_type: str
            The type of plot to update e.g. image, probe, scatter
        plot_func: str
            The name of the function used to update the plots
        init_attr: str
            The name of the attribute that stores the current plot key of the plot
            type e.g. self.probe_init
        dual_cb_name: str
            The name of the dual colorbar object to use
        plugin_event: str
            The plugin event name to link plugin callbacks
        data_only: bool
            Whether the plot can be generated without histology data
        """
        # Update which plot was last selected
        setattr(self, init_attr, plot_key)

        if self.show_feature:
            self.show_feature = False
            self.on_view_changed()
            return

        if self.model.selected_config == 'both':
            results = self._plot_panels(plot_key, plot_type, plot_func, data_only,
                                        **kwargs)
            if dual_cb_name:
                self.plot_dual_colorbar(results, dual_cb_name)
        else:
            self._plot_panels(plot_key, plot_type, plot_func, data_only, **kwargs)

        # Optional plugin event
        if plugin_event:
            self.execute_plugins(plugin_event, plot_key, plot_type)

    @shank_loop
    def _plot_panels(
            self,
            items: ShankController,
            plot_key: str,
            plot_type: str,
            plot_func: str,
            data_only: bool = True,
            **kwargs
    ) -> Bunch:
        """
        Plot the panel per shank and config.

        Returns
        -------
        Bunch
            A bunch containing the cbar object as well as the shank and config it belongs to
        """
        plot_func = getattr(items, plot_func)
        cbar = plot_func(plot_key)

        if plot_type == 'scatter':
            self.connect_cluster_plugin(items)

        return Bunch(shank=kwargs.get('shank'), config=kwargs.get('config'), cbar=cbar)

    def plot_scatter_panels(self, plot_key: str, data_only: bool = True, **kwargs) -> None:
        """
        Plot scatter panels per shank and config.

        Parameters
        ----------
        plot_key: str
            The key of the scatter plot to display
        data_only: bool
            Whether the plot can be generated without histology data
        """
        self.plot_panels(plot_key, plot_type='scatter', plot_func='plot_scatter',
                         init_attr='img_init', dual_cb_name='fig_dual_img_cb',
                         plugin_event='plot_scatter_panels', data_only=data_only, **kwargs)

    def plot_image_panels(self, plot_key: str, data_only: bool = True, **kwargs) -> None:
        """
        Plot image panels per shank and config.

        Parameters
        ----------
        plot_key: str
            The key of the image plot to display
        data_only: bool
            Whether the plot can be generated without histology data
        """
        self.plot_panels(plot_key, plot_type='image', plot_func='plot_image',
                         init_attr='img_init', dual_cb_name='fig_dual_img_cb',
                         plugin_event='plot_image_panels', data_only=data_only, **kwargs)

    def plot_probe_panels(self, plot_key: str, data_only: bool = True, **kwargs) -> None:
        """
        Plot probe panels per shank and config.

        Parameters
        ----------
        plot_key: str
            The key of the probe plot to display
        data_only: bool
            Whether the plot can be generated without histology data
        """
        self.plot_panels(plot_key, plot_type='probe', plot_func='plot_probe',
                         init_attr='probe_init', dual_cb_name='fig_dual_probe_cb',
                         plugin_event='plot_probe_panels', data_only=data_only, **kwargs)

    def plot_dual_colorbar(self, results: Bunch, fig: str) -> None:
        """
        Update colorbar based on config selection.

        When the selected_config is both, update the colorbar displayed to show the levels of
        both configs. The levels of the default config are shown on the top axis, and the levels
        of the non-default config on the bottom axis.

        Parameters
        ----------
        results: Bunch
            A bunch containing the cbar figures per shank and config
        fig: str
            The name of the plot item to show the updated dual colorbar on
        """
        cbs = defaultdict(Bunch)
        for res in results:
            cbs[res['shank']][res['config']] = res['cbar']

        for shank in cbs:
            cb_default = cbs[shank].get(self.model.default_config)
            cb_non_default = cbs[shank].get(self.model.non_default_config)
            cmap = cb_non_default.cmap_name \
                if cb_non_default else cb_default.cmap_name if cb_default else None
            if not cmap:
                continue
            fig_cb = getattr(self.shank_items[shank][self.model.default_config].view, fig)
            cbar = ColorBar(cmap, plot_item=fig_cb)
            if cb_default:
                cbar.set_axis(cb_default.ticks, cb_default.label, loc='top', extent=20)
            else:
                cbar.set_axis([], cb_non_default.label, loc='top', extent=20)
            if cb_non_default:
                cbar.set_axis(cb_non_default.ticks, loc='bottom', extent=20)

    def update_plots(self, shanks: tuple | list = ()) -> None:
        """
        Update all plots and displays to reflect the current alignment state.

        Parameters
        ----------
        shanks: tuple
            The list of shanks to update plots for
        """
        self.get_scaled_histology(shanks=shanks)
        self.plot_histology_panels(shanks=shanks)
        self.plot_scale_factor_panels(shanks=shanks)
        self.plot_fit_panels(shanks=shanks)
        if self.model.selected_config == 'both':
            self.plot_channel_panels(shanks=shanks)
        else:
            self.plot_channel_panels(shanks=shanks, configs=[self.model.selected_config])
        self.remove_reference_lines_from_display(shanks=shanks)
        self.add_reference_lines_to_display(shanks=shanks)
        self.align_reference_lines(shanks=shanks)
        self.set_yaxis_range('fig_hist', shanks=shanks)
        self.update_string()
        self.execute_plugins('update_plots')

    def set_ephys_plots(self) -> None:
        """Set the ephys plots to the values stored in the init variables."""
        self.blockPlugins = True
        if not self.show_feature:
            self.view.trigger_menu_option('image', self.img_init)
            self.view.trigger_menu_option('line', self.line_init)
            self.view.trigger_menu_option('probe', self.probe_init)
        else:
            self.view.trigger_menu_option('feature', self.feature_init)
        self.blockPlugins = False

    # --------------------------------------------------------------------------------------------
    # Selection
    # --------------------------------------------------------------------------------------------
    def on_subject_selected(self, idx: int) -> None:
        """
        Triggered when a subject/ session is selected from the subject dropdown list.

        Parameters
        ----------
        idx: int
            The index selected in the dropdown list
        """
        self.loaded = None
        self.view.clear_selection_dropdown('session')
        sessions = self.model.get_sessions(idx)
        self.view.populate_selection_dropdown('session', sessions)
        self.on_session_selected(0)
        self.view.activate_selection_button()

    def on_session_selected(self, idx: int) -> None:
        """
        Triggered when a session/ probe is selected from the session dropdown list.

        Parameters
        ----------
        idx: int
            The index selected in the dropdown list
        """
        self.loaded = None
        self.view.clear_selection_dropdown(['shank', 'config'])
        self.model.get_config(0)
        shanks = self.model.get_shanks(idx)
        self.view.populate_selection_dropdown('shank', shanks)
        self.on_shank_selected(0)
        self.view.activate_selection_button()

    def on_shank_selected(self, idx: int) -> None:
        """
        Triggered when a shank is selected from the shank dropdown list.

        Updates the alignment dropdown list with any previous alignments for this shank.
        If the data is already loaded, the display is updated to highlight the selected shank.
        If the layout is in tab mode, the fit lines and points on the fit plot are only shown
        for the selected shank.

        Parameters
        ----------
        idx: int
            The index selected in the dropdown list
        """
        self.view.clear_selection_dropdown('align')
        self.model.set_info(idx)
        self.view.populate_selection_dropdown('align', self.model.get_previous_alignments())
        # Load the initial alignment
        self.model.get_starting_alignment(0)
        if self.loaded is not None:
            # If in tab view, update the tab to display the selected shank
            self.view.set_tabs(idx)
            # Remove points from the fit figure from the previous shank
            self.remove_points_from_display()
            # Add points to the fit figure for the selected shank
            self.add_points_to_display()
            if not self.view.is_grid:
                # If we are in tab view, remove the fit lines on the fit figure from the
                # previous shank
                self.remove_fit_panels()
                # Add the fit lines for the selected shank
                self.plot_fit_panels(shanks=[self.model.selected_shank])
            else:
                # If we are in grid view highlight the header of the selected shank
                self.set_shank_header()

    def on_alignment_selected(self, idx: int) -> None:
        """
        Triggered when an alignment is selected from the alignment dropdown list.

        Updates the reference lines to the selected alignment.

        Parameters
        ----------
        idx: int
            The index selected in the dropdown list
        """
        # Load the selected alignment
        self.model.get_starting_alignment(idx)
        if self.loaded is not None:
            # Remove previous reference lines
            self.remove_reference_lines_from_display(shanks=[self.model.selected_shank])
            # Reset arrays that track the reference lines
            self.reset_reference_line_arrays(shanks=[self.model.selected_shank])
            # Add the reference lines for the selected alignment
            self.set_init_reference_lines(shanks=[self.model.selected_shank])
            # Update the plots
            self.update_plots(shanks=[self.model.selected_shank])

    def on_config_selected(self, idx: int, init: bool = False) -> None:
        """
        Triggered when a config is selected from the config dropdown list.

        Parameters
        ----------
        idx: int
            The index selected in the dropdown list
        init: bool
            Whether this is the first time loading the probe or not
        """
        self.model.get_config(idx)
        self.setup(init=init)

        if not init:
            self.execute_plugins('on_config_selected')
            self.view.focus()

    def on_folder_selected(self, folder_path: str | None = None) -> None:
        """Triggered in offline mode when the folder button is clicked."""
        self.loaded = None
        self.view.clear_selection_dropdown(['align', 'shank'])
        if folder_path:
            self.view.set_selected_path(folder_path)
        else:
            folder_path = self.view.get_selected_path()
        shank_options = self.model.get_shanks(folder_path)
        self.view.populate_selection_dropdown('shank', shank_options)
        self.on_shank_selected(0)
        self.view.activate_selection_button()

    def on_view_changed(self):
        """Triggered when the view is changed between feature and ephys plots."""
        self.setup(init=False)
        self.execute_plugins('on_view_changed')

    # --------------------------------------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------------------------------------
    def data_button_pressed(self) -> None:
        """
        Load in all the relevant data and instantiate the GUI display.

        Triggered when data button is pressed.
        """
        if self.loaded:
            return
        start = time.time()
        # Get the list of shanks
        self.all_shanks = list(self.model.shanks.keys())
        # Load and prepare all data
        self.load_data()
        # Load in all the plots
        self.load_plots()
        # Add all the plot options to the menubar
        self.populate_menubar()
        # If csv add the config options
        if self.view.config:
            self.view.populate_selection_dropdown('config', self.model.possible_configs)
        # Load in the shank panels and configure figures for initial config
        self.on_config_selected(0, init=True)
        # Execute any plugins linked to the current method
        self.execute_plugins('data_button_pressed')
        # Setup the view
        self.view.init_view()
        # Change colour of data button to indicate data has been loaded
        self.view.deactivate_selection_button()
        self.view.focus()
        print(time.time() - start)

    def setup(self, init=True) -> None:
        """
        Set up the GUI display according to the config used.

        Parameters
        ----------
        init: bool
            Whether the GUI is being loaded for a new probe or if the config is just changing
        """
        # Remove the reference lines so that we can add them back onto the new plots
        if not init:
            self.remove_reference_lines_from_display()

        # Reset the view
        self.view.reset_view()

        if not init:
            # Reset shank plot items
            self.reset_shanks(data_only=True)

        self.init_shanks()

        # Set the probe lims for each shank and config
        self.set_probe_lims(data_only=True)
        self.set_yaxis_lims()

        # Initialise ephys plots
        self.set_ephys_plots()

        # Initialise histology plots
        self.view.trigger_menu_option('slice', self.slice_init)
        self.get_scaled_histology()
        self.plot_histology_ref_panels()
        self.plot_histology_panels()
        self.plot_scale_factor_panels()
        self.show_labels = False
        self.toggle_labels()
        self.update_string()

        # Add reference lines to the display
        if init:
            self.set_init_reference_lines()
        else:
            self.add_reference_lines_to_display()
            # Ensure the slice images have the same lut as was set before config changed
            self.view.lut_widget.set_lut_levels()
        # Add reference points for selected shank
        self.remove_points_from_display()
        self.add_points_to_display()
        # Add fit lines for all shanks
        self.plot_fit_panels()
        # Select highlighted shank
        self.set_shank_header()

    def filter_unit_pressed(self, filter_type: str, data_only: bool = True) -> None:
        """
        Filter the ephys plots according to the type of unit selected.

        Parameters
        ----------
        filter_type: str
            The unit type
        data_only: bool
            Whether the plot can be generated without histology data
        """
        if filter_type == self.filter_init:
            return

        self.filter_init = filter_type
        self._filter_units(filter_type, data_only=data_only)
        self.set_ephys_plots()
        self.execute_plugins('filter_unit_pressed')

    @shank_loop
    def _filter_units(
            self,
            items: ShankController,
            *args,
            data_only: bool = True,
            **kwargs
    ) -> None:
        """"See :meth:`ShankController.filter_units` for details."""
        items.filter_units(*args)

    # --------------------------------------------------------------------------------------------
    # Upload data
    # --------------------------------------------------------------------------------------------
    def complete_button_pressed(self) -> None:
        """
        Triggered when complete button or Shift+U is pressed.

        Saves channel locations and alignments.
        """
        if len(self.all_shanks) > 1:
            shanks_to_upload = display_upload_dialog(self)
        else:
            shanks_to_upload = self.all_shanks

        for shank in shanks_to_upload:

            self.model.selected_shank = shank
            self.model.current_shank = shank

            if not self.offline:
                accepted = display_qc_dialog(self, shank)
                if accepted == 0:
                    break

            upload = self.view.upload_prompt()
            if upload:
                info = self.model.upload_data()
                self.view.populate_selection_dropdown('align',
                                                      self.model.load_previous_alignments())
                self.model.get_starting_alignment(0)
                self.view.upload_info(upload, info)
            else:
                self.view.upload_info(upload)

    # --------------------------------------------------------------------------------------------
    # Fitting functions
    # --------------------------------------------------------------------------------------------
    @shank_loop
    def offset_hist_data(self, items: ShankController, *args, **kwargs) -> None:
        """See :meth:`ShankController.offset_hist_data` for details."""
        items.offset_hist_data(*args)

    @shank_loop
    def scale_hist_data(self, items: ShankController, **kwargs) -> None:
        """Scale brain regions along the probe track based on reference lines."""
        items.scale_hist_data(self.extend_feature, self.lin_fit)

    @shank_loop
    def get_scaled_histology(self, items: ShankController, **kwargs) -> None:
        """See :meth:`ShankController.get_scaled_histology` for details."""
        items.get_scaled_histology()

    def apply_fit(self, fit_function: Callable, **kwargs) -> None:
        """
        Apply a given fitting function to histology data and update all relevant plots.

        Parameters
        ----------
        fit_function : Callable
            A function that modifies the alignment
        **kwargs :
            Additional arguments passed to `fit_function`.
        """
        fit_function(**kwargs)
        self.update_plots(shanks=[self.model.selected_shank])

    def offset_button_pressed(self) -> None:
        """
        Apply an offset to the selected shank based on location of the probe tip line.

        Called when the offset button or O key is pressed.
        """
        self.apply_fit(self.offset_hist_data, shanks=[self.model.selected_shank])

    def movedown_button_pressed(self) -> None:
        """
        Offset the probe tip of selected shank by 100 µm downwards.

        Called when Shift+down arrow is pressed.
        """
        self.apply_fit(self.offset_hist_data, shanks=[self.model.selected_shank], val=-100/1e6)

    def moveup_button_pressed(self) -> None:
        """
        Offset the probe tip of selected shank by 100 µm upwards.

        Called when Shift+up arrow is pressed.
        """
        self.apply_fit(self.offset_hist_data, shanks=[self.model.selected_shank], val=100/1e6)

    def fit_button_pressed(self) -> None:
        """
        Scale the regions using reference lines and updates plots.

        Called when the fit button or Enter is pressed.
        """
        self.apply_fit(self.scale_hist_data, shanks=[self.model.selected_shank])

    def next_button_pressed(self) -> None:
        """
        Update the display with next alignment stored in the alignment buffer.

        Ensures user cannot go past latest move. Called when the prev button or Shift+right
        arrow is pressed.
        """
        if self.model.next_idx():
            self.update_plots(shanks=[self.model.selected_shank])

    def prev_button_pressed(self) -> None:
        """
        Update the display with previous alignment stored in the alignment buffer.

        Called when next button or Shift+left arrow is pressed.
        """
        if self.model.prev_idx():
            self.update_plots(shanks=[self.model.selected_shank])

    def reset_button_pressed(self) -> None:
        """
        Reset the feature and track alignment to initial starting alignment and updates plots.

        Called when reset button or Shift+R is pressed.
        """
        self.remove_reference_lines_from_display(shanks=[self.model.selected_shank])
        self.reset_reference_line_arrays(shanks=[self.model.selected_shank])
        self.reset_feature_and_tracks(shanks=[self.model.selected_shank])
        self.set_init_reference_lines(shanks=[self.model.selected_shank])
        self.update_plots(shanks=[self.model.selected_shank])

    @shank_loop
    def reset_feature_and_tracks(self, items: ShankController, **kwargs) -> None:
        """See :meth:`ShankHandler.reset_features_and_tracks` for details."""
        items.model.reset_features_and_tracks()

    def lin_fit_option_changed(self, state: int) -> None:
        """
        Toggle the use of linear fit for scaling histology data.

        Parameters
        ----------
        state : int
            0 disables linear fit, any other value enables it.
        """
        self.lin_fit = bool(state)
        self.fit_button_pressed()

    def update_string(self) -> None:
        """Update on-screen text showing current and total alignment steps."""
        self.view.set_labels(self.model.current_idx, self.model.total_idx)

    # --------------------------------------------------------------------------------------------
    # Mouse interactions
    # --------------------------------------------------------------------------------------------
    def on_mouse_double_clicked(self, event, idx: int) -> None:
        """
        Handle a mouse double-click event on the ephys or histology plots.

        Adds a movable reference line on the ephys and histology plots.

        Parameters
        ----------
        event : pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
            The mouse double-click event.
        idx: int
            The index of the panel that the mouse click event occured
        """
        if event.double():
            if idx != self.model.selected_idx:
                self.view.set_selection_dropdown('shank', idx)
                self.on_shank_selected(idx)

            if len(self.model.configs) > 1:
                config = self.get_config()
                items = self.shank_items[self.model.selected_shank][config]
                pos = items.view.ephys_plot.mapFromScene(event.scenePos())
                y_scale = items.view.y_scale
                for config in self.model.configs:
                    items = self.shank_items[self.model.selected_shank][config]
                    self.create_reference_line(pos.y() * y_scale, items)
            else:
                items = self.shank_items[self.model.selected_shank][self.model.selected_config]
                pos = items.view.ephys_plot.mapFromScene(event.scenePos())
                self.create_reference_line(pos.y() * items.view.y_scale, items)

    def on_mouse_hover(
            self,
            hover_items: list[pg.GraphicsObject],
            name: str,
            idx: int,
            config: str
    ) -> None:
        """
        Handle a mouse hover event over the pyqtgraph plot items.

        Identifies reference lines or linear regions the mouse is hovering over to allow
        interactive operations like deletion or displaying additional info.

        Parameters
        ----------
        hover items : list of pyqtgraph.GraphicsObject
            List of items under the mouse cursor.
        name: str
            The name of the tab being hovered over
        idx:
            Then index of the tab that is being hovered over
        config:
            The config of the tab being hovered over
        """
        self.hover_idx = idx
        self.hover_shank = name
        self.hover_config = config

        if len(hover_items) > 1:
            self.hover_line = None
            hover_item0, hover_item1 = hover_items[0], hover_items[1]
            if isinstance(hover_item0, pg.InfiniteLine):
                self.hover_line = hover_item0
            elif isinstance(hover_item1, pg.LinearRegionItem):
                items = self.shank_items[name][config]
                # Check if we are on the fig_scale plot
                if hover_item0 == items.view.fig_scale:
                    items.set_scale_title(hover_item1)
                    return
                # Check if we are on the histology plot
                if hover_item0 == items.view.fig_hist:
                    self.hover_region = hover_item1
                    return
                if hover_item0 == items.view.fig_hist_ref:
                    self.hover_region = hover_item1
                    return
            elif self.show_feature and isinstance(hover_item1, pg.ImageItem):
                items = self.shank_items[name][config]
                title = getattr(hover_item1, 'feature_name', None)
                items.set_feature_title(title)
            else:
                items = self.shank_items[name][config]
                items.set_feature_title(None)

    # --------------------------------------------------------------------------------------------
    # Display options
    # --------------------------------------------------------------------------------------------
    def toggle_labels(self) -> None:
        """
        Toggle visibility of brain region labels on histology plot.

        Triggered by pressing Shift+L.
        """
        self.show_labels = not self.show_labels
        self._toggle_labels()

    @shank_loop
    def _toggle_labels(self,  items: ShankController, **kwargs) -> None:
        """See :meth:`ShankController.toggle_labels` for details."""
        items.toggle_labels(self.show_labels)

    def toggle_reference_lines(self) -> None:
        """
        Toggle visibility of reference lines.

        Triggered by pressing Shift+H.
        """
        self.show_lines = not self.show_lines
        if not self.show_lines:
            self.remove_reference_lines_from_display()
        else:
            self.add_reference_lines_to_display()

    def toggle_channels(self) -> None:
        """
        Toggle visibility of channels and trajectory lines on the slice image.

        Triggered by pressing Shift+C.
        """
        self.show_channels = not self.show_channels
        self._toggle_channels()

    @shank_loop
    def _toggle_channels(self, items: ShankController, **kwargs) -> None:
        """See :meth:`ShankController.toggle_channels` for details."""
        items.toggle_channels(self.slice_figs[kwargs.get('shank')], self.show_channels)

    def toggle_plots(self, plot_type: str, direction: int) -> None:
        """
        Toggle through the different plot types.

        Can toggle through the image, line, probe and slice plots by pressing the
        keys Alt+1, Alt+2, Alt+3 and Alt+4 respectively.

        Parameters
        ----------
        plot_type: str
            The type of plot to toggle through e.g. image, probe, scatter
        direction: int
            The direction to toggle in, -1 for previous, +1 for next
        """
        self.view.toggle_menu_option(plot_type, direction)

    # --------------------------------------------------------------------------------------------
    # Plot display interactions
    # --------------------------------------------------------------------------------------------
    def on_reset_levels(self) -> None:
        """
        Reset the levels of all plots to the default range.

        Triggered by pressing Shift+R.
        """
        self.reset_levels()

    @shank_loop
    def reset_levels(self, items: ShankController, **kwargs) -> None:
        """See :meth:`ShankController.reset_levels` for details."""
        items.reset_levels()

    def reset_axis_button_pressed(self) -> None:
        """
        Reset plot axis to default values.

        Triggered by pressing Shift+A.
        """
        self.set_yaxis_range('fig_hist')
        self.set_yaxis_range('fig_hist_ref')
        if self.show_feature:
            self.set_yaxis_range('fig_feature')
            self.set_xaxis_range('fig_feature')
        else:
            self.set_yaxis_range('fig_img')
            self.set_xaxis_range('fig_img')

        if self.model.selected_config == 'both':
            self.reset_slice_axis(configs=[self.model.default_config])
        else:
            self.reset_slice_axis(configs=[self.model.selected_config])

    @shank_loop
    def reset_slice_axis(self, items: ShankController, **kwargs) -> None:
        """See :meth:`ShankController.reset_slice_axis` for details."""
        items.reset_slice_axis()

    @shank_loop
    def set_xaxis_range(self, items: ShankController, *args, **kwargs) -> None:
        """See :meth:`ShankController.set_xaxis_range` for details."""
        items.set_xaxis_range(*args)

    @shank_loop
    def set_yaxis_range(self, items: ShankController, *args, **kwargs) -> None:
        """See :meth:`ShankController.set_yaxis_range` for details."""
        items.set_yaxis_range(*args)

    @shank_loop
    def set_probe_lims(self, items: ShankController, data_only: bool = True, **kwargs) -> None:
        """See :meth:`ShankController.set_probe_lims` for details."""
        items.set_probe_lims()

    def set_yaxis_lims(self) -> None:
        """
        Set the y-axis limits for all shanks based on stored values.

        Parameters
        ----------
        data_only: bool
            Whether the plot can be generated without histology data
        """
        results = self._get_yaxis_lims()
        if self.model.selected_config == 'both':
            ylims = Bunch.fromkeys(self.all_shanks, [])
            for res in results:
                ylims[res['shank']] += res['ylim']
            lims = defaultdict(Bunch)
            for shank in self.all_shanks:
                for config in self.model.configs:
                    lims[shank][config] = [np.nanmin(ylims[shank]), np.nanmax(ylims[shank])]
        else:
            lims = defaultdict(Bunch)
            for res in results:
                lims[res['shank']][res['config']] = res['ylim']

        self._set_yaxis_lims(lims)

    @shank_loop
    def _set_yaxis_lims(self, items: ShankController, lims, data_only=True, **kwargs) -> None:
        """See :meth:`ShankController.set_yaxis_lims` for details."""
        ylims = lims[kwargs.get('shank')][kwargs.get('config')]
        items.set_yaxis_lims(*ylims)

    @shank_loop
    def _get_yaxis_lims(self, items: ShankController, data_only=True, **kwargs) -> Bunch:
        """See :meth:`ShankController.get_yaxis_lims` for details."""
        return Bunch(shank=kwargs.get('shank'), config=kwargs.get('config'), ylim=items.get_yaxis_lims())

    # --------------------------------------------------------------------------------------------
    # Grid / Tab display interactions
    # --------------------------------------------------------------------------------------------
    @shank_loop
    def set_shank_header(self, items: ShankController, **kwargs) -> None:
        """See :meth:`ShankController.set_header_style` for details."""
        items.set_header_style(kwargs.get('shank') == self.model.selected_shank)

    def loop_through_tabs(self, direction: int):
        """
        Move between shank tabs using left and right arrow keys.

        Parameters
        ----------
        direction: int
            The direction to move in, -1 for previous, +1 for next
        """
        idx = np.mod(self.model.selected_idx + direction, len(self.all_shanks))
        self.view.set_selection_dropdown('shank', idx)
        self.on_shank_selected(idx)

    def tab_layout_changed(self) -> None:
        """Triggered when the tab layout is changed to a grid layout."""
        self.view.set_selection_dropdown('shank', self.model.selected_idx)
        self.on_shank_selected(self.model.selected_idx)

    def shank_tab_changed(self, idx: int):
        """
        Triggered when the tab on the shank tabs view is changed.

        Parameters
        ----------
        idx: int
            The index of the newly selected tab
        """
        self.view.set_selection_dropdown('shank', idx)
        self.on_shank_selected(idx)
        self.view.set_slice_tab(idx)
        self.remove_fit_panels()
        self.plot_fit_panels(shanks=[self.model.selected_shank])

    def slice_tab_changed(self, idx: int) -> None:
        """
        Triggered when the tab on the slice tabs view is changed.

        Parameters
        ----------
        idx: int
            The index of the newly selected tab
        """
        self.view.set_shank_tab(idx)

    def toggle_layout(self) -> None:
        """
        Toggle the layout between the grid and shank views.

        Triggered when T is pressed.
        """
        self.view.toggle_tabs(self.model.selected_idx)
        if self.view.is_grid:
            self.plot_fit_panels()
            self.set_shank_header()
        else:
            self.remove_fit_panels()

    # --------------------------------------------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------------------------------------------
    @shank_loop
    def set_init_reference_lines(self, items: ShankController, **kwargs) -> None:
        """Find the initial alignment for specified shanks and creates reference lines."""
        self.model.set_init_alignment()
        feature_prev = items.model.feature_prev
        if np.any(feature_prev):
            self.create_reference_lines(feature_prev[1:-1] * 1e6, items)

    def create_reference_line(self, pos: float, items: ShankController) -> None:
        """
        Create a reference line and a corresponding scatter point.

        It creates:
        - A track line in the histology figure
        - Feature lines in the image, line, and probe figures that are synchronized
        - A scatter point in the fit figure indicating the correspondence

        Parameters
        ----------
        pos : float
            Y-axis position at which to create the reference line.
        """
        # Create lines and point
        line_track, line_features, point = (
            items.create_reference_line_and_point(pos, fix_colour=len(self.all_shanks) > 1))
        # Add callbacks
        line_track.sigPositionChanged.connect(
            lambda track, i=items.index, c=items.config:
            self.update_track_reference_line(track, i, c))
        for line_feature in line_features:
            line_feature.sigPositionChanged.connect(
                lambda feature, i=items.index, c=items.config:
                self.update_feature_reference_line(feature, i, c))
        # Add point to fit figure
        self.view.add_point(point)

    def create_reference_lines(
            self,
            positions: np.ndarray | list[float],
            items: ShankController
    ) -> None:
        """
        Create reference lines across at multiple positions.

        Parameters
        ----------
        positions : array-like of float
            List or array of y-axis positions at which to create reference lines.
        """
        for pos in positions:
            self.create_reference_line(pos, items)

    def delete_reference_line(self) -> None:
        """
        Delete the currently selected reference line from all plots.

        Triggered when the user hovers over a reference line and presses Shift+D.
        """
        if not self.hover_line:
            return

        line_idx = None
        configs_to_check = self.model.configs if self.hover_config == 'both' else \
            [self.hover_config]

        for config in configs_to_check:
            items = self.shank_items[self.hover_shank][config]
            # Attempt to find selected line in feature lines
            line_idx, _ = items.match_feature_line(self.hover_line)
            if line_idx is not None:
                break
            # If not found, try in track lines
            line_idx = items.match_track_line(self.hover_line)
            if line_idx is not None:
                break

        if line_idx is None:
            return

        self._delete_reference_line(line_idx, shanks=[self.hover_shank])

    @shank_loop
    def _delete_reference_line(self, items: ShankController, line_idx: int,  **kwargs) -> None:
        """
        Delete a reference line from the display and remove from tracking arrays.

        Parameters
        ----------
        line_idx: int
            The index in the tracking arrays of the reference line to remove
        """
        # Remove line items from plots
        items.remove_reference_line(line_idx)
        # Remove the point from the fig fit
        self.view.remove_point(items.points[line_idx])
        # Remove from tracking arrays
        items.delete_reference_line_and_point(line_idx)

    def update_feature_reference_line(
            self,
            feature_line: pg.InfiniteLine,
            idx: int,
            config: str
    ) -> None:
        """
        Triggered when a reference line is moved in one of the electrophysiology plots.

        This function ensures the line's new position is synchronized across the other
        ephys plots, and updates the corresponding scatter point in the fit plot.

        Parameters
        ----------
        feature_line : pyqtgraph.InfiniteLine
            The line instance that was moved by the user.
        idx: int
            The panel number that the line instance belongs to, used to update the selected_shank
        config: str
            The config of the panel that the line instance belongs to
        """
        if idx != self.model.selected_idx:
            self.view.set_selection_dropdown('shank', idx)
            self.on_shank_selected(idx)

        items = self.shank_items[self.model.selected_shank][config]
        line_idx, fig_idx = items.match_feature_line(feature_line)
        self._update_feature_reference_line(feature_line, line_idx, fig_idx,
                                            shanks=[self.model.selected_shank])

    @shank_loop
    def _update_feature_reference_line(self, items: ShankController, *args, **kwargs) -> None:
        """See :meth:`ShankController.update_feature_reference_line_and_point` for details."""
        items.update_feature_reference_line_and_point(*args)

    def update_track_reference_line(
            self,
            track_line: pg.InfiniteLine,
            idx: int,
            config: str
    ) -> None:
        """
        Triggered when a reference line in the histology plot is moved.

        This updates the corresponding scatter point in the fit plot.

        Parameters
        ----------
        track_line : pyqtgraph.InfiniteLine
            The line instance that was moved by the user.
        idx: int
            The panel number that the line instance belongs to, used to update the selected_shank
        config: str
            The config of the panel that the line instance belongs to
        """
        if idx != self.model.selected_idx:
            self.view.set_selection_dropdown('shank', idx)
            self.on_shank_selected(idx)

        items = self.shank_items[self.model.selected_shank][config]
        line_idx = items.match_track_line(track_line)
        self._update_track_reference_line(track_line, line_idx,
                                          shanks=[self.model.selected_shank])

    @shank_loop
    def _update_track_reference_line(self, items: ShankController, *args, **kwargs) -> None:
        """See :meth:`ShankController.update_track_reference_line_and_point` for details."""
        items.update_track_reference_line_and_point(*args)

    @shank_loop
    def align_reference_lines(self, items: ShankController, **kwargs) -> None:
        """See :meth:`ShankController.align_reference_lines` for details."""
        items.align_reference_lines_and_points()

    @shank_loop
    def remove_points_from_display(self, items: ShankController, **kwargs) -> None:
        """Remove all reference points from the fit plot."""
        self.view.remove_points_from_display(items.points)

    def add_points_to_display(self) -> None:
        """Add reference points to the fit plot for the selected shank."""
        config = self.get_config()
        items = self.shank_items[self.model.selected_shank][config]
        self.view.add_points_to_display(items.points)

    @shank_loop
    def remove_reference_lines_from_display(self, items: ShankController, **kwargs) -> None:
        """Remove all reference lines and scatter points from the respective plots."""
        items.remove_reference_lines_from_display()
        self.view.remove_points_from_display(items.points)

    @shank_loop
    def add_reference_lines_to_display(self, items: ShankController, **kwargs) -> None:
        """Add previously created reference lines and scatter points to their respective plots."""
        shank = kwargs.get('shank')
        items.add_reference_lines_to_display()
        if shank == self.model.selected_shank:
            self.view.add_points_to_display(items.points)
