from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.app.widgets.custom_widgets import CheckBoxGroup, PopupWindow, SliderWidget
from ibl_alignment_gui.utils.helpers import shank_loop
from iblutil.util import Bunch

if TYPE_CHECKING:
    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController

PLUGIN_NAME = 'Range Controller'


def setup(controller: 'AlignmentGUIController') -> None:
    """
    Set up the Range Controller plugin.

    Adds a submenu to the main menu to open the plugin and attaches necessary callbacks
    to the controller.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = Bunch()
    range_controller = RangeController(PLUGIN_NAME, controller)
    controller.plugins[PLUGIN_NAME]['loader'] = range_controller
    controller.plugins[PLUGIN_NAME]['activated'] = False

    # Attach callbacks to methods in the controller
    controller.plugins[PLUGIN_NAME]['plot_image_panels'] = range_controller.on_plot_changed
    controller.plugins[PLUGIN_NAME]['plot_probe_panels'] = range_controller.on_plot_changed
    controller.plugins[PLUGIN_NAME]['plot_scatter_panels'] = range_controller.on_plot_changed
    controller.plugins[PLUGIN_NAME]['plot_line_panels'] = range_controller.on_plot_changed
    controller.plugins[PLUGIN_NAME]['on_view_changed'] = range_controller.disable_sliders

    # Add a submenu to the main menu
    action = QtWidgets.QAction(PLUGIN_NAME, controller.view)
    action.triggered.connect(lambda: callback(controller))
    controller.plugin_options.addAction(action)


def callback(controller: 'AlignmentGUIController') -> None:
    """
    Open the Range Controller plugin window.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME]['activated'] = True
    controller.plugins[PLUGIN_NAME]['loader'].setup()


class RangeControllerView(PopupWindow):
    """
    The GUI view for the Range Controller plugin.

    Parameters
    ----------
    title : str
        The title of the popup window.
    controller : AlignmentGUIController
        The main application controller.
    """

    def __init__(self, title: str, controller: 'AlignmentGUIController'):
        # A high number of steps so that dragging the slider is fine grained even when the
        # extremes of the data are far apart
        self.steps: int = 1000
        self.controller: AlignmentGUIController = controller

        super().__init__(title, controller.view, size=(500, 600), graphics=False)

    def setup(self):
        """Add widgets to the popup window."""
        self.sliders = Bunch()
        self.labels = Bunch()

        for plot in ['image', 'probe', 'line']:
            self.sliders[plot] = SliderWidget(steps=self.steps, slider_type=plot, editable=True)
            self.labels[plot] = QtWidgets.QLabel()

        self.shank_options = self.create_checkbox_group(self.controller.all_shanks, 'All')
        self.shank_options.setup_callback(self.on_shank_button_clicked)

        self.config_options = self.create_checkbox_group(self.controller.model.configs, 'both')
        self.config_options.setup_callback(self.on_config_button_clicked)

        self.layout.addWidget(self.labels['image'])
        self.layout.addWidget(self.sliders['image'])
        self.layout.addWidget(self.labels['line'])
        self.layout.addWidget(self.sliders['line'])
        self.layout.addWidget(self.labels['probe'])
        self.layout.addWidget(self.sliders['probe'])
        self.layout.addWidget(QtWidgets.QLabel('Shanks:'))
        self.layout.addWidget(self.shank_options)
        self.layout.addWidget(QtWidgets.QLabel('Configurations:'))
        self.layout.addWidget(self.config_options)

    @staticmethod
    def create_checkbox_group(options: list[str], select_all: str) -> CheckBoxGroup:
        """
        Create a group of checkboxes with all of the options checked by default.

        The extra option that toggles all the others at once is only added if there is more
        than one option to choose between. If there is only one option, it is checked and
        disabled, as there is no choice to be made.

        Parameters
        ----------
        options: list of str
            The options to create checkboxes for.
        select_all: str
            The text of the extra option that toggles all the others at once.

        Returns
        -------
        CheckBoxGroup
            The group of checkboxes.
        """
        checkbox_group = CheckBoxGroup()

        if len(options) > 1:
            options = [select_all] + options

        checkbox_group.add_options(options)
        checkbox_group.set_checked(options)

        if len(options) == 1:
            checkbox_group.checkboxes[options[0]].setEnabled(False)

        return checkbox_group

    def on_shank_button_clicked(self, checked: bool, button: str) -> None:
        """
        Triggered when a shank button is clicked.

        If the clicked button is 'All', it checks or unchecks all shank buttons accordingly.

        Parameters
        ----------
        checked: bool
            Whether the button is checked or not.
        button: str
            The text of the button that was clicked.
        """
        if button == 'All' and checked:
            for _, checkbox in self.shank_options.checkboxes.items():
                checkbox.setChecked(True)
        elif button == 'All' and not checked:
            for _, checkbox in self.shank_options.checkboxes.items():
                checkbox.setChecked(False)

    def on_config_button_clicked(self, checked: bool, button: str) -> None:
        """
        Triggered when a configuration button is clicked.

        If the clicked button is 'both', it checks or unchecks all configuration
        buttons accordingly.

        Parameters
        ----------
        checked: bool
            Whether the button is checked or not.
        button: str
            The text of the button that was clicked.
        """
        if button == 'both' and checked:
            for _, checkbox in self.config_options.checkboxes.items():
                checkbox.setChecked(True)
        elif button == 'both' and not checked:
            for _, checkbox in self.config_options.checkboxes.items():
                checkbox.setChecked(False)

    def get_selected_shanks(self) -> list[str]:
        """
        Get the list of selected shanks from the GUI.

        Returns
        -------
        list
            A list of selected shank names.
        """
        return [shank for shank in self.shank_options.get_checked() if 'All' not in shank]

    def get_selected_configs(self) -> list[str]:
        """
        Get the list of selected configurations from the GUI.

        Returns
        -------
        list
            A list of selected configuration names.
        """
        return [config for config in self.config_options.get_checked() if 'both' not in config]


class RangeController:
    """
    The Range Controller plugin for the alignment GUI.

    Parameters
    ----------
    title : str
        The title of the plugin window.
    controller : AlignmentGUIController
        The main application controller.

    Attributes
    ----------
    title : str
        The title of the plugin window.
    controller : AlignmentGUIController
        The main application controller.
    view : RangeControllerView
        The GUI view for the Range Controller plugin.
    disable : bool
        A flag to disable callbacks during updates.
    plot_keys : Bunch
        A collection of current plot keys for image, probe, and line plots.

    """

    def __init__(self, title: str, controller: 'AlignmentGUIController'):
        self.title = title
        self.controller = controller
        self.view = None
        self.disable = False

        self.plot_keys = Bunch()
        self.plot_keys['image'] = None
        self.plot_keys['probe'] = None
        self.plot_keys['line'] = None

    def setup(self) -> None:
        """Set up the plugin GUI and connect callbacks."""
        self.view = RangeControllerView._get_or_create(self.title, self.controller)
        self.view.closed.connect(self.on_close)
        for slider_widget in self.view.sliders.values():
            slider_widget.released.connect(self.on_slider_moved)
            slider_widget.levels_changed.connect(self.on_slider_moved)
            slider_widget.reset.connect(self.on_reset_button_pressed)

        self.set_init_levels()

    def on_close(self) -> None:
        """
        Triggered when the plugin window is closed.

        Deactivate the plugin and callbacks.
        """
        self.controller.plugins[PLUGIN_NAME]['activated'] = False

    def disable_sliders(self) -> None:
        """Disable sliders when the view is changed to feature view."""
        for slider_widget in self.view.sliders.values():
            slider_widget.set_enabled(not self.controller.show_feature)

        if not self.controller.show_feature:
            self.set_init_levels()

    def set_init_levels(self) -> None:
        """Set the initial levels for image, probe, and line plots based on controller settings."""
        self.plot_keys['image'] = self.controller.img_init
        self.view.labels['image'].setText(f'Image: {self.plot_keys["image"]}')
        self.on_plot_changed(
            self.plot_keys['image'], self.get_image_plot_type(self.plot_keys['image'])
        )

        self.plot_keys['probe'] = self.controller.probe_init
        self.view.labels['probe'].setText(f'Probe: {self.plot_keys["probe"]}')
        self.on_plot_changed(self.plot_keys['probe'], 'probe')

        self.plot_keys['line'] = self.controller.line_init
        self.view.labels['line'].setText(f'Line: {self.plot_keys["line"]}')
        self.on_plot_changed(self.plot_keys['line'], 'line')

    def get_image_plot_type(self, plot_key: str) -> str | None:
        """
        Determine the plot type for a given plot key.

        Parameters
        ----------
        plot_key: str

        Returns
        -------
        str | None
            The plot type ('image' or 'scatter') or None if not found.
        """
        if plot_key in self.controller.model.image_keys:
            return 'image'
        elif plot_key in self.controller.model.scatter_keys:
            return 'scatter'
        else:
            return None

    def on_plot_changed(self, plot_key: str, plot_type: str) -> None:
        """
        Triggered when a plot is changed in the main controller.

        If no view is set or callbacks are disabled, it returns immediately.

        Parameters
        ----------
        plot_key: str
            The key of the plot that was changed.
        plot_type: str
            The type of the plot ('image', 'probe', 'line', or 'scatter').
        """
        if self.view is None or self.disable:
            return

        key = plot_type if plot_type != 'scatter' else 'image'
        self.plot_keys[key] = plot_key

        # Find the extremes across all shanks and configs. The default levels are used, as these
        # are the extremes of the data, while the current levels are the ones that have been
        # applied to the plot. The current levels are included so that levels lying outside of
        # the data can still be shown on the slider.
        data = [
            dat['data']
            for dat in get_levels(self.controller, plot_key, plot_type)
            if dat['data'] is not None
        ]
        all_levels = np.array([dat.default_levels for dat in data] + [dat.levels for dat in data])
        bounds = [np.nanmin(all_levels), np.nanmax(all_levels)]
        self.view.sliders[key].set_slider_intervals(bounds)

        # Find the levels currently applied to the selected shanks and configs. Where these
        # differ between shanks, the widest range is shown, so that the levels of no shank lie
        # outside of the ones displayed.
        data = get_levels(
            self.controller,
            plot_key,
            plot_type,
            shanks=self.view.get_selected_shanks(),
            configs=self.view.get_selected_configs(),
        )
        levels = np.array([dat['data'].levels for dat in data if dat['data'] is not None])
        levels = bounds if len(levels) == 0 else [np.nanmin(levels[:, 0]), np.nanmax(levels[:, 1])]

        # Update the slider and label
        self.view.sliders[key].set_slider_values(levels)
        self.view.labels[key].setText(f'{key.capitalize()}: {self.plot_keys[key]}')

    def plot_panels(self, plot_key: str, plot_type: str) -> None:
        """
        Re-plot the panels in the main view for the given plot key and type.

        Parameters
        ----------
        plot_key: str
            The key of the plot to update.
        plot_type: str
            The type of the plot ('image', 'probe', 'line', 'scatter' or 'feature')
        """
        if plot_type == 'probe':
            self.controller.plot_probe_panels(plot_key)
        elif plot_type == 'line':
            self.controller.plot_line_panels(plot_key)
        elif plot_type == 'scatter':
            self.controller.plot_scatter_panels(plot_key)
        elif plot_type == 'image':
            self.controller.plot_image_panels(plot_key)

    def on_slider_moved(self, slider_widget: SliderWidget, plot_type: str) -> None:
        """
        Triggered when a slider is moved.

        Updates the slider values and updates the plots.

        Parameters
        ----------
        slider_widget: SliderWidget
            The slider widget that was moved.
        plot_type: str
            The type of the plot
        """
        # If not feature plot shown and feature slider moved, return
        if self.controller.show_feature:
            return

        # Ensure we don't go into on_plot_changed
        self.disable = True

        levels = slider_widget.get_slider_values()
        plot_key = self.plot_keys[plot_type]
        if plot_type == 'image':
            plot_type = self.get_image_plot_type(plot_key)

        set_levels(
            self.controller,
            plot_key,
            plot_type,
            levels,
            shanks=self.view.get_selected_shanks(),
            configs=self.view.get_selected_configs(),
        )

        slider_widget.set_slider_values(levels)

        self.plot_panels(plot_key, plot_type)
        self.disable = False

    def on_reset_button_pressed(self, _, plot_type: str) -> None:
        """
        Triggered when the reset button is pressed.

        Resets the levels to default and updates the plots.

        Parameters
        ----------
        plot_type: str
            The type of the plot
        """
        self.disable = True

        plot_key = self.plot_keys[plot_type]
        if plot_type == 'image':
            plot_type = self.get_image_plot_type(plot_key)

        reset_default_levels(self.controller, plot_key, plot_type)
        self.plot_panels(plot_key, plot_type)
        self.disable = False
        # Call this to update the slider values
        self.on_plot_changed(plot_key, plot_type)


@shank_loop
def set_levels(
    controller: 'AlignmentGUIController',
    items: 'ShankController',
    plot_key: str,
    plot_type: str,
    levels: tuple[float, float],
    **kwargs,
) -> None:
    """
    Set the levels for a given plot key and type.

    Parameters
    ----------
    plot_key: str
        The key of the plot to update.
    plot_type: str
        The type of the plot ('image', 'probe', 'line', or 'scatter').
    levels: list or tuple
        The new levels to set.
    """
    data = get_plot_group(items, plot_type).get(plot_key, None)

    if data is None:
        return
    data.levels = np.copy(levels)


@shank_loop
def reset_default_levels(
    controller: 'AlignmentGUIController',
    items: 'ShankController',
    plot_key: str,
    plot_type: str,
    **kwargs,
) -> None:
    """
    Reset the levels to default for a given plot key and type.

    Parameters
    ----------
    plot_key: str
        The key of the plot to reset.
    plot_type: str
        The type of the plot ('image', 'probe', 'line', or 'scatter').
    """
    data = get_plot_group(items, plot_type).get(plot_key, None)

    if data is None:
        return
    data.levels = np.copy(data.default_levels)


@shank_loop
def get_levels(
    controller: 'AlignmentGUIController',
    items: 'ShankController',
    plot_key: str,
    plot_type: str,
    **kwargs,
) -> dict[str, Any]:
    """
    Get the levels for a given plot key and type.

    Parameters
    ----------
    plot_key: str
        The key of the plot.
    plot_type: str
        The type of the plot ('image', 'probe', 'line', or 'scatter').

    Returns
    -------
    dict
        A dictionary containing the data, shank, and config.
    """
    data = get_plot_group(items, plot_type).get(plot_key, None)
    if data is None:
        return {'data': None, 'shank': kwargs['shank'], 'config': kwargs['config']}

    return {'data': data, 'shank': kwargs['shank'], 'config': kwargs['config']}


def get_plot_group(items: 'ShankController', plot_type: str) -> Bunch:
    """
    Get the plot group for a given plot type.

    Parameters
    ----------
    plot_type: The type of the plot ('image', 'probe', 'line', or 'scatter').

    Returns
    -------
    Bunch
        The plot group corresponding to the plot type.
    """
    plot_groups = {
        'probe': items.model.probe_plots,
        'line': items.model.line_plots,
        'scatter': items.model.scatter_plots,
        'image': items.model.image_plots,
    }
    return plot_groups[plot_type]
