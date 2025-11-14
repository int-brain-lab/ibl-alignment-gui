import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
import oursin as urchin
from matplotlib import cm
from matplotlib.colors import Normalize, rgb2hex
from qtpy import QtCore, QtWidgets

from iblutil.util import Bunch

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController
    from iblatlas.atlas import AllenAtlas

from ibl_alignment_gui.utils.utils import shank_loop

PLUGIN_NAME = "3D features"

SHANK_COLOURS = {
    '0': '#000000',
    '1': '#000000',
    'a': '#000000',
    'b': '#30B666',
    'c': '#ff0044',
    'd': '#0000ff'
}


def setup(controller: 'AlignmentGUIController') -> None:
    """
    Set up the 3D Features plugin.

    Adds a submenu to the main menu with options to show/hide regions and adjust point size.
    Also attaches necessary callbacks to the controller.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = Bunch()

    feature3d_plugin = Features3D(controller)
    controller.plugins[PLUGIN_NAME]['loader'] = feature3d_plugin
    controller.plugins[PLUGIN_NAME]['activated'] = False

    # Attach callbacks to methods in the controller
    controller.plugins[PLUGIN_NAME]['data_button_pressed'] = feature3d_plugin.data_button_pressed
    controller.plugins[PLUGIN_NAME]['on_config_selected'] = feature3d_plugin.update_plots
    controller.plugins[PLUGIN_NAME]['filter_unit_pressed'] = feature3d_plugin.update_plots
    controller.plugins[PLUGIN_NAME]['update_plots'] = feature3d_plugin.update_plots
    controller.plugins[PLUGIN_NAME]['plot_probe_panels'] = feature3d_plugin.plot_channels
    controller.plugins[PLUGIN_NAME]['plot_scatter_panels'] = feature3d_plugin.plot_clusters

    # Add a submenu to the main menu
    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, controller.view)
    controller.plugin_options.addMenu(plugin_menu)

    # Show the 3D viewer setup
    show_action = QtWidgets.QAction('Show 3D Viewer', controller.view)
    show_action.triggered.connect(lambda _, c=controller: callback(_, c))
    show_action.setCheckable(True)
    show_action.setChecked(False)
    plugin_menu.addAction(show_action)

    # Toggle action to show / hide regions
    region_action = QtWidgets.QAction('Show Regions', controller.view)
    region_action.setCheckable(True)
    region_action.setChecked(False)
    region_action.triggered.connect(
        lambda a=region_action: feature3d_plugin.toggle_regions(region_action.isChecked()))
    plugin_menu.addAction(region_action)
    feature3d_plugin.region_toggle = region_action

    # Slider widget to change size of displayed points
    slider_min = QtWidgets.QLabel('0.1')
    slider_max = QtWidgets.QLabel('1')
    slider_max.setAlignment(QtCore.Qt.AlignRight)
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(1)
    slider.setMaximum(10)
    slider.setValue(5)
    slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
    slider.setTickInterval(1)
    slider.sliderReleased.connect(lambda s=slider: feature3d_plugin.on_point_size_changed(s))
    slider_layout = QtWidgets.QGridLayout()
    slider_layout.setVerticalSpacing(0)
    slider_layout.addWidget(slider, 0, 0, 1, 10)
    slider_layout.addWidget(slider_min, 1, 0, 1, 1)
    slider_layout.addWidget(slider_max, 1, 9, 1, 1)
    slider_widget = QtWidgets.QWidget()
    slider_widget.setLayout(slider_layout)
    slider_action = QtWidgets.QWidgetAction(controller.view)
    slider_action.setDefaultWidget(slider_widget)
    plugin_menu.addAction(slider_action)


def callback(_, controller: 'AlignmentGUIController') -> None:
    """Open the 3D viewer."""
    if not controller.plugins[PLUGIN_NAME]['activated']:
        controller.plugins[PLUGIN_NAME]['activated'] = True
        controller.plugins[PLUGIN_NAME]['loader'].setup()


class Features3D:
    """
    A class to manage viewing features in a 3D Urchin viewer.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.

    Attributes
    ----------
    controller: AlignmentGUIController
        The main application controller.
    particles: oursin.particles.ParticleSystem or None
        The particle system for displaying points.
    markers: oursin.particles.ParticleSystem or None
        The particle system for displaying markers.
    regions: list or np.ndarray
        The list of brain regions displayed.
    texts: list
        The list of text objects displayed.
    point_size: float
        The size of the points displayed.
    plot: str
        The type of plot currently displayed ('channels' or 'clusters').
    plot_type: str or None
        The specific plot name currently displayed.
    side: oursin.utils.Side
        The hemisphere side for displaying regions.
    region_toggle: QtWidgets.QAction or None
        The action to toggle region visibility.
    ba: AllenAtlas
        A brain atlas instance.
    """

    def __init__(self, controller: 'AlignmentGUIController'):
        self.controller = controller

        # Initialize variables
        self.particles: urchin.particles.ParticleSystem | None = None
        self.markers: urchin.particles.ParticleSystem | None = None
        self.regions: list | np.ndarray = []
        self.texts: list = []
        self.point_size: float = 0.05
        self.plot: str = 'channels'
        self.plot_type: str | None = None
        self.side: urchin.utils.Side = urchin.utils.Side.LEFT
        self.region_toggle: QtWidgets.QAction | None = None
        self.ba: AllenAtlas = self.controller.model.brain_atlas

    def setup(self):
        """Launch the 3D Urchin viewer and display the initial probe channels."""
        # Initialize urchin
        urchin.setup()
        time.sleep(5)
        # Load the CCF25 brain atlas
        urchin.ccf25.load()
        time.sleep(5)
        # Add the brain root
        urchin.ccf25.root.set_visibility(True)
        urchin.ccf25.root.set_material('transparent-lit')
        urchin.ccf25.root.set_alpha(0.5)

        self.plot_channels(self.controller.probe_init)

    def data_button_pressed(self) -> None:
        """
        Add regions and plots to the 3D view for the chosen probe.

        Called when the data button is pressed in the main application.
        """
        # Remove existing data
        for text in self.texts:
            text.delete()
        self.toggle_regions(False)
        self.particles = None
        self.markers = None
        self.regions = []
        self.texts = []

        # Find new regions
        regions = get_regions(self.controller)
        regions = np.unique(np.concatenate(regions))
        self.add_regions(regions)
        self.region_toggle.setChecked(True)

        # Plot initial channels
        self.plot_channels(self.controller.probe_init)

    def add_regions(self, regions: list | np.ndarray, hemisphere: int = -1) -> None:
        """
        Add a list of brain regions to the 3D view.

        Parameters
        ----------
        regions: list or np.ndarray
            A list of region names to add.
        hemisphere: int, optional
            The hemisphere to display the regions in (-1 for left, 1 for right). Default is -1.
        """
        regions = [r for r in regions if r not in ['void', 'root']]
        self.side = urchin.utils.Side.LEFT if hemisphere == -1 else urchin.utils.Side.RIGHT
        self.regions = urchin.ccf25.get_areas(regions)
        urchin.ccf25.set_visibilities(self.regions, True, self.side)
        urchin.ccf25.set_materials(self.regions, 'transparent-lit', 'left')
        urchin.ccf25.set_alphas(self.regions, 0.25, 'left')

    def toggle_regions(self, display: bool) -> None:
        """
        Show or hide the brain regions in the 3D view.

        Parameters
        ----------
        display: bool
            Whether to display the brain regions.
        """
        if len(self.regions) > 0:
            urchin.ccf25.set_visibilities(self.regions, display, self.side)

    def set_points(self, points: dict[str, list]) -> None:
        """
        Add a set of points to the 3D view.

        Parameters
        ----------
        points: dict
            The position and color of the points to add.
        """
        self.particles = urchin.particles.ParticleSystem(n=len(points['pos']))
        self.particles.set_material('circle')
        self.particles.set_positions(points['pos'])
        self.particles.set_colors(points['col'])
        self.set_point_size(self.point_size)

    def on_point_size_changed(self, slider: QtWidgets.QSlider) -> None:
        """
        Update the size of the points in the 3D view based on the slider value.

        Parameters
        ----------
        slider: QtWidgets.QSlider
            A slider widget with values from 1 to 10 representing point size.
        """
        self.set_point_size(slider.value() / 100)

    def set_point_size(self, point_size: float) -> None:
        """
        Set the size of the points in the 3D view.

        Parameters
        ----------
        point_size: float
            The size of the points to set.
        """
        self.point_size = point_size
        self.particles.set_sizes(list(np.ones(self.particles.data.n) * self.point_size * 1000))

    def set_markers(self, markers: list) -> None:
        """
        Add markers and text to indicate shank positions in the 3D view.

        Parameters
        ----------
        markers: dict
            A list of marker positions, colors, and names.
        """
        self.markers = urchin.particles.ParticleSystem(n=len(markers))
        self.markers.set_material('circle')
        self.markers.set_positions([m['pos'] for m in markers])
        self.markers.set_colors([m['col'] for m in markers])
        self.markers.set_sizes(list(np.ones(self.markers.data.n) * 250))

        if len(self.texts) == 0 and len(markers) > 0:
            text = sorted(markers, key=lambda x: x['name'])
            self.texts = urchin.text.create(len(text))
            urchin.text.set_texts(self.texts, [t['name'] for t in text])
            urchin.text.set_positions(self.texts,
                                      [[-0.95, 0.95], [-0.95, 0.9], [-0.95, 0.85], [-0.95, 0.8]])
            urchin.text.set_font_sizes(self.texts, [24, 24, 24, 24])
            urchin.text.set_colors(self.texts, [t['col'] for t in text])

    def update_plots(self) -> None:
        """Update the plots in the 3D view based on the current selection."""
        self.plot_channels(self.plot_type) if self.plot == 'channels' \
            else self.plot_clusters(self.plot_type)

    def plot_channels(self, plot_key: str, *args) -> None:
        """
        Plot channel data in the 3D view.

        Parameters
        ----------
        plot_key: str
            The name of channel plot to display.
        """
        self.plot = 'channels'
        self._plot_data(plot_key, update_channels)

    def plot_clusters(self, plot_key: str, *args) -> None:
        """
        Plot cluster data in the 3D view.

        Parameters
        ----------
        plot_key: str
            The name of the cluster plot to display.
        """
        if plot_key == 'Amplitude':
            return
        self.plot = 'clusters'
        self._plot_data(plot_key, update_clusters)

    def _plot_data(self, plot_key: str, update_function: Callable) -> None:
        """
        Plot data in the 3D view.

        Parameters
        ----------
        plot_key: str
            The name of plot to display.
        update_function: Callable
            The function to use for getting the data.
        """
        self.plot_type = plot_key

        if (self.controller.model.selected_config == 'both'
                or not self.controller.model.selected_config):
            data = update_function(self.controller, plot_key)
        else:
            data = update_function(self.controller, plot_key,
                                   configs=[self.controller.model.selected_config])

        colours = []
        positions = []
        markers = []

        for dat in data:

            if dat['xyz'] is None:
                continue

            cols = dat['values']
            xyz = dat['xyz']
            mlapdv = self.ba.xyz2ccf(xyz, mode='clip')
            shank = dat['shank']

            for i, loc in enumerate(mlapdv):
                colours.append(cols[i])
                # convert to ap ml dv order
                positions.append([loc[1], loc[0], loc[2]])

            # Find the position to put the shank indicators
            min_idx = np.argmin(mlapdv[:, 2])
            sh_info = {'name': shank,
                       'pos': [mlapdv[min_idx, 1], mlapdv[min_idx, 0], mlapdv[min_idx, 2] - 200],
                       'col': SHANK_COLOURS[shank[-1]]}
            if self.controller.model.selected_config != 'both' or dat['config'] == 'quarter':
                markers.append(sh_info)

        urchin.particles.clear()
        if len(positions) > 0:
            self.set_points({'pos': positions, 'col': colours})
            self.set_markers(markers)


def data_to_colors(
        data: list | np.ndarray,
        cmap: str,
        vmin: float,
        vmax: float
) -> list:
    """
    Convert data values to hex color codes.

    Parameters
    ----------
    data: list or np.ndarray
        The data values to convert to colors.
    cmap: str
        The name of the matplotlib colormap to use.
    vmin: float
        The minimum value for the colormap.
    vmax: float
        The maximum value for the colormap.

    Returns
    -------
    chex: list
        A list of hex color codes corresponding to each data value.
    """
    cmap = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=mpl.colormaps[cmap])
    cvals = cmap.to_rgba(data)
    chex = [rgb2hex(c) for c in cvals]
    return chex


@shank_loop
def get_regions(_, items: 'ShankController', **kwargs):
    """
    Get unique brain regions that the shank passes through.

    Parameters
    ----------
    items: ShankController
        A ShankController instance containing model data.

    Returns
    -------
    np.ndarray
        An array of unique region names.
    """
    return np.unique(items.model.hist_data['axis_label'][:, 1])


@shank_loop
def update_clusters(_, items: 'ShankController', plot_key: str, **kwargs) -> dict[str, Any]:
    """
    Get the cluster data for the chosen plot.

    Parameters
    ----------
    items: ShankController
        A ShankController instance containing model data.
    plot_key: str
        The name of the plot to display data for.

    Returns
    -------
    dict
        A dictionary containing data for 3D plotting.
    """
    xyz = items.model.xyz_clusters
    data = items.model.scatter_plots.get(plot_key, None)
    if data is None:
        return {'xyz': None, 'values': None, 'shank': kwargs['shank'], 'config': kwargs['config']}

    values = data_to_colors(data.colours, data.cmap, data.levels[0], data.levels[1])

    return {'xyz': xyz, 'values': values, 'shank': kwargs['shank'], 'config': kwargs['config']}


@shank_loop
def update_channels(_, items: 'ShankController', plot_key: str, **kwargs) -> dict[str, Any]:
    """
    Get the channel data for the chosen plot.

    Parameters
    ----------
    items: ShankController
        A ShankController instance containing model data.
    plot_key: str
        The name of the plot to display data for.

    Returns
    -------
    dict
        A dictionary containing data for 3D plotting.
    """
    xyz = items.model.xyz_channels
    data = items.model.probe_plots.get(plot_key, None)
    if data is None:
        return {'xyz': None, 'values': None, 'shank': kwargs['shank'], 'config': kwargs['config']}

    # We need to do this because the probe plots are split by bank
    vals = np.concatenate(data.img, axis=1)[0]
    idx = np.concatenate(data.idx)
    vals = vals[idx]

    values = data_to_colors(vals, data.cmap, data.levels[0], data.levels[1])

    return {'xyz': xyz, 'values': values, 'shank': kwargs['shank'], 'config': kwargs['config']}
