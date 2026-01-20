import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from datoviz.backends.pyqt6 import QtServer
from ibl_datoviz.viewer import Viewer
from ibl_datoviz.points import PointsController


import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, rgb2hex
from qtpy import QtCore, QtWidgets

from iblutil.util import Bunch

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController
    from iblatlas.atlas import AllenAtlas

from ibl_alignment_gui.utils.utils import shank_loop
from ibl_alignment_gui.utils.qt.custom_widgets import PopupWindow

PLUGIN_NAME = "3D features"

SHANK_COLOURS = {
    'a': [0, 255, 0, 255],
    'b': [48, 182, 102, 255],
    'c': [255, 0, 0, 255],
    'd': [0, 0, 255, 255],
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

    # # Add a submenu to the main menu
    # plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, controller.view)
    # controller.plugin_options.addMenu(plugin_menu)
    #
    # # Show the 3D viewer setup
    # show_action = QtWidgets.QAction('Show 3D Viewer', controller.view)
    # show_action.triggered.connect(lambda _, c=controller: callback(_, c))
    # show_action.setCheckable(True)
    # show_action.setChecked(False)
    # plugin_menu.addAction(show_action)

    action = QtWidgets.QAction(PLUGIN_NAME, controller.view)
    action.triggered.connect(lambda: callback(controller))
    controller.plugin_options.addAction(action)


def callback(controller: 'AlignmentGUIController') -> None:
    """Open the 3D viewer."""
    if not controller.plugins[PLUGIN_NAME]['activated']:
        controller.plugins[PLUGIN_NAME]['activated'] = True
        controller.plugins[PLUGIN_NAME]['loader'].setup()



class TestViewer(PopupWindow):

    def __init__(self, title: str, controller: 'AlignmentGUIController'):
        self.controller: AlignmentGUIController = controller

        super().__init__(title, controller.view, size=(500, 600), graphics=False)

    def setup(self):

        self.qt_server = QtServer(background='black')
        w, h = 800, 600
        self.qfig = self.qt_server.figure(w, h)
        self.panel = self.qfig.panel((0, 0), (w, h))
        self.panel.arcball()
        self.panel.gui()
        self.layout.addWidget(self.qfig)

        # Checkbox to show / hide regions
        self.regions = QtWidgets.QCheckBox('Show regions')
        self.regions.setChecked(True)

        # Checkbox to show / hide picks
        self.picks = QtWidgets.QCheckBox('Show picks')
        self.picks.setChecked(False)

        # Slider widget to change size of displayed points
        slider_min = QtWidgets.QLabel('0.1')
        slider_max = QtWidgets.QLabel('1')
        slider_max.setAlignment(QtCore.Qt.AlignRight)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(10)
        self.slider.setValue(5)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider.setTickInterval(1)
        slider_layout = QtWidgets.QGridLayout()
        slider_layout.setVerticalSpacing(0)
        slider_layout.addWidget(self.slider, 0, 0, 1, 10)
        slider_layout.addWidget(slider_min, 1, 0, 1, 1)
        slider_layout.addWidget(slider_max, 1, 9, 1, 1)
        slider_layout.addWidget(self.regions, 2, 0)
        slider_layout.addWidget(self.picks, 3, 0)
        slider_widget = QtWidgets.QWidget()
        slider_widget.setLayout(slider_layout)


        self.layout.addWidget(slider_widget)


        # TODO show trajectories
        # TODO show xyz picks
        # TODO add the gui elements to the view


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
        self.regions: list | np.ndarray = []
        self.texts: list = []
        self.point_size: float = 3
        self.plot: str = 'channels'
        self.plot_type: str | None = None
        self.side: str = 'left'
        self.region_toggle: QtWidgets.QAction | None = None
        self.ba: AllenAtlas = self.controller.model.brain_atlas

    def setup(self):
        """Launch the 3D Urchin viewer and display the initial probe channels."""

        self.view = TestViewer(PLUGIN_NAME, self.controller)
        self.view.closed.connect(self.on_close)
        self.view.slider.sliderReleased.connect(lambda s=self.view.slider: self.on_point_size_changed(s))
        self.view.regions.clicked.connect(lambda: self.toggle_regions(self.view.regions.isChecked()))
        self.view.picks.clicked.connect(lambda: self.toggle_picks(self.view.picks.isChecked()))
        self.viewer = Viewer(self.view.qt_server, self.view.panel)
        # Add an additional points controller for picks
        self.viewer.picks = PointsController(self.view.qt_server, self.view.panel,
                                             self.viewer.offset, scale=self.viewer.scale)
        self.point_size = 3

        self.regions: list | np.ndarray = []
        self.texts: list = []

        self.data_button_pressed()


    def on_close(self) -> None:
        """
        Triggered when the plugin window is closed.

        Deactivate the plugin and callbacks.
        """
        self.controller.plugins[PLUGIN_NAME]['activated'] = False

    def data_button_pressed(self) -> None:
        """
        Add regions and plots to the 3D view for the chosen probe.

        Called when the data button is pressed in the main application.
        """
        # Remove existing data
        for text in self.texts:
            text.delete()
        # TODO should delete the regions from the viewer
        self.remove_regions()
        self.texts = []

        # Find new regions
        regions = get_regions(self.controller)
        regions = np.unique(np.concatenate(regions))
        region_ids = self.controller.model.brain_atlas.regions.acronym2id(regions)
        keep_regions = []
        for rid, acr in zip(region_ids, regions):
            region_info = self.controller.model.brain_atlas.regions.ancestors(rid)
            if 'fiber tracts' not in region_info['acronym']:
                keep_regions.append(acr)

        self.add_regions(keep_regions)
        self.view.regions.setChecked(True)

        # Plot initial channels
        self.plot_channels(self.controller.probe_init)
        self.plot_picks()
        self.view.picks.setChecked(False)
        self.toggle_picks(False)

        self.view.panel.update()

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
        self.regions = [r for r in regions if r not in ['void', 'root']]
        self.side = 'left' if hemisphere == -1 else 'right'
        self.viewer.meshes.add_regions(self.regions, hemisphere=self.side)
        for reg in self.regions:
            self.viewer.meshes.set_alpha(60, reg)

        self.view.qfig.update_image()

    def toggle_regions(self, display: bool) -> None:
        """
        Show or hide the brain regions in the 3D view.

        Parameters
        ----------
        display: bool
            Whether to display the brain regions.
        """
        if len(self.regions) > 0:
            if display:
                self.viewer.meshes.show_regions(self.regions)
            else:
                self.viewer.meshes.hide_regions(self.regions)

        self.view.qfig.update_image()

    def remove_regions(self):
        """Remove all brain regions from the 3D view."""
        if len(self.regions) > 0:
            self.viewer.meshes.remove_regions(self.regions)
            self.regions = []

        self.view.qfig.update_image()

    def set_points(self, points: dict[str, np.ndarray]) -> None:
        """
        Add a set of points to the 3D view.

        Parameters
        ----------
        points: dict
            The position and color of the points to add.
        """
        self.viewer.points.add_points(points['pos'], points['col'], self.point_size)
        self.view.qfig.update_image()

    def set_picks(self, points: dict[str, np.ndarray]) -> None:
        """
        Add a set of pick points to the 3D view.

        Parameters
        ----------
        points: dict
            The position and color of the pick points to add.
        """
        self.viewer.picks.add_points(points['pos'], points['col'], self.point_size)
        self.view.qfig.update_image()

    def on_point_size_changed(self, slider: QtWidgets.QSlider) -> None:
        """
        Update the size of the points in the 3D view based on the slider value.

        Parameters
        ----------
        slider: QtWidgets.QSlider
            A slider widget with values from 1 to 10 representing point size.
        """
        self.set_point_size(slider.value())

    def set_point_size(self, point_size: float) -> None:
        """
        Set the size of the points in the 3D view.

        Parameters
        ----------
        point_size: float
            The size of the points to set.
        """
        self.viewer.points.set_size(point_size)
        self.view.qfig.update_image()

    def set_markers(self, markers: list) -> None:
        """
        Add markers and text to indicate shank positions in the 3D view.

        Parameters
        ----------
        markers: dict
            A list of marker positions, colors, and names.
        """
        for marker in markers:
            self.viewer.texts.add_text(marker['name'], marker['pos'], marker['col'], 1)


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

    def toggle_picks(self, display: bool) -> None:
        """
        Show or hide the pick points in the 3D view.

        Parameters
        ----------
        display: bool
            Whether to display the pick points.
        """
        if display:
            self.viewer.picks.show_points()
        else:
            self.viewer.picks.hide_points()

        self.view.qfig.update_image()

    def plot_picks(self) -> None:
        """
        Plot channel data in the 3D view.

        Parameters
        ----------
        plot_key: str
            The name of channel plot to display.
        """
        # TODO this only needs to be done onece
        data = get_xyz_picks(self.controller)
        colours = []
        positions = []

        for dat in data:

            if dat['xyz'] is None:
                continue

            colours.append(dat['values'])
            positions.append(dat['xyz'])

        positions = np.ascontiguousarray(np.vstack(positions).astype(np.float32))
        colours = np.ascontiguousarray(np.vstack(colours).astype(np.uint8))

        self.viewer.picks.add_points(positions, colours, 5)
        self.view.qfig.update_image()

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

            colours.append(dat['values'])
            positions.append(dat['xyz'])

            # Find the position to put the shank indicators
            min_idx = np.argmax(dat['xyz'][:, 2])

            sh_info = {'name': dat['shank'][-1],
                       'pos': [dat['xyz'][min_idx, 0], dat['xyz'][min_idx, 1], dat['xyz'][min_idx, 2] + 200 / 1e6],
                       'col': SHANK_COLOURS.get(dat['shank'][-1], create_random_color())}
            if self.controller.model.selected_config != 'both' or dat['config'] == 'quarter':
                markers.append(sh_info)

        if len(positions) != 0:
            positions = np.ascontiguousarray(np.vstack(positions).astype(np.float32))
            colours = np.ascontiguousarray(np.vstack(colours).astype(np.uint8))
            if len(positions) > 0:
                self.set_points({'pos': positions, 'col': colours})
                self.set_markers(markers)


def create_random_color() -> str:
    """
    Create a random hex color code.

    Returns
    -------
    str
        A hex color code.
    """
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return np.array([r, g, b, 255], dtype=np.uint8)
    # return rgb2hex((r / 255, g / 255, b / 255))


def data_to_colors(
        data: list | np.ndarray,
        cmap: str,
        vmin: float,
        vmax: float
) -> list:
    """
    Convert data values to RGBA color codes.

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
        A list of RGBA color codes corresponding to each data value.
    """
    cmap = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=mpl.colormaps[cmap])
    cvals = (cmap.to_rgba(data) * 255).astype(np.uint8)
    return cvals


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
def get_xyz_picks(_, items: 'ShankController', **kwargs) -> dict[str, Any]:
    """
    Get the xyz coordinates of the picks for the shank.

    Parameters
    ----------
    items: ShankController
        A ShankController instance containing model data.

    Returns
    -------
    np.ndarray
        An array of xyz coordinates of the picks.
    """
    xyz = items.model.xyz_picks
    values = np.array([[255, 0, 0, 255]] * xyz.shape[0], dtype=np.uint8)

    return {'xyz': xyz, 'values': values, 'shank': kwargs['shank'], 'config': kwargs['config']}


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
    jitter = np.random.uniform(-1 * 1e-5, 1 * 1e-5, size=xyz.shape)
    data = items.model.probe_plots.get(plot_key, None)
    if data is None or data.data is None:
        return {'xyz': None, 'values': None, 'shank': kwargs['shank'], 'config': kwargs['config']}

    values = data_to_colors(data.data, data.cmap, data.levels[0], data.levels[1])

    return {'xyz': xyz + jitter, 'values': values, 'shank': kwargs['shank'], 'config': kwargs['config']}
