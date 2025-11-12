from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.utils.utils import shank_loop
from iblutil.util import Bunch

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController

PLUGIN_NAME = 'Channel Prediction'


def setup(controller: 'AlignmentGUIController') -> None:
    """
    Set up the Channel Prediction plugin.

    Adds menu options to select different prediction models.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = Bunch()
    channel_prediction = ChannelPrediction(controller)
    controller.plugins[PLUGIN_NAME]['loader'] = channel_prediction

    # Add menu bar for selecting what to show
    controller.plugins[PLUGIN_NAME]['activated'] = False

    # Add a submenu to the main menu
    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, controller.view)
    controller.plugin_options.addMenu(plugin_menu)

    action_group = QtWidgets.QActionGroup(plugin_menu)
    action_group.setExclusive(True)

    # Add the different prediction model options

    predictions_models = {
        'Original': None,
        'Cosmos': compute_cosmos_predictions,
        'Random': compute_random_predictions
    }

    for model, model_func in predictions_models.items():
        action = QtWidgets.QAction(model, controller.view)
        action.setCheckable(True)
        action.setChecked(model == 'Original')
        action.triggered.connect(lambda _, m=model, func=model_func:
                                 channel_prediction.plot_regions(_, m, func))
        action_group.addAction(action)
        plugin_menu.addAction(action)


class ChannelPrediction:
    """
    Class to handle channel prediction plotting in the alignment GUI.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """

    def __init__(self, controller: 'AlignmentGUIController') -> None:
        self.controller = controller

    def plot_regions(self, _, model: str, func: Callable) -> None:
        """
        Plot the brain regions based on the selected model.

        Parameters
        ----------
        model: str
            The name of the model to use for predictions.
        func: Callable
            The function to compute the predictions.
        """
        # Plot the regions based on the action
        if model == 'Original':
            plot_original_regions(self.controller)
        else:
            plot_predicted_regions(self.controller, model, func)


@shank_loop
def plot_original_regions(_, items: 'ShankController', **kwargs) -> None:
    """Plot the original histology regions on the reference histology plot."""
    items.view.plot_histology(items.view.fig_hist_ref, items.model.hist_data_ref, ax='right')


@shank_loop
def plot_predicted_regions(
        controller: 'AlignmentGUIController',
        items: 'ShankController',
        model: str,
        func: Callable,
        **kwargs
) -> None:
    """
    Plot the model predictions on the reference histology plot.

    Parameters
    ----------
    model: str
        The name of the model.
    func: Callable
        The function to compute the predictions.
    """
    if not getattr(items.model, 'predictions', None):
        items.model.predictions = Bunch()

    results = items.model.predictions.get(model, None)
    if results is None:
        items.model.predictions[model] = func(controller, items)

    items.view.plot_histology(items.view.fig_hist_ref, items.model.predictions[model], ax='right')


def compute_cosmos_predictions(
        controller: 'AlignmentGUIController',
        items: 'ShankController'
) -> Bunch[str, np.ndarray]:
    """
    Example prediction model that returns cosmos brain regions.

    Returns
    -------
    Bunch
        A bunch containing the predicted brain regions.
    """
    # xyz coordinates sampled at 10 um along histology track from bottom or brain to top
    xyz_samples = items.model.align_handle.xyz_samples
    # depths of these coordinates along the track
    depth_samples = items.model.align_handle.ephysalign.sampling_trk

    region_ids = controller.model.brain_atlas.get_labels(xyz_samples, mapping='Cosmos')
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depth_samples)


def compute_random_predictions(
        controller: 'AlignmentGUIController',
        items: 'ShankController'
) -> Bunch[str, np.ndarray]:
    """
    Example prediction model that uses the spikes data to assign random brain regions.

    Returns
    -------
    Bunch
        A bunch containing the predicted brain regions.
    """
    # xyz coordinates sampled at 10 um along histology track from bottom or brain to top
    xyz_samples = items.model.align_handle.xyz_samples
    # depths of these coordinates along the track
    depth_samples = items.model.align_handle.ephysalign.sampling_trk

    # Spikes and other data can be accessed in this way if needed
    spikes = items.model.raw_data['spikes']

    def random_chunked_array(n, n_vals=20, seed=None):
        rng = np.random.default_rng(seed)
        cuts = np.sort(rng.choice(np.arange(1, n), size=n_vals - 1, replace=False))
        chunks = np.diff(np.r_[0, cuts, n])
        vals = rng.choice(np.arange(1001), size=n_vals, replace=False)
        chosen = rng.choice(vals, size=len(chunks), replace=True)
        return np.repeat(chosen, chunks)

    random = random_chunked_array(len(depth_samples), n_vals=20, seed=42)
    region_ids = controller.model.brain_atlas.regions.id[random]
    regions = controller.model.brain_atlas.regions.get(region_ids)
    return get_region_boundaries(regions, depth_samples)


def get_region_boundaries(regions: dict, depths: np.ndarray) -> Bunch[str, np.ndarray]:
    """
    Get the boundaries of brain regions along the histology track.

    Parameters
    ----------
    regions: dict
        The brain regions along the histology track.
    depths:
        The depths along the histology track.

    Returns
    -------
    Bunch
        A bunch containing the region boundaries, labels, and colours.
    """
    boundaries = np.where(np.diff(regions.id))[0]

    n_regions = len(boundaries) + 1
    region = np.empty((n_regions, 2))
    region_label = np.empty((n_regions, 2), dtype=object)
    region_colour = np.empty((n_regions, 3), dtype=int)

    for i in range(n_regions):
        # Compute start and end indices for this region
        start = 0 if i == 0 else boundaries[i - 1] + 1
        end = boundaries[i] if i < len(boundaries) else regions.id.size - 1

        region[i, :] = depths[[start, end]] * 1e6
        region_label[i, :] = (np.mean(depths[[start, end]]) * 1e6, regions.acronym[end])
        region_colour[i, :] = regions.rgb[end]

    data = Bunch(
        region=region,
        axis_label=region_label,
        colour=region_colour
    )

    return data
