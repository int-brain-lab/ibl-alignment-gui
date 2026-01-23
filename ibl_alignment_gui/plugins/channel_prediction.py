from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.utils.utils import shank_loop
from iblutil.util import Bunch
from iblatlas.atlas import AllenAtlas

import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial
import ibl_alignment_gui.plugins.ephys_atlas.inference as inference

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController

PLUGIN_NAME = 'Channel Prediction'


def setup(controller: 'AlignmentGUIController') -> None:
    controller.plugins[PLUGIN_NAME] = Bunch()
    controller.plugins[PLUGIN_NAME]['activated'] = True

    channel_prediction = ChannelPrediction(controller)
    controller.plugins[PLUGIN_NAME]['loader'] = channel_prediction

    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, controller.view)
    controller.plugin_options.addMenu(plugin_menu)

    action_group = QtWidgets.QActionGroup(plugin_menu)
    action_group.setExclusive(True)

    predictions_models = {
        'Original': None,
        'Cosmos': compute_cosmos_predictions,
        'Spatial Encoder': compute_spatial_encoder_predictions,
        'Inference Model': compute_inference_predictions,
        'Inference Cumulative': compute_cumulative_predictions
    }

    for model, model_func in predictions_models.items():
        action = QtWidgets.QAction(model, controller.view)
        action.setCheckable(True)
        action.setChecked(model == 'Original')
        action.triggered.connect(lambda _, m=model, func=model_func:
                                 channel_prediction.plot_regions(_, m, func))
        action_group.addAction(action)
        plugin_menu.addAction(action)

    controller.plugins[PLUGIN_NAME]['data_button_pressed'] = lambda: callback(action_group)


def callback(group) -> None:
    """Reset action group to 'Original' selection."""
    group.setEnabled(False)
    for action in group.actions():
        if action.text() == 'Original':
            action.setChecked(True)
        else:
            action.setChecked(False)
    group.setEnabled(True)


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
        self.ba: AllenAtlas = self.controller.model.brain_atlas

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

    if 'probability' in items.model.predictions[model]:
        items.view.plot_histology_cumulative(items.view.fig_hist_ref,items.model.predictions[model])
    else:
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


def compute_spatial_encoder_predictions(
    controller: 'AlignmentGUIController',
    items: 'ShankController'
) -> Bunch[str, np.ndarray]:
    """
    Prediction model using the spatial encoder.

    Returns
    -------
    Bunch
        The predicted brain regions along the probe.
    """

    region_ids, depths = spatial.predict(controller, items)
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depths)


def compute_inference_predictions(
    controller: 'AlignmentGUIController',
    items: 'ShankController'
) -> Bunch[str, np.ndarray]:
    """
    Prediction model using the inference model.

    Returns
    -------
    Bunch
        The predicted brain regions along the probe.
    """

    region_ids, depths = inference.predict(controller, items)
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depths / 1e6)


def compute_cumulative_predictions(
        controller: 'AlignmentGUIController',
        items: 'ShankController'
) -> Bunch[str, np.ndarray]:
    """
    Cumulative prediction model using the inference model.
    Returns
    -------
    Bunch
        A bunch containing the probability of predicted brain regions along the probe.
    """

    cprobas, depths, colours, regions = inference.predict_cumulative(controller, items)

    data = Bunch(
        depths=depths,
        regions=regions,
        colours=colours,
        probability=cprobas
    )

    return data


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
