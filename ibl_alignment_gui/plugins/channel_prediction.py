import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal
from ibl_alignment_gui.utils.utils import shank_loop
from iblutil.util import Bunch
from iblatlas.atlas import AllenAtlas

# NB: ``spatial_encoder`` (torch) and ``inference`` (ephysatlas) are imported lazily inside the
# compute functions below so this plugin can be set up in offline mode without those heavy/optional
# dependencies installed.

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController

PLUGIN_NAME = 'Channel Prediction'


def setup(controller: 'AlignmentGUIController') -> None:
    """Register the Channel Prediction plugin and (when available) its menu.

    Always installs the plugin state and its :class:`ChannelPrediction` loader. When ``ephysatlas``
    is importable, also adds the "Channel Prediction" menu (load inference/spatial models, load a
    features file) and registers a data-loaded callback that exposes the model region options.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = Bunch()
    controller.plugins[PLUGIN_NAME]['activated'] = True
    channel_prediction = ChannelPrediction(controller)
    controller.plugins[PLUGIN_NAME]['loader'] = channel_prediction

    if importlib.util.find_spec('ephysatlas') is None:
        return


    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, controller.view)
    controller.plugin_options.addMenu(plugin_menu)

    for label, handler in (
        ('Load inference model', _load_inference_model),
        ('Load spatial model',   _load_spatial_model),
        ('Load features file…',  _set_local_features),
    ):
        action = QtWidgets.QAction(label, controller.view)
        action.triggered.connect(lambda _=False, h=handler: h(controller))
        plugin_menu.addAction(action)

    # TODO add these so they are only added once the model has been loaded
    def _add_model_options(controller=controller):
        # All models are offline-capable from local assets: the inference (xgboost) model via a
        # local model dir, and the Spatial Encoder via a local encoder dir + bank dir (set through
        # the dialogs above). They fall back to S3/ONE only when no local source is set.
        model_keys = []
        if importlib.util.find_spec('torch') is not None:
            model_keys.append('Spatial Encoder')
        model_keys.append('Inference Model')
        model_keys.append('Inference Cumulative')
        controller.view.populate_menu_tab(
            'region', controller.plot_region_ref_panels, model_keys, set_checked=False
        )

    controller.plugins[PLUGIN_NAME]['data_button_pressed'] = _add_model_options

def _set_local_features(controller: 'AlignmentGUIController') -> None:
    # TODO we need to make this work with 4 shanks, if the feature files are all in individual folders
    # Alternatively if it is one feature file we need to split per shank
    """Prompt for a per-channel features parquet and use it for inference."""
    import ibl_alignment_gui.plugins.ephys_atlas.inference as inference
    parent = controller.view
    chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent, 'Select per-channel features file', filter='Parquet (*.pqt *.parquet)')
    if not chosen:
        return

    path = Path(chosen)
    feats = FeatureLoaderLocal(path).load_features()
    if not feats.get('exists', False):
        QtWidgets.QMessageBox.warning(parent, PLUGIN_NAME, f'No features found in:\n{path}')
        return

    controller.plugins[PLUGIN_NAME]['features_path'] = path
    # Inject into any already-loaded shanks so inference (and re-runs) use the new file; if data
    # is not loaded yet, ephys_atlas.inference._get_features_df will load it lazily from this path.
    for shank_dict in controller.model.shanks.values():
        for shank_handler in shank_dict.values():
            if getattr(shank_handler, 'raw_data', None) is not None:
                shank_handler.raw_data['features'] = feats
    inference.invalidate_predictions(controller)
    logger.info('Local features file set to %s', path)


def _load_inference_model(controller: 'AlignmentGUIController') -> None:
    """Load inference model via GUI dialog; invalidate cache and refresh on success."""
    import ibl_alignment_gui.plugins.ephys_atlas.inference as inference
    if inference.load_model_dialog(controller):
        controller.view.trigger_menu_option('region', inference.PREDICTION_KEY)


def _load_spatial_model(controller: 'AlignmentGUIController') -> None:
    """Load the spatial model via dialog (builds the engine); invalidate cache and refresh."""
    import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial
    if spatial.load_model_dialog(controller):
        controller.view.trigger_menu_option('region', spatial.PREDICTION_KEY)


class ChannelPrediction:
    """Plugin loader that computes and plots per-channel region predictions.

    Registered as the Channel Prediction plugin's ``loader``; :meth:`plot_regions` dispatches the
    selected region model to the matching ``compute_*`` function and draws it on each shank.
    """

    def __init__(self, controller: 'AlignmentGUIController') -> None:
        """Store the controller and cache its brain atlas.

        Parameters
        ----------
        controller : AlignmentGUIController
            The main application controller.
        """
        self.controller = controller
        self.ba: AllenAtlas = self.controller.model.brain_atlas
        self.func_map = {
            'Beryl': compute_beryl_predictions,
            'Cosmos': compute_cosmos_predictions,
            'Spatial Encoder': compute_spatial_encoder_predictions,
            'Inference Model': compute_inference_predictions,
            'Inference Cumulative': compute_cumulative_predictions,
        }

    def plot_regions(self, model: str, data_only: bool = True) -> None:
        """Compute and plot the selected region model across all shanks.

        Looks up ``model`` in the dispatch map and runs the matching ``compute_*`` function on
        each shank.
        
        Parameters
        ----------
        model : str
            Region-model key (e.g. 'Beryl', 'Cosmos', 'Spatial Encoder', 'Inference Model',
            'Inference Cumulative'). Unknown keys are ignored.
        data_only : bool
            Reserved for signature compatibility with the region-plot callback; not used here.
        """
        self.controller.region_init = model
        func = self.func_map.get(model)
        if func is None:
            return

        _plot_region_panels(self.controller, model, func)



@shank_loop
def _plot_region_panels(
    controller: 'AlignmentGUIController',
    items: 'ShankController',
    model: str,
    func: Callable,
    **kwargs,
) -> None:
    """Compute (and cache) a shank's prediction for ``model`` and draw it.

    Decorated with :func:`shank_loop`, so a single call iterates over every shank/config (the
    injected ``shank``/``config`` keywords are absorbed via ``**kwargs``). The prediction is
    computed once per shank and cached on ``items.model.predictions``; cumulative results are
    drawn as stacked bands, others as a region histology column.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank to compute and draw.
    model : str
        Region-model key, also the per-shank prediction cache key.
    func : Callable
        The ``compute_*`` function producing the prediction Bunch for ``model``.
    """
    if not getattr(items.model, 'predictions', None):
        items.model.predictions = Bunch()

    if items.model.predictions.get(model) is None:
        items.model.predictions[model] = func(controller, items)

    pred = items.model.predictions[model]
    if pred is not None:
        if 'probability' in pred:
            items.view.plot_histology_cumulative(items.view.fig_hist_ref, pred)
        else:
            items.view.plot_histology(items.view.fig_hist_ref, pred, ax='right')


def compute_mapping_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController', mapping: str = 'Beryl'
) -> Bunch[str, np.ndarray]:
    """
    Example prediction model that returns brain regions based on a specified atlas mapping.

    Parameters
    ----------
    controller: 'AlignmentGUIController'
        The main application controller.
    items: 'ShankController'
        The shank controller containing the model and view for the current shank.
    mapping: str
        The atlas mapping to use for predictions (e.g., 'Beryl' or 'Cosmos').

    Returns
    -------
    Bunch
        A bunch containing the predicted brain regions.
    """

    # xyz coordinates sampled at 10 um along histology track from bottom or brain to top
    xyz_samples = items.model.align_handle.xyz_samples
    # depths of these coordinates along the track
    depth_samples = items.model.align_handle.ephysalign.sampling_trk

    region_ids = controller.model.brain_atlas.get_labels(xyz_samples, mapping=mapping)
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depth_samples)


def compute_cosmos_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
) -> Bunch[str, np.ndarray]:
    """
    Example prediction model that returns cosmos brain regions.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.

    Returns
    -------
    Bunch
        A bunch containing the predicted brain regions.
    """
    return compute_mapping_predictions(controller, items, mapping='Cosmos')


def compute_beryl_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
) -> Bunch[str, np.ndarray]:
    """
    Example prediction model that returns beryl brain regions.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.

    Returns
    -------
    Bunch
        A bunch containing the predicted brain regions.
    """
    return compute_mapping_predictions(controller, items, mapping='Beryl')


def compute_spatial_encoder_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
) -> Bunch[str, np.ndarray] | None:
    """
    Prediction model using the spatial encoder.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.

    Returns
    -------
    Bunch or None
        The predicted brain regions along the probe, or None if no prediction is available.
    """
    # Lazy import: pulls in torch + the spatial encoder model; online-only.
    import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial

    result = spatial.predict(controller, items)
    if result is None:
        return
    region_ids, depths = result
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depths)


def compute_inference_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
) -> Bunch[str, np.ndarray] | None:
    """
    Prediction model using the inference model.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.

    Returns
    -------
    Bunch or None
        The predicted brain regions along the probe, or None if no prediction is available.
    """
    # Lazy import: ephysatlas is an optional dependency, only needed when inference runs.
    import ibl_alignment_gui.plugins.ephys_atlas.inference as inference

    result = inference.predict(controller, items)
    if result is None:
        return

    region_ids, depths = result
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depths / 1e6)


def compute_cumulative_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
) -> Bunch[str, np.ndarray] | None:
    """
    Cumulative prediction model using the inference model.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.

    Returns
    -------
    Bunch or None
        A bunch containing the probability of predicted brain regions along the probe, or None if
        no prediction is available.
    """
    # Lazy import: ephysatlas is an optional dependency, only needed when inference runs.
    import ibl_alignment_gui.plugins.ephys_atlas.inference as inference

    result = inference.predict_cumulative(controller, items)
    if result is None:
        return

    cprobas, depths, colours, regions = result
    data = Bunch(depths=depths, regions=regions, colours=colours, probability=cprobas)

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

    data = Bunch(region=region, axis_label=region_label, colour=region_colour)

    return data
