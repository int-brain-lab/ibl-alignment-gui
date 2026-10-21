import importlib.util
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    AUTO_ALIGNMENT_CONFIDENCE_KEY,
    AUTO_ALIGNMENT_KEY,
    M_TO_UM,
    PredictionError,
    close_alignment_windows,
    get_region_boundaries,
    is_model_loaded,
    load_model_dialog,
    show_alignment_windows,
)
from ibl_alignment_gui.plugins.ephys_atlas.inference import CUMULATIVE_SUFFIX, DECODERS
from ibl_alignment_gui.utils.helpers import shank_loop
from iblutil.util import Bunch

# NB: ``spatial_encoder`` (torch) is imported lazily inside the compute functions below, and
# ``inference`` defers its ``ephysatlas`` imports, so this plugin can be set up in offline mode
# without those heavy/optional dependencies installed.

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController
    from iblatlas.atlas import AllenAtlas

PLUGIN_NAME = 'Channel Prediction'

# Region-plot dropdown options contributed by loadable models. Each entry maps the plugin-state
# key the backend caches its model under (its ``MODEL_NAME``) to the region keys it enables, plus
# whether torch is required. In offline mode an option is only shown once its model is loaded.
_MODEL_OPTIONS = (
    ('Encoding', [AUTO_ALIGNMENT_KEY, AUTO_ALIGNMENT_CONFIDENCE_KEY], True),
    *(
        (name, [name, name + CUMULATIVE_SUFFIX], decoder.needs_torch)
        for name, decoder in DECODERS.items()
    ),
)


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

    has_torch = importlib.util.find_spec('torch') is not None
    # One load action per region decoder; the torch ones only when torch is installed.
    menu_actions = [
        (f'Load {name} model (Hugging Face)…', partial(_load_decoder, name=name))
        for name, decoder in DECODERS.items()
        if has_torch or not decoder.needs_torch
    ]
    # The alignment model needs torch; only offer it when torch is installed.
    if has_torch:
        menu_actions.append(('Load alignment model (Hugging Face)…', _load_spatial_model))
    menu_actions.append(('Load features file…', _set_local_features))

    for label, handler in menu_actions:
        action = QtWidgets.QAction(label, controller.view)
        action.triggered.connect(lambda _=False, h=handler: h(controller))
        plugin_menu.addAction(action)

    if has_torch:
        # Re-opens the auto-alignment result windows; greyed out until a shank has been aligned.
        action = QtWidgets.QAction('Show auto-alignment results', controller.view)
        action.triggered.connect(lambda _=False: show_alignment_windows(controller))
        action.setEnabled(False)
        plugin_menu.addAction(action)
        controller.plugins[PLUGIN_NAME]['alignment_results_action'] = action

    controller.plugins[PLUGIN_NAME]['data_button_pressed'] = partial(_on_data_loaded, controller)


def _set_local_features(controller: 'AlignmentGUIController') -> None:
    """Prompt for a per-channel features parquet and use it for inference."""
    from ibl_alignment_gui.plugins.ephys_atlas import inference

    parent = controller.view
    chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent, 'Select per-channel features file', filter='Parquet (*.pqt *.parquet)'
    )
    if not chosen:
        return

    path = Path(chosen)
    loader = FeatureLoaderLocal(path)
    if not loader.load_features().get('exists', False):
        QtWidgets.QMessageBox.warning(parent, PLUGIN_NAME, f'No features found in:\n{path}')
        return

    controller.plugins[PLUGIN_NAME]['features_path'] = path
    # Inject into any already-loaded shanks so inference (and re-runs) use the new file; a single
    # combined file is split per shank via shank_sites['raw_ind']. If data is not loaded yet,
    # ephys_atlas._common._get_features_df will load (and split) it lazily from this path.
    for shank_dict in controller.model.shanks.values():
        for shank_handler in shank_dict.values():
            if getattr(shank_handler, 'raw_data', None) is not None:
                shank_sites = shank_handler.loaders['geom'].get_sites_for_shank(
                    shank_handler.shank_idx
                )
                shank_handler.raw_data['features'] = loader.load_features(shank_sites)
    inference.invalidate_predictions(controller)
    if is_model_loaded(controller, 'Encoding'):
        import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial

        spatial.invalidate_predictions(controller)
    logger.info('Local features file set to %s', path)


def _load_decoder(controller: 'AlignmentGUIController', name: str) -> None:
    """Load region decoder ``name`` via dialog; reveal its region options, refresh on success."""
    from ibl_alignment_gui.plugins.ephys_atlas import inference

    if load_model_dialog(controller, inference.DECODER_SPECS[name]):
        # Reveal the now-loaded model's region options (offline only adds them once loaded).
        _refresh_model_options(controller)
        controller.view.trigger_menu_option('region', name)


def _load_spatial_model(controller: 'AlignmentGUIController') -> None:
    """Load the alignment model via dialog; reveal its region options and refresh on success."""
    import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial

    if load_model_dialog(controller, spatial.ENCODER_SPEC):
        # Reveal the now-loaded model's region option (offline only adds it once loaded).
        _refresh_model_options(controller)
        controller.view.trigger_menu_option('region', AUTO_ALIGNMENT_KEY)


def _on_data_loaded(controller: 'AlignmentGUIController') -> None:
    """Data-load hook: reset per-session plugin state and expose available model region options.

    Runs on every data load (i.e. each new session). Drops any manual features override so it does
    not leak across sessions (each session supplies its own features), closes the previous
    session's auto-alignment result windows, then refreshes which model region-plot options are
    offered.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME]['features_path'] = None
    close_alignment_windows(controller)
    _refresh_model_options(controller)


def _refresh_model_options(controller: 'AlignmentGUIController') -> None:
    """Add the region-plot options for models that should currently be available.

    Online, every model's option is offered (the model loads on demand). Offline, an option is
    added only once its model has been loaded, so the dropdown never lists a model the user cannot
    run. Options already present are left untouched, so this is safe to call repeatedly (e.g. after
    each successful model load and on every data reload).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    new_keys = []
    for state_key, region_keys, needs_torch in _MODEL_OPTIONS:
        if needs_torch and importlib.util.find_spec('torch') is None:
            continue
        if controller.offline and not is_model_loaded(controller, state_key):
            continue
        new_keys.extend(
            key for key in region_keys if not controller.view.has_menu_option('region', key)
        )
    if new_keys:
        controller.view.populate_menu_tab(
            'region', controller.plot_region_ref_panels, new_keys, set_checked=False
        )


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
            'Beryl': partial(compute_mapping_predictions, mapping='Beryl'),
            'Cosmos': partial(compute_mapping_predictions, mapping='Cosmos'),
            AUTO_ALIGNMENT_KEY: compute_spatial_encoder_predictions,
            AUTO_ALIGNMENT_CONFIDENCE_KEY: compute_spatial_encoder_predictions,
        }
        # Each region decoder: its most probable region, and its cumulative probabilities.
        for name in DECODERS:
            self.func_map[name] = partial(compute_inference_predictions, name=name)
            self.func_map[name + CUMULATIVE_SUFFIX] = partial(
                compute_cumulative_predictions, name=name
            )

    def plot_regions(self, model: str, data_only: bool = True) -> None:
        """Compute and plot the selected region model across all shanks.

        Looks up ``model`` in the dispatch map and runs the matching ``compute_*`` function on
        each shank. When the user selects the auto-alignment option, every aligned shank's result
        window is shown again (if closed); a programmatic re-selection (e.g. after a reload) does
        not open windows.

        Parameters
        ----------
        model : str
            Region-model key (e.g. 'Beryl', 'Cosmos', 'Auto-alignment',
            'XGBoost region prediction', 'XGBoost region prediction (cumulative)'). Unknown keys
            are ignored.
        data_only : bool
            Reserved for signature compatibility with the region-plot callback; not used here.
        """
        self.controller.region_init = model
        func = self.func_map.get(model)
        if func is None:
            return

        errors = _plot_region_panels(self.controller, model, func)
        # One dialog for all shanks; the shanks that succeeded are still drawn.
        messages = list(dict.fromkeys(error for error in errors if error))
        if messages:
            QtWidgets.QMessageBox.warning(self.controller.view, model, '\n\n'.join(messages))


@shank_loop
def _plot_region_panels(
    controller: 'AlignmentGUIController',
    items: 'ShankController',
    model: str,
    func: Callable[..., dict[str, Bunch] | None],
    **kwargs,
) -> str | None:
    """Compute (and cache) a shank's prediction for ``model`` and draw it.

    Decorated with :func:`shank_loop`, so a single call iterates over every shank/config (the
    injected ``shank``/``config`` keywords are absorbed via ``**kwargs``). The prediction is
    computed once per shank and cached on ``items.model.predictions``; cumulative results are
    drawn as stacked bands, others as a region histology column. Every view ``func`` returns is
    cached, so a model's other region-plot options are later drawn without computing again.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank to compute and draw.
    model : str
        Region-model key, also the per-shank prediction cache key.
    func : Callable
        The ``compute_*`` function for ``model``: it returns the views of every region-plot
        option one run of its model gives, ``{option: view}`` (including ``model``), or None
        when the user cancelled.

    Returns
    -------
    str or None
        The error message when the prediction failed, else None. Errors are returned rather than
        shown so that the caller reports them once for all shanks.
    """
    if not getattr(items.model, 'predictions', None):
        items.model.predictions = Bunch()

    if items.model.predictions.get(model) is None:
        try:
            views = func(controller, items)
        except PredictionError as exc:
            return str(exc)
        if views is not None:
            items.model.predictions.update(views)

    pred = items.model.predictions.get(model)
    if pred is not None:
        if 'probability' in pred:
            items.view.plot_histology_cumulative(items.view.fig_hist_ref, pred)
        else:
            items.view.plot_histology(
                items.view.fig_hist_ref, pred, ax='right', boundaries=pred.get('boundaries', True)
            )


def compute_mapping_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController', mapping: str = 'Beryl'
) -> dict[str, Bunch]:
    """
    Return brain regions based on a specified atlas mapping, as an example prediction model.

    Parameters
    ----------
    controller: 'AlignmentGUIController'
        The main application controller.
    items: 'ShankController'
        The shank controller containing the model and view for the current shank.
    mapping: str
        The atlas mapping to use for predictions (e.g., 'Beryl' or 'Cosmos'), also its
        region-plot option.

    Returns
    -------
    dict of {str: Bunch}
        The predicted brain regions, keyed by the region-plot option (the mapping).
    """
    # xyz coordinates sampled at 10 um along histology track from bottom or brain to top
    xyz_samples = items.model.align_handle.xyz_samples
    # depths of these coordinates along the track
    depth_samples = items.model.align_handle.ephysalign.sampling_trk

    region_ids = controller.model.brain_atlas.get_labels(xyz_samples, mapping=mapping)
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return {mapping: get_region_boundaries(regions, depth_samples)}


def _compute_region_id_predictions(
    controller: 'AlignmentGUIController',
    items: 'ShankController',
    predict: Callable[['AlignmentGUIController', 'ShankController'], tuple | None],
    depth_scale: float = 1.0,
) -> Bunch[str, np.ndarray] | None:
    """Run a region-id ``predict`` callable and convert its output to region boundaries.

    Shared pipeline for the model-backed predictors: it calls ``predict`` (which returns
    ``(region_ids, depths)`` or ``None``), maps the ids to atlas regions, and reduces them to
    contiguous region boundaries. ``depth_scale`` converts the predictor's native depth unit to
    meters (the unit expected by :func:`get_region_boundaries`).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.
    predict : Callable
        Predictor returning ``(region_ids, depths)`` for the shank, or ``None`` if unavailable.
    depth_scale : float
        Multiplier converting the predictor's depths to meters.

    Returns
    -------
    Bunch or None
        The predicted brain regions along the probe, or None if no prediction is available.
    """
    result = predict(controller, items)
    if result is None:
        return None
    region_ids, depths = result
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depths * depth_scale)


def compute_spatial_encoder_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
) -> dict[str, Bunch] | None:
    """
    Automatic alignment with the Ephys Atlas channel-level model.

    Warps the shank's recorded features along its histology track (behind a progress dialog,
    opening the result figure in a pop-up window) and returns both of its region-plot views: the
    predicted regions and the alignment confidence along the probe.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.

    Returns
    -------
    dict of {str: Bunch} or None
        Both views, keyed by their region-plot options ('Auto-alignment' and
        'Auto-alignment (confidence)'), or None if no alignment is available.
    """
    # Lazy import: pulls in torch + the alignment model.
    import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial

    return spatial.align_shank(controller, items)


def compute_inference_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController', name: str
) -> dict[str, Bunch] | None:
    """
    Prediction model using a region decoder (its most probable region per channel).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.
    name : str
        The region decoder, a key of ``inference.DECODERS``, also its region-plot option.

    Returns
    -------
    dict of {str: Bunch} or None
        The predicted brain regions along the probe, keyed by the region-plot option, or None if
        no prediction is available.
    """
    # Lazy import: ephysatlas is an optional dependency, only needed when inference runs.
    from ibl_alignment_gui.plugins.ephys_atlas import inference

    result = inference.predict(controller, items, name)
    if result is None:
        return None

    regions, depths = result

    return {name: get_region_boundaries(regions, depths / M_TO_UM)}


def compute_cumulative_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController', name: str
) -> dict[str, Bunch] | None:
    """
    Cumulative prediction model using a region decoder.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.
    name : str
        The region decoder, a key of ``inference.DECODERS``.

    Returns
    -------
    dict of {str: Bunch} or None
        The probability of predicted brain regions along the probe, keyed by the region-plot
        option (``name`` + ``' (cumulative)'``), or None if no prediction is available.
    """
    # Lazy import: ephysatlas is an optional dependency, only needed when inference runs.
    from ibl_alignment_gui.plugins.ephys_atlas import inference

    result = inference.predict_cumulative(controller, items, name)
    if result is None:
        return None

    cprobas, depths, colours, regions = result
    data = Bunch(depths=depths, regions=regions, colours=colours, probability=cprobas)

    return {name + CUMULATIVE_SUFFIX: data}
