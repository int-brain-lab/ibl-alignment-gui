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
    AUTO_ALIGNMENT_KEY,
    DISPLAY_CONFIDENCE,
    DISPLAY_REGIONS,
    DISPLAY_STATE_KEY,
    auto_alignment_display,
    is_model_loaded,
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

# Track depths are stored along the histology track in meters; region plots expect microns.
M_TO_UM = 1e6

# Region-plot dropdown options contributed by loadable models. Each entry maps the plugin-state
# key the backend caches its model under (its ``MODEL_NAME``) to the region keys it enables, plus
# whether torch is required. In offline mode an option is only shown once its model is loaded.
_MODEL_OPTIONS = (
    ('Encoding', [AUTO_ALIGNMENT_KEY], True),
    *(
        (name, [name, name + CUMULATIVE_SUFFIX], decoder.needs_torch)
        for name, decoder in DECODERS.items()
    ),
)
# Colourings of the auto-alignment column: (menu label, display mode).
_DISPLAY_OPTIONS = (
    ('Regions (IBL colours)', DISPLAY_REGIONS),
    ('Confidence (green high → red low)', DISPLAY_CONFIDENCE),
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

    if importlib.util.find_spec('torch') is not None:
        _add_display_menu(controller, plugin_menu)

    controller.plugins[PLUGIN_NAME]['data_button_pressed'] = partial(_on_data_loaded, controller)


def _add_display_menu(controller: 'AlignmentGUIController', plugin_menu: QtWidgets.QMenu) -> None:
    """Add the 'Auto-alignment colours' submenu: predicted regions or alignment confidence."""
    controller.plugins[PLUGIN_NAME][DISPLAY_STATE_KEY] = DISPLAY_REGIONS
    menu = plugin_menu.addMenu('Auto-alignment colours')
    group = QtWidgets.QActionGroup(menu)
    group.setExclusive(True)
    for label, mode in _DISPLAY_OPTIONS:
        action = QtWidgets.QAction(label, menu)
        action.setCheckable(True)
        action.setChecked(mode == DISPLAY_REGIONS)
        action.triggered.connect(
            lambda _=False, m=mode: _set_auto_alignment_display(controller, m)
        )
        group.addAction(action)
        menu.addAction(action)
    controller.plugins[PLUGIN_NAME]['display_actions'] = group


def _set_auto_alignment_display(controller: 'AlignmentGUIController', mode: str) -> None:
    """Colour the auto-alignment column by ``mode``; redraw it (without re-aligning) if shown."""
    if auto_alignment_display(controller) == mode:
        return
    controller.plugins[PLUGIN_NAME][DISPLAY_STATE_KEY] = mode
    if not is_model_loaded(controller, 'Encoding'):
        return
    import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial

    spatial.clear_display(controller)
    if controller.region_init == AUTO_ALIGNMENT_KEY:
        _plot_region_panels(controller, AUTO_ALIGNMENT_KEY, compute_spatial_encoder_predictions)


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

    if inference.load_model_dialog(controller, name):
        # Reveal the now-loaded model's region options (offline only adds them once loaded).
        _refresh_model_options(controller)
        controller.view.trigger_menu_option('region', name)


def _load_spatial_model(controller: 'AlignmentGUIController') -> None:
    """Load the alignment model via dialog; reveal its region options and refresh on success."""
    import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial

    if spatial.load_model_dialog(controller):
        # Reveal the now-loaded model's region option (offline only adds it once loaded).
        _refresh_model_options(controller)
        controller.view.trigger_menu_option('region', spatial.PREDICTION_KEY)


def _on_data_loaded(controller: 'AlignmentGUIController') -> None:
    """Data-load hook: reset per-session plugin state and expose available model region options.

    Runs on every data load (i.e. each new session). Drops any manual features override so it does
    not leak across sessions (each session supplies its own features), then refreshes which model
    region-plot options are offered.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME]['features_path'] = None
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

        _plot_region_panels(self.controller, model, func)

        user_selected = not getattr(self.controller.view, 'triggering_menu_option', False)
        if model == AUTO_ALIGNMENT_KEY and user_selected:
            import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial

            spatial.show_result_windows(self.controller)


@shank_loop
def _plot_region_panels(
    controller: 'AlignmentGUIController',
    items: 'ShankController',
    model: str,
    func: Callable[..., Bunch | None],
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
            items.view.plot_histology(
                items.view.fig_hist_ref, pred, ax='right', boundaries=pred.get('boundaries', True)
            )


def compute_mapping_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController', mapping: str = 'Beryl'
) -> Bunch[str, np.ndarray]:
    """
    Return brain regions based on a specified atlas mapping, as an example prediction model.

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
) -> Bunch[str, np.ndarray] | None:
    """
    Automatic alignment with the Ephys Atlas channel-level model.

    Warps the shank's recorded features along its histology track (once per shank, behind a
    progress dialog, opening the result figure in a pop-up window) and returns the column to draw:
    the predicted regions, or the alignment confidence along the probe, per the 'Auto-alignment
    colours' menu.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank controller containing the model and view for the current shank.

    Returns
    -------
    Bunch or None
        The column along the probe, or None if no alignment is available.
    """
    # Lazy import: pulls in torch + the alignment model.
    import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial

    if auto_alignment_display(controller) == DISPLAY_CONFIDENCE:
        return spatial.predict_confidence(controller, items)
    result = spatial.predict_regions(controller, items)
    if result is None:
        return
    region_ids, depths = result
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depths)


def compute_inference_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController', name: str
) -> Bunch[str, np.ndarray] | None:
    """
    Prediction model using a region decoder (its most probable region per channel).

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
    Bunch or None
        The predicted brain regions along the probe, or None if no prediction is available.
    """
    # Lazy import: ephysatlas is an optional dependency, only needed when inference runs.
    from ibl_alignment_gui.plugins.ephys_atlas import inference

    result = inference.predict(controller, items, name)
    if result is None:
        return

    regions, depths = result

    return get_region_boundaries(regions, depths / M_TO_UM)


def compute_cumulative_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController', name: str
) -> Bunch[str, np.ndarray] | None:
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
    Bunch or None
        A bunch containing the probability of predicted brain regions along the probe, or None if
        no prediction is available.
    """
    # Lazy import: ephysatlas is an optional dependency, only needed when inference runs.
    from ibl_alignment_gui.plugins.ephys_atlas import inference

    result = inference.predict_cumulative(controller, items, name)
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
    depths: np.ndarray
        The depths along the histology track, in meters.

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

        region[i, :] = depths[[start, end]] * M_TO_UM
        region_label[i, :] = (np.mean(depths[[start, end]]) * M_TO_UM, regions.acronym[end])
        region_colour[i, :] = regions.rgb[end]

    data = Bunch(region=region, axis_label=region_label, colour=region_colour)

    return data
