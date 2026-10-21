import importlib.util
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy import QtWidgets

from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal
from ibl_alignment_gui.plugins.ephys_atlas import spatial_encoder as spatial
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    PredictionError,
    close_alignment_windows,
    get_model,
    get_region_boundaries,
    is_model_loaded,
    load_model_dialog,
    show_alignment_windows,
)
from ibl_alignment_gui.plugins.ephys_atlas.inference import DECODER_SPECS
from ibl_alignment_gui.utils.helpers import shank_loop
from iblutil.util import Bunch

# NB: both backends import ``ephysatlas`` (and torch) only when a model is used, so this plugin
# can be set up without those heavy/optional dependencies installed.

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController
    from ibl_alignment_gui.plugins.ephys_atlas._common import ModelSpec
    from iblatlas.atlas import AllenAtlas

PLUGIN_NAME = 'Channel Prediction'

# Every model the plugin offers, as the backends describe them (see ModelSpec).
MODELS = (*DECODER_SPECS.values(), spatial.ENCODER_SPEC)
# The model behind each of their region-plot options.
MODEL_FOR_OPTION = {option: spec for spec in MODELS for option in spec.options}


def has_torch() -> bool:
    """Return whether torch is installed, without importing it."""
    return importlib.util.find_spec('torch') is not None


def available_models() -> list['ModelSpec']:
    """Return the models offered: those needing torch only when it is installed."""
    torch = has_torch()
    return [spec for spec in MODELS if torch or not spec.needs_torch]


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

    # One load action per model; those needing torch only when it is installed.
    models = available_models()
    menu_actions = [
        (f'{spec.title} (Hugging Face)…', partial(_load_model, spec=spec)) for spec in models
    ]
    menu_actions.append(('Load features file…', _set_local_features))

    for label, handler in menu_actions:
        action = QtWidgets.QAction(label, controller.view)
        action.triggered.connect(lambda _=False, h=handler: h(controller))
        plugin_menu.addAction(action)

    if spatial.ENCODER_SPEC in models:
        # Re-opens the auto-alignment result windows; greyed out until a shank has been aligned.
        action = QtWidgets.QAction('Show auto-alignment results', controller.view)
        action.triggered.connect(lambda _=False: show_alignment_windows(controller))
        action.setEnabled(False)
        plugin_menu.addAction(action)
        controller.plugins[PLUGIN_NAME]['alignment_results_action'] = action

    controller.plugins[PLUGIN_NAME]['data_button_pressed'] = partial(_on_data_loaded, controller)


def _set_local_features(controller: 'AlignmentGUIController') -> None:
    """Prompt for a per-channel features parquet and use it for inference."""
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
    _invalidate_loaded_models(controller)
    logger.info('Local features file set to %s', path)


def _invalidate_loaded_models(controller: 'AlignmentGUIController') -> None:
    """Drop every loaded model's cached predictions, e.g. after a new features file."""
    for spec in MODELS:
        if is_model_loaded(controller, spec.key):
            spec.invalidate(controller)


def _load_model(controller: 'AlignmentGUIController', spec: 'ModelSpec') -> None:
    """Load ``spec``'s model via its dialog and show its first region-plot option."""
    if load_model_dialog(controller, spec):
        _refresh_model_options(controller)  # make sure its options are offered
        controller.view.trigger_menu_option('region', spec.options[0])


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
    """Add the region-plot options of every available model, the same online and offline.

    A model is loaded when one of its options is first selected (or from the Plugins menu). Models
    that need torch are offered only when it is installed (:func:`available_models`). Options
    already present are left untouched, so this is safe to call repeatedly (e.g. after each model
    load and on every data reload).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    new_keys = [
        option
        for spec in available_models()
        for option in spec.options
        if not controller.view.has_menu_option('region', option)
    ]
    if new_keys:
        controller.view.populate_menu_tab(
            'region', controller.plot_region_ref_panels, new_keys, set_checked=False
        )


class ChannelPrediction:
    """Plugin loader that computes and plots per-channel region predictions.

    Registered as the Channel Prediction plugin's ``loader``; :meth:`plot_regions` dispatches the
    selected region option to its compute function and draws it on each shank.
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
            # One model run gives the views of all of the model's options.
            **{option: spec.compute for option, spec in MODEL_FOR_OPTION.items()},
        }

    def plot_regions(self, model: str, data_only: bool = True) -> None:
        """Compute and plot the selected region model across all shanks.

        Looks up ``model`` in the dispatch map and runs the matching compute function on each
        shank (an atlas mapping, or the model's ``compute``), getting the option's model once
        first. Problems are reported in one warning for all shanks.

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

        # Get the option's model once, before the shanks: its load dialog (or a failing preset
        # source) then happens once, and cancelling it stops the whole plot.
        spec = MODEL_FOR_OPTION.get(model)
        if spec is not None:
            try:
                if get_model(self.controller, spec) is None:
                    return  # the user cancelled
            except PredictionError as exc:
                QtWidgets.QMessageBox.warning(self.controller.view, model, str(exc))
                return
            except Exception as exc:
                logger.exception('Getting the %s model failed', model)
                QtWidgets.QMessageBox.warning(
                    self.controller.view,
                    model,
                    f'Unexpected error in {model}: {type(exc).__name__}: {exc}\n\n'
                    'See the log for details.',
                )
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
        The error message when the prediction failed, else None: a ``PredictionError``'s message,
        or a short message naming the error when something unexpected failed (its traceback is
        logged). Errors are returned rather than shown so that the caller reports them once for
        all shanks.
    """
    if not getattr(items.model, 'predictions', None):
        items.model.predictions = Bunch()

    if items.model.predictions.get(model) is None:
        try:
            views = func(controller, items)
        except PredictionError as exc:
            return str(exc)
        except Exception as exc:
            # Anything else is a bug or an unforeseen failure. Keep it out of Qt (an uncaught
            # exception in a callback aborts the app), log the traceback, and report it with the
            # other shanks' errors.
            logger.exception(
                '%s failed on shank %s (%s)', model, kwargs.get('shank'), kwargs.get('config')
            )
            return (
                f'Unexpected error in {model}: {type(exc).__name__}: {exc}\n\n'
                'See the log for details.'
            )
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
