import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from qtpy import QtWidgets

from ibl_alignment_gui.app.load_worker import TaskCancelledError
from ibl_alignment_gui.plugins.ephys_atlas import spatial_encoder as spatial
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    NO_FEATURES_MESSAGE,
    PredictionError,
    choose_source,
    close_alignment_windows,
    has_features,
    is_model_loaded,
    load_model,
    plugin_state,
    show_alignment_windows,
    store_model,
)
from ibl_alignment_gui.plugins.ephys_atlas.inference import DECODER_SPECS
from iblutil.util import Bunch

# NB: both backends import ``ephysatlas`` (and torch) only when a model is used, so this plugin
# can be set up without those heavy/optional dependencies installed.

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController
    from ibl_alignment_gui.plugins.ephys_atlas._common import ModelSpec

PLUGIN_NAME = 'Channel Prediction'
# Title of the dialogs of the all predictions view.
ALL_TITLE = 'All predictions'
# Shown in the all predictions view on the figures of a model that is not loaded.
NOT_LOADED_MESSAGE = 'Not loaded: Plugins -> Channel Prediction'

# Every model the plugin offers, as the backends describe them (see ModelSpec).
MODELS = (*DECODER_SPECS.values(), spatial.ENCODER_SPEC)
# The model behind each of their prediction options.
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
    features file), offers the models' options in Prediction Plots and registers a data-loaded
    callback that resets the per-session plugin state.

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
    # The models need features, which come with a session: greyed out until one has them, with a
    # hint saying how to provide them
    controller.plugin_options.setToolTipsVisible(True)
    controller.plugins[PLUGIN_NAME]['menu'] = plugin_menu
    _enable_menu(plugin_menu, False)

    # One load action per model; those needing torch only when it is installed.
    models = available_models()
    # Their options, offered in Prediction Plots. A model is loaded when one of its options is
    # first selected (or from the Plugins menu).
    controller.prediction_keys = [option for spec in models for option in spec.options]
    menu_actions = [
        (f'{spec.title} (Hugging Face)…', partial(_load_model, spec=spec)) for spec in models
    ]

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


def _load_model(controller: 'AlignmentGUIController', spec: 'ModelSpec') -> None:
    """Load a release of ``spec``'s model chosen in its dialog, then show its first option.

    The dialog runs here (main thread); the model loads in the background, behind a progress
    dialog with Cancel. A source that is already loaded is not loaded again. Cancelling the dialog
    keeps the current model (and still shows it, if there is one).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    spec : ModelSpec
        The model to load.
    """
    if not has_features(controller):
        QtWidgets.QMessageBox.warning(controller.view, PLUGIN_NAME, NO_FEATURES_MESSAGE)
        return
    state = plugin_state(controller).get(spec.key) or {}
    loaded = state.get('model') is not None
    source = choose_source(controller, spec)
    if source is None:  # dialog cancelled: keep (and show) the current model, if any
        if loaded:
            _show_first_option(controller, spec)
        return
    if loaded and tuple(state['source']) == source:  # already loaded: not loaded again
        _show_first_option(controller, spec)
        return
    started = controller.run_in_thread(
        _load_task,
        spec,
        source,
        on_finished=partial(_load_done, controller, spec),
        on_error=partial(_task_failed, controller, spec.title),
        busy_message='Starting…',
        title=spec.title,
        report_progress=True,
        cancellable=True,
    )
    if not started:
        _busy_warning(controller, spec.title)


def _load_task(spec: 'ModelSpec', source: tuple[str, str], progress_callback) -> '_Outcome':
    """Load ``source`` (background thread, no Qt)."""
    outcome = _Outcome()
    try:
        outcome.model = load_model(spec, source, _fraction_progress(progress_callback))
        outcome.loaded_from = source
    except TaskCancelledError:
        outcome.cancelled = True
    except PredictionError as exc:
        outcome.errors.append(str(exc))
    return outcome


def _load_done(controller: 'AlignmentGUIController', spec: 'ModelSpec', outcome: '_Outcome'):
    """Keep the loaded model and show its first option, or report the failure (main thread)."""
    if outcome.errors:
        QtWidgets.QMessageBox.critical(controller.view, spec.title, outcome.errors[0])
        return
    if outcome.loaded_from is None:
        return  # cancelled: the current model, if any, is kept
    store_model(controller, spec, outcome.model, outcome.loaded_from)
    _show_first_option(controller, spec)


def _show_first_option(controller: 'AlignmentGUIController', spec: 'ModelSpec') -> None:
    """Select the model's first prediction option, which shows its predictions.

    In the all predictions view, the view is redrawn instead, with the model's predictions.
    """
    if getattr(controller, 'show_predictions', False):
        _predict_all(controller)
        return
    controller.view.trigger_menu_option('prediction', spec.options[0])


def _on_data_loaded(controller: 'AlignmentGUIController') -> None:
    """Data-load hook: reset per-session plugin state.

    Runs on every data load (i.e. each new session). Closes the previous session's auto-alignment
    result windows, and enables the plugin's menu only when the session has features.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    close_alignment_windows(controller)
    menu = controller.plugins[PLUGIN_NAME].get('menu')
    if menu is not None:
        _enable_menu(menu, has_features(controller))


def _enable_menu(menu: QtWidgets.QMenu, enabled: bool) -> None:
    """Enable the plugin's menu, or grey it out with a hint saying a features file is needed."""
    menu.setEnabled(enabled)
    menu.menuAction().setToolTip('' if enabled else NO_FEATURES_MESSAGE)


class ChannelPrediction:
    """Plugin loader that computes and plots per-channel region predictions.

    Registered as the Channel Prediction plugin's ``loader``; :meth:`plot_predictions` shows the
    selected model option (``MODEL_FOR_OPTION``) on each shank's prediction panel, computed in
    the background.
    """

    def __init__(self, controller: 'AlignmentGUIController') -> None:
        """Store the controller.

        Parameters
        ----------
        controller : AlignmentGUIController
            The main application controller.
        """
        self.controller = controller

    def ready(self) -> bool:
        """Return whether there are features to predict on, warning when there are none.

        Returns
        -------
        bool
            True when a shank has features (or a features file was chosen), else False after
            showing a warning that says how to load a features file.
        """
        if has_features(self.controller):
            return True
        QtWidgets.QMessageBox.warning(self.controller.view, PLUGIN_NAME, NO_FEATURES_MESSAGE)
        return False

    def plot_predictions(self, option: str) -> None:
        """Show the selected model option on every aligned shank.

        A model's predictions are drawn from the cache, or computed first in the background
        (:func:`_predict`). Problems are reported in one warning for all shanks.

        Parameters
        ----------
        option : str
            Prediction option (e.g. 'Auto-alignment', 'XGBoost region prediction',
            'XGBoost region prediction (cumulative)'). Unknown options are ignored.
        """
        spec = MODEL_FOR_OPTION.get(option)
        if spec is not None:
            _predict(self.controller, spec, option)

    def plot_all_predictions(self) -> None:
        """Show every option of the loaded models side by side on every aligned shank.

        The options of models that are not loaded say so; their models are not loaded here. The
        missing views are computed first, for all models in one background task
        (:func:`_predict_all`).
        """
        _predict_all(self.controller)


# -----------------------------------------------------------------------------
# Model predictions, computed in the background
# -----------------------------------------------------------------------------


@dataclass
class _Outcome:
    """What a background task did, handed back to the main thread when it ends."""

    model: object = None
    loaded_from: tuple[str, str] | None = None  # set when the task loaded the model
    results: list = field(default_factory=list)  # (items, views, extra) per finished shank
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False


def _fraction_progress(progress_callback: Callable, prefix: str = '') -> Callable:
    """Adapt the worker's ``(message, current, total)`` to the models' ``(fraction, message)``.

    A ``fraction`` of None shows a busy bar. Each report also stops the task after Cancel.
    """

    def progress(fraction: float | None, message: str) -> None:
        if fraction is None:
            progress_callback(prefix + message, 0, 0)
        else:
            progress_callback(prefix + message, int(round(1000 * float(fraction))), 1000)

    return progress


def _unexpected(title: str, exc: Exception) -> str:
    """Return the message for an unexpected error, whose traceback is logged."""
    return f'Unexpected error in {title}: {type(exc).__name__}: {exc}\n\nSee the log for details.'


def _prediction_task(
    controller: 'AlignmentGUIController',
    spec: 'ModelSpec',
    model,
    source: tuple[str, str] | None,
    shanks: list['ShankController'],
    progress_callback: Callable,
) -> _Outcome:
    """Load ``spec``'s model if needed, then run it on each shank (background thread, no Qt).

    Each shank's error is kept and the next shank is computed. After Cancel, the shanks finished
    so far (and a model already loaded) are kept and the rest skipped.
    """
    outcome = _Outcome(model=model)
    try:
        if model is None:
            outcome.model = load_model(spec, source, _fraction_progress(progress_callback))
            outcome.loaded_from = source
        for i, items in enumerate(shanks, 1):
            step = _fraction_progress(progress_callback, f'{items.name} ({i}/{len(shanks)}): ')
            step(None, 'starting…')  # also where Cancel stops the task between shanks
            try:
                views, extra = spec.compute(controller, items, outcome.model, step)
            except TaskCancelledError:
                raise
            except PredictionError as exc:
                outcome.errors.append(str(exc))
                continue
            except Exception as exc:
                # A bug or an unforeseen failure: log it and report it with the other shanks.
                logger.exception('%s failed on shank %s', spec.key, items.name)
                outcome.errors.append(_unexpected(spec.options[0], exc))
                continue
            outcome.results.append((items, views, extra))
    except TaskCancelledError:
        outcome.cancelled = True
    except PredictionError as exc:  # the model could not be loaded
        outcome.errors.append(str(exc))
    except Exception as exc:
        logger.exception('Getting the %s model failed', spec.key)
        outcome.errors.append(_unexpected(spec.options[0], exc))
    return outcome


def _predict(controller: 'AlignmentGUIController', spec: 'ModelSpec', option: str) -> None:
    """Show ``option`` on every aligned shank, computing the missing views in the background.

    The model is chosen here if it is not loaded (its preset source, or its dialog); loading it
    and running it on each shank then happen in one background task, behind one progress dialog
    whose Cancel stops it. The shanks are drawn when it ends (:func:`_prediction_done`).
    """
    shanks = _aligned_shanks(controller)
    todo = [items for items in shanks if _cache(items).get(option) is None]
    if not todo:
        _draw_all(shanks, option)
        return
    state = plugin_state(controller).get(spec.key) or {}
    model, source = state.get('model'), None
    if model is None:
        source = tuple(state['source']) if state.get('source') else choose_source(controller, spec)
        if source is None:
            return  # cancelled, or no features (the user was warned)
    started = controller.run_in_thread(
        _prediction_task,
        controller,
        spec,
        model,
        source,
        todo,
        on_finished=partial(_prediction_done, controller, spec, option, shanks),
        on_error=partial(_task_failed, controller, option),
        busy_message='Starting…',
        title=option,
        report_progress=True,
        cancellable=True,
    )
    if not started:
        _busy_warning(controller, option)


def _prediction_done(
    controller: 'AlignmentGUIController',
    spec: 'ModelSpec',
    option: str,
    shanks: list['ShankController'],
    outcome: _Outcome,
) -> None:
    """Keep and draw what the task computed, and report its warnings and errors once (main thread).

    The warnings are those the task queued in the plugin state's ``pending_warnings`` (e.g. local
    features of another feature version than the model's).
    """
    _keep(controller, spec, outcome)
    _draw_all(shanks, option)
    _report(controller, option, outcome.errors)


def _keep(controller: 'AlignmentGUIController', spec: 'ModelSpec', outcome: _Outcome) -> None:
    """Keep the model a task loaded and the views it computed (main thread)."""
    if outcome.loaded_from is not None:
        store_model(controller, spec, outcome.model, outcome.loaded_from)
    for items, views, extra in outcome.results:
        _cache(items).update(views)
        if spec.on_computed is not None:
            spec.on_computed(controller, items, extra)


def _report(controller: 'AlignmentGUIController', title: str, errors: list[str]) -> None:
    """Show the warnings queued by the task and its errors in one dialog, if there are any.

    The warnings are those the task queued in the plugin state's ``pending_warnings`` (e.g. local
    features of another feature version than the model's).
    """
    warnings = plugin_state(controller).pop('pending_warnings', [])
    messages = list(dict.fromkeys(warnings + errors))
    if messages:
        QtWidgets.QMessageBox.warning(controller.view, title, '\n\n'.join(messages))


def _all_predictions_task(
    controller: 'AlignmentGUIController', jobs: list, progress_callback: Callable
) -> list[tuple['ModelSpec', _Outcome]]:
    """Run each loaded model on its shanks, one after the other (background thread, no Qt).

    ``jobs`` holds each model's ``(spec, model, shanks)``. A model's errors are kept and the next
    model is run; after Cancel, the models finished so far are kept and the rest skipped.
    """
    results = []
    for spec, model, shanks in jobs:

        def progress(message, current, total, name=spec.key):
            progress_callback(f'{name}, {message}', current, total)

        outcome = _prediction_task(controller, spec, model, None, shanks, progress)
        results.append((spec, outcome))
        if outcome.cancelled:
            break
    return results


def _predict_all(controller: 'AlignmentGUIController') -> None:
    """Show every option of the loaded models on every aligned shank, side by side.

    The views missing for the loaded models are computed in one background task, behind one
    progress dialog whose Cancel stops it, and the shanks are drawn when it ends
    (:func:`_all_predictions_done`). Models that are not loaded are not loaded here.
    """
    shanks = _aligned_shanks(controller)
    jobs = []
    for spec in available_models():
        if not is_model_loaded(controller, spec.key):
            continue
        todo = [
            items
            for items in shanks
            if any(_cache(items).get(option) is None for option in spec.options)
        ]
        if todo:
            jobs.append((spec, plugin_state(controller)[spec.key]['model'], todo))
    if not jobs:
        _draw_all_predictions(controller, shanks)
        return
    started = controller.run_in_thread(
        _all_predictions_task,
        controller,
        jobs,
        on_finished=partial(_all_predictions_done, controller, shanks),
        on_error=partial(_task_failed, controller, ALL_TITLE),
        busy_message='Starting…',
        title=ALL_TITLE,
        report_progress=True,
        cancellable=True,
    )
    if not started:
        _busy_warning(controller, ALL_TITLE)


def _all_predictions_done(
    controller: 'AlignmentGUIController',
    shanks: list['ShankController'],
    results: list[tuple['ModelSpec', _Outcome]],
) -> None:
    """Keep and draw what each model computed, and report all problems once (main thread)."""
    errors = []
    for spec, outcome in results:
        _keep(controller, spec, outcome)
        errors.extend(outcome.errors)
    _draw_all_predictions(controller, shanks)
    _report(controller, ALL_TITLE, errors)


def _task_failed(controller: 'AlignmentGUIController', title: str, traceback: str) -> None:
    """Report a task that failed outside its own error handling (the worker logged it)."""
    QtWidgets.QMessageBox.warning(
        controller.view, title, f'Unexpected error in {title}.\n\nSee the log for details.'
    )


def _busy_warning(controller: 'AlignmentGUIController', title: str) -> None:
    """Tell the user another background task is still running."""
    QtWidgets.QMessageBox.warning(
        controller.view, title, 'Another task is running; try again once it has finished.'
    )


def _aligned_shanks(controller: 'AlignmentGUIController') -> list['ShankController']:
    """Return the shanks a prediction is shown on: the aligned ones, in shank_loop's order."""
    return [
        controller.shank_items[shank][config]
        for config in controller.model.configs
        for shank in controller.all_shanks
        if controller.model.get_current_shank(shank, config).align_exists
    ]


def _cache(items: 'ShankController') -> Bunch:
    """Return the shank's cached prediction views, by option."""
    if not getattr(items.model, 'predictions', None):
        items.model.predictions = Bunch()
    return items.model.predictions


def _draw(items: 'ShankController', option: str, fig=None) -> None:
    """Draw the shank's cached view of ``option`` on a prediction figure, or clear the figure.

    The figure is the shank's prediction panel unless given.
    """
    fig = items.view.fig_pred if fig is None else fig
    pred = _cache(items).get(option)
    if pred is None:  # e.g. the prediction failed on this shank
        items.view.clear_prediction(fig, option)
    else:
        items.view.plot_prediction(fig, pred, option)


def _draw_all(shanks: list['ShankController'], option: str) -> None:
    """Draw ``option`` on each shank that has it."""
    for items in shanks:
        _draw(items, option)


def _draw_all_predictions(
    controller: 'AlignmentGUIController', shanks: list['ShankController']
) -> None:
    """Draw every option on its own figure of each shank, saying which models are not loaded."""
    for items in shanks:
        for option, fig in items.view.fig_preds.items():
            spec = MODEL_FOR_OPTION.get(option)
            if spec is None or not is_model_loaded(controller, spec.key):
                items.view.clear_prediction(fig, option, NOT_LOADED_MESSAGE)
            else:
                _draw(items, option, fig)
