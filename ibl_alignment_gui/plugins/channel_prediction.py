import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy import QtWidgets

from ibl_alignment_gui.app.load_worker import TaskCancelledError
from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal
from ibl_alignment_gui.plugins.ephys_atlas import spatial_encoder as spatial
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    ALL_SHANKS,
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

    # One load action per model; those needing torch only when it is installed.
    models = available_models()
    # Their options, offered in Prediction Plots. A model is loaded when one of its options is
    # first selected (or from the Plugins menu).
    controller.prediction_keys = [option for spec in models for option in spec.options]
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


class FeaturesFilesDialog(QtWidgets.QDialog):
    """Choose a features file for each shank and configuration, or one file for all shanks.

    Each row opens with the shank's current file; a row left as it is (or emptied) keeps the
    shank's current features.

    Parameters
    ----------
    parent : QtWidgets.QWidget or None
        The parent window.
    rows : list of ((str, str), Path or None)
        Each shank's ``(shank, config)`` and its current features file.
    """

    def __init__(self, parent, rows: list) -> None:
        super().__init__(parent)
        self.setWindowTitle('Load features files')
        self.setMinimumWidth(640)
        self._initial = {key: '' if path is None else str(path) for key, path in rows}

        self._all = QtWidgets.QCheckBox('One file for all shanks (split by channel)')
        self._all_field = QtWidgets.QLineEdit()
        all_row = self._file_row(self._all_field)
        self._all_widget = QtWidgets.QWidget()
        self._all_widget.setLayout(all_row)
        self._all_widget.setVisible(False)

        grid = QtWidgets.QGridLayout()
        self._fields = {}
        for row, (key, _) in enumerate(rows):
            shank, config = key
            field = QtWidgets.QLineEdit(self._initial[key])
            self._fields[key] = field
            grid.addWidget(QtWidgets.QLabel(shank), row, 0)
            grid.addWidget(QtWidgets.QLabel(config), row, 1)
            grid.addLayout(self._file_row(field), row, 2)
        self._rows_widget = QtWidgets.QWidget()
        self._rows_widget.setLayout(grid)
        self._all.toggled.connect(self._all_widget.setVisible)
        self._all.toggled.connect(lambda on: self._rows_widget.setVisible(not on))

        note = QtWidgets.QLabel("Rows left as they are keep the shank's current features.")
        note.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        for widget in (self._all, self._all_widget, self._rows_widget, note, buttons):
            layout.addWidget(widget)

    def _file_row(self, field: QtWidgets.QLineEdit) -> QtWidgets.QHBoxLayout:
        """Return a file field with a Browse button."""
        browse = QtWidgets.QPushButton('Browse…')
        browse.clicked.connect(lambda: self._browse(field))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(field)
        row.addWidget(browse)
        return row

    def _browse(self, field: QtWidgets.QLineEdit) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select per-channel features file', filter='Parquet (*.pqt *.parquet)'
        )
        if chosen:
            field.setText(chosen)

    def chosen(self) -> dict:
        """Return the files to apply: one for all shanks, or the changed rows per shank."""
        if self._all.isChecked():
            text = self._all_field.text().strip()
            return {ALL_SHANKS: Path(text)} if text else {}
        return {
            key: Path(field.text().strip())
            for key, field in self._fields.items()
            if field.text().strip() and field.text().strip() != self._initial[key]
        }


def _shank_handlers(controller: 'AlignmentGUIController') -> list:
    """Return every shank's ``((shank, config), ShankHandler)``."""
    return [
        ((shank, config), handler)
        for shank, configs in controller.model.shanks.items()
        for config, handler in configs.items()
    ]


def _current_features_file(controller: 'AlignmentGUIController', key, handler) -> Path | None:
    """Return the features file a shank uses now: the one chosen before, or its session's file."""
    paths = plugin_state(controller).get('features_paths') or {}
    if key in paths or ALL_SHANKS in paths:
        return paths.get(key, paths.get(ALL_SHANKS))
    loader = handler.loaders.get('features')
    return loader.features_path if isinstance(loader, FeatureLoaderLocal) else None


def _check_features_files(controller: 'AlignmentGUIController', chosen: dict) -> list[str]:
    """Return a line for each chosen file that has no features for its shank (none if all do).

    A file for all shanks only needs features in it, as it is split by channel. A file that cannot
    be read (e.g. not a parquet file) is reported the same way.
    """
    handlers = dict(_shank_handlers(controller))
    problems = []
    for key, path in chosen.items():
        if key == ALL_SHANKS:
            label, sites = 'All shanks', None
        else:
            handler = handlers[key]
            label = f'{key[0]} ({key[1]})'
            sites = (
                handler.loaders['geom'].get_sites_for_shank(handler.shank_idx)
                if handler.data_loaded
                else None
            )
        try:
            exists = FeatureLoaderLocal(path).load_features(sites).get('exists', False)
        except Exception as exc:
            problems.append(f'{label}: {path} could not be read ({type(exc).__name__}: {exc})')
            continue
        if not exists:
            problems.append(f'{label}: no features for this shank in {path}')
    return problems


def _apply_features_files(controller: 'AlignmentGUIController', chosen: dict) -> None:
    """Remember the chosen files, load them into the loaded shanks and clear old predictions.

    A file for all shanks replaces any shank's own file; a shank's own file overrides the file
    for all shanks. Shanks whose data is not loaded yet load their file when they are predicted
    on (``ephys_atlas._common._get_features_df``).
    """
    paths = (
        {} if ALL_SHANKS in chosen else dict(plugin_state(controller).get('features_paths') or {})
    )
    paths.update(chosen)
    plugin_state(controller)['features_paths'] = paths
    for key, handler in _shank_handlers(controller):
        path = chosen.get(key, chosen.get(ALL_SHANKS))
        if path is not None and handler.data_loaded:
            handler.load_ephys_features(FeatureLoaderLocal(path))
    _invalidate_loaded_models(controller)


def _set_local_features(controller: 'AlignmentGUIController') -> None:
    """Choose features files for the shanks ("Load features file…") and use them for the models.

    With one shank and one configuration, a plain file dialog; otherwise a file per shank and
    configuration (:class:`FeaturesFilesDialog`). The files are checked before any is applied.
    """
    rows = _shank_handlers(controller)
    if len(rows) <= 1:
        chosen_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            controller.view, 'Select per-channel features file', filter='Parquet (*.pqt *.parquet)'
        )
        chosen = {ALL_SHANKS: Path(chosen_file)} if chosen_file else {}
    else:
        dialog = FeaturesFilesDialog(
            controller.view,
            [(key, _current_features_file(controller, key, handler)) for key, handler in rows],
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        chosen = dialog.chosen()
    if not chosen:
        return

    problems = _check_features_files(controller, chosen)
    if problems:
        QtWidgets.QMessageBox.warning(
            controller.view,
            PLUGIN_NAME,
            'No features found for:\n'
            + '\n'.join(problems)
            + '\n\nNo features files were changed.',
        )
        return
    _apply_features_files(controller, chosen)
    logger.info('Features files set: %s', {str(k): str(v) for k, v in chosen.items()})


def _invalidate_loaded_models(controller: 'AlignmentGUIController') -> None:
    """Drop every loaded model's cached predictions, e.g. after a new features file."""
    for spec in MODELS:
        if is_model_loaded(controller, spec.key):
            spec.invalidate(controller)


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
    """Select the model's first prediction option, which shows its predictions."""
    controller.view.trigger_menu_option('prediction', spec.options[0])


def _on_data_loaded(controller: 'AlignmentGUIController') -> None:
    """Data-load hook: reset per-session plugin state.

    Runs on every data load (i.e. each new session). Drops any manual features override so it does
    not leak across sessions (each session supplies its own features) and closes the previous
    session's auto-alignment result windows.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME]['features_paths'] = None
    close_alignment_windows(controller)


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
    if outcome.loaded_from is not None:
        store_model(controller, spec, outcome.model, outcome.loaded_from)
    for items, views, extra in outcome.results:
        _cache(items).update(views)
        if spec.on_computed is not None:
            spec.on_computed(controller, items, extra)
    _draw_all(shanks, option)
    warnings = plugin_state(controller).pop('pending_warnings', [])
    messages = list(dict.fromkeys(warnings + outcome.errors))
    if messages:
        QtWidgets.QMessageBox.warning(controller.view, option, '\n\n'.join(messages))


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


def _draw(items: 'ShankController', option: str) -> None:
    """Draw the shank's cached view of ``option`` on its prediction panel, or clear the panel."""
    pred = _cache(items).get(option)
    if pred is None:  # e.g. the prediction failed on this shank
        items.view.clear_prediction(items.view.fig_pred, option)
    else:
        items.view.plot_prediction(items.view.fig_pred, pred, option)


def _draw_all(shanks: list['ShankController'], option: str) -> None:
    """Draw ``option`` on each shank that has it."""
    for items in shanks:
        _draw(items, option)
