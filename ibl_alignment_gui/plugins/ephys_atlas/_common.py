"""
Provider-agnostic helpers shared by the channel-prediction model backends.

Both :mod:`ibl_alignment_gui.plugins.ephys_atlas.inference` (XGBoost) and
:mod:`ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder` (torch) cache their state on the
Channel Prediction plugin, check for features the same way and clear per-shank prediction caches
identically. Those pieces live here.

This module intentionally avoids the heavy/optional ``torch`` and ``ephysatlas`` imports so it can
be imported in offline mode alongside ``inference`` (which defers those imports itself).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy import QtCore, QtWidgets

from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal, FeatureLoaderOne

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController
    from iblutil.util import Bunch

logger = logging.getLogger(__name__)

PLUGIN_KEY = 'Channel Prediction'

# Auto-alignment: its region-panel option (also the per-shank prediction cache key) and how its
# column is coloured -- the predicted regions, or the alignment confidence along the probe.
AUTO_ALIGNMENT_KEY = 'Auto-alignment'
DISPLAY_STATE_KEY = 'auto_alignment_display'
DISPLAY_REGIONS = 'Regions'
DISPLAY_CONFIDENCE = 'Confidence'


class PredictionError(Exception):
    """A model could not make a prediction; the message is meant for the user."""


class AlignmentCancelledError(Exception):
    """Raised from the progress callback when the user presses Cancel."""


@dataclass(frozen=True)
class ModelSpec:
    """A loadable Ephys Atlas model, as the shared load dialog and model cache need it.

    Attributes
    ----------
    key : str
        The plugin-state key the loaded model is cached under: a decoder's name, or 'Encoding'.
    title : str
        Title of the load dialog, e.g. 'Load alignment model'.
    repo : str
        Default Hugging Face repository.
    revision : str
        Default release tag, e.g. '2026_W39'.
    note : str
        Explanation shown in the load dialog.
    load : Callable
        ``load(source, progress) -> model``: loads ``source`` (``(repo_or_dir, revision)``),
        reporting through ``progress(fraction, message)``. It does its heavy imports itself, so
        this module stays light.
    invalidate : Callable
        ``invalidate(controller)``: drops the shanks' cached predictions of this model.
    """

    key: str
    title: str
    repo: str
    revision: str
    note: str
    load: Callable[[tuple[str, str], Callable[[float, str], None]], object]
    invalidate: Callable[[AlignmentGUIController], None]


class ModelSourceDialog(QtWidgets.QDialog):
    """Choose a model release: a Hugging Face repository and release tag, or a local release.

    Parameters
    ----------
    parent : QtWidgets.QWidget or None
        The parent window.
    spec : ModelSpec
        The model: its dialog title and note, and the defaults used for empty fields.
    repo : str
        The repository the dialog opens with.
    revision : str
        The release tag the dialog opens with.
    local_dir : Path or None
        The local release directory the dialog opens with, if any.
    """

    def __init__(
        self, parent, spec: ModelSpec, repo: str, revision: str, local_dir: Path | None
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self.setWindowTitle(spec.title)
        self.setMinimumWidth(560)
        form = QtWidgets.QFormLayout()
        self._repo = QtWidgets.QLineEdit(repo)
        self._revision = QtWidgets.QLineEdit(revision)
        form.addRow('Hugging Face repository', self._repo)
        form.addRow('Release tag', self._revision)

        local_row = QtWidgets.QHBoxLayout()
        self._local = QtWidgets.QLineEdit('' if local_dir is None else str(local_dir))
        self._local.setPlaceholderText(
            'optional: a local release directory (containing ephysatlas_model.json) instead of '
            'the Hub'
        )
        browse = QtWidgets.QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        local_row.addWidget(self._local)
        local_row.addWidget(browse)
        form.addRow('Local release', local_row)

        note = QtWidgets.QLabel(spec.note)
        note.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        # Loading reports a directory that is not a release (no manifest).
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select a local release directory (containing ephysatlas_model.json)'
        )
        if chosen:
            self._local.setText(chosen)

    def source(self) -> tuple[str, str]:
        """``(repo_id_or_directory, revision)`` to load; empty fields use the spec's defaults."""
        local = self._local.text().strip()
        revision = self._revision.text().strip() or self._spec.revision
        return (local or self._repo.text().strip() or self._spec.repo), revision


NO_FEATURES_MESSAGE = (
    'No features found for this probe. '
    'Set via Plugins -> Channel Prediction -> Load features file...'
)


def _load_source(controller: AlignmentGUIController, spec: ModelSpec, source: tuple[str, str]):
    """Load ``source`` with ``spec.load`` behind a progress dialog.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    spec : ModelSpec
        The model to load.
    source : tuple of (str, str)
        ``(repo_id_or_directory, revision)`` to load.

    Returns
    -------
    object
        The loaded model.

    Raises
    ------
    PredictionError
        If the model cannot be loaded, naming the source and the reason.
    """
    try:
        return run_with_progress(
            controller, spec.title, lambda progress: spec.load(source, progress), cancellable=False
        )
    except Exception as exc:
        raise PredictionError(f'Could not load {source[0]}@{source[1]}:\n{exc}') from exc


def load_model_dialog(controller: AlignmentGUIController, spec: ModelSpec) -> bool:
    """Ask which release of ``spec``'s model to use and load it (only when the source changed).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    spec : ModelSpec
        The model to load.

    Returns
    -------
    bool
        True when a model is loaded, False when the user cancelled or loading failed.
    """
    plugin = plugin_state(controller)
    state = plugin.get(spec.key) or dict(model=None, source=None)
    plugin[spec.key] = state
    if not has_features(controller):
        QtWidgets.QMessageBox.warning(controller.view, PLUGIN_KEY, NO_FEATURES_MESSAGE)
        return False

    repo, revision = state['source'] or (spec.repo, spec.revision)
    local = Path(repo) if Path(str(repo)).is_dir() else None
    dialog = ModelSourceDialog(
        controller.view, spec, spec.repo if local else repo, revision, local
    )
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return state['model'] is not None

    source = dialog.source()
    if state['model'] is not None and tuple(state['source']) == source:
        return True
    try:
        model = _load_source(controller, spec, source)
    except PredictionError as exc:
        QtWidgets.QMessageBox.critical(controller.view, spec.title, str(exc))
        return False
    state.update(model=model, source=source)
    spec.invalidate(controller)
    logger.info('%s set to %s@%s', spec.key, *source)
    return True


def get_model(controller: AlignmentGUIController, spec: ModelSpec):
    """Return ``spec``'s loaded model, loading it first when needed.

    A ``source`` preset on the plugin state (``(repo_id_or_dir, revision)``, e.g. by a launcher
    script) loads without the dialog; otherwise the load dialog opens.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    spec : ModelSpec
        The model to return.

    Returns
    -------
    object or None
        The loaded model, or None if the user cancelled or loading from the dialog failed.

    Raises
    ------
    PredictionError
        If a preset source cannot be loaded.
    """
    state = plugin_state(controller).get(spec.key)
    if state is not None and state.get('model') is None and state.get('source'):
        state['model'] = _load_source(controller, spec, tuple(state['source']))
    if state is None or state.get('model') is None:
        if not load_model_dialog(controller, spec):
            return None
        state = plugin_state(controller)[spec.key]
    return state['model']


def run_with_progress(
    controller: AlignmentGUIController, title: str, work: Callable, *, cancellable: bool = True
):
    """Run ``work(progress)`` behind a modal progress dialog showing its fraction and message.

    ``progress(fraction, message)`` updates the bar and its text and keeps the GUI responsive;
    Cancel raises :class:`AlignmentCancelledError` inside ``work`` at its next progress report.
    """
    dialog = QtWidgets.QProgressDialog(
        title, 'Cancel' if cancellable else None, 0, 1000, controller.view
    )
    dialog.setWindowTitle(title)
    dialog.setWindowModality(QtCore.Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setMinimumWidth(420)
    dialog.setAutoClose(False)
    dialog.setValue(0)
    QtWidgets.QApplication.processEvents()

    def progress(fraction: float, message: str) -> None:
        if cancellable and dialog.wasCanceled():
            raise AlignmentCancelledError()
        dialog.setLabelText(f'{title}\n\n{message}')
        dialog.setValue(int(round(1000 * float(fraction))))
        QtWidgets.QApplication.processEvents()

    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        return work(progress)
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
        dialog.close()


def auto_alignment_display(controller: AlignmentGUIController) -> str:
    """How the auto-alignment column is coloured: ``DISPLAY_REGIONS`` or ``DISPLAY_CONFIDENCE``."""
    return plugin_state(controller).get(DISPLAY_STATE_KEY) or DISPLAY_REGIONS


def plugin_state(controller: AlignmentGUIController) -> Bunch:
    """Return the Channel Prediction plugin state Bunch.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    Bunch
        The plugin's mutable state container.
    """
    return controller.plugins[PLUGIN_KEY]


def is_model_loaded(controller: AlignmentGUIController, state_key: str) -> bool:
    """Return whether a model has been loaded under ``state_key`` on the plugin.

    Light enough to call without importing the (heavy) backend module: it only reads the cached
    plugin state.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    state_key : str
        The plugin-state key the backend caches its model under (its ``MODEL_NAME``).

    Returns
    -------
    bool
        True when a model is cached under ``state_key``, else False.
    """
    state = plugin_state(controller).get(state_key)
    return bool(state) and state.get('model') is not None


def has_local_features(controller: AlignmentGUIController) -> bool:
    """Return whether the user chose a features file with ``Load features file…``.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    bool
        True when a local features file overrides the session's features, else False.
    """
    return plugin_state(controller).get('features_path') is not None


def has_features(controller: AlignmentGUIController) -> bool:
    """Return whether per-channel features are available to run a prediction on.

    True when a local features file is configured on the plugin (the offline source) or any loaded
    shank already holds a non-empty features table (the online source).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    bool
        True when features are available for at least one loaded shank, else False.
    """
    if has_local_features(controller):
        return True
    for shank_dict in controller.model.shanks.values():
        for shank_handler in shank_dict.values():
            raw_data = getattr(shank_handler, 'raw_data', None)
            if raw_data is None:
                continue
            features = raw_data.get('features')
            if features is not None and features.get('exists', False):
                return True
    return False


def clear_predictions(items: ShankController, *keys: str) -> None:
    """Drop cached predictions for ``keys`` on a shank so the next click recomputes.

    Parameters
    ----------
    items : ShankController
        The shank whose cached predictions are cleared.
    keys
        The per-shank prediction cache keys to remove.
    """
    preds = getattr(items.model, 'predictions', None)
    if preds:
        for key in keys:
            preds.pop(key, None)


def _get_features_df(
    controller: AlignmentGUIController, items: ShankController
) -> pd.DataFrame | None:
    """Return the per-channel features DataFrame for a shank, or None if unavailable.

    Uses the already-loaded features (online ONE path) when present. Otherwise, in offline/yaml
    mode where no features loader exists, loads them from the local parquet configured on the
    plugin (``features_path``) and injects the result into ``raw_data['features']`` so the rest of
    the inference path is unchanged.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.

    Returns
    -------
    pandas.DataFrame or None
        The features DataFrame, or None when no features are available.
    """
    feats = items.model.raw_data.get('features')
    if feats is None or not feats.get('exists', False):
        features_path = plugin_state(controller).get('features_path')
        if features_path is None:
            return None
        # Split a combined features file down to this shank using its raw channel indices.
        shank_sites = items.model.loaders['geom'].get_sites_for_shank(items.model.shank_idx)
        feats = FeatureLoaderLocal(features_path).load_features(shank_sites)
        items.model.raw_data['features'] = feats
        if not feats.get('exists', False):
            return None
    return feats['df']


def features_for_vintage(
    controller: AlignmentGUIController, items: ShankController, vintage: str | None
) -> pd.DataFrame | None:
    """Return the shank's features from the feature table a model was trained on.

    The GUI loads the latest ephys-atlas feature table, whose feature set can differ from the one
    a model was trained on (columns are renamed or dropped between vintages). When the features
    come from ONE, the table labelled ``vintage`` is loaded instead, and cached on the shank so the
    feature plots keep the latest table. A local features file (from the session or chosen with
    ``Load features file…``) is used as is.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.
    vintage : str or None
        Label of the feature table the model was trained on (e.g. ``2026_W26``); None uses the
        GUI's features.

    Returns
    -------
    pandas.DataFrame or None
        The features DataFrame, or None when no features are available.
    """
    loader = items.model.loaders.get('features')
    if (
        has_local_features(controller)
        or not isinstance(loader, FeatureLoaderOne)
        or vintage is None
        or loader.feature_version == vintage
    ):
        return _get_features_df(controller, items)

    cache = items.model.raw_data.setdefault('model_features', {})
    if vintage not in cache:
        vintage_loader = copy.copy(loader)
        vintage_loader.feature_version = vintage
        logger.info(
            'Loading %s features for the model (GUI features are %s)',
            vintage,
            loader.feature_version,
        )
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            cache[vintage] = vintage_loader.load_features()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    feats = cache[vintage]
    if not feats.get('exists', False):
        logger.warning('No %s features for this insertion; using the GUI features', vintage)
        return _get_features_df(controller, items)
    return feats['df']
