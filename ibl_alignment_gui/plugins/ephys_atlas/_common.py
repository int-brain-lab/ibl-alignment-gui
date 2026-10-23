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

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.app.load_worker import TaskCancelledError
from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal, FeatureLoaderOne
from iblutil.util import Bunch

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController

logger = logging.getLogger(__name__)

PLUGIN_KEY = 'Channel Prediction'

# Track depths are stored along the histology track in meters; region plots expect microns.
M_TO_UM = 1e6

# Auto-alignment's region-plot options (also the per-shank prediction cache keys): the predicted
# regions, and the alignment confidence along the probe.
AUTO_ALIGNMENT_KEY = 'Auto-alignment'
AUTO_ALIGNMENT_CONFIDENCE_KEY = 'Auto-alignment (confidence)'


class PredictionError(Exception):
    """A model could not make a prediction; the message is meant for the user."""


@dataclass(frozen=True)
class ModelSpec:
    """A loadable Ephys Atlas model, as the shared load dialog and model cache need it.

    Attributes
    ----------
    key : str
        The plugin-state key the loaded model is cached under: a decoder's name, or 'Encoding'.
    title : str
        Title of the load dialog, e.g. 'Load alignment model'.
    default_source : Callable
        ``default_source() -> (repo, revision)``: the default Hugging Face repository and release
        tag. Called only when the load dialog opens, so it may import ``ephysatlas`` itself.
    note : str
        Explanation shown in the load dialog.
    load : Callable
        ``load(source, progress) -> model``: loads ``source`` (``(repo_or_dir, revision)``),
        reporting through ``progress(fraction, message)``. It does its heavy imports itself, so
        this module stays light.
    invalidate : Callable
        ``invalidate(controller)``: drops the shanks' cached predictions of this model.
    options : tuple of str
        The region-plot options the model offers, e.g. its most probable region and its
        cumulative probabilities. The first is selected after the model is loaded from the menu.
    compute : Callable
        ``compute(controller, items, model, progress) -> (views, extra)``: runs the loaded
        ``model`` on a shank, without Qt (it may run on a background thread). ``progress(fraction,
        message)`` reports each step (``fraction`` None while a step's length is unknown, e.g. a
        download) and raises ``TaskCancelledError`` after Cancel. Returns the views of all its
        options, ``{option: view}``, and anything ``on_computed`` needs.
    needs_torch : bool
        Whether the model needs torch; it is only offered when torch is installed.
    model_class : str
        The ``model_class`` its release manifest (``ephysatlas_model.json``) declares, which
        identifies a local release directory as this model.
    on_computed : Callable or None
        ``on_computed(controller, items, extra)``: run on the main thread after ``compute``,
        e.g. to open a result window; None when the model needs nothing.
    """

    key: str
    title: str
    default_source: Callable[[], tuple[str, str]]
    note: str
    load: Callable[[tuple[str, str], Callable[[float, str], None]], object]
    invalidate: Callable[[AlignmentGUIController], None]
    options: tuple[str, ...]
    compute: Callable[..., tuple[dict, object]]
    needs_torch: bool
    model_class: str
    on_computed: Callable[[AlignmentGUIController, ShankController, object], None] | None = None


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
        self._default_repo, self._default_revision = spec.default_source()
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
        revision = self._revision.text().strip() or self._default_revision
        return (local or self._repo.text().strip() or self._default_repo), revision


# The ``features_paths`` key of a features file for every shank (split by channel), as opposed to
# a ``(shank, config)`` key for one shank's file.
ALL_SHANKS = 'all shanks'

NO_FEATURES_MESSAGE = (
    'No features found for this probe. '
    'Set via Plugins -> Channel Prediction -> Load features file...'
)


def model_vintage(index: dict | None) -> str:
    """Return the feature vintage a model release was trained on (its manifest ``vintage``).

    Parameters
    ----------
    index : dict or None
        The release's manifest (``ephysatlas_model.json``), as loaded with the model.

    Returns
    -------
    str
        The vintage, e.g. ``2026_W39``.

    Raises
    ------
    PredictionError
        If the manifest does not name one.
    """
    vintage = (index or {}).get('vintage')
    if not vintage:
        raise PredictionError(
            "The model's release does not say which feature vintage it was trained on (no "
            "'vintage' in its ephysatlas_model.json), so the features to run it on are unknown."
        )
    return vintage


def features_vintage(df: pd.DataFrame) -> str | None:
    """Return the vintage a features table was computed with, when it says so.

    The vintage is the table's ``vintage`` attribute (``df.attrs['vintage']``), which pandas keeps
    in a parquet file: the GUI sets it on the ephys-atlas tables it reads, and a features file
    written with it set carries it too.

    Parameters
    ----------
    df : pandas.DataFrame
        A features table.

    Returns
    -------
    str or None
        The vintage, e.g. ``2026_W26``, or None if the table does not say.
    """
    return df.attrs.get('vintage') or None


def missing_features_error(
    missing: list[str], vintage: str, data_vintage: str | None = None
) -> PredictionError:
    """Return the error for a features table lacking columns a model needs.

    It also points to the other way out: a release of the model matching the features' vintage.

    Parameters
    ----------
    missing : list of str
        The columns the model needs that the table lacks.
    vintage : str
        The feature vintage the model was trained on.
    data_vintage : str or None
        The vintage the features were computed with (:func:`features_vintage`), if known.

    Returns
    -------
    PredictionError
        The error, naming the missing columns, the vintage and the release to load instead.
    """
    message = (
        f'Features file is missing required column(s): {missing}\n\n'
        f'The model ({vintage}) needs features computed with the {vintage} feature set.'
    )
    if data_vintage and data_vintage != vintage:
        message += (
            f'\n\nAlternatively, load a {data_vintage} release of the model (Plugins -> '
            f'Channel Prediction -> Load ... model, release tag {data_vintage}) to match your '
            'features.'
        )
    else:
        message += (
            '\n\nAlternatively, load a release of the model whose vintage matches your features '
            '(Plugins -> Channel Prediction -> Load ... model, with that vintage as the release '
            'tag).'
        )
    return PredictionError(message)


def choose_source(controller: AlignmentGUIController, spec: ModelSpec) -> tuple[str, str] | None:
    """Ask which release of ``spec``'s model to use (main thread).

    The dialog opens with the model's current source, or its default.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    spec : ModelSpec
        The model to choose a release of.

    Returns
    -------
    tuple of (str, str) or None
        The chosen ``(repo_id_or_directory, revision)``; None if the user cancelled, or if no
        shank has features yet (the user is warned).
    """
    if not has_features(controller):
        QtWidgets.QMessageBox.warning(controller.view, PLUGIN_KEY, NO_FEATURES_MESSAGE)
        return None

    state = plugin_state(controller).get(spec.key) or {}
    default_repo, default_revision = spec.default_source()
    repo, revision = state.get('source') or (default_repo, default_revision)
    local = Path(repo) if Path(str(repo)).is_dir() else None
    dialog = ModelSourceDialog(
        controller.view, spec, default_repo if local else repo, revision, local
    )
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.source()


def load_model(spec: ModelSpec, source: tuple[str, str], progress: Callable) -> object:
    """Load ``source`` with ``spec.load``, without Qt (it may run on a background thread).

    Parameters
    ----------
    spec : ModelSpec
        The model to load.
    source : tuple of (str, str)
        ``(repo_id_or_directory, revision)`` to load.
    progress : Callable
        ``progress(fraction, message)``, told as loading proceeds.

    Returns
    -------
    object
        The loaded model.

    Raises
    ------
    PredictionError
        If the model cannot be loaded, naming the source and the reason.
    TaskCancelledError
        If Cancel was pressed, at the next progress report.
    """
    try:
        return spec.load(source, progress)
    except TaskCancelledError:
        raise
    except Exception as exc:
        raise PredictionError(f'Could not load {source[0]}@{source[1]}:\n{exc}') from exc


def store_model(
    controller: AlignmentGUIController, spec: ModelSpec, model, source: tuple[str, str]
) -> None:
    """Keep a loaded model in the plugin state and clear its old predictions (main thread).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    spec : ModelSpec
        The model.
    model : object
        The loaded model.
    source : tuple of (str, str)
        The ``(repo_id_or_directory, revision)`` it was loaded from.
    """
    plugin_state(controller)[spec.key] = dict(model=model, source=tuple(source))
    spec.invalidate(controller)
    logger.info('%s set to %s@%s', spec.key, *source)


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


def alignment_windows(controller: AlignmentGUIController) -> dict:
    """Return the auto-alignment result windows, created empty on first use.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    dict
        The ``AlignmentResultWindow`` of each aligned shank, keyed by ``(shank, config)``.
    """
    return plugin_state(controller).setdefault('alignment_windows', {})


def close_alignment_windows(controller: AlignmentGUIController) -> None:
    """Close every auto-alignment result window and forget them (e.g. on a new session).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    windows = alignment_windows(controller)
    for window in windows.values():
        window.close()
    windows.clear()
    update_results_action(controller)


def show_alignment_windows(controller: AlignmentGUIController) -> None:
    """Show every kept auto-alignment result window again, bringing it to the front.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    for window in alignment_windows(controller).values():
        if not window.isVisible():
            window.show()
        window.raise_()
        window.activateWindow()


def update_results_action(controller: AlignmentGUIController) -> None:
    """Enable the 'Show auto-alignment results' menu item only while a result window exists.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """
    action = plugin_state(controller).get('alignment_results_action')
    if action is not None:
        action.setEnabled(bool(alignment_windows(controller)))


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


def local_features_path(controller: AlignmentGUIController, items: ShankController) -> Path | None:
    """Return the features file chosen for this shank and configuration, if any.

    Files chosen with ``Load features file…`` are kept per ``(shank, config)`` in the plugin
    state's ``features_paths``; the :data:`ALL_SHANKS` key holds one file for every shank, split
    by channel.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank.

    Returns
    -------
    Path or None
        The shank's own file, else the file for all shanks, else None.
    """
    paths = plugin_state(controller).get('features_paths') or {}
    return paths.get((items.name, items.config), paths.get(ALL_SHANKS))


def has_local_features(controller: AlignmentGUIController, items: ShankController) -> bool:
    """Return whether a features file was chosen for this shank with ``Load features file…``.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank.

    Returns
    -------
    bool
        True when a local features file overrides this shank's features, else False.
    """
    return local_features_path(controller, items) is not None


def has_features(controller: AlignmentGUIController) -> bool:
    """Return whether per-channel features are available to run a prediction on.

    True when a features file was chosen for any shank (the offline source) or any loaded shank
    already holds a non-empty features table (the online source).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    bool
        True when features are available for at least one loaded shank, else False.
    """
    if plugin_state(controller).get('features_paths'):
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
    mode where no features loader exists, loads the features file chosen for this shank
    (:func:`local_features_path`) through the shank (:meth:`ShankHandler.load_ephys_features`),
    which keeps them for the next prediction.

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
        features_path = local_features_path(controller, items)
        if features_path is None:
            return None
        # The shank loads the file, split down to its own channels.
        feats = items.model.load_ephys_features(FeatureLoaderLocal(features_path))
        if not feats.get('exists', False):
            return None
    return feats['df']


def features_for_vintage(
    controller: AlignmentGUIController,
    items: ShankController,
    vintage: str,
    progress: Callable[[float | None, str], None] | None = None,
) -> pd.DataFrame | None:
    """Return the shank's features from the feature table a model was trained on.

    The GUI loads the latest ephys-atlas feature table, whose feature set can differ from the one
    a model was trained on (columns are renamed or dropped between vintages). When the features
    come from ONE, the table labelled ``vintage`` is loaded instead, and cached on the shank so the
    feature plots keep the latest table. A local features file (from the session or chosen with
    ``Load features file…``) is used as is.

    The ephys-atlas tables are already outlier-treated when ephysatlas builds them (the alpha
    features), so the features are not treated again here.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.
    vintage : str
        Label of the feature table the model was trained on (e.g. ``2026_W26``).
    progress : Callable or None
        ``progress(fraction, message)``, told when the vintage table is loaded (which may download
        it). No Qt is used here, so this can run on a background thread.

    Returns
    -------
    pandas.DataFrame or None
        The features DataFrame, or None when no features are available.

    Raises
    ------
    PredictionError
        If the model's vintage table cannot be loaded, or has no features for this insertion.
    """
    loader = items.model.loaders.get('features')
    if (
        has_local_features(controller, items)
        or not isinstance(loader, FeatureLoaderOne)
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
        if progress is not None:
            progress(None, f'Loading the {vintage} features for {items.name}…')
        try:
            cache[vintage] = vintage_loader.load_features()
        except Exception as exc:
            raise PredictionError(
                f'The {vintage} features could not be loaded for this insertion:\n{exc}'
            ) from exc
    feats = cache[vintage]
    if not feats.get('exists', False):
        # Features of another vintage would give a confident but wrong prediction. Instead, point
        # to the other way out: a release of the model trained on this session's vintage (one may
        # not exist for every model, so this names the vintage rather than promising a release).
        message = (
            f'No {vintage} features are available for this insertion; the model needs features '
            f'computed with the {vintage} feature set.'
        )
        if loader.feature_version:
            message += (
                f'\n\nAlternatively, load a {loader.feature_version} release of the model '
                f'(Plugins -> Channel Prediction -> Load ... model, release tag '
                f'{loader.feature_version}) to match the features of this session.'
            )
        raise PredictionError(message)
    return feats['df']


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
