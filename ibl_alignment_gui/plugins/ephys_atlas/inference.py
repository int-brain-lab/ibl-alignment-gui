"""
Region-classifier inference for the channel-prediction plugin.

Loads the released fold-based XGBoost region classifier with
``ephysatlas.regionclassifier.RegionClassifier`` -- from the Hugging Face Hub (a repository and a
release tag) or from a local release directory -- and runs its ``predict`` to get per-channel
Cosmos regions from an ephys-feature table. A release is described by its ``ephysatlas_model.json``
manifest (features, classes and folds), so the GUI needs nothing else. Heavy ``ephysatlas`` imports
are deferred so this module can be imported without it installed; it is only required when a model
is loaded.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtCore, QtWidgets

from ibl_alignment_gui.plugins.ephys_atlas._common import (
    clear_predictions,
    features_for_vintage,
    has_features,
    plugin_state,
)
from ibl_alignment_gui.utils.helpers import shank_loop

if TYPE_CHECKING:
    import pandas as pd
    from ephysatlas.anatomy import ClassifierRegions
    from ephysatlas.regionclassifier import RegionClassifier

    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController

logger = logging.getLogger(__name__)

# Released channel-level region classifier on the Hugging Face Hub, pinned to the release tag of
# its vintage (the feature set it was trained on) rather than ``main``, which moves whenever a new
# release is published. Update the tag when a new release is adopted.
MODEL_REPO_ID = 'int-brain-lab/ea-decoder-channel-xgboost'
MODEL_REVISION = '2026_W26'
MANIFEST_FILE = 'ephysatlas_model.json'
MODEL_NAME = 'Inference'  # key under which the loaded model is cached on the plugin
PREDICTION_KEY = 'Inference Model'  # cache key for the argmax prediction
CUMULATIVE_KEY = 'Inference Cumulative'  # cache key for the cumulative prediction


@dataclass(frozen=True)
class InferenceModel:
    """A loaded region classifier and its feature/class contract.

    Attributes
    ----------
    classifier : RegionClassifier
        The loaded classifier; its ``predict`` averages the fold models.
    features : list of str
        Feature columns the model requires, in order.
    classes : list of int
        Region ids the model predicts, in column order.
    source : tuple of (str, str)
        ``(repo_id_or_directory, revision)`` the model was loaded from.
    vintage : str or None
        Label of the feature table the model was trained on (e.g. ``2026_W26``).
    """

    classifier: RegionClassifier
    features: list[str]
    classes: list[int]
    source: tuple[str, str]
    vintage: str | None = None


# -----------------------------------------------------------------------------
# GUI interaction
# -----------------------------------------------------------------------------


class _InferenceModelDialog(QtWidgets.QDialog):
    """Choose the released region classifier: a Hugging Face repo + tag, or a local release."""

    def __init__(self, parent, repo_id: str, revision: str, local_dir: Path | None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Load inference model')
        self.setMinimumWidth(560)
        form = QtWidgets.QFormLayout()
        self._repo = QtWidgets.QLineEdit(repo_id)
        self._revision = QtWidgets.QLineEdit(revision)
        form.addRow('Hugging Face repository', self._repo)
        form.addRow('Release tag', self._revision)

        local_row = QtWidgets.QHBoxLayout()
        self._local = QtWidgets.QLineEdit('' if local_dir is None else str(local_dir))
        self._local.setPlaceholderText(
            f'optional: a local release directory (containing {MANIFEST_FILE}) instead of the Hub'
        )
        browse = QtWidgets.QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        local_row.addWidget(self._local)
        local_row.addWidget(browse)
        form.addRow('Local release', local_row)

        note = QtWidgets.QLabel(
            'The model is downloaded once from the Hugging Face Hub and cached '
            '(~/.cache/ephysatlas/models). A local release takes precedence over the Hub.'
        )
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
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, f'Select a local release directory (containing {MANIFEST_FILE})'
        )
        if not chosen:
            return
        if not validate_model_folder(Path(chosen)):
            QtWidgets.QMessageBox.warning(
                self, 'Inference model', f'No {MANIFEST_FILE} found in:\n{chosen}'
            )
            return
        self._local.setText(chosen)

    def source(self) -> tuple[str, str]:
        """``(repo_id_or_directory, revision)`` to load."""
        local = self._local.text().strip()
        revision = self._revision.text().strip() or MODEL_REVISION
        return (local or self._repo.text().strip() or MODEL_REPO_ID), revision


def load_model_dialog(controller: AlignmentGUIController) -> bool:
    """Ask which release to use and load it (only when the source changed).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    bool
        True when a model is loaded, False when the user cancelled or loading failed.
    """
    plugin = plugin_state(controller)
    state = plugin.get(MODEL_NAME) or dict(model=None, source=None)
    plugin[MODEL_NAME] = state
    if not has_features(controller):
        QtWidgets.QMessageBox.warning(
            controller.view,
            'Channel Prediction',
            'No features found for this probe. '
            'Set via Plugins -> Channel Prediction -> Load features file...',
        )
        return False

    repo, revision = state['source'] or (MODEL_REPO_ID, MODEL_REVISION)
    local = Path(repo) if Path(str(repo)).is_dir() else None
    dialog = _InferenceModelDialog(
        controller.view, MODEL_REPO_ID if local else repo, revision, local
    )
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return state['model'] is not None

    source = dialog.source()
    if state['model'] is not None and state['source'] == source:
        return True
    if not _load_into_state(controller, state, source):
        return False
    invalidate_predictions(controller)
    logger.info('Inference model set to %s@%s', *source)
    return True


def _load_into_state(controller: AlignmentGUIController, state: dict, source: tuple) -> bool:
    """Load ``source`` behind a wait cursor and cache it on ``state``; report failures."""
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        model = load_inference_model(source)
    except Exception as exc:  # noqa: BLE001 - show any loading problem to the user
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.critical(
            controller.view,
            'Load inference model',
            f'Could not load {source[0]}@{source[1]}:\n{exc}',
        )
        return False
    QtWidgets.QApplication.restoreOverrideCursor()
    state.update(model=model, source=model.source)
    return True


@shank_loop
def invalidate_predictions(
    controller: AlignmentGUIController, items: ShankController, **kwargs
) -> None:
    """Drop cached inference predictions on a shank so the next click recomputes.

    Decorated with :func:`shank_loop`, so a single call iterates over every shank/config; the
    ``shank`` and ``config`` keywords injected by the decorator are absorbed via ``**kwargs``.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank whose cached predictions are cleared.
    """
    clear_predictions(items, PREDICTION_KEY, CUMULATIVE_KEY)


# -----------------------------------------------------------------------------
# Loading utils
# -----------------------------------------------------------------------------


def get_model(controller: AlignmentGUIController) -> InferenceModel | None:
    """Return the loaded model, loading it first when needed (None if the user cancelled).

    A ``source`` pre-set on the plugin state (``(repo_id_or_dir, revision)``, e.g. by a launcher
    script) loads without the dialog; otherwise the load dialog opens.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    InferenceModel or None
        The loaded model, or None if the user cancelled or loading failed.
    """
    state = plugin_state(controller).get(MODEL_NAME)
    if (
        state is not None
        and state.get('model') is None
        and state.get('source')
        and not _load_into_state(controller, state, tuple(state['source']))
    ):
        return None
    if state is None or state.get('model') is None:
        if not load_model_dialog(controller):
            return None
        state = plugin_state(controller).get(MODEL_NAME)
    return state['model']


def load_inference_model(source: tuple[str, str]) -> InferenceModel:
    """Load the region classifier from a local release directory or the Hugging Face Hub.

    Parameters
    ----------
    source : tuple of (str, str)
        ``(repo_id_or_directory, revision)``. A directory is loaded as is (the revision is then
        ignored); otherwise the Hugging Face repository is downloaded at ``revision`` and cached.

    Returns
    -------
    InferenceModel
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If a local directory has no ``ephysatlas_model.json`` manifest (e.g. a model exported in
        the old ``FOLD0X/meta.yaml``-only layout, which must be re-exported).
    """
    from ephysatlas.regionclassifier import RegionClassifier  # noqa: PLC0415 - heavy, lazy

    repo, revision = source
    local = Path(repo).is_dir()
    if local:
        if not validate_model_folder(Path(repo)):
            raise FileNotFoundError(
                f'{repo} has no {MANIFEST_FILE}; models in the old folds/FOLD0X/meta.yaml layout '
                'are no longer supported and must be re-exported with ibleatools.'
            )
        classifier = RegionClassifier(Path(repo))
    else:
        classifier = RegionClassifier.from_pretrained(repo, revision=revision)

    model = InferenceModel(
        classifier=classifier,
        features=list(classifier.inputs['features']),
        classes=[int(c) for c in classifier.config['classes']],
        # A local release ignores the revision, so record its vintage there for the dialog to show.
        source=(str(repo), classifier.index.get('vintage') or revision if local else revision),
        vintage=classifier.index.get('vintage'),
    )
    logger.info(
        'Loaded inference model %s (%d features, %d classes) from %s',
        classifier.index.get('vintage'),
        len(model.features),
        len(model.classes),
        classifier.path_model,
    )
    return model


def validate_model_folder(model_dir: Path) -> bool:
    """Return whether ``model_dir`` is a release directory (it holds the model manifest).

    Parameters
    ----------
    model_dir : Path
        The local model directory to validate.

    Returns
    -------
    bool
        True if ``model_dir`` contains ``ephysatlas_model.json``, else False.
    """
    return model_dir.joinpath(MANIFEST_FILE).is_file()


def validate_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Ensure the features DataFrame has every column the model expects.

    Fills an absent ``outside`` column with ``0.0`` (in-brain default, matching the upstream
    inference scripts); any other missing required column is a hard error.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-channel features (one row per channel).
    features : list of str
        Feature columns the model requires.

    Returns
    -------
    pandas.DataFrame
        The DataFrame, with ``outside`` added if it was missing.

    Raises
    ------
    ValueError
        If a required column other than ``outside`` is absent.
    """
    if 'outside' in features and 'outside' not in df.columns:
        df = df.copy()
        df['outside'] = 0.0
        logger.info("Filled missing 'outside' column with 0.0 (in-brain default).")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f'Features file is missing required column(s): {missing}')

    return df


@functools.cache
def classifier_regions() -> ClassifierRegions:
    """Brain regions that resolve every class a classifier can emit.

    The Allen regions plus ``void_fluid`` (id 2000), which the classifiers predict but the GUI's
    atlas does not know.

    Returns
    -------
    ClassifierRegions
        The regions, with ``void_fluid`` added.
    """
    from ephysatlas.anatomy import NEW_VOID, ClassifierRegions  # noqa: PLC0415 - heavy, lazy

    regions = ClassifierRegions()
    regions.add_new_region(NEW_VOID)
    return regions


# -----------------------------------------------------------------------------
# Model prediction
# -----------------------------------------------------------------------------


def _fold_mean_probas(
    controller: AlignmentGUIController, items: ShankController
) -> tuple[np.ndarray, np.ndarray, InferenceModel] | None:
    """Run the region classifier and return its fold-averaged probabilities.

    Shared core of :func:`predict` and :func:`predict_cumulative`: loads the shank's features,
    validates them against the model contract and runs the classifier.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, InferenceModel) or None
        The fold-averaged class probabilities (``n_channels`` x ``n_classes``, in
        ``model.classes`` order), the channel depths (µm) and the loaded model; or None when no
        model or no features are available.
    """
    # Resolve the model first so its load dialog (and its "load a features file first" guard)
    # runs before we bail on missing features, matching the spatial encoder path.
    model = get_model(controller)
    if model is None:
        return None

    df = features_for_vintage(controller, items, model.vintage)
    if df is None:
        QtWidgets.QMessageBox.warning(
            controller.view,
            'Channel Prediction',
            'No features found for this probe. Set via '
            'Plugins -> Channel Prediction -> Load features file...',
        )
        return None

    try:
        df = validate_features(df, model.features)
    except ValueError as exc:
        QtWidgets.QMessageBox.warning(
            controller.view,
            'Channel Prediction',
            f'{exc}\n\nThe inference model ({model.vintage}) needs features computed with the '
            f'{model.vintage} feature set.',
        )
        return None
    out = model.classifier.predict(df)
    # predict writes one p_<acronym> column per class, in the manifest's class order.
    mean_probas = out.filter(regex=r'^p_').to_numpy()
    depths = df['axial_um'].values

    return mean_probas, depths, model


def predict(
    controller: AlignmentGUIController, items: ShankController
) -> tuple[dict, np.ndarray] | None:
    """Predict the per-channel region (argmax over fold-averaged probabilities).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.

    Returns
    -------
    tuple of (dict, np.ndarray) or None
        The predicted regions (id, acronym, rgb, ... per channel) and channel depths (µm), or None
        if no prediction is available.
    """
    result = _fold_mean_probas(controller, items)
    if result is None:
        return None

    mean_probas, depths, model = result
    region_ids = np.array(model.classes)[np.argmax(mean_probas, axis=1)]

    return classifier_regions().get(region_ids), depths


def predict_cumulative(
    controller: AlignmentGUIController, items: ShankController
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray] | None:
    """Return cumulative region probabilities by depth for the stacked-band view.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, list of np.ndarray, np.ndarray) or None
        ``(cprobas, depths, colours, region_ids)`` or None if no prediction is available.
    """
    result = _fold_mean_probas(controller, items)
    if result is None:
        return None

    mean_probas, depths, model = result
    cprobas = mean_probas.cumsum(axis=1)
    region_ids = np.array(model.classes).astype(int)
    colours = list(classifier_regions().get(region_ids).rgb)

    return cprobas, depths, colours, region_ids
