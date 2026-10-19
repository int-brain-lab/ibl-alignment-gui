"""
Region-decoder inference for the channel-prediction plugin.

Loads the released channel-level region decoders with ``ephysatlas.load_pretrained`` -- from the
Hugging Face Hub (a repository and a release tag) or from a local release directory -- and runs
their ``predict`` to get per-channel Cosmos regions from an ephys-feature table. Two decoders are
offered (:data:`DECODERS`): the fold-averaged XGBoost classifier and the ProbeTransformer seed
ensemble. Both ``predict`` outputs carry the best region (``predicted_atlas_id``) and one
``p_<acronym>`` probability column per class.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets

from ibl_alignment_gui.plugins.ephys_atlas._common import (
    clear_predictions,
    features_for_vintage,
    has_features,
    plugin_state,
)
from ibl_alignment_gui.utils.helpers import shank_loop

if TYPE_CHECKING:
    from ephysatlas.anatomy import ClassifierRegions

    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController

logger = logging.getLogger(__name__)


class Decoder(NamedTuple):
    """A released region decoder.

    Attributes
    ----------
    repo : str
        Its Hugging Face repository.
    model_class : str
        The ``model_class`` its release manifest (``ephysatlas_model.json``) declares, which
        identifies a local release directory as this decoder.
    needs_torch : bool
        Whether it needs torch (only offered when torch is installed).
    """

    repo: str
    model_class: str
    needs_torch: bool


# Released channel-level region decoders on the Hugging Face Hub, by region-plot option name (also
# the plugin-state key the loaded model is cached under). The ProbeTransformer is a torch model;
# the XGBoost decoder is not.
DECODERS = {
    'XGBoost region prediction': Decoder(
        repo='int-brain-lab/ea-decoder-channel-xgboost',
        model_class='xgboost.sklearn.XGBClassifier',
        needs_torch=False,
    ),
    'ProbeTransformer region prediction': Decoder(
        repo='int-brain-lab/ea-decoder-channel-transformer',
        model_class='ProbeTransformer',
        needs_torch=True,
    ),
}
# Release tag of the vintage (the feature set the decoders were trained on), pinned rather than
# ``main``, which moves whenever a new release is published. Update it when adopting a new release.
MODEL_REVISION = '2026_W39'
# Suffix of the region-plot option showing a decoder's cumulative class probabilities.
CUMULATIVE_SUFFIX = ' (cumulative)'


# -----------------------------------------------------------------------------
# GUI interaction
# -----------------------------------------------------------------------------


class _InferenceModelDialog(QtWidgets.QDialog):
    """Choose a released region decoder: a Hugging Face repo + tag, or a local release."""

    def __init__(
        self, parent, title: str, repo_id: str, revision: str, local_dir: Path | None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self._default_repo = repo_id
        form = QtWidgets.QFormLayout()
        self._repo = QtWidgets.QLineEdit(repo_id)
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

        note = QtWidgets.QLabel(
            'The model is loaded with ephysatlas.load_pretrained, downloaded once from the '
            'Hugging Face Hub and cached (~/.cache/ephysatlas/models). A local release takes '
            'precedence over the Hub.'
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
        # load_pretrained reports a directory that is not a release (no manifest) when loading.
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select a local release directory (containing ephysatlas_model.json)'
        )
        if chosen:
            self._local.setText(chosen)

    def source(self) -> tuple[str, str]:
        """``(repo_id_or_directory, revision)`` to load."""
        local = self._local.text().strip()
        revision = self._revision.text().strip() or MODEL_REVISION
        return (local or self._repo.text().strip() or self._default_repo), revision


def load_model_dialog(controller: AlignmentGUIController, name: str) -> bool:
    """Ask which release of decoder ``name`` to use and load it (only when the source changed).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    name : str
        The decoder, a key of :data:`DECODERS`.

    Returns
    -------
    bool
        True when a model is loaded, False when the user cancelled or loading failed.
    """
    plugin = plugin_state(controller)
    state = plugin.get(name) or dict(model=None, source=None)
    plugin[name] = state
    if not has_features(controller):
        QtWidgets.QMessageBox.warning(
            controller.view,
            'Channel Prediction',
            'No features found for this probe. '
            'Set via Plugins -> Channel Prediction -> Load features file...',
        )
        return False

    default_repo = DECODERS[name].repo
    repo, revision = state['source'] or (default_repo, MODEL_REVISION)
    local = Path(repo) if Path(str(repo)).is_dir() else None
    dialog = _InferenceModelDialog(
        controller.view, f'Load {name} model', default_repo if local else repo, revision, local
    )
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return state['model'] is not None

    source = dialog.source()
    if state['model'] is not None and state['source'] == source:
        return True
    if not _load_into_state(controller, state, source):
        return False
    invalidate_predictions(controller)
    logger.info('%s model set to %s@%s', name, *source)
    return True


def _load_into_state(controller: AlignmentGUIController, state: dict, source: tuple) -> bool:
    """Load ``source`` behind a wait cursor and cache it on ``state``; report failures."""
    from ephysatlas import load_pretrained  # noqa: PLC0415 - heavy, lazy

    repo, revision = source
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        # A Hub repo id or a local release directory; the manifest picks the model class.
        model = load_pretrained(repo, revision=revision)
    except Exception as exc:  # noqa: BLE001 - show any loading problem to the user
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.critical(
            controller.view,
            'Load region decoder',
            f'Could not load {repo}@{revision}:\n{exc}',
        )
        return False
    QtWidgets.QApplication.restoreOverrideCursor()
    state.update(model=model, source=(str(repo), revision))
    logger.info(
        'Loaded region decoder %s %s (%d features, %d classes)',
        model.index.get('model_class'),
        model.index.get('vintage'),
        len(model.inputs['features']),
        len(model.config['classes']),
    )
    return True


@shank_loop
def invalidate_predictions(
    controller: AlignmentGUIController, items: ShankController, **kwargs
) -> None:
    """Drop every decoder's cached predictions on a shank so the next click recomputes.

    Decorated with :func:`shank_loop`, so a single call iterates over every shank/config; the
    ``shank`` and ``config`` keywords injected by the decorator are absorbed via ``**kwargs``.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank whose cached predictions are cleared.
    """
    clear_predictions(items, *DECODERS, *(name + CUMULATIVE_SUFFIX for name in DECODERS))


# -----------------------------------------------------------------------------
# Loading utils
# -----------------------------------------------------------------------------


def get_model(controller: AlignmentGUIController, name: str):
    """Return the loaded decoder ``name``, loading it first when needed (None if cancelled).

    A ``source`` pre-set on the plugin state (``(repo_id_or_dir, revision)``, e.g. by a launcher
    script) loads without the dialog; otherwise the load dialog opens.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    name : str
        The decoder, a key of :data:`DECODERS`.

    Returns
    -------
    RegionClassifier or ProbeTransformerClassifier or None
        The model returned by ``ephysatlas.load_pretrained``, or None if the user cancelled or
        loading failed.
    """
    state = plugin_state(controller).get(name)
    if (
        state is not None
        and state.get('model') is None
        and state.get('source')
        and not _load_into_state(controller, state, tuple(state['source']))
    ):
        return None
    if state is None or state.get('model') is None:
        if not load_model_dialog(controller, name):
            return None
        state = plugin_state(controller).get(name)
    return state['model']


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


def _predict_df(
    controller: AlignmentGUIController, items: ShankController, name: str
) -> tuple[pd.DataFrame, np.ndarray, object] | None:
    """Run decoder ``name`` on the shank's features and return its prediction table.

    Shared core of :func:`predict` and :func:`predict_cumulative`: loads the shank's features
    from the table of the model's vintage, gives them the ``(pid, channel)`` index the decoders
    expect and runs the model's ``predict``.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.
    name : str
        The decoder, a key of :data:`DECODERS`.

    Returns
    -------
    tuple of (pandas.DataFrame, np.ndarray, model) or None
        The model's ``predict`` output (one row per predicted channel, with ``predicted_atlas_id``
        and a ``p_<acronym>`` column per class, in the manifest's class order), the depths (µm)
        of those channels and the loaded model; or None when no model or no features are
        available, or the features lack a column the model needs.
    """
    # Resolve the model first so its load dialog (and its "load a features file first" guard)
    # runs before we bail on missing features, matching the spatial encoder path.
    model = get_model(controller, name)
    if model is None:
        return None

    vintage = model.index.get('vintage')
    df = features_for_vintage(controller, items, vintage)
    if df is None:
        QtWidgets.QMessageBox.warning(
            controller.view,
            'Channel Prediction',
            'No features found for this probe. Set via '
            'Plugins -> Channel Prediction -> Load features file...',
        )
        return None

    if 'outside' in model.inputs['features'] and 'outside' not in df.columns:
        # In-brain default, matching the upstream inference scripts.
        df = df.assign(outside=0.0)
        logger.info("Filled missing 'outside' column with 0.0 (in-brain default).")
    if list(df.index.names) != ['pid', 'channel']:
        # The decoders take one probe per pid (the ProbeTransformer attends across all channels
        # of a pid): a flat table holds this one shank, so give all its rows the same pid.
        pids = np.zeros(len(df), dtype=int)
        df = df.set_index(pd.MultiIndex.from_arrays([pids, df.index], names=['pid', 'channel']))

    try:
        out = model.predict(df)
    except KeyError as exc:  # a feature the model needs is missing, named in the message
        QtWidgets.QMessageBox.warning(
            controller.view,
            'Channel Prediction',
            f'{exc}\n\nThe {name} model ({vintage}) needs features computed with the {vintage} '
            'feature set.',
        )
        return None
    # The ProbeTransformer drops channels with non-finite features: take the depths of the rows
    # that were predicted.
    depths = df.loc[out.index, 'axial_um'].to_numpy()

    return out, depths, model


def predict(
    controller: AlignmentGUIController, items: ShankController, name: str
) -> tuple[dict, np.ndarray] | None:
    """Predict the per-channel region with decoder ``name`` (its most probable class).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.
    name : str
        The decoder, a key of :data:`DECODERS`.

    Returns
    -------
    tuple of (dict, np.ndarray) or None
        The predicted regions (id, acronym, rgb, ... per channel) and channel depths (µm), or None
        if no prediction is available.
    """
    result = _predict_df(controller, items, name)
    if result is None:
        return None

    out, depths, _ = result

    return classifier_regions().get(out['predicted_atlas_id'].to_numpy()), depths


def predict_cumulative(
    controller: AlignmentGUIController, items: ShankController, name: str
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray] | None:
    """Return decoder ``name``'s cumulative region probabilities by depth for the stacked view.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.
    name : str
        The decoder, a key of :data:`DECODERS`.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, list of np.ndarray, np.ndarray) or None
        ``(cprobas, depths, colours, region_ids)`` or None if no prediction is available.
    """
    result = _predict_df(controller, items, name)
    if result is None:
        return None

    out, depths, model = result
    # One p_<acronym> column per class, in the manifest's class order.
    cprobas = out.filter(regex=r'^p_').to_numpy().cumsum(axis=1)
    region_ids = np.array(model.config['classes']).astype(int)
    colours = list(classifier_regions().get(region_ids).rgb)

    return cprobas, depths, colours, region_ids
