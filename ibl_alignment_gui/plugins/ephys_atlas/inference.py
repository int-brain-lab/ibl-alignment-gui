"""
Region-classifier inference for the channel-prediction plugin.

Loads a fold-based XGBoost region classifier — either from a local directory or downloaded from
S3 — and runs ``ephysatlas.regionclassifier.infer_regions`` to predict per-channel Cosmos regions
from an ephys-feature table. Heavy ``ephysatlas`` imports are deferred so this module can be
imported in offline mode without it installed; it is only required when inference actually runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from qtpy import QtWidgets

from one.api import ONE

from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    clear_predictions,
    has_features,
    has_one_connection,
    needs_reload,
    plugin_state,
    s3_cache_root,
)
from ibl_alignment_gui.utils.utils import shank_loop
from iblutil.numerical import ismember

if TYPE_CHECKING:
    import pandas as pd

    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController

logger = logging.getLogger(__name__)

# Default model downloaded from S3 when neither a local model dir nor an explicit S3 model name
# is configured (preserves the out-of-the-box S3 download from prediction_models_051526).
MODEL_VINTAGE = 'xgboost_channels/2026_W12_Cosmos_careless-clover-dingo'
MODEL_NAME = 'Inference'  # key under which the loaded model is cached on the plugin
MAX_FOLDS = 10  # upper bound when discovering FOLD0X directories
PREDICTION_KEY = 'Inference Model'  # cache key for the argmax prediction
CUMULATIVE_KEY = 'Inference Cumulative'  # cache key for the cumulative prediction

# Named S3 models offered in the online load dialog (first entry is the default vintage).
S3_MODEL_NAMES = [
    MODEL_VINTAGE,
    '2024_W50_Cosmos_lid-basket-sense/2024_W50_Cosmos_lid-basket-sense',
]


@dataclass(frozen=True)
class InferenceModel:
    """A loaded fold-based region classifier and its feature/class contract.

    Attributes
    ----------
    features : list of str
        Feature columns the model requires, in order.
    classes : list of int
        Cosmos region ids the model predicts, in column order.
    model_path : Path
        The folds directory passed to ``infer_regions``.
    n_folds : int
        Number of folds discovered during validation.
    """

    features: list[str]
    classes: list[int]
    model_path: Path
    n_folds: int



# -----------------------------------------------------------------------------
# GUI interaction
# -----------------------------------------------------------------------------

class _InferenceModelDialog(QtWidgets.QDialog):
    """Inference-model selection dialog with optional dropdown and a local-folder picker.

    A "Browse…" button lets the user pick (and validate) a local model directory; this is the
    only source in offline mode. When ``options`` is given (online mode) a dropdown of named S3
    models is shown above the folder row, and a chosen local directory takes precedence over the
    dropdown selection. Folder validation happens on selection: an invalid directory raises a
    warning and leaves the dialog open.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        title: str,
        options: list[str] | None = None,
        current: str | None = None,
        current_dir: Path | None = None,
    ) -> None:
        """Build the dialog.

        Parameters
        ----------
        parent : QtWidgets.QWidget
            Parent widget for the dialog.
        title : str
            Window title.
        options : list of str or None
            Named S3 models to offer in a dropdown (online mode). When None or empty no dropdown is
            shown and only the local-folder picker is available (offline mode).
        current : str or None
            Model name to pre-select in the dropdown, if present in ``options``.
        current_dir : Path or None
            Local model directory to pre-fill in the line edit (the currently-loaded local model).
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self._local_dir: Path | None = Path(current_dir) if current_dir else None
        layout = QtWidgets.QVBoxLayout(self)

        # Dropdown of named S3 models (online mode only).
        self._combo: QtWidgets.QComboBox | None = None
        if options:
            layout.addWidget(QtWidgets.QLabel('Select model:'))
            self._combo = QtWidgets.QComboBox()
            self._combo.addItems(options)
            if current in options:
                self._combo.setCurrentIndex(options.index(current))
            layout.addWidget(self._combo)

        # Local model directory row (takes precedence over the dropdown when set). Pre-filled with
        # the currently-loaded local model dir, if any, so it reflects the active source on open.
        self._dir_edit = QtWidgets.QLineEdit()
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setPlaceholderText('local inference model dir (containing folds/FOLD00/)')
        if current_dir is not None:
            self._dir_edit.setText(str(current_dir))
        browse = QtWidgets.QPushButton('Browse…')
        browse.clicked.connect(self._browse_model)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Local folder:'))
        row.addWidget(self._dir_edit)
        row.addWidget(browse)
        layout.addLayout(row)

        # Choosing a dropdown model clears any local folder so the dropdown selection takes effect.
        # 'activated' fires only on user interaction, so the init-time setCurrentIndex above and
        # any pre-filled local dir are left untouched.
        if self._combo is not None:
            self._combo.activated.connect(self._clear_dir)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _browse_model(self) -> None:
        """Pick the inference model dir, validating its folds/FOLD00 structure."""
        chosen_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select model directory (containing folds/FOLD00/)')
        if not chosen_path:
            return
        chosen_path = Path(chosen_path)
        if not validate_model_folder(chosen_path):
            QtWidgets.QMessageBox.warning(
                self, 'Inference model',
                f'No "folds/FOLD00" (or "FOLD00") found under:\n{chosen_path}')
            return
        self._local_dir = chosen_path
        self._dir_edit.setText(str(chosen_path))

    def _clear_dir(self, *_) -> None:
        """Drop a chosen/pre-filled local dir so the dropdown selection is used instead."""
        self._local_dir = None
        self._dir_edit.clear()

    @property
    def local_dir(self) -> Path | None:
        """Return the validated local model directory, or None if none was chosen."""
        return self._local_dir

    def selected_model(self) -> str | None:
        """Return the dropdown model name, or None in offline (no-dropdown) mode."""
        return self._combo.currentText() if self._combo is not None else None


def _current_local_dir(current_model: dict | None) -> Path | None:
    """Reconstruct the local model folder the user picked from a cached model.

    A local model is cached with ``model_name`` None and a ``model_path`` pointing at its folds
    directory. The folder the user originally selected is that path with a trailing ``folds``
    segment stripped, so the dialog can pre-fill the line edit with the model dir rather than the
    nested folds subdirectory.

    Parameters
    ----------
    current_model : dict or None
        The cached Channel Prediction model state, or None if none is loaded.

    Returns
    -------
    Path or None
        The local model directory, or None if no local model is cached.
    """
    if current_model is None or current_model['model_name'] is not None:
        return None
    folds_path = Path(current_model['local_inference_dir'])
    return folds_path.parent if folds_path.name == 'folds' else folds_path


def load_model_dialog(controller: AlignmentGUIController) -> bool:
    """Run the full inference-model load GUI and cache the result.

    Shows an intermediate dialog in both modes: offline offers only a local-folder picker, online
    adds a dropdown of named S3 models. A chosen local folder takes precedence over the dropdown.
    Calls :func:`load_inference_model` when the user selects a new or different model.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    bool
        True if a model was loaded, False if the user cancelled.
    """
    plugin = plugin_state(controller)
    current_model = plugin.get(MODEL_NAME, None)

    if current_model is None:
        current_model = dict(
            model=None,
            model_name=None,
            local_inference_dir=None,
        )
        plugin[MODEL_NAME] = current_model

    current_dir = current_model['local_inference_dir']

    has_one, one = has_one_connection(controller)

    if not has_one:
        # Offline inference reads features from a locally-loaded parquet; without one there is
        # nothing to predict on, so steer the user to load it before choosing a model.
        if not has_features(controller):
            QtWidgets.QMessageBox.warning(
                controller.view, 'Channel Prediction',
                'Load a features file first via "Load features file…" before loading a model.')
            return False
        dialog = _InferenceModelDialog(
            controller.view, 'Load Inference Model', current_dir=current_dir)
    else:
        # Online the model loads without a local features file, so check here that the insertion
        # actually has features to predict on before letting the user pick a model.
        if not has_features(controller):
            QtWidgets.QMessageBox.warning(
                controller.view, 'Channel Prediction',
                'No features found for this insertion.')
            return False
        dialog = _InferenceModelDialog(
            controller.view, 'Load Inference Model', S3_MODEL_NAMES,
            current_model['model_name'] or MODEL_VINTAGE, current_dir)

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return False

    if dialog.local_dir is not None:
        # Local folder takes precedence over the dropdown in both modes.
        if not needs_reload(current_model, local_inference_dir=dialog.local_dir):
            return True

        load_inference_model(controller, model_dir=dialog.local_dir, one=one)
        invalidate_predictions(controller)
        logger.info('Inference model set to %s', dialog.local_dir)
        return True

    # TODO put this into _on_accept
    if not has_one:
        QtWidgets.QMessageBox.warning(
            controller.view, 'Channel Prediction',
            'Offline mode needs a local inference model dir.')
        # No dropdown offline: OK without a folder selection means nothing to load.
        return False

    model_name = dialog.selected_model()
    if not needs_reload(current_model, model_name=model_name):
        return True
    load_inference_model(controller, model_name=model_name, one=one)
    invalidate_predictions(controller)
    logger.info('Inference model set to %s', model_name)
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

def is_model_loaded(controller: AlignmentGUIController) -> bool:
    """Return whether an inference model is cached on the plugin.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    bool
        True if a model has been loaded, else False.
    """
    return plugin_state(controller).get(MODEL_NAME) is not None


def get_model(controller: AlignmentGUIController) -> InferenceModel | None:
    """Return the cached inference model, loading or prompting for it on first use.

    If no Channel Prediction model state exists yet, opens the load dialog and retries. If state
    exists but the model has not been built, rebuilds it from the cached local dir when present
    (otherwise prompts via the dialog) and retries. Returns None if the user cancels the dialog.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    InferenceModel or None
        The loaded model, or None if the user cancelled loading.
    """
    plugin = plugin_state(controller).get(MODEL_NAME, None)

    # First time loading: prompt the user; bail out if they cancel.
    if plugin is None:
        if not load_model_dialog(controller):
            return None
        return get_model(controller)

    if plugin.get('model', None) is None:
        if plugin.get('local_inference_dir', None) is not None:
            load_inference_model(
                controller,
                model_dir=plugin['local_inference_dir'],
            )
        elif not load_model_dialog(controller):
            return None
        return get_model(controller)

    return plugin['model']


def load_inference_model(
    controller: AlignmentGUIController,
    model_dir: str | Path | None = None,
    model_name: str | None = None,
    one: ONE | None = None
) -> None:
    """Load the region-classifier model from a local directory or download it from S3.

    Caches the result on the Channel Prediction plugin: the loaded :class:`InferenceModel` under
    ``[MODEL_NAME]['model']``, plus the chosen source (``local_inference_dir`` and ``model_name``).

    Parameters
    ----------
    controller : AlignmentGUIController
        Provides ONE access for the S3 download path.
    model_dir : str or Path or None
        If given, use this local model directory and skip S3. The fold models are expected under
        ``<model_dir>/folds/FOLD0X`` (or directly under ``<model_dir>`` if there is no ``folds``
        subdirectory).
    model_name : str or None
        S3 model name to download when ``model_dir`` is None. May be a nested name such as
        ``xgboost_channels/2026_W12_Cosmos_careless-clover-dingo``. Defaults to
        :data:`MODEL_VINTAGE` when not provided.
    one : ONE or None
        ONE connection used for the S3 download.

    Raises
    ------
    RuntimeError
        If an S3 download is requested but no ONE/Alyx connection is available.
    """

    plugin = plugin_state(controller)[MODEL_NAME]

    if model_dir is not None:
        # Local model: point straight at the folds directory; no ONE / S3 access needed.
        folds_path = _get_model_path_from_local(Path(model_dir))
        plugin['local_inference_dir'] = folds_path
        plugin['model_name'] = None
    else:
        s3_name = model_name or MODEL_VINTAGE
        folds_path = _get_model_path_from_s3(one, s3_name)

        plugin['local_inference_dir'] = folds_path
        plugin['model_name'] = s3_name

    if folds_path is None:
        raise RuntimeError(
            'Could not resolve an inference model: the S3 download needs a ONE/Alyx connection. '
            'Set a local model directory instead.')

    # Validation contract: confirm all folds agree and extract FEATURES / CLASSES.
    features, classes, n_folds = validate_model(folds_path)

    # model_name is None for local models, marking the source as a local folder (vs S3).
    plugin['model'] = InferenceModel(
        features=features,
        classes=classes,
        model_path=folds_path,
        n_folds=n_folds,
    )


def _get_model_path_from_local(model_dir: Path) -> Path:
    """Resolve the folds directory for a local model directory.

    Parameters
    ----------
    model_dir : Path
        The local model directory, holding either ``folds/FOLD0X`` or ``FOLD0X`` directly.

    Returns
    -------
    Path
        ``<model_dir>/folds`` when that subdirectory exists, otherwise ``model_dir``.
    """
    folds_dir = model_dir.joinpath('folds')
    folds_path = folds_dir if folds_dir.is_dir() else model_dir
    logger.info('Using local inference model at %s', folds_path)

    return folds_path


def _get_model_path_from_s3(one: ONE, model_name: str) -> Path | None:
    """Download a named S3 model and return its folds directory.

    Parameters
    ----------
    one : ONE
        ONE connection used for the download.
    model_name : str
        S3 model name (possibly nested, e.g. ``xgboost_channels/<vintage>``).

    Returns
    -------
    Path or None
        The downloaded model's folds directory, or None when no ONE/Alyx connection is available.
    """
    import ephysatlas.regionclassifier  # noqa: PLC0415

    # download_model downloads aggregates/atlas/models/<model_name> into cache_root/<model_name>
    # and returns that path (handles nested names without re-nesting).
    cache_root = s3_cache_root(one)
    cache_root.mkdir(parents=True, exist_ok=True)
    model_path = ephysatlas.regionclassifier.download_model(cache_root, model_name, one)
    folds_path = model_path.joinpath('folds')
    logger.info('Using S3 inference model %s at %s', model_name, folds_path)

    return folds_path


# -----------------------------------------------------------------------------
# Model validation
# -----------------------------------------------------------------------------

def validate_model(
    folds_path: str | Path, max_folds: int = MAX_FOLDS
) -> tuple[list[str], list[int], int]:
    """Validate a fold-based model directory and extract its feature/class contract.

    Reads each ``FOLD0X/meta.yaml`` under ``folds_path`` and asserts every fold agrees on its
    ``FEATURES`` and ``CLASSES`` lists. This is what makes averaging across folds (and switching
    between models) safe — the GUI relies on a single, consistent FEATURES/CLASSES ordering.

    Parameters
    ----------
    folds_path : str or Path
        Directory containing ``FOLD0X`` subdirectories, each with a ``meta.yaml``.
    max_folds : int
        Upper bound on the number of folds to probe (folds are discovered, not assumed).

    Returns
    -------
    tuple of (list of str, list of int, int)
        The shared FEATURES list, the shared CLASSES list (Cosmos region ids), and the number of
        folds discovered.

    Raises
    ------
    FileNotFoundError
        If no ``FOLD0X/meta.yaml`` is found under ``folds_path``.
    ValueError
        If the folds disagree on FEATURES or CLASSES.
    """
    folds_path = Path(folds_path)
    features: list[str] | None = None
    classes: list[int] | None = None
    n_folds = 0

    for fold in range(max_folds):
        meta_path = folds_path.joinpath(f'FOLD0{fold}', 'meta.yaml')
        if not meta_path.is_file():
            break
        with meta_path.open() as fh:
            meta = yaml.safe_load(fh)
        fold_features = list(meta['FEATURES'])
        fold_classes = [int(c) for c in meta['CLASSES']]
        if features is None:
            features, classes = fold_features, fold_classes
        else:
            if fold_features != features:
                raise ValueError(f'FEATURES differ between FOLD00 and FOLD0{fold}.')
            if fold_classes != classes:
                raise ValueError(f'CLASSES differ between FOLD00 and FOLD0{fold}.')
        n_folds += 1

    if n_folds == 0 or features is None or classes is None:
        raise FileNotFoundError(f'No FOLD0X/meta.yaml found under {folds_path}.')

    logger.info(
        'Validated %d folds at %s (%d features, %d classes).',
        n_folds,
        folds_path,
        len(features),
        len(classes),
    )
    return features, classes, n_folds


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


def validate_model_folder(model_dir: Path) -> bool:
    """Check the model directory has the expected fold structure.

    Parameters
    ----------
    model_dir : Path
        The local model directory to validate.

    Returns
    -------
    bool
        True if ``model_dir`` contains ``folds/FOLD00`` or ``FOLD00`` directly, else False.
    """
    return (
        model_dir.joinpath('folds', 'FOLD00').is_dir()
        or model_dir.joinpath('FOLD00').is_dir()
    )


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
        feats = FeatureLoaderLocal(features_path).load_features()
        items.model.raw_data['features'] = feats
        if not feats.get('exists', False):
            return None
    return feats['df']


# -----------------------------------------------------------------------------
# Model prediction
# -----------------------------------------------------------------------------

def _fold_mean_probas(
    controller: AlignmentGUIController, items: ShankController
) -> tuple[np.ndarray, np.ndarray, InferenceModel] | None:
    """Run the region classifier and average its per-fold probabilities.

    Shared core of :func:`predict` and :func:`predict_cumulative`: loads the shank's features,
    validates them against the model contract, runs ``infer_regions`` and averages the per-fold
    probabilities.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, InferenceModel) or None
        The fold-averaged class probabilities (``n_channels`` x ``n_classes``), the channel
        depths (µm) and the loaded model; or None when no features are available.
    """
    import ephysatlas.regionclassifier  # noqa: PLC0415 - lazy import keeps offline startup safe

    # Resolve the model first so its load dialog (and the "load a features file first" offline
    # guard inside it) runs before we bail on missing features, matching the spatial encoder path.
    model = get_model(controller)
    if model is None:
        return None

    df = _get_features_df(controller, items)
    if df is None:
        # Online the model loads without a features file (no offline guard runs), so warn here
        # when the insertion has no features available either on the plugin or in the loaded data.
        has_one, _ = has_one_connection(controller)
        if has_one:
            QtWidgets.QMessageBox.warning(
                controller.view, 'Channel Prediction',
                'No features found for this insertion.',
            )
        return None

    df = validate_features(df, model.features)
    predicted_probas, _ = ephysatlas.regionclassifier.infer_regions(
        df, path_model=model.model_path, n_folds=model.n_folds,
    )
    mean_probas = np.mean(predicted_probas, axis=0)
    depths = df['axial_um'].values

    return mean_probas, depths, model


def predict(
    controller: AlignmentGUIController, items: ShankController
) -> tuple[np.ndarray, np.ndarray] | None:
    """Predict the per-channel region (argmax over fold-averaged probabilities).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or None
        Predicted region ids and channel depths (µm), or None if no features are available.
    """
    result = _fold_mean_probas(controller, items)
    if result is None:
        return None

    mean_probas, depths, model = result
    region_ids = np.array(model.classes)[np.argmax(mean_probas, axis=1)]

    return region_ids, depths


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
        ``(cprobas, depths, colours, region_ids)`` or None if no features are available.
    """
    result = _fold_mean_probas(controller, items)
    if result is None:
        return None

    mean_probas, depths, model = result
    cprobas = mean_probas.cumsum(axis=1)
    region_ids = np.array(model.classes).astype(int)

    _, region_idxs = ismember(region_ids, controller.model.brain_atlas.regions.id)
    colours = [controller.model.brain_atlas.regions.rgb[idx] for idx in region_idxs]

    return cprobas, depths, colours, region_ids
