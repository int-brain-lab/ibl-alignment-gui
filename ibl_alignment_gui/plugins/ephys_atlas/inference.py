"""
Region-classifier inference for the channel-prediction plugin.

Loads a fold-based XGBoost region classifier — either from a local directory or downloaded from
S3 — and runs ``ephysatlas.regionclassifier.infer_regions`` to predict per-channel Cosmos regions
from an ephys-feature table. Heavy ``ephysatlas`` imports are deferred so this module can be
imported in offline mode without it installed; it is only required when inference actually runs.
"""

import logging
from pathlib import Path

import numpy as np
import yaml

from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal
from iblutil.numerical import ismember
from iblutil.util import Bunch
from one.api import ONE

logger = logging.getLogger(__name__)

# Default model downloaded from S3 when neither a local model dir nor an explicit S3 model name
# is configured (preserves the out-of-the-box S3 download from prediction_models_051526).
MODEL_VINTAGE = '2026_W12_Cosmos_careless-clover-dingo'
MODEL_NAME = 'Inference'  # key under which the loaded model is cached on the plugin
MAX_FOLDS = 10  # upper bound when discovering FOLD0X directories


def _plugin(controller):
    """Return the Channel Prediction plugin state Bunch."""
    return controller.plugins['Channel Prediction']


def ensure_model(controller):
    """
    Lazily load and cache the inference model, reloading when the source changes.

    The model is cached on the Channel Prediction plugin so the folds are loaded only once. The
    cache is keyed by the configured source (a local model directory or an S3 model name), so
    switching source through the plugin dialogs transparently triggers a reload.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    Bunch
        The loaded model Bunch (see :func:`load_inference_model`).
    """
    plugin = _plugin(controller)
    source = (plugin.get('local_model_dir'), plugin.get('model_name'))
    if plugin.get(MODEL_NAME) is None or plugin.get('_model_source') != source:
        plugin[MODEL_NAME] = load_inference_model(
            controller,
            model_dir=plugin.get('local_model_dir'),
            model_name=plugin.get('model_name'),
        )
        plugin['_model_source'] = source
    return plugin[MODEL_NAME]


def load_inference_model(controller, model_dir=None, model_name=None):
    """
    Load the region-classifier model from a local directory or download it from S3.

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

    Returns
    -------
    Bunch
        ``info`` (dict with ``FEATURES`` and ``CLASSES``), ``path`` (the folds directory passed
        to ``infer_regions``), and ``n_folds`` (number of folds from the validation contract).
    """
    if model_dir is not None:
        # Local model: point straight at the folds directory; no ONE / S3 access needed.
        model_dir = Path(model_dir)
        folds_dir = model_dir.joinpath('folds')
        folds_path = folds_dir if folds_dir.is_dir() else model_dir
        logger.info('Using local inference model at %s', folds_path)
    else:
        # S3 model: download the named model from S3 into the ONE cache.
        # noqa: PLC0415 below - lazy import keeps offline startup ephysatlas-free.
        import ephysatlas.regionclassifier  # noqa: PLC0415

        # Online backends (ProbeHandlerONE/CSV) expose a ready ONE on the model. Offline/YAML mode
        # does not, so fall back to a standalone ONE() — works when ONE/Alyx is configured and the
        # network is reachable; otherwise give an actionable message instead of an AttributeError.
        one = getattr(controller.model, 'one', None)
        if one is None:
            try:
                one = ONE()
            except Exception as exc:
                raise RuntimeError(
                    'S3 model download needs a configured ONE/Alyx connection, which is '
                    'unavailable in this offline/YAML session. Set a local model directory via '
                    'Channel Prediction → "Set local model dir…" instead.'
                ) from exc
        # Fall back to the default packaged model vintage when no S3 name is configured, so the
        # S3 download works out of the box (override via "Set S3 model name…").
        model_name = model_name or MODEL_VINTAGE
        cache_root = one.cache_dir.joinpath('ephys_atlas_features')
        cache_root.mkdir(parents=True, exist_ok=True)
        # download_model downloads aggregates/atlas/models/<model_name> into
        # cache_root/<model_name> and returns that path (handles nested names without re-nesting).
        model_path = ephysatlas.regionclassifier.download_model(cache_root, model_name, one)
        folds_path = model_path.joinpath('folds')
        logger.info('Using S3 inference model %s at %s', model_name, folds_path)

    # Validation contract: confirm all folds agree and extract FEATURES / CLASSES.
    features, classes, n_folds = validate_model(folds_path)

    return Bunch(info={'FEATURES': features, 'CLASSES': classes}, path=folds_path, n_folds=n_folds)


def validate_model(folds_path, max_folds=MAX_FOLDS):
    """
    Validate a fold-based model directory and extract its feature/class contract.

    Reads each ``FOLD0X/meta.yaml`` under ``folds_path`` and asserts every fold agrees on its
    ``FEATURES`` and ``CLASSES`` lists. This is what makes averaging across folds (and switching
    between models) safe — the GUI relies on a single, consistent FEATURES/CLASSES ordering.

    Parameters
    ----------
    folds_path : Path
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


def validate_features(df, features):
    """
    Ensure the features DataFrame has every column the model expects.

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


def _get_features_df(controller, items):
    """
    Return the per-channel features DataFrame for a shank, or None if unavailable.

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
        features_path = _plugin(controller).get('features_path')
        if features_path is None:
            return None
        feats = FeatureLoaderLocal(features_path).load_features()
        items.model.raw_data['features'] = feats
        if not feats.get('exists', False):
            return None
    return feats['df']


def predict(controller, items):
    """
    Predict the per-channel Cosmos region (argmax over fold-averaged probabilities).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or None
        Predicted Cosmos region ids and channel depths (µm), or None if no features are available.
    """
    import ephysatlas.regionclassifier  # noqa: PLC0415 - lazy import keeps offline startup safe

    df = _get_features_df(controller, items)
    if df is None:
        return None

    model = ensure_model(controller)
    df = validate_features(df, model['info']['FEATURES'])
    predicted_probas, _ = ephysatlas.regionclassifier.infer_regions(
        df, path_model=model['path'], n_folds=model['n_folds']
    )

    cosmos_ids = np.array(model['info']['CLASSES'])[
        np.argmax(np.mean(predicted_probas, axis=0), axis=1)
    ]
    depths = df['axial_um'].values

    return cosmos_ids, depths


def predict_cumulative(controller, items):
    """
    Return cumulative region probabilities by depth for the stacked-band view.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.

    Returns
    -------
    tuple or None
        ``(cprobas, depths, colours, region_ids)`` or None if no features are available.
    """
    import ephysatlas.regionclassifier  # noqa: PLC0415 - lazy import keeps offline startup safe

    df = _get_features_df(controller, items)
    if df is None:
        return None

    model = ensure_model(controller)
    df = validate_features(df, model['info']['FEATURES'])
    predicted_probas, _ = ephysatlas.regionclassifier.infer_regions(
        df, path_model=model['path'], n_folds=model['n_folds']
    )

    cprobas = np.mean(predicted_probas, axis=0).cumsum(axis=1)
    region_ids = np.array(model['info']['CLASSES']).astype(int)
    depths = df['axial_um'].values

    _, region_idxs = ismember(region_ids, controller.model.brain_atlas.regions.id)
    colours = [controller.model.brain_atlas.regions.rgb[idx] for idx in region_idxs]

    return cprobas, depths, colours, region_ids
