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
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from ibl_alignment_gui.plugins.ephys_atlas._common import (
    NO_FEATURES_MESSAGE,
    ModelSpec,
    PredictionError,
    clear_predictions,
    features_for_vintage,
    get_model,
)
from ibl_alignment_gui.utils.helpers import shank_loop

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

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
# Explanation shown in the decoders' load dialog.
DECODER_NOTE = (
    'The model is loaded with ephysatlas.load_pretrained, downloaded once from the Hugging Face '
    'Hub and cached (~/.cache/ephysatlas/models). A local release takes precedence over the Hub.'
)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def _load_decoder(source: tuple[str, str], progress: Callable[[float, str], None]):
    """Load a region decoder release.

    Parameters
    ----------
    source : tuple of (str, str)
        ``(repo_id_or_directory, revision)``: a Hugging Face repository and release tag, or a
        local release directory.
    progress : Callable
        ``progress(fraction, message)``, called as loading proceeds.

    Returns
    -------
    RegionClassifier or ProbeTransformerClassifier
        The model returned by ``ephysatlas.load_pretrained``.
    """
    from ephysatlas import load_pretrained  # noqa: PLC0415 - heavy, lazy

    repo, revision = source
    progress(0.1, f'Loading {repo}@{revision} (downloaded on first use)…')
    # A Hub repo id or a local release directory; the manifest picks the model class.
    model = load_pretrained(repo, revision=revision)
    logger.info(
        'Loaded region decoder %s %s (%d features, %d classes)',
        model.index.get('model_class'),
        model.index.get('vintage'),
        len(model.inputs['features']),
        len(model.config['classes']),
    )
    progress(1.0, 'Model ready')
    return model


@shank_loop
def invalidate_predictions(
    controller: AlignmentGUIController,
    items: ShankController,
    names: Iterable[str] = tuple(DECODERS),
    **kwargs,
) -> None:
    """Drop decoders' cached predictions on a shank so the next click recomputes.

    Decorated with :func:`shank_loop`, so a single call iterates over every shank/config; the
    ``shank`` and ``config`` keywords injected by the decorator are absorbed via ``**kwargs``.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank whose cached predictions are cleared.
    names : iterable of str
        The decoders whose predictions are cleared, keys of :data:`DECODERS`; all by default.
    """
    clear_predictions(items, *names, *(name + CUMULATIVE_SUFFIX for name in names))


# The decoders, as the shared load dialog and model cache need them.
DECODER_SPECS = {
    name: ModelSpec(
        key=name,
        title=f'Load {name} model',
        repo=decoder.repo,
        revision=MODEL_REVISION,
        note=DECODER_NOTE,
        load=_load_decoder,
        invalidate=functools.partial(invalidate_predictions, names=(name,)),
    )
    for name, decoder in DECODERS.items()
}


# -----------------------------------------------------------------------------
# Loading utils
# -----------------------------------------------------------------------------


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


def run_decoder(model, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Run a loaded region decoder on a features table.

    Fills a missing ``outside`` column (in-brain default) and gives a flat table the
    ``(pid, channel)`` index the decoders expect, as one probe. No GUI is involved, so failures
    are raised for the caller to report.

    Parameters
    ----------
    model : RegionClassifier or ProbeTransformerClassifier
        A decoder returned by ``ephysatlas.load_pretrained``.
    df : pandas.DataFrame
        The probe's features, with an ``axial_um`` column.

    Returns
    -------
    pandas.DataFrame
        The model's ``predict`` output: one row per predicted channel, with ``predicted_atlas_id``
        and a ``p_<acronym>`` column per class, in the manifest's class order.
    np.ndarray
        The depths (µm) of the predicted channels.

    Raises
    ------
    PredictionError
        If ``df`` lacks a feature the model needs.
    """
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
        vintage = model.index.get('vintage')
        raise PredictionError(
            f'{exc}\n\nThe model ({vintage}) needs features computed with the {vintage} '
            'feature set.'
        ) from exc
    # The ProbeTransformer drops channels with non-finite features: take the depths of the rows
    # that were predicted.
    depths = df.loc[out.index, 'axial_um'].to_numpy()

    return out, depths


def _predict_df(
    controller: AlignmentGUIController, items: ShankController, name: str
) -> tuple[pd.DataFrame, np.ndarray, object] | None:
    """Run decoder ``name`` on the shank's features and return its prediction table.

    Shared core of :func:`predict` and :func:`predict_cumulative`: loads the model (asking the
    user when none is loaded yet) and the shank's features from the table of the model's vintage,
    then runs :func:`run_decoder`.

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
        The model's ``predict`` output, the depths (µm) of the predicted channels and the loaded
        model; or None when the user cancelled loading the model.

    Raises
    ------
    PredictionError
        If the shank has no features, or they lack a column the model needs.
    """
    # Resolve the model first so its load dialog (and its "load a features file first" guard)
    # runs before we bail on missing features, matching the spatial encoder path.
    model = get_model(controller, DECODER_SPECS[name])
    if model is None:
        return None

    df = features_for_vintage(controller, items, model.index.get('vintage'))
    if df is None:
        raise PredictionError(NO_FEATURES_MESSAGE)
    out, depths = run_decoder(model, df)

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
        if the user cancelled loading the model.

    Raises
    ------
    PredictionError
        If the shank has no features, or they lack a column the model needs.
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
        ``(cprobas, depths, colours, region_ids)``, or None if the user cancelled loading the
        model.

    Raises
    ------
    PredictionError
        If the shank has no features, or they lack a column the model needs.
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
