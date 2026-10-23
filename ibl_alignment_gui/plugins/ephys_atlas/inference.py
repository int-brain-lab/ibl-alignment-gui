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
    M_TO_UM,
    NO_FEATURES_MESSAGE,
    ModelSpec,
    PredictionError,
    clear_predictions,
    features_for_vintage,
    get_region_boundaries,
    missing_features_error,
    model_vintage,
)
from ibl_alignment_gui.utils.helpers import shank_loop
from iblutil.util import Bunch

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
    KeyError
        If the model raises one that is not about a missing feature.
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
    except KeyError as exc:
        missing = [c for c in model.inputs['features'] if c not in df.columns]
        if not missing:  # not a missing feature: an error in the model, for the safety net
            raise
        raise missing_features_error(missing, model.index.get('vintage')) from exc
    # The ProbeTransformer drops channels with non-finite features: take the depths of the rows
    # that were predicted.
    depths = df.loc[out.index, 'axial_um'].to_numpy()

    return out, depths


def _decode(
    controller: AlignmentGUIController,
    items: ShankController,
    model,
    progress: Callable[[float | None, str], None],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run a loaded decoder on the shank's features, without Qt.

    The features come from the table of the model's vintage (:func:`features_for_vintage`), then
    :func:`run_decoder` runs the model on them.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.
    model : RegionClassifier or ProbeTransformerClassifier
        The loaded decoder.
    progress : Callable
        ``progress(fraction, message)``, told of each step.

    Returns
    -------
    pandas.DataFrame
        The model's ``predict`` output.
    np.ndarray
        The depths (µm) of the predicted channels.

    Raises
    ------
    PredictionError
        If the model's release names no feature vintage, or the shank has no features, or they
        lack a column the model needs.
    """
    df = features_for_vintage(controller, items, model_vintage(model.index), progress)
    if df is None:
        raise PredictionError(NO_FEATURES_MESSAGE)
    progress(None, 'Predicting the regions…')

    return run_decoder(model, df)


def decode_shank(
    controller: AlignmentGUIController,
    items: ShankController,
    model,
    progress: Callable[[float | None, str], None],
    name: str,
) -> tuple[dict[str, Bunch], None]:
    """Run decoder ``name`` once on the shank and return both of its region-plot views.

    The decoder's ``compute`` (see :class:`ModelSpec`): no Qt is used, so it can run on a
    background thread.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank being predicted on.
    model : RegionClassifier or ProbeTransformerClassifier
        The loaded decoder.
    progress : Callable
        ``progress(fraction, message)``, told of each step.
    name : str
        The decoder, a key of :data:`DECODERS`, also its region-plot option.

    Returns
    -------
    dict of {str: Bunch}
        The most probable region per channel (:func:`predict`) under ``name`` and the cumulative
        class probabilities (:func:`predict_cumulative`) under ``name`` + ``' (cumulative)'``.
    None
        Nothing for ``on_computed``.

    Raises
    ------
    PredictionError
        If the shank has no features, or they lack a column the model needs.
    """
    out, depths = _decode(controller, items, model, progress)
    views = {
        name: predict(out, depths),
        name + CUMULATIVE_SUFFIX: predict_cumulative(out, depths, model),
    }

    return views, None


def predict(out: pd.DataFrame, depths: np.ndarray) -> Bunch:
    """Return the view of the most probable region per channel.

    Parameters
    ----------
    out : pandas.DataFrame
        A decoder's ``predict`` output, with ``predicted_atlas_id``.
    depths : np.ndarray
        The depths (µm) of the predicted channels.

    Returns
    -------
    Bunch
        The region boundaries, labels and colours along the probe, for ``plot_histology``.
    """
    regions = classifier_regions().get(out['predicted_atlas_id'].to_numpy())

    return get_region_boundaries(regions, np.asarray(depths, dtype=float) / M_TO_UM)


def predict_cumulative(out: pd.DataFrame, depths: np.ndarray, model) -> Bunch:
    """Return the view of the cumulative class probabilities by depth, for the stacked plot.

    Parameters
    ----------
    out : pandas.DataFrame
        A decoder's ``predict`` output, with one ``p_<acronym>`` column per class, in the
        manifest's class order.
    depths : np.ndarray
        The depths (µm) of the predicted channels.
    model : RegionClassifier or ProbeTransformerClassifier
        The decoder, for its classes.

    Returns
    -------
    Bunch
        ``probability`` (cumulative, channels x classes), ``depths``, ``regions`` (the class
        region ids) and ``colours`` (one per class), for ``plot_histology_cumulative``.
    """
    # One p_<acronym> column per class, in the manifest's class order.
    cprobas = out.filter(regex=r'^p_').to_numpy().cumsum(axis=1)
    region_ids = np.array(model.config['classes']).astype(int)
    colours = list(classifier_regions().get(region_ids).rgb)

    return Bunch(depths=depths, regions=region_ids, colours=colours, probability=cprobas)


# -----------------------------------------------------------------------------
# Model specs
# -----------------------------------------------------------------------------


# The decoders, as the shared load dialog and model cache need them.
DECODER_SPECS = {
    name: ModelSpec(
        key=name,
        title=f'Load {name} model',
        # The release tag is pinned here until ibleatools publishes the decoders' vintage.
        default_source=functools.partial(lambda repo: (repo, MODEL_REVISION), decoder.repo),
        note=DECODER_NOTE,
        load=_load_decoder,
        invalidate=functools.partial(invalidate_predictions, names=(name,)),
        options=(name, name + CUMULATIVE_SUFFIX),
        compute=functools.partial(decode_shank, name=name),
        needs_torch=decoder.needs_torch,
        model_class=decoder.model_class,
    )
    for name, decoder in DECODERS.items()
}
