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
from typing import TYPE_CHECKING

from qtpy import QtCore, QtWidgets

from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal, FeatureLoaderOne

if TYPE_CHECKING:
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
