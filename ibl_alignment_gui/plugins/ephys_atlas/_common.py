"""
Provider-agnostic helpers shared by the channel-prediction model backends.

Both :mod:`ibl_alignment_gui.plugins.ephys_atlas.inference` (XGBoost) and
:mod:`ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder` (torch) cache their state on the
Channel Prediction plugin, resolve a ONE/Alyx connection the same way, decide when to rebuild a
model from cached state, and clear per-shank prediction caches identically. Those pieces live here.

This module intentionally avoids the heavy/optional ``torch`` and ``ephysatlas`` imports so it can
be imported in offline mode alongside ``inference`` (which defers those imports itself).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from iblutil.util import Bunch
from one.api import ONE

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController

logger = logging.getLogger(__name__)

PLUGIN_KEY = 'Channel Prediction'


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
    if plugin_state(controller).get('features_path') is not None:
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


def has_one_connection(
    controller: AlignmentGUIController,
    base_url: str = 'https://alyx.internationalbrainlab.org',
) -> tuple[bool, ONE | None]:
    """Return whether a ONE/Alyx connection to ``base_url`` is available.

    Reuses the data backend's ONE when it already targets ``base_url``; otherwise tries to open a
    standalone connection.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    base_url : str
        Alyx database URL the connection must target.

    Returns
    -------
    tuple of (bool, ONE or None)
        ``(True, one)`` when a connection is available, otherwise ``(False, None)``.
    """
    one = getattr(controller.model, 'one', None)
    if one is None or one.alyx.base_url != base_url:
        try:
            one = ONE(base_url=base_url, silent=True)
        except Exception:
            return False, None
    return True, one


def s3_cache_root(one: ONE) -> Path:
    """Return the local cache directory used for S3 model/feature downloads.

    Parameters
    ----------
    one : ONE
        ONE connection whose cache directory roots the download tree.

    Returns
    -------
    Path
        The ``ephys_atlas_features`` directory under the ONE cache.
    """
    return Path(one.cache_dir).joinpath('ephys_atlas_features')


def needs_reload(current_model: dict, **expected: object) -> bool:
    """Return whether a cached model must be (re)built for a newly-chosen source.

    Parameters
    ----------
    current_model : dict
        The cached model state, holding a ``model`` entry plus the source keys to compare.
    expected
        The newly-selected source values keyed by their cache-state name (e.g.
        ``model_name='…'`` or ``local_inference_dir=Path(…)``).

    Returns
    -------
    bool
        True when no model is built yet or any ``expected`` value differs from the cached one.
    """
    if current_model['model'] is None:
        return True
    return any(current_model.get(key) != value for key, value in expected.items())


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
