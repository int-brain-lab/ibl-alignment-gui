"""Helpers for plugin tests that use modules depending on optional packages."""

import functools
import importlib
import sys
import types
from unittest.mock import patch

SPATIAL_ENCODER = 'ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder'


@functools.cache
def import_spatial_encoder() -> types.ModuleType:
    """Import ``spatial_encoder``, with stand-ins for ``ephysatlas`` when it is not installed.

    The stand-ins hold only the names ``spatial_encoder`` imports, and are in ``sys.modules`` only
    while it is imported. Tests replace whatever they use from them (e.g. ``ChannelModel``).

    The module is imported once and the same object is returned on every call, so that all test
    files patch the module the plugin uses (importing it again would make a new module object and
    leave the package attribute pointing to whichever was imported last).

    Returns
    -------
    types.ModuleType
        The ``spatial_encoder`` module.
    """
    try:
        return importlib.import_module(SPATIAL_ENCODER)
    except ImportError:
        pass
    alignment = types.ModuleType('ephysatlas.alignment')
    alignment.CHANNEL_MODEL_REPO_ID = 'int-brain-lab/ea-encoder-channel'
    alignment.VINTAGE = '2026_W39'
    alignment.ChannelModel = object
    alignment.align_histology = None
    data = types.ModuleType('ephysatlas.alignment.data')
    data.load_probe = None
    utils = types.ModuleType('ephysatlas.spatial_encoder.utils')
    utils.FEATURE_LIST = []
    stubs = {
        'ephysatlas': types.ModuleType('ephysatlas'),
        'ephysatlas.alignment': alignment,
        'ephysatlas.alignment.data': data,
        'ephysatlas.spatial_encoder': types.ModuleType('ephysatlas.spatial_encoder'),
        'ephysatlas.spatial_encoder.utils': utils,
    }
    with patch.dict(sys.modules, stubs):
        return importlib.import_module(SPATIAL_ENCODER)
