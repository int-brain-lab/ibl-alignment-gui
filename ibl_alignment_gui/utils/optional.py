"""Helpers for importing optional dependencies.

Some features of the GUI (raw-data streaming, alignment upload/QC and passive/RF
plots) rely on ``ibllib``. It is an optional dependency installed via the ``ibl``
extra so that the offline mode can run without it. The helpers here import those
optional modules lazily and raise a consistent, actionable error when they are
missing.
"""

import importlib
import importlib.util
from types import ModuleType

IBL_EXTRA_HINT = (
    'This feature requires the IBL online dependencies. Install them with '
    "'pip install ibl_alignment_gui[ibl]'."
)


def has_ibllib() -> bool:
    """Return whether ``ibllib`` is importable.

    Returns
    -------
    bool
        True if ``ibllib`` is installed, False otherwise.
    """
    return importlib.util.find_spec('ibllib') is not None


def require_ibllib(feature: str, module: str) -> ModuleType:
    """Import an ``ibllib``/``brainbox`` submodule or raise a helpful error.

    Parameters
    ----------
    feature : str
        Human-readable name of the feature requesting the import, used in the
        error message (e.g. ``'Alignment upload'``).
    module : str
        Dotted path of the module to import (e.g. ``'ibllib.pipes.histology'``).

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the module cannot be imported because ``ibllib`` is not installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(f'{feature} is unavailable: {IBL_EXTRA_HINT}') from err
