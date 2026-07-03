"""Helpers for querying optional dependencies.

Some features rely on optional dependency stacks installed via extras: ``ibllib`` (the ``ibl``
extra, for the online ONE/Alyx mode) and the ``aind``/``boto3`` stack (the ``allen`` extra, for
the Allen/Code Ocean DocDB mode). The helpers here report whether those stacks are installed
(without importing them) so the launchers can fail fast with an actionable message; the modules
that use them import them lazily at the point of use.
"""

import importlib.util

IBL_EXTRA_HINT = (
    'This feature requires the IBL online dependencies. Install them with '
    "'pip install ibl_alignment_gui[ibl]'."
)

ALLEN_EXTRA_HINT = (
    'This feature requires the Allen institute dependencies. Install them with '
    "'pip install ibl_alignment_gui[allen]'."
)


def has_ibllib() -> bool:
    """Return whether the IBL (ibllib) dependency stack is importable.

    Uses ``importlib.util.find_spec`` so the check does not import the heavy dependencies.

    Returns
    -------
    bool
        True if ``ibllib`` is installed, False otherwise.
    """
    return importlib.util.find_spec('ibllib') is not None


def has_allen() -> bool:
    """Return whether the Allen (DocDB) dependency stack is importable.

    Uses ``importlib.util.find_spec`` so the check does not import the heavy dependencies.

    Returns
    -------
    bool
        True if ``aind_data_access_api`` is installed, False otherwise.
    """
    return importlib.util.find_spec('aind_data_access_api') is not None
