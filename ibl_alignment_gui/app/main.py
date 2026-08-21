import os
import sys

# The spatial-encoder (torch) and inference (xgboost, via ephysatlas.regionclassifier) plugins
# each bundle their own OpenMP runtime. On macOS, having both loaded in one process leaves two
# independent libomp.dylib copies mapped in-process; as soon as either does a parallel OpenMP
# region (e.g. XGBoost's model loading), their internal thread-pool state collides and segfaults
# - confirmed via crash log + reproduction, independent of xgboost's own n_jobs setting. Forcing
# a single OpenMP thread avoids the thread-pool bring-up that triggers the corruption. Must be
# set before torch/xgboost are ever imported (they're lazily imported by the plugins later), so
# this needs to happen at process start, before those imports occur anywhere.
# Only macOS is affected; applying the limit elsewhere would needlessly single-thread the
# OpenMP-backed numpy/scipy work that the rest of the GUI depends on.
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse

from qtpy import QtWidgets

from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
from ibl_alignment_gui.utils.optional import (
    ALLEN_EXTRA_HINT,
    IBL_EXTRA_HINT,
    has_allen,
    has_ibllib,
)


def _require_extra(available: bool, hint: str) -> None:
    """Exit with an actionable message if a required optional dependency is missing.

    Parameters
    ----------
    available : bool
        Whether the required optional dependency is installed.
    hint : str
        The install instructions shown when the dependency is missing.

    Raises
    ------
    SystemExit
        If ``available`` is False, exiting the process with the hint as the message.
    """
    if not available:
        raise SystemExit(hint)


def launch_app() -> None:
    """Launch the alignment GUI application in offline mode with optional YAML file."""
    parser = argparse.ArgumentParser()

    parser.add_argument('-y', '--yaml', required=False, type=str, help='Path to the YAML file')

    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = AlignmentGUIController(offline=True, csv=None, yaml=args.yaml)
    mainapp.view.show()
    app.exec_()


def launch_app_ibl() -> None:
    """Launch the alignment GUI application in IBL mode.

    Optionally accepts a CSV file or a probe insertion id (pid) to auto-load. When a pid is
    given the subject, session and shank dropdowns are configured to that insertion and its
    data is loaded automatically.
    """
    # IBL online mode requires ibllib (raw-data streaming, alignment upload/QC).
    _require_extra(has_ibllib(), IBL_EXTRA_HINT)

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--csv', required=False, type=str, help='Path to the CSV file')
    parser.add_argument(
        '-p', '--pid', required=False, type=str, help='Probe insertion id to auto-load'
    )

    args = parser.parse_args()

    if args.csv is not None and args.pid is not None:
        parser.error('--pid cannot be used together with --csv')

    app = QtWidgets.QApplication([])
    try:
        mainapp = AlignmentGUIController(offline=False, csv=args.csv, yaml=None, pid=args.pid)
    except ValueError as err:
        parser.error(str(err))
    mainapp.view.show()
    app.exec_()


def launch_app_allen() -> None:
    """Launch the alignment GUI in Allen/Code Ocean mode with DocDB support.

    Runs offline from a session YAML (chosen from the source button, or passed with ``-y``) and
    adds a DocDB checkbox that toggles whether previous alignments are read from, and results
    written to, the Allen DocDB (ticked) or the local files (unticked).
    """
    # Allen mode requires the aind/DocDB dependency stack.
    _require_extra(has_allen(), ALLEN_EXTRA_HINT)

    parser = argparse.ArgumentParser()

    parser.add_argument('-y', '--yaml', required=False, type=str, help='Path to the YAML file')

    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = AlignmentGUIController(offline=True, csv=None, yaml=args.yaml, allen=True)
    mainapp.view.show()
    app.exec_()


if __name__ == '__main__':
    launch_app()
