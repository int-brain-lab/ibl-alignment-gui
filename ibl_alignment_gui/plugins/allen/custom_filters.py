from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ibl_alignment_gui.loaders import plot_loader
from iblutil.util import Bunch

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController


PLUGIN_NAME = 'Custom Filters'


def setup(controller: 'AlignmentGUIController') -> None:
    """
    Example to show how to add custom unit filters to the GUI.

    The filters registered here appear as extra options in the 'Filter units' dropdown menu,
    alongside the built-in 'All', 'KS good', 'KS mua' and 'IBL good' filters. Selecting one keeps
    only the clusters (and their spikes) for which the filter's predicate is True.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = Bunch()
    controller.plugins[PLUGIN_NAME]['activated'] = True
    # Register the filters on each data load, before the Filter menu is populated.
    controller.plugins[PLUGIN_NAME]['load_data'] = add_filters


def add_filters(controller: 'AlignmentGUIController') -> None:
    """
    Register custom unit filters.

    Each entry maps a filter name (shown in the dropdown) to a predicate. The predicate receives
    the clusters ``metrics`` table (a pandas DataFrame, one row per cluster) and returns a boolean
    mask selecting the clusters to keep. If a metric is missing or the predicate raises, the GUI
    falls back to showing all units (see :meth:`PlotLoader.filter_units`).

    Add your own filters by extending the dictionary below.

    Parameters
    ----------
    _ : AlignmentGUIController
        The main application controller (unused).
    """
    plot_loader.CUSTOM_FILTERS['aind_qc'] = filter_aind_qc
    plot_loader.CUSTOM_FILTERS['unitrefine_sua'] = filter_unit_refine_sua
    plot_loader.CUSTOM_FILTERS['unitrefine_label'] = filter_unit_refine_neural

def filter_aind_qc(metrics: pd.DataFrame) -> np.ndarray:
    return metrics['default_qc'].values

def filter_unit_refine_sua(metrics: pd.DataFrame) -> np.ndarray:
    return metrics["unitrefine_label"] == "sua"

def filter_unit_refine_neural(metrics: pd.DataFrame) -> np.ndarray:
    return metrics["unitrefine_label"] != "noise"

