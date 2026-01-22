from ibl_alignment_gui.loaders.plot_loader import ScatterData, LineData, ImageData, ProbeData, skip_missing
from ibl_alignment_gui.utils.utils import shank_loop
from iblutil.util import Bunch
import numpy as np
from typing import Any
from types import MethodType

PLUGIN_NAME = 'Additional Plots'


def setup(controller: 'AlignmentGUIController') -> None:
    """
    Example to show how to add additional plots to the GUI.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = Bunch()
    controller.plugins[PLUGIN_NAME]['activated'] = True
    # Attach callbacks to methods in the controller
    controller.plugins[PLUGIN_NAME]['load_data'] = add_plots


@shank_loop
def add_plots(_, items, **kwargs):
    # Add the additional data that may be required for the plots
    items.model.raw_data['clusters']['predicted_region'] = np.random.randint(
        0, 500, size=items.model.raw_data['clusters']['peakToTrough'].shape)
    items.model.loaders['plots'].scatter_amp_depth_prediction = (
        MethodType(scatter_amp_depth_prediction, items.model.loaders['plots']))


@skip_missing(['spikes'])
def scatter_amp_depth_prediction(self) -> dict[str, Any]:
    """
    Generate data for a scatter plot of cluster depth vs. cluster amplitude.

    Clusters are coloured by their predicted region.

    Returns
    -------
    Dict
        A dict containing a ScatterData object with key 'Cluster Amp vs Depth vs Duration'.
    """
    levels = np.array([0, 500])

    scatter = ScatterData(
        x=self.avg_amp[self.cluster_idx],
        y=self.avg_depth[self.cluster_idx],
        levels=levels,
        default_levels=np.copy(levels),
        colours=self.data['clusters']['predicted_region'][self.cluster_idx],
        pen='k',
        size=np.array(8),
        symbol=np.array('o'),
        xrange=np.array([0.9 * np.nanmin(self.avg_amp[self.cluster_idx]),
                         1.1 * np.nanmax(self.avg_amp[self.cluster_idx])]),
        xaxis='Amplitude (uV)',
        title='Prediction',
        cmap='Purples',
        cluster=True
    )

    return {'Cluster Amp vs Depth vs Prediction': scatter}


