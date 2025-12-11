from ibl_alignment_gui.loaders.plot_loader import ScatterData, LineData, ImageData, ProbeData, skip_missing
from ibl_alignment_gui.utils.utils import shank_loop
from iblutil.util import Bunch
import numpy as np
from typing import Any
from types import MethodType

PLUGIN_NAME = 'C4: Cerebellar Cell Types'


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
    from pathlib import Path
    import pandas as pd
    df = []
    for tsv in Path('/datadisk/Data/paper-ephys-atlas/c4/869bac48-ad1b-46b8-be00-cf6b26aea40e').glob('*.tsv'):
        df.append(pd.read_csv(tsv, sep='\t'))
    df_c4 = pd.concat(df, axis=0).groupby('cluster_id').first()
    df_c4['predicted'], categories = pd.factorize(df_c4['predicted_cell_type'])
    items.model.raw_data['clusters']['c4'] = df_c4
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
    levels = np.array([0, 4])
    cid = self.cluster_idx
    scatter = ScatterData(
        x=self.avg_amp[cid],
        y=self.avg_depth[cid],
        levels=levels,
        default_levels=np.copy(levels),
        colours=self.data['clusters']['c4'].loc[cid, 'predicted'].values,
        pen='k',
        size=np.array(8),
        symbol=np.array('o'),
        xrange=np.array([0.9 * np.nanmin(self.avg_amp[cid]),
                         1.1 * np.nanmax(self.avg_amp[cid])]),
        xaxis='Amplitude (uV)',
        title='Prediction',
        cmap='tab20b',
        cluster=True
    )

    return {'Cluster Amp vs Depth vs Prediction': scatter}
