from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.utils.utils import shank_loop
from iblutil.util import Bunch

from ibl_alignment_gui.plugins.auto_align import *

from ephysatlas.feature_computation import compute_features_from_pid
import time

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController

PLUGIN_NAME = 'Channel Prediction'


def setup(controller: 'AlignmentGUIController') -> None:
    controller.plugins[PLUGIN_NAME] = Bunch()
    controller.plugins[PLUGIN_NAME]['activated'] = False
    controller.plugins[PLUGIN_NAME]['engine'] = None  # cache slot

    channel_prediction = ChannelPrediction(controller)
    controller.plugins[PLUGIN_NAME]['loader'] = channel_prediction

    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, controller.view)
    controller.plugin_options.addMenu(plugin_menu)

    action_group = QtWidgets.QActionGroup(plugin_menu)
    action_group.setExclusive(True)

    predictions_models = {
        'Original': None,
        'Cosmos': compute_cosmos_predictions,
        'Random': compute_random_predictions,
        'ephys_based': compute_ephys_based_predictions,
    }

    for model, model_func in predictions_models.items():
        action = QtWidgets.QAction(model, controller.view)
        action.setCheckable(True)
        action.setChecked(model == 'Original')
        action.triggered.connect(lambda _, m=model, func=model_func:
                                 channel_prediction.plot_regions(_, m, func))
        action_group.addAction(action)
        plugin_menu.addAction(action)


class ChannelPrediction:
    """
    Class to handle channel prediction plotting in the alignment GUI.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """

    def __init__(self, controller: 'AlignmentGUIController') -> None:
        self.controller = controller
        self.ba: AllenAtlas = self.controller.model.brain_atlas

    def plot_regions(self, _, model: str, func: Callable) -> None:
        """
        Plot the brain regions based on the selected model.

        Parameters
        ----------
        model: str
            The name of the model to use for predictions.
        func: Callable
            The function to compute the predictions.
        """
        # Plot the regions based on the action
        if model == 'Original':
            plot_original_regions(self.controller)
        else:
            plot_predicted_regions(self.controller, model, func)


@shank_loop
def plot_original_regions(_, items: 'ShankController', **kwargs) -> None:
    """Plot the original histology regions on the reference histology plot."""
    items.view.plot_histology(items.view.fig_hist_ref, items.model.hist_data_ref, ax='right')


@shank_loop
def plot_predicted_regions(
        controller: 'AlignmentGUIController',
        items: 'ShankController',
        model: str,
        func: Callable,
        **kwargs
) -> None:
    """
    Plot the model predictions on the reference histology plot.

    Parameters
    ----------
    model: str
        The name of the model.
    func: Callable
        The function to compute the predictions.
    """
    if not getattr(items.model, 'predictions', None):
        items.model.predictions = Bunch()

    results = items.model.predictions.get(model, None)
    if results is None:
        items.model.predictions[model] = func(controller, items)

    items.view.plot_histology(items.view.fig_hist_ref, items.model.predictions[model], ax='right')


def compute_cosmos_predictions(
        controller: 'AlignmentGUIController',
        items: 'ShankController'
) -> Bunch[str, np.ndarray]:
    """
    Example prediction model that returns cosmos brain regions.

    Returns
    -------
    Bunch
        A bunch containing the predicted brain regions.
    """
    # xyz coordinates sampled at 10 um along histology track from bottom or brain to top
    xyz_samples = items.model.align_handle.xyz_samples
    # depths of these coordinates along the track
    depth_samples = items.model.align_handle.ephysalign.sampling_trk

    region_ids = controller.model.brain_atlas.get_labels(xyz_samples, mapping='Cosmos')
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depth_samples)


def compute_random_predictions(
        controller: 'AlignmentGUIController',
        items: 'ShankController'
) -> Bunch[str, np.ndarray]:
    """
    Example prediction model that uses the spikes data to assign random brain regions.

    Returns
    -------
    Bunch
        A bunch containing the predicted brain regions.
    """
    # xyz coordinates sampled at 10 um along histology track from bottom or brain to top
    xyz_samples = items.model.align_handle.xyz_samples
    # depths of these coordinates along the track
    depth_samples = items.model.align_handle.ephysalign.sampling_trk

    # Spikes and other data can be accessed in this way if needed
    spikes = items.model.raw_data['spikes']

    def random_chunked_array(n, n_vals=20, seed=None):
        rng = np.random.default_rng(seed)
        cuts = np.sort(rng.choice(np.arange(1, n), size=n_vals - 1, replace=False))
        chunks = np.diff(np.r_[0, cuts, n])
        vals = rng.choice(np.arange(1001), size=n_vals, replace=False)
        chosen = rng.choice(vals, size=len(chunks), replace=True)
        return np.repeat(chosen, chunks)

    random = random_chunked_array(len(depth_samples), n_vals=20, seed=42)
    region_ids = controller.model.brain_atlas.regions.id[random]
    regions = controller.model.brain_atlas.regions.get(region_ids)
    return get_region_boundaries(regions, depth_samples)

def compute_ephys_based_predictions(
    controller: 'AlignmentGUIController',
    items: 'ShankController'
) -> Bunch[str, np.ndarray]:

    # -------------- one-time caches (lazy) --------------
    engine = ensure_engine(controller)
    device = engine.device
    ctx_manager = engine.ctx_manager
    model = engine.model
    handles = engine.handles
    e_mean, e_std = engine.e_mean, engine.e_std
    optimization_features = engine.optimization_features
    M_MAX = engine.M_MAX
    RADIUS_UM = engine.RADIUS_UM
    pid_str = engine.pid_str
    ephys = engine.ephys

    brain_atlas = AllenAtlas()

    # -------------- prediction pipeline (fast) --------------
    print("Ephys feature computation")
    t0 = time.time()
    # TODO: Currently flipping the probe since my model is trained on probes that goes from top to bottom,
    #  need to update the model and change that
    xyz_samples = items.model.align_handle.xyz_samples[::-1, :]
    pid = controller.model.shank_labels[0]['id']
    p_ind = np.where(pid == pid_str)[0]

    if len(p_ind) <= 0:
        print("Probe has no precomputed ephys features.")
        region_ids = xyz_to_region_ids(xyz_samples, brain_atlas)
        depth_samples = items.model.align_handle.ephysalign.sampling_trk
        regions = controller.model.brain_atlas.regions.get(region_ids)
        return get_region_boundaries(regions, depth_samples)

    recorded_ephys_probe = ephys[p_ind][0]

    print(f"Time: {time.time() - t0:.2f}s")

    C = recorded_ephys_probe.shape[0]

    print("Ephys feature prediction along the probe trace")
    t0 = time.time()
    mu_std_full, logvar_full = predict_features_at_xyz_v2(
        model, ctx_manager, handles, e_mean.shape[-1],
        xyz_samples, batch_size=512, radius_um=RADIUS_UM, M_max=M_MAX, device=device
    )
    print(f"Time: {time.time() - t0:.2f}s")

    print("Alignment pipeline")
    t0 = time.time()
    mu_std_full_np = mu_std_full.cpu().numpy().astype(np.float64)
    B_std = mu_std_full_np[:, optimization_features]

    Y_rec = recorded_ephys_probe.astype(np.float32)
    ch_mask = ~(np.all(Y_rec == 0.0, axis=1))
    rec_idx = np.where(ch_mask)[0]
    assert rec_idx.size >= 2, "Need at least 2 recorded channels."

    Y_std = ((torch.from_numpy(Y_rec) - e_mean.cpu()) / (e_std.cpu() + 1e-8)).numpy().astype(np.float64)
    A_std = Y_std[rec_idx][:, optimization_features]

    Cmat = build_cost(A_std, B_std, logvar_full)

    j_start, j_end, path, total_cost = sdtw(Cmat, lam_d=0.0, lam_u=5.0, lam_l=5.0)

    min_overlap_channels = int(0.9*C)
    if (j_end - j_start + 1) < min_overlap_channels:
        best_k, best_mse = 0, np.inf
        Nr = A_std.shape[0]
        for k in range(0, B_std.shape[0] - Nr + 1):
            m = ((B_std[k:k + Nr] - A_std) ** 2).mean()
            if m < best_mse:
                best_mse, best_k = m, k
        j_start, j_end = best_k, best_k + Nr - 1
        path = [(i, best_k + i) for i in range(Nr)]

    i_seq = np.array([ij[0] for ij in path], dtype=int)
    j_seq = np.array([ij[1] for ij in path], dtype=int)
    j_for_i = np.full((A_std.shape[0],), np.nan)
    for ii, jj in zip(i_seq, j_seq):
        j_for_i[ii] = jj
    for ii in range(1, j_for_i.shape[0]):
        if np.isnan(j_for_i[ii]):
            j_for_i[ii] = j_for_i[ii - 1]
    for ii in range(j_for_i.shape[0] - 2, -1, -1):
        if np.isnan(j_for_i[ii]):
            j_for_i[ii] = j_for_i[ii + 1]
    j_for_i = np.clip(j_for_i.astype(int), 0, B_std.shape[0] - 1)

    j_map_all = np.interp(np.arange(C), rec_idx, j_for_i.astype(float))
    j_map_all_i = np.clip(np.round(j_map_all).astype(int), 0, B_std.shape[0] - 1)
    est_xyz = xyz_samples[j_map_all_i]

    # TODO: Currently flipping the probe since my model is trained on probes that goes from top to bottom,
    #  need to update the model and change that
    trk = items.model.align_handle.ephysalign.sampling_trk.copy()[::-1]
    depth_samples = (trk - trk[j_end])[::-1]

    # TODO: Flipping the probe back for consistency
    region_ids = xyz_to_region_ids(xyz_samples, brain_atlas)[::-1]

    print(f"Time: {time.time() - t0:.2f}s")

    regions = controller.model.brain_atlas.regions.get(region_ids)
    return get_region_boundaries(regions, depth_samples)


def get_region_boundaries(regions: dict, depths: np.ndarray) -> Bunch[str, np.ndarray]:
    """
    Get the boundaries of brain regions along the histology track.

    Parameters
    ----------
    regions: dict
        The brain regions along the histology track.
    depths:
        The depths along the histology track.

    Returns
    -------
    Bunch
        A bunch containing the region boundaries, labels, and colours.
    """
    boundaries = np.where(np.diff(regions.id))[0]

    n_regions = len(boundaries) + 1
    region = np.empty((n_regions, 2))
    region_label = np.empty((n_regions, 2), dtype=object)
    region_colour = np.empty((n_regions, 3), dtype=int)

    for i in range(n_regions):
        # Compute start and end indices for this region
        start = 0 if i == 0 else boundaries[i - 1] + 1
        end = boundaries[i] if i < len(boundaries) else regions.id.size - 1

        region[i, :] = depths[[start, end]] * 1e6
        region_label[i, :] = (np.mean(depths[[start, end]]) * 1e6, regions.acronym[end])
        region_colour[i, :] = regions.rgb[end]

    data = Bunch(
        region=region,
        axis_label=region_label,
        colour=region_colour
    )

    return data
