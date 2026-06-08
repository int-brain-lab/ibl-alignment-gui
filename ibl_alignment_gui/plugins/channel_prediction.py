import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal
from ibl_alignment_gui.utils.utils import shank_loop
from iblutil.util import Bunch
from iblatlas.atlas import AllenAtlas

# NB: ``spatial_encoder`` (torch) and ``inference`` (ephysatlas) are imported lazily inside the
# compute functions below so this plugin can be set up in offline mode without those heavy/optional
# dependencies installed.

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController

PLUGIN_NAME = 'Channel Prediction'


def setup(controller: 'AlignmentGUIController') -> None:
    controller.plugins[PLUGIN_NAME] = Bunch()
    controller.plugins[PLUGIN_NAME]['activated'] = True
    # Source configuration for the inference model + local features, set via the dialogs below and
    # consumed by ephys_atlas.inference.ensure_model / _get_features_df.
    controller.plugins[PLUGIN_NAME]['local_model_dir'] = None
    controller.plugins[PLUGIN_NAME]['features_path'] = None
    controller.plugins[PLUGIN_NAME]['model_name'] = None
    # Spatial Encoder (automatic alignment) sources: a local encoder model dir and the
    # reference-bank root, consumed by ephys_atlas.spatial_encoder.load_alignment_engine.
    controller.plugins[PLUGIN_NAME]['local_encoder_dir'] = None
    controller.plugins[PLUGIN_NAME]['local_encoder_data'] = None

    channel_prediction = ChannelPrediction(controller)
    controller.plugins[PLUGIN_NAME]['loader'] = channel_prediction

    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, controller.view)
    controller.plugin_options.addMenu(plugin_menu)

    action_group = QtWidgets.QActionGroup(plugin_menu)
    action_group.setExclusive(True)

    # All models are offline-capable from local assets: the inference (xgboost) model via a local
    # model dir, and the Spatial Encoder (automatic alignment) via a local encoder dir + bank dir
    # (set through the dialogs below). They fall back to S3/ONE only when no local source is set.
    predictions_models = {
        'Original': None,
        'Cosmos': compute_cosmos_predictions,
        'Spatial Encoder': compute_spatial_encoder_predictions,
        'Inference Model': compute_inference_predictions,
        'Inference Cumulative': compute_cumulative_predictions,
    }

    for model, model_func in predictions_models.items():
        action = QtWidgets.QAction(model, controller.view)
        action.setCheckable(True)
        action.setChecked(model == 'Original')
        action.triggered.connect(
            lambda _, m=model, func=model_func: channel_prediction.plot_regions(_, m, func)
        )
        action_group.addAction(action)
        plugin_menu.addAction(action)

    # Dialogs to point the inference model + features at local paths or an S3 model name, and the
    # Spatial Encoder (automatic alignment) at a local encoder dir + reference-bank dir.
    plugin_menu.addSeparator()
    for label, handler in (
        ('Set local features file…', _set_local_features),
        ('Set local model dir…', _set_local_model_dir),
        ('Set S3 model name…', _set_s3_model_name),
        ('Set local Spatial Encoder dir…', _set_local_encoder_dir),
        ('Set Spatial Encoder bank dir…', _set_local_encoder_data),
    ):
        action = QtWidgets.QAction(label, controller.view)
        action.triggered.connect(lambda _=False, h=handler: h(controller))
        plugin_menu.addAction(action)

    controller.plugins[PLUGIN_NAME]['data_button_pressed'] = lambda: callback(action_group)
    # Kept so the 'Set …' dialogs can re-trigger the currently-selected prediction after they
    # change the model/features source (see _refresh_current_prediction).
    controller.plugins[PLUGIN_NAME]['action_group'] = action_group


def _refresh_current_prediction(controller: 'AlignmentGUIController') -> None:
    """
    Re-run the currently-selected prediction so the visible plot reflects a new source.

    Called after a ``Set …`` dialog changes the model/features source and clears the prediction
    cache. Re-firing the checked action recomputes + redraws the active plot (for the inference
    models) or simply replots the current view (Original/Cosmos).
    """
    group = controller.plugins[PLUGIN_NAME].get('action_group')
    action = group.checkedAction() if group is not None else None
    if action is not None:
        action.trigger()


def _invalidate_predictions(controller: 'AlignmentGUIController') -> None:
    """Drop cached inference predictions on every shank so the next click recomputes."""
    for shank_dict in controller.model.shanks.values():
        for shank_handler in shank_dict.values():
            preds = getattr(shank_handler, 'predictions', None)
            if preds:
                for key in ('Inference Model', 'Inference Cumulative'):
                    preds.pop(key, None)


def _set_local_features(controller: 'AlignmentGUIController') -> None:
    """Prompt for a per-channel features parquet and use it for inference."""
    parent = controller.view
    chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent, 'Select per-channel features file', filter='Parquet (*.pqt *.parquet)')
    if not chosen:
        return

    path = Path(chosen)
    feats = FeatureLoaderLocal(path).load_features()
    if not feats.get('exists', False):
        QtWidgets.QMessageBox.warning(parent, PLUGIN_NAME, f'No features found in:\n{path}')
        return

    controller.plugins[PLUGIN_NAME]['features_path'] = path
    # Inject into any already-loaded shanks so inference (and re-runs) use the new file; if data
    # is not loaded yet, ephys_atlas.inference._get_features_df will load it lazily from this path.
    for shank_dict in controller.model.shanks.values():
        for shank_handler in shank_dict.values():
            if getattr(shank_handler, 'raw_data', None) is not None:
                shank_handler.raw_data['features'] = feats
    _invalidate_predictions(controller)
    logger.info('Local features file set to %s', path)
    _refresh_current_prediction(controller)


def _set_local_model_dir(controller: 'AlignmentGUIController') -> None:
    """Prompt for a local model directory (containing folds/FOLD00/)."""
    parent = controller.view
    chosen = QtWidgets.QFileDialog.getExistingDirectory(
        parent, 'Select model directory (containing folds/FOLD00/)')
    if not chosen:
        return

    model_dir = Path(chosen)
    # Accept either <dir>/folds/FOLD00 or <dir>/FOLD00 (the dir already being the folds directory).
    has_folds = model_dir.joinpath('folds', 'FOLD00').is_dir() or model_dir.joinpath('FOLD00').is_dir()
    if not has_folds:
        QtWidgets.QMessageBox.warning(
            parent, PLUGIN_NAME, f'No "folds/FOLD00" (or "FOLD00") found under:\n{model_dir}')
        return

    controller.plugins[PLUGIN_NAME]['local_model_dir'] = model_dir
    controller.plugins[PLUGIN_NAME]['model_name'] = None  # local model takes precedence over S3
    _invalidate_predictions(controller)
    logger.info('Local model dir set to %s', model_dir)
    _refresh_current_prediction(controller)


def _set_s3_model_name(controller: 'AlignmentGUIController') -> None:
    """Prompt for an S3 model name to download (e.g. xgboost_channels/2026_W12_Cosmos_...)."""
    parent = controller.view
    current = controller.plugins[PLUGIN_NAME].get('model_name') or ''
    name, ok = QtWidgets.QInputDialog.getText(
        parent, PLUGIN_NAME,
        'S3 model name (e.g. xgboost_channels/2026_W12_Cosmos_careless-clover-dingo):',
        text=current)
    if not ok:
        return

    controller.plugins[PLUGIN_NAME]['model_name'] = name or None
    controller.plugins[PLUGIN_NAME]['local_model_dir'] = None  # S3 takes precedence over local dir
    _invalidate_predictions(controller)
    logger.info('S3 model name set to %s', name or '<unset>')
    _refresh_current_prediction(controller)


def _invalidate_engine(controller: 'AlignmentGUIController') -> None:
    """Drop the cached Spatial Encoder engine so the next click rebuilds it."""
    controller.plugins[PLUGIN_NAME].pop('Spatial encoder', None)


def _set_local_encoder_dir(controller: 'AlignmentGUIController') -> None:
    """Prompt for a local Spatial Encoder model dir (SE_model_*.pt + *_vol_pca.npy)."""
    parent = controller.view
    chosen = QtWidgets.QFileDialog.getExistingDirectory(
        parent, 'Select Spatial Encoder model dir (SE_model_*.pt + *_vol_pca.npy)')
    if not chosen:
        return

    enc_dir = Path(chosen)
    if not any(enc_dir.glob('SE_model_*.pt')):
        QtWidgets.QMessageBox.warning(
            parent, PLUGIN_NAME, f'No "SE_model_*.pt" found under:\n{enc_dir}')
        return

    controller.plugins[PLUGIN_NAME]['local_encoder_dir'] = enc_dir
    _invalidate_engine(controller)
    logger.info('Local Spatial Encoder dir set to %s', enc_dir)
    _refresh_current_prediction(controller)


def _set_local_encoder_data(controller: 'AlignmentGUIController') -> None:
    """Prompt for the Spatial Encoder reference-bank root."""
    parent = controller.view
    chosen = QtWidgets.QFileDialog.getExistingDirectory(
        parent, 'Select Spatial Encoder bank root (contains <project>/<vintage>/agg_full/)')
    if not chosen:
        return

    controller.plugins[PLUGIN_NAME]['local_encoder_data'] = Path(chosen)
    _invalidate_engine(controller)
    logger.info('Spatial Encoder bank dir set to %s', Path(chosen))
    _refresh_current_prediction(controller)


def callback(group) -> None:
    """Reset action group to 'Original' selection."""
    group.setEnabled(False)
    for action in group.actions():
        if action.text() == 'Original':
            action.setChecked(True)
        else:
            action.setChecked(False)
    group.setEnabled(True)


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
    **kwargs,
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

    if items.model.predictions[model] is not None:
        if 'probability' in items.model.predictions[model]:
            items.view.plot_histology_cumulative(
                items.view.fig_hist_ref, items.model.predictions[model]
            )
        else:
            items.view.plot_histology(
                items.view.fig_hist_ref, items.model.predictions[model], ax='right'
            )


def compute_cosmos_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
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


def compute_spatial_encoder_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
) -> Bunch[str, np.ndarray] | None:
    """
    Prediction model using the spatial encoder.

    Returns
    -------
    Bunch
        The predicted brain regions along the probe.
    """
    # Lazy import: pulls in torch + the spatial encoder model; online-only.
    import ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder as spatial

    result = spatial.predict(controller, items)
    if result is None:
        return
    region_ids, depths = result
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depths)


def compute_inference_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
) -> Bunch[str, np.ndarray] | None:
    """
    Prediction model using the inference model.

    Returns
    -------
    Bunch
        The predicted brain regions along the probe.
    """
    # Lazy import: ephysatlas is an optional dependency, only needed when inference runs.
    import ibl_alignment_gui.plugins.ephys_atlas.inference as inference

    result = inference.predict(controller, items)
    if result is None:
        return

    region_ids, depths = result
    regions = controller.model.brain_atlas.regions.get(region_ids)

    return get_region_boundaries(regions, depths / 1e6)


def compute_cumulative_predictions(
    controller: 'AlignmentGUIController', items: 'ShankController'
) -> Bunch[str, np.ndarray] | None:
    """
    Cumulative prediction model using the inference model.
    Returns
    -------
    Bunch
        A bunch containing the probability of predicted brain regions along the probe.
    """
    # Lazy import: ephysatlas is an optional dependency, only needed when inference runs.
    import ibl_alignment_gui.plugins.ephys_atlas.inference as inference

    result = inference.predict_cumulative(controller, items)
    if result is None:
        return

    cprobas, depths, colours, regions = result
    data = Bunch(depths=depths, regions=regions, colours=colours, probability=cprobas)

    return data


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

    data = Bunch(region=region, axis_label=region_label, colour=region_colour)

    return data
