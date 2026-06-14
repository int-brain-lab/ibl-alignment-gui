"""
Spatial-encoder inference and automatic alignment for the channel-prediction plugin.

Builds the neighbour-inpainting "alignment engine" — either from a local encoder directory or
downloaded from S3 — and uses it to predict per-channel features along a probe, warp the recorded
features onto the predicted trace (dynamic time warping with a rigid fallback) and read out Cosmos
regions. The heavy ``torch``/``ephysatlas`` imports load at module import, so the plugin imports
this module lazily, only when the spatial encoder is actually used.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch
from qtpy import QtWidgets
from torch.utils.data import DataLoader

from ephysatlas.spatial_encoder.model import (
    NeighborInpaintingModel,
    ProbeConfidenceTrainConfig,
    ProbeSequenceConfidenceTransformer,
    predict_probe_confidence_classes,
)
from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig,
    ContextAtlasManager,
    FEATURE_LIST,
    GridDS,
    LoadInsertionData,
    NeighborCollate,
    build_channels_plus_emptyvoxels_with_neighbors,
    region_ids_from_xyz,
)
from iblatlas.atlas import AllenAtlas
from one.api import ONE

from ibl_alignment_gui.plugins.ephys_atlas._common import (
    clear_predictions,
    has_features,
    has_one_connection,
    needs_reload,
    plugin_state,
    s3_cache_root,
)
from ibl_alignment_gui.utils.utils import shank_loop

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController

logger = logging.getLogger(__name__)

MODEL_VINTAGE = '2026_W12'
MODEL_NAME = 'Encoding'  # key under which the alignment engine is cached on the plugin
PREDICTION_KEY = 'Spatial Encoder'  # per-shank cache key for the spatial-encoder prediction

S3_MODEL_NAMES = [
    f'encoding_models/{MODEL_VINTAGE}',
]

# -----------------------------------------------------------------------------
# GUI interaction
# -----------------------------------------------------------------------------
class _SpatialModelDialog(QtWidgets.QDialog):
    """Spatial-model selection dialog: optional S3 dropdown plus local encoder/feature folder rows.

    The two "Browse…" rows pick a local encoder dir (validated for ``SE_model_*.pt``) and a
    feature data dir, pre-filled from the currently-configured local paths. When ``options`` is
    given (online mode) a dropdown of named S3 models is shown above the rows; the local folders
    take precedence over the dropdown, and choosing a dropdown model clears both so the encoder
    downloads from S3. Offline mode (no ``options``) shows only the two folder rows.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        title: str,
        options: list[str] | None = None,
        current: str | None = None,
        current_dir: Path | None = None,
        current_data: Path | None = None,
    ) -> None:
        """Build the dialog.

        Parameters
        ----------
        parent : QtWidgets.QWidget
            Parent widget for the dialog.
        title : str
            Window title.
        options : list of str or None
            Named S3 models to offer in a dropdown (online mode). When None or empty no dropdown is
            shown and only the local-folder rows are available (offline mode).
        current : str or None
            Model name to pre-select in the dropdown, if present in ``options``.
        current_dir : Path or None
            Local encoder model directory to pre-fill in the first row.
        current_data : Path or None
            Local feature data directory to pre-fill in the second row.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.enc_dir: Path | None = Path(current_dir) if current_dir else None
        self.enc_data: Path | None = Path(current_data) if current_data else None
        self.enc_model: str | None = None
        layout = QtWidgets.QVBoxLayout(self)

        # Dropdown of named S3 models (online mode only).
        self._combo: QtWidgets.QComboBox | None = None
        if options:
            layout.addWidget(QtWidgets.QLabel('Select model:'))
            self._combo = QtWidgets.QComboBox()
            self._combo.addItems(options)
            if current in options:
                self._combo.setCurrentIndex(options.index(current))
            layout.addWidget(self._combo)

        # Row 1 — local encoder model dir (takes precedence over the dropdown when set).
        self._dir_edit = QtWidgets.QLineEdit()
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setPlaceholderText('directory containing SE_model_*.pt')
        if self.enc_dir is not None:
            self._dir_edit.setText(str(self.enc_dir))
        browse1 = QtWidgets.QPushButton('Browse…')
        browse1.clicked.connect(self._browse_model)
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel('Model:'))
        row1.addWidget(self._dir_edit)
        row1.addWidget(browse1)
        layout.addLayout(row1)

        # Row 2 — local feature dir.
        self._data_edit = QtWidgets.QLineEdit()
        self._data_edit.setReadOnly(True)
        self._data_edit.setPlaceholderText('directory containing feature data raw_ephys_features*.pqt')
        if self.enc_data is not None:
            self._data_edit.setText(str(self.enc_data))
        browse2 = QtWidgets.QPushButton('Browse…')
        browse2.clicked.connect(self._browse_data)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel('Features:'))
        row2.addWidget(self._data_edit)
        row2.addWidget(browse2)
        layout.addLayout(row2)

        # Choosing a dropdown model clears the local folders so the S3 model is used instead.
        # 'activated' fires only on user interaction, so the init-time setCurrentIndex above and
        # any pre-filled local folders are left untouched.
        if self._combo is not None:
            self._combo.activated.connect(self._clear_local)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _browse_model(self) -> None:
        """Pick the encoder model dir, requiring at least one SE_model_*.pt inside."""
        chosen_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Spatial Encoder model dir (SE_model_*.pt + *_vol_pca.npy)')
        if not chosen_path:
            return
        chosen_path = Path(chosen_path)
        if not validate_encoder_folder(chosen_path):
            QtWidgets.QMessageBox.warning(
                self, 'Channel Prediction', f'No "SE_model_*.pt" found under:\n{chosen_path}')
            return
        self.enc_dir = chosen_path
        self.enc_model = chosen_path.name
        self._dir_edit.setText(str(chosen_path))

    def _browse_data(self) -> None:
        """Pick the features table dir, requiring the vintage feature tables inside."""
        chosen_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Spatial Encoder features')
        if not chosen_path:
            return
        chosen_path = Path(chosen_path)
        if not validate_feature_folder(chosen_path):
            QtWidgets.QMessageBox.warning(
                self, 'Channel Prediction',
                f'No "{MODEL_VINTAGE}" feature tables (raw_ephys_features*.pqt) found under:\n{chosen_path}')
            return
        self.enc_data = chosen_path
        self._data_edit.setText(str(chosen_path))

    def _clear_local(self, *_) -> None:
        """Drop chosen/pre-filled local folders so the dropdown model is used instead."""
        self.enc_dir = None
        self.enc_data = None
        self.enc_model = None
        self._dir_edit.clear()
        self._data_edit.clear()

    def _on_accept(self) -> None:
        """Validate the selection on OK; warn and keep the dialog open if it is invalid.

        Closes (accepts) only when the choice is loadable: a complete pair of valid local folders,
        or — online — the dropdown model. An invalid folder, a half-filled local pair, or offline
        with nothing chosen shows a warning and leaves the dialog open to retry.
        """
        error = _selection_error(self.enc_dir, self.enc_data, self._combo is not None)
        if error is not None:
            QtWidgets.QMessageBox.warning(self, 'Channel Prediction', error)
            return
        self.accept()

    def selected_model(self) -> str | None:
        """Return the dropdown model name, or None in offline (no-dropdown) mode."""
        return self._combo.currentText() if self._combo is not None else None


def load_model_dialog(controller: AlignmentGUIController) -> bool:
    """Run the spatial-model load GUI and (re)build the alignment engine as needed.

    Shows the local encoder + feature folder rows in both modes, with a dropdown of named S3
    models added when a ONE connection is available. A chosen pair of local folders takes
    precedence and loads from disk; otherwise the selected dropdown model is downloaded via S3.
    Offline with incomplete local folders warns and aborts. The engine is rebuilt only when the
    source changed (see :func:`ibl_alignment_gui.plugins.ephys_atlas._common.needs_reload`).

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    bool
        True if the engine was (re)loaded, False if the user cancelled or nothing changed.
    """
    plugin = plugin_state(controller)

    current_model = plugin.get(MODEL_NAME, None)

    if current_model is None:
        current_model = dict(
            model=None,
            model_name=None,
            local_encoder_dir=None,
            local_encoder_data=None,
        )
        plugin[MODEL_NAME] = current_model

    current_dir = current_model['local_encoder_dir']
    current_data = current_model['local_encoder_data']

    has_one, one = has_one_connection(controller)

    if not has_one:
        # Offline prediction reads features from a locally-loaded parquet; without one there is
        # nothing to predict on, so steer the user to load it before choosing a model.
        if not has_features(controller):
            QtWidgets.QMessageBox.warning(
                controller.view, 'Channel Prediction',
                'Load a features file first via "Load features file…" before loading a model.')
            return False
        dialog = _SpatialModelDialog(
            controller.view, 'Load Spatial Model',
            current_dir=current_dir, current_data=current_data)
    else:
        # Online the model loads without a local features file, so check here that the insertion
        # actually has features to predict on before letting the user pick a model.
        if not has_features(controller):
            QtWidgets.QMessageBox.warning(
                controller.view, 'Channel Prediction',
                'No features found for this insertion.')
            return False
        dialog = _SpatialModelDialog(
            controller.view, 'Load Spatial Model', options=S3_MODEL_NAMES, current=MODEL_VINTAGE,
            current_dir=current_dir, current_data=current_data)

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return False

    # The dialog only accepts a loadable selection: either both local folders, or (online) the
    # dropdown model. Rebuild the engine only when that source actually changed.
    if dialog.enc_dir is not None and dialog.enc_data is not None:
        # Local source: load the encoder + features from disk.
        if not needs_reload(
            current_model,
            local_encoder_dir=dialog.enc_dir,
            local_encoder_data=dialog.enc_data,
        ):
            return True
        load_alignment_engine(
            controller, model_path=dialog.enc_dir, data_path=dialog.enc_data, one=one)
        invalidate_predictions(controller)
        return True

    # S3 source: download the selected dropdown model.
    model_name = dialog.selected_model()
    if not needs_reload(current_model, model_name=model_name):
        return True
    load_alignment_engine(controller, model_name=model_name, one=one)
    invalidate_predictions(controller)
    return True

# -----------------------------------------------------------------------------
# Validation utils
# -----------------------------------------------------------------------------
def validate_encoder_folder(enc_dir: Path) -> bool:
    """Check the encoder model directory has the expected weights.

    Parameters
    ----------
    enc_dir : Path
        The local encoder model directory to validate.

    Returns
    -------
    bool
        True if ``enc_dir`` contains at least one ``SE_model_*.pt`` file, else False.
    """
    return any(enc_dir.glob('SE_model_*.pt'))


def validate_feature_folder(feature_dir: Path) -> bool:
    """Check the feature data directory holds a per-channel feature table.

    Parameters
    ----------
    feature_dir : Path
        The local feature data directory to validate.

    Returns
    -------
    bool
        True if ``feature_dir`` contains a ``raw_ephys_features*.pqt`` file, else False.
    """
    return any(feature_dir.glob('raw_ephys_features*.pqt'))


def _selection_error(
    enc_dir: Path | None, enc_data: Path | None, has_dropdown: bool
) -> str | None:
    """Return a warning for an invalid/incomplete spatial-model selection, else None.

    A selection is loadable when it is either a complete pair of valid local folders or — when a
    dropdown is available (online) — nothing local (the dropdown model is then used).

    Parameters
    ----------
    enc_dir : Path or None
        Chosen local encoder model directory, or None.
    enc_data : Path or None
        Chosen local feature data directory, or None.
    has_dropdown : bool
        Whether the dialog offers the S3 model dropdown (i.e. a ONE connection is available).

    Returns
    -------
    str or None
        A user-facing warning message when the selection cannot be loaded, else None.
    """
    if enc_dir is not None and not validate_encoder_folder(enc_dir):
        return f'No "SE_model_*.pt" found under:\n{enc_dir}'
    if enc_data is not None and not validate_feature_folder(enc_data):
        return f'No feature tables (raw_ephys_features*.pqt) found under:\n{enc_data}'
    # Local folders are all-or-nothing; offline (no dropdown) requires the full pair.
    if (enc_dir is not None) != (enc_data is not None):
        return 'Select both a model dir and a feature dir, or pick a model from the dropdown.'
    if enc_dir is None and not has_dropdown:
        return 'Offline mode needs both a local model dir and a feature dir.'
    return None


def _get_date_from_vintage(model_name: str) -> str:
    """ Get the date from the model vintage string

    which is expected to be in the format '<vintage>_SE_Model'. or xxxx/<vintage>

    """
    if len(model_name.split('/')) > 1:
        return model_name.split('/')[1]

    return model_name[:8]


@shank_loop
def invalidate_predictions(
    controller: AlignmentGUIController, items: ShankController, **kwargs
) -> None:
    """Drop the cached spatial-encoder prediction on a shank so the next click recomputes.

    Decorated with :func:`shank_loop`, so a single call iterates over every shank/config; the
    ``shank`` and ``config`` keywords injected by the decorator are absorbed via ``**kwargs``.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    items : ShankController
        The shank whose cached prediction is cleared.
    """
    clear_predictions(items, PREDICTION_KEY)

# -----------------------------------------------------------------------------
# Loading utils
# -----------------------------------------------------------------------------
@dataclass
class AlignmentEngine:
    device: torch.device
    cfg: AtlasPCAConfig
    ctx_manager: ContextAtlasManager
    model: NeighborInpaintingModel
    handles: dict
    e_mean: torch.Tensor
    e_std: torch.Tensor
    ctx_mean: torch.Tensor
    ctx_std: torch.Tensor
    M_MAX: int
    RADIUS_UM: float
    optimization_features: np.ndarray
    model_name: str | None
    local_path: Path | None
    conf_model: Optional[torch.nn.Module] = None


def alignment_handles_from_loader(train_loader):
    collate = train_loader.collate_fn
    return dict(
        bank_xyz=collate.bank_xyz,
        bank_feat=collate.bank_feat,
        bank_pid=collate.bank_pid,
        nn_bank=collate.nn,
    )


def _as_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_optional_conf_model(*, model_path: Path, device: torch.device, f_ctx: int, f_e: int):
    # Accept the canonical name or the local encoder dir's `Confidence_model_<VINTAGE>.pt`.
    conf_path = model_path.joinpath('probe_conf_model.pt')
    if not conf_path.exists():
        alt = sorted(model_path.glob('Confidence_model_*.pt'))
        if alt:
            conf_path = alt[0]
    if not conf_path.exists():
        print(
            f'[Alignment engine] No confidence model found at {conf_path}; continuing without it.'
        )
        return None

    # The confidence model is optional and checkpoint layouts vary (`conf_model_state` vs
    # `model_state`, with or without a saved `cfg`). Any incompatibility degrades to "no conf
    # model" rather than failing the whole engine build.
    try:
        ckpt = torch.load(conf_path, map_location=device)
        conf_cfg = ProbeConfidenceTrainConfig(**ckpt.get('cfg', {}))
        conf_model = ProbeSequenceConfidenceTransformer(
            f_ctx=f_ctx,
            f_e=f_e,
            d_model=conf_cfg.d_model,
            nhead=conf_cfg.nhead,
            depth=conf_cfg.depth,
            mlp_ratio=conf_cfg.mlp_ratio,
            drop=conf_cfg.drop,
        ).to(device)
        conf_model.load_state_dict(ckpt.get('conf_model_state', ckpt.get('model_state')))
        conf_model.eval()
        return conf_model
    except Exception as exc:
        print(
            f'[Alignment engine] Confidence model at {conf_path} is incompatible ({exc}); '
            'continuing without it.'
        )
        return None


def _build_context_manager(
    cfg: AtlasPCAConfig, *, model_name: str, local_path: Path, model_path: Path
):
    """Compatibility wrapper for old/new ContextAtlasManager signatures."""
    try:
        # Old GUI/debug version sometimes accepted model_name and output_dir=local_path.
        return ContextAtlasManager(
            cfg,
            regenerate_context=False,
            model_name=model_name,
            output_dir=local_path,
        )
    except TypeError:
        # New split utils.py signature: ContextAtlasManager(cfg, regenerate_context, output_dir).
        # The downloaded PCA files are usually inside model_path.
        return ContextAtlasManager(
            cfg,
            regenerate_context=False,
            output_dir=model_path,
        )


def _unpack_loader_outputs(loaders):
    """Support both the new 9-item and older 7-item dataset builder returns."""
    if len(loaders) == 9:
        (
            train_loader,
            _conf_train_loader,
            _val_loader,
            _test_loader,
            e_mean,
            e_std,
            ctx_mean,
            ctx_std,
            split_info,
        ) = loaders
    elif len(loaders) == 7:
        train_loader, _val_loader, _test_loader, e_mean, e_std, ctx_mean, ctx_std = loaders
        split_info = None
    else:
        raise RuntimeError(f'Unexpected loader return length: {len(loaders)}')
    return train_loader, e_mean, e_std, ctx_mean, ctx_std, split_info


def _get_encoder_path_from_s3(one: ONE, model_name: str) -> Path | None:
    """Download the named encoder model from S3 and return its local directory.

    Parameters
    ----------
    one : ONE
        ONE connection used for the download.
    model_name : str
        Name of the encoder model directory to download under the ONE cache.

    Returns
    -------
    Path or None
        The downloaded model directory, or None if the download failed.
    """
    cache_root = s3_cache_root(one)
    cache_root.joinpath(model_name).mkdir(parents=True, exist_ok=True)
    try:
        from ephysatlas.regionclassifier import download_model  # noqa: PLC0415

        model_path = download_model(cache_root, model_name, one=one)
    except Exception as exc:
        logger.warning('download_model skipped/failed: %s', exc)
        return None

    return model_path


def _get_encoder_data_from_s3(one: ONE, feature_vintage: str = MODEL_VINTAGE) -> Path | None:
    """Download the feature tables for a vintage from S3 and return their local directory.

    Parameters
    ----------
    one : ONE
        ONE connection used for the download.
    feature_vintage : str
        Vintage label whose feature tables to download (defaults to :data:`MODEL_VINTAGE`).

    Returns
    -------
    Path or None
        The directory holding the downloaded feature tables, or None if the download failed.
    """
    cache_root = s3_cache_root(one)
    try:
        from ephysatlas.data import download_tables # noqa: PLC0415

        data_path = download_tables(cache_root, feature_vintage, one=one)
    except Exception as exc:
        logger.warning('download_tables skipped/failed: %s', exc)
        return None
    return data_path

def load_alignment_engine(
    controller: AlignmentGUIController,
    model_path: Path | None = None,
    data_path: Path | None = None,
    model_name: str | None = None,
    one: ONE | None = None,
) -> None:
    """Build the Spatial Encoder alignment engine and cache it on the plugin.

    Resolves the encoder weights and feature tables — from local ``model_path`` / ``data_path``
    when both are provided, otherwise downloading ``model_name`` (and its feature vintage) from S3
    via ``one`` — then builds the model, context manager and reference-bank handles and stores the
    resulting :class:`AlignmentEngine` under
    ``controller.plugins['Channel Prediction'][MODEL_NAME]['model']``.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    model_path : Path or None
        Local encoder model directory. When given together with ``data_path``, the engine loads
        from disk and S3 is not used.
    data_path : Path or None
        Local feature data directory (see ``model_path``).
    model_name : str or None
        S3 model name to download when both local directories are not provided.
    one : ONE or None
        ONE connection used for the S3 download.

    Raises
    ------
    RuntimeError
        If the source cannot be resolved (no ONE and both local directories missing), or a
        provided local directory fails validation.
    """
    print('Data loading and model initialization (one-time)')
    t0 = time.time()
    device = _as_device()

    plugin = plugin_state(controller)[MODEL_NAME]

    if one is None and (data_path is None or model_path is None):
        raise RuntimeError(
            'No ONE connection found, must specify both local encoder and local feature directories')

    if data_path is not None and model_path is not None:
        # TODO do we need this validation here given that we have done it before?
        if not validate_encoder_folder(model_path):
            raise RuntimeError(f'No "SE_model_*.pt" found under the given model path: {model_path}')

        if not validate_feature_folder(data_path):
            raise RuntimeError(f'No "{MODEL_VINTAGE}" feature tables (raw_ephys_features*.pqt) found under the given data path: {data_path}')

        plugin['local_encoder_dir'] = model_path
        plugin['local_encoder_data'] = data_path
        plugin['model_name'] = None
    else:

        data_path = _get_encoder_data_from_s3(one, _get_date_from_vintage(model_name))
        model_path = _get_encoder_path_from_s3(one, model_name)

        plugin['local_encoder_dir'] = model_path
        plugin['local_encoder_data'] = data_path
        plugin['model_name'] = model_name


    optimization_features = np.arange(len(FEATURE_LIST), dtype=int)

    cfg = AtlasPCAConfig()
    ctx_manager = _build_context_manager(
        cfg,
        model_name=model_name,
        local_path=model_path,
        model_path=model_path,
    )

    pid_str, ephys, probe_positions, _ = LoadInsertionData(
        VINTAGE=MODEL_VINTAGE,
        path_data=data_path,
    )

    M_MAX = 8
    RADIUS_UM = 500

    loaders = build_channels_plus_emptyvoxels_with_neighbors(
        ctx_manager=ctx_manager,
        ephys=ephys,
        probe_positions=probe_positions,
        RADIUS_UM=RADIUS_UM,
        M_MAX=M_MAX,
        pid_names=pid_str,
    )

    train_loader, e_mean, e_std, ctx_mean, ctx_std, _split_info = _unpack_loader_outputs(loaders)
    handles = alignment_handles_from_loader(train_loader)

    F_ctx = int(ctx_mean.numel())
    F_e = int(ephys.shape[-1])

    model = NeighborInpaintingModel(
        f_ctx=F_ctx,
        f_ephys=F_e,
        f_out=F_e,
        e_mean=e_mean,
        e_std=e_std,
        ctx_mean=ctx_mean,
        ctx_std=ctx_std,
        d_model=128,
        nhead=8,
        depth=2,
        drop=0.15,
    ).to(device)

    ckpt_path = model_path / f'SE_model_{MODEL_VINTAGE}.pt'
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
    model.eval()
    torch.set_grad_enabled(False)

    conf_model = _load_optional_conf_model(
        model_path=model_path,
        device=device,
        f_ctx=F_ctx,
        f_e=F_e,
    )

    print(f'[Alignment engine ready] build time: {time.time() - t0:.2f}s')

    plugin['model'] = AlignmentEngine(
        device=device,
        cfg=cfg,
        ctx_manager=ctx_manager,
        model=model,
        handles=handles,
        e_mean=e_mean,
        e_std=e_std,
        ctx_mean=ctx_mean,
        ctx_std=ctx_std,
        M_MAX=M_MAX,
        RADIUS_UM=RADIUS_UM,
        optimization_features=optimization_features,
        model_name=model_name,
        local_path=model_path,
        conf_model=conf_model,
    )


def get_model(controller: AlignmentGUIController) -> AlignmentEngine | None:
    """Return the cached alignment engine, loading or prompting for it on first use.

    If no Channel Prediction model state exists yet, opens the load dialog and retries. If state
    exists but the engine has not been built, rebuilds it from the cached local paths when present
    (otherwise prompts via the dialog) and retries. Returns None if the user cancels the load
    dialog.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.

    Returns
    -------
    AlignmentEngine or None
        The loaded alignment engine, or None if the user cancelled loading.
    """
    plugin = plugin_state(controller).get(MODEL_NAME, None)

    # First time loading: prompt the user; bail out if they cancel.
    if plugin is None:
        if not load_model_dialog(controller):
            return None
        return get_model(controller)

    if plugin.get('model', None) is None:
        if (plugin.get('local_encoder_dir', None) is not None
                and plugin.get('local_encoder_data', None) is not None):
            load_alignment_engine(
                controller,
                model_path=plugin['local_encoder_dir'],
                data_path=plugin['local_encoder_data'],
            )
        elif not load_model_dialog(controller):
            return None
        return get_model(controller)

    return plugin['model']

# -----------------------------------------------------------------------------
# Feature & geometry utils
# -----------------------------------------------------------------------------
def _extract_recorded_features(items):
    if not items.model.raw_data['features']['exists']:
        raise RuntimeError('No raw ephys feature table is available for this insertion.')

    df = items.model.raw_data['features']['df'].copy()
    df = df.sort_values('axial_um', ascending=True).reset_index(drop=True)

    recorded_full = df[FEATURE_LIST].to_numpy(dtype=np.float32).copy()
    recorded_full[~np.isfinite(recorded_full)] = 0.0

    return recorded_full, df


def _get_current_pid(controller, items) -> str:
    for obj in (items.model, controller.model):
        for attr in ('pid', 'probe_id', 'eid'):
            val = getattr(obj, attr, None)
            if val is not None:
                return str(val)
    return 'unknown_pid'


def _depths_for_extended_trace_fixed(
        *, df, sampling_trk, j_start, j_end, trace_len
):
    depth_probe = df["axial_um"].to_numpy(dtype=float) / 1e6
    trk = np.asarray(sampling_trk, dtype=float)

    if trk.shape[0] != trace_len:
        trk = np.arange(trace_len, dtype=float) * 20e-6

    j_start = int(np.clip(j_start, 0, trace_len - 1))
    j_end = int(np.clip(j_end, j_start, trace_len - 1))

    # Extension before aligned probe: should be above the probe, i.e. before depth_probe[0]
    depths_before = depth_probe[0] - (trk[j_start] - trk[:j_start])

    # Extension after aligned probe: should continue after depth_probe[-1]
    depths_after = depth_probe[-1] + (trk[j_end + 1:] - trk[j_end])

    depth_samples = np.concatenate(
        [
            depths_before,
            depth_probe,
            depths_after,
        ]
    )

    return depth_samples


def gui_region_ids_from_xyz(xyz_m, brain_atlas):
    return np.asarray(brain_atlas.get_labels(xyz_m, mode='clip')).astype(int).reshape(-1)


def extend_xyz_samples_to_brain(
    xyz_samples: np.ndarray,  # [C,3] meters (ground-truth channel positions; may include zeros)
    *,
    n_edge: int = 100,
    max_extra: int = 4096,
    brain_atlas=None,
    mapping: str = 'Cosmos',
) -> np.ndarray:
    """
    Extends xyz_samples on both ends by estimating a CONSTANT step (gradient) separately
    for the top and bottom edges, then linearly extrapolating until leaving the brain (rid==0).

    This is tailored to probes where positions repeat in pairs (e.g. every two channels
    share the exact same xyz), so the "effective" step is captured by robustly averaging
    non-zero deltas within each edge window.
    """
    if brain_atlas is None:
        brain_atlas = AllenAtlas()

    xyz = np.asarray(xyz_samples, dtype=np.float64)
    if not (xyz.ndim == 2 and xyz.shape[1] == 3):
        raise ValueError(f'xyz_samples must be (C,3), got {xyz.shape}')

    # Valid (non-zero) channels
    valid = np.isfinite(xyz).all(axis=1) & ~(np.all(xyz == 0.0, axis=1))
    if valid.sum() < 2:
        return xyz_samples.astype(np.float32)

    # Keep contiguous valid block
    idx = np.where(valid)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    xyzv = xyz[i0 : i1 + 1]  # [Cv,3]
    Cv = xyzv.shape[0]
    if Cv < 2:
        return xyz_samples.astype(np.float32)

    n_edge = int(min(n_edge, Cv))
    if n_edge < 2:
        return xyz_samples.astype(np.float32)

    def _first_rid0_index(xarr: np.ndarray) -> int | None:
        xarr = np.asarray(xarr, dtype=np.float32)
        if xarr.ndim == 1:
            xarr = xarr[None, :]
        rids = region_ids_from_xyz(brain_atlas, xarr, mapping=mapping, mode='clip')
        rids = np.atleast_1d(np.asarray(rids))
        bad = np.where(rids == 0)[0]
        return int(bad[0]) if bad.size > 0 else None

    def _estimate_constant_step(edge_xyz: np.ndarray) -> np.ndarray:
        """
        Estimate constant step from a window of points [K,3] by averaging non-zero
        consecutive deltas. If everything is repeated (all deltas zero), fall back to
        the farthest difference / (K-1).
        """
        edge_xyz = np.asarray(edge_xyz, dtype=np.float64)
        if edge_xyz.shape[0] < 2:
            return np.zeros((3,), dtype=np.float64)

        d = edge_xyz[1:] - edge_xyz[:-1]  # [K-1,3]
        mag = np.linalg.norm(d, axis=1)
        nz = mag > 0  # ignore repeated pairs (zero deltas)

        if np.any(nz):
            step = d[nz].mean(axis=0)
        else:
            # Fully repeated? Use overall displacement as fallback (might still be zero).
            step = (edge_xyz[-1] - edge_xyz[0]) / max(1, (edge_xyz.shape[0] - 1))

        return step.astype(np.float64)

    # Top edge (near xyzv[0]) and bottom edge (near xyzv[-1])
    top_edge = xyzv[:n_edge]
    bot_edge = xyzv[-n_edge:]

    step_top = _estimate_constant_step(top_edge)  # direction "downwards" along probe from top
    step_bot = _estimate_constant_step(bot_edge)  # direction "downwards" along probe near bottom

    # If one side ended up ~0 (degenerate), reuse the other if it exists
    if np.linalg.norm(step_top) < 1e-12 and np.linalg.norm(step_bot) >= 1e-12:
        step_top = step_bot.copy()
    if np.linalg.norm(step_bot) < 1e-12 and np.linalg.norm(step_top) >= 1e-12:
        step_bot = step_top.copy()

    # If still degenerate, can't extend meaningfully
    if np.linalg.norm(step_top) < 1e-12 and np.linalg.norm(step_bot) < 1e-12:
        return xyz_samples.astype(np.float32)

    # ---- extend BEFORE (prepend): go "upwards" opposite to top-step direction ----
    pre = []
    cur = xyzv[0].copy()
    for _ in range(int(max_extra)):
        cur = cur - step_top
        # stop when outside brain (rid==0)
        if _first_rid0_index(cur) is not None:
            break
        pre.append(cur.copy())
    if len(pre) > 0:
        pre = pre[::-1]  # earliest -> latest

    # ---- extend AFTER (append): go "downwards" following bottom-step direction ----
    post = []
    cur = xyzv[-1].copy()
    for _ in range(int(max_extra)):
        cur = cur + step_bot
        if _first_rid0_index(cur) is not None:
            break
        post.append(cur.copy())

    pre_arr = np.asarray(pre, dtype=np.float64).reshape(-1, 3)
    post_arr = np.asarray(post, dtype=np.float64).reshape(-1, 3)

    xyz_ext = np.concatenate([pre_arr, xyzv, post_arr], axis=0).astype(np.float32)

    return xyz_ext


# -----------------------------------------------------------------------------
# Automatic alignment utils
# -----------------------------------------------------------------------------


def _concat_context(cell_pc: np.ndarray, gene_pc: np.ndarray) -> np.ndarray:
    return np.concatenate([cell_pc, gene_pc], axis=1).astype(np.float32)


@torch.no_grad()
def _sample_and_standardize_ctx_for_xyz(
    ctx_manager,
    xyz_m: np.ndarray,
    ctx_mean: torch.Tensor,
    ctx_std: torch.Tensor,
    *,
    chunk: int = 8192,
) -> torch.Tensor:
    assert xyz_m.ndim == 2 and xyz_m.shape[1] == 3

    ctx_list = []
    for s in range(0, xyz_m.shape[0], chunk):
        xyz_chunk = xyz_m[s : s + chunk].astype(np.float32, copy=False)
        pack = ctx_manager.sample_context_numpy_m(xyz_chunk, mode='clip')
        ctx_chunk = _concat_context(pack['cell_pc'], pack['gene_pc'])
        ctx_list.append(ctx_chunk)

    ctx = np.concatenate(ctx_list, axis=0).astype(np.float32)
    ctx_t = torch.from_numpy(ctx).float()

    ctx_mean = ctx_mean.detach().cpu()
    ctx_std = ctx_std.detach().cpu()

    has_ctx = ctx_t.abs().sum(dim=1) != 0
    ctx_t[has_ctx] = (ctx_t[has_ctx] - ctx_mean) / (ctx_std + 1e-8)

    return ctx_t


@torch.no_grad()
def predict_features_at_xyz(
    model,
    ctx_manager,
    handles: dict,
    xyz_m: np.ndarray,
    *,
    batch_size: int = 512,
    radius_um: float,
    M_max: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()

    xyz_m = np.asarray(xyz_m, dtype=np.float32)
    xyz_t = torch.from_numpy(xyz_m).float()

    ctx_q = _sample_and_standardize_ctx_for_xyz(
        ctx_manager,
        xyz_m,
        model.ctx_mean,
        model.ctx_std,
        chunk=8192,
    )

    F_e = int(model.e_mean.numel())
    qds = GridDS(ctx_q, xyz_t, F_e)

    collate = NeighborCollate(
        ctx_manager,
        handles['bank_xyz'],
        handles['bank_feat'],
        handles['bank_pid'],
        handles['nn_bank'],
        e_feat_dim=F_e,
        M_max=M_max,
        radius_um=radius_um,
    )

    dl = DataLoader(
        qds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )

    mu_all = []
    device_type = device.type
    use_autocast = device_type == 'cuda'

    for batch in dl:
        ctx_b, p_b, e_n, p_n, mask, *_ = [x.to(device) if torch.is_tensor(x) else x for x in batch]

        with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
            _, mu = model(ctx_b, p_b, e_n, p_n, mask)

        mu_all.append(mu.detach().cpu())

    return torch.cat(mu_all, dim=0)


def build_cost_matrix(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True).T
    AB = A @ B.T

    return (AA + BB - 2.0 * AB).clip(min=0.0)


def dynamic_time_warping_debug(C, lam_d=0.0, lam_u=0.1, lam_l=0.1, band=None, open_begin=True):
    C = np.asarray(C, dtype=np.float64)
    C = np.where(np.isfinite(C), C, np.inf)

    N, M = C.shape
    D = np.full((N, M), np.inf, dtype=np.float64)
    P = np.full((N, M), -1, dtype=np.int8)

    if band is None:
        band = np.ones((N, M), dtype=bool)
    else:
        band = np.asarray(band, dtype=bool)

    if band[0, 0]:
        D[0, 0] = C[0, 0]

    for j in range(1, M):
        if not band[0, j]:
            continue
        if open_begin:
            D[0, j] = C[0, j]
            P[0, j] = -1
        else:
            D[0, j] = C[0, j] + D[0, j - 1] + lam_l
            P[0, j] = 2

    for i in range(1, N):
        if not band[i, 0]:
            continue
        D[i, 0] = C[i, 0] + D[i - 1, 0] + lam_u
        P[i, 0] = 1

    for i in range(1, N):
        for j in range(1, M):
            if not band[i, j]:
                continue

            candidates = [
                D[i - 1, j - 1] + lam_d,
                D[i - 1, j] + lam_u,
                D[i, j - 1] + lam_l,
            ]

            k = int(np.argmin(candidates))
            D[i, j] = C[i, j] + candidates[k]
            P[i, j] = k

    j_end = int(np.nanargmin(D[N - 1]))
    total = float(D[N - 1, j_end])

    i, j = N - 1, j_end
    path = [(i, j)]

    while i > 0 or (not open_begin and j > 0):
        k = P[i, j]

        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i, j = i - 1, j
        elif k == 2:
            i, j = i, j - 1
        else:
            break

        path.append((i, j))

    path.reverse()
    j_start = path[0][1]

    return j_start, j_end, path, total, D, P


def rigid_assignment(A, B):
    best_k, best_mse = 0, np.inf
    Nr = A.shape[0]

    for k in range(0, B.shape[0] - Nr + 1):
        m = ((B[k : k + Nr] - A) ** 2).mean()
        if m < best_mse:
            best_mse, best_k = m, k

    j_start = best_k
    j_end = best_k + Nr - 1
    path = [(i, best_k + i) for i in range(Nr)]

    return j_start, j_end, path


def _scatter_recorded_onto_trace(
    recorded_full: np.ndarray,
    j_map_all_i: np.ndarray,
    trace_len: int,
    *,
    kp_mask: Optional[np.ndarray] = None,
):
    recorded_full = np.asarray(recorded_full)
    j_map_all_i = np.asarray(j_map_all_i, dtype=int)

    C_rec, F = recorded_full.shape
    L = int(trace_len)

    if kp_mask is None:
        kp_mask = np.ones((C_rec,), dtype=bool)
    else:
        kp_mask = np.asarray(kp_mask, dtype=bool)

    sums = np.zeros((L, F), dtype=np.float64)
    counts = np.zeros((L,), dtype=np.int64)

    for c in range(C_rec):
        if not kp_mask[c]:
            continue
        j = int(j_map_all_i[c])
        if 0 <= j < L:
            sums[j] += recorded_full[c]
            counts[j] += 1

    recorded_on_trace_raw = np.full((L, F), np.nan, dtype=np.float64)
    recorded_on_trace_filled = np.zeros((L, F), dtype=np.float64)

    hit = counts > 0
    recorded_on_trace_raw[hit] = sums[hit] / counts[hit, None]
    recorded_on_trace_filled[hit] = sums[hit] / counts[hit, None]

    return recorded_on_trace_raw, recorded_on_trace_filled, counts


@torch.no_grad()
def classify_aligned_probe_channels(
    *,
    conf_model,
    model,
    ctx_manager,
    recorded_full: np.ndarray,
    est_xyz: np.ndarray,
    mu_std_est: np.ndarray | torch.Tensor,
    device: torch.device,
):
    conf_model.eval()

    rec_raw = np.asarray(recorded_full, dtype=np.float32)
    xyz_np = np.asarray(est_xyz, dtype=np.float32)

    C, F_e = rec_raw.shape

    if torch.is_tensor(mu_std_est):
        pred_std_np = mu_std_est.detach().cpu().numpy().astype(np.float32)
    else:
        pred_std_np = np.asarray(mu_std_est, dtype=np.float32)

    rec_is_finite = np.isfinite(rec_raw).all(axis=1)
    rec_has_signal = ~np.all(np.nan_to_num(rec_raw, nan=0.0) == 0.0, axis=1)
    xyz_is_finite = np.isfinite(xyz_np).all(axis=1)
    pred_is_finite = np.isfinite(pred_std_np).all(axis=1)

    valid_mask = rec_is_finite & rec_has_signal & xyz_is_finite & pred_is_finite
    valid_t = torch.from_numpy(valid_mask).bool()

    e_mean = model.e_mean.detach().cpu().numpy().astype(np.float32)
    e_std = model.e_std.detach().cpu().numpy().astype(np.float32)

    rec_raw_safe = np.nan_to_num(rec_raw, nan=0.0, posinf=0.0, neginf=0.0)
    rec_std = (rec_raw_safe - e_mean) / (e_std + 1e-8)
    rec_std[~valid_mask] = 0.0

    pred_std_np = np.nan_to_num(pred_std_np, nan=0.0, posinf=0.0, neginf=0.0)
    pred_std_np[~valid_mask] = 0.0

    ctx_std_t = _sample_and_standardize_ctx_for_xyz(
        ctx_manager,
        xyz_np,
        model.ctx_mean,
        model.ctx_std,
        chunk=8192,
    ).float()
    ctx_std_t[~valid_t] = 0.0

    logits, probs, _ = predict_probe_confidence_classes(
        conf_model=conf_model,
        rec_std=torch.from_numpy(rec_std).float(),
        pred_std=torch.from_numpy(pred_std_np).float(),
        ctx_std=ctx_std_t,
        valid_mask=valid_t,
        device=device,
    )

    probs_cpu = probs.detach().cpu().float()
    pred_cls = probs_cpu.argmax(dim=1).numpy().astype(np.int64)
    pred_cls[~valid_mask] = -1

    probs_np = probs_cpu.numpy().astype(np.float32)
    probs_np[~valid_mask] = np.nan

    return pred_cls, probs_np


@torch.no_grad()
def align(
    model,
    ctx_manager,
    xyz_samples_ext,
    recorded_full,
    handles,
    optimization_features,
    RADIUS_UM,
    M_MAX,
    device,
    conf_model=None,
    return_debug: bool = True,
    brain_atlas=None,
):
    C_full = recorded_full.shape[0]
    L_trace = xyz_samples_ext.shape[0]

    kp_mask = ~np.all(recorded_full == 0.0, axis=1)
    if kp_mask.sum() < 2:
        print(
            'Need at least 2 recorded (non-zero) channels with non-zero features for spatial encoding.'
        )
        return None

    recorded_std = (
        (
            (torch.from_numpy(recorded_full.copy()) - model.e_mean.cpu())
            / (model.e_std.cpu() + 1e-8)
        )
        .numpy()
        .astype(np.float64)
    )
    recorded_opt = recorded_std[kp_mask][:, optimization_features]

    # full-trace prediction
    pred_std_full = predict_features_at_xyz(
        model,
        ctx_manager,
        handles,
        xyz_samples_ext,
        batch_size=512,
        radius_um=RADIUS_UM,
        M_max=M_MAX,
        device=device,
    )
    pred_std_full_np = pred_std_full.detach().cpu().numpy().astype(np.float64)
    pred_std_opt = pred_std_full_np[:, optimization_features]

    ephys_cost_matrix = build_cost_matrix(recorded_opt, pred_std_opt)

    region_cost_matrix = None
    region_cost_norm = None
    ephys_cost_norm = None
    has_region_cost = None
    trace_region_target_idx = None
    trace_region_target_name = None

    cost_matrix = ephys_cost_matrix

    W_ephys = np.ones_like(ephys_cost_matrix, dtype=np.float64)
    W_region = np.zeros_like(ephys_cost_matrix, dtype=np.float64)

    finite_cost = cost_matrix[np.isfinite(cost_matrix)]
    max_cost = float(np.median(np.nan_to_num(finite_cost)))
    jump_frac = 0.5

    lam_u = jump_frac * max_cost
    lam_l = jump_frac * max_cost

    j_start, j_end, path, total_cost, D, P = dynamic_time_warping_debug(
        cost_matrix,
        lam_d=0.0,
        lam_u=lam_u,
        lam_l=lam_l,
        open_begin=True,
    )

    min_overlap_channels = int(0.9 * int(kp_mask.sum()))
    if (j_end - j_start + 1) < min_overlap_channels:
        print(f'Trace too short - resorting to rigid optimization')
        j_start, j_end, path = rigid_assignment(recorded_opt, pred_std_opt)

    i_seq, j_seq = np.array(path, dtype=int).T
    j_for_i = np.full(recorded_opt.shape[0], np.nan)
    j_for_i[i_seq] = j_seq
    j_for_i = (
        pd.Series(j_for_i)
        .ffill()
        .bfill()
        .astype(int)
        .clip(0, pred_std_opt.shape[0] - 1)
        .to_numpy()
    )

    # map from ALL recorded channels -> full trace indices
    j_map = np.interp(np.arange(C_full), np.where(kp_mask)[0], j_for_i.astype(float))
    j_map_i = np.clip(np.round(j_map).astype(int), 0, pred_std_opt.shape[0] - 1)

    est_xyz = xyz_samples_ext[j_map_i]

    if not return_debug:
        return est_xyz

    # aligned-window prediction as before
    mu_std_est = pred_std_full_np[j_map_i]

    # create full-trace recorded array with NaNs outside aligned channels
    recorded_on_trace_raw, recorded_on_trace_filled, recorded_on_trace_counts = (
        _scatter_recorded_onto_trace(
            recorded_full=recorded_full,
            j_map_all_i=j_map_i,
            trace_len=L_trace,
            kp_mask=kp_mask,
        )
    )

    pred_cls_est = None
    cls_probs_est = None
    pred_cls_trace = None
    cls_probs_trace = None

    if conf_model is not None:
        # per-channel class/confidence on aligned estimated probe (same as before)
        pred_cls_est, cls_probs_est = classify_aligned_probe_channels(
            conf_model=conf_model,
            model=model,
            ctx_manager=ctx_manager,
            recorded_full=recorded_full,
            est_xyz=est_xyz,
            mu_std_est=mu_std_est,
            device=device,
        )

        # full-trace class/confidence
        # Use the NaN-padded trace for plotting and the zero-filled trace for inference.
        pred_cls_trace, cls_probs_trace = classify_aligned_probe_channels(
            conf_model=conf_model,
            model=model,
            ctx_manager=ctx_manager,
            recorded_full=recorded_on_trace_raw,  # not recorded_on_trace_filled
            est_xyz=xyz_samples_ext,
            mu_std_est=pred_std_full_np,
            device=device,
        )

    return dict(
        est_xyz=est_xyz,
        kp_mask=kp_mask,
        j_map_all_i=j_map_i,
        cost_matrix=cost_matrix,
        path=np.array(path, dtype=int),
        total_cost=float(total_cost),
        j_start=int(j_start),
        j_end=int(j_end),
        pred_cls_est=pred_cls_est,
        cls_probs_est=cls_probs_est,
        mu_std_est=mu_std_est,
        # full-trace outputs
        xyz_samples_ext=xyz_samples_ext,
        mu_std_trace=pred_std_full_np,
        pred_cls_trace=pred_cls_trace,
        cls_probs_trace=cls_probs_trace,
        recorded_on_trace_raw=recorded_on_trace_raw,
        recorded_on_trace_counts=recorded_on_trace_counts,
        ephys_cost_matrix=ephys_cost_matrix,
    )


# -----------------------------------------------------------------------------
# Model prediction
# -----------------------------------------------------------------------------


def predict(controller, items):
    engine = get_model(controller)
    if engine is None:
        # User cancelled the load dialog; nothing to predict with.
        return None

    try:
        recorded_full, df = _extract_recorded_features(items)
    except RuntimeError as e:
        print(e)
        print(
            'Could not extract ephys feature table. The automated alignment would not be computed'
        )
        return None

    # align() expects the native GUI/histology trace order. Do not reverse here.
    xyz_samples = items.model.align_handle.xyz_samples.copy().astype(np.float32)

    xyz_samples_ext = extend_xyz_samples_to_brain(
        xyz_samples,
        brain_atlas=controller.model.brain_atlas,
        mapping='Cosmos',
    ).astype(np.float32)

    out = align(
        engine.model,
        engine.ctx_manager,
        xyz_samples_ext,
        recorded_full,
        engine.handles,
        engine.optimization_features,
        engine.RADIUS_UM,
        engine.M_MAX,
        engine.device,
        conf_model=engine.conf_model,
        return_debug=True,
        brain_atlas=controller.model.brain_atlas,
    )

    if out is None:
        return None

    est_xyz = out['est_xyz']
    j_start = int(out['j_start'])
    j_end = int(out['j_end'])

    sampling_trk = items.model.align_handle.ephysalign.sampling_trk.copy()

    region_ids_before = gui_region_ids_from_xyz(
        out['xyz_samples_ext'][:j_start],
        controller.model.brain_atlas,
    )
    region_ids_probe = gui_region_ids_from_xyz(
        est_xyz,
        controller.model.brain_atlas,
    )
    region_ids_after = gui_region_ids_from_xyz(
        out['xyz_samples_ext'][j_end + 1 :],
        controller.model.brain_atlas,
    )

    region_ids = np.concatenate(
        [
            region_ids_before,
            region_ids_probe,
            region_ids_after,
        ],
        axis=0,
    )

    depth_samples = _depths_for_extended_trace_fixed(
        df=df,
        sampling_trk=sampling_trk,
        j_start=j_start,
        j_end=j_end,
        trace_len=out["xyz_samples_ext"].shape[0],
    )

    if len(region_ids) != len(depth_samples):
        print(
            '[Alignment engine] WARNING: region_ids/depth_samples length mismatch:',
            len(region_ids),
            len(depth_samples),
        )

    print('[Alignment debug]')
    print('j_start/j_end:', j_start, j_end)
    print('region_ids len:', len(region_ids))
    print('depth_samples len:', len(depth_samples))
    print('sampling_trk first/last:', sampling_trk[0], sampling_trk[-1])
    print('xyz_ext z first/last:', out['xyz_samples_ext'][0, 2], out['xyz_samples_ext'][-1, 2])
    print('selected xyz z first/last:', est_xyz[0, 2], est_xyz[-1, 2])

    return region_ids, depth_samples
