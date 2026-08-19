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
from ephysatlas.spatial_encoder.model_registry import (
    EphysAtlasReleaseRegistry,
    RegistryError,
    split_manifest_to_builder_format,
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
    _get_features_df
)
from ibl_alignment_gui.utils.utils import shank_loop

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController

logger = logging.getLogger(__name__)

MODEL_VINTAGE = '2026_W26'
HF_REPO_ID = 'AlonSaguy/ephys-atlas-models'
MODEL_NAME = 'Encoding'  # key under which the alignment engine is cached on the plugin
PREDICTION_KEY = 'Spatial Encoder'  # per-shank cache key for the spatial-encoder prediction

S3_MODEL_NAMES = [MODEL_VINTAGE]  # kept name for GUI compatibility; releases now come from Hugging Face

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
        self._dir_edit.setPlaceholderText('2026_W26 release directory (models/, context/, preprocessing/)')
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
            self, 'Select Spatial Encoder release directory')
        if not chosen_path:
            return
        chosen_path = Path(chosen_path)
        if not validate_encoder_folder(chosen_path):
            QtWidgets.QMessageBox.warning(
                self, 'Channel Prediction', f'No complete {MODEL_VINTAGE} release found under:\n{chosen_path}')
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
                'No features found for this probe. Set via Plugins -> Channel Prediction -> Load features file...')
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
                'No features found for this probe. Set via Plugins -> Channel Prediction -> Load features file...')
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
    """Check that ``enc_dir`` is a complete spatial-encoder release directory."""
    enc_dir = Path(enc_dir)
    return all(
        p.exists()
        for p in [
            enc_dir / 'models' / 'channel' / 'spatial_encoder.pt',
            enc_dir / 'models' / 'channel' / 'confidence_model.pt',
            enc_dir / 'context' / 'agea_vol_pca.npy',
            enc_dir / 'context' / 'merfish_vol_pca.npy',
            enc_dir / 'preprocessing' / 'channel_stats.npz',
            enc_dir / 'split.json',
            enc_dir / 'config.json',
            enc_dir / 'features.json',
        ]
    )

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
        return f'No complete {MODEL_VINTAGE} release found under:\n{enc_dir}'
    if enc_data is not None and not validate_feature_folder(enc_data):
        return f'No features table (raw_ephys_features*.pqt) found under:\n{enc_data}'
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
    conf_path = Path(model_path) / 'models' / 'channel' / 'confidence_model.pt'
    if not conf_path.exists():
        print(f'[Alignment engine] No confidence model found at {conf_path}; continuing without it.')
        return None
    try:
        ckpt = torch.load(conf_path, map_location=device)
        arch = ckpt.get('architecture', {})
        conf_model = ProbeSequenceConfidenceTransformer(
            f_ctx=f_ctx,
            f_e=f_e,
            d_model=int(arch.get('d_model', 64)),
            nhead=int(arch.get('nhead', 4)),
            depth=int(arch.get('depth', 2)),
            mlp_ratio=float(arch.get('mlp_ratio', 2.0)),
            drop=float(arch.get('drop', 0.1)),
        ).to(device)
        conf_model.load_state_dict(ckpt['model_state'], strict=True)
        conf_model.eval()
        return conf_model
    except Exception as exc:
        print(f'[Alignment engine] Confidence model at {conf_path} is incompatible ({exc}); continuing without it.')
        return None

def _build_context_manager(cfg: AtlasPCAConfig, *, model_name: str | None, local_path: Path, model_path: Path):
    """Load the frozen PCA context from the release bundle."""
    return ContextAtlasManager(
        cfg,
        regenerate_context=False,
        output_dir=Path(model_path) / 'context',
    )

def _unpack_loader_outputs(loaders):
    """Unpack the current release-aware dataset builder, with old fallbacks."""
    if len(loaders) == 10:
        (train_loader, _conf_train_loader, _val_loader, _test_loader,
         e_mean, e_std, ctx_mean, ctx_std, split_info, _stats) = loaders
    elif len(loaders) == 9:
        (train_loader, _conf_train_loader, _val_loader, _test_loader,
         e_mean, e_std, ctx_mean, ctx_std, split_info) = loaders
    elif len(loaders) == 7:
        train_loader, _val_loader, _test_loader, e_mean, e_std, ctx_mean, ctx_std = loaders
        split_info = None
    else:
        raise RuntimeError(f'Unexpected loader return length: {len(loaders)}')
    return train_loader, e_mean, e_std, ctx_mean, ctx_std, split_info

def _get_encoder_path_from_s3(one: ONE, model_name: str) -> Path | None:
    """Compatibility name: resolve the tagged release from Hugging Face."""
    try:
        registry = EphysAtlasReleaseRegistry()
        return registry.resolve_release(
            str(model_name or MODEL_VINTAGE),
            repo_id=HF_REPO_ID,
            require_weights=True,
        )
    except Exception as exc:
        logger.warning('Hugging Face spatial-encoder release download failed: %s', exc)
        return None

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
    """Build the GUI engine from the authoritative 2026_W26 release bundle.

    Alignment, histology usage and channel-order handling below this loader are unchanged.
    """
    print('Data loading and model initialization (one-time)')
    t0 = time.time()
    device = _as_device()
    plugin = plugin_state(controller)[MODEL_NAME]

    if model_path is None:
        model_path = _get_encoder_path_from_s3(one, model_name or MODEL_VINTAGE)
        if model_path is None:
            raise RuntimeError(f'Could not resolve Hugging Face release {MODEL_VINTAGE}')
    model_path = Path(model_path)
    if not validate_encoder_folder(model_path):
        raise RuntimeError(f'Incomplete spatial-encoder release under: {model_path}')

    if data_path is None:
        if one is None:
            one = ONE(base_url='https://alyx.internationalbrainlab.org')
        data_path = _get_encoder_data_from_s3(one, MODEL_VINTAGE)
        if data_path is None:
            raise RuntimeError(f'Could not load feature tables for {MODEL_VINTAGE}')
    data_path = Path(data_path)

    plugin['local_encoder_dir'] = model_path
    plugin['local_encoder_data'] = data_path
    plugin['model_name'] = str(model_name or MODEL_VINTAGE)

    registry = EphysAtlasReleaseRegistry()
    # model_path may be a manually-selected copy rather than registry.release_dir().
    # Read its release artifacts directly.
    import json
    with (model_path / 'config.json').open('r', encoding='utf-8') as f:
        release_config = json.load(f)
    with (model_path / 'split.json').open('r', encoding='utf-8') as f:
        release_split = json.load(f)
    with (model_path / 'features.json').open('r', encoding='utf-8') as f:
        release_features = json.load(f)['features']
    if list(release_features) != list(FEATURE_LIST):
        raise RuntimeError('FEATURE_LIST does not match the ordered feature list in the release.')
    with np.load(model_path / 'preprocessing' / 'channel_stats.npz', allow_pickle=False) as z:
        preprocessing_stats = {k: z[k].copy() for k in z.files}

    context_cfg = release_config.get('context', {})
    channel_cfg = release_config.get('channel_level', {})
    arch = channel_cfg.get('architecture', {})
    neigh = channel_cfg.get('neighbors', {})
    cfg = AtlasPCAConfig(
        n_cell_pcs=int(context_cfg.get('n_cell_pcs', 50)),
        n_gene_pcs=int(context_cfg.get('n_gene_pcs', 50)),
    )
    ctx_manager = _build_context_manager(
        cfg, model_name=model_name, local_path=model_path, model_path=model_path
    )

    pid_str, ephys, probe_positions, _ = LoadInsertionData(
        project=release_config.get('data', {}).get('project', 'ea_active'),
        agg=release_config.get('data', {}).get('agg', 'agg_full'),
        VINTAGE=MODEL_VINTAGE,
        path_data=data_path,
    )
    pid_str = [str(x) for x in pid_str]

    M_MAX = int(neigh.get('m_max', 8))
    RADIUS_UM = float(neigh.get('radius_um', 500))
    split_manifest = split_manifest_to_builder_format(release_split)

    try:
        loaders = build_channels_plus_emptyvoxels_with_neighbors(
            ctx_manager=ctx_manager,
            ephys=ephys,
            probe_positions=probe_positions,
            RADIUS_UM=RADIUS_UM,
            M_MAX=M_MAX,
            pid_names=pid_str,
            split_manifest=split_manifest,
            preprocessing_stats=preprocessing_stats,
            return_preprocessing_stats=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            'The checked-out ephysatlas branch is too old for the 2026_W26 release-aware '
            'builder API. Switch to the branch containing split_manifest/preprocessing_stats support.'
        ) from exc

    train_loader, e_mean, e_std, ctx_mean, ctx_std, _split_info = _unpack_loader_outputs(loaders)
    handles = alignment_handles_from_loader(train_loader)
    F_ctx = int(ctx_mean.numel())
    F_e = int(e_mean.numel())

    model = NeighborInpaintingModel(
        f_ctx=F_ctx, f_ephys=F_e, f_out=F_e,
        e_mean=e_mean, e_std=e_std, ctx_mean=ctx_mean, ctx_std=ctx_std,
        d_model=int(arch.get('d_model', 128)),
        nhead=int(arch.get('nhead', 8)),
        depth=int(arch.get('depth', 2)),
        drop=float(arch.get('drop', 0.15)),
    ).to(device)
    ckpt = torch.load(model_path / 'models' / 'channel' / 'spatial_encoder.pt', map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()
    torch.set_grad_enabled(False)

    conf_model = _load_optional_conf_model(
        model_path=model_path, device=device, f_ctx=F_ctx, f_e=F_e
    )
    optimization_features = np.arange(len(FEATURE_LIST), dtype=int)

    print(f'[Alignment engine ready] build time: {time.time() - t0:.2f}s')
    plugin['model'] = AlignmentEngine(
        device=device, cfg=cfg, ctx_manager=ctx_manager, model=model, handles=handles,
        e_mean=e_mean, e_std=e_std, ctx_mean=ctx_mean, ctx_std=ctx_std,
        M_MAX=M_MAX, RADIUS_UM=RADIUS_UM, optimization_features=optimization_features,
        model_name=str(model_name or MODEL_VINTAGE), local_path=model_path, conf_model=conf_model,
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

    cost_matrix = ephys_cost_matrix

    finite_cost = cost_matrix[np.isfinite(cost_matrix)]
    max_cost = float(np.median(np.nan_to_num(finite_cost)))

    lam_u = 0.5 * max_cost
    lam_l = 0.1 * max_cost

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


def _build_warped_region_ids_and_depths(
    *,
    xyz_samples_gui_order: np.ndarray,
    j_map_work_order: np.ndarray,
    channel_depth_um_gui_order: np.ndarray,
    brain_atlas,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Return full histology region trace, with depth_samples warped into probe-depth space.

    GUI trace order:
        bottom -> top

    Alignment work order:
        top -> bottom

    Output:
        region_ids:
            one region id per original histology trace sample

        depth_samples:
            one depth value per original histology trace sample, in meters,
            where estimated probe tip is ~0 and estimated probe top is ~3840 um

        j_map_gui_order:
            channel -> histology trace index, in GUI order
    """
    xyz_samples_gui_order = np.asarray(xyz_samples_gui_order, dtype=np.float32)
    j_map_work_order = np.asarray(j_map_work_order, dtype=int)
    channel_depth_um_gui_order = np.asarray(channel_depth_um_gui_order, dtype=float)

    trace_len = xyz_samples_gui_order.shape[0]
    n_channels = j_map_work_order.shape[0]

    if channel_depth_um_gui_order.shape[0] != n_channels:
        raise ValueError(
            f"channel_depth_um length {channel_depth_um_gui_order.shape[0]} "
            f"does not match j_map length {n_channels}"
        )

    # work channel order was flipped relative to GUI channel order.
    # work trace was also flipped relative to GUI trace order.
    j_map_gui_order = (trace_len - 1) - j_map_work_order[::-1]
    j_map_gui_order = np.clip(j_map_gui_order.astype(int), 0, trace_len - 1)

    # Fit: histology_trace_index -> displayed_probe_depth_um
    #
    # This is the important part. We fit using the actual channel depths,
    # not np.arange(trace_len), because the GUI depth axis is physical depth.
    valid = np.isfinite(channel_depth_um_gui_order) & np.isfinite(j_map_gui_order)

    if np.sum(valid) < 2:
        print("[Alignment engine] WARNING: not enough valid points for affine warp")
        trace_idx = np.arange(trace_len, dtype=float)
        depth_um = trace_idx * 10.0
        scale_um_per_trace_sample = 10.0
        offset_um = 0.0
    else:
        x = j_map_gui_order[valid].astype(float)
        y = channel_depth_um_gui_order[valid].astype(float)

        # Robust-ish affine fit. If DTW is basically rigid, this should be close
        # to the native trace sampling scale.
        scale_um_per_trace_sample, offset_um = np.polyfit(x, y, deg=1)

        trace_idx = np.arange(trace_len, dtype=float)
        depth_um = scale_um_per_trace_sample * trace_idx + offset_um

    region_ids = gui_region_ids_from_xyz(
        xyz_samples_gui_order,
        brain_atlas,
    ).astype(int)

    depth_samples = depth_um.astype(float) / 1e6

    warp_info = dict(
        trace_len=int(trace_len),
        n_channels=int(n_channels),
        j_map_first=int(j_map_gui_order[0]),
        j_map_last=int(j_map_gui_order[-1]),
        j_map_min=int(np.min(j_map_gui_order)),
        j_map_max=int(np.max(j_map_gui_order)),
        channel_depth_first_um=float(channel_depth_um_gui_order[0]),
        channel_depth_last_um=float(channel_depth_um_gui_order[-1]),
        output_depth_first_um=float(depth_um[0]),
        output_depth_last_um=float(depth_um[-1]),
        estimated_probe_tip_depth_um=float(
            scale_um_per_trace_sample * j_map_gui_order[0] + offset_um
        ),
        estimated_probe_top_depth_um=float(
            scale_um_per_trace_sample * j_map_gui_order[-1] + offset_um
        ),
        scale_um_per_trace_sample=float(scale_um_per_trace_sample),
        offset_um=float(offset_um),
    )

    return region_ids, depth_samples, j_map_gui_order, warp_info


# -----------------------------------------------------------------------------
# Model prediction
# -----------------------------------------------------------------------------


def predict(controller, items):
    engine = get_model(controller)
    if engine is None:
        # User cancelled the load dialog; nothing to predict with.
        return None

    recorded_full_gui_order, df = _extract_recorded_features(items)
    if df is None:
        QtWidgets.QMessageBox.warning(
            controller.view, 'Channel Prediction',
            'No features found for this probe. Set via Plugins -> Channel Prediction -> Load features file...',
        )
        return None

    # ------------------------------------------------------------------
    # GUI histology trace is bottom -> top.
    # Local/batch alignment convention is top -> bottom.
    #
    # Therefore flip BOTH the histology trace and the recorded ephys table
    # before calling align().
    # ------------------------------------------------------------------
    xyz_samples_gui_order = (
        items.model.align_handle.xyz_samples.copy().astype(np.float32)
    )

    xyz_samples_work_order = xyz_samples_gui_order[::-1].copy()
    recorded_full_work_order = recorded_full_gui_order[::-1].copy()

    # ------------------------------------------------------------------
    # No extension.
    # Use only the original GUI histology samples.
    # ------------------------------------------------------------------
    out = align(
        engine.model,
        engine.ctx_manager,
        xyz_samples_work_order,
        recorded_full_work_order,
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

    # ------------------------------------------------------------------
    # Build full warped region-id vector.
    #
    # This keeps regions beyond the probe boundaries visible when possible.
    # Missing/out-of-range samples are set to 0.
    # ------------------------------------------------------------------
    # Channel depth axis: should be 0 at tip and ~3840 at top.
    channel_depth_um_gui_order = df["axial_um"].to_numpy(dtype=float)

    region_ids, depth_samples, j_map_gui_order, warp_info = (
        _build_warped_region_ids_and_depths(
            xyz_samples_gui_order=xyz_samples_gui_order,
            j_map_work_order=out["j_map_all_i"],
            channel_depth_um_gui_order=channel_depth_um_gui_order,
            brain_atlas=controller.model.brain_atlas,
        )
    )

    if len(region_ids) != len(depth_samples):
        print(
            "[Alignment engine] WARNING: region_ids/depth_samples length mismatch:",
            len(region_ids),
            len(depth_samples),
        )

    print("[Alignment debug]")
    print("[Alignment debug]")
    print("No trace extension used")
    print("work-order j_start/j_end:", int(out["j_start"]), int(out["j_end"]))
    print("region_ids len:", len(region_ids))
    print("depth_samples len:", len(depth_samples))
    print("channel_depth_um first/last:", channel_depth_um_gui_order[0],
          channel_depth_um_gui_order[-1])
    print("j_map GUI-order first/last/min/max:",
          j_map_gui_order[0], j_map_gui_order[-1],
          np.min(j_map_gui_order), np.max(j_map_gui_order))
    print("warp info:", warp_info)

    return region_ids, depth_samples
