from iblatlas.atlas import AllenAtlas
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import torch
import ephysatlas.data

MODEL_VINTAGE = "2026_W12"
MODEL_NAME = "Spatial encoder"

# -----------------------------------------------------------------------------
# Ephys Atlas repository imports
# -----------------------------------------------------------------------------
from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig,
    ContextAtlasManager,
    LoadInsertionData,
    build_channels_plus_emptyvoxels_with_neighbors,
    FEATURE_LIST,
    region_ids_from_xyz,
    GridDS,
    NeighborCollate,
)

from ephysatlas.spatial_encoder.model import (
    NeighborInpaintingModel,
    ProbeConfidenceTrainConfig,
    ProbeSequenceConfidenceTransformer,
    predict_probe_confidence_classes,
)

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
    model_name: str
    local_path: Path
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_optional_conf_model(*, model_path: Path, device: torch.device, f_ctx: int, f_e: int):
    conf_path = model_path / "probe_conf_model.pt"
    if not conf_path.exists():
        print(f"[Alignment engine] No confidence model found at {conf_path}; continuing without it.")
        return None

    ckpt = torch.load(conf_path, map_location=device)
    conf_cfg = ProbeConfidenceTrainConfig(**ckpt.get("cfg", {}))
    conf_model = ProbeSequenceConfidenceTransformer(
        f_ctx=f_ctx,
        f_e=f_e,
        d_model=conf_cfg.d_model,
        nhead=conf_cfg.nhead,
        depth=conf_cfg.depth,
        mlp_ratio=conf_cfg.mlp_ratio,
        drop=conf_cfg.drop,
    ).to(device)
    conf_model.load_state_dict(ckpt["conf_model_state"])
    conf_model.eval()
    return conf_model


def _build_context_manager(cfg: AtlasPCAConfig, *, model_name: str, local_path: Path, model_path: Path):
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
        train_loader, _conf_train_loader, _val_loader, _test_loader, e_mean, e_std, ctx_mean, ctx_std, split_info = loaders
    elif len(loaders) == 7:
        train_loader, _val_loader, _test_loader, e_mean, e_std, ctx_mean, ctx_std = loaders
        split_info = None
    else:
        raise RuntimeError(f"Unexpected loader return length: {len(loaders)}")
    return train_loader, e_mean, e_std, ctx_mean, ctx_std, split_info


def load_alignment_engine(controller) -> AlignmentEngine:
    print("Data loading and model initialization (one-time)")
    t0 = time.time()
    device = _as_device()

    model_name = f"{MODEL_VINTAGE}_SE_model"

    one = controller.model.one
    local_path = Path(one.cache_dir).joinpath("ephys_atlas_features")
    model_path = local_path / model_name
    model_path.mkdir(parents=True, exist_ok=True)

    try:
        from ephysatlas.regionclassifier import download_model
        model_path = download_model(model_path, f"encoding_models/{MODEL_VINTAGE}", one=one)
    except Exception as e:
        print(f"[Alignment engine] download_model skipped/failed: {e}")

    optimization_features = np.arange(len(FEATURE_LIST), dtype=int)

    cfg = AtlasPCAConfig()
    ctx_manager = _build_context_manager(
        cfg,
        model_name=model_name,
        local_path=local_path,
        model_path=model_path,
    )

    pid_str, ephys, probe_positions, _ = LoadInsertionData(
        VINTAGE=MODEL_VINTAGE,
        path_data=local_path,
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

    ckpt_path = model_path / f"SE_model_{MODEL_VINTAGE}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()
    torch.set_grad_enabled(False)

    conf_model = _load_optional_conf_model(
        model_path=model_path,
        device=device,
        f_ctx=F_ctx,
        f_e=F_e,
    )

    print(f"[Alignment engine ready] build time: {time.time() - t0:.2f}s")

    return AlignmentEngine(
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
        local_path=local_path,
        conf_model=conf_model,
    )


def ensure_engine(controller) -> AlignmentEngine:
    plug = controller.plugins.setdefault("Channel Prediction", {})
    if MODEL_NAME not in plug or plug[MODEL_NAME] is None:
        plug[MODEL_NAME] = load_alignment_engine(controller)
    return plug[MODEL_NAME]


def _extract_recorded_features(items):
    if not items.model.raw_data["features"]["exists"]:
        raise RuntimeError("No raw ephys feature table is available for this insertion.")

    df = items.model.raw_data["features"]["df"].copy()
    df = df.sort_values("axial_um", ascending=True).reset_index(drop=True)

    recorded_full = df[FEATURE_LIST].to_numpy(dtype=np.float32).copy()
    recorded_full[~np.isfinite(recorded_full)] = 0.0

    return recorded_full, df


def _get_current_pid(controller, items) -> str:
    for obj in (items.model, controller.model):
        for attr in ("pid", "probe_id", "eid"):
            val = getattr(obj, attr, None)
            if val is not None:
                return str(val)
    return "unknown_pid"


def _depths_for_extended_trace(*, df: pd.DataFrame, sampling_trk: np.ndarray, j_start: int, j_end: int, trace_len: int):
    """Preserve old GUI depth convention: channel depths plus extra depths."""
    depth_samples = df["axial_um"].to_numpy(dtype=float) / 1e6
    trk = np.asarray(sampling_trk, dtype=float)

    if trk.shape[0] != trace_len:
        trk = np.arange(trace_len, dtype=float) * 20e-6

    j_start = int(np.clip(j_start, 0, trace_len - 1))
    j_end = int(np.clip(j_end, j_start, trace_len - 1))

    depths_top = (trk[:j_start] - trk[j_start] + depth_samples[-1])[::-1]
    depths_bottom = (trk[j_end + 1:] - trk[j_end])[::-1]
    return depths_bottom, depth_samples, depths_top


def gui_region_ids_from_xyz(xyz_m, brain_atlas):
    return np.asarray(brain_atlas.get_labels(xyz_m, mode="clip")).astype(int).reshape(-1)


def extend_xyz_samples_to_brain(
    xyz_samples: np.ndarray,   # [C,3] meters (ground-truth channel positions; may include zeros)
    *,
    n_edge: int = 100,
    max_extra: int = 4096,
    brain_atlas=None,
    mapping: str = "Cosmos",
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
        raise ValueError(f"xyz_samples must be (C,3), got {xyz.shape}")

    # Valid (non-zero) channels
    valid = np.isfinite(xyz).all(axis=1) & ~(np.all(xyz == 0.0, axis=1))
    if valid.sum() < 2:
        return xyz_samples.astype(np.float32)

    # Keep contiguous valid block
    idx = np.where(valid)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    xyzv = xyz[i0:i1 + 1]  # [Cv,3]
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
        rids = region_ids_from_xyz(brain_atlas, xarr, mapping=mapping, mode="clip")
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

    step_top = _estimate_constant_step(top_edge)   # direction "downwards" along probe from top
    step_bot = _estimate_constant_step(bot_edge)   # direction "downwards" along probe near bottom

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

    pre_arr  = np.asarray(pre, dtype=np.float64).reshape(-1, 3)
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
        pack = ctx_manager.sample_context_numpy_m(xyz_chunk, mode="clip")
        ctx_chunk = _concat_context(pack["cell_pc"], pack["gene_pc"])
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
        handles["bank_xyz"],
        handles["bank_feat"],
        handles["bank_pid"],
        handles["nn_bank"],
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
    use_autocast = device_type == "cuda"

    for batch in dl:
        ctx_b, p_b, e_n, p_n, mask, *_ = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

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
        print("Need at least 2 recorded (non-zero) channels with non-zero features for spatial encoding.")
        return None

    recorded_std = (
        (torch.from_numpy(recorded_full.copy()) - model.e_mean.cpu())
        / (model.e_std.cpu() + 1e-8)
    ).numpy().astype(np.float64)
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
        print(f"Trace too short - resorting to rigid optimization")
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
    recorded_on_trace_raw, recorded_on_trace_filled, recorded_on_trace_counts = _scatter_recorded_onto_trace(
        recorded_full=recorded_full,
        j_map_all_i=j_map_i,
        trace_len=L_trace,
        kp_mask=kp_mask,
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


def predict(controller, items):
    engine = ensure_engine(controller)

    try:
        recorded_full, df = _extract_recorded_features(items)
    except RuntimeError as e:
        print(e)
        print("Could not extract ephys feature table. The automated alignment would not be computed")
        return None

    # align() expects the native GUI/histology trace order. Do not reverse here.
    xyz_samples = items.model.align_handle.xyz_samples.copy().astype(np.float32)

    xyz_samples_ext = extend_xyz_samples_to_brain(
        xyz_samples,
        brain_atlas=controller.model.brain_atlas,
        mapping="Cosmos",
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

    est_xyz = out["est_xyz"]
    j_start = int(out["j_start"])
    j_end = int(out["j_end"])

    sampling_trk = items.model.align_handle.ephysalign.sampling_trk.copy()

    region_ids_before = gui_region_ids_from_xyz(
        out["xyz_samples_ext"][:j_start],
        controller.model.brain_atlas,
    )
    region_ids_probe = gui_region_ids_from_xyz(
        est_xyz,
        controller.model.brain_atlas,
    )
    region_ids_after = gui_region_ids_from_xyz(
        out["xyz_samples_ext"][j_end + 1:],
        controller.model.brain_atlas,
    )

    region_ids = np.concatenate(
        [region_ids_before, region_ids_probe, region_ids_after],
        axis=0,
    )

    depth_samples = sampling_trk.copy()

    if len(region_ids) != len(depth_samples):
        print(
            "[Alignment engine] WARNING: region_ids/depth_samples length mismatch:",
            len(region_ids),
            len(depth_samples),
        )

    print("[Alignment debug]")
    print("j_start/j_end:", j_start, j_end)
    print("region_ids len:", len(region_ids))
    print("depth_samples len:", len(depth_samples))
    print("sampling_trk first/last:", sampling_trk[0], sampling_trk[-1])
    print("xyz_ext z first/last:", out["xyz_samples_ext"][0, 2], out["xyz_samples_ext"][-1, 2])
    print("selected xyz z first/last:", est_xyz[0, 2], est_xyz[-1, 2])

    return region_ids, depth_samples
