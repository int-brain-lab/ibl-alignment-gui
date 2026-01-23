import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import ephysatlas.data

from ephysatlas.spatial_encoder.utils import (ContextAtlasManager, AtlasPCAConfig, GridDS, LoadInsertionData,
                                              build_channels_plus_emptyvoxels_with_neighbors, NeighborCollate, FEATURE_LIST)
from ephysatlas.spatial_encoder.model import NeighborInpaintingModel

from typing import Optional
from dataclasses import dataclass


MODEL_NAME = 'Spatial encoder'

@dataclass
class AlignmentEngine:
    device: str
    cfg: 'AtlasPCAConfig'
    ctx_manager: 'ContextAtlasManager'
    model: 'NeighborInpaintingModel'
    handles: dict
    e_mean: 'torch.Tensor'
    e_std:  'torch.Tensor'
    ctx_mean: 'torch.Tensor'
    ctx_std:  'torch.Tensor'
    ephys: np.ndarray
    probe_positions: np.ndarray
    probe_planned_positions: np.ndarray
    pid_str: np.ndarray
    M_MAX: int
    RADIUS_UM: float
    optimization_features: np.ndarray

def load_alignment_engine(controller: 'AlignmentGUIController') -> AlignmentEngine:
    print('Data loading and model initialization (one-time)')
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = '2024_W43_SE_model'
    one = controller.model.one
    local_path = one.cache_dir.joinpath('ephys_atlas_features')
    _ = ephysatlas.regionclassifier.download_model(local_path=local_path, model_name=model_name, one=one)

    optimization_features = np.arange(len(FEATURE_LIST))

    # Context manager + config
    cfg = AtlasPCAConfig()
    ctx_manager = ContextAtlasManager(cfg, regenerate_context=False, model_name=model_name, output_dir=local_path)

    # Load ephys & probe positions
    pid_str, ephys, probe_positions, probe_planned_positions = LoadInsertionData(VINTAGE='2025_W52')

    # Build dataset & splits ONCE
    M_MAX = 8
    RADIUS_UM = 500
    train_loader, val_loader, test_loader, e_mean, e_std, ctx_mean, ctx_std = build_channels_plus_emptyvoxels_with_neighbors(
        ctx_manager=ctx_manager,
        ephys=ephys,
        probe_positions=probe_positions,
        RADIUS_UM=RADIUS_UM,
        M_MAX=M_MAX
    )

    handles = alignment_handles_from_loader(train_loader)

    # Model (reuse across calls)
    F_ctx = cfg.n_cell_pcs + cfg.n_gene_pcs
    F_e = ephys.shape[-1]
    F_REG = 0
    heteroscedastic = False
    model = NeighborInpaintingModel(
        f_ctx=F_ctx, f_ephys=F_e, f_out=F_e, f_region=F_REG,
        e_mean=e_mean, e_std=e_std, ctx_mean=ctx_mean, ctx_std=ctx_std,
        d_model=128, nhead=8, depth=2, neighbor_self_attn=False,
        heteroscedastic=heteroscedastic, drop=0.15
    ).to(device)
    model.load_state_dict(torch.load(local_path.joinpath(model_name, 'SE_model.pth'), map_location=device))
    model.eval()
    torch.set_grad_enabled(False)

    print(f"[Alignment engine ready] build time: {time.time() - t0:.2f}s")

    return AlignmentEngine(
        device=device, cfg=cfg, ctx_manager=ctx_manager, model=model,
        handles=handles, e_mean=e_mean, e_std=e_std, ctx_mean=ctx_mean, ctx_std=ctx_std,
        ephys=ephys, probe_positions=probe_positions, probe_planned_positions=probe_planned_positions,
        pid_str=pid_str, M_MAX=M_MAX, RADIUS_UM=RADIUS_UM, optimization_features=optimization_features
    )

def ensure_engine(controller: 'AlignmentGUIController') -> AlignmentEngine:
    plug = controller.plugins['Channel Prediction']
    if MODEL_NAME not in plug or plug[MODEL_NAME] is None:
        plug[MODEL_NAME] = load_alignment_engine(controller)
    return plug[MODEL_NAME]

def get_query_repr_and_pred(model, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask_nei):
    p_rel = p_n - p_q[:, None, :]
    h_q = model.qenc(ctx_q, reg_q, p_q)                    # [B,1,D]
    h_n = model.nenc(e_n, reg_n, p_n, p_rel, mask_nei)     # [B,M,D]
    for blk in model.blocks:
        h_q = blk(h_q, h_n, mask_nei)                      # [B,1,D]
    mu, logvar = model.pred(h_q)                           # [B,F], [B,F] or None
    return h_q.squeeze(1), mu, logvar

# ==== alignment algorithm utils ====
@torch.no_grad()
def predict_features_at_xyz(
    engine: AlignmentEngine,
    xyz: np.ndarray,      # [K,3] meters
    batch_size: int = 512,
    radius_um: float = 500.0,
    M_max: int = 8,
    device: Optional[str] = None,
):
    """
    Returns standardized predictions at given XYZ:
      mu_std: [K, F], logvar: [K, F] or None
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = engine.model.to(device).eval()

    # Build RAW context at query xyz
    pack = engine.ctx_manager.sample_context_numpy_m(xyz, mode='clip')
    ctx_raw = torch.from_numpy(np.concatenate([pack['cell_pc'], pack['gene_pc']], axis=1)).float()
    xyz_t = torch.from_numpy(xyz.copy()).float()

    C = xyz_t.shape[0]

    allen = torch.ones(C) * pack['allen_ix'][0]

    qds = GridDS(ctx_raw, allen, xyz_t, engine.e_mean.shape[-1])

    # Collate with neighbors
    collate = NeighborCollate(
        engine.ctx_manager,
        engine.handles["bank_xyz"], engine.handles["bank_feat"], engine.handles["bank_pid"], engine.handles["nn_bank"],
        e_feat_dim=engine.e_mean.shape[-1], M_max=M_max, radius_um=radius_um
    )

    dl = DataLoader(
        qds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=collate
    )

    mu_std, logvar_list = [], []
    for batch in dl:
        (ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask, has_ephys, y_e, vox_count, *_) = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                _, mu, logvar = get_query_repr_and_pred(model, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask)
        else:
            _, mu, logvar = get_query_repr_and_pred(model, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask)
        mu_std.append(mu.detach().float().cpu())
        logvar_list.append(None if logvar is None else logvar.detach().float().cpu())

    mu_std = torch.cat(mu_std, dim=0)
    logvar = None if logvar_list[0] is None else torch.cat(logvar_list, dim=0)
    return mu_std, logvar

def alignment_handles_from_loader(train_loader):
    """
    Extract the neighbor bank and full-channel lookup from NeighborCollateV2
    attached to the train_loader. Also returns ctx stats and ephys stats.
    """
    collate = train_loader.collate_fn

    # Neighbor bank (TRAIN-ONLY)
    bank_xyz = collate.bank_xyz
    bank_feat = collate.bank_feat
    bank_pid = collate.bank_pid
    nn_bank = collate.nn

    return dict(
        bank_xyz=bank_xyz, bank_feat=bank_feat, bank_pid=bank_pid, nn_bank=nn_bank,
    )

def build_cost_matrix(A, B, logvar, use_nll=False):
    """
    Compute cost matrix between actual features A and predicted features B.
    """
    if use_nll and (logvar is not None):
        sigma2 = np.exp(logvar.cpu().numpy().astype(np.float64)) # [M,F_all]
        C = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
        for j in range(B.shape[0]):
            diff2 = (A - B[j])**2
            C[:, j] = 0.5 * (diff2 / (sigma2[j][:B.shape[1]] + 1e-12) + np.log(sigma2[j][:B.shape[1]] + 1e-12)).sum(axis=1)
        return C

    # L2 in standardized space
    AA = np.sum(A*A, axis=1, keepdims=True)
    BB = np.sum(B*B, axis=1, keepdims=True).T
    AB = A @ B.T
    return (AA + BB - 2.0*AB).clip(min=0.0)

def soft_dynamic_time_warping(C, lam_d, lam_u, lam_l, band=None):
    """
    Soft Dynamic Time Warping (DTW) for aligning two sequences.

    Finds the optimal alignment path between two sequences by minimizing
    the cumulative cost, allowing for diagonal, upward, and leftward moves
    with different penalties.
    """
    N, M = C.shape

    # Initialize cumulative cost matrix (D) and path direction matrix (P)
    D = np.full((N, M), np.inf, dtype=np.float64)
    P = np.full((N, M), -1, dtype=np.int8)  # -1 = unvisited, 0 = diagonal, 1 = up, 2 = left

    # Initialize first row with base costs (no prior path)
    D[0, :] = C[0, :] if band is None else np.where(band[0, :], C[0, :], np.inf)

    for i in range(1, N):
        # Determine which columns to consider (all or band-constrained)
        j_iter = range(1, M) if band is None else np.where(band[i, :])[0]
        for j in j_iter:
            # Skip positions outside the band
            if band is not None and not band[i, j]:
                continue
            # Calculate costs for three possible moves:
            # 1. Diagonal: match i with j
            c_diag = D[i - 1, j - 1] + lam_d if j - 1 >= 0 else np.inf
            # 2. Up: skip j in second sequence (insertion)
            c_up = D[i - 1, j] + lam_u
            # 3. Left: skip i in first sequence (deletion)
            c_left = D[i, j - 1] + lam_l if j - 1 >= 0 else np.inf
            # Choose the move with the minimum cost
            if c_diag <= c_up and c_diag <= c_left:
                D[i, j] = C[i, j] + c_diag
                P[i, j] = 0
            elif c_up <= c_left:
                D[i, j] = C[i, j] + c_up
                P[i, j] = 1
            else:
                D[i, j] = C[i, j] + c_left
                P[i, j] = 2

    # Find the ending position with minimum cost in the last row
    j_end = int(np.argmin(D[N - 1, :]))
    total = float(D[N - 1, j_end])
    i, j = N - 1, j_end

    # Backtrack to find the optimal path
    path = [(i, j)]
    while i > 0:
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

    return path[0][1], j_end, path, total


def rigid_assignment(A, B):
    """Find the best rigid alignment using sliding window L2 matching."""

    best_k, best_mse = 0, np.inf
    Nr = A.shape[0]
    for k in range(0, B.shape[0] - Nr + 1):
        m = ((B[k:k + Nr] - A) ** 2).mean()
        if m < best_mse:
            best_mse, best_k = m, k
    j_start, j_end = best_k, best_k + Nr - 1
    path = [(i, best_k + i) for i in range(Nr)]

    return j_start, j_end, path


def xyz_to_region_ids(xyz_m, brain_atlas):
    """
    xyz_m: [C, 3] in meters. Convert to µm for atlas, return [C] region ids (int).
    """
    # brain_atlas.get_labels should broadcast over channels
    region_ids = brain_atlas.get_labels(xyz_m, mode='clip')

    # ensure 1D int array
    return np.asarray(region_ids).astype(int).reshape(-1)


def predict(controller, items):
    # -------------- one-time caches (lazy) --------------

    engine = ensure_engine(controller)

    # -------------- prediction pipeline (fast) --------------
    print("Ephys feature computation")
    # TODO: Currently flipping the probe since my model is trained on probes that goes from top to bottom,
    #  need to update the model and change that
    xyz_samples = items.model.align_handle.xyz_samples.copy()[::-1, :]
    pid = controller.model.shank_labels[0]['id']
    p_ind = np.where(pid == engine.pid_str)[0]
    if len(p_ind) == 0:
        print(f'Probe ID {pid} not found in ephys atlas database.')
        return

    # Get the ephys features recorded on the probe
    # These features are flipped
    recorded = engine.ephys[p_ind][0].astype(np.float32)

    # This would get the features that are used in the GUI for the current pid but they
    # are not flipped
    df = items.model.raw_data['features']['df']
    # TODO check if it is the same if we flip the recorded features
    # recorded = df[FEATURE_LIST].to_numpy()

    # Number of channels on the probe
    nc = recorded.shape[0]
    # Remove channels that were not recorded (all zeros)
    kp_mask = ~np.all(recorded == 0.0, axis=1)
    if kp_mask.sum() < 2:
        print('Need at least 2 recorded channels with non-zero features for spatial encoding.')
        return
    # Z score the recorded features so that they are in the same space as predicted features
    recorded = ((torch.from_numpy(recorded) - engine.e_mean.cpu()) / (engine.e_std.cpu() + 1e-8)).numpy().astype(np.float64)
    # Recorded features
    recorded = recorded[kp_mask][:, engine.optimization_features]

    # Get the predicted ephys features along the probe trajectory
    # Predict the features at all sample points along the probe
    predicted, logvar_full = predict_features_at_xyz(
        engine, xyz_samples, batch_size=512, radius_um=engine.RADIUS_UM, M_max=engine.M_MAX, device=engine.device
    )
    predicted = predicted.cpu().numpy().astype(np.float64)
    predicted = predicted[:, engine.optimization_features]

    # Compute cost matrix
    cost_matrix = build_cost_matrix(recorded, predicted, logvar_full)

    # Find path that minimizes cost with soft DTW
    j_start, j_end, path, total_cost = soft_dynamic_time_warping(cost_matrix, lam_d=0.0, lam_u=5.0, lam_l=5.0)

    # Ensure at least 90% of channels are used in the alignment, else use rigid assignment
    min_overlap_channels = int(0.9*nc)
    if (j_end - j_start + 1) < min_overlap_channels:
        j_start, j_end, path = rigid_assignment(recorded, predicted)

    # Convert path to arrays
    i_seq, j_seq = np.array(path, dtype=int).T

    # Initialize mapping (A index → B index)
    j_for_i = np.full(recorded.shape[0], np.nan)
    j_for_i[i_seq] = j_seq

    # Fill gaps (forward then backward)
    j_for_i = (
        pd.Series(j_for_i)
        .ffill()
        .bfill()
        .astype(int)
        .clip(0, predicted.shape[0] - 1)
        .to_numpy()
    )

    # Interpolate to get mapping for all channels
    j_map_all = np.interp(np.arange(nc), np.where(kp_mask)[0], j_for_i.astype(float))
    j_map_all_i = np.clip(np.round(j_map_all).astype(int), 0, predicted.shape[0] - 1)
    est_xyz = xyz_samples[j_map_all_i]

    # TODO: Currently flipping the probe since my model is trained on probes that goes from top to bottom,
    #  need to update the model and change that

    # TODO: Flipping the probe back for consistency
    trk = items.model.align_handle.ephysalign.sampling_trk.copy()[::-1]

    # Get the regions within the probe
    region_ids = xyz_to_region_ids(est_xyz, controller.model.brain_atlas)[::-1]
    depth_samples = df['axial_um'].to_numpy() / 1e6

    # Get the regions outside the probe
    # Above the probe (from top of probe to top of brain)
    region_ids_top = xyz_to_region_ids(xyz_samples[:j_start], controller.model.brain_atlas)[::-1]
    depths_top = (trk[:j_start] - trk[j_start] + depth_samples[-1])[::-1]

    # Below the probe (from bottom of probe to bottom of brain)
    region_ids_bottom =  xyz_to_region_ids(xyz_samples[j_end+1:], controller.model.brain_atlas)[::-1]
    depths_bottom = (trk[j_end+1:] - trk[j_end])[::-1]

    region_ids = np.concatenate([region_ids_bottom, region_ids, region_ids_top], axis=0)
    depth_samples = np.concatenate([depths_bottom, depth_samples, depths_top], axis=0)

    return region_ids, depth_samples