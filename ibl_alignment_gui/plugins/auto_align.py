import json, os, gc

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset

import ephysatlas.data
import ephysatlas.anatomy
import ephysatlas.fixtures

from ephysatlas.data import download_tables, read_features_from_disk

from dataclasses import asdict

from typing import Optional, Tuple

import numpy as np
from qtpy import QtWidgets

from iblutil.util import Bunch
from typing import TYPE_CHECKING

from iblatlas.genomics import agea, merfish
from iblatlas.atlas import AllenAtlas
from one.api import ONE

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from collections import defaultdict, Counter

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.shank_controller import ShankController
    from iblatlas.atlas import AllenAtlas

from ibl_alignment_gui.utils.utils import shank_loop

FEATURE_LIST = [
    'rms_lf', 'psd_lfp', 'psd_alpha', 'psd_beta', 'psd_gamma', 'psd_delta', 'psd_theta',
    'psd_lfp_csd', 'psd_alpha_csd', 'psd_beta_csd', 'psd_gamma_csd', 'psd_delta_csd', 'psd_theta_csd',
    'rms_lf_csd', 'psd_residual_lfp', 'psd_residual_alpha', 'psd_residual_beta', 'psd_residual_gamma', 'psd_residual_delta', 'psd_residual_theta',
    'decay_fit_error', 'decay_fit_r_squared', 'decay_n_peaks', 'aperiodic_exponent', 'aperiodic_offset',
    'cor_ratio', 'rms_ap', 'alpha_mean', 'alpha_std', 'spike_count',
    'tip_time_secs', 'recovery_time_secs', 'peak_time_secs', 'trough_time_secs',
    'trough_val', 'tip_val', 'peak_val',
    'recovery_slope', 'depolarisation_slope', 'repolarisation_slope', 'polarity',
     #'channel_labels'
]

PLUGIN_NAME = "Automatic Alignment"

SHANK_COLOURS = {
    '0': '#000000',
    '1': '#000000',
    'a': '#000000',
    'b': '#30B666',
    'c': '#ff0044',
    'd': '#0000ff'
}

def setup(controller: 'AlignmentGUIController') -> None:
    """
    Set up the Automatic Alignment plugin.

    Also attaches necessary callbacks to the controller.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = Bunch()

    auto_align_plugin = AutoAlign(controller)
    controller.plugins[PLUGIN_NAME]['loader'] = auto_align_plugin

    # Attach callbacks to methods in the controller
    controller.plugins[PLUGIN_NAME]['data_button_pressed'] = auto_align_plugin.data_button_pressed

    # Add a submenu to the main menu
    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, controller.view)
    controller.plugin_options.addMenu(plugin_menu)

    # Checkable menu option for using the automatic alignment
    align_checkbox = QtWidgets.QAction('Automatic Alignment', controller.view)
    align_checkbox.setCheckable(True)
    align_checkbox.setChecked(False)
    align_checkbox.toggled.connect(auto_align_plugin.predict_alignment)
    plugin_menu.addAction(align_checkbox)
    auto_align_plugin.predict_alignment = align_checkbox

class RecDS(Dataset):
    """Recorded voxels: (context, allen, xyz_m, ephys, pid, vox_count, has_ephys=True)."""
    def __init__(self, ctx, allen, xyz_m, ephys, pid, vox_count):
        self.ctx, self.allen, self.xyz = ctx, allen, xyz_m
        self.ephys, self.pid = ephys, pid
        self.vox_count = vox_count
        self.has = torch.ones(len(self.ctx), dtype=torch.bool)
    def __len__(self): return self.ctx.shape[0]
    def __getitem__(self, i):
        return (i, self.ctx[i], self.allen[i], self.xyz[i],
                self.ephys[i], self.pid[i], self.has[i], self.vox_count[i])

class GridDS(Dataset):
    """Grid-only voxels: (context, allen, xyz_m, empty ephys, pid=0, vox_count=1, has_ephys=False)."""
    def __init__(self, ctx, allen, xyz_m, f_e):
        self.ctx, self.allen, self.xyz = ctx, allen, xyz_m
        self._empty = torch.zeros(f_e, dtype=torch.float32)
        self._empty_pid = torch.tensor(0.0, dtype=torch.float32)   # scalar, not [1]
        self._count = torch.tensor(1.0, dtype=torch.float32)       # scalar, not [1]
        self.has = torch.zeros(len(self.ctx), dtype=torch.bool)
    def __len__(self): return self.ctx.shape[0]
    def __getitem__(self, i):
        return (i, self.ctx[i], self.allen[i], self.xyz[i],
                self._empty, self._empty_pid, self.has[i], self._count)

# ---------- small utilities ----------
def downsample_keys_from_xyz(ctx_manager, xyz_m, ds_rate=8):
    Xh, Zh, Yh = ctx_manager.cell_pca.shape[1:]
    ijk = ctx_manager.bc.xyz2i(xyz_m, mode='clip')
    xi = np.clip(np.round(ijk[:,0] / ds_rate).astype(int), 0, Xh-1)
    yi = np.clip(np.round(ijk[:,1] / ds_rate).astype(int), 0, Yh-1)
    zi = np.clip(np.round(ijk[:,2] / ds_rate).astype(int), 0, Zh-1)
    return xi, zi, yi

def compute_voxel_with_ephys(ctx_manager, probe_positions, xi, yi, zi):
    N = xi.size
    if(len(probe_positions.shape) == 2):
        P, _ = probe_positions.shape
        ch_xyz = probe_positions
    else:
        P, C, _ = probe_positions.shape
        ch_xyz = probe_positions.reshape(P*C, 3)

    xic, zic, yic = downsample_keys_from_xyz(ctx_manager, ch_xyz)
    has = np.zeros(N, dtype=bool)

    # Map grid tuple -> flat index
    key2flat = { (int(xi[i]), int(zi[i]), int(yi[i])): i for i in range(N) }

    for x, z, y in zip(xic, zic, yic):
        if (x,z,y) in key2flat:
            has[key2flat[(x,z,y)]] = True
    return has

# ---------- KDTree neighbors (train bank) ----------
try:
    from sklearn.neighbors import KDTree
    _HAS_KDT = True
except Exception:
    _HAS_KDT = False

class ChannelNN:
    def __init__(self, ch_xyz_m: np.ndarray):
        self.X = ch_xyz_m.astype(np.float64)
        self.tree = KDTree(self.X, leaf_size=40) if (self.X.shape[0] and _HAS_KDT) else None
    def query_radius(self, q_xyz_m: np.ndarray, r_m: float, k_cap: int = 8):
        if self.tree is not None:
            inds, _ = self.tree.query_radius(q_xyz_m, r=r_m, return_distance=True, sort_results=True)
            return [ii[:k_cap] for ii in inds]
        # brute force
        out = []
        X = self.X
        for q in q_xyz_m:
            if X.shape[0]==0: out.append(np.array([], dtype=int)); continue
            d2 = np.sum((X - q[None,:])**2, axis=1)
            I = np.where(d2 <= (r_m**2))[0]
            if I.size > 8:
                J = np.argpartition(d2[I], 8)[:8]
                I = I[J]
            out.append(I)
        return out

# ---------- collate that injects neighbors ----------
class NeighborCollate:
    """
    Takes per-sample (idx, ctx, allen, xyz_m, ephys, has_ephys) and adds:
      - e_n [B,M,F_e], p_n [B,M,3], mask [B,M]
      - y_e [B,F_e] from dataset
    Uses a TRAIN-ONLY neighbor bank and excludes same-probe neighbors for recorded voxels.
    Assumes inputs are already standardized.
    """
    def __init__(self,
                 ctx_manager,
                 bank_xyz_m, bank_feat_stdzd, bank_pid, kdtree_bank,
                 e_feat_dim: int,
                 M_max=64, radius_um=600.0, allow_same_probe=False):
        self.ctx_manager = ctx_manager
        self.bank_xyz  = bank_xyz_m
        self.bank_feat = bank_feat_stdzd
        self.bank_pid  = bank_pid
        self.nn        = kdtree_bank
        self.F_e       = int(e_feat_dim)
        self.M         = int(M_max)
        self.r_m       = float(radius_um) * 1e-6
        self.F_reg     = 0
        self.allow_same_probe = allow_same_probe
    def __call__(self, batch_items):
        # unpack
        (idxs, ctxs, allens, xyzs, ephys, pids, has, counts) = zip(*batch_items)

        B = len(idxs)
        ctx_q  = torch.stack(ctxs,   dim=0)        # [B,F_ctx] (already standardized)
        allen  = torch.stack(allens, dim=0)        # [B]
        p_q    = torch.stack(xyzs,   dim=0)        # [B,3] m
        y_e    = torch.stack([
                   t if t.numel() else torch.zeros(self.F_e, dtype=torch.float32)
                 for t in ephys], dim=0)           # [B,F_e]
        has_ephys = torch.stack(has, dim=0).bool() # [B]
        vox_count = torch.stack(counts, dim=0).float().clamp_min(1.0)  # [B]

        # placeholders
        e_n   = torch.zeros(B, self.M, self.F_e, dtype=torch.float32)
        reg_q = torch.zeros(B, self.F_reg, dtype=torch.float32)
        reg_n = torch.zeros(B, self.M, self.F_reg, dtype=torch.float32)
        p_n   = torch.zeros(B, self.M, 3, dtype=torch.float32)
        mask  = torch.zeros(B, self.M, dtype=torch.bool)

        # voxel keys for exclusion / target lookup
        xi, zi, yi = downsample_keys_from_xyz(self.ctx_manager, p_q.numpy())

        # neighbor candidates from train bank
        neigh_lists = self.nn.query_radius(p_q.numpy(), r_m=self.r_m, k_cap=8*self.M)

        for b in range(B):
            key = (xi[b], zi[b], yi[b])

            # Exclude same-probe neighbors for recorded voxels
            exclude_pids = set()
            if has_ephys[b] and self.allow_same_probe == False:
                exclude_pids = {pids[b].item()}

            # build neighbor set
            cand = [ci for ci in neigh_lists[b] if int(self.bank_pid[ci]) not in exclude_pids]
            L = len(cand)

            if L > self.M:
                # random subset, not just first-M
                sel = np.random.choice(cand, size=self.M, replace=False)
                cand = sel.tolist()

            # neighbor dropout
            keep = []
            for ci in cand:
                if np.random.rand() > 0.3:
                    keep.append(ci)
            cand = keep

            L = len(cand)

            if L > 0:
                e_n[b, :L] = torch.from_numpy(self.bank_feat[cand])
                p_n[b, :L] = torch.from_numpy(self.bank_xyz[cand])
                mask[b, :L] = True

            # If recorded voxel but dataset provided empty ephys (edge), fall back to full map mean
            if has_ephys[b] and (y_e[b].abs().sum() == 0):
                chs_here = self.vox2chs.get(key, [])
                if chs_here:
                    y_e[b] = torch.from_numpy(self.bank_feat[chs_here].mean(axis=0))

        batch = (ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask, has_ephys, y_e, vox_count, allen, pids)
        return batch

# ---------- channel catalog (good probes only) ----------
def build_channel_catalog(ephys_np: np.ndarray, probe_xyz_np: np.ndarray):
    """
    ephys_np: [P, C, F], probe_xyz_np: [P, C, 3], good_idx: [Pg]
    Returns flat arrays:
      ch_xyz: [Nch,3] (meters), ch_feat: [Nch,F], ch_pid: [Nch] int
    Filters out channels whose xyz are all-zero.
    """
    feats, xyzs, pids = [], [], []
    for p in range(probe_xyz_np.shape[0]):
        xyz = probe_xyz_np[p]          # [C,3]
        ef  = ephys_np[p]              # [C,F]
        valid = ~(np.all(xyz == 0.0, axis=1))
        if not valid.any():
            continue
        xyzs.append(xyz[valid])
        feats.append(ef[valid])
        pids.append(np.full(valid.sum(), p, dtype=np.int32))

    if len(xyzs) == 0:
        return (np.zeros((0,3), np.float32),
                np.zeros((0, ephys_np.shape[-1]), np.float32),
                np.zeros((0,), np.int32))
    ch_xyz  = np.concatenate(xyzs, axis=0).astype(np.float32)
    ch_feat = np.concatenate(feats, axis=0).astype(np.float32)
    ch_pid  = np.concatenate(pids,  axis=0).astype(np.int32)
    return ch_xyz, ch_feat, ch_pid

# ---------- voxel↔channels index ----------
def build_voxel_to_channels(ctx_manager, ch_xyz_m: np.ndarray):
    """
    Map each rounded atlas voxel (xi,zi,yi) to list of channel indices.
    """
    from collections import defaultdict

    if ch_xyz_m.shape[0] == 0:
        return defaultdict(list)
    ijk = ctx_manager.bc.xyz2i(ch_xyz_m) # float indices (x,y,z)
    Xh, Zh, Yh = ctx_manager.cell_pca.shape[1:]
    xi = np.clip(np.round(ijk[:,0] / 8).astype(int), 0, Xh-1)
    yi = np.clip(np.round(ijk[:,1] / 8).astype(int), 0, Yh-1)
    zi = np.clip(np.round(ijk[:,2] / 8).astype(int), 0, Zh-1)
    vox2chs = defaultdict(list)
    for ci,(x, z, y) in enumerate(zip(xi, zi, yi)):
        vox2chs[(x,z,y)].append(ci)
    return vox2chs

# ---------- KDTree (optional, falls back to numpy) ----------
try:
    from sklearn.neighbors import KDTree
    _HAS_KDT = True
except Exception:
    _HAS_KDT = False

class ChannelNN:
    def __init__(self, ch_xyz_m: np.ndarray):
        self.X = ch_xyz_m.astype(np.float64)
        if self.X.shape[0] and _HAS_KDT:
            self.tree = KDTree(self.X, leaf_size=40)
        else:
            self.tree = None
    def query_radius(self, q_xyz_m: np.ndarray, r_m: float, k_cap: int = 8):
        if self.tree is not None:
            inds, _ = self.tree.query_radius(q_xyz_m, r=r_m, return_distance=True, sort_results=True)
            # cap K to avoid gigantic neighbor sets:
            capped = [ii[:k_cap] for ii in inds]
            return capped
        # fallback: brute force (batch)
        out = []
        X = self.X
        for q in q_xyz_m:
            if X.shape[0]==0:
                out.append(np.array([], dtype=int)); continue
            d2 = np.sum((X - q[None,:])**2, axis=1)
            I = np.where(d2 <= (r_m**2))[0]
            if I.size > k_cap:
                # keep closest k_cap
                J = np.argpartition(d2[I], k_cap)[:k_cap]
                I = I[J]
            out.append(I)
        return out

# ---------- dataset wrapper so collate knows index ----------
class IndexedVoxelDataset(Dataset):
    def __init__(self, base_ds: torch.utils.data.TensorDataset):
        assert len(base_ds.tensors) == 4
        self.base = base_ds
    def __len__(self): return len(self.base.tensors[0])
    def __getitem__(self, idx):
        context, allen_ix, xyz_m, has_ephys = (t[idx] for t in self.base.tensors)
        return idx, context, allen_ix, xyz_m, has_ephys

class AtlasPCAConfig:
    n_cell_pcs: int = 50  # 338
    n_gene_pcs: int = 50  # 4345
    batch_size_vox: int = 2_000_000
    blur_features: bool = False
    no_pca: bool = False

class ContextAtlasManager:
    """
    Handles loading raw atlases, computing PCs, caching transforms, sampling,
    and now: saving/loading precomputed context to/from disk.
    """

    def __init__(self, cfg: AtlasPCAConfig, regenerate_context: bool = False):
        brain_atlas = AllenAtlas()
        self.bc = brain_atlas.bc
        self.cfg = cfg

        # Save scales for persistence and for BCProxy on reload
        self.xscale = np.asarray(self.bc.xscale)  # meters
        self.yscale = np.asarray(self.bc.yscale)
        self.zscale = np.asarray(self.bc.zscale)

        # -------- Load brain regions --------
        Allen_regions = brain_atlas._get_mapping(mapping='Allen')[brain_atlas.label]
        np.save('ibl_alignment_gui/plugins/auto_align_data/Allen_regions.npy', Allen_regions)

        suffix = '_blur' if cfg.blur_features else ''

        if regenerate_context:
            # -------- AGEA → PCA --------
            df_genes, gene_vols, atlas = agea.load(label='processed')  # [G, Xc, Zc, Yc]
            if cfg.blur_features:
                gene_vols = apply_3d_gaussian_blur(torch.tensor(gene_vols, device='cuda'),
                                                   kernel_size=15, sigma=3).cpu().numpy()
            size_x, size_z, size_y = gene_vols.shape[1:]
            xgenes = gene_vols.reshape(gene_vols.shape[0], -1).T.astype(np.float32)
            scaler = StandardScaler().fit(xgenes)
            xgenes = scaler.transform(xgenes)
            pca = PCA(n_components=cfg.n_gene_pcs).fit(xgenes)
            X_pca = pca.transform(xgenes).reshape(size_x, size_z, size_y, cfg.n_gene_pcs)
            gene_exp_vol = np.moveaxis(X_pca, -1, 0).astype(np.float32)  # [P_gene, Xc, Zc, Yc]

            # -------- MERFISH → PCA --------
            merfish.load()
            LEVEL = 'subclass'
            path = AllenAtlas._get_cache_dir().joinpath('merfish')
            cell_type_vol = torch.tensor(np.load(path.joinpath(f'merfish_{LEVEL}.npy')))
            zero_ind = torch.where(cell_type_vol.sum(dim=0) == 0)
            if cfg.blur_features:
                cell_type_vol = apply_3d_gaussian_blur(cell_type_vol.to(device='cuda'),
                                                       kernel_size=9, sigma=3.0).detach().cpu()
            cell_type_vol = cell_type_vol.numpy()
            size_x, size_z, size_y = cell_type_vol.shape[1:]

            xcells = cell_type_vol.reshape(cell_type_vol.shape[0], -1).T.astype(np.float32)
            scaler = StandardScaler().fit(xcells)
            xcells = scaler.transform(xcells)
            pca = PCA(n_components=cfg.n_cell_pcs).fit(xcells)
            X_pca = pca.transform(xcells).reshape(size_x, size_z, size_y, cfg.n_cell_pcs)
            cell_type_vol = np.moveaxis(X_pca, -1, 0).astype(np.float32)  # [P_cell, Xc, Zc, Yc]

            # zero out empty MERFISH sites in both modalities
            cell_type_vol[:, zero_ind[0], zero_ind[1], zero_ind[2]] = 0
            gene_exp_vol[:, zero_ind[0], zero_ind[1], zero_ind[2]] = 0

            np.save(f'auto_align_data/agea_vol_pca{suffix}', gene_exp_vol)
            np.save(f'auto_align_data/merfish_vol_pca{suffix}', cell_type_vol)
        else:
            gene_exp_vol = np.load(f'ibl_alignment_gui/plugins/auto_align_data/agea_vol_pca{suffix}.npy')
            cell_type_vol = np.load(f'ibl_alignment_gui/plugins/auto_align_data/merfish_vol_pca{suffix}.npy')

        self.cell_pca = cell_type_vol  # [P_cell, Xh, Zh, Yh]
        self.gene_pca = gene_exp_vol  # [P_gene, Xh, Zh, Yh]
        self.allen_idx = Allen_regions  # [Yh, Xh, Zh]
        assert self.cell_pca.ndim == 4 and self.gene_pca.ndim == 4
        assert self.allen_idx.ndim == 3
    def get_allen_ind(self):
        return np.unique(self.allen_idx)

    def get_brain_dim(self):
        y_dim, x_dim, z_dim = self.allen_idx.shape
        return (x_dim, y_dim, z_dim)

    def sample_context_numpy_m(self, xyz_m: np.ndarray, mode='raise'):
        """
        Sample at world positions (meters). Uses bc.xyz2i -> coarsen to your 8-voxel grid.
        """
        indices = self.bc.xyz2i(xyz_m, mode=mode)  # fractional high-res indices

        # Coarsen to your [Xh, Yh, Zh] grid (original code divides by 8 and rounds)
        Yh, Xh, Zh = self.allen_idx.shape
        xi = np.clip((np.round(indices[:, 0] / 8)).astype(int), 0, Xh - 1)
        yi = np.clip((np.round(indices[:, 1] / 8)).astype(int), 0, Yh - 1)
        zi = np.clip((np.round(indices[:, 2] / 8)).astype(int), 0, Zh - 1)

        cell_pc = self.cell_pca[:, xi, zi, yi].T.astype(np.float32)
        gene_pc = self.gene_pca[:, xi, zi, yi].T.astype(np.float32)
        return {
            'cell_pc': cell_pc,
            'gene_pc': gene_pc,
            'allen_ix': self.allen_idx[yi, xi, zi].astype(np.int32),
        }

    def sample_context_numpy_i(self, xyz_i: np.ndarray, s_xyz: np.ndarray = np.array([8, 8, 8])):
        cell_pc = self.cell_pca[:, xyz_i[:, 0], xyz_i[:, 2], xyz_i[:, 1]].T.astype(np.float32)
        gene_pc = self.gene_pca[:, xyz_i[:, 0], xyz_i[:, 2], xyz_i[:, 1]].T.astype(np.float32)

        iy = np.clip(xyz_i[:, 1] * s_xyz[1], 0, len(self.yscale) - 1)
        ix = np.clip(xyz_i[:, 0] * s_xyz[0], 0, len(self.xscale) - 1)
        iz = np.clip(xyz_i[:, 2] * s_xyz[2], 0, len(self.zscale) - 1)

        return {
            'cell_pc': cell_pc,
            'gene_pc': gene_pc,
            'allen_ix': self.allen_idx[iy, ix, iz].astype(np.int32),
        }

def mlp(d_in, d_hidden, d_out, n_layers=2, drop=0.0):
    layers = [nn.Linear(d_in, d_hidden), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(drop)]
    layers += [nn.Linear(d_hidden, d_out)]
    return nn.Sequential(*layers)

class PosEnc3D(nn.Module):
    """Encode absolute and relative 3D positions."""
    def __init__(self, d_out):
        super().__init__()
        self.pe = mlp(6, max(64, d_out), d_out)  # [xyz_abs(3), xyz_rel(3)] -> d_out

    def forward(self, p_abs, p_rel):
        # p_abs, p_rel: [B, M, 3] (for neighbors) or [B, 1, 3] (for query)
        x = torch.cat([p_abs, p_rel], dim=-1)
        return self.pe(x)

class NeighborEncoder(nn.Module):
    def __init__(self, f_ephys, f_region, d_model, d_pos=64, drop=0.1):
        super().__init__()
        self.pos = PosEnc3D(d_pos)
        self.embed = mlp(f_ephys + f_region + d_pos, d_model, d_model, n_layers=2, drop=drop)

    def forward(self, e_n, reg_n, p_n_abs, p_n_rel, mask):  # e_n: [B,M,Fe]
        pos = self.pos(p_n_abs, p_n_rel)                     # [B,M,d_pos]
        x = torch.cat([e_n, reg_n, pos], dim=-1)
        h = self.embed(x)                                    # [B,M,d_model]
        h = h * mask[..., None]                              # zero out pads
        return h

class QueryEncoder(nn.Module):
    def __init__(self, f_ctx, f_region, d_model, d_pos=64, drop=0.1):
        super().__init__()
        self.pos = PosEnc3D(d_pos)
        self.embed = mlp(f_ctx + f_region + d_pos, d_model, d_model, n_layers=2, drop=drop)

    def forward(self, ctx_q, reg_q, p_q_abs):
        # broadcast rel=0 for the query token
        B = ctx_q.size(0)
        p_rel0 = torch.zeros(B, 1, 3, device=ctx_q.device, dtype=ctx_q.dtype)
        p_abs = p_q_abs[:, None, :]
        pos = self.pos(p_abs, p_rel0)                        # [B,1,d_pos]
        x = torch.cat([ctx_q[:, None, :], reg_q[:, None, :], pos], dim=-1)
        h = self.embed(x)                                    # [B,1,d_model]
        return h

class CrossBlock(nn.Module):
    """Optional neighbor self-attn, then query->neighbor cross-attn."""
    def __init__(self, d_model, nhead=8, drop=0.1, neighbor_self_attn=True):
        super().__init__()
        self.nei_self = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=drop, batch_first=True
        ) if neighbor_self_attn else None
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=drop, batch_first=True)
        self.ff = mlp(d_model, 4*d_model, d_model, n_layers=2, drop=drop)
        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_q2 = nn.LayerNorm(d_model)

    def forward(self, h_q, h_n, mask_nei):
        # h_q: [B,1,D], h_n: [B,M,D], mask_nei: [B,M] (True=real, False=pad)
        B, M, D = h_n.shape
        no_nei = ~mask_nei.any(dim=1)          # [B]
        if no_nei.any():
            # append a dummy zero neighbor and mark it valid ONLY for empty rows
            dummy = h_n.new_zeros(B, 1, D)
            h_n = torch.cat([h_n, dummy], dim=1)               # [B, M+1, D]
            pad = mask_nei.new_zeros(B, 1)
            pad[no_nei, 0] = True
            mask_nei = torch.cat([mask_nei, pad], dim=1)       # [B, M+1]

        # neighbor self-attn (optional)
        if self.nei_self is not None:
            key_padding_mask = (~mask_nei.bool())              # True=ignore
            h_n = self.nei_self(h_n, src_key_padding_mask=key_padding_mask)

        # query <- neighbors cross-attn
        key_padding_mask = (~mask_nei.bool())
        h_q2, _ = self.cross(h_q, h_n, h_n, key_padding_mask=key_padding_mask)
        h_q = self.norm_q1(h_q + h_q2)
        h_q = self.norm_q2(h_q + self.ff(h_q))
        return h_q

class EphysPredictor(nn.Module):
    def __init__(self, d_model, f_out, heteroscedastic=True):
        super().__init__()
        self.mu_head = mlp(d_model, 2*d_model, f_out, n_layers=2, drop=0.0)
        self.het = heteroscedastic
        if heteroscedastic:
            self.logvar_head = mlp(d_model, 2*d_model, f_out, n_layers=2, drop=0.0)

    def forward(self, h_q):  # [B,1,D]
        h = h_q.squeeze(1)   # [B,D]
        mu = self.mu_head(h)
        if self.het:
            logvar = self.logvar_head(h).clamp(-6.0, 4.0)
            return mu, logvar
        return mu, None

class NeighborInpaintingModel(nn.Module):
    """
    Predict ephys for a single query channel using context of that channel
    and a variable-size set of neighbor ephys from *other* probes.
    """
    def __init__(self, f_ctx, f_ephys, f_region, f_out, e_mean=None, e_std=None, ctx_mean=None, ctx_std=None,
                 d_model=256, nhead=8, depth=2, neighbor_self_attn=True, heteroscedastic=True, drop=0.1):
        super().__init__()
        self.qenc = QueryEncoder(f_ctx, f_region, d_model, drop=drop)
        self.nenc = NeighborEncoder(f_ephys, f_region, d_model, drop=drop)
        self.blocks = nn.ModuleList([
            CrossBlock(d_model, nhead=nhead, drop=drop, neighbor_self_attn=neighbor_self_attn)
            for _ in range(depth)
        ])
        self.pred = EphysPredictor(d_model, f_out, heteroscedastic=heteroscedastic)

        if(e_mean is not None and e_std is not None and ctx_mean is not None and ctx_std is not None):
            # register as buffers so they're saved/loaded with state_dict
            self.register_buffer("e_mean", e_mean.clone().detach())
            self.register_buffer("e_std",  e_std.clone().detach())
            self.register_buffer("ctx_mean", ctx_mean.clone().detach())
            self.register_buffer("ctx_std", ctx_std.clone().detach())

    def forward(self, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask_nei):
        """
        ctx_q: [B, F_ctx]
        reg_q: [B, F_reg]   (e.g., one-hot Allen/Cosmos or learned emb)
        p_q:   [B, 3]       absolute (voxel/world) coords
        e_n:   [B, M, F_e]  neighbor ephys
        reg_n: [B, M, F_reg]
        p_n:   [B, M, 3]    neighbor absolute coords
        mask_nei: [B, M]    1 = real neighbor, 0 = pad
        """
        # relative pos of neighbors to query
        p_q_b = p_q[:, None, :].expand_as(p_n)
        p_rel = p_n - p_q_b

        h_q = self.qenc(ctx_q, reg_q, p_q)                   # [B,1,D]
        h_n = self.nenc(e_n, reg_n, p_n, p_rel, mask_nei)    # [B,M,D]

        for blk in self.blocks:
            h_q = blk(h_q, h_n, mask_nei)

        mu, logvar = self.pred(h_q)                          # [B,F_out], [B,F_out] or None
        return mu, logvar

def get_query_repr_and_pred(model, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask_nei):
    p_rel = p_n - p_q[:, None, :]
    h_q = model.qenc(ctx_q, reg_q, p_q)                    # [B,1,D]
    h_n = model.nenc(e_n, reg_n, p_n, p_rel, mask_nei)     # [B,M,D]
    for blk in model.blocks:
        h_q = blk(h_q, h_n, mask_nei)                      # [B,1,D]
    mu, logvar = model.pred(h_q)                           # [B,F], [B,F] or None
    return h_q.squeeze(1), mu, logvar

class AutoAlign:
    """
    A class to manage viewing the automatic alignment visualizations in Urchin viewer.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.

    Attributes
    ----------
    controller: AlignmentGUIController
        The main application controller.
    align_checkbox: QtWidgets.QCheckbox or None
        Visualize the predicted alignment.
    ba: AllenAtlas
        A brain atlas instance.
    """

    def __init__(self, controller: 'AlignmentGUIController'):
        self.controller = controller

        # Initialize variables
        self.align_checkbox: QtWidgets.QCheckBox | None = None
        self.ba: AllenAtlas = self.controller.model.brain_atlas

    def data_button_pressed(self) -> None:
        """
        Prepare NeigborCollate function and VoxelDataset.

        Called when the data button is pressed in the main application.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = AtlasPCAConfig()
        print("Generating context grid")
        ctx_manager = ContextAtlasManager(cfg, regenerate_context=False)

        # Load ephys & probe positions and filter indices of MISALIGNED or bad probes
        print("Loading insertion data")
        pid_str, ephys, probe_positions, probe_planned_positions = LoadInsertionData()

        # Build the 200 µm dataset & splits
        M_MAX = 8
        RADIUS_UM = 500

        train_loader, val_loader, test_loader, e_mean, e_std, ctx_mean, ctx_std = build_channels_plus_emptyvoxels_with_neighbors(
            ctx_manager=ctx_manager,
            ephys=ephys,
            probe_positions=probe_positions,
            RADIUS_UM=RADIUS_UM,
            M_MAX=M_MAX)

        # Instantiate model
        F_ctx = cfg.n_cell_pcs + cfg.n_gene_pcs
        F_e = ephys.shape[-1]
        F_REG = 0
        heteroscedastic = False

        self.model = NeighborInpaintingModel(
            f_ctx=F_ctx, f_ephys=F_e, f_out=F_e, f_region=F_REG, e_mean=e_mean, e_std=e_std, ctx_mean=ctx_mean,
            ctx_std=ctx_std,
            d_model=128, nhead=8, depth=2, neighbor_self_attn=False, heteroscedastic=heteroscedastic, drop=0.15
        ).to(self.device)
        self.model.load_state_dict(torch.load('ibl_alignment_gui/plugins/auto_align_data/SE_model.pth'))

        # #TODO: Get estimated xyz from controller
        # # If we already have a fit, skip this part
        # est_xyz = self.controller.model.loaders['align'].get_starting_alignment(0)
        #
        # # Fit linear line through the estimated xyz positions
        # p_fit, v_fit = self.fit_line_least_squares(est_xyz)
        #
        # bc = ctx_manager.bc
        # bounds_m = (
        #     (float(bc.xscale.min()), float(bc.xscale.max())),
        #     (float(bc.yscale.min()), float(bc.yscale.max())),
        #     (float(bc.zscale.min()), float(bc.zscale.max())),
        # )
        # t_min, t_max = self.line_aabb_intersection(p_fit, v_fit, bounds_m)
        #
        # seg_P0 = p_fit + t_min * v_fit
        # seg_P1 = p_fit + t_max * v_fit
        # seg_len_m = float(np.linalg.norm(seg_P1 - seg_P0))
        # v_unit = v_fit / (np.linalg.norm(v_fit) + 1e-20)
        #
        # # Build relative positions along true probe (for pitch estimate only)
        # r_rel, L_ref = self.build_rel_spacing_from_true(est_xyz, p_fit, v_fit)  # r_rel in [0,1] on valid channels
        #
        # # TODO: Get the recorded ephys
        # #self.recorded_ephys_probe = get.......
        # self.C = int(self.recorded_ephys_probe.shape[0])
        #
        # # Sample the FULL top-to-bottom line at (approx) channel pitch to get M points
        # M = int(np.clip(np.ceil(seg_len_m / L_ref), self.C, 4096))  # cap to avoid extreme M
        # ts = np.linspace(0.0, seg_len_m, M, dtype=np.float32)
        # self.xyz_full = (seg_P0[None, :] + ts[:, None] * v_unit[None, :]).astype(np.float32)   # [M,3] meters

    def predict_alignment(self, radius_um=500, M_max=8, lambda_diag=0.0, lambda_vert=50.0, lambda_horz=0.5,
                          min_overlap_channels=350):
        # Predict features along the FULL line (standardized space)
        mu_std_full, logvar_full = self.predict_features_at_xyz(
            self.model, self.ctx_manager, self.dataset, self.nn_bank, self.bank_xyz, self.bank_feat, self.bank_pid,
            self.vox2chs_all, self.full_feat, self.full_pid, self.model.e_mean, self.model.e_std, self.model.ctx_stats,
            self.xyz_full, batch_size=512, radius_um=radius_um, M_max=M_max, device=self.device
        )  # mu_std_full: [M, F], logvar_full: [M, F] or None
        B_std = mu_std_full.cpu().numpy().astype(np.float64)  # [M, F]

        # Prepare recorded features in standardized space
        Y_rec = self.recorded_ephys_probe.astype(np.float32)  # [C, F] original
        ch_mask = ~(np.all(Y_rec == 0.0, axis=1))  # True where we have data
        rec_idx = np.where(ch_mask)[0]

        Y_std = ((torch.from_numpy(Y_rec) - self.model.e_mean.cpu()) / (self.model.e_std.cpu() + 1e-8)).numpy().astype(np.float64)  # [C, F]
        A_std = Y_std  # [N, F] where N = number of recorded channels

        Cmat = self.build_cost_matrix(A_std, B_std, logvar_full, use_nll_cost=True)  # [N,M]
        N, M = Cmat.shape
        j_start, j_end, path, total_cost = self.sdtw_subsequence(
            Cmat, lambda_diag, lambda_vert, lambda_horz
        )

        # Guard: ensure sufficient overlap
        if (j_end - j_start + 1) < min_overlap_channels:
            # Fallback: maximize sliding window MSE as in your old code (simple, robust)
            # This keeps the function from failing; you can choose to raise instead.
            best_k = 0
            best_mse = np.inf
            Nr = N
            for k in range(0, M - Nr + 1):
                m = ((B_std[k:k + Nr] - A_std) ** 2).mean()
                if m < best_mse:
                    best_mse = m
                    best_k = k
            j_start, j_end = best_k, best_k + Nr - 1
            path = [(i, best_k + i) for i in range(N)]
            total_cost = best_mse * N
            print("Fallback to maximizing sliding window MSE")
        else:
            print("Path length: {}".format(len(path)))

        # path: list[(i,j)] from DTW backtrack, i in [0..N-1] for recorded-index order
        i_seq = np.array([ij[0] for ij in path], dtype=int)
        j_seq = np.array([ij[1] for ij in path], dtype=int)

        # For each i, take the last j visited at that i (so we advance when i increments)
        # This yields exactly N pairs (one per recorded index)
        j_for_i = np.full((A_std.shape[0],), np.nan)  # N
        for ii, jj in zip(i_seq, j_seq):
            j_for_i[ii] = jj
        # Safety: forward/backward fill if any i were skipped (shouldn’t happen, but guard)
        for ii in range(1, j_for_i.shape[0]):
            if np.isnan(j_for_i[ii]):
                j_for_i[ii] = j_for_i[ii - 1]
        for ii in range(j_for_i.shape[0] - 2, -1, -1):
            if np.isnan(j_for_i[ii]):
                j_for_i[ii] = j_for_i[ii + 1]

        j_for_i = np.clip(j_for_i.astype(int), 0, M - 1)  # length N

        # Now interpolate to ALL C channels using absolute channel indices rec_idx
        j_map_all = np.interp(np.arange(self.C), rec_idx, j_for_i.astype(float))  # float
        j_map_all_i = np.clip(np.round(j_map_all).astype(int), 0, M - 1)
        est_xyz = self.xyz_full[j_map_all_i]  # [C,3] meters

        # Predictions for ALIGNED placement to compute visuals & metrics
        # Build predicted features at the aligned C indices for visual parity w/ your previous code
        mu_std_aligned = B_std[j_map_all_i]  # [C, F] standardized
        mu_aligned = (mu_std_aligned * (self.model.e_std.cpu().numpy() + 1e-8) + self.model.e_mean.cpu().numpy())  # original units

        # Overlap window for recorded channels inside aligned array (place recorded where they map)
        rec_win_idx = rec_idx
        pred_win_idx = j_map_all_i[rec_idx]  # the corresponding positions in the full line for those channels

        # TODO: Send to the controller the final alignment and update plots

def LoadInsertionData(
    raw_date: bool = False,
    project: str = 'ea_active',
    agg: str = 'agg_full',
    VINTAGE: str = '2025_W43',
):
    """
    Loads table-based ephys features and concatenates per-channel averaged waveform latents
    assigned by nearest channel in xyz for each probe.

    Returns:
      unique_pids, context [N,384,(cell_pc+gene_pc)], allen_ix [N,384],
      ephys_concat [N,384,F+L], probe_positions [N,384,3], probe_planned_positions [N,384,3], filter_indices
    """

    print("Loading ephys features")
    if raw_date:
        df_features = pd.read_parquet('../ephys-atlas-decoding/features/2025_W27/raw_ephys_features.pqt')
        channels = pd.read_parquet('../ephys-atlas-decoding/features/2025_W27/channels.pqt')
    else:
        one = ONE(base_url='https://alyx.internationalbrainlab.org')
        path_data = Path('../ephys-atlas-decoding/features')
        path_data = download_tables(path_data, label=VINTAGE, project=project, one=one, agg_level=agg)
        df_features = read_features_from_disk(path_data, strict=False)

    # Pre-allocate containers
    probe_positions = []
    probe_planned_positions = []
    ephys_per_probe = []

    # Iterate probes
    for pid, df_pid in df_features.groupby(level='pid'):
        pid_str = str(pid)

        # --- Prepare channel xyz (actual + planned), preserving your up->down reversal ---
        xyz = np.zeros((384, 3), dtype=np.float32)
        xyz_planned = np.zeros((384, 3), dtype=np.float32)

        if raw_date:
            channel_indices = channels.loc[pid].index.get_level_values('channel').to_numpy()
            xyz_values = channels.loc[pid][['x', 'y', 'z']].values
            xyz_planned_values = channels.loc[pid][['x_target', 'y_target', 'z_target']].values
        else:
            channel_indices = df_pid.index.get_level_values('channel').to_numpy()
            xyz_values = df_pid[['x', 'y', 'z']].values
            xyz_planned_values = df_pid[['x_target', 'y_target', 'z_target']].values

        # Reverse order to be up -> down (same as your existing code)
        xyz[channel_indices] = xyz_values[::-1, :].copy()
        xyz_planned[channel_indices] = xyz_planned_values[::-1, :].copy()

        probe_positions.append(xyz)
        probe_planned_positions.append(xyz_planned)

        # --- Table features per probe ---
        ephys_probe = np.zeros((384, len(FEATURE_LIST)), dtype=np.float32)
        channel_idx = df_pid.index.get_level_values('channel').to_numpy()
        values = np.stack([df_pid[feat].values for feat in FEATURE_LIST], axis=-1)
        ephys_probe[channel_idx] = values

        # Keep your final reversal (up->down)
        ephys_per_probe.append(ephys_probe[::-1, :].copy())

    # Stack all probes
    ephys = np.stack(ephys_per_probe)  # [N, 384, F(+L)]
    ephys[np.where(np.isinf(ephys))] = 0.0
    probe_positions = np.stack(probe_positions)          # [N, 384, 3]
    probe_planned_positions = np.stack(probe_planned_positions)

    # PIDs in the df order
    unique_pids = df_features.index.get_level_values('pid').unique()

    # Filter bad/misaligned
    MISALIGNED_PIDS = ephysatlas.fixtures.misaligned_pids
    if agg == 'agg_full':
        block_extra = [unique_pids[944]]
    else:
        block_extra = []
    block_set = set(MISALIGNED_PIDS + block_extra)
    filter_indices = [i for i, item in enumerate(unique_pids) if item not in block_set]

    filter_pids = unique_pids[filter_indices]
    filter_ephys = ephys[filter_indices]
    filter_probe_positions = probe_positions[filter_indices]
    filter_probe_planned_positions = probe_planned_positions[filter_indices]

    # import statsmodels.api as sm
    # for p in range(int(0.7*filter_ephys.shape[0])):
    #     for f in range(len(FEATURE_LIST)):
    #         filter_ephys[p, :, f] = sm.nonparametric.lowess(filter_ephys[p, :, f], np.arange(len(filter_ephys[p, :, f])), frac=0.1, it=3)[:, 1]

    return filter_pids, filter_ephys, filter_probe_positions, filter_probe_planned_positions

def build_channels_plus_emptyvoxels_with_neighbors(
    ctx_manager: ContextAtlasManager,
    ephys: np.ndarray,                     # [P, C, F_e]
    probe_positions: np.ndarray,           # [P, C, 3] meters
    RADIUS_UM: int,
    M_MAX: int,
    *,
    batch_size_train: int = 1024,
    batch_size_eval: int = 1024,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    New behavior:
      • GRID DATASET = voxels that do NOT contain any ephys channels.
      • RECORDED DATASET = per-channel samples (no voxel averaging):
            one sample per (probe, channel) with valid xyz; context/allen are sampled at that xyz.

    Context standardization: mean/std over ALL grid voxels (full atlas grid, same as before).
    Ephys standardization: mean/std over TRAIN split of recorded channels only.

    Returns:
      train_loader, val_loader, test_loader, e_mean, e_std, ctx_mean, ctx_std
    """

    # ----- 200 µm grid over the whole atlas -----
    P_cell, Xh, Zh, Yh = ctx_manager.cell_pca.shape
    sx, sy, sz = compute_grid_strides_200um(ctx_manager.bc)

    xs = np.arange(0, Xh, dtype=int)
    ys = np.arange(0, Yh, dtype=int)
    zs = np.arange(0, Zh, dtype=int)
    XX, ZZ, YY = np.meshgrid(xs, zs, ys, indexing='ij')
    xi = XX.reshape(-1); zi = ZZ.reshape(-1); yi = YY.reshape(-1)
    N  = xi.size

    ijk   = np.stack([xi, yi, zi], axis=1)
    xyz_m = ctx_manager.bc.i2xyz(ijk * 8).astype(np.float32)  # [N,3] meters

    # Sample context + Allen for ALL voxels (used both for grid_ds creation and context stats)
    allen_all, ctx_all = [], []
    for i in range(N):
        ctx = ctx_manager.sample_context_numpy_i(
            np.array((xi[i], yi[i], zi[i]))[None, :],
            np.array((sx, sy, sz))
        )
        ctx_all.append(concat_context(ctx['cell_pc'], ctx['gene_pc'])[0])
        allen_all.append(ctx['allen_ix'][0])

    ctx_all   = np.asarray(ctx_all, dtype=np.float32)   # [N, F_ctx]
    allen_all = np.asarray(allen_all, dtype=np.int64)
    F_ctx     = int(ctx_all.shape[1])
    F_e       = int(ephys.shape[-1])

    # --- mark voxels that have any ephys channels, to keep only "empty" ones for grid_ds ---
    has_eph = compute_voxel_with_ephys(ctx_manager, probe_positions, xi, yi, zi)
    has_ctx = ~(ctx_all.sum(axis=1) == 0)

    has_vox = has_eph & has_ctx

    grid_mask = ~has_vox
    # Context stats over ALL grid voxels (rec + non-rec) per your original rule
    ctx_all_t = torch.from_numpy(ctx_all).float()
    ctx_mean  = ctx_all_t[grid_mask].mean(dim=0)
    ctx_std   = ctx_all_t[grid_mask].std(dim=0, unbiased=False).clamp_min(1e-6)

    def _stdz_ctx(t):
        mask = np.where(t.sum(axis=1) != 0)[0]
        t_clone = t.clone()
        t_clone[mask] = (t[mask] - ctx_mean) / ctx_std
        return t_clone

    # GRID DATASET (only voxels WITHOUT ephys)
    ctx_grid   = _stdz_ctx(torch.from_numpy(ctx_all[grid_mask]).float())
    allen_grid = torch.from_numpy(allen_all[grid_mask]).long()
    xyz_grid   = torch.from_numpy(xyz_m[grid_mask]).float()
    grid_ds    = GridDS(ctx_grid, allen_grid, xyz_grid, F_e)

    # ----- RECORDED CHANNEL DATASET (per-channel; NO voxel averaging) -----
    P, C, _ = probe_positions.shape
    rec_ctx_list, rec_allen_list, rec_xyz_list, rec_ephys_list, rec_pid_list = [], [], [], [], []

    for p in range(P):
        xyz_p = probe_positions[p].astype(np.float32)       # [C,3]
        eph_p = ephys[p].astype(np.float32)                  # [C,F]
        valid = ~(np.all(xyz_p == 0.0, axis=1))
        if not valid.any():
            continue

        xyz_valid = xyz_p[valid]                             # [C_valid,3]
        # sample context/allen at channel coordinates
        pack = ctx_manager.sample_context_numpy_m(xyz_valid, mode='clip')
        ctx_p = np.concatenate([pack['cell_pc'], pack['gene_pc']], axis=1).astype(np.float32)
        allen_p = pack['allen_ix'].astype(np.int64)
        eph_valid = eph_p[valid]                             # keep as original units for now

        rec_ctx_list.append(ctx_p)
        rec_allen_list.append(allen_p)
        rec_xyz_list.append(xyz_valid)
        rec_ephys_list.append(eph_valid)
        rec_pid_list.append(p * np.ones(C))


    if len(rec_ctx_list) == 0:
        raise RuntimeError("No valid recorded channels found to build recorded dataset.")

    rec_ctx   = torch.from_numpy(np.concatenate(rec_ctx_list, axis=0)).float()      # [Nc,F_ctx]
    rec_allen = torch.from_numpy(np.concatenate(rec_allen_list, axis=0)).long()     # [Nc]
    rec_xyz   = torch.from_numpy(np.concatenate(rec_xyz_list, axis=0)).float()      # [Nc,3]
    rec_ephys = torch.from_numpy(np.concatenate(rec_ephys_list, axis=0)).float()    # [Nc,F_e]
    rec_pids  = torch.from_numpy(np.concatenate(rec_pid_list, axis=0)).float()      # [Nc,]

    # Standardize context (use global grid stats)
    rec_ctx_std = _stdz_ctx(rec_ctx)

    # ----- Split RECORDED by PROBE (recommended) or keep your index split -----
    Nc = rec_ctx_std.shape[0]
    indices = np.arange(Nc)
    n_tr = int(round(0.7 * Nc))
    n_va = int(round(0.1 * Nc))
    I_tr, I_va, I_te = indices[:n_tr], indices[n_tr:n_tr + n_va], indices[n_tr + n_va:]

    # ----- Per-channel voxel key using the SAME rounding used elsewhere -----
    xi_all, zi_all, yi_all = downsample_keys_from_xyz(ctx_manager, rec_xyz.numpy())  # arrays length Nc
    rec_keys = list(zip(xi_all.tolist(), zi_all.tolist(), yi_all.tolist()))

    # Count TRAIN occupancy per voxel
    train_key_counts = Counter(rec_keys[i] for i in I_tr)

    # For every recorded sample, store how many TRAIN channels exist in its voxel.
    # (For val/test, this is still computed against TRAIN; use min=1 to avoid div0 later.)
    vox_count_all = np.array([max(1, train_key_counts.get(k, 0)) for k in rec_keys], dtype=np.float32)

    vox_count = torch.from_numpy(vox_count_all).float()
    vox_count_tr = vox_count[I_tr]
    vox_count_va = vox_count[I_va]
    vox_count_te = vox_count[I_te]

    # EPHYS stats from TRAIN ONLY
    e_mean = rec_ephys[I_tr].mean(dim=0)
    e_std = rec_ephys[I_tr].std(dim=0, unbiased=False).clamp_min(1e-6)

    def _stdz_e(t):
        return (t - e_mean) / e_std

    rec_ephys_std = rec_ephys.clone()
    rec_ephys_std[I_tr] = _stdz_e(rec_ephys[I_tr])
    rec_ephys_std[I_va] = _stdz_e(rec_ephys[I_va])
    # keep test unstandardized if you prefer original-scale R² elsewhere

    # Build REC datasets (now with vox_count)
    rec_train = RecDS(rec_ctx_std[I_tr], rec_allen[I_tr], rec_xyz[I_tr],
                      rec_ephys_std[I_tr], rec_pids[I_tr], vox_count_tr)
    rec_val = RecDS(rec_ctx_std[I_va], rec_allen[I_va], rec_xyz[I_va],
                    rec_ephys_std[I_va], rec_pids[I_va], vox_count_va)
    # test can carry vox_count too (not used for loss weighting)
    rec_test = RecDS(rec_ctx_std[I_te], rec_allen[I_te], rec_xyz[I_te],
                     rec_ephys_std[I_te], rec_pids[I_te], vox_count_te)

    # Final TRAIN = recorded_train + empty_grid  (shuffle in DataLoader)
    train_concat = ConcatDataset([rec_train, grid_ds])

    # =========================
    # Neighbor bank (TRAIN-ONLY, from recorded-train channels)
    # =========================
    # Flatten ALL channels for exclusion/targets (stdzd with train ephys stats)
    bank_xyz, bank_feat, bank_pid = build_channel_catalog(ephys, probe_positions)
    bank_feat_std = ((torch.from_numpy(bank_feat) - e_mean) / e_std).numpy()

    # Neighbor bank = channels whose voxel key is within recorded-train keys
    nn_bank = ChannelNN(bank_xyz[I_tr])

    # Collate with neighbors
    collate = NeighborCollate(
        ctx_manager,
        bank_xyz[I_tr], bank_feat_std[I_tr], bank_pid[I_tr], nn_bank,
        e_feat_dim=F_e, M_max=M_MAX, radius_um=RADIUS_UM
    )

    train_loader = DataLoader(
        train_concat, batch_size=batch_size_train, shuffle=shuffle_train,
        num_workers=0, pin_memory=False, drop_last=False, collate_fn=collate
    )
    val_loader = DataLoader(
        rec_val, batch_size=batch_size_eval, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, collate_fn=collate
    )
    test_loader = DataLoader(
        rec_test, batch_size=batch_size_eval, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, collate_fn=collate
    )

    return train_loader, val_loader, test_loader, e_mean, e_std, ctx_mean, ctx_std

def compute_grid_strides_200um(bc) -> Tuple[int, int, int]:
    """
    Convert 200 µm into strides in index units along x,y,z, rounding to at least 1.
    """
    target_m = 200e-6
    dx = _axis_step_in_indices(bc, 0)
    dy = _axis_step_in_indices(bc, 1)
    dz = _axis_step_in_indices(bc, 2)

    sx = max(1, int(round(target_m / max(dx, 1e-12))))
    sy = max(1, int(round(target_m / max(dy, 1e-12))))
    sz = max(1, int(round(target_m / max(dz, 1e-12))))
    return sx, sy, sz

def _axis_step_in_indices(bc, axis: int) -> float:
    """
    Estimate physical spacing (meters) per +1 index step along a given axis in the high-res grid.
    axis: 0->x, 1->y, 2->z
    """
    a0 = np.zeros((1, 3), dtype=np.int64)
    a1 = a0.copy()
    a1[0, axis] = 1
    p0 = bc.i2xyz(a0)  # [1,3], meters
    p1 = bc.i2xyz(a1)
    return float(np.linalg.norm(p1 - p0))

def concat_context(cell_pc: np.ndarray, gene_pc: np.ndarray) -> np.ndarray:
    return np.concatenate([cell_pc, gene_pc], axis=-1)

def apply_3d_gaussian_blur(x, kernel_size=5, sigma=1.0):
    """
    x: tensor of shape [C, H, W, D]
    Returns: blurred tensor of same shape
    """
    device = x.device
    with torch.no_grad():
        C = x.shape[0]

        kernel = get_gaussian_kernel3d(kernel_size, sigma, device=device)  # [K, K, K]
        kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)  # [1, 1, K, K, K]
        kernel = kernel.repeat(C, 1, 1, 1, 1)  # [C, 1, K, K, K]
        kernel = kernel.to(x.dtype)

        x = x.unsqueeze(0)  # [1, C, H, W, D]
        padding = kernel_size // 2
        blurred = F.conv3d(x, kernel, padding=padding, groups=C)  # [1, C, H, W, D]
    return blurred.squeeze(0)  # [C, H, W, D]

def get_gaussian_kernel3d(kernel_size, sigma, device='cpu'):
    """Create a 3D Gaussian kernel."""
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2. * sigma**2))
    kernel /= kernel.sum()
    return kernel