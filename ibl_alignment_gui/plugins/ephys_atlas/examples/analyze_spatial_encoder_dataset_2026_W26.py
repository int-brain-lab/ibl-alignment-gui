"""Run the 2026_W26 spatial-encoder alignment over every Ephys Atlas insertion.

Outputs
-------
<result_dir>/all_probe_metrics.csv
<result_dir>/filtered_no_misaligned.csv
<result_dir>/filtered_good_lfp_qc.csv
<result_dir>/filtered_no_misaligned_good_lfp_qc.csv
<result_dir>/<pid>_alignement_vis.png
<result_dir>/summary_plots/<scenario>_{all,test}.png
<result_dir>/run_manifest.json

Important conventions
---------------------
* The feature-table channel convention is copied from the existing Ephys Atlas loader:
  xyz[channel_indices] = table_xyz[::-1] and ephys_probe is finally reversed.
* No PID is removed from the evaluation input table.
* The model/reference bank is built from the authoritative split and preprocessing artifacts
  stored in the tagged 2026_W26 Hugging Face release.
* Alignment uses only the histology xyz_picks trajectory. Human annotated channel xyz is used
  for evaluation, never to construct the candidate trace.
"""
from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import ephysatlas.fixtures
from ephysatlas.data import download_tables, read_features_from_disk
from ephysatlas.spatial_encoder.model import (
    NeighborInpaintingModel,
    ProbeSequenceConfidenceTransformer,
)
from ephysatlas.spatial_encoder.model_registry import (
    EphysAtlasReleaseRegistry,
    split_manifest_to_builder_format,
)
from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig,
    ContextAtlasManager,
    FEATURE_LIST,
    build_channels_plus_emptyvoxels_with_neighbors,
    region_ids_from_xyz,
)
from iblatlas.atlas import AllenAtlas
from iblatlas.plots import plot_points_on_slice
from one.api import ONE

try:
    from ibl_alignment_gui.plugins.ephys_atlas.spatial_encoder import align
except ImportError:
    from spatial_encoder import align


MODEL_VINTAGE = "2026_W26"
HF_REPO_ID = "AlonSaguy/ephys-atlas-models"
PROJECT = "ea_active"
AGG = "agg_full"


@dataclass
class Engine:
    model: NeighborInpaintingModel
    conf_model: Optional[ProbeSequenceConfidenceTransformer]
    ctx_manager: ContextAtlasManager
    handles: dict
    device: torch.device
    radius_um: float
    m_max: int
    optimization_features: np.ndarray
    split_manifest: dict
    release_dir: Path


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_load(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _loader_handles(train_loader) -> dict:
    c = train_loader.collate_fn
    return {"bank_xyz": c.bank_xyz, "bank_feat": c.bank_feat, "bank_pid": c.bank_pid, "nn_bank": c.nn}


def _unpack_builder(out):
    if len(out) == 10:
        tr, _, _, _, e_mean, e_std, ctx_mean, ctx_std, split_info, _ = out
    elif len(out) == 9:
        tr, _, _, _, e_mean, e_std, ctx_mean, ctx_std, split_info = out
    else:
        raise RuntimeError(
            f"Expected release-aware builder to return 9/10 values, got {len(out)}. "
            "Switch ephysatlas to the branch with split_manifest/preprocessing_stats support."
        )
    return tr, e_mean, e_std, ctx_mean, ctx_std, split_info


def load_all_insertion_data(*, data_dir: Path, one: ONE):
    """Load every PID without applying ephysatlas.fixtures.misaligned_pids."""
    path_data = download_tables(data_dir, label=MODEL_VINTAGE, project=PROJECT, one=one, agg_level=AGG)
    df_features = read_features_from_disk(path_data, strict=False)

    pids, ephys_all, xyz_all, planned_all = [], [], [], []
    for pid, df_pid in df_features.groupby(level="pid", sort=False):
        C = len(df_pid)
        channel_indices = df_pid.index.get_level_values("channel").to_numpy(dtype=int)

        xyz = np.zeros((C, 3), dtype=np.float32)
        xyz_planned = np.zeros((C, 3), dtype=np.float32)
        xyz[channel_indices] = df_pid[["x", "y", "z"]].to_numpy()[::-1].copy()
        xyz_planned[channel_indices] = df_pid[["x_target", "y_target", "z_target"]].to_numpy()[::-1].copy()

        eph = np.zeros((C, len(FEATURE_LIST)), dtype=np.float32)
        values = np.stack([df_pid[f].to_numpy() for f in FEATURE_LIST], axis=-1)
        eph[channel_indices] = values
        eph = eph[::-1].copy()
        eph[~np.isfinite(eph)] = 0.0

        pids.append(str(pid))
        ephys_all.append(eph)
        xyz_all.append(xyz)
        planned_all.append(xyz_planned)

    return (
        np.asarray(pids, dtype=str),
        np.stack(ephys_all),
        np.stack(xyz_all),
        np.stack(planned_all),
        Path(path_data),
    )


def build_engine(*, data_dir: Path, one: ONE, hf_repo_id: str = HF_REPO_ID) -> Engine:
    device = _device()
    registry = EphysAtlasReleaseRegistry()
    release_dir = registry.resolve_release(MODEL_VINTAGE, repo_id=hf_repo_id, require_weights=True)
    registry.validate_feature_order(MODEL_VINTAGE, FEATURE_LIST)

    config = registry.load_config(MODEL_VINTAGE)
    split_raw = registry.load_split(MODEL_VINTAGE)
    split_manifest = split_manifest_to_builder_format(split_raw)
    preprocessing_stats = registry.load_channel_preprocessing_stats(MODEL_VINTAGE)

    context_cfg = config.get("context", {})
    channel_cfg = config.get("channel_level", {})
    arch = channel_cfg.get("architecture", {})
    neigh = channel_cfg.get("neighbors", {})

    ctx_manager = ContextAtlasManager(
        AtlasPCAConfig(
            n_cell_pcs=int(context_cfg.get("n_cell_pcs", 50)),
            n_gene_pcs=int(context_cfg.get("n_gene_pcs", 50)),
        ),
        regenerate_context=False,
        output_dir=release_dir / "context",
    )

    # Use the standard loader for the model reference bank. The authoritative split controls
    # which loaded probes can enter the train-neighbour bank.
    from ephysatlas.spatial_encoder.utils import LoadInsertionData
    pid_bank, ephys_bank, xyz_bank, _ = LoadInsertionData(
        project=config.get("data", {}).get("project", PROJECT),
        agg=config.get("data", {}).get("agg", AGG),
        VINTAGE=MODEL_VINTAGE,
        path_data=data_dir,
    )
    pid_bank = [str(x) for x in pid_bank]

    radius_um = float(neigh.get("radius_um", 500))
    m_max = int(neigh.get("m_max", 8))
    try:
        built = build_channels_plus_emptyvoxels_with_neighbors(
            ctx_manager,
            ephys_bank,
            xyz_bank,
            RADIUS_UM=radius_um,
            M_MAX=m_max,
            pid_names=pid_bank,
            split_manifest=split_manifest,
            preprocessing_stats=preprocessing_stats,
            return_preprocessing_stats=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Your checked-out ephysatlas utils.py is older than run_spatial_encoder.py. "
            "The 2026_W26 analysis requires build_channels_plus_emptyvoxels_with_neighbors(..., "
            "split_manifest=..., preprocessing_stats=..., return_preprocessing_stats=True)."
        ) from exc

    train_loader, e_mean, e_std, ctx_mean, ctx_std, _ = _unpack_builder(built)
    handles = _loader_handles(train_loader)
    f_ctx, f_e = int(ctx_mean.numel()), int(e_mean.numel())

    model = NeighborInpaintingModel(
        f_ctx=f_ctx,
        f_ephys=f_e,
        f_out=f_e,
        e_mean=e_mean,
        e_std=e_std,
        ctx_mean=ctx_mean,
        ctx_std=ctx_std,
        d_model=int(arch.get("d_model", 128)),
        nhead=int(arch.get("nhead", 8)),
        depth=int(arch.get("depth", 2)),
        drop=float(arch.get("drop", 0.15)),
    ).to(device)
    base_ckpt = torch.load(release_dir / "models" / "channel" / "spatial_encoder.pt", map_location=device)
    model.load_state_dict(base_ckpt["model_state"], strict=True)
    model.eval()

    conf_model = None
    conf_path = release_dir / "models" / "channel" / "confidence_model.pt"
    if conf_path.exists():
        ckpt = torch.load(conf_path, map_location=device)
        ca = ckpt.get("architecture", {})
        conf_model = ProbeSequenceConfidenceTransformer(
            f_ctx=f_ctx,
            f_e=f_e,
            d_model=int(ca.get("d_model", 64)),
            nhead=int(ca.get("nhead", 4)),
            depth=int(ca.get("depth", 2)),
            mlp_ratio=float(ca.get("mlp_ratio", 2.0)),
            drop=float(ca.get("drop", 0.1)),
        ).to(device)
        conf_model.load_state_dict(ckpt["model_state"], strict=True)
        conf_model.eval()

    return Engine(
        model=model,
        conf_model=conf_model,
        ctx_manager=ctx_manager,
        handles=handles,
        device=device,
        radius_um=radius_um,
        m_max=m_max,
        optimization_features=np.arange(len(FEATURE_LIST), dtype=int),
        split_manifest=split_raw,
        release_dir=release_dir,
    )


def _infer_xyz_m(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    if xyz.size and np.nanmax(np.abs(xyz)) > 0.1:
        xyz = xyz / 1e6
    return xyz.astype(np.float32)


def histology_picks(one: ONE, pid: str) -> np.ndarray:
    recs = one.alyx.rest("insertions", "list", id=str(pid))
    if not recs:
        raise RuntimeError("No Alyx insertion")
    picks = (recs[0].get("json") or {}).get("xyz_picks")
    if picks is None:
        raise RuntimeError("No xyz_picks in insertion json")
    picks = _infer_xyz_m(np.asarray(picks))
    valid = np.isfinite(picks).all(axis=1) & ~np.all(picks == 0, axis=1)
    picks = picks[valid]
    if len(picks) < 2:
        raise RuntimeError("Need >=2 valid histology picks")
    return picks


def resample_curve(xyz: np.ndarray, step_um: float = 10.0) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    seg_um = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1e6
    s = np.r_[0.0, np.cumsum(seg_um)]
    keep = np.r_[True, np.diff(s) > 1e-6]
    xyz, s = xyz[keep], s[keep]
    if len(xyz) < 2 or s[-1] <= 0:
        return xyz.astype(np.float32)
    sq = np.arange(0.0, s[-1] + 0.5 * step_um, step_um)
    out = np.column_stack([np.interp(sq, s, xyz[:, d]) for d in range(3)])
    return out.astype(np.float32)


def _region_ids(ba: AllenAtlas, xyz: np.ndarray, mapping: str) -> np.ndarray:
    return np.asarray(region_ids_from_xyz(ba, np.asarray(xyz, np.float32), mapping=mapping, mode="clip")).reshape(-1)


def _metrics(true_xyz, est_xyz, conf_cls, ba: AllenAtlas, subset: Optional[int] = None) -> dict:
    true_xyz = np.asarray(true_xyz, float)
    est_xyz = np.asarray(est_xyz, float)
    valid = (
        np.isfinite(true_xyz).all(axis=1)
        & np.isfinite(est_xyz).all(axis=1)
        & ~np.all(true_xyz == 0, axis=1)
        & ~np.all(est_xyz == 0, axis=1)
    )
    if subset is not None:
        if conf_cls is None:
            valid &= False
        else:
            valid &= np.asarray(conf_cls) == int(subset)

    out = {"n_channels": int(valid.sum()), "l2_um": np.nan, "cosmos_acc": np.nan, "beryl_acc": np.nan}
    if not valid.any():
        return out
    out["l2_um"] = float(np.mean(np.linalg.norm(est_xyz[valid] - true_xyz[valid], axis=1)) * 1e6)
    for mapping, key in [("Cosmos", "cosmos_acc"), ("Beryl", "beryl_acc")]:
        t = _region_ids(ba, true_xyz[valid], mapping)
        p = _region_ids(ba, est_xyz[valid], mapping)
        m = t != 0
        out[key] = float(np.mean(t[m] == p[m])) if m.any() else np.nan
    return out


def get_lfp_qc(pid: str, h5_path: Optional[Path], max_saturated_fraction: float) -> dict:
    out = {
        "lfp_qc_available": False,
        "lfp_saturated_fraction": np.nan,
        "lfp_saturation_n_intervals": np.nan,
        "bad_lfp_qc": False,
        "lfp_qc_error": "",
    }
    if h5_path is None:
        return out
    try:
        from lfpack import LFPackReader
        sr = LFPackReader(str(h5_path), recording=str(pid))
        summary = dict(sr.saturation_summary or {})
        frac = float(summary.get("saturated_fraction", 0.0) or 0.0)
        out.update(
            lfp_qc_available=True,
            lfp_saturated_fraction=frac,
            lfp_saturation_n_intervals=int(summary.get("n_intervals", 0) or 0),
            bad_lfp_qc=bool(frac > max_saturated_fraction),
        )
    except Exception as exc:
        out["lfp_qc_error"] = str(exc)
    return out


def _choose_features():
    names = list(FEATURE_LIST)
    def take(cands):
        out = []
        for x in cands:
            if x in names and names.index(x) not in out:
                out.append(names.index(x))
            if len(out) == 3:
                break
        return out
    return {
        "LF": take(["rms_lf", "psd_lfp", "psd_gamma", "psd_delta", "psd_theta"]),
        "AP": take(["rms_ap", "cor_ratio", "alpha_mean", "alpha_std"]),
        "spike": take(["spike_count", "peak_val", "trough_val", "recovery_time_secs"]),
    }


def _unstd(mu_std: np.ndarray, model) -> np.ndarray:
    mean = model.e_mean.detach().cpu().numpy()
    std = model.e_std.detach().cpu().numpy()
    return np.asarray(mu_std) * (std + 1e-8) + mean


def _plot_trace_panel(ax, view, ba, traces: dict[str, np.ndarray], half_span_um: float):
    all_xyz = np.concatenate([x for x in traces.values() if x is not None and len(x)], axis=0)
    valid = np.isfinite(all_xyz).all(axis=1)
    center = np.nanmean(all_xyz[valid], axis=0) if valid.any() else np.zeros(3)
    coord = int((center[1] if view == "coronal" else center[0]) * 1e6)
    try:
        plot_points_on_slice(torch.zeros((2, 3)), coord=coord, slice=view, ax=ax, cmap="Greys")
    except Exception:
        pass
    for label, xyz in traces.items():
        if xyz is None or len(xyz) == 0:
            continue
        xyz = np.asarray(xyz, float)
        x = xyz[:, 0] if view == "coronal" else xyz[:, 1]
        z = xyz[:, 2]
        ax.plot(x * 1e6, z * 1e6, lw=2, label=label)
    cx = (center[0] if view == "coronal" else center[1]) * 1e6
    cz = center[2] * 1e6
    ax.set_xlim(cx - half_span_um, cx + half_span_um)
    ax.set_ylim(cz - half_span_um, cz + half_span_um)
    ax.set_box_aspect(1)
    ax.set_title(view.capitalize())
    ax.set_xlabel("X (µm)" if view == "coronal" else "Y (µm)")
    ax.set_ylabel("Z (µm)")
    ax.legend(fontsize=7, loc="best")


def _stripe(ax, values, title, cmap="viridis", vmin=None, vmax=None):
    v = np.asarray(values)
    if v.ndim == 1:
        v = v[:, None]
    ax.imshow(v, aspect="auto", interpolation="nearest", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def make_probe_figure(
    *, out_path: Path, pid: str, hist_xyz: np.ndarray, true_xyz: np.ndarray, planned_xyz: np.ndarray,
    out: dict, engine: Engine, ba: AllenAtlas, is_misaligned: bool, bad_lfp: bool,
):
    est_xyz = np.asarray(out["est_xyz"], float)
    traces = {"histology": hist_xyz, "human": true_xyz, "planned": planned_xyz, "predicted": est_xyz}
    vals = []
    for view_axis in [0, 1]:
        for xyz in traces.values():
            if xyz is not None and len(xyz):
                vals.extend((xyz[:, view_axis] * 1e6).tolist())
                vals.extend((xyz[:, 2] * 1e6).tolist())
    half_span = max(700.0, 0.55 * (np.nanmax(vals) - np.nanmin(vals))) if vals else 1000.0

    fig = plt.figure(figsize=(23, 12))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.35], hspace=0.28, wspace=0.25)
    _plot_trace_panel(fig.add_subplot(gs[0, 0]), "coronal", ba, traces, half_span)
    _plot_trace_panel(fig.add_subplot(gs[0, 1]), "sagittal", ba, traces, half_span)

    ax = fig.add_subplot(gs[0, 2])
    C = np.asarray(out["cost_matrix"], float)
    finite = C[np.isfinite(C)]
    vmin, vmax = (np.nanpercentile(finite, [1, 99]) if finite.size else (0, 1))
    im = ax.imshow(C, aspect="auto", cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
    p = np.asarray(out["path"], int)
    if p.ndim == 2 and p.shape[1] == 2:
        ax.plot(p[:, 1], p[:, 0], color="white", lw=1.4)
    ax.set_title("Alignment cost + inferred path")
    ax.set_xlabel("Histology trace index")
    ax.set_ylabel("Recorded channel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    sub = gs[1, :].subgridspec(1, 11, wspace=0.12)
    true_cosmos = _region_ids(ba, true_xyz, "Cosmos")
    pred_cosmos = _region_ids(ba, est_xyz, "Cosmos")
    true_beryl = _region_ids(ba, true_xyz, "Beryl")
    pred_beryl = _region_ids(ba, est_xyz, "Beryl")
    for j, (arr, title) in enumerate([
        (true_cosmos, "Human\nCosmos"), (pred_cosmos, "Pred\nCosmos"),
        (true_beryl, "Human\nBeryl"), (pred_beryl, "Pred\nBeryl"),
    ]):
        rgb = ba._label2rgb(arr.astype(int))
        _stripe(fig.add_subplot(sub[0, j]), rgb, title, cmap=None)

    conf = out.get("pred_cls_est")
    conf_img = np.full((len(est_xyz), 1, 3), 0.65)
    if conf is not None:
        conf = np.asarray(conf)
        conf_img[conf == 0, 0] = [0.1, 0.75, 0.1]
        conf_img[conf == 1, 0] = [0.9, 0.15, 0.15]
    _stripe(fig.add_subplot(sub[0, 4]), conf_img, "Confidence\ngreen=high\nred=low", cmap=None)

    rec = np.asarray(out["recorded_on_trace_raw"], float)
    pred = _unstd(np.asarray(out["mu_std_trace"], float), engine.model)
    groups = _choose_features()
    col = 5
    for group, idx in groups.items():
        if not idx:
            continue
        rv, pv = rec[:, idx], pred[:, idx]
        both = np.r_[rv[np.isfinite(rv)], pv[np.isfinite(pv)]]
        if both.size:
            lo, hi = np.nanpercentile(both, [1, 99])
        else:
            lo, hi = None, None
        axr = fig.add_subplot(sub[0, col]); col += 1
        _stripe(axr, rv, f"{group}\nrecorded", "viridis", lo, hi)
        axr.set_xticks(np.arange(len(idx))); axr.set_xticklabels([FEATURE_LIST[i] for i in idx], rotation=90, fontsize=6)
        axp = fig.add_subplot(sub[0, col]); col += 1
        _stripe(axp, pv, f"{group}\npredicted", "viridis", lo, hi)
        axp.set_xticks(np.arange(len(idx))); axp.set_xticklabels([FEATURE_LIST[i] for i in idx], rotation=90, fontsize=6)

    fig.suptitle(
        f"{pid}_alignement_vis | misaligned_list={is_misaligned} | bad_LFP_QC={bad_lfp}",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scenario_masks(df: pd.DataFrame):
    return {
        "all_probes": np.ones(len(df), dtype=bool),
        "no_misaligned": ~df["is_misaligned"].fillna(False).to_numpy(bool),
        "good_lfp_qc": ~df["bad_lfp_qc"].fillna(False).to_numpy(bool),
        "no_misaligned_good_lfp_qc": (
            ~df["is_misaligned"].fillna(False).to_numpy(bool)
            & ~df["bad_lfp_qc"].fillna(False).to_numpy(bool)
        ),
    }


def write_filtered_csvs(df: pd.DataFrame, result_dir: Path):
    names = {
        "all_probes": "all_probe_metrics.csv",
        "no_misaligned": "filtered_no_misaligned.csv",
        "good_lfp_qc": "filtered_good_lfp_qc.csv",
        "no_misaligned_good_lfp_qc": "filtered_no_misaligned_good_lfp_qc.csv",
    }
    for scenario, mask in _scenario_masks(df).items():
        df.loc[mask].to_csv(result_dir / names[scenario], index=False)


def update_summary_plots(df: pd.DataFrame, result_dir: Path):
    summary_dir = result_dir / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics = [("l2_um", "L2 distance (µm)"), ("cosmos_acc", "Cosmos accuracy"), ("beryl_acc", "Beryl accuracy")]
    for scenario, mask0 in _scenario_masks(df).items():
        for scope in ["all", "test"]:
            mask = mask0.copy()
            if scope == "test":
                mask &= df["split"].eq("test").to_numpy()
            d = df.loc[mask]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
            for ax, (base, title) in zip(axes, metrics):
                high = pd.to_numeric(d[f"{base}_high_conf"], errors="coerce").dropna().to_numpy()
                low = pd.to_numeric(d[f"{base}_low_conf"], errors="coerce").dropna().to_numpy()
                bins = 30
                if len(high): ax.hist(high, bins=bins, alpha=0.55, label=f"high conf (n={len(high)})")
                if len(low): ax.hist(low, bins=bins, alpha=0.55, label=f"low conf (n={len(low)})")
                ax.set_title(title); ax.set_ylabel("Probes"); ax.legend(fontsize=8)
            fig.suptitle(f"{scenario} | {scope} | completed probes={len(df)} | included={len(d)}")
            fig.tight_layout()
            fig.savefig(summary_dir / f"{scenario}_{scope}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=Path("results/spatial_encoder_2026_W26_all_probes"))
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--hf-repo-id", default=HF_REPO_ID)
    ap.add_argument("--lfpack-h5", type=Path, default=None, help="Path to lf_compressed_all_bwm.h5")
    ap.add_argument("--lfp-max-saturated-fraction", type=float, default=0.0,
                    help="A PID fails LFP QC when saturated_fraction is greater than this value.")
    ap.add_argument("--summary-update-every", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Debug only: analyze first N PIDs")
    args = ap.parse_args()

    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary_plots").mkdir(exist_ok=True)
    progress_csv = result_dir / "all_probe_metrics.csv"

    one = ONE(base_url="https://alyx.internationalbrainlab.org")
    ba = AllenAtlas()
    engine = build_engine(data_dir=args.data_dir, one=one, hf_repo_id=args.hf_repo_id)
    pids, ephys, true_xyz, planned_xyz, feature_dir = load_all_insertion_data(data_dir=args.data_dir, one=one)

    split_map = {}
    for split_key, label in [("train_pids", "train"), ("validation_pids", "validation"), ("test_pids", "test")]:
        for pid in engine.split_manifest.get(split_key, []):
            split_map[str(pid)] = label
    misaligned = {str(x) for x in ephysatlas.fixtures.misaligned_pids}

    existing = pd.read_csv(progress_csv) if progress_csv.exists() and not args.overwrite else pd.DataFrame()
    done = set(existing.get("pid", pd.Series(dtype=str)).astype(str))
    rows = existing.to_dict("records")

    indices = range(len(pids)) if args.limit is None else range(min(args.limit, len(pids)))
    for count, i in enumerate(tqdm(indices, desc="align all Ephys Atlas probes"), start=1):
        pid = str(pids[i])
        if pid in done and not args.overwrite:
            continue
        row: dict[str, Any] = {
            "pid": pid,
            "split": split_map.get(pid, "unassigned"),
            "is_misaligned": pid in misaligned,
            "alignment_ok": False,
            "error": "",
        }
        row.update(get_lfp_qc(pid, args.lfpack_h5, args.lfp_max_saturated_fraction))
        try:
            picks = histology_picks(one, pid)
            trace = resample_curve(picks, step_um=10.0)
            with torch.no_grad():
                out = align(
                    engine.model,
                    engine.ctx_manager,
                    trace,
                    ephys[i],
                    engine.handles,
                    engine.optimization_features,
                    engine.radius_um,
                    engine.m_max,
                    engine.device,
                    conf_model=engine.conf_model,
                    return_debug=True,
                    brain_atlas=ba,
                )
            if out is None:
                raise RuntimeError("align returned None")

            conf_cls = out.get("pred_cls_est")
            all_m = _metrics(true_xyz[i], out["est_xyz"], conf_cls, ba, None)
            hi_m = _metrics(true_xyz[i], out["est_xyz"], conf_cls, ba, 0)
            lo_m = _metrics(true_xyz[i], out["est_xyz"], conf_cls, ba, 1)
            row.update({
                "alignment_ok": True,
                "l2_um": all_m["l2_um"], "cosmos_acc": all_m["cosmos_acc"], "beryl_acc": all_m["beryl_acc"],
                "n_eval_channels": all_m["n_channels"],
                "l2_um_high_conf": hi_m["l2_um"], "cosmos_acc_high_conf": hi_m["cosmos_acc"], "beryl_acc_high_conf": hi_m["beryl_acc"],
                "n_high_conf_channels": hi_m["n_channels"],
                "l2_um_low_conf": lo_m["l2_um"], "cosmos_acc_low_conf": lo_m["cosmos_acc"], "beryl_acc_low_conf": lo_m["beryl_acc"],
                "n_low_conf_channels": lo_m["n_channels"],
                "alignment_cost_per_path_step": float(out["total_cost"] / max(1, len(out["path"]))),
            })
            make_probe_figure(
                out_path=result_dir / f"{pid}_alignement_vis.png",
                pid=pid, hist_xyz=picks, true_xyz=true_xyz[i], planned_xyz=planned_xyz[i], out=out,
                engine=engine, ba=ba, is_misaligned=row["is_misaligned"], bad_lfp=row["bad_lfp_qc"],
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        rows = [r for r in rows if str(r.get("pid")) != pid] + [row]
        df = pd.DataFrame(rows)
        write_filtered_csvs(df, result_dir)
        if count % max(1, args.summary_update_every) == 0 or count == 1:
            update_summary_plots(df, result_dir)

    df = pd.DataFrame(rows)
    write_filtered_csvs(df, result_dir)
    update_summary_plots(df, result_dir)
    manifest = {
        "model_vintage": MODEL_VINTAGE,
        "hf_repo_id": args.hf_repo_id,
        "release_dir": str(engine.release_dir),
        "feature_dir": str(feature_dir),
        "lfpack_h5": None if args.lfpack_h5 is None else str(args.lfpack_h5),
        "lfp_bad_definition": f"saturated_fraction > {args.lfp_max_saturated_fraction}",
        "n_loaded_probes_no_filter": int(len(pids)),
        "n_completed_rows": int(len(df)),
        "confidence_definition": "high=confidence model class 0 (good), low=class 1 (suspicious)",
    }
    with (result_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Results: {result_dir.resolve()}")


if __name__ == "__main__":
    main()
