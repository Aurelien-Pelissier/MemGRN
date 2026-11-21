"""Human EMT Multi-TF Cross-Ligand Induction & Withdrawal Benchmark (GSE147405, 53,290 cells).

Evaluates source-isolated asymmetric memory under real human cross-ligand correlation shift
and 7-day induction + 3-day ligand withdrawal hysteresis on Cook & Vanderhyden (2020,
Nature Communications, GSE147405: 53,290 QC-passed human single cells across 4 cancer
cell lines [A549, DU145, MCF7, OVCA420], 3 EMT-inducing ligands [TGFB1, EGF, TNF], and
8 longitudinal timepoints [0d, 8h, 1d, 3d, 7d, 8h_rm, 1d_rm, 3d_rm]).

Protocol (3-fold Leave-One-Ligand-Out across TGFB1, EGF, TNF):
  - For each held-out ligand (4 cell lines = 4 full 10-day induction+withdrawal trajectories):
    * Train on the other 2 ligands across 3 cell lines (6 trajectories), validate on the
      4th cell line of the source ligands (2 trajectories; rotating validation cell line
      across folds, strictly zero leakage from held-out ligand).
    * Inputs: piecewise-linear measured trajectories of the top-12 source-dynamic canonical
      EMT/signaling TFs (selected strictly on source training RMS, scaled by training RMS).
    * Targets: all 76 canonical hallmark EMT/epithelial/mesenchymal/signaling target genes
      (scaled strictly by training RMS).
    * Evaluation (zero-shot on held-out ligand across all 4 cell lines):
        1. Full 10-day induction + withdrawal MSE (all 76 targets & top-30 dynamic EMT targets).
        2. Post-withdrawal hysteresis MSE strictly on 8h_rm, 1d_rm, 3d_rm.
        3. Trajectory Pearson correlation r and real-human post-withdrawal peaking MSE.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_env_var] = "1"
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    DynamicalModelFamily,
    ModelSpec,
    fit_dynamical_model,
)
from memgrn.datasets import fetch_and_prepare_emt_withdrawal  # noqa: E402

OUT_DIR = ROOT / "results" / "emt_ligand_shift"
RAW_DIR = ROOT / "data" / "raw" / "emt_gse147405"

DT_DAYS = 1.0 / 6.0  # 4 hours
N_STEPS = 60  # 10.0 days
OBS_STEP_INDICES = (0, 2, 6, 18, 42, 44, 48, 60)

MODELS: tuple[DynamicalModelFamily, ...] = (
    DynamicalModelFamily.MEMGRN,
    DynamicalModelFamily.MEMGRN_ADDITIVE,
    DynamicalModelFamily.LOCAL_CONST,
    DynamicalModelFamily.LTC_LOCAL,
    DynamicalModelFamily.COUPLED_ADAPTIVE,
    DynamicalModelFamily.COUPLED_CONST,
    DynamicalModelFamily.LTC_COUPLED,
    DynamicalModelFamily.PERTURB_ODE,
    DynamicalModelFamily.LATENT_ODE,
)


def _interp_tf_stage_grid(tf_obs: np.ndarray) -> np.ndarray:
    """Interpolate (B, 8, S) TF trajectories onto RK4 (N_STEPS, B, S, 3) knots."""
    batch, _, n_s = tf_obs.shape
    obs_t = np.asarray(OBS_STEP_INDICES, dtype=np.float64)
    s0 = np.arange(N_STEPS, dtype=np.float64)
    sm = s0 + 0.5
    s1 = s0 + 1.0
    out = np.zeros((N_STEPS, batch, n_s, 3), dtype=np.float64)
    for b_i in range(batch):
        for s_i in range(n_s):
            v = tf_obs[b_i, :, s_i]
            out[:, b_i, s_i, 0] = np.interp(s0, obs_t, v)
            out[:, b_i, s_i, 1] = np.interp(sm, obs_t, v)
            out[:, b_i, s_i, 2] = np.interp(s1, obs_t, v)
    return out


def _pack_dense_targets(y_8: np.ndarray, active_eval_indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Place 8 irregular observation knots onto obs_stride=2 dense grid (31 knots: steps 0,2,4,...,60)."""
    # Every obs index in (0, 2, 6, 18, 42, 44, 48, 60) is EVEN -> exact knot index k // 2 in [0..30]!
    batch, _, n_g = y_8.shape
    y_31 = np.zeros((batch, 31, n_g), dtype=np.float64)
    m_31 = np.zeros((batch, 31, n_g), dtype=np.float64)
    for obs_pos, step_i in enumerate(OBS_STEP_INDICES):
        k31 = step_i // 2
        y_31[:, k31, :] = y_8[:, obs_pos, :]
        if obs_pos in active_eval_indices:
            m_31[:, k31, :] = 1.0
    return y_31, m_31


def _run_ligand_fold(job: tuple[int, str]) -> tuple[str, str, dict]:
    """Run one Leave-One-Ligand-Out fold for one model family."""
    lig_i, fam_val = job
    data = fetch_and_prepare_emt_withdrawal(RAW_DIR)
    held_lig = data.ligands[lig_i]
    val_cl = data.cell_lines[lig_i % len(data.cell_lines)]

    test_idx = [i for i, (_cl, lig) in enumerate(data.conditions) if lig == held_lig]
    val_idx = [i for i, (cl, lig) in enumerate(data.conditions) if lig != held_lig and cl == val_cl]
    train_idx = [i for i, (cl, lig) in enumerate(data.conditions) if lig != held_lig and cl != val_cl]

    n_tf_all = len(data.tf_indices)
    tgt_idx = np.arange(n_tf_all, len(data.genes), dtype=np.int64)  # 76 hallmark EMT targets

    # Strictly source-only TF selection (top-12 dynamic TFs) & RMS scaling
    tr_raw = data.delta_trajectories[train_idx]  # (6, 8, 112)
    tf_rms = np.sqrt(np.mean(tr_raw[:, :, data.tf_indices] ** 2, axis=(0, 1)))
    sel_tf = np.sort(data.tf_indices[np.argsort(-tf_rms)[:12]])

    tf_scale = np.maximum(0.10, np.sqrt(np.mean(tr_raw[:, :, sel_tf] ** 2, axis=(0, 1))))
    tgt_scale = np.maximum(0.10, np.sqrt(np.mean(tr_raw[:, :, tgt_idx] ** 2, axis=(0, 1))))

    src_tgt_rms = np.sqrt(np.mean(tr_raw[:, :, tgt_idx] ** 2, axis=(0, 1)))
    top30_mask = np.zeros(len(tgt_idx), dtype=bool)
    top30_mask[np.argsort(-src_tgt_rms)[:30]] = True

    def _prep(idxs: list[int]):
        u_grid = _interp_tf_stage_grid(data.delta_trajectories[idxs][:, :, sel_tf] / tf_scale[None, None, :])
        y_scaled = data.delta_trajectories[idxs][:, :, tgt_idx] / tgt_scale[None, None, :]
        y_31, m_31 = _pack_dense_targets(y_scaled, active_eval_indices=(1, 2, 3, 4, 5, 6, 7))
        return u_grid, y_31, m_31, y_scaled

    tr_u, tr_y31, tr_m31, _ = _prep(train_idx)
    va_u, va_y31, va_m31, _ = _prep(val_idx)
    te_u, _, _, te_y8 = _prep(test_idx)

    fam = DynamicalModelFamily(fam_val)
    spec = ModelSpec(
        family=fam,
        n_sources=len(sel_tf),
        n_targets=len(tgt_idx),
        n_channels=3,
        hidden_width=5,
        latent_dim=18,
        module_dim=18,
        rate_min=0.05,
        rate_max=8.0,
    )
    fitted = fit_dynamical_model(
        spec=spec,
        train_stage_u=tr_u,
        train_y=tr_y31,
        train_mask=tr_m31,
        val_stage_u=va_u,
        val_y=va_y31,
        val_mask=va_m31,
        dt=DT_DAYS,
        obs_stride=2,
        seeds=(101, 202),
        reg_grid=(1e-3, 5e-3, 2e-2, 1e-1),
        total_updates=1100,
        lr_max=0.008,
        gate_shrinkage_grid=(1.0, 4.0),
    )
    pred_31 = fitted.predict_fn(te_u, None)  # (4, 31, 76)
    obs_k31 = [s // 2 for s in OBS_STEP_INDICES]
    pred_8 = pred_31[:, obs_k31, :]  # (4, 8, 76)

    # Metrics on held-out ligand across all 4 human cancer cell lines
    mse_full = float(np.mean((pred_8[:, 1:, :] - te_y8[:, 1:, :]) ** 2))
    mse_ind = float(np.mean((pred_8[:, 1:5, :] - te_y8[:, 1:5, :]) ** 2))
    mse_wdr = float(np.mean((pred_8[:, 5:, :] - te_y8[:, 5:, :]) ** 2))
    mse_top30 = float(np.mean((pred_8[:, 1:, top30_mask] - te_y8[:, 1:, top30_mask]) ** 2))
    mse_wdr_top30 = float(np.mean((pred_8[:, 5:, top30_mask] - te_y8[:, 5:, top30_mask]) ** 2))

    # Post-withdrawal peaking gene-condition pairs on the held-out ligand
    y7 = te_y8[:, 4, :]
    peak_mask = (np.abs(y7) > 0.35) & (
        ((y7 > 0) & (np.maximum(te_y8[:, 5, :], te_y8[:, 6, :]) > y7 + 0.08))
        | ((y7 < 0) & (np.minimum(te_y8[:, 5, :], te_y8[:, 6, :]) < y7 - 0.08))
    )
    if np.any(peak_mask):
        pm_3 = np.broadcast_to(peak_mask[:, None, :], pred_8[:, 5:, :].shape)
        mse_peak = float(np.mean(((pred_8[:, 5:, :] - te_y8[:, 5:, :]) ** 2)[pm_3]))
    else:
        mse_peak = mse_wdr

    corrs = []
    for b_i in range(pred_8.shape[0]):
        pb = pred_8[b_i, 1:, :].ravel()
        tb = te_y8[b_i, 1:, :].ravel()
        corrs.append(0.0 if np.std(pb) < 1e-8 else float(np.corrcoef(tb, pb)[0, 1]))

    return (
        held_lig,
        fam_val,
        {
            "n_params": fitted.n_params,
            "full_10d_mse": mse_full,
            "induction_mse": mse_ind,
            "withdrawal_mse": mse_wdr,
            "top30_full_mse": mse_top30,
            "top30_withdrawal_mse": mse_wdr_top30,
            "post_withdrawal_peak_mse": mse_peak,
            "pearson_r": float(np.mean(corrs)),
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_and_prepare_emt_withdrawal(RAW_DIR)

    design = {
        "benchmark": "emt_cross_ligand_induction_withdrawal_gse147405",
        "accession": "GSE147405",
        "reference": "Cook & Vanderhyden, Nature Communications 11, 2142 (2020)",
        "total_singlet_cells": data.total_singlet_cells,
        "cell_lines": list(data.cell_lines),
        "ligands": list(data.ligands),
        "n_selected_tfs": 12,
        "n_targets": len(data.genes) - len(data.tf_indices),
        "time_labels": list(data.time_labels),
        "models": [m.value for m in MODELS],
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    jobs = [(l_i, m.value) for l_i in range(len(data.ligands)) for m in MODELS]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=24, mp_context=ctx) as pool:
        raw_out = list(pool.map(_run_ligand_fold, jobs))
    elapsed = time.time() - t0

    by_model: dict[str, dict[str, dict]] = {m.value: {} for m in MODELS}
    for lig, mk, res in raw_out:
        by_model[mk][lig] = res

    summary: dict[str, dict] = {}
    for m in MODELS:
        mk = m.value
        vals = list(by_model[mk].values())
        summary[mk] = {
            "n_params": vals[0]["n_params"],
            "full_10d_mse": float(np.mean([v["full_10d_mse"] for v in vals])),
            "induction_mse": float(np.mean([v["induction_mse"] for v in vals])),
            "withdrawal_mse": float(np.mean([v["withdrawal_mse"] for v in vals])),
            "top30_full_mse": float(np.mean([v["top30_full_mse"] for v in vals])),
            "top30_withdrawal_mse": float(np.mean([v["top30_withdrawal_mse"] for v in vals])),
            "post_withdrawal_peak_mse": float(np.mean([v["post_withdrawal_peak_mse"] for v in vals])),
            "pearson_r": float(np.mean([v["pearson_r"] for v in vals])),
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "by_ligand": by_model,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": elapsed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
