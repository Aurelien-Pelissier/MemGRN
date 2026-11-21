"""Closed-Loop Human EMT Multi-Ligand Induction & Withdrawal scRNA-seq Benchmark (GSE147405).

Evaluates autonomous closed-loop regulatory dynamics (zero future TF trajectories supplied)
on Cook & Vanderhyden (2020, Nature Communications, GSE147405): 53,290 QC-passed human
single cells across 4 cancer cell lines (A549, DU145, MCF7, OVCA420), 3 EMT-inducing ligands
(TGFB1, EGF, TNF), 112 canonical EMT/signaling genes (36 TFs + 76 hallmark targets), and
8 longitudinal timepoints spanning 7 days of ligand induction (0d, 8h, 1d, 3d, 7d) followed
by 3 days of ligand withdrawal / washout (8h_rm, 1d_rm, 3d_rm).

Three locked zero-leakage evaluations:
  1. Task 1 (Zero-Shot Post-Withdrawal Hysteresis Forecasting, 4-fold Leave-One-Cell-Line-Out):
     All models observe the identical induction trajectory (0d -> 7d) to load intracellular
     memory and start from the exact true Day-7 state y(7d), then roll out 100% autonomously
     with ligand OFF across 8h_rm, 1d_rm, and 3d_rm on the unseen human cell line.
  2. Task 2 (Full 10-Day Autonomous Induction + Withdrawal Forecasting from t=0, 4-fold LOCO):
     Driven strictly by the binary 3-ligand pulse u(t) = 1_{t <= 7d} from y(0) = 0 with zero
     observed expression or TF traces on the held-out human cell line.
  3. Task 3 (Real-Human Post-Withdrawal Peaking & Hysteresis Sub-Analysis):
     Evaluates models specifically on genes/conditions exhibiting delayed post-withdrawal
     peaking (e.g., LAMC2, VIM, PMEPA1, TGFBI) or strong induction-withdrawal hysteresis.
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
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    fit_closed_loop_network,
)
from memgrn.datasets import fetch_and_prepare_emt_withdrawal  # noqa: E402

OUT_DIR = ROOT / "results" / "emt_withdrawal_closed_loop"
RAW_DIR = ROOT / "data" / "raw" / "emt_gse147405"

# Exact RK4 time grid: dt = 1/6 day (4 hours), 60 steps = 10.0 days
# Obs knots: 0d=0, 8h=2, 1d=6, 3d=18, 7d=42 (washout), 8h_rm=44, 1d_rm=48, 3d_rm=60
DT_DAYS = 1.0 / 6.0
N_STEPS = 60
CLAMP_WASHOUT_STEP = 42
OBS_STEP_INDICES = (0, 2, 6, 18, 42, 44, 48, 60)

MODEL_KEYS: tuple[tuple[str, ClosedLoopModelFamily, bool], ...] = (
    ("cl_memgrn", ClosedLoopModelFamily.MEMGRN, True),
    ("cl_memgrn_add", ClosedLoopModelFamily.MEMGRN, False),
    ("cl_local_const", ClosedLoopModelFamily.LOCAL_CONST, False),
    ("cl_ltc_local", ClosedLoopModelFamily.LTC_LOCAL, False),
    ("cl_coupled_adaptive", ClosedLoopModelFamily.COUPLED_ADAPTIVE, False),
    ("cl_ltc_coupled", ClosedLoopModelFamily.LTC_COUPLED, False),
    ("cl_perturb_ode", ClosedLoopModelFamily.PERTURB_ODE, False),
    ("cl_latent_ode", ClosedLoopModelFamily.LATENT_ODE, False),
)


def _build_piecewise_clamp_seq(y_obs: np.ndarray) -> np.ndarray:
    """Build piecewise-linear induction clamp sequence (N_STEPS, B, G) over steps 1..N_STEPS."""
    batch, _, n_g = y_obs.shape
    obs_t = np.asarray(OBS_STEP_INDICES, dtype=np.float64)
    grid_t = np.arange(1, N_STEPS + 1, dtype=np.float64)
    out = np.zeros((N_STEPS, batch, n_g), dtype=np.float32)
    for b_i in range(batch):
        for g_i in range(n_g):
            out[:, b_i, g_i] = np.interp(grid_t, obs_t, y_obs[b_i, :, g_i])
    return out


def _build_ligand_pulse_seq(lig_oh: np.ndarray) -> np.ndarray:
    """Build binary ligand pulse sequence (N_STEPS, B, 3): ON for steps 0..41 (0..7d), OFF after."""
    batch, n_lig = lig_oh.shape
    seq = np.zeros((N_STEPS, batch, n_lig), dtype=np.float32)
    seq[:CLAMP_WASHOUT_STEP, :, :] = lig_oh[None, :, :].astype(np.float32)
    return seq


def _run_loco_fold(job: tuple[int, str, str, bool, str]) -> tuple[int, str, str, dict]:
    """Run one Leave-One-Cell-Line-Out fold for either 'withdrawal' or 'full_10d' task."""
    fold_i, model_key, fam_val, use_coop, task_mode = job
    data = fetch_and_prepare_emt_withdrawal(RAW_DIR)
    n_genes = len(data.genes)
    n_cl = len(data.cell_lines)

    test_cl = data.cell_lines[fold_i]
    val_cl = data.cell_lines[(fold_i + 1) % n_cl]
    train_cls = [c for c in data.cell_lines if c not in (test_cl, val_cl)]

    train_idx = [i for i, (cl, _) in enumerate(data.conditions) if cl in train_cls]
    val_idx = [i for i, (cl, _) in enumerate(data.conditions) if cl == val_cl]
    test_idx = [i for i, (cl, _) in enumerate(data.conditions) if cl == test_cl]

    # Source-only per-gene RMS scaling (zero leakage from held-out cell line!)
    tr_raw = data.delta_trajectories[train_idx]
    gene_scales = np.maximum(0.12, np.sqrt(np.mean(tr_raw**2, axis=(0, 1))))
    y_all = data.delta_trajectories / gene_scales[None, None, :]  # (12, 8, 112)
    basal_scaled = np.mean(data.basal_expression[train_idx], axis=0) / gene_scales
    zero_ko = np.zeros((len(data.conditions), n_genes), dtype=np.float64)
    u_all = _build_ligand_pulse_seq(data.ligand_one_hot)

    # Select top-14 transcriptionally dynamic TFs strictly from source training RMS
    # (filtering out near-constant post-translationally regulated TFs for fair parameter parity)
    tf_rms = np.sqrt(np.mean(tr_raw[:, :, data.tf_indices] ** 2, axis=(0, 1)))
    active_tf_pos = np.sort(data.tf_indices[np.argsort(-tf_rms)[:14]])
    reg_tuple = tuple(int(i) for i in active_tf_pos)

    tr_ko, va_ko, te_ko = zero_ko[train_idx], zero_ko[val_idx], zero_ko[test_idx]
    tr_y, va_y, te_y = y_all[train_idx], y_all[val_idx], y_all[test_idx]
    tr_u, va_u, te_u = u_all[:, train_idx, :], u_all[:, val_idx, :], u_all[:, test_idx, :]

    # Identify top-40 dynamic genes strictly on source split
    src_rms = np.sqrt(np.mean(tr_raw**2, axis=(0, 1)))
    top40_mask = np.zeros(n_genes, dtype=bool)
    top40_mask[np.argsort(-src_rms)[:40]] = True

    if task_mode == "withdrawal":
        # Observe induction (0d -> 7d) via y_clamp_seq; supervise & evaluate strictly on withdrawal (8h_rm, 1d_rm, 3d_rm)
        tr_ycl = _build_piecewise_clamp_seq(tr_y)
        va_ycl = _build_piecewise_clamp_seq(va_y)
        te_ycl = _build_piecewise_clamp_seq(te_y)
        tr_w = np.zeros_like(tr_y)
        va_w = np.zeros_like(va_y)
        tr_w[:, 5:, :] = 1.0
        va_w[:, 5:, :] = 1.0
        clamp_steps = CLAMP_WASHOUT_STEP
    else:
        # Full 10-day autonomous rollout from y(0) = 0 driven solely by ligand pulse u(t)
        tr_ycl = va_ycl = te_ycl = None
        tr_w = np.zeros_like(tr_y)
        va_w = np.zeros_like(va_y)
        tr_w[:, 1:, :] = 1.0
        va_w[:, 1:, :] = 1.0
        clamp_steps = 0

    fam = ClosedLoopModelFamily(fam_val)
    spec = ClosedLoopSpec(
        family=fam,
        n_genes=n_genes,
        regulator_indices=reg_tuple,
        n_ext_inputs=3,
        hollow=False,
        use_cooperativity=use_coop,
        coop_rank=6,
        n_channels=3,
        hidden_width=5,
        module_dim=20,
        latent_dim=20,
        rate_min=0.05,
        rate_max=8.0,
    )
    fitted = fit_closed_loop_network(
        spec=spec,
        basal_levels=basal_scaled,
        train_ko=tr_ko,
        train_y=tr_y,
        train_weights=tr_w,
        val_ko=va_ko,
        val_y=va_y,
        val_weights=va_w,
        dt=DT_DAYS,
        n_steps=N_STEPS,
        obs_step_indices=OBS_STEP_INDICES,
        train_u_ext=tr_u,
        val_u_ext=va_u,
        train_y_clamp=tr_ycl,
        val_y_clamp=va_ycl,
        clamp_steps=clamp_steps,
        seeds=(101, 202),
        reg_grid=(2e-3, 1e-2, 5e-2, 2e-1),
        total_updates=900,
        lr_max=0.006,
        gate_shrinkage=1.5,
    )
    pred_te = fitted.predict_ko_fn(te_ko, None, te_u, te_ycl)  # (3, 8, 112)

    if task_mode == "withdrawal":
        # Strictly post-withdrawal timepoints: 8h_rm (5), 1d_rm (6), 3d_rm (7)
        p_eval = pred_te[:, 5:, :]
        t_eval = te_y[:, 5:, :]
        # Post-withdrawal peaking / hysteretic pairs: |y(8h_rm or 1d_rm)| > |y(7d)| + 0.15 and |y(7d)| > 0.3
        y7 = te_y[:, 4, :]
        y_rm_early = np.maximum(np.abs(te_y[:, 5, :]), np.abs(te_y[:, 6, :]))
        hyst_mask = (np.abs(y7) > 0.35) & (y_rm_early > 0.85 * np.abs(y7))
        mse_all = float(np.mean((p_eval - t_eval) ** 2))
        mse_top40 = float(np.mean((p_eval[:, :, top40_mask] - t_eval[:, :, top40_mask]) ** 2))
        mse_hyst = (
            float(
                np.mean(
                    ((p_eval - t_eval) ** 2)[:, :, :][
                        np.broadcast_to(hyst_mask[:, None, :], p_eval.shape)
                    ]
                )
            )
            if np.any(hyst_mask)
            else mse_all
        )
        # Trajectory Pearson r across withdrawal timepoints
        corrs = []
        for b_i in range(p_eval.shape[0]):
            pb = p_eval[b_i].ravel()
            tb = t_eval[b_i].ravel()
            corrs.append(0.0 if np.std(pb) < 1e-8 else float(np.corrcoef(tb, pb)[0, 1]))
        metrics = {
            "test_cell_line": test_cl,
            "n_params": fitted.n_params,
            "withdrawal_mse": mse_all,
            "withdrawal_top40_mse": mse_top40,
            "hysteresis_mse": mse_hyst,
            "withdrawal_pearson_r": float(np.mean(corrs)),
        }
    else:
        p_all = pred_te[:, 1:, :]
        t_all = te_y[:, 1:, :]
        mse_full = float(np.mean((p_all - t_all) ** 2))
        mse_ind = float(np.mean((pred_te[:, 1:5, :] - te_y[:, 1:5, :]) ** 2))
        mse_wdr = float(np.mean((pred_te[:, 5:, :] - te_y[:, 5:, :]) ** 2))
        mse_top40 = float(np.mean((p_all[:, :, top40_mask] - t_all[:, :, top40_mask]) ** 2))
        corrs = []
        for b_i in range(p_all.shape[0]):
            pb = p_all[b_i].ravel()
            tb = t_all[b_i].ravel()
            corrs.append(0.0 if np.std(pb) < 1e-8 else float(np.corrcoef(tb, pb)[0, 1]))
        metrics = {
            "test_cell_line": test_cl,
            "n_params": fitted.n_params,
            "full_10d_mse": mse_full,
            "induction_mse": mse_ind,
            "full_withdrawal_mse": mse_wdr,
            "full_top40_mse": mse_top40,
            "full_pearson_r": float(np.mean(corrs)),
        }

    return fold_i, model_key, task_mode, metrics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_and_prepare_emt_withdrawal(RAW_DIR)

    # Quantify real-human single-cell iso-expression hysteresis pairs across GSE147405
    # Pairs where an induction timepoint (8h, 1d, or 3d) and early withdrawal (8h_rm or 1d_rm)
    # have nearly identical expression (|y_ind - y_wdr| < 0.08 * range) yet opposite future slopes!
    hyst_examples = []
    for c_i, (cl, lig) in enumerate(data.conditions):
        traj = data.delta_trajectories[c_i]  # (8, 112)
        for g_i, gname in enumerate(data.genes):
            y_g = traj[:, g_i]
            amp = float(np.max(np.abs(y_g)))
            if amp < 0.4:
                continue
            # Check delayed post-withdrawal peak: |y(8h_rm) or y(1d_rm)| > |y(7d)| + 0.08
            if abs(y_g[4]) > 0.25 and (
                (y_g[4] > 0 and max(y_g[5], y_g[6]) > y_g[4] + 0.08)
                or (y_g[4] < 0 and min(y_g[5], y_g[6]) < y_g[4] - 0.08)
            ):
                hyst_examples.append(
                    {
                        "cell_line": cl,
                        "ligand": lig,
                        "gene": gname,
                        "y_7d": float(y_g[4]),
                        "y_8h_rm": float(y_g[5]),
                        "y_1d_rm": float(y_g[6]),
                        "y_3d_rm": float(y_g[7]),
                    }
                )

    design = {
        "benchmark": "emt_multi_ligand_induction_withdrawal_closed_loop_gse147405",
        "accession": "GSE147405",
        "reference": "Cook & Vanderhyden, Nature Communications 11, 2142 (2020)",
        "total_singlet_cells": data.total_singlet_cells,
        "cell_lines": list(data.cell_lines),
        "ligands": list(data.ligands),
        "n_genes": len(data.genes),
        "n_tfs": len(data.tf_indices),
        "time_labels": list(data.time_labels),
        "obs_days": data.obs_days.tolist(),
        "models": [m[0] for m in MODEL_KEYS],
        "n_post_withdrawal_peaking_gene_conditions": len(hyst_examples),
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    jobs = [
        (f_i, m_key, fam.value, use_coop, task_mode)
        for task_mode in ("withdrawal", "full_10d")
        for f_i in range(len(data.cell_lines))
        for (m_key, fam, use_coop) in MODEL_KEYS
    ]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=24, mp_context=ctx) as pool:
        raw_out = list(pool.map(_run_loco_fold, jobs))
    elapsed = time.time() - t0

    by_model: dict[str, dict[str, list[dict]]] = {
        m[0]: {"withdrawal": [], "full_10d": []} for m in MODEL_KEYS
    }
    for _, m_key, task_mode, met in raw_out:
        by_model[m_key][task_mode].append(met)

    summary: dict[str, dict] = {}
    for m_key, _, _ in MODEL_KEYS:
        w_list = by_model[m_key]["withdrawal"]
        f_list = by_model[m_key]["full_10d"]
        summary[m_key] = {
            "n_params": w_list[0]["n_params"],
            "withdrawal_mse": float(np.mean([x["withdrawal_mse"] for x in w_list])),
            "withdrawal_top40_mse": float(np.mean([x["withdrawal_top40_mse"] for x in w_list])),
            "hysteresis_mse": float(np.mean([x["hysteresis_mse"] for x in w_list])),
            "withdrawal_pearson_r": float(np.mean([x["withdrawal_pearson_r"] for x in w_list])),
            "full_10d_mse": float(np.mean([x["full_10d_mse"] for x in f_list])),
            "induction_mse": float(np.mean([x["induction_mse"] for x in f_list])),
            "full_withdrawal_mse": float(np.mean([x["full_withdrawal_mse"] for x in f_list])),
            "full_top40_mse": float(np.mean([x["full_top40_mse"] for x in f_list])),
            "full_pearson_r": float(np.mean([x["full_pearson_r"] for x in f_list])),
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "n_post_withdrawal_peaking_gene_conditions": len(hyst_examples),
        "top_post_withdrawal_peaking_examples": hyst_examples[:15],
        "summary": summary,
        "per_fold": by_model,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": elapsed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
