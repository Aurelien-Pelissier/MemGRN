"""Zero-Shot Human Cancer EMT Withdrawal Forecasting Benchmark (GSE147405, 53,290 cells).

Evaluates real human 3-day ligand-withdrawal hysteresis and post-withdrawal peaking on
Cook & Vanderhyden (2020, Nature Communications, GSE147405: 53,290 QC-passed human single
cells across 4 cancer cell lines [A549, DU145, MCF7, OVCA420], 3 EMT-inducing ligands
[TGFB1, EGF, TNF], and 8 timepoints spanning 7-day induction [0d, 8h, 1d, 3d, 7d] and
3-day ligand withdrawal [8h_rm, 1d_rm, 3d_rm]).

Protocol (4-fold Leave-One-Cell-Line-Withdrawal-Out, zero leakage of held-out withdrawal):
  - For each held-out human cancer cell line c in {A549, DU145, MCF7, OVCA420}:
      * Training: 7-day induction (8h..7d) of all 12 (cell_line, ligand) trajectories PLUS
        full 3-day withdrawal (8h_rm..3d_rm) of 2 training cell lines (6 withdrawal curves).
      * Validation: full 3-day withdrawal of the 3rd cell line (3 withdrawal curves).
      * Zero-shot test: NEVER-SEEN 3-day withdrawal (8h_rm, 1d_rm, 3d_rm) of held-out
        cell line c across all 3 ligands (TGFB1, EGF, TNF) across all 76 hallmark EMT targets,
        top-30 source-dynamic EMT targets, and delayed post-withdrawal peaking gene-condition
        pairs, plus counterfactual observed-TF attribution drift vs. standalone MemGRN.
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
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    DynamicalModelFamily,
    ModelSpec,
    counterfactual_source_attribution,
    fit_dynamical_model,
    fit_two_stage_memgrn_ortho,
)
from memgrn.datasets import fetch_and_prepare_emt_withdrawal  # noqa: E402

OUT_DIR = ROOT / "results" / "emt_withdrawal_forecast"
RAW_DIR = ROOT / "data" / "raw" / "emt_gse147405"

DT_DAYS = 1.0 / 6.0  # 4 hours
N_STEPS = 60  # 10.0 days
OBS_STEP_INDICES = (0, 2, 6, 18, 42, 44, 48, 60)

MODEL_CONFIGS: tuple[tuple[str, DynamicalModelFamily, int | None, bool | None], ...] = (
    ("memgrn_ortho", DynamicalModelFamily.MEMGRN_ORTHO, 6, True),
    ("memgrn", DynamicalModelFamily.MEMGRN, 6, True),
    ("memgrn_additive", DynamicalModelFamily.MEMGRN_ADDITIVE, None, False),
    ("local_const", DynamicalModelFamily.LOCAL_CONST, None, False),
    ("local_const_coop", DynamicalModelFamily.LOCAL_CONST, 6, True),
    ("ltc_local", DynamicalModelFamily.LTC_LOCAL, None, False),
    ("ltc_local_coop", DynamicalModelFamily.LTC_LOCAL, 6, True),
    ("coupled_adaptive", DynamicalModelFamily.COUPLED_ADAPTIVE, None, False),
    ("ltc_coupled", DynamicalModelFamily.LTC_COUPLED, None, False),
    ("selective_ssm", DynamicalModelFamily.SELECTIVE_SSM, None, False),
    ("cfc", DynamicalModelFamily.CFC, None, False),
    ("neural_cde", DynamicalModelFamily.NEURAL_CDE, None, False),
    ("perturb_ode", DynamicalModelFamily.PERTURB_ODE, None, False),
    ("latent_ode", DynamicalModelFamily.LATENT_ODE, None, False),
)


def _interp_tf_stage_grid(tf_obs: np.ndarray) -> np.ndarray:
    batch, _, n_s = tf_obs.shape
    obs_t = np.asarray(OBS_STEP_INDICES, dtype=np.float64)
    s0 = np.arange(N_STEPS, dtype=np.float64)
    out = np.zeros((N_STEPS, batch, n_s, 3), dtype=np.float64)
    for b_i in range(batch):
        for s_i in range(n_s):
            v = tf_obs[b_i, :, s_i]
            out[:, b_i, s_i, 0] = np.interp(s0, obs_t, v)
            out[:, b_i, s_i, 1] = np.interp(s0 + 0.5, obs_t, v)
            out[:, b_i, s_i, 2] = np.interp(s0 + 1.0, obs_t, v)
    return out


def _run_cellline_fold(job: tuple[int, str, str, int | None, bool | None]) -> tuple[str, str, dict]:
    cl_i, model_key, fam_val, coop_rank, use_coop = job
    data = fetch_and_prepare_emt_withdrawal(RAW_DIR)
    n_cl = len(data.cell_lines)
    held_cl = data.cell_lines[cl_i]
    val_cl = data.cell_lines[(cl_i + 1) % n_cl]

    test_rows = [i for i, (cl, _l) in enumerate(data.conditions) if cl == held_cl]
    val_rows = [i for i, (cl, _l) in enumerate(data.conditions) if cl == val_cl]
    train_wdr_rows = [i for i in range(len(data.conditions)) if i not in test_rows and i not in val_rows]

    y_all_raw = data.delta_trajectories  # (12, 8, 112)
    n_tf_all = len(data.tf_indices)
    tgt_idx = np.arange(n_tf_all, len(data.genes), dtype=np.int64)

    tr_mask_ct = np.zeros((12, 8), dtype=bool)
    tr_mask_ct[:, 1:5] = True  # induction frames observed across all 12 conditions
    tr_mask_ct[train_wdr_rows, 5:] = True  # withdrawal observed strictly on 2 train cell lines
    va_mask_ct = np.zeros((12, 8), dtype=bool)
    va_mask_ct[val_rows, 5:] = True  # withdrawal observed strictly on validation cell line

    # Zero-leakage TF selection & RMS scaling from observed training frames ONLY
    tf_rms = np.sqrt(np.mean(y_all_raw[:, :, data.tf_indices][tr_mask_ct] ** 2, axis=0))
    sel_tf = np.sort(data.tf_indices[np.argsort(-tf_rms)[:12]])
    tf_sc = np.maximum(0.10, np.sqrt(np.mean(y_all_raw[:, :, sel_tf][tr_mask_ct] ** 2, axis=0)))
    tg_sc = np.maximum(0.10, np.sqrt(np.mean(y_all_raw[:, :, tgt_idx][tr_mask_ct] ** 2, axis=0)))

    src_tgt_rms = np.sqrt(np.mean(y_all_raw[:, :, tgt_idx][tr_mask_ct] ** 2, axis=0))
    top30_mask = np.zeros(len(tgt_idx), dtype=bool)
    top30_mask[np.argsort(-src_tgt_rms)[:30]] = True

    u_all = _interp_tf_stage_grid(y_all_raw[:, :, sel_tf] / tf_sc[None, None, :])
    y8_all = y_all_raw[:, :, tgt_idx] / tg_sc[None, None, :]

    y31 = np.zeros((12, 31, len(tgt_idx)), dtype=np.float64)
    tr_m31 = np.zeros((12, 31, len(tgt_idx)), dtype=np.float64)
    va_m31 = np.zeros((12, 31, len(tgt_idx)), dtype=np.float64)
    for p, st in enumerate(OBS_STEP_INDICES):
        k31 = st // 2
        y31[:, k31, :] = y8_all[:, p, :]
        tr_m31[:, k31, :] = tr_mask_ct[:, p : p + 1].astype(np.float64)
        va_m31[:, k31, :] = va_mask_ct[:, p : p + 1].astype(np.float64)
    for p in (5, 6, 7):
        tr_m31[:, OBS_STEP_INDICES[p] // 2, :] *= 2.0

    fam = DynamicalModelFamily(fam_val)
    if fam == DynamicalModelFamily.MEMGRN_ORTHO:
        base_spec = ModelSpec(
            family=DynamicalModelFamily.MEMGRN,
            n_sources=len(sel_tf),
            n_targets=len(tgt_idx),
            n_channels=3,
            hidden_width=5,
            coop_rank=coop_rank,
            use_cooperativity=True,
            base_rates=(0.6, 2.0, 5.5),
            rate_min=0.05,
            rate_max=8.0,
        )
        base_fit = fit_dynamical_model(
            spec=base_spec,
            train_stage_u=u_all,
            train_y=y31,
            train_mask=tr_m31,
            val_stage_u=u_all,
            val_y=y31,
            val_mask=va_m31,
            dt=DT_DAYS,
            obs_stride=2,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2, 5e-2),
            total_updates=1000,
            lr_max=0.008,
            gate_shrinkage_grid=(0.25, 1.0),
            hysteresis_shrinkage_grid=(0.5, 3.0),
        )
        fitted = fit_two_stage_memgrn_ortho(
            base_memgrn=base_fit,
            train_stage_u=u_all,
            train_y=y31,
            train_mask=tr_m31,
            val_stage_u=u_all,
            val_y=y31,
            val_mask=va_m31,
            dt=DT_DAYS,
            obs_stride=2,
            latent_dim=14,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2),
            total_updates=900,
            lr_max=0.007,
        )
    else:
        spec = ModelSpec(
            family=fam,
            n_sources=len(sel_tf),
            n_targets=len(tgt_idx),
            n_channels=3,
            hidden_width=5,
            latent_dim=18,
            module_dim=18,
            coop_rank=coop_rank,
            use_cooperativity=use_coop,
            base_rates=(0.6, 2.0, 5.5),
            rate_min=0.05,
            rate_max=8.0,
        )
        fitted = fit_dynamical_model(
            spec=spec,
            train_stage_u=u_all,
            train_y=y31,
            train_mask=tr_m31,
            val_stage_u=u_all,
            val_y=y31,
            val_mask=va_m31,
            dt=DT_DAYS,
            obs_stride=2,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2, 5e-2),
            total_updates=1000,
            lr_max=0.008,
            gate_shrinkage_grid=(0.25, 1.0),
            hysteresis_shrinkage_grid=(0.5, 3.0),
        )

    obs_k31 = [s // 2 for s in OBS_STEP_INDICES]
    pred_8 = fitted.predict_fn(u_all, None)[test_rows][:, obs_k31, :]  # (3, 8, 76)
    te_y8 = y8_all[test_rows]

    wdr_pred = pred_8[:, 5:, :]
    wdr_true = te_y8[:, 5:, :]
    wdr_mse = float(np.mean((wdr_pred - wdr_true) ** 2))
    wdr_var = float(np.mean(wdr_true**2)) + 1e-8
    wdr_r2 = 1.0 - wdr_mse / wdr_var
    wdr_top30 = float(np.mean((wdr_pred[:, :, top30_mask] - wdr_true[:, :, top30_mask]) ** 2))
    wdr_r = float(np.corrcoef(wdr_pred.ravel(), wdr_true.ravel())[0, 1]) if np.std(wdr_pred) > 1e-8 else 0.0

    # Post-withdrawal peaking gene-condition pairs on held-out cell line
    y7 = te_y8[:, 4, :]
    peak_mask = (np.abs(y7) > 0.35) & (
        ((y7 > 0) & (np.maximum(te_y8[:, 5, :], te_y8[:, 6, :]) > y7 + 0.08))
        | ((y7 < 0) & (np.minimum(te_y8[:, 5, :], te_y8[:, 6, :]) < y7 - 0.08))
    )
    if np.any(peak_mask):
        pm_3 = np.broadcast_to(peak_mask[:, None, :], wdr_pred.shape)
        wdr_peak = float(np.mean(((wdr_pred - wdr_true) ** 2)[pm_3]))
    else:
        wdr_peak = wdr_mse

    # Per-ligand withdrawal MSE (3 conditions per fold -> 12 conditions across 4 folds!)
    per_cond_wdr = [float(np.mean((wdr_pred[k] - wdr_true[k]) ** 2)) for k in range(3)]

    # Counterfactual attribution vector on held-out cell line withdrawal (for drift audit)
    attr_fn = (
        fitted.predict_observed_only_fn
        if fitted.predict_observed_only_fn is not None
        else fitted.predict_fn
    )
    eff = counterfactual_source_attribution(attr_fn, u_all[:, test_rows, :, :], None)[:, :, obs_k31, :][:, :, 5:, :]
    attr_vec = eff.ravel()

    return (
        held_cl,
        model_key,
        {
            "n_params": fitted.n_params,
            "withdrawal_mse": wdr_mse,
            "withdrawal_r2": wdr_r2,
            "withdrawal_r": wdr_r,
            "top30_withdrawal_mse": wdr_top30,
            "post_withdrawal_peak_mse": wdr_peak,
            "per_ligand_wdr_mse": per_cond_wdr,
            "attr_vec": attr_vec.tolist(),
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_and_prepare_emt_withdrawal(RAW_DIR)

    design = {
        "benchmark": "emt_zero_shot_human_cancer_withdrawal_gse147405",
        "accession": "GSE147405",
        "reference": "Cook & Vanderhyden, Nature Communications 11, 2142 (2020)",
        "total_singlet_cells": data.total_singlet_cells,
        "cell_lines": list(data.cell_lines),
        "ligands": list(data.ligands),
        "n_selected_tfs": 12,
        "n_targets": len(data.genes) - len(data.tf_indices),
        "models": [cfg[0] for cfg in MODEL_CONFIGS],
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    jobs = [
        (cl_i, mk, fam.value, cr, uc)
        for cl_i in range(len(data.cell_lines))
        for (mk, fam, cr, uc) in MODEL_CONFIGS
    ]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=24, mp_context=ctx) as pool:
        raw_out = list(pool.map(_run_cellline_fold, jobs))
    elapsed = time.time() - t0

    by_model: dict[str, dict[str, dict]] = {mk: {} for mk, _, _, _ in MODEL_CONFIGS}
    attr_by_model: dict[str, dict[str, np.ndarray]] = {mk: {} for mk, _, _, _ in MODEL_CONFIGS}
    for cl, mk, res in raw_out:
        attr_by_model[mk][cl] = np.asarray(res.pop("attr_vec"), dtype=np.float64)
        by_model[mk][cl] = res

    # Compute observed-TF attribution drift relative to standalone MemGRN across all 4 folds
    summary: dict[str, dict] = {}
    for mk, _, _, _ in MODEL_CONFIGS:
        vals = [by_model[mk][cl] for cl in data.cell_lines]
        cond_12 = np.array([x for v in vals for x in v["per_ligand_wdr_mse"]], dtype=np.float64)
        drifts = []
        for cl in data.cell_lines:
            ref = attr_by_model["memgrn"][cl]
            cur = attr_by_model[mk][cl]
            denom = float(np.linalg.norm(ref)) + 1e-12
            drifts.append(float(np.linalg.norm(cur - ref) / denom))
        summary[mk] = {
            "n_params": vals[0]["n_params"],
            "withdrawal_mse": float(np.mean([v["withdrawal_mse"] for v in vals])),
            "withdrawal_mse_se": float(np.std(cond_12, ddof=1) / np.sqrt(len(cond_12))),
            "withdrawal_r2": float(np.mean([v["withdrawal_r2"] for v in vals])),
            "withdrawal_r": float(np.mean([v["withdrawal_r"] for v in vals])),
            "top30_withdrawal_mse": float(np.mean([v["top30_withdrawal_mse"] for v in vals])),
            "post_withdrawal_peak_mse": float(np.mean([v["post_withdrawal_peak_mse"] for v in vals])),
            "obs_tf_attr_drift": float(np.mean(drifts)),
        }

    # Paired Wilcoxon across the 12 (cell_line, ligand) held-out withdrawal curves vs memgrn_ortho & memgrn
    ref_ortho = np.array(
        [x for cl in data.cell_lines for x in by_model["memgrn_ortho"][cl]["per_ligand_wdr_mse"]]
    )
    ref_mem = np.array(
        [x for cl in data.cell_lines for x in by_model["memgrn"][cl]["per_ligand_wdr_mse"]]
    )
    wilcoxon_vs_mem: dict[str, dict] = {}
    for mk, _, _, _ in MODEL_CONFIGS:
        oth = np.array([x for cl in data.cell_lines for x in by_model[mk][cl]["per_ligand_wdr_mse"]])
        p_o = float(wilcoxon(ref_ortho, oth, alternative="less").pvalue) if np.any(ref_ortho != oth) else 1.0
        p_m = float(wilcoxon(ref_mem, oth, alternative="less").pvalue) if np.any(ref_mem != oth) else 1.0
        wilcoxon_vs_mem[mk] = {
            "ortho_wins_12": int(np.sum(ref_ortho < oth)),
            "ortho_wilcoxon_p": p_o,
            "memgrn_wins_12": int(np.sum(ref_mem < oth)),
            "memgrn_wilcoxon_p": p_m,
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "wilcoxon_12_conditions": wilcoxon_vs_mem,
        "by_cell_line": by_model,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": round(elapsed, 1), "summary": summary, "wilcoxon": wilcoxon_vs_mem}, indent=2))


if __name__ == "__main__":
    main()
