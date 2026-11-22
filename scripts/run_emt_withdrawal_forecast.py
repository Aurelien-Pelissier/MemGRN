"""Zero-Shot Human Cancer EMT Withdrawal Forecasting & Regulon Benchmark (GSE147405, 53,290 cells).

Evaluates real human 3-day ligand-withdrawal hysteresis, post-withdrawal peaking, and external
human TF->target regulon discovery on Cook & Vanderhyden (2020, Nature Communications, GSE147405:
53,290 QC-passed human single cells across 4 cancer cell lines [A549, DU145, MCF7, OVCA420],
3 EMT-inducing ligands [TGFB1, EGF, TNF], and 8 timepoints spanning 7-day induction [0d..7d] and
3-day ligand withdrawal [8h_rm, 1d_rm, 3d_rm]).

Protocol (4-fold Leave-One-Cell-Line-Withdrawal-Out, zero leakage of held-out withdrawal):
  - For each held-out human cancer cell line c in {A549, DU145, MCF7, OVCA420}:
      * Source cell lines: the other 3 cell lines (9 full induction + withdrawal curves).
      * Latin-square stratified source validation: 3 withdrawal curves (1 from EVERY source cell
        line and 1 from EVERY ligand), preventing single-cell-line validation collapse while keeping
        the held-out cell line's 3-day withdrawal 100% unseen.
      * Training: 7-day induction (8h..7d) of all 12 trajectories PLUS the remaining 6 source
        withdrawal curves (2 per source cell line, 2 per ligand).
      * Zero-shot test: NEVER-SEEN 3-day withdrawal (8h_rm, 1d_rm, 3d_rm) of held-out cell line c
        across all 3 ligands (TGFB1, EGF, TNF), scored on all 76 EMT targets, top-30 source-dynamic
        targets, delayed post-withdrawal peaking gene-condition pairs, and independent external human
        CollecTRI regulons (167 curated TF->target edges) + hg38 ChIP-Atlas binding (203 edges).
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
from memgrn.datasets import (  # noqa: E402
    fetch_and_prepare_emt_withdrawal,
    fetch_emt_collectri_and_chip_masks,
)

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


def _auroc_and_ap(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores)
    y_true = labels[order].astype(np.float64)
    n_pos = float(np.sum(y_true))
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5, 0.0
    cum_pos = np.cumsum(y_true)
    ranks = np.arange(1, len(y_true) + 1, dtype=np.float64)
    ap = float(np.sum((cum_pos / ranks) * y_true) / n_pos)
    auroc = float(np.sum((1.0 - (ranks - cum_pos) / n_neg) * y_true) / n_pos)
    return auroc, ap


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
    held_cl = data.cell_lines[cl_i]
    src_cls = [c for c in data.cell_lines if c != held_cl]

    test_rows = [i for i, (cl, _l) in enumerate(data.conditions) if cl == held_cl]
    # Latin-square stratified source validation: 1 withdrawal curve per source cell line & per ligand
    val_pairs = {(src_cls[k], data.ligands[(k + cl_i) % len(data.ligands)]) for k in range(3)}
    val_rows = [i for i, pair in enumerate(data.conditions) if pair in val_pairs]
    train_wdr_rows = [i for i in range(len(data.conditions)) if i not in test_rows and i not in val_rows]

    y_all_raw = data.delta_trajectories  # (12, 8, 112)
    n_tf_all = len(data.tf_indices)
    tgt_idx = np.arange(n_tf_all, len(data.genes), dtype=np.int64)

    tr_mask_ct = np.zeros((12, 8), dtype=bool)
    tr_mask_ct[:, 1:5] = True  # induction frames observed across all 12 conditions
    tr_mask_ct[train_wdr_rows, 5:] = True  # 6 source withdrawal curves (2/cell line, 2/ligand)
    va_mask_ct = np.zeros((12, 8), dtype=bool)
    va_mask_ct[val_rows, 5:] = True  # 3 stratified source withdrawal curves (1/cell line, 1/ligand)

    # Zero-leakage TF selection & RMS scaling from observed source frames ONLY
    src_obs_mask = tr_mask_ct | va_mask_ct
    tf_rms = np.sqrt(np.mean(y_all_raw[:, :, data.tf_indices][src_obs_mask] ** 2, axis=0))
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
    pred_full31 = fitted.predict_fn(u_all, None)[test_rows]  # (3, 31, 76)
    pred_8 = pred_full31[:, obs_k31, :]  # (3, 8, 76)
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

    per_cond_wdr = [float(np.mean((wdr_pred[k] - wdr_true[k]) ** 2)) for k in range(3)]
    per_cond_t30 = [
        float(np.mean((wdr_pred[k, :, top30_mask] - wdr_true[k, :, top30_mask]) ** 2))
        for k in range(3)
    ]

    # Counterfactual TF->target attribution across all 12 conditions (for external CollecTRI/ChIP scoring)
    attr_fn = (
        fitted.predict_observed_only_fn
        if fitted.predict_observed_only_fn is not None
        else fitted.predict_fn
    )
    eff_all = counterfactual_source_attribution(attr_fn, u_all, None)  # (12_tf, 12_cond, 31, 76)
    # Causal edge score matrix (12_tf, 76_target): mean |Delta_{i->j}(t)| across observed timepoints
    edge_mat = np.mean(np.abs(eff_all[:, :, obs_k31, :][:, :, 1:, :]), axis=(1, 2))
    attr_vec_test = eff_all[:, test_rows, :, :][:, :, obs_k31, :][:, :, 5:, :].ravel()

    out_d = {
        "n_params": fitted.n_params,
        "withdrawal_mse": wdr_mse,
        "withdrawal_r2": wdr_r2,
        "withdrawal_r": wdr_r,
        "top30_withdrawal_mse": wdr_top30,
        "post_withdrawal_peak_mse": wdr_peak,
        "per_ligand_wdr_mse": per_cond_wdr,
        "per_ligand_top30_mse": per_cond_t30,
        "edge_mat": edge_mat.tolist(),
        "attr_vec": attr_vec_test.tolist(),
        "selected_tfs": [data.genes[int(i)] for i in sel_tf],
    }
    if held_cl == "A549" and model_key in ("memgrn", "memgrn_ortho", "ltc_local", "neural_cde", "local_const"):
        out_d["a549_pred31"] = pred_full31.tolist()
        out_d["a549_true8"] = te_y8.tolist()
    return held_cl, model_key, out_d


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_and_prepare_emt_withdrawal(RAW_DIR)

    # Confirm the 12 source-dynamic EMT TFs & fetch SHA-256 cached external CollecTRI + ChIP-Atlas masks
    n_tf_all = len(data.tf_indices)
    tgt_names = tuple(data.genes[n_tf_all:])
    tf_rms = np.sqrt(np.mean(data.delta_trajectories[:, :, data.tf_indices] ** 2, axis=(0, 1)))
    sel_tf_names = tuple(data.genes[int(i)] for i in np.sort(data.tf_indices[np.argsort(-tf_rms)[:12]]))
    collectri_mask, chip_mask, ext_hashes = fetch_emt_collectri_and_chip_masks(
        RAW_DIR, sel_tf_names, tgt_names
    )

    design = {
        "benchmark": "emt_zero_shot_human_cancer_withdrawal_gse147405_latin_cv_collectri",
        "accession": "GSE147405",
        "reference": "Cook & Vanderhyden, Nature Communications 11, 2142 (2020)",
        "external_regulon_ground_truth": "CollecTRI (Muller-Dott et al., NAR 2023) + hg38 ChIP-Atlas (MACS2>=500)",
        "validation_split": "latin_square_stratified_3_source_cell_lines_x_3_ligands",
        "total_singlet_cells": data.total_singlet_cells,
        "cell_lines": list(data.cell_lines),
        "ligands": list(data.ligands),
        "selected_tfs": list(sel_tf_names),
        "n_targets": len(tgt_names),
        "n_collectri_edges": int(np.sum(collectri_mask)),
        "n_chip_edges": int(np.sum(chip_mask)),
        "models": [cfg[0] for cfg in MODEL_CONFIGS],
        "file_sha256": {**data.file_sha256, **ext_hashes},
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
    edge_by_model: dict[str, list[np.ndarray]] = {mk: [] for mk, _, _, _ in MODEL_CONFIGS}
    traj_export: dict[str, np.ndarray] = {
        "obs_days": np.asarray(data.obs_days, dtype=np.float64),
        "t31_days": np.arange(31, dtype=np.float64) * (2.0 * DT_DAYS),
        "targets": np.asarray(tgt_names),
        "ligands": np.asarray(data.ligands),
    }
    for cl, mk, res in raw_out:
        attr_by_model[mk][cl] = np.asarray(res.pop("attr_vec"), dtype=np.float64)
        edge_by_model[mk].append(np.asarray(res.pop("edge_mat"), dtype=np.float64))
        res.pop("selected_tfs", None)
        if "a549_pred31" in res:
            traj_export[f"a549_pred31_{mk}"] = np.asarray(res.pop("a549_pred31"), dtype=np.float64)
            traj_export["a549_true8"] = np.asarray(res.pop("a549_true8"), dtype=np.float64)
        by_model[mk][cl] = res
    np.savez(OUT_DIR / "example_trajectories.npz", **traj_export)

    col_base_rate = float(np.mean(collectri_mask))
    chip_base_rate = float(np.mean(chip_mask))

    summary: dict[str, dict] = {}
    for mk, _, _, _ in MODEL_CONFIGS:
        vals = [by_model[mk][cl] for cl in data.cell_lines]
        cond_12 = np.array([x for v in vals for x in v["per_ligand_wdr_mse"]], dtype=np.float64)
        t30_12 = np.array([x for v in vals for x in v["per_ligand_top30_mse"]], dtype=np.float64)

        # Average counterfactual edge attribution across all 4 folds -> (12, 76)
        mean_edge = np.mean(np.stack(edge_by_model[mk], axis=0), axis=0)
        # Column-normalized per target (standard regulon ranking across candidate TFs)
        col_norm = mean_edge / np.maximum(1e-8, np.max(mean_edge, axis=0, keepdims=True))
        col_auroc_raw, col_ap_raw = _auroc_and_ap(mean_edge.ravel(), collectri_mask.ravel())
        col_auroc, col_ap = _auroc_and_ap(col_norm.ravel(), collectri_mask.ravel())
        chip_auroc, chip_ap = _auroc_and_ap(col_norm.ravel(), chip_mask.ravel())

        # Exact attribution drift relative to standalone MemGRN (primarily to audit MemGRN-Ortho invariance)
        drifts = []
        for cl in data.cell_lines:
            ref = attr_by_model["memgrn"][cl]
            cur = attr_by_model[mk][cl]
            drifts.append(float(np.linalg.norm(cur - ref) / (float(np.linalg.norm(ref)) + 1e-12)))

        summary[mk] = {
            "n_params": vals[0]["n_params"],
            "withdrawal_mse": float(np.mean([v["withdrawal_mse"] for v in vals])),
            "withdrawal_mse_se": float(np.std(cond_12, ddof=1) / np.sqrt(len(cond_12))),
            "withdrawal_r2": float(np.mean([v["withdrawal_r2"] for v in vals])),
            "withdrawal_r": float(np.mean([v["withdrawal_r"] for v in vals])),
            "top30_withdrawal_mse": float(np.mean([v["top30_withdrawal_mse"] for v in vals])),
            "top30_withdrawal_mse_se": float(np.std(t30_12, ddof=1) / np.sqrt(len(t30_12))),
            "post_withdrawal_peak_mse": float(np.mean([v["post_withdrawal_peak_mse"] for v in vals])),
            "collectri_auroc": col_auroc,
            "collectri_ap": col_ap,
            "collectri_ap_over_random": float(col_ap / max(1e-8, col_base_rate)),
            "collectri_raw_auroc": col_auroc_raw,
            "collectri_raw_ap_over_random": float(col_ap_raw / max(1e-8, col_base_rate)),
            "chip_auroc": chip_auroc,
            "chip_ap_over_random": float(chip_ap / max(1e-8, chip_base_rate)),
            "obs_tf_attr_drift_vs_memgrn": float(np.mean(drifts)),
        }

    # Paired Wilcoxon across the 12 (cell_line, ligand) held-out withdrawal curves
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
