"""Closed-Loop Longitudinal Human iPSC CRISPR Perturb-seq Benchmark (GSE213069).

Evaluates autonomous closed-loop regulatory dynamics (zero future TF trajectories supplied)
across 14,945 human iPSCs, 103 pluripotency genes (23 knocked-out pluripotency TFs + 80
downstream responder genes), and Days 2, 3, 4, 5 (Ishikawa et al., 2023), plus ChIP-seq
regulatory network recovery against human PSC ChIP-Atlas binding profiles:
  1. Task 1 (Unseen CRISPR Knockout Trajectory Forecasting, Day 2 -> Days 3, 4, 5):
     5-fold CV across the 23 pluripotency TFs; given nascent Day-2 state y(d=2) and ko_k=1
     for an unseen knocked-out TF, forecast autonomous 103-gene trajectories over Days 3-5.
  2. Task 2 (Longitudinal Day-5 Horizon Extrapolation from t=0):
     Train from t=0 on Days 2-4 across all 23 KOs, extrapolate zero-shot to held-out Day 5.
  3. Task 3 (Human PSC ChIP-seq Direct Network Discovery):
     Evaluate inferred 23-TF -> 103-gene causal graph against ChIP-Atlas human PSC binding.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
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
from sklearn.linear_model import Ridge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    fit_closed_loop_network,
)
from memgrn.datasets import fetch_and_prepare_hipsc_perturbseq  # noqa: E402

OUT_DIR = ROOT / "results" / "hipsc_closed_loop"
RAW_DIR = ROOT / "data" / "raw" / "hipsc_perturbseq"

CLOSED_LOOP_FAMILIES: tuple[ClosedLoopModelFamily, ...] = (
    ClosedLoopModelFamily.MEMGRN,
    ClosedLoopModelFamily.LOCAL_CONST,
    ClosedLoopModelFamily.LTC_LOCAL,
    ClosedLoopModelFamily.COUPLED_ADAPTIVE,
    ClosedLoopModelFamily.LTC_COUPLED,
    ClosedLoopModelFamily.PERTURB_ODE,
    ClosedLoopModelFamily.LATENT_ODE,
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


def _fit_renge_power_baseline(
    train_ko: np.ndarray,
    train_y: np.ndarray,
    basal: np.ndarray,
    reg_indices: np.ndarray,
    n_powers: int = 3,
) -> tuple[np.ndarray, callable]:
    """Fit RENGE static knockout power-series propagation y(t) = sum_{m=1}^K w_{tm} A^m X_ko."""
    n_tr, n_days, n_g = train_y.shape
    x_ko = np.zeros_like(train_ko)
    x_ko[:, reg_indices] = -train_ko[:, reg_indices] * basal[None, reg_indices]
    y_mean = np.mean(train_y, axis=1)
    ridge = Ridge(alpha=2.0, fit_intercept=False)
    ridge.fit(x_ko, y_mean)
    a_mat = ridge.coef_.copy() * (1.0 - np.eye(n_g))
    norm_a = np.linalg.norm(a_mat, ord=2)
    if norm_a > 0.85:
        a_mat = a_mat * (0.85 / norm_a)

    powers_tr = []
    cur = x_ko @ a_mat.T
    for _ in range(n_powers):
        powers_tr.append(cur)
        cur = (cur * (1.0 - train_ko)) @ a_mat.T
    p_stack = np.stack(powers_tr, axis=-1)

    day_weights = np.zeros((n_days, n_powers), dtype=np.float64)
    for d_i in range(n_days):
        feat_d = p_stack.reshape(n_tr * n_g, n_powers)
        tgt_d = train_y[:, d_i, :].reshape(n_tr * n_g)
        day_weights[d_i], _, _, _ = np.linalg.lstsq(
            feat_d.T @ feat_d + 1e-2 * np.eye(n_powers), feat_d.T @ tgt_d, rcond=None
        )

    def _predict(ko_test: np.ndarray) -> np.ndarray:
        xk = np.zeros_like(ko_test)
        xk[:, reg_indices] = -ko_test[:, reg_indices] * basal[None, reg_indices]
        p_list = []
        c = xk @ a_mat.T
        for _ in range(n_powers):
            p_list.append(c)
            c = (c * (1.0 - ko_test)) @ a_mat.T
        ps = np.stack(p_list, axis=-1)
        return np.einsum("bgm,dm->bdg", ps, day_weights)

    return np.abs(a_mat.T), _predict


def _run_unseen_ko_fold(job: tuple[int, str, list[list[int]]]) -> tuple[int, str, dict]:
    """Task 1: Given nascent Day-2 state y(d=2) & ko_k=1 of unseen KO, predict Days 3, 4, 5."""
    fold_i, fam_val, folds = job
    data = fetch_and_prepare_hipsc_perturbseq(RAW_DIR)
    n_ko = len(data.ko_genes)
    n_genes = len(data.genes)
    reg_tuple = tuple(int(i) for i in data.ko_gene_indices)

    test_idx = folds[fold_i]
    val_idx = folds[(fold_i + 1) % len(folds)]
    train_idx = [i for i in range(n_ko) if i not in test_idx and i not in val_idx]

    tr_raw = data.ko_delta_trajectories[train_idx]
    gene_scales = np.maximum(0.08, np.sqrt(np.mean(tr_raw**2, axis=(0, 1))))
    basal_scaled = data.basal_expression / gene_scales

    def _pack(idxs: list[int]):
        ko_b = data.ko_binary_masks[idxs]
        y_d = data.ko_delta_trajectories[idxs] / gene_scales[None, None, :]  # (B, 4, 103) at d=2,3,4,5
        y_d2 = y_d[:, 0, :]  # Initial post-transduction state at Day 2
        y_full = np.concatenate([np.zeros((len(idxs), 1, n_genes)), y_d], axis=1)
        # Evaluate strictly on future autonomous days d=3, 4, 5 (indices 2, 3, 4; d=0 and d=2 masked!)
        w_full = np.concatenate(
            [np.zeros((len(idxs), 1, n_genes)), data.eval_target_masks[idxs]], axis=1
        )
        w_full[:, 1, :] = 0.0  # Day 2 is supplied as initial state y0!
        return ko_b, y_d2, y_full, w_full

    tr_ko, tr_y0, tr_y, tr_w = _pack(train_idx)
    va_ko, va_y0, va_y, va_w = _pack(val_idx)
    te_ko, te_y0, te_y, te_w = _pack(test_idx)

    if fam_val == "renge":
        # Static knockout power-series RENGE has zero column weights for unseen KOs;
        # evaluate standard RENGE prediction on Days 3, 4, 5
        _, pred_fn = _fit_renge_power_baseline(
            tr_ko, tr_y[:, 1:, :], basal_scaled, data.ko_gene_indices
        )
        pred_te = pred_fn(te_ko)[:, 1:, :]  # Days 3, 4, 5
        true_d = te_y[:, 2:, :]
        mask_d = te_w[:, 2:, :]
        mse = float(np.sum(((pred_te - true_d) ** 2) * mask_d) / np.sum(mask_d))
        return fold_i, fam_val, {"mse": mse, "pearson_r": 0.0, "n_params": len(reg_tuple) * n_genes + 12}

    fam = ClosedLoopModelFamily(fam_val)
    spec = ClosedLoopSpec(
        family=fam,
        n_genes=n_genes,
        regulator_indices=reg_tuple,
        n_channels=3,
        hidden_width=6,
        module_dim=20,
        latent_dim=20,
    )
    # Warm-start ramp t=0->Day 2 (steps 0..8, clamp_steps=8) loads pre-Day-2 intracellular memory,
    # then releases y(t) from y(Day 2) for autonomous closed-loop integration across Days 3, 4, 5!
    fitted = fit_closed_loop_network(
        spec=spec,
        basal_levels=basal_scaled,
        train_ko=tr_ko,
        train_y=tr_y,
        train_weights=tr_w,
        val_ko=va_ko,
        val_y=va_y,
        val_weights=va_w,
        dt=0.25,
        n_steps=20,
        obs_step_indices=(0, 8, 12, 16, 20),
        train_y0=tr_y0,
        val_y0=va_y0,
        clamp_steps=8,
        seeds=(101, 202),
        reg_grid=(1e-3, 5e-3, 2e-2),
        total_updates=800,
        lr_max=0.006,
    )
    pred_full = fitted.predict_ko_fn(te_ko, te_y0)  # (B, 5, 103)
    pred_d = pred_full[:, 2:, :]  # Days 3, 4, 5
    true_d = te_y[:, 2:, :]
    mask_d = te_w[:, 2:, :]
    mse = float(np.sum(((pred_d - true_d) ** 2) * mask_d) / np.sum(mask_d))

    corrs = []
    for b_i in range(len(test_idx)):
        valid = mask_d[b_i, 0, :] > 0.5
        tb = true_d[b_i, :, valid].ravel()
        pb = pred_d[b_i, :, valid].ravel()
        corrs.append(0.0 if np.std(pb) < 1e-8 else float(np.corrcoef(tb, pb)[0, 1]))

    return (
        fold_i,
        fam_val,
        {
            "mse": mse,
            "pearson_r": float(np.mean(corrs)),
            "n_params": fitted.n_params,
        },
    )


def _run_day5_and_chip_model(fam_val: str) -> tuple[str, dict]:
    """Task 2 (t=0 -> Day 5 horizon extrapolation) & Task 3 (Human PSC ChIP-seq GRN recovery)."""
    data = fetch_and_prepare_hipsc_perturbseq(RAW_DIR)
    n_ko = len(data.ko_genes)
    n_genes = len(data.genes)
    reg_tuple = tuple(int(i) for i in data.ko_gene_indices)

    gene_scales = np.maximum(
        0.08, np.sqrt(np.mean(data.ko_delta_trajectories[:, :3, :] ** 2, axis=(0, 1)))
    )
    basal_scaled = data.basal_expression / gene_scales
    y_scaled = data.ko_delta_trajectories / gene_scales[None, None, :]
    y_full = np.concatenate([np.zeros((n_ko, 1, n_genes)), y_scaled], axis=1)
    m_full = np.concatenate([np.zeros((n_ko, 1, n_genes)), data.eval_target_masks], axis=1)

    # Train strictly on Days 2-3 + 17/23 KOs on Day 4; validate on Day 4 of 6 KOs; test on Day 5 of ALL 23 KOs!
    tr_w = m_full.copy()
    tr_w[:, 4, :] = 0.0
    val_ko_subset = [0, 4, 8, 12, 16, 20]
    tr_w[val_ko_subset, 3, :] = 0.0
    va_w = np.zeros_like(m_full)
    va_w[val_ko_subset, 3, :] = m_full[val_ko_subset, 3, :]

    if fam_val == "renge":
        _, pred_fn3 = _fit_renge_power_baseline(
            data.ko_binary_masks, y_scaled[:, :3, :], basal_scaled, data.ko_gene_indices
        )
        pred_234 = pred_fn3(data.ko_binary_masks)
        pred_day5 = pred_234[:, 2, :] + (pred_234[:, 2, :] - pred_234[:, 1, :])
        d5_mask = m_full[:, 4, :]
        day5_mse = float(np.sum(((pred_day5 - y_full[:, 4, :]) ** 2) * d5_mask) / np.sum(d5_mask))
        graph = data.renge_reference_graph
        n_p = len(reg_tuple) * n_genes + 17
    else:
        fam = ClosedLoopModelFamily(fam_val)
        spec = ClosedLoopSpec(
            family=fam,
            n_genes=n_genes,
            regulator_indices=reg_tuple,
            n_channels=3,
            hidden_width=6,
            module_dim=20,
            latent_dim=20,
        )
        fitted_d5 = fit_closed_loop_network(
            spec=spec,
            basal_levels=basal_scaled,
            train_ko=data.ko_binary_masks,
            train_y=y_full,
            train_weights=tr_w,
            val_ko=data.ko_binary_masks,
            val_y=y_full,
            val_weights=va_w,
            dt=0.25,
            n_steps=20,
            obs_step_indices=(0, 8, 12, 16, 20),
            seeds=(101, 202),
            reg_grid=(1e-3, 5e-3, 2e-2),
            total_updates=850,
            lr_max=0.006,
        )
        pred_all = fitted_d5.predict_ko_fn(data.ko_binary_masks)
        d5_mask = m_full[:, 4, :]
        day5_mse = float(
            np.sum(((pred_all[:, 4, :] - y_full[:, 4, :]) ** 2) * d5_mask) / np.sum(d5_mask)
        )

        # Task 3: Full 4-day closed-loop network fit for ChIP-Atlas PSC network discovery
        tr_all_w = m_full.copy()
        tr_all_w[val_ko_subset, 4, :] = 0.0
        va_all_w = np.zeros_like(m_full)
        va_all_w[val_ko_subset, 4, :] = m_full[val_ko_subset, 4, :]
        fitted_full = fit_closed_loop_network(
            spec=spec,
            basal_levels=basal_scaled,
            train_ko=data.ko_binary_masks,
            train_y=y_full,
            train_weights=tr_all_w,
            val_ko=data.ko_binary_masks,
            val_y=y_full,
            val_weights=va_all_w,
            dt=0.25,
            n_steps=20,
            obs_step_indices=(0, 8, 12, 16, 20),
            seeds=(101, 202),
            reg_grid=(1e-3, 5e-3, 2e-2),
            total_updates=900,
            lr_max=0.006,
        )
        graph = fitted_full.interaction_matrix
        n_p = fitted_full.n_params

    # Normalize target columns across the 23 TFs (following Ishikawa et al. example_hiPSC.ipynb Cell 7)
    sub_g = graph[data.ko_gene_indices, :].copy()  # (23, 103)
    col_max = np.maximum(1e-8, np.max(sub_g, axis=0, keepdims=True))
    norm_graph = np.zeros_like(graph)
    norm_graph[data.ko_gene_indices, :] = sub_g / col_max

    psc_rows = data.chip_psc_tf_indices
    psc_scores_list = []
    psc_labels_list = []
    per_tf_aurocs = []
    for tf_i in psc_rows:
        valid_tgt = np.arange(n_genes) != tf_i
        sc_row = norm_graph[tf_i, valid_tgt]
        sc_z = (sc_row - np.mean(sc_row)) / (np.std(sc_row) + 1e-12)
        lb_row = data.chip_psc_edge_mask[tf_i, valid_tgt]
        psc_scores_list.append(sc_z)
        psc_labels_list.append(lb_row)
        au_tf, _ = _auroc_and_ap(sc_row, lb_row)
        per_tf_aurocs.append(au_tf)

    pooled_sc = np.concatenate(psc_scores_list)
    pooled_lb = np.concatenate(psc_labels_list)
    chip_psc_auroc, chip_psc_ap = _auroc_and_ap(pooled_sc, pooled_lb)
    base_rate = float(np.mean(pooled_lb))

    any_rows = data.chip_any_tf_indices
    any_au_list = []
    for tf_i in any_rows:
        valid_tgt = np.arange(n_genes) != tf_i
        au_a, _ = _auroc_and_ap(
            norm_graph[tf_i, valid_tgt], data.chip_any_edge_mask[tf_i, valid_tgt]
        )
        any_au_list.append(au_a)

    return (
        fam_val,
        {
            "n_params": n_p,
            "day5_extrapolation_mse": day5_mse,
            "chip_psc_pooled_auroc": chip_psc_auroc,
            "chip_psc_mean_tf_auroc": float(np.mean(per_tf_aurocs)),
            "chip_psc_ap": chip_psc_ap,
            "chip_psc_ap_over_random": float(chip_psc_ap / max(1e-8, base_rate)),
            "chip_any_mean_tf_auroc": float(np.mean(any_au_list)),
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_and_prepare_hipsc_perturbseq(RAW_DIR)

    n_ko = len(data.ko_genes)
    folds = [list(range(f, n_ko, 5)) for f in range(5)]
    all_model_keys = [*[m.value for m in CLOSED_LOOP_FAMILIES], "renge"]

    design = {
        "benchmark": "hipsc_closed_loop_longitudinal_perturbseq_gse213069",
        "accession": "GSE213069",
        "reference": "Ishikawa et al., Communications Biology 6, 1290 (2023)",
        "repo": "masastat/RENGE",
        "commit": "473b4f9b33072d84bbae3b730d3998434370714b",
        "n_cells": 14945,
        "n_genes": len(data.genes),
        "n_ko_tfs": n_ko,
        "ko_genes": list(data.ko_genes),
        "obs_days": [2, 3, 4, 5],
        "chip_psc_tfs": [data.genes[i] for i in data.chip_psc_tf_indices],
        "unseen_ko_folds": folds,
        "models": all_model_keys,
        "future_tf_traces_supplied": False,
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    ko_jobs = [(f_i, m_key, folds) for f_i in range(5) for m_key in all_model_keys]
    with ProcessPoolExecutor(max_workers=24) as pool:
        ko_raw = list(pool.map(_run_unseen_ko_fold, ko_jobs))
        d5_chip_raw = dict(pool.map(_run_day5_and_chip_model, all_model_keys))
    elapsed = time.time() - t0

    by_model_ko: dict[str, list[dict]] = {m: [] for m in all_model_keys}
    for _, m_key, res_d in ko_raw:
        by_model_ko[m_key].append(res_d)

    summary: dict[str, dict] = {}
    for m_key in all_model_keys:
        ko_list = by_model_ko[m_key]
        extra = d5_chip_raw[m_key]
        summary[m_key] = {
            "n_params": extra["n_params"],
            "unseen_ko_mse": float(np.mean([x["mse"] for x in ko_list])),
            "unseen_ko_pearson_r": float(np.mean([x["pearson_r"] for x in ko_list])),
            "day5_extrapolation_mse": extra["day5_extrapolation_mse"],
            "chip_psc_pooled_auroc": extra["chip_psc_pooled_auroc"],
            "chip_psc_mean_tf_auroc": extra["chip_psc_mean_tf_auroc"],
            "chip_psc_ap": extra["chip_psc_ap"],
            "chip_psc_ap_over_random": extra["chip_psc_ap_over_random"],
            "chip_any_mean_tf_auroc": extra["chip_any_mean_tf_auroc"],
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "summary": summary,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
