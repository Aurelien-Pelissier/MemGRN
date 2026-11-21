"""Real 4-TF Mammalian Macrophage Innate-Immune Benchmark (Cheng et al., 2017; Sheu et al., 2023).

Evaluates 7-fold out-of-sample stimulus generalization and counterfactual regulator
attribution across 14 innate immune stimuli, 4 measured regulators (NF-kB, IRF, AP-1, p38),
and 477 experimentally GRS-annotated target genes.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    DynamicalModelFamily,
    ModelSpec,
    counterfactual_source_attribution,
    fit_dynamical_model,
)
from memgrn.datasets import fetch_and_prepare_macrophage_multitf  # noqa: E402

OUT_DIR = ROOT / "results" / "macrophage_multitf"
RAW_DIR = ROOT / "data" / "raw" / "macrophage"

# 7 disjoint stimulus pairs covering all 14 stimuli; Fold 0 holds out (TNF, IFNB),
# which cleanly breaks TLR NF-kB/p38 and NF-kB/IRF correlations.
STIMULUS_FOLDS: tuple[tuple[str, str], ...] = (
    ("TNF", "IFNB"),
    ("FLA", "VSV"),
    ("LPS", "MCMV"),
    ("PAM", "PIC"),
    ("FSL", "tPIC"),
    ("R848", "LM"),
    ("CPG", "STREP"),
)

MODELS_TO_RUN: tuple[DynamicalModelFamily, ...] = (
    DynamicalModelFamily.LOCAL_CONST,
    DynamicalModelFamily.MEMGRN,
    DynamicalModelFamily.MEMGRN_INT,
    DynamicalModelFamily.COUPLED_CONST,
    DynamicalModelFamily.COUPLED_ADAPTIVE,
    DynamicalModelFamily.LTC_COUPLED,
    DynamicalModelFamily.LTC_LOCAL,
    DynamicalModelFamily.PERTURB_ODE,
)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    y_true = labels[order].astype(np.float64)
    n_pos = float(np.sum(y_true))
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5
    ranks = np.arange(1, len(y_true) + 1, dtype=np.float64)
    cum_neg = ranks - np.cumsum(y_true)
    return float(np.sum((1.0 - cum_neg / n_neg) * y_true) / n_pos)


def _run_fold_model(job: tuple[int, str]) -> tuple[int, str, dict]:
    fold_idx, fam_val = job
    fam = DynamicalModelFamily(fam_val)
    data = fetch_and_prepare_macrophage_multitf(RAW_DIR)

    n_stim = len(data.stimuli)
    n_genes = len(data.genes)
    held_pair = STIMULUS_FOLDS[fold_idx]
    val_pair = STIMULUS_FOLDS[(fold_idx + 1) % len(STIMULUS_FOLDS)]

    test_idx = [data.stimuli.index(s) for s in held_pair]
    val_idx = [data.stimuli.index(s) for s in val_pair]
    train_idx = [i for i in range(n_stim) if i not in test_idx and i not in val_idx]

    # Standardize each gene solely by training-stimulus RMS (leakage-free)
    train_raw = data.expression_by_stimulus[train_idx]  # (10, 4, 477)
    gene_scales = np.maximum(0.25, np.sqrt(np.mean(train_raw[:, 1:, :] ** 2, axis=(0, 1))))

    # Build hourly target arrays (batch, 9, n_genes) with mask=1 at hours (1, 3, 8)
    obs_hour_indices = (1, 3, 8)

    def _pack_targets(idxs: list[int]) -> tuple[np.ndarray, np.ndarray]:
        y_full = np.zeros((len(idxs), 9, n_genes), dtype=np.float64)
        mask = np.zeros((len(idxs), 9, n_genes), dtype=np.float64)
        scaled = data.expression_by_stimulus[idxs] / gene_scales[None, None, :]
        for k_pos, hr in enumerate(obs_hour_indices, start=1):
            y_full[:, hr, :] = scaled[:, k_pos, :]
            mask[:, hr, :] = 1.0
        return y_full, mask

    tr_y, tr_mask = _pack_targets(train_idx)
    va_y, va_mask = _pack_targets(val_idx)
    te_y, te_mask = _pack_targets(test_idx)

    tr_u = data.stage_inputs[:, train_idx, :, :]
    va_u = data.stage_inputs[:, val_idx, :, :]
    te_u = data.stage_inputs[:, test_idx, :, :]

    spec = ModelSpec(
        family=fam,
        n_sources=4,
        n_targets=n_genes,
        n_channels=3,
        hidden_width=6,
        module_dim=20,
    )
    fitted = fit_dynamical_model(
        spec=spec,
        train_stage_u=tr_u,
        train_y=tr_y,
        train_mask=tr_mask,
        val_stage_u=va_u,
        val_y=va_y,
        val_mask=va_mask,
        dt=data.dt_hours,
        obs_stride=data.obs_stride,
        seeds=(301, 402),
        reg_grid=(1e-5, 1e-4, 5e-4),
        total_updates=900,
        lr_max=0.008,
    )

    pred_te = fitted.predict_fn(te_u, None)
    # Extract predicted values at observed hours (1, 3, 8)
    pred_obs = pred_te[:, list(obs_hour_indices), :]  # (2, 3, 477)
    true_obs = te_y[:, list(obs_hour_indices), :]
    _ = te_mask

    per_stim_mse = {
        held_pair[s_k]: float(np.mean((pred_obs[s_k] - true_obs[s_k]) ** 2))
        for s_k in range(len(held_pair))
    }
    per_gene_sq = np.mean((pred_obs - true_obs) ** 2, axis=(0, 1))  # (477,)

    # Counterfactual regulator attribution across held-out stimuli
    effects = counterfactual_source_attribution(fitted.predict_fn, te_u, None)
    # RMS effect over observed hours (1, 3, 8): shape (4, 477)
    eff_obs = effects[:, :, list(obs_hour_indices), :]
    reg_scores = np.sqrt(np.mean(eff_obs**2, axis=(1, 2)))

    return (
        fold_idx,
        fam_val,
        {
            "n_params": fitted.n_params,
            "val_mse": fitted.best_val_loss,
            "per_stim_mse": per_stim_mse,
            "per_gene_sq": per_gene_sq.tolist(),
            "reg_scores": reg_scores.tolist(),
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_and_prepare_macrophage_multitf(RAW_DIR)

    design = {
        "benchmark": "macrophage_4tf_14stimuli_477genes",
        "repo": "signalingsystemslab/ResponseSpecificity",
        "commit": "5336be7e3fe121a29bee1c8e4f39fe3ecf667a85",
        "regulators": list(data.regulators),
        "stimuli": list(data.stimuli),
        "n_genes": len(data.genes),
        "folds": [list(f) for f in STIMULUS_FOLDS],
        "models": [m.value for m in MODELS_TO_RUN],
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    jobs = [
        (f_i, fam.value)
        for f_i in range(len(STIMULUS_FOLDS))
        for fam in MODELS_TO_RUN
    ]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=28) as pool:
        raw_results = list(pool.map(_run_fold_model, jobs))
    elapsed = time.time() - t0

    by_model: dict[str, list[dict]] = {m.value: [] for m in MODELS_TO_RUN}
    for _, fam_val, payload in raw_results:
        by_model[fam_val].append(payload)

    grs_arr = np.array(data.grs_labels)
    selective_stimuli = ("TNF", "IFNB", "FLA", "VSV")

    summary: dict[str, dict] = {}
    for fam in MODELS_TO_RUN:
        m_key = fam.value
        folds_list = by_model[m_key]
        stim_mse_map: dict[str, float] = {}
        for item in folds_list:
            stim_mse_map.update(item["per_stim_mse"])

        gene_mse = np.mean([np.asarray(it["per_gene_sq"]) for it in folds_list], axis=0)
        # Aggregate counterfactual regulator attribution across all 7 held-out folds (all 14 stimuli)
        agg_reg_scores = np.sqrt(
            np.mean([np.asarray(it["reg_scores"]) ** 2 for it in folds_list], axis=0)
        )  # (4, 477)

        global_auroc = _auroc(agg_reg_scores.ravel(), data.causal_regulator_mask.ravel())
        # Top-1 regulator precision: fraction of genes whose highest-attributed TF is a true GRS regulator
        top1_tf = np.argmax(agg_reg_scores, axis=0)
        top1_acc = float(
            np.mean([data.causal_regulator_mask[top1_tf[g], g] for g in range(len(data.genes))])
        )

        # Combinatorial cluster MSEs (NFkB&p38 AND-gate and NFkB|IRF OR-gate)
        and_gate_mse = float(np.mean(gene_mse[grs_arr == "NFkB&p38"]))
        or_gate_mse = float(np.mean(gene_mse[grs_arr == "NFkB|IRF"]))

        summary[m_key] = {
            "n_params": folds_list[0]["n_params"],
            "all_14_stimuli_mse": float(np.mean(list(stim_mse_map.values()))),
            "selective_decorrelated_mse": float(
                np.mean([stim_mse_map[s] for s in selective_stimuli])
            ),
            "nfkb_and_p38_cluster_mse": and_gate_mse,
            "nfkb_or_irf_cluster_mse": or_gate_mse,
            "regulator_attribution_auroc": global_auroc,
            "top1_regulator_accuracy": top1_acc,
            "per_stimulus_mse": stim_mse_map,
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "n_genes": len(data.genes),
        "summary": summary,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"design_sha256": design_sha, "elapsed_s": round(elapsed, 1), "summary": {k: {mk: mv for mk, mv in v.items() if mk != "per_stimulus_mse"} for k, v in summary.items()}}, indent=2))


if __name__ == "__main__":
    main()
