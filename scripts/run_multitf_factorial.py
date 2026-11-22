"""Locked 2x2 Multi-TF Correlation-Shift Benchmark + LTC and PerturbODE Comparators.

Isolates source-isolated memory from adaptive kinetics across 24 fresh 8-TF, 24-target
out-of-class biochemical worlds (12 additive buffering + 12 cooperative enhancer worlds)
under correlation-breaking test interventions.
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
from scipy.stats import t as student_t, wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    DynamicalModelFamily,
    ModelSpec,
    counterfactual_source_attribution,
    fit_dynamical_model,
)
from memgrn.multitf_benchmark import generate_multitf_world  # noqa: E402

OUT_DIR = ROOT / "results" / "multitf_factorial"

MODELS_TO_RUN: tuple[DynamicalModelFamily, ...] = (
    DynamicalModelFamily.LOCAL_CONST,
    DynamicalModelFamily.MEMGRN,
    DynamicalModelFamily.MEMGRN_ADDITIVE,
    DynamicalModelFamily.COUPLED_CONST,
    DynamicalModelFamily.COUPLED_ADAPTIVE,
    DynamicalModelFamily.LTC_COUPLED,
    DynamicalModelFamily.LTC_LOCAL,
    DynamicalModelFamily.SELECTIVE_SSM,
    DynamicalModelFamily.CFC,
    DynamicalModelFamily.NEURAL_CDE,
    DynamicalModelFamily.PERTURB_ODE,
)


def _auroc_and_ap(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute exact AUROC and average precision for binary labels."""
    order = np.argsort(-scores)
    y_true = labels[order].astype(np.float64)
    n_pos = float(np.sum(y_true))
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5, 0.0
    cum_pos = np.cumsum(y_true)
    ranks = np.arange(1, len(y_true) + 1, dtype=np.float64)
    ap = float(np.sum((cum_pos / ranks) * y_true) / n_pos)
    # Mann-Whitney U statistic for AUROC
    cum_neg = ranks - cum_pos
    auroc = float(np.sum((1.0 - cum_neg / n_neg) * y_true) / n_pos)
    return auroc, ap


def _evaluate_one_world(job: tuple[int, str]) -> dict:
    seed, regime = job
    world = generate_multitf_world(seed=seed, regime=regime, n_sources=8, n_targets=24)

    train_mask = np.ones_like(world.train_y)
    train_mask[:, 0, :] = 0.0
    val_mask = np.ones_like(world.val_y)
    val_mask[:, 0, :] = 0.0

    world_res: dict[str, dict] = {}
    cats = np.array(world.test_categories)

    for fam in MODELS_TO_RUN:
        spec = ModelSpec(
            family=fam,
            n_sources=world.n_sources,
            n_targets=world.n_targets,
            n_channels=3,
            hidden_width=6,
            module_dim=14,
        )
        fitted = fit_dynamical_model(
            spec=spec,
            train_stage_u=world.train_stage_u,
            train_y=world.train_y,
            train_mask=train_mask,
            val_stage_u=world.val_stage_u,
            val_y=world.val_y,
            val_mask=val_mask,
            dt=world.dt,
            obs_stride=world.obs_stride,
            seeds=(101, 202),
            reg_grid=(1e-5, 1e-4, 1e-3),
            total_updates=1000,
            lr_max=0.008,
        )
        pred_test = fitted.predict_fn(world.test_stage_u, None)
        sq_err = (pred_test[:, 1:, :] - world.test_y[:, 1:, :]) ** 2
        test_mse = float(np.mean(sq_err))
        cat_mse = {
            cat_name: float(np.mean(sq_err[cats == cat_name]))
            for cat_name in ("single_ko_in_module", "phase_lag_shift", "cross_module_combo")
        }

        # Common model-agnostic counterfactual source-intervention attribution
        est_effects = counterfactual_source_attribution(
            fitted.predict_fn, world.test_stage_u, None
        )
        attr_diff = est_effects[:, :, 1:, :] - world.test_true_effects[:, :, 1:, :]
        attr_rmse = float(np.sqrt(np.mean(attr_diff**2)))

        # False cross-talk RMSE on non-causal (source, target) pairs
        non_causal = ~world.parent_mask
        spurious_sq = (est_effects[:, :, 1:, :] ** 2)[
            non_causal[:, None, None, :].repeat(est_effects.shape[1], axis=1).repeat(
                est_effects.shape[2] - 1, axis=2
            )
        ]
        spurious_rmse = float(np.sqrt(np.mean(spurious_sq)))

        # Edge recovery from counterfactual attribution L2 norms
        edge_scores = np.sqrt(np.mean(est_effects[:, :, 1:, :] ** 2, axis=(1, 2)))
        auroc, ap = _auroc_and_ap(edge_scores.ravel(), world.parent_mask.ravel())

        world_res[fam.value] = {
            "n_params": fitted.n_params,
            "val_mse": fitted.best_val_loss,
            "chosen_reg": fitted.chosen_reg,
            "test_mse": test_mse,
            "category_mse": cat_mse,
            "attribution_rmse": attr_rmse,
            "spurious_crosstalk_rmse": spurious_rmse,
            "edge_auroc": auroc,
            "edge_ap": ap,
        }

    return {"seed": seed, "regime": regime, "models": world_res}


def _paired_ci(vals_a: list[float], vals_b: list[float]) -> dict[str, float | list[float] | int]:
    """Compute paired mean difference (A - B), 95% t-CI, and win count (A < B)."""
    arr_a = np.asarray(vals_a, dtype=np.float64)
    arr_b = np.asarray(vals_b, dtype=np.float64)
    diff = arr_a - arr_b
    n = len(diff)
    mean_d = float(np.mean(diff))
    se_d = float(np.std(diff, ddof=1) / np.sqrt(n))
    crit = float(student_t.ppf(0.975, df=n - 1))
    w_p = float(wilcoxon(arr_a, arr_b, alternative="less").pvalue) if np.any(diff != 0.0) else 1.0
    return {
        "mean_diff": mean_d,
        "pct_reduction_vs_b": float(100.0 * (1.0 - np.mean(arr_a) / np.mean(arr_b))),
        "ci_95": [mean_d - crit * se_d, mean_d + crit * se_d],
        "wins_a_lt_b": int(np.sum(arr_a < arr_b)),
        "wilcoxon_p": w_p,
        "n": n,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    design = {
        "benchmark": "locked_2x2_multitf_correlation_shift",
        "n_sources": 8,
        "n_targets": 24,
        "additive_seeds": list(range(5001, 5013)),
        "cooperative_seeds": list(range(6001, 6013)),
        "models": [m.value for m in MODELS_TO_RUN],
        "train_schedules": 28,
        "val_schedules": 8,
        "test_schedules": 24,
        "test_categories": [
            "single_ko_in_module",
            "phase_lag_shift",
            "cross_module_combo",
        ],
        "primary_outcomes": ["test_mse", "attribution_rmse"],
        "secondary_outcomes": ["spurious_crosstalk_rmse", "edge_ap", "edge_auroc"],
        "decision_rule": (
            "Unified source-local adaptive memory (memgrn) must reduce counterfactual "
            "attribution_rmse relative to coupled_adaptive and ltc_coupled while improving "
            "or matching held-out OOD intervention test_mse."
        ),
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    jobs: list[tuple[int, str]] = [
        *[(s, "additive") for s in design["additive_seeds"]],
        *[(s, "cooperative") for s in design["cooperative_seeds"]],
    ]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=24) as pool:
        world_outputs = list(pool.map(_evaluate_one_world, jobs))
    elapsed = time.time() - t0

    summary: dict[str, dict] = {}
    for regime in ("additive", "cooperative"):
        sub = [w for w in world_outputs if w["regime"] == regime]
        reg_means: dict[str, dict] = {}
        for fam in MODELS_TO_RUN:
            m_key = fam.value
            tm = np.array([w["models"][m_key]["test_mse"] for w in sub])
            ar = np.array([w["models"][m_key]["attribution_rmse"] for w in sub])
            ap_arr = np.array([w["models"][m_key]["edge_ap"] for w in sub])
            reg_means[m_key] = {
                "n_params": sub[0]["models"][m_key]["n_params"],
                "test_mse": float(np.mean(tm)),
                "test_mse_se": float(np.std(tm, ddof=1) / np.sqrt(len(tm))),
                "single_ko_mse": float(
                    np.mean(
                        [w["models"][m_key]["category_mse"]["single_ko_in_module"] for w in sub]
                    )
                ),
                "phase_lag_mse": float(
                    np.mean([w["models"][m_key]["category_mse"]["phase_lag_shift"] for w in sub])
                ),
                "cross_combo_mse": float(
                    np.mean(
                        [w["models"][m_key]["category_mse"]["cross_module_combo"] for w in sub]
                    )
                ),
                "attribution_rmse": float(np.mean(ar)),
                "attribution_rmse_se": float(np.std(ar, ddof=1) / np.sqrt(len(ar))),
                "spurious_crosstalk_rmse": float(
                    np.mean([w["models"][m_key]["spurious_crosstalk_rmse"] for w in sub])
                ),
                "edge_auroc": float(np.mean([w["models"][m_key]["edge_auroc"] for w in sub])),
                "edge_ap": float(np.mean(ap_arr)),
                "edge_ap_se": float(np.std(ap_arr, ddof=1) / np.sqrt(len(ap_arr))),
            }

        hero = "memgrn"
        comparisons = {
            f"{hero}_vs_coupled_adaptive_mse": _paired_ci(
                [w["models"][hero]["test_mse"] for w in sub],
                [w["models"]["coupled_adaptive"]["test_mse"] for w in sub],
            ),
            f"{hero}_vs_coupled_adaptive_attr": _paired_ci(
                [w["models"][hero]["attribution_rmse"] for w in sub],
                [w["models"]["coupled_adaptive"]["attribution_rmse"] for w in sub],
            ),
            f"{hero}_vs_ltc_coupled_mse": _paired_ci(
                [w["models"][hero]["test_mse"] for w in sub],
                [w["models"]["ltc_coupled"]["test_mse"] for w in sub],
            ),
            f"{hero}_vs_ltc_coupled_attr": _paired_ci(
                [w["models"][hero]["attribution_rmse"] for w in sub],
                [w["models"]["ltc_coupled"]["attribution_rmse"] for w in sub],
            ),
            f"{hero}_vs_selective_ssm_mse": _paired_ci(
                [w["models"][hero]["test_mse"] for w in sub],
                [w["models"]["selective_ssm"]["test_mse"] for w in sub],
            ),
            f"{hero}_vs_cfc_mse": _paired_ci(
                [w["models"][hero]["test_mse"] for w in sub],
                [w["models"]["cfc"]["test_mse"] for w in sub],
            ),
            f"{hero}_vs_neural_cde_mse": _paired_ci(
                [w["models"][hero]["test_mse"] for w in sub],
                [w["models"]["neural_cde"]["test_mse"] for w in sub],
            ),
            f"{hero}_vs_perturb_ode_mse": _paired_ci(
                [w["models"][hero]["test_mse"] for w in sub],
                [w["models"]["perturb_ode"]["test_mse"] for w in sub],
            ),
            f"{hero}_vs_local_const_mse": _paired_ci(
                [w["models"][hero]["test_mse"] for w in sub],
                [w["models"]["local_const"]["test_mse"] for w in sub],
            ),
        }
        summary[regime] = {"means": reg_means, "comparisons": comparisons}

    out_payload = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "worlds": world_outputs,
    }
    (OUT_DIR / "scores.json").write_text(
        json.dumps(out_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"design_sha256": design_sha, "elapsed_s": round(elapsed, 1), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
