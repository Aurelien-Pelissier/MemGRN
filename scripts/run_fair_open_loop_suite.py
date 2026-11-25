"""Strictly Equal-Budget Fair Open-Loop Benchmark Suite (Addressing R1-M1/M6/M7 & R2-M1..M7).

Runs ALL open-loop benchmarks under strictly identical validation budgets (zero MEMGRN-only
shrinkage-grid inflation), diagonal-free cooperativity, matched cooperativity (+Gamma) across
all source-isolated architectures, the new Depthwise (source-isolated) Selective SSM ablation,
a genuinely jointly trained unprojected cofactor ODE (MEMGRN_UNPROJ), uniform single-backbone
MemGRN-Ortho (LATENT_ODE Stage 2, global scalar beta, no multi-family ensemble), both Stage-1
and full-model counterfactual attribution audits, and two-sided Holm-Bonferroni Wilcoxon tests:

  1. Locked 4-Season UCI Beijing Air Quality (raw u AND anchored Delta u = u(t) - u(0),
     506 episodes, plus 6-driver Partial Observability).
  2. Equal-Budget 24-World Multi-TF Correlation-Shift (12 additive + 12 cooperative) with
     matched +Gamma controls across MemGRN, Local Constant, Source-Local LTC, & Depthwise SSM.
  3. Ground-Truth 6-of-8 Partial Observability (12 worlds) comparing MemGRN, true joint
     MEMGRN_UNPROJ, uniform two-stage MemGRN-Ortho, and joint Col-Proj MemGRN-Ortho on BOTH
     Stage-1 observed-branch AND full-model attribution + Part B LTC/Depthwise-SSM separation.
  4. Clean Human EMT Withdrawal (GSE147405, strict source-cell-line-only TF selection, equal
     8-trial budget, uniform single-spec MemGRN-Ortho with direct Stage-1 CollecTRI/ChIP).
  5. Equal-Budget Binance 16-Asset & 14-Altcoin Partial Observability (all ODEs + Depthwise SSM
     + Discrete DLinear/GRU/Transformer + uniform MemGRN-Ortho & joint MEMGRN_UNPROJ).
  6. Equal-Budget Yeast Msn2 Joint Models (16 single pulses -> 10 pulse trains) transparently
     reporting CV-selected asymmetric memgrn, memgrn_sym, Depthwise SSM, and x/10 train wins.
  7. Complete S=8..512 Scaling Benchmark (dense MemGRN, rank-24 MemGRN, rank-24 Depthwise SSM,
     Coupled Adaptive, Coupled LTC reporting mem_params, total_params, and ms/step).
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    DynamicalModelFamily,
    ModelSpec,
    build_closed_loop_jax_rollout,
    counterfactual_source_attribution,
    fit_dynamical_model,
    fit_two_stage_memgrn_ortho,
)
from memgrn.cross_domain_datasets import (  # noqa: E402
    load_beijing_air_quality,
    load_binance_market_contagion,
)
from memgrn.datasets import (  # noqa: E402
    MSN2_PROMOTERS,
    build_rk4_stage_grid,
    fetch_and_prepare_emt_withdrawal,
    fetch_emt_collectri_and_chip_masks,
    load_archived_json,
    load_msn2_population_trace,
)
from memgrn.discrete_baselines import (  # noqa: E402
    DiscreteModelFamily,
    discrete_counterfactual_attribution,
    fit_discrete_open_loop,
)
from memgrn.multitf_benchmark import generate_multitf_world  # noqa: E402

OUT_DIR = ROOT / "results" / "fair_open_loop"
RAW_ROOT = ROOT / "data" / "raw"


def _build_multi_stage_grid(drivers: np.ndarray, substeps_per_knot: int) -> np.ndarray:
    b, t_len, n_s = drivers.shape
    frac = np.arange(substeps_per_knot, dtype=np.float64)[None, None, :, None] / substeps_per_knot
    lo = drivers[:, :-1, None, :]
    delta = np.diff(drivers, axis=1)[:, :, None, :]

    def _interp(offset: float) -> np.ndarray:
        vals = lo + delta * (frac + offset / substeps_per_knot)
        return vals.reshape(b, (t_len - 1) * substeps_per_knot, n_s).transpose(1, 0, 2)

    return np.stack([_interp(0.0), _interp(0.5), _interp(1.0)], axis=-1)


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


def _two_sided_holm_stats(ref: np.ndarray, others: dict[str, np.ndarray]) -> dict[str, dict]:
    """Compute exact two-sided Wilcoxon p-values, Holm-Bonferroni adjusted p-values, and win counts."""
    raw = {}
    keys = list(others.keys())
    p_list = []
    for k in keys:
        oth = others[k]
        diff = ref - oth
        if np.any(diff != 0.0):
            p_two = float(wilcoxon(ref, oth, alternative="two-sided").pvalue)
        else:
            p_two = 1.0
        p_list.append(p_two)
        raw[k] = {
            "pct_reduction": float(100.0 * (1.0 - np.mean(ref) / max(1e-12, np.mean(oth)))),
            "wins": int(np.sum(ref < oth)),
            "total": int(len(ref)),
            "wilcoxon_p_two_sided": p_two,
        }
    # Step-down Holm-Bonferroni adjustment
    m = len(keys)
    order = np.argsort(p_list)
    adj = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p_list[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    for idx, k in enumerate(keys):
        raw[k]["holm_p_two_sided"] = float(adj[idx])
    return raw


# =============================================================================
# Task 1: Locked 4-Season Beijing Air Quality (Raw u AND Anchored Delta u)
# =============================================================================
BQ_MODELS = (
    ("memgrn", "memgrn", 6, True),
    ("memgrn_additive", "memgrn_additive", None, False),
    ("local_const_coop", "local_const", 6, True),
    ("local_const", "local_const", None, False),
    ("ltc_local_coop", "ltc_local", 6, True),
    ("ltc_local", "ltc_local", None, False),
    ("depthwise_ssm_coop", "depthwise_ssm", 6, True),
    ("depthwise_ssm", "depthwise_ssm", None, False),
    ("coupled_adaptive", "coupled_adaptive", None, False),
    ("ltc_coupled", "ltc_coupled", None, False),
    ("selective_ssm", "selective_ssm", None, False),
    ("cfc", "cfc", None, False),
    ("neural_cde", "neural_cde", None, False),
    ("perturb_ode", "perturb_ode", None, False),
    ("latent_ode", "latent_ode", None, False),
    ("dlinear", "dlinear", None, False),
    ("gru", "gru", None, False),
    ("transformer", "transformer", None, False),
)

BQ_PO_MODELS = (
    ("memgrn", "memgrn", 6, True),
    ("memgrn_ortho", "memgrn_ortho_2stage", 6, True),
    ("memgrn_unproj", "memgrn_unproj", 6, True),
    ("depthwise_ssm_coop", "depthwise_ssm", 6, True),
    ("local_const_coop", "local_const", 6, True),
    ("latent_ode", "latent_ode", None, False),
    ("dlinear", "dlinear", None, False),
    ("gru", "gru", None, False),
    ("transformer", "transformer", None, False),
)


def _run_beijing_job(job: tuple) -> tuple[str, int, str, dict]:
    mode, sea_i, mk, fam_str, coop_rank, use_coop = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    data = load_beijing_air_quality(RAW_ROOT / "beijing_aq", episode_hours=48)
    test_idx = np.where(data.seasons == sea_i)[0]
    src_idx = np.where(data.seasons != sea_i)[0]
    val_idx = src_idx[4::5]
    train_idx = np.array([i for i in src_idx if i not in set(val_idx.tolist())], dtype=np.int64)

    partial_obs = mode == "po6"
    delta_u = mode == "delta_u"
    drv_cols = [0, 1, 2, 3, 4, 7] if partial_obs else list(range(8))
    n_s, n_t = len(drv_cols), len(data.stations)

    tr_d = data.drivers[train_idx][:, :, drv_cols]
    d_mu = np.mean(tr_d, axis=(0, 1), keepdims=True)
    d_sd = np.maximum(1e-3, np.std(tr_d, axis=(0, 1), keepdims=True))
    y_sc = np.maximum(0.25, np.sqrt(np.mean(data.targets[train_idx] ** 2, axis=(0, 1), keepdims=True)))

    def _prep(idxs: np.ndarray):
        u_obs = (data.drivers[idxs][:, :, drv_cols] - d_mu) / d_sd
        if delta_u:
            u_obs = u_obs - u_obs[:, :1, :]
        u_stage = _build_multi_stage_grid(u_obs, substeps_per_knot=2)
        y_n = data.targets[idxs] / y_sc
        m_n = np.ones_like(y_n)
        m_n[:, 0, :] = 0.0
        return u_obs, u_stage, y_n, m_n

    tr_uo, tr_u, tr_y, tr_m = _prep(train_idx)
    va_uo, va_u, va_y, va_m = _prep(val_idx)
    te_uo, te_u, te_y, _ = _prep(test_idx)
    dt_h, obs_st = 0.5, 2  # 2 substeps/hour -> 1 model unit = 1.0 h, fast & uniform

    # Strictly EQUAL validation budget: 2 seeds x 3 regs = 6 trials for EVERY model
    seeds = (101, 202)
    reg_grid = (1e-4, 1e-3, 1e-2)
    attr_full_mag = None
    attr_obs_mag = None

    if fam_str in ("dlinear", "gru", "transformer"):
        fit_d = fit_discrete_open_loop(
            family=DiscreteModelFamily(fam_str),
            train_u=tr_uo,
            train_y=tr_y,
            train_mask=tr_m,
            val_u=va_uo,
            val_y=va_y,
            val_mask=va_m,
            hidden_dim=32,
            seeds=seeds,
            reg_grid=reg_grid,
            total_updates=500,
            lr_max=0.005,
        )
        pred = fit_d.predict_fn(te_uo)
        n_params = fit_d.n_params
        if partial_obs:
            a_f = discrete_counterfactual_attribution(fit_d.predict_fn, te_uo[:16])
            attr_full_mag = np.sqrt(np.mean(a_f[:, :, 1:, :] ** 2, axis=(1, 2)))
            attr_obs_mag = attr_full_mag
    elif fam_str == "memgrn_ortho_2stage":
        base_spec = ModelSpec(
            family=DynamicalModelFamily.MEMGRN,
            n_sources=n_s,
            n_targets=n_t,
            n_channels=3,
            hidden_width=5,
            coop_rank=coop_rank,
            use_cooperativity=True,
            base_rates=(0.06, 0.25, 1.0),
            rate_min=0.015,
            rate_max=3.0,
        )
        base_fit = fit_dynamical_model(
            spec=base_spec,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=dt_h,
            obs_stride=obs_st,
            seeds=seeds,
            reg_grid=reg_grid,
            gate_shrinkage_grid=(1.0,),
            hysteresis_shrinkage_grid=(1.0,),
            total_updates=500,
            lr_max=0.007,
        )
        # Uniform single-spec MemGRN-Ortho: single LATENT_ODE Stage 2, global scalar beta
        fitted = fit_two_stage_memgrn_ortho(
            base_memgrn=base_fit,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=dt_h,
            obs_stride=obs_st,
            latent_dim=14,
            seeds=seeds,
            reg_grid=reg_grid,
            total_updates=450,
            lr_max=0.007,
            residual_families=(DynamicalModelFamily.LATENT_ODE,),
            per_target_blend=False,
        )
        pred = fitted.predict_fn(te_u, te_y[:, 0, :])
        n_params = fitted.n_params
        a_obs = counterfactual_source_attribution(fitted.predict_observed_only_fn, te_u[:, :16], te_y[:16, 0, :])
        a_tot = counterfactual_source_attribution(fitted.predict_fn, te_u[:, :16], te_y[:16, 0, :])
        attr_obs_mag = np.sqrt(np.mean(a_obs[:, :, 1:, :] ** 2, axis=(1, 2)))
        attr_full_mag = np.sqrt(np.mean(a_tot[:, :, 1:, :] ** 2, axis=(1, 2)))
    else:
        fam = DynamicalModelFamily(fam_str)
        spec = ModelSpec(
            family=fam,
            n_sources=n_s,
            n_targets=n_t,
            n_channels=3,
            hidden_width=5,
            latent_dim=24,
            ortho_dim=14,
            module_dim=24,
            coop_rank=coop_rank,
            use_cooperativity=use_coop,
            base_rates=(0.06, 0.25, 1.0),
            rate_min=0.015,
            rate_max=3.0,
        )
        fitted = fit_dynamical_model(
            spec=spec,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=dt_h,
            obs_stride=obs_st,
            seeds=seeds,
            reg_grid=reg_grid,
            gate_shrinkage_grid=(1.0,),
            hysteresis_shrinkage_grid=(1.0,),
            total_updates=500,
            lr_max=0.007,
        )
        pred = fitted.predict_fn(te_u, te_y[:, 0, :])
        n_params = fitted.n_params
        if partial_obs:
            a_tot = counterfactual_source_attribution(fitted.predict_fn, te_u[:, :16], te_y[:16, 0, :])
            attr_full_mag = np.sqrt(np.mean(a_tot[:, :, 1:, :] ** 2, axis=(1, 2)))
            attr_obs_mag = attr_full_mag

    ep_mse = np.mean((pred[:, 1:, :] - te_y[:, 1:, :]) ** 2, axis=(1, 2))
    wash_m = data.washout_flags[test_idx]
    fold_mse = float(np.mean(ep_mse))
    fold_r2 = 1.0 - fold_mse / (float(np.mean(te_y[:, 1:, :] ** 2)) + 1e-8)
    out_d = {
        "n_params": n_params,
        "test_indices": test_idx.tolist(),
        "per_ep_mse": ep_mse.tolist(),
        "fold_mse": fold_mse,
        "fold_r2": fold_r2,
        "washout_mse": float(np.mean(ep_mse[wash_m])) if np.any(wash_m) else fold_mse,
    }
    if attr_full_mag is not None:
        out_d["attr_obs_mag"] = attr_obs_mag.tolist()
        out_d["attr_full_mag"] = attr_full_mag.tolist()
    return mode, sea_i, mk, out_d


# =============================================================================
# Task 2: Equal-Budget 24-World Multi-TF with Matched +Gamma & Depthwise SSM
# =============================================================================
MTF_MODELS = (
    ("memgrn", DynamicalModelFamily.MEMGRN, True),
    ("local_const_coop", DynamicalModelFamily.LOCAL_CONST, True),
    ("ltc_local_coop", DynamicalModelFamily.LTC_LOCAL, True),
    ("depthwise_ssm_coop", DynamicalModelFamily.DEPTHWISE_SSM, True),
    ("memgrn_additive", DynamicalModelFamily.MEMGRN_ADDITIVE, False),
    ("local_const", DynamicalModelFamily.LOCAL_CONST, False),
    ("ltc_local", DynamicalModelFamily.LTC_LOCAL, False),
    ("depthwise_ssm", DynamicalModelFamily.DEPTHWISE_SSM, False),
    ("coupled_const", DynamicalModelFamily.COUPLED_CONST, False),
    ("coupled_adaptive", DynamicalModelFamily.COUPLED_ADAPTIVE, False),
    ("ltc_coupled", DynamicalModelFamily.LTC_COUPLED, False),
    ("selective_ssm", DynamicalModelFamily.SELECTIVE_SSM, False),
    ("cfc", DynamicalModelFamily.CFC, False),
    ("neural_cde", DynamicalModelFamily.NEURAL_CDE, False),
    ("perturb_ode", DynamicalModelFamily.PERTURB_ODE, False),
)


def _run_mtf_world(job: tuple[int, str]) -> dict:
    seed, regime = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    world = generate_multitf_world(seed=seed, regime=regime, n_sources=8, n_targets=24)
    tr_m = np.ones_like(world.train_y)
    tr_m[:, 0, :] = 0.0
    va_m = np.ones_like(world.val_y)
    va_m[:, 0, :] = 0.0

    world_res = {}
    for mk, fam, use_coop in MTF_MODELS:
        spec = ModelSpec(
            family=fam,
            n_sources=world.n_sources,
            n_targets=world.n_targets,
            n_channels=3,
            hidden_width=6,
            module_dim=14,
            use_cooperativity=use_coop,
        )
        fitted = fit_dynamical_model(
            spec=spec,
            train_stage_u=world.train_stage_u,
            train_y=world.train_y,
            train_mask=tr_m,
            val_stage_u=world.val_stage_u,
            val_y=world.val_y,
            val_mask=va_m,
            dt=world.dt,
            obs_stride=world.obs_stride,
            seeds=(101, 202),
            reg_grid=(1e-5, 1e-4, 1e-3),
            gate_shrinkage_grid=(1.0,),
            hysteresis_shrinkage_grid=(1.0,),
            total_updates=950,
            lr_max=0.008,
        )
        pred_te = fitted.predict_fn(world.test_stage_u, None)
        test_mse = float(np.mean((pred_te[:, 1:, :] - world.test_y[:, 1:, :]) ** 2))
        eff = counterfactual_source_attribution(fitted.predict_fn, world.test_stage_u, None)
        attr_rmse = float(np.sqrt(np.mean((eff[:, :, 1:, :] - world.test_true_effects[:, :, 1:, :]) ** 2)))
        non_c = ~world.parent_mask
        spur = float(
            np.sqrt(
                np.mean(
                    (eff[:, :, 1:, :] ** 2)[
                        non_c[:, None, None, :].repeat(eff.shape[1], axis=1).repeat(eff.shape[2] - 1, axis=2)
                    ]
                )
            )
        )
        edge_s = np.sqrt(np.mean(eff[:, :, 1:, :] ** 2, axis=(1, 2)))
        auroc, ap = _auroc_and_ap(edge_s.ravel(), world.parent_mask.ravel())
        world_res[mk] = {
            "n_params": fitted.n_params,
            "test_mse": test_mse,
            "attribution_rmse": attr_rmse,
            "spurious_crosstalk_rmse": spur,
            "edge_auroc": auroc,
            "edge_ap": ap,
        }
    return {"seed": seed, "regime": regime, "models": world_res}


# =============================================================================
# Task 3: Ground-Truth 6-of-8 Partial Observability (True Joint MEMGRN_UNPROJ!)
# =============================================================================
def _run_po6of8_world(job: tuple[int, str]) -> dict:
    seed, regime = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    world = generate_multitf_world(seed=seed, regime=regime, n_sources=8, n_targets=24)
    obs_tf_idx = [0, 1, 3, 4, 6, 7]
    tr_u6 = world.train_stage_u[:, :, obs_tf_idx, :]
    va_u6 = world.val_stage_u[:, :, obs_tf_idx, :]
    te_u6 = world.test_stage_u[:, :, obs_tf_idx, :]
    tr_y, va_y, te_y = world.train_y, world.val_y, world.test_y
    tr_m, va_m = np.ones_like(tr_y), np.ones_like(va_y)
    tr_m[:, 0, :] = 0.0
    va_m[:, 0, :] = 0.0
    true_eff_6 = world.test_true_effects[obs_tf_idx]
    true_edge_6 = world.parent_mask[obs_tf_idx, :]

    specs = {
        "local_const": ModelSpec(family=DynamicalModelFamily.LOCAL_CONST, n_sources=6, n_targets=24, n_channels=3, use_cooperativity=True),
        "depthwise_ssm": ModelSpec(family=DynamicalModelFamily.DEPTHWISE_SSM, n_sources=6, n_targets=24, n_channels=3, use_cooperativity=True),
        "memgrn": ModelSpec(family=DynamicalModelFamily.MEMGRN, n_sources=6, n_targets=24, n_channels=3, hidden_width=5, use_cooperativity=True),
        "memgrn_unproj": ModelSpec(family=DynamicalModelFamily.MEMGRN_UNPROJ, n_sources=6, n_targets=24, n_channels=3, hidden_width=5, ortho_dim=14, use_cooperativity=True),
        "memgrn_ortho_joint": ModelSpec(family=DynamicalModelFamily.MEMGRN_ORTHO, n_sources=6, n_targets=24, n_channels=3, hidden_width=5, ortho_dim=14, use_cooperativity=False),
        "perturb_ode": ModelSpec(family=DynamicalModelFamily.PERTURB_ODE, n_sources=6, n_targets=24, module_dim=14),
        "latent_ode": ModelSpec(family=DynamicalModelFamily.LATENT_ODE, n_sources=6, n_targets=24, latent_dim=14),
    }
    fits = {}
    for mk, sp in specs.items():
        fits[mk] = fit_dynamical_model(
            spec=sp,
            train_stage_u=tr_u6,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u6,
            val_y=va_y,
            val_mask=va_m,
            dt=world.dt,
            obs_stride=world.obs_stride,
            seeds=(31, 47),
            reg_grid=(1e-4, 1e-3, 1e-2),
            gate_shrinkage_grid=(1.0,),
            hysteresis_shrinkage_grid=(1.0,),
            total_updates=950,
            lr_max=0.009,
        )
    # Uniform single-spec two-stage MemGRN-Ortho (LATENT_ODE Stage 2, global scalar beta)
    fits["memgrn_ortho_2stage"] = fit_two_stage_memgrn_ortho(
        base_memgrn=fits["memgrn"],
        train_stage_u=tr_u6,
        train_y=tr_y,
        train_mask=tr_m,
        val_stage_u=va_u6,
        val_y=va_y,
        val_mask=va_m,
        dt=world.dt,
        obs_stride=world.obs_stride,
        latent_dim=14,
        seeds=(31, 47),
        reg_grid=(1e-4, 1e-3, 1e-2),
        total_updates=950,
        lr_max=0.009,
        residual_families=(DynamicalModelFamily.LATENT_ODE,),
        per_target_blend=False,
    )

    pure_eff = counterfactual_source_attribution(fits["memgrn"].predict_fn, te_u6, None)
    pure_norm = float(np.linalg.norm(pure_eff)) + 1e-12
    out = {}
    for mk, ft in fits.items():
        y_hat = ft.predict_fn(te_u6, None)
        test_mse = float(np.mean((y_hat[:, 1:, :] - te_y[:, 1:, :]) ** 2))
        # Always evaluate FULL-MODEL attribution (predict_fn) honestly!
        eff_full = counterfactual_source_attribution(ft.predict_fn, te_u6, None)
        full_rmse = float(np.sqrt(np.mean((eff_full[:, :, 1:, :] - true_eff_6[:, :, 1:, :]) ** 2)))
        full_edges = np.sqrt(np.mean(eff_full[:, :, 1:, :] ** 2, axis=(1, 2)))
        full_auroc, full_ap = _auroc_and_ap(full_edges.ravel(), true_edge_6.ravel())
        full_drift = float(np.linalg.norm(eff_full - pure_eff) / pure_norm)

        # Also evaluate observed-only branch when present
        obs_fn = ft.predict_observed_only_fn if ft.predict_observed_only_fn is not None else ft.predict_fn
        eff_obs = counterfactual_source_attribution(obs_fn, te_u6, None)
        obs_rmse = float(np.sqrt(np.mean((eff_obs[:, :, 1:, :] - true_eff_6[:, :, 1:, :]) ** 2)))
        obs_edges = np.sqrt(np.mean(eff_obs[:, :, 1:, :] ** 2, axis=(1, 2)))
        obs_auroc, obs_ap = _auroc_and_ap(obs_edges.ravel(), true_edge_6.ravel())
        obs_drift = float(np.linalg.norm(eff_obs - pure_eff) / pure_norm)

        out[mk] = {
            "n_params": ft.n_params,
            "test_mse": test_mse,
            "full_model_gt_attr_rmse": full_rmse,
            "full_model_gt_edge_ap": full_ap,
            "full_model_gt_edge_auroc": full_auroc,
            "full_model_drift_vs_memgrn": full_drift,
            "obs_branch_gt_attr_rmse": obs_rmse,
            "obs_branch_gt_edge_ap": obs_ap,
            "obs_branch_gt_edge_auroc": obs_auroc,
            "obs_branch_drift_vs_memgrn": obs_drift,
        }
    return {"seed": seed, "regime": regime, "models": out}


def _run_ltc_and_dssm_separation() -> dict:
    """Part B: Parameter-matched MemGRN vs Source-Local LTC vs Depthwise SSM vs Local Const."""
    rng = np.random.default_rng(20261121)
    n_src, n_tgt, dt, n_steps, obs_stride = 4, 12, 0.05, 120, 4
    d_true = np.zeros((n_src, n_tgt), dtype=np.float64)
    b_true = np.zeros((n_src, n_tgt), dtype=np.float64)
    for j in range(n_tgt):
        p1, p2 = j % n_src, (j + 1) % n_src
        b_true[p1, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 1.5)
        b_true[p2, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.1)
    delta_true = rng.uniform(0.8, 1.5, size=n_tgt)
    lam_load = np.array([3.2, 2.5, 4.0, 2.8])
    lam_clear_base = np.array([0.45, 0.65, 0.55, 0.75])

    def _sim(stage_u: np.ndarray) -> np.ndarray:
        batch = stage_u.shape[1]
        z, h, y = np.zeros((batch, n_src)), np.zeros((batch, n_src)), np.zeros((batch, n_tgt))
        ys = [y.copy()]
        for st in range(n_steps):
            a0, am, a1 = np.tanh(stage_u[st, :, :, 0]), np.tanh(stage_u[st, :, :, 1]), np.tanh(stage_u[st, :, :, 2])
            def _rhs(zs, hs, ys_, av):
                lu = lam_load[None] / (1.0 + 0.6 * np.abs(hs))
                ld = lam_clear_base[None] / (1.0 + 1.2 * np.abs(hs))
                return lu * (av - zs), ld * (zs - hs), av @ d_true + hs @ b_true - delta_true[None] * ys_
            k1 = _rhs(z, h, y, a0)
            k2 = _rhs(z + 0.5 * dt * k1[0], h + 0.5 * dt * k1[1], y + 0.5 * dt * k1[2], am)
            k3 = _rhs(z + 0.5 * dt * k2[0], h + 0.5 * dt * k2[1], y + 0.5 * dt * k2[2], am)
            k4 = _rhs(z + dt * k3[0], h + dt * k3[1], y + dt * k3[2], a1)
            z += (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            h += (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            y += (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            if (st + 1) % obs_stride == 0:
                ys.append(y.copy())
        return np.stack(ys, axis=1)

    t3 = np.stack([np.arange(n_steps) * dt, (np.arange(n_steps) + 0.5) * dt, (np.arange(n_steps) + 1.0) * dt], axis=-1)
    def _mk(n_b, dr, ar, sp):
        r = np.random.default_rng(sp)
        u = np.zeros((n_steps, n_b, n_src, 3))
        for b in range(n_b):
            for s in range(n_src):
                if r.random() < 0.75:
                    dur, amp, ton = r.uniform(*dr), r.uniform(*ar), r.uniform(0.05, 0.35)
                    u[:, b, s, :] = amp * ((t3 >= ton) & (t3 <= ton + dur))
        return u

    tr_u, va_u = _mk(24, (0.4, 2.5), (0.3, 1.2), 11), _mk(8, (0.4, 2.5), (0.3, 1.2), 22)
    te_peak_u = _mk(16, (0.25, 0.55), (0.6, 1.5), 33)
    cl_lo, cl_hi = np.zeros((n_steps, n_src, n_src, 3)), np.zeros((n_steps, n_src, n_src, 3))
    for s in range(n_src):
        cl_lo[:, s, s, :], cl_hi[:, s, s, :] = 0.25, 1.20
    tr_y = _sim(tr_u) + rng.normal(0.0, 0.008, size=(24, 31, n_tgt))
    va_y = _sim(va_u) + rng.normal(0.0, 0.008, size=(8, 31, n_tgt))
    tr_y[:, 0, :], va_y[:, 0, :] = 0.0, 0.0
    te_peak_y, te_lo_y, te_hi_y = _sim(te_peak_u), _sim(cl_lo), _sim(cl_hi)
    tr_m, va_m = np.ones_like(tr_y), np.ones_like(va_y)
    tr_m[:, 0, :], va_m[:, 0, :] = 0.0, 0.0

    specs = {
        "memgrn_matched": ModelSpec(family=DynamicalModelFamily.MEMGRN_ADDITIVE, n_sources=4, n_targets=12, n_channels=2, hidden_width=4),
        "ltc_local_matched": ModelSpec(family=DynamicalModelFamily.LTC_LOCAL, n_sources=4, n_targets=12, n_channels=3),
        "depthwise_ssm_matched": ModelSpec(family=DynamicalModelFamily.DEPTHWISE_SSM, n_sources=4, n_targets=12, n_channels=3),
        "local_const_matched": ModelSpec(family=DynamicalModelFamily.LOCAL_CONST, n_sources=4, n_targets=12, n_channels=3),
    }
    res = {}
    for nm, sp in specs.items():
        fit = fit_dynamical_model(
            spec=sp, train_stage_u=tr_u, train_y=tr_y, train_mask=tr_m,
            val_stage_u=va_u, val_y=va_y, val_mask=va_m,
            dt=dt, obs_stride=obs_stride, seeds=(17, 29, 43), reg_grid=(1e-5, 1e-4, 1e-3),
            gate_shrinkage_grid=(1.0,), hysteresis_shrinkage_grid=(1.0,), total_updates=1400, lr_max=0.01,
        )
        p_pk, p_lo, p_hi = fit.predict_fn(te_peak_u, None), fit.predict_fn(cl_lo, None), fit.predict_fn(cl_hi, None)
        w_lo, w_hi = p_lo[:, -1, :] / np.tanh(0.25), p_hi[:, -1, :] / np.tanh(1.20)
        wt_lo, wt_hi = te_lo_y[:, -1, :] / np.tanh(0.25), te_hi_y[:, -1, :] / np.tanh(1.20)
        res[nm] = {
            "n_params": fit.n_params,
            "delayed_peak_post_mse": float(np.mean((p_pk[:, 4:, :] - te_peak_y[:, 4:, :]) ** 2)),
            "steady_state_amp_invariance_error": float(np.linalg.norm(w_hi - w_lo) / (np.linalg.norm(wt_hi) + 1e-12)),
            "steady_state_weight_rel_error": float(
                0.5 * (np.linalg.norm(w_lo - wt_lo) / np.linalg.norm(wt_lo) + np.linalg.norm(w_hi - wt_hi) / np.linalg.norm(wt_hi))
            ),
        }
    return res


# =============================================================================
# Task 4: Clean Equal-Budget Human EMT Withdrawal (GSE147405)
# =============================================================================
EMT_MODELS = (
    ("memgrn_ortho", "memgrn_ortho_2stage", 6, True),
    ("memgrn", "memgrn", 6, True),
    ("memgrn_additive", "memgrn_additive", None, False),
    ("local_const_coop", "local_const", 6, True),
    ("ltc_local_coop", "ltc_local", 6, True),
    ("depthwise_ssm_coop", "depthwise_ssm", 6, True),
    ("coupled_adaptive", "coupled_adaptive", None, False),
    ("ltc_coupled", "ltc_coupled", None, False),
    ("selective_ssm", "selective_ssm", None, False),
    ("cfc", "cfc", None, False),
    ("neural_cde", "neural_cde", None, False),
    ("perturb_ode", "perturb_ode", None, False),
    ("latent_ode", "latent_ode", None, False),
)


def _run_emt_job(job: tuple) -> tuple[str, str, dict]:
    cl_i, mk, fam_str, coop_rank, use_coop = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    data = fetch_and_prepare_emt_withdrawal(RAW_ROOT / "emt_gse147405")
    held_cl = data.cell_lines[cl_i]
    src_cls = [c for c in data.cell_lines if c != held_cl]
    test_rows = [i for i, (cl, _l) in enumerate(data.conditions) if cl == held_cl]
    src_rows = [i for i, (cl, _l) in enumerate(data.conditions) if cl != held_cl]
    val_pairs = {(src_cls[k], data.ligands[(k + cl_i) % len(data.ligands)]) for k in range(3)}
    val_rows = [i for i, pair in enumerate(data.conditions) if pair in val_pairs]
    train_wdr_rows = [i for i in src_rows if i not in val_rows]

    y_all_raw = data.delta_trajectories
    n_tf_all = len(data.tf_indices)
    tgt_idx = np.arange(n_tf_all, len(data.genes), dtype=np.int64)

    tr_mask_ct = np.zeros((12, 8), dtype=bool)
    tr_mask_ct[:, 1:5] = True
    tr_mask_ct[train_wdr_rows, 5:] = True
    va_mask_ct = np.zeros((12, 8), dtype=bool)
    va_mask_ct[val_rows, 5:] = True

    # STRICT ZERO-LEAKAGE TF SELECTION: exclude held-out cell line induction frames!
    src_only_mask = np.zeros((12, 8), dtype=bool)
    src_only_mask[src_rows, 1:] = True
    tf_rms = np.sqrt(np.mean(y_all_raw[:, :, data.tf_indices][src_only_mask] ** 2, axis=0))
    sel_tf = np.sort(data.tf_indices[np.argsort(-tf_rms)[:12]])
    tf_sc = np.maximum(0.10, np.sqrt(np.mean(y_all_raw[:, :, sel_tf][tr_mask_ct] ** 2, axis=0)))
    tg_sc = np.maximum(0.10, np.sqrt(np.mean(y_all_raw[:, :, tgt_idx][tr_mask_ct] ** 2, axis=0)))

    src_tgt_rms = np.sqrt(np.mean(y_all_raw[:, :, tgt_idx][tr_mask_ct] ** 2, axis=0))
    top30_mask = np.zeros(len(tgt_idx), dtype=bool)
    top30_mask[np.argsort(-src_tgt_rms)[:30]] = True

    dt_days, n_steps = 1.0 / 6.0, 60
    obs_steps = (0, 2, 6, 18, 42, 44, 48, 60)
    obs_t = np.asarray(obs_steps, dtype=np.float64)
    s0 = np.arange(n_steps, dtype=np.float64)
    tf_obs = y_all_raw[:, :, sel_tf] / tf_sc[None, None, :]
    u_all = np.zeros((n_steps, 12, len(sel_tf), 3), dtype=np.float64)
    for b_i in range(12):
        for s_i in range(len(sel_tf)):
            v = tf_obs[b_i, :, s_i]
            u_all[:, b_i, s_i, 0] = np.interp(s0, obs_t, v)
            u_all[:, b_i, s_i, 1] = np.interp(s0 + 0.5, obs_t, v)
            u_all[:, b_i, s_i, 2] = np.interp(s0 + 1.0, obs_t, v)

    y8_all = y_all_raw[:, :, tgt_idx] / tg_sc[None, None, :]
    y31 = np.zeros((12, 31, len(tgt_idx)))
    tr_m31, va_m31 = np.zeros_like(y31), np.zeros_like(y31)
    for p, st in enumerate(obs_steps):
        k31 = st // 2
        y31[:, k31, :] = y8_all[:, p, :]
        tr_m31[:, k31, :] = tr_mask_ct[:, p : p + 1].astype(np.float64)
        va_m31[:, k31, :] = va_mask_ct[:, p : p + 1].astype(np.float64)
    for p in (5, 6, 7):
        tr_m31[:, obs_steps[p] // 2, :] *= 2.0

    # Strictly EQUAL 8-trial validation budget for EVERY model (2 seeds x 4 regs)
    seeds = (101, 202)
    reg_grid = (1e-4, 1e-3, 1e-2, 5e-2)
    if fam_str == "memgrn_ortho_2stage":
        base_spec = ModelSpec(
            family=DynamicalModelFamily.MEMGRN, n_sources=12, n_targets=len(tgt_idx),
            n_channels=3, hidden_width=5, coop_rank=coop_rank, use_cooperativity=True,
            base_rates=(0.6, 2.0, 5.5), rate_min=0.05, rate_max=8.0,
        )
        base_fit = fit_dynamical_model(
            spec=base_spec, train_stage_u=u_all, train_y=y31, train_mask=tr_m31,
            val_stage_u=u_all, val_y=y31, val_mask=va_m31, dt=dt_days, obs_stride=2,
            seeds=seeds, reg_grid=reg_grid, gate_shrinkage_grid=(1.0,), hysteresis_shrinkage_grid=(1.0,),
            total_updates=900, lr_max=0.008,
        )
        fitted = fit_two_stage_memgrn_ortho(
            base_memgrn=base_fit, train_stage_u=u_all, train_y=y31, train_mask=tr_m31,
            val_stage_u=u_all, val_y=y31, val_mask=va_m31, dt=dt_days, obs_stride=2,
            latent_dim=14, seeds=seeds, reg_grid=reg_grid, total_updates=800, lr_max=0.007,
            residual_families=(DynamicalModelFamily.LATENT_ODE,), per_target_blend=False,
        )
    else:
        spec = ModelSpec(
            family=DynamicalModelFamily(fam_str), n_sources=12, n_targets=len(tgt_idx),
            n_channels=3, hidden_width=5, latent_dim=18, module_dim=18,
            coop_rank=coop_rank, use_cooperativity=use_coop,
            base_rates=(0.6, 2.0, 5.5), rate_min=0.05, rate_max=8.0,
        )
        fitted = fit_dynamical_model(
            spec=spec, train_stage_u=u_all, train_y=y31, train_mask=tr_m31,
            val_stage_u=u_all, val_y=y31, val_mask=va_m31, dt=dt_days, obs_stride=2,
            seeds=seeds, reg_grid=reg_grid, gate_shrinkage_grid=(1.0,), hysteresis_shrinkage_grid=(1.0,),
            total_updates=900, lr_max=0.008,
        )

    obs_k31 = [s // 2 for s in obs_steps]
    pred_8 = fitted.predict_fn(u_all, None)[test_rows][:, obs_k31, :]
    te_y8 = y8_all[test_rows]
    wdr_p, wdr_t = pred_8[:, 5:, :], te_y8[:, 5:, :]
    wdr_mse = float(np.mean((wdr_p - wdr_t) ** 2))
    wdr_r2 = 1.0 - wdr_mse / (float(np.mean(wdr_t**2)) + 1e-8)
    wdr_t30 = float(np.mean((wdr_p[:, :, top30_mask] - wdr_t[:, :, top30_mask]) ** 2))

    # Compute CollecTRI edges directly on this model's Stage-1 observed branch AND full model!
    attr_obs_fn = fitted.predict_observed_only_fn if fitted.predict_observed_only_fn is not None else fitted.predict_fn
    eff_obs = counterfactual_source_attribution(attr_obs_fn, u_all, None)
    edge_obs = np.mean(np.abs(eff_obs[:, :, obs_k31, :][:, :, 1:, :]), axis=(1, 2))
    eff_full = counterfactual_source_attribution(fitted.predict_fn, u_all, None)
    edge_full = np.mean(np.abs(eff_full[:, :, obs_k31, :][:, :, 1:, :]), axis=(1, 2))

    return held_cl, mk, {
        "n_params": fitted.n_params,
        "withdrawal_mse": wdr_mse,
        "withdrawal_r2": wdr_r2,
        "top30_withdrawal_mse": wdr_t30,
        "per_ligand_wdr_mse": [float(np.mean((wdr_p[k] - wdr_t[k]) ** 2)) for k in range(3)],
        "per_ligand_top30_mse": [float(np.mean((wdr_p[k, :, top30_mask] - wdr_t[k, :, top30_mask]) ** 2)) for k in range(3)],
        "selected_tfs": [data.genes[int(i)] for i in sel_tf],
        "edge_obs": edge_obs.tolist(),
        "edge_full": edge_full.tolist(),
    }


# =============================================================================
# Task 5: Equal-Budget Binance 16-Asset & 14-Altcoin PO
# =============================================================================
FIN_MODELS = (
    ("memgrn", "memgrn", 8, True),
    ("memgrn_additive", "memgrn_additive", None, False),
    ("local_const_coop", "local_const", 8, True),
    ("ltc_local_coop", "ltc_local", 8, True),
    ("depthwise_ssm_coop", "depthwise_ssm", 8, True),
    ("coupled_adaptive", "coupled_adaptive", None, False),
    ("ltc_coupled", "ltc_coupled", None, False),
    ("selective_ssm", "selective_ssm", None, False),
    ("cfc", "cfc", None, False),
    ("neural_cde", "neural_cde", None, False),
    ("perturb_ode", "perturb_ode", None, False),
    ("latent_ode", "latent_ode", None, False),
    ("dlinear", "dlinear", None, False),
    ("gru", "gru", None, False),
    ("transformer", "transformer", None, False),
)

FIN_PO_MODELS = (
    ("memgrn", "memgrn", 8, True),
    ("memgrn_ortho", "memgrn_ortho_2stage", 8, True),
    ("memgrn_unproj", "memgrn_unproj", 8, True),
    ("depthwise_ssm_coop", "depthwise_ssm", 8, True),
    ("local_const_coop", "local_const", 8, True),
    ("latent_ode", "latent_ode", None, False),
    ("dlinear", "dlinear", None, False),
    ("gru", "gru", None, False),
    ("transformer", "transformer", None, False),
)


def _run_fin_job(job: tuple) -> tuple[bool, str, dict]:
    hide_btc_eth, mk, fam_str, coop_rank, use_coop = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    data = load_binance_market_contagion(RAW_ROOT / "financial_defi", episode_bars=48)
    tr_idx = np.where(data.month_ids <= 6)[0]
    va_idx = np.where(data.month_ids == 7)[0]
    te_idx = np.where(data.month_ids == 8)[0]
    cols = list(range(2, 16)) if hide_btc_eth else list(range(16))
    n_s = len(cols)
    y_sc = np.maximum(0.10, np.sqrt(np.mean(data.targets[tr_idx][:, :, cols] ** 2, axis=(0, 1), keepdims=True)))

    def _prep(idxs: np.ndarray):
        u_obs = data.drivers[idxs][:, :, cols]
        u_stage = _build_multi_stage_grid(u_obs, substeps_per_knot=2)
        y_n = data.targets[idxs][:, :, cols] / y_sc
        m_n = np.ones_like(y_n)
        m_n[:, 0, :] = 0.0
        return u_obs, u_stage, y_n, m_n

    tr_uo, tr_u, tr_y, tr_m = _prep(tr_idx)
    va_uo, va_u, va_y, va_m = _prep(va_idx)
    te_uo, te_u, te_y, _ = _prep(te_idx)
    seeds, reg_grid = (101, 202), (1e-4, 1e-3, 1e-2)

    if fam_str in ("dlinear", "gru", "transformer"):
        fit_d = fit_discrete_open_loop(
            family=DiscreteModelFamily(fam_str), train_u=tr_uo, train_y=tr_y, train_mask=tr_m,
            val_u=va_uo, val_y=va_y, val_mask=va_m, hidden_dim=32, seeds=seeds, reg_grid=reg_grid,
            total_updates=500, lr_max=0.005,
        )
        pred = fit_d.predict_fn(te_uo)
        n_params = fit_d.n_params
    elif fam_str == "memgrn_ortho_2stage":
        base_spec = ModelSpec(
            family=DynamicalModelFamily.MEMGRN, n_sources=n_s, n_targets=n_s,
            n_channels=3, hidden_width=5, coop_rank=coop_rank, use_cooperativity=True,
            base_rates=(0.2, 0.8, 2.5), rate_min=0.05, rate_max=5.0,
        )
        base_fit = fit_dynamical_model(
            spec=base_spec, train_stage_u=tr_u, train_y=tr_y, train_mask=tr_m,
            val_stage_u=va_u, val_y=va_y, val_mask=va_m, dt=0.25, obs_stride=2,
            seeds=seeds, reg_grid=reg_grid, gate_shrinkage_grid=(1.0,), hysteresis_shrinkage_grid=(1.0,),
            total_updates=550, lr_max=0.006,
        )
        fitted = fit_two_stage_memgrn_ortho(
            base_memgrn=base_fit, train_stage_u=tr_u, train_y=tr_y, train_mask=tr_m,
            val_stage_u=va_u, val_y=va_y, val_mask=va_m, dt=0.25, obs_stride=2,
            latent_dim=16, seeds=seeds, reg_grid=reg_grid, total_updates=450, lr_max=0.006,
            residual_families=(DynamicalModelFamily.LATENT_ODE,), per_target_blend=False,
        )
        pred = fitted.predict_fn(te_u, te_y[:, 0, :])
        n_params = fitted.n_params
    else:
        spec = ModelSpec(
            family=DynamicalModelFamily(fam_str), n_sources=n_s, n_targets=n_s,
            n_channels=3, hidden_width=5, latent_dim=32, ortho_dim=16, module_dim=32,
            coop_rank=coop_rank, use_cooperativity=use_coop,
            base_rates=(0.2, 0.8, 2.5), rate_min=0.05, rate_max=5.0,
        )
        fitted = fit_dynamical_model(
            spec=spec, train_stage_u=tr_u, train_y=tr_y, train_mask=tr_m,
            val_stage_u=va_u, val_y=va_y, val_mask=va_m, dt=0.25, obs_stride=2,
            seeds=seeds, reg_grid=reg_grid, gate_shrinkage_grid=(1.0,), hysteresis_shrinkage_grid=(1.0,),
            total_updates=550, lr_max=0.006,
        )
        pred = fitted.predict_fn(te_u, te_y[:, 0, :])
        n_params = fitted.n_params

    ep_mse = np.mean((pred[:, 1:, :] - te_y[:, 1:, :]) ** 2, axis=(1, 2))
    idio_m = data.idiosyncratic_flags[te_idx]
    all_mse = float(np.mean(ep_mse))
    return hide_btc_eth, mk, {
        "n_params": n_params,
        "aug_test_mse": all_mse,
        "aug_test_mse_se": float(np.std(ep_mse, ddof=1) / np.sqrt(len(ep_mse))),
        "aug_test_r2": 1.0 - all_mse / (float(np.mean(te_y[:, 1:, :] ** 2)) + 1e-8),
        "idio_shock_mse": float(np.mean(ep_mse[idio_m])),
        "idio_shock_mse_se": float(np.std(ep_mse[idio_m], ddof=1) / np.sqrt(np.sum(idio_m))),
        "per_ep_mse": ep_mse.tolist(),
    }


# =============================================================================
# Task 6: Equal-Budget Yeast Msn2 Joint Models + Depthwise SSM (CV Transparent)
# =============================================================================
YEAST_MODELS = (
    "memgrn",
    "memgrn_sym",
    "local_const",
    "ltc_local",
    "depthwise_ssm",
    "coupled_adaptive",
    "ltc_coupled",
    "selective_ssm",
    "cfc",
    "neural_cde",
    "perturb_ode",
    "latent_ode",
)


def _run_yeast_model(fam_str: str) -> tuple[str, dict]:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    stage3 = ROOT / "results" / "stage3"
    rep_design = load_archived_json(stage3, "msn2_repeated_pulses/design.json")
    schedules = rep_design["schedules"]
    last_off = [s["n"] * 5.0 + (s["n"] - 1) * s["gap"] for s in schedules]
    with np.load(stage3 / "neural_rate_msn2_transfer" / "training.npz") as tr_f:
        train_y_obs = np.asarray(tr_f["y"], dtype=np.float64)
        scales = np.asarray(tr_f["scales"], dtype=np.float64)
        tf_single = np.asarray(tr_f["tf"], dtype=np.float64)
        obs_min = np.asarray(tr_f["minutes"], dtype=np.float64)
    with np.load(stage3 / "msn2_repeated_pulses" / "predictions.npz") as rep_f:
        rep_drivers = np.asarray(rep_f["drivers"], dtype=np.float64)
    rep_truth = (
        np.stack(
            [
                np.stack(
                    [
                        load_msn2_population_trace(
                            RAW_ROOT / "msn2" / f"{g}_{s['id']}_size.mat", "YFP_molecules", obs_min
                        )[0]
                        for g in MSN2_PROMOTERS
                    ],
                    axis=-1,
                )
                for s in schedules
            ],
            axis=0,
        )
        / scales[None, None, :]
    )
    substeps = 4
    dt_h = 2.5 / (60.0 * substeps)
    obs_stride = 2 * substeps
    single_stage = build_rk4_stage_grid(tf_single[:16], substeps)
    rep_stage = build_rk4_stage_grid(rep_drivers, substeps)
    single_y = np.concatenate([np.zeros((16, 1, 7)), train_y_obs], axis=1)
    val_idx = [1, 6, 10, 15]
    tr_idx = [i for i in range(16) if i not in val_idx]
    tr_y, va_y = single_y[tr_idx], single_y[val_idx]
    tr_m, va_m = np.ones_like(tr_y), np.ones_like(va_y)
    tr_m[:, 0, :] = 0.0
    va_m[:, 0, :] = 0.0

    spec = ModelSpec(
        family=DynamicalModelFamily(fam_str), n_sources=1, n_targets=7,
        n_channels=3, hidden_width=4, latent_dim=16, module_dim=12,
        rate_min=0.05, rate_max=14.0,
    )
    # Strictly EQUAL 9-trial budget for EVERY model (3 seeds x 3 regs, no extra shrinkage grids!)
    fit = fit_dynamical_model(
        spec=spec,
        train_stage_u=single_stage[:, tr_idx],
        train_y=tr_y,
        train_mask=tr_m,
        val_stage_u=single_stage[:, val_idx],
        val_y=va_y,
        val_mask=va_m,
        dt=dt_h,
        obs_stride=obs_stride,
        seeds=(101, 202, 303),
        reg_grid=(1e-5, 1e-4, 1e-3),
        gate_shrinkage_grid=(1.0,),
        hysteresis_shrinkage_grid=(1.0,),
        total_updates=1400,
        lr_max=0.01,
    )
    pred = fit.predict_fn(rep_stage, None)[:, 1:, :]
    per_train = [
        float(np.mean((pred[i, obs_min > stop] - rep_truth[i, obs_min > stop]) ** 2))
        for i, stop in enumerate(last_off)
    ]
    return fam_str, {
        "n_params": fit.n_params,
        "val_loss": float(fit.best_val_loss),
        "ten_train_post_mse": float(np.mean(per_train)),
        "ten_train_post_mse_se": float(np.std(per_train, ddof=1) / np.sqrt(len(per_train))),
        "per_train_mse": per_train,
    }


# =============================================================================
# Task 7: Complete S=8..512 Scaling Benchmark (Dense + Rank-24 + Coupled LTC)
# =============================================================================
def _run_complete_scaling() -> list[dict]:
    s_values = (8, 16, 32, 64, 128, 256, 512)
    batch, n_steps, dt = 4, 20, 0.25
    obs_idx = (0, 5, 10, 15, 20)
    readout_keys = {"direct", "delayed", "delayed_u", "delayed_v", "readout", "decay", "in_gain", "coop_u", "coop_v", "coop_w", "self_mem"}
    rows = []
    for s_tf in s_values:
        n_g = 4 * s_tf
        reg_idx = tuple(range(s_tf))
        basal = np.ones(n_g, dtype=np.float32)
        rng_sc = np.random.default_rng(1000 + s_tf)
        ko_np = np.zeros((batch, n_g), dtype=np.float32)
        for b_i in range(batch):
            ko_np[b_i, b_i % s_tf] = 1.0
        ko = jnp.asarray(ko_np)
        y0 = jnp.asarray(rng_sc.normal(0.0, 0.25, size=(batch, n_g)).astype(np.float32))
        target = jnp.asarray(
            rng_sc.normal(0.0, 0.25, size=(batch, len(obs_idx), n_g)).astype(np.float32)
        )
        row = {"n_tfs": s_tf, "n_genes": n_g}
        configs = [
            ("memgrn_dense", ClosedLoopSpec(family=ClosedLoopModelFamily.MEMGRN, n_genes=n_g, regulator_indices=reg_idx, use_cooperativity=True, coop_rank=8, n_channels=3, hidden_width=4)),
            ("memgrn_rank24", ClosedLoopSpec(family=ClosedLoopModelFamily.MEMGRN, n_genes=n_g, regulator_indices=reg_idx, use_cooperativity=True, coop_rank=8, readout_rank=24, n_channels=3, hidden_width=4)),
            ("depthwise_ssm_rank24", ClosedLoopSpec(family=ClosedLoopModelFamily.DEPTHWISE_SSM, n_genes=n_g, regulator_indices=reg_idx, use_cooperativity=True, coop_rank=8, readout_rank=24, n_channels=3)),
            ("coupled_adaptive", ClosedLoopSpec(family=ClosedLoopModelFamily.COUPLED_ADAPTIVE, n_genes=n_g, regulator_indices=reg_idx, n_channels=3, hidden_width=4)),
            ("coupled_ltc", ClosedLoopSpec(family=ClosedLoopModelFamily.LTC_COUPLED, n_genes=n_g, regulator_indices=reg_idx, latent_dim=min(3 * s_tf, 768))),
        ]
        for label, spec in configs:
            fwd, unpack, flat0 = build_closed_loop_jax_rollout(spec, basal_levels=basal, dt=dt, n_steps=n_steps, obs_step_indices=obs_idx)
            p_tree = unpack(jnp.asarray(flat0))
            mem_p = int(sum(int(v.size) for k, v in p_tree.items() if k not in readout_keys))
            tot_p = int(flat0.size)
            th = jnp.asarray(flat0)

            @jax.jit
            def _lg(tv):
                return jax.value_and_grad(lambda x: jnp.mean((fwd(x, ko, y0) - target) ** 2))(tv)

            _, g = _lg(th)
            g.block_until_ready()
            if not bool(jnp.all(jnp.isfinite(g))) or float(jnp.linalg.norm(g)) <= 0.0:
                raise RuntimeError(f"Degenerate scaling gradient for {label} at S={s_tf}")
            n_reps = 5 if s_tf >= 256 else 10
            ts = time.perf_counter()
            for _ in range(n_reps):
                _, g = _lg(th)
                g.block_until_ready()
            ms = 1000.0 * (time.perf_counter() - ts) / n_reps
            row[f"{label}_mem_params"] = mem_p
            row[f"{label}_total_params"] = tot_p
            row[f"{label}_ms_per_step"] = float(ms)
        row["coupled_ltc_latent_dim_capped"] = min(3 * s_tf, 768)
        rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")

    # 1. Beijing 4-season (raw_u, delta_u, po6)
    bq_jobs = (
        [("raw_u", s, mk, fs, cr, uc) for s in range(4) for mk, fs, cr, uc in BQ_MODELS]
        + [("delta_u", s, mk, fs, cr, uc) for s in range(4) for mk, fs, cr, uc in BQ_MODELS]
        + [("po6", s, mk, fs, cr, uc) for s in range(4) for mk, fs, cr, uc in BQ_PO_MODELS]
    )
    with ProcessPoolExecutor(max_workers=16, mp_context=ctx) as pool:
        bq_raw = list(pool.map(_run_beijing_job, bq_jobs))
    bq_data = load_beijing_air_quality(RAW_ROOT / "beijing_aq", episode_hours=48)
    n_bq = len(bq_data.seasons)

    def _agg_bq(mode: str, model_keys: list[str], ref_key: str) -> dict:
        by_m: dict[str, list[dict]] = {m: [] for m in model_keys}
        for md, _s, mk, r in bq_raw:
            if md == mode and mk in by_m:
                by_m[mk].append(r)
        ep_map, summ = {}, {}
        for mk in model_keys:
            fl = by_m[mk]
            vec = np.zeros(n_bq, dtype=np.float64)
            for fd in fl:
                for i_e, v_e in zip(fd["test_indices"], fd["per_ep_mse"]):
                    vec[i_e] = v_e
            ep_map[mk] = vec
            entry = {
                "n_params": fl[0]["n_params"],
                "loso_mse": float(np.mean(vec)),
                "loso_mse_se": float(np.std(vec, ddof=1) / np.sqrt(n_bq)),
                "loso_r2": float(np.mean([fd["fold_r2"] for fd in fl])),
                "washout_mse": float(np.mean(vec[bq_data.washout_flags])),
                "winter_mse": float(np.mean(vec[bq_data.seasons == 3])),
            }
            if mode == "po6":
                entry["_attr_obs"] = np.mean([np.asarray(fd["attr_obs_mag"]) for fd in fl], axis=0)
                entry["_attr_full"] = np.mean([np.asarray(fd["attr_full_mag"]) for fd in fl], axis=0)
            summ[mk] = entry
        if mode == "po6":
            ref_a = summ["memgrn"]["_attr_obs"]
            rn = float(np.linalg.norm(ref_a)) + 1e-12
            for mk in model_keys:
                ao = summ[mk].pop("_attr_obs")
                af = summ[mk].pop("_attr_full")
                summ[mk]["stage1_drift_vs_memgrn"] = float(np.linalg.norm(ao - ref_a) / rn)
                summ[mk]["full_model_drift_vs_memgrn"] = float(np.linalg.norm(af - ref_a) / rn)
        stats = _two_sided_holm_stats(ep_map[ref_key], {k: v for k, v in ep_map.items() if k != ref_key})
        return {"summary": summ, "holm_two_sided_vs_" + ref_key: stats}

    bq_results = {
        "raw_u_4season": _agg_bq("raw_u", [m[0] for m in BQ_MODELS], "memgrn"),
        "delta_u_4season": _agg_bq("delta_u", [m[0] for m in BQ_MODELS], "memgrn"),
        "po6_4season": _agg_bq("po6", [m[0] for m in BQ_PO_MODELS], "memgrn_ortho"),
    }

    # 2. Multi-TF 24-world fair
    mtf_jobs = [(5001 + i, "additive") for i in range(12)] + [(6001 + i, "cooperative") for i in range(12)]
    with ProcessPoolExecutor(max_workers=16, mp_context=ctx) as pool:
        mtf_raw = list(pool.map(_run_mtf_world, mtf_jobs))
    mtf_summary = {}
    for regime in ("additive", "cooperative"):
        sub = [w for w in mtf_raw if w["regime"] == regime]
        means = {}
        mse_vecs, attr_vecs = {}, {}
        for mk, _, _ in MTF_MODELS:
            tm = np.array([w["models"][mk]["test_mse"] for w in sub])
            ar = np.array([w["models"][mk]["attribution_rmse"] for w in sub])
            sp = np.array([w["models"][mk]["spurious_crosstalk_rmse"] for w in sub])
            ap = np.array([w["models"][mk]["edge_ap"] for w in sub])
            mse_vecs[mk], attr_vecs[mk] = tm, ar
            means[mk] = {
                "n_params": sub[0]["models"][mk]["n_params"],
                "test_mse": float(np.mean(tm)),
                "test_mse_se": float(np.std(tm, ddof=1) / np.sqrt(len(tm))),
                "attribution_rmse": float(np.mean(ar)),
                "attribution_rmse_se": float(np.std(ar, ddof=1) / np.sqrt(len(ar))),
                "spurious_crosstalk_rmse": float(np.mean(sp)),
                "edge_ap": float(np.mean(ap)),
                "edge_ap_se": float(np.std(ap, ddof=1) / np.sqrt(len(ap))),
            }
        mtf_summary[regime] = {
            "means": means,
            "holm_mse_vs_memgrn": _two_sided_holm_stats(mse_vecs["memgrn"], {k: v for k, v in mse_vecs.items() if k != "memgrn"}),
            "holm_attr_vs_memgrn": _two_sided_holm_stats(attr_vecs["memgrn"], {k: v for k, v in attr_vecs.items() if k != "memgrn"}),
        }

    # 3. 6-of-8 PO + LTC/Depthwise-SSM separation
    po_jobs = [(100 + i, "additive") for i in range(6)] + [(200 + i, "cooperative") for i in range(6)]
    with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as pool:
        po_raw = list(pool.map(_run_po6of8_world, po_jobs))
    po_keys = list(po_raw[0]["models"].keys())
    po_sum = {}
    po_mse_vecs, po_full_attr_vecs = {}, {}
    for mk in po_keys:
        tm = np.array([w["models"][mk]["test_mse"] for w in po_raw])
        fa = np.array([w["models"][mk]["full_model_gt_attr_rmse"] for w in po_raw])
        oa = np.array([w["models"][mk]["obs_branch_gt_attr_rmse"] for w in po_raw])
        fap = np.array([w["models"][mk]["full_model_gt_edge_ap"] for w in po_raw])
        oap = np.array([w["models"][mk]["obs_branch_gt_edge_ap"] for w in po_raw])
        po_mse_vecs[mk], po_full_attr_vecs[mk] = tm, fa
        po_sum[mk] = {
            "n_params": po_raw[0]["models"][mk]["n_params"],
            "test_mse": float(np.mean(tm)),
            "test_mse_se": float(np.std(tm, ddof=1) / np.sqrt(len(tm))),
            "full_model_gt_attr_rmse": float(np.mean(fa)),
            "full_model_gt_attr_rmse_se": float(np.std(fa, ddof=1) / np.sqrt(len(fa))),
            "full_model_gt_edge_ap": float(np.mean(fap)),
            "full_model_drift_vs_memgrn": float(np.mean([w["models"][mk]["full_model_drift_vs_memgrn"] for w in po_raw])),
            "obs_branch_gt_attr_rmse": float(np.mean(oa)),
            "obs_branch_gt_edge_ap": float(np.mean(oap)),
            "obs_branch_drift_vs_memgrn": float(np.mean([w["models"][mk]["obs_branch_drift_vs_memgrn"] for w in po_raw])),
        }
    ltc_sep = _run_ltc_and_dssm_separation()

    # 4. Clean EMT (GSE147405)
    emt_jobs = [(cl_i, mk, fs, cr, uc) for cl_i in range(4) for mk, fs, cr, uc in EMT_MODELS]
    with ProcessPoolExecutor(max_workers=16, mp_context=ctx) as pool:
        emt_raw = list(pool.map(_run_emt_job, emt_jobs))
    emt_data = fetch_and_prepare_emt_withdrawal(RAW_ROOT / "emt_gse147405")
    tgt_names = tuple(emt_data.genes[len(emt_data.tf_indices) :])
    emt_by_m: dict[str, list[dict]] = {m[0]: [] for m in EMT_MODELS}
    for _cl, mk, r in emt_raw:
        emt_by_m[mk].append(r)
    emt_sum = {}
    for mk, _, _, _ in EMT_MODELS:
        fl = emt_by_m[mk]
        # Score CollecTRI & ChIP per fold on that fold's exact source-selected TFs!
        col_aps_obs, col_aps_full, col_aurocs_obs = [], [], []
        for fd in fl:
            c_mask, _ch_mask, _ = fetch_emt_collectri_and_chip_masks(
                RAW_ROOT / "emt_gse147405", tuple(fd["selected_tfs"]), tgt_names
            )
            br = float(np.mean(c_mask))
            e_obs = np.asarray(fd["edge_obs"])
            e_full = np.asarray(fd["edge_full"])
            cn_obs = e_obs / np.maximum(1e-8, np.max(e_obs, axis=0, keepdims=True))
            cn_full = e_full / np.maximum(1e-8, np.max(e_full, axis=0, keepdims=True))
            au_o, ap_o = _auroc_and_ap(cn_obs.ravel(), c_mask.ravel())
            _, ap_f = _auroc_and_ap(cn_full.ravel(), c_mask.ravel())
            col_aurocs_obs.append(au_o)
            col_aps_obs.append(ap_o / max(1e-8, br))
            col_aps_full.append(ap_f / max(1e-8, br))
        c12 = np.array([x for fd in fl for x in fd["per_ligand_wdr_mse"]])
        t30_12 = np.array([x for fd in fl for x in fd["per_ligand_top30_mse"]])
        emt_sum[mk] = {
            "n_params": fl[0]["n_params"],
            "withdrawal_mse": float(np.mean(c12)),
            "withdrawal_mse_se": float(np.std(c12, ddof=1) / np.sqrt(len(c12))),
            "withdrawal_r2": float(np.mean([fd["withdrawal_r2"] for fd in fl])),
            "top30_withdrawal_mse": float(np.mean(t30_12)),
            "top30_withdrawal_mse_se": float(np.std(t30_12, ddof=1) / np.sqrt(len(t30_12))),
            "collectri_auroc_stage1": float(np.mean(col_aurocs_obs)),
            "collectri_ap_over_random_stage1": float(np.mean(col_aps_obs)),
            "collectri_ap_over_random_full": float(np.mean(col_aps_full)),
        }

    # 5. Binance 16-asset & 14-altcoin PO
    fin_jobs = (
        [(False, mk, fs, cr, uc) for mk, fs, cr, uc in FIN_MODELS]
        + [(True, mk, fs, cr, uc) for mk, fs, cr, uc in FIN_PO_MODELS]
    )
    with ProcessPoolExecutor(max_workers=16, mp_context=ctx) as pool:
        fin_raw = list(pool.map(_run_fin_job, fin_jobs))
    fin_full, fin_po = {}, {}
    fin_full_eps, fin_po_eps = {}, {}
    for is_po, mk, r in fin_raw:
        ep = np.asarray(r.pop("per_ep_mse"), dtype=np.float64)
        if is_po:
            fin_po[mk] = r
            fin_po_eps[mk] = ep
        else:
            fin_full[mk] = r
            fin_full_eps[mk] = ep
    fin_results = {
        "full_16asset": fin_full,
        "full_holm_vs_memgrn": _two_sided_holm_stats(
            fin_full_eps["memgrn"], {k: v for k, v in fin_full_eps.items() if k != "memgrn"}
        ),
        "partial_14alt": fin_po,
        "partial_holm_vs_memgrn_ortho": _two_sided_holm_stats(
            fin_po_eps["memgrn_ortho"], {k: v for k, v in fin_po_eps.items() if k != "memgrn_ortho"}
        ),
    }

    # 6. Yeast Msn2 Joint Models + Depthwise SSM
    with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as pool:
        yeast_raw = dict(pool.map(_run_yeast_model, YEAST_MODELS))
    yeast_eps = {k: np.asarray(v["per_train_mse"], dtype=np.float64) for k, v in yeast_raw.items()}
    yeast_results = {
        "models": yeast_raw,
        "holm_vs_memgrn": _two_sided_holm_stats(
            yeast_eps["memgrn"], {k: v for k, v in yeast_eps.items() if k != "memgrn"}
        ),
        "holm_vs_memgrn_sym": _two_sided_holm_stats(
            yeast_eps["memgrn_sym"], {k: v for k, v in yeast_eps.items() if k != "memgrn_sym"}
        ),
    }

    # 7. Complete S=8..512 Scaling
    scaling_rows = _run_complete_scaling()
    elapsed = time.time() - t0

    design = {
        "benchmark": "fair_equal_budget_open_loop_suite",
        "equal_validation_budgets": True,
        "gate_shrinkage_grid_all_models": [1.0],
        "hysteresis_shrinkage_grid_all_models": [1.0],
        "diagonal_free_cooperativity": True,
        "memgrn_ortho_spec": "single_backbone_latent_ode_global_scalar_beta",
        "emt_tf_selection": "strict_source_cell_lines_only",
        "stats_test": "two_sided_wilcoxon_holm_bonferroni",
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "beijing_air_quality": bq_results,
        "multitf_24world": mtf_summary,
        "partial_obs_6of8": {
            "summary": po_sum,
            "holm_mse_vs_memgrn_ortho_2stage": _two_sided_holm_stats(
                po_mse_vecs["memgrn_ortho_2stage"],
                {k: v for k, v in po_mse_vecs.items() if k != "memgrn_ortho_2stage"},
            ),
            "holm_full_attr_vs_memgrn_ortho_2stage": _two_sided_holm_stats(
                po_full_attr_vecs["memgrn_ortho_2stage"],
                {k: v for k, v in po_full_attr_vecs.items() if k != "memgrn_ortho_2stage"},
            ),
        },
        "ltc_and_dssm_separation": ltc_sep,
        "emt_gse147405": emt_sum,
        "binance_contagion": fin_results,
        "yeast_msn2_joint": yeast_results,
        "scaling_s8_to_s512": scaling_rows,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 1),
                "beijing_delta_u_top5": sorted(
                    [(k, round(v["loso_mse"], 4)) for k, v in bq_results["delta_u_4season"]["summary"].items()],
                    key=lambda x: x[1],
                )[:6],
                "beijing_raw_u_top5": sorted(
                    [(k, round(v["loso_mse"], 4)) for k, v in bq_results["raw_u_4season"]["summary"].items()],
                    key=lambda x: x[1],
                )[:6],
                "mtf_coop_top5": sorted(
                    [(k, round(v["test_mse"], 4), round(v["attribution_rmse"], 4)) for k, v in mtf_summary["cooperative"]["means"].items()],
                    key=lambda x: x[1],
                )[:6],
                "po6of8": {
                    k: (round(v["test_mse"], 4), round(v["full_model_gt_attr_rmse"], 4), round(v["obs_branch_gt_attr_rmse"], 4))
                    for k, v in po_sum.items()
                },
                "ltc_sep": {k: (round(v["delayed_peak_post_mse"], 5), round(v["steady_state_amp_invariance_error"], 4), round(v["steady_state_weight_rel_error"], 4)) for k, v in ltc_sep.items()},
                "emt_top5": sorted(
                    [(k, round(v["top30_withdrawal_mse"], 4), round(v["collectri_ap_over_random_stage1"], 2)) for k, v in emt_sum.items()],
                    key=lambda x: x[1],
                )[:6],
                "yeast_top5": sorted(
                    [(k, round(v["val_loss"], 4), round(v["ten_train_post_mse"], 4)) for k, v in yeast_raw.items()],
                    key=lambda x: x[2],
                )[:6],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
