"""Strictly Matched Closed-Loop Benchmark Suite (Addressing R1-M2/M6/M7, R2-M1/M2/M3/M6, R3-M2).

Runs ALL closed-loop benchmarks using the repaired `baselines.py`:
  - Exact diagonal-free, hollow-masked factorized cooperativity.
  - dt-independent, rest-centered continuous closed-loop CfC.
  - Restored direct path (act @ d_hol) in closed-loop Latent ODE.
  - New Depthwise (source-isolated) Selective SSM ablation (`cl_depthwise_ssm`).
  - 100% matched `readout_rank`, `use_cooperativity`, `group_l21_scale`, and `use_self_memory`
    across ALL four source-isolated families (`cl_memgrn`, `cl_local_const`, `cl_ltc_local`,
    `cl_depthwise_ssm`).
  - Two-sided Holm-Bonferroni corrected Wilcoxon tests, separate `mem_params` vs `n_params`,
    and separated raw vs column-normalized graph metrics (zero mixing!).

Benchmarks:
  1. Matched Full S=207 METR-LA Traffic (120/40/160 episodic subsample, 15 models).
  2. Matched IEEE 39-Bus Physics-Inspired Cascading-Load ODE (12 worlds; BOTH hollow and
     +self_memory across all 4 source-isolated models + all coupled baselines).
  3. Strictly Non-Overlapping SNAP Reddit (S=24, stride=24 test windows = 12 independent
     24-day episodes, zero overlap; plus 47-window summary for continuity).
  4. Matched Primary Human CD4+ T-Cell Perturb-seq (GSE314342, S=128 TFs, N=256 genes, 5 folds;
     matched readout_rank=24 / coop / group_l21 across all 4 source-isolated models; RENGE
     scored strictly on CollecTRI graph recovery only, not unseen-KO forecasting).
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
from sklearn.linear_model import Ridge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    _init_closed_loop_tree,
    build_closed_loop_jax_rollout,
    fit_closed_loop_network,
)
from memgrn.cross_domain_datasets import (  # noqa: E402
    load_metr_la_traffic,
    load_reddit_temporal_network,
)
from memgrn.datasets import fetch_and_prepare_marson_cd4t_perturbseq  # noqa: E402
from memgrn.discrete_baselines import (  # noqa: E402
    DiscreteModelFamily,
    fit_discrete_closed_loop,
)
from memgrn.powergrid import generate_ieee39_powergrid_world  # noqa: E402

OUT_DIR = ROOT / "results" / "fair_closed_loop"
RAW_ROOT = ROOT / "data" / "raw"

READOUT_KEYS = {
    "direct",
    "delayed",
    "delayed_u",
    "delayed_v",
    "readout",
    "decay",
    "in_gain",
    "coop_u",
    "coop_v",
    "coop_w",
    "self_mem",
}

SOURCE_ISOLATED_CL = (
    ClosedLoopModelFamily.MEMGRN,
    ClosedLoopModelFamily.LOCAL_CONST,
    ClosedLoopModelFamily.LTC_LOCAL,
    ClosedLoopModelFamily.DEPTHWISE_SSM,
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


def _two_sided_holm_stats(ref: np.ndarray, others: dict[str, np.ndarray]) -> dict[str, dict]:
    keys = list(others.keys())
    p_list, raw = [], {}
    for k in keys:
        oth = others[k]
        p_two = float(wilcoxon(ref, oth, alternative="two-sided").pvalue) if np.any(ref != oth) else 1.0
        p_list.append(p_two)
        raw[k] = {
            "pct_reduction": float(100.0 * (1.0 - np.mean(ref) / max(1e-12, np.mean(oth)))),
            "wins": int(np.sum(ref < oth)),
            "total": int(len(ref)),
            "wilcoxon_p_two_sided": p_two,
        }
    m = len(keys)
    order = np.argsort(p_list)
    adj = np.zeros(m, dtype=np.float64)
    rmax = 0.0
    for rank, idx in enumerate(order):
        rmax = max(rmax, min(1.0, p_list[idx] * (m - rank)))
        adj[idx] = rmax
    for idx, k in enumerate(keys):
        raw[k]["holm_p_two_sided"] = float(adj[idx])
    return raw


# =============================================================================
# Task 1: Matched Full S=207 METR-LA (120/40/160 episodic subsample)
# =============================================================================
def _run_metr207_job(job: tuple[str, bool]) -> tuple[str, dict]:
    model_key, use_self_mem = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    n_g = 207
    data = load_metr_la_traffic(
        RAW_ROOT / "metr_la", n_sensors=207, max_train_eps=120, max_val_eps=40, max_test_eps=160
    )
    t0 = time.perf_counter()
    if model_key == "persistence":
        pred_fut = np.repeat(data.test_episodes[:, 11:12, :], 12, axis=1)
        n_params, mem_params = 0, 0
        graph = np.zeros((n_g, n_g))
        out_key = "persistence"
    elif model_key in ("dlinear", "gru", "transformer"):
        fit_d = fit_discrete_closed_loop(
            family=DiscreteModelFamily(model_key),
            train_hist=data.train_episodes[:, :12, :],
            train_fut=data.train_episodes[:, 12:, :],
            train_fut_mask=data.train_masks[:, 12:, :],
            val_hist=data.val_episodes[:, :12, :],
            val_fut=data.val_episodes[:, 12:, :],
            val_fut_mask=data.val_masks[:, 12:, :],
            hidden_dim=64,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2),
            total_updates=450,
            lr_max=0.005,
        )
        pred_fut = fit_d.predict_fn(data.test_episodes[:, :12, :])
        n_params, mem_params = fit_d.n_params, fit_d.n_params
        graph = fit_d.interaction_matrix
        out_key = model_key
    else:
        fam = ClosedLoopModelFamily(model_key)
        is_iso = fam in SOURCE_ISOLATED_CL

        def _pack(eps, msk):
            b = len(eps)
            ko = np.zeros((b, n_g))
            yo = np.concatenate([np.zeros((b, 1, n_g)), eps], axis=1)
            wo = np.zeros_like(yo)
            wo[:, 13:, :] = msk[:, 12:, :]
            yc = np.zeros((24, b, n_g))
            yc[:12] = np.transpose(eps[:, :12, :], (1, 0, 2))
            return ko, yo, wo, yc

        tr_ko, tr_y, tr_w, tr_cl = _pack(data.train_episodes, data.train_masks)
        va_ko, va_y, va_w, va_cl = _pack(data.val_episodes, data.val_masks)
        te_ko, _, _, te_cl = _pack(data.test_episodes, data.test_masks)
        # IDENTICAL features across ALL 4 source-isolated families!
        spec = ClosedLoopSpec(
            family=fam,
            n_genes=n_g,
            regulator_indices=tuple(range(n_g)),
            hollow=True,
            use_self_memory=use_self_mem,
            use_cooperativity=is_iso,
            coop_rank=8,
            readout_rank=(24 if is_iso else None),
            n_channels=3,
            hidden_width=5,
            module_dim=48,
            latent_dim=48,
            base_rates=(0.25, 1.0, 3.0),
        )
        fit = fit_closed_loop_network(
            spec=spec,
            basal_levels=np.ones(n_g),
            train_ko=tr_ko,
            train_y=tr_y,
            train_weights=tr_w,
            train_y_clamp=tr_cl,
            val_ko=va_ko,
            val_y=va_y,
            val_weights=va_w,
            val_y_clamp=va_cl,
            dt=0.25,
            n_steps=24,
            obs_step_indices=tuple(range(25)),
            clamp_steps=12,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2),
            group_l21_scale=(0.005 if is_iso else 0.0),
            total_updates=450,
            lr_max=0.006,
        )
        pred_fut = fit.predict_ko_fn(te_ko, None, None, te_cl)[:, 13:, :]
        n_params = fit.n_params
        t0_tree = _init_closed_loop_tree(spec, seed=0)
        mem_params = int(sum(int(v.size) for k, v in t0_tree.items() if k not in READOUT_KEYS))
        graph = fit.interaction_matrix
        out_key = f"{model_key}_selfmem" if use_self_mem else (f"{model_key}_hollow" if is_iso else model_key)

    rt = time.perf_counter() - t0
    true_fut, msk_fut = data.test_episodes[:, 12:, :], data.test_masks[:, 12:, :]
    err2 = ((pred_fut - true_fut) ** 2) * msk_fut
    abse = np.abs(pred_fut - true_fut) * msk_fut * 20.0
    ep_mse = np.sum(err2, axis=(1, 2)) / np.maximum(1.0, np.sum(msk_fut, axis=(1, 2)))
    ep_mae = np.sum(abse, axis=(1, 2)) / np.maximum(1.0, np.sum(msk_fut, axis=(1, 2)))
    shock_m = data.test_shock_flags

    off = ~np.eye(n_g, dtype=bool)
    labels = data.adj_binary[off]
    br = float(np.mean(labels))
    if np.max(graph[off]) > 1e-12:
        g_norm = graph / np.maximum(1e-8, np.max(graph, axis=0, keepdims=True))
        au_r, ap_r = _auroc_and_ap(graph[off], labels)
        au_n, ap_n = _auroc_and_ap(g_norm[off], labels)
    else:
        au_r, ap_r, au_n, ap_n = 0.5, br, 0.5, br

    return out_key, {
        "n_params": n_params,
        "mem_params": mem_params,
        "runtime_seconds": rt,
        "test_mse": float(np.mean(ep_mse)),
        "test_mse_se": float(np.std(ep_mse, ddof=1) / np.sqrt(len(ep_mse))),
        "test_mae_mph": float(np.mean(ep_mae)),
        "mae_15min_mph": float(np.sum(abse[:, 2, :]) / max(1.0, np.sum(msk_fut[:, 2, :]))),
        "mae_30min_mph": float(np.sum(abse[:, 5, :]) / max(1.0, np.sum(msk_fut[:, 5, :]))),
        "mae_60min_mph": float(np.sum(abse[:, 11, :]) / max(1.0, np.sum(msk_fut[:, 11, :]))),
        "shockwave_mse": float(np.mean(ep_mse[shock_m])),
        "shockwave_mae_mph": float(np.mean(ep_mae[shock_m])),
        "road_graph_raw_auroc": au_r,
        "road_graph_raw_ap_over_random": float(ap_r / max(1e-8, br)),
        "road_graph_colnorm_auroc": au_n,
        "road_graph_colnorm_ap_over_random": float(ap_n / max(1e-8, br)),
        "per_ep_mse": ep_mse.tolist(),
    }


# =============================================================================
# Task 2: Matched IEEE 39-Bus Physics-Inspired Cascading-Load ODE (12 worlds)
# =============================================================================
def _run_pg_job(job: tuple[int, str, bool]) -> tuple[int, str, dict]:
    seed, fam_val, use_self_mem = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    w = generate_ieee39_powergrid_world(seed)
    fam = ClosedLoopModelFamily(fam_val)
    is_iso = fam in SOURCE_ISOLATED_CL
    spec = ClosedLoopSpec(
        family=fam,
        n_genes=10,
        regulator_indices=tuple(range(10)),
        hollow=True,
        use_self_memory=use_self_mem,
        use_cooperativity=is_iso,
        coop_rank=6,
        n_channels=3,
        hidden_width=5,
        module_dim=20,
        latent_dim=24,
        base_rates=(0.3, 1.0, 3.0),
    )
    obs_tr = tuple(range(0, 101, 5))
    obs_ext = tuple(range(0, 181, 5))
    fit = fit_closed_loop_network(
        spec=spec,
        basal_levels=w.basal_load,
        train_ko=w.train_ko,
        train_y=w.train_y,
        train_weights=w.train_weights,
        val_ko=w.val_ko,
        val_y=w.val_y,
        val_weights=w.val_weights,
        dt=0.05,
        n_steps=100,
        obs_step_indices=obs_tr,
        seeds=(101, 202),
        reg_grid=(1e-4, 1e-3, 5e-3, 2e-2),
        l1_graph_ratio=0.25,
        group_l21_scale=(0.01 if is_iso else 0.0),
        total_updates=800,
        lr_max=0.006,
    )

    def _em(kos, nt):
        m = np.ones((len(kos), nt, 10))
        m[:, 0, :] = 0.0
        for b, kv in enumerate(kos):
            m[b, :, kv > 0.0] = 0.0
        return m

    p1 = fit.predict_ko_fn(w.test_single_ko, None, None, None)
    m1 = _em(w.test_single_ko, len(obs_tr))
    p2 = fit.predict_ko_fn(w.test_double_ko, None, None, None)
    m2 = _em(w.test_double_ko, len(obs_tr))

    ext_fwd, _, _ = build_closed_loop_jax_rollout(
        spec, basal_levels=w.basal_load, dt=0.05, n_steps=180, obs_step_indices=obs_ext
    )
    import jax.numpy as jnp

    pe = np.asarray(
        ext_fwd(
            jnp.asarray(fit.theta, dtype=jnp.float32),
            jnp.asarray(w.test_double_ko, dtype=jnp.float32),
            jnp.zeros((len(w.test_double_ko), 10), dtype=jnp.float32),
            None,
            None,
        ),
        dtype=np.float64,
    )
    me = _em(w.test_double_ko, len(obs_ext))
    me[:, : len(obs_tr), :] = 0.0

    off = ~np.eye(10, dtype=bool)
    auroc, ap = _auroc_and_ap(fit.interaction_matrix[off], w.adj_binary[off])
    t0_tree = _init_closed_loop_tree(spec, seed=0)
    mem_p = int(sum(int(v.size) for k, v in t0_tree.items() if k not in READOUT_KEYS))
    key = f"{fam_val}_selfmem" if use_self_mem else (f"{fam_val}_hollow" if is_iso else fam_val)
    return seed, key, {
        "n_params": fit.n_params,
        "mem_params": mem_p,
        "single_n1_mse": float(np.sum(((p1 - w.test_single_y) ** 2) * m1) / np.sum(m1)),
        "double_n2_mse": float(np.sum(((p2 - w.test_double_y) ** 2) * m2) / np.sum(m2)),
        "extrap_n2_mse": float(np.sum(((pe - w.test_double_y_extrap) ** 2) * me) / np.sum(me)),
        "graph_auroc": auroc,
        "graph_ap": ap,
    }


# =============================================================================
# Task 3: Strictly Non-Overlapping SNAP Reddit (S=24, stride=24 test windows)
# =============================================================================
def _run_reddit_job(job: tuple[str, bool]) -> tuple[str, dict]:
    model_key, use_self_mem = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    data = load_reddit_temporal_network(RAW_ROOT / "reddit_temporal", n_subreddits=24, episode_days=24)
    n_g = 24
    # Strictly non-overlapping 24-day test windows (stride=24 = every 4th stride-6 window -> 0% overlap!)
    nonov_idx = np.arange(0, len(data.test_episodes), 4, dtype=np.int64)

    if model_key in ("dlinear", "gru", "transformer"):
        fit_d = fit_discrete_closed_loop(
            family=DiscreteModelFamily(model_key),
            train_hist=data.train_episodes[:, :12, :],
            train_fut=data.train_episodes[:, 12:, :],
            train_fut_mask=np.ones_like(data.train_episodes[:, 12:, :]),
            val_hist=data.val_episodes[:, :12, :],
            val_fut=data.val_episodes[:, 12:, :],
            val_fut_mask=np.ones_like(data.val_episodes[:, 12:, :]),
            hidden_dim=32,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2, 5e-2),
            total_updates=600,
            lr_max=0.005,
        )
        pred_all = fit_d.predict_fn(data.test_episodes[:, :12, :])
        n_params, mem_params = fit_d.n_params, fit_d.n_params
        graph = fit_d.interaction_matrix
        out_key = model_key
    else:
        fam = ClosedLoopModelFamily(model_key)
        is_iso = fam in SOURCE_ISOLATED_CL

        def _pack(eps):
            b = len(eps)
            ko = np.zeros((b, n_g))
            yo = np.concatenate([np.zeros((b, 1, n_g)), eps], axis=1)
            wo = np.zeros_like(yo)
            wo[:, 13:, :] = 1.0
            yc = np.zeros((24, b, n_g))
            yc[:12] = np.transpose(eps[:, :12, :], (1, 0, 2))
            return ko, yo, wo, yc

        tr_ko, tr_y, tr_w, tr_cl = _pack(data.train_episodes)
        va_ko, va_y, va_w, va_cl = _pack(data.val_episodes)
        te_ko, _, _, te_cl = _pack(data.test_episodes)
        spec = ClosedLoopSpec(
            family=fam,
            n_genes=n_g,
            regulator_indices=tuple(range(n_g)),
            hollow=True,
            use_self_memory=use_self_mem,
            use_cooperativity=is_iso,
            coop_rank=6,
            n_channels=3,
            hidden_width=5,
            module_dim=24,
            latent_dim=24,
            base_rates=(0.25, 1.0, 3.0),
        )
        fit = fit_closed_loop_network(
            spec=spec,
            basal_levels=np.ones(n_g),
            train_ko=tr_ko,
            train_y=tr_y,
            train_weights=tr_w,
            train_y_clamp=tr_cl,
            val_ko=va_ko,
            val_y=va_y,
            val_weights=va_w,
            val_y_clamp=va_cl,
            dt=0.25,
            n_steps=24,
            obs_step_indices=tuple(range(25)),
            clamp_steps=12,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2, 5e-2),
            group_l21_scale=(0.01 if is_iso else 0.0),
            total_updates=650,
            lr_max=0.006,
        )
        pred_all = fit.predict_ko_fn(te_ko, None, None, te_cl)[:, 13:, :]
        n_params = fit.n_params
        t0_tree = _init_closed_loop_tree(spec, seed=0)
        mem_params = int(sum(int(v.size) for k, v in t0_tree.items() if k not in READOUT_KEYS))
        graph = fit.interaction_matrix
        out_key = f"{model_key}_selfmem" if use_self_mem else (f"{model_key}_hollow" if is_iso else model_key)

    true_all = data.test_episodes[:, 12:, :]
    ep_all = np.mean((pred_all - true_all) ** 2, axis=(1, 2))
    ep_no = ep_all[nonov_idx]
    no_mse = float(np.mean(ep_no))
    no_var = float(np.mean(true_all[nonov_idx] ** 2)) + 1e-8
    off = ~np.eye(n_g, dtype=bool)
    labels = data.test_adj_binary[off]
    br = float(np.mean(labels))
    au_r, ap_r = _auroc_and_ap(graph[off], labels) if np.max(graph[off]) > 1e-12 else (0.5, br)
    return out_key, {
        "n_params": n_params,
        "mem_params": mem_params,
        "nonoverlap_n_episodes": int(len(ep_no)),
        "nonoverlap_test_mse": no_mse,
        "nonoverlap_test_mse_se": float(np.std(ep_no, ddof=1) / np.sqrt(len(ep_no))),
        "nonoverlap_test_r2": 1.0 - no_mse / no_var,
        "overlap47_test_mse": float(np.mean(ep_all)),
        "hyperlink_graph_auroc": au_r,
        "hyperlink_graph_ap_over_random": float(ap_r / max(1e-8, br)),
        "per_nonoverlap_ep_mse": ep_no.tolist(),
    }


# =============================================================================
# Task 4: Matched Primary Human CD4+ T-Cell Perturb-seq (GSE314342, S=128)
# =============================================================================
def _run_cd4t_job(job: tuple[int, str, list[list[int]]]) -> tuple[int, str, dict]:
    fold_i, fam_val, folds = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    n_tfs, n_targets = 128, 128
    data = fetch_and_prepare_marson_cd4t_perturbseq(RAW_ROOT / "marson_cd4t", n_tfs, n_targets)
    n_ko, n_genes = len(data.ko_genes), len(data.genes)
    reg_tuple = tuple(int(i) for i in data.ko_gene_indices)
    test_idx = folds[fold_i]
    val_idx = folds[(fold_i + 1) % len(folds)]
    train_idx = [i for i in range(n_ko) if i not in test_idx and i not in val_idx]

    tr_raw = data.ko_delta_trajectories[train_idx]
    gene_scales = np.maximum(0.10, np.sqrt(np.mean(tr_raw**2, axis=(0, 1))))
    basal_scaled = data.basal_expression / gene_scales
    top50_idx = np.argsort(-np.sqrt(np.mean(tr_raw[:, 2, :] ** 2, axis=0)))[:50]

    def _pack(idxs):
        ko_b = data.ko_binary_masks[idxs]
        y3 = data.ko_delta_trajectories[idxs] / gene_scales[None, None, :]
        y_obs = np.concatenate([np.zeros((len(idxs), 1, n_genes)), y3], axis=1)
        w_obs = np.zeros_like(y_obs)
        w_obs[:, 3, :] = data.eval_target_masks[idxs, 2, :]
        y_cl = np.zeros((20, len(idxs), n_genes))
        for s in range(2):
            y_cl[s] = ((s + 1) / 2.0) * y3[:, 0, :]
        for s in range(2, 4):
            f = (s - 1) / 2.0
            y_cl[s] = (1.0 - f) * y3[:, 0, :] + f * y3[:, 1, :]
        return ko_b, y3, y_obs, w_obs, y_cl

    tr_ko, tr_y3, tr_y, tr_w, tr_cl = _pack(train_idx)
    va_ko, _, va_y, va_w, va_cl = _pack(val_idx)
    te_ko, te_y3, te_y, te_w, te_cl = _pack(test_idx)

    t0 = time.perf_counter()
    if fam_val == "zero":
        pred48 = np.zeros_like(te_y[:, 3, :])
        n_params, mem_params = 0, 0
        graph = np.zeros((n_tfs, n_genes))
    elif fam_val == "persistence":
        pred48 = te_y3[:, 1, :].copy()
        n_params, mem_params = 0, 0
        graph = np.zeros((n_tfs, n_genes))
    elif fam_val == "renge_graph_only":
        # Static knockout ridge graph recovery ONLY (no degenerate zero unseen-KO forecast!)
        x_ko = np.zeros_like(tr_ko)
        x_ko[:, data.ko_gene_indices] = -tr_ko[:, data.ko_gene_indices] * basal_scaled[None, data.ko_gene_indices]
        ridge = Ridge(alpha=5.0, fit_intercept=False).fit(x_ko, np.mean(tr_y3, axis=1))
        a_mat = ridge.coef_.copy() * (1.0 - np.eye(n_genes))
        graph = np.abs(a_mat.T[:n_tfs, :])
        pred48 = None
        n_params, mem_params = n_tfs * n_genes, 0
    else:
        fam = ClosedLoopModelFamily(fam_val)
        is_iso = fam in SOURCE_ISOLATED_CL
        # IDENTICAL readout_rank=24, cooperativity, & group_l21 across ALL 4 source-isolated families!
        spec = ClosedLoopSpec(
            family=fam,
            n_genes=n_genes,
            regulator_indices=reg_tuple,
            hollow=True,
            use_cooperativity=is_iso,
            coop_rank=8,
            readout_rank=(24 if is_iso else None),
            base_rates=(0.4, 1.2, 3.5),
            n_channels=3,
            hidden_width=5,
            module_dim=48,
            latent_dim=48,
        )
        fit = fit_closed_loop_network(
            spec=spec,
            basal_levels=basal_scaled,
            train_ko=tr_ko,
            train_y=tr_y,
            train_weights=tr_w,
            train_y_clamp=tr_cl,
            val_ko=va_ko,
            val_y=va_y,
            val_weights=va_w,
            val_y_clamp=va_cl,
            dt=0.25,
            n_steps=20,
            obs_step_indices=(0, 2, 4, 20),
            clamp_steps=4,
            seeds=(101, 202),
            reg_grid=(1e-3, 1e-2, 5e-2),
            group_l21_scale=(0.01 if is_iso else 0.0),
            total_updates=450,
            lr_max=0.005,
        )
        pred48 = fit.predict_ko_fn(te_ko, None, None, te_cl)[:, 3, :]
        n_params = fit.n_params
        t0_tree = _init_closed_loop_tree(spec, seed=0)
        mem_params = int(sum(int(v.size) for k, v in t0_tree.items() if k not in READOUT_KEYS))
        graph = fit.interaction_matrix[:n_tfs, :]

    fsec = time.perf_counter() - t0
    true48, mask48 = te_y[:, 3, :], te_w[:, 3, :]
    top50_mask = np.zeros_like(mask48)
    top50_mask[:, top50_idx] = mask48[:, top50_idx]
    if pred48 is None:
        return fold_i, fam_val, {
            "n_params": n_params,
            "mem_params": mem_params,
            "fold_seconds": fsec,
            "graph": graph.tolist(),
            "test_tf_indices": test_idx,
            "forecast_supported": False,
        }

    per_tf_mse, per_tf_t50, per_tf_r = [], [], []
    for b_i in range(len(test_idx)):
        m = mask48[b_i] > 0.5
        e2 = (pred48[b_i] - true48[b_i]) ** 2
        per_tf_mse.append(float(np.mean(e2[m])))
        per_tf_t50.append(float(np.mean(e2[top50_mask[b_i] > 0.5])))
        if np.std(pred48[b_i, m]) > 1e-8 and np.std(true48[b_i, m]) > 1e-8:
            per_tf_r.append(float(np.corrcoef(pred48[b_i, m], true48[b_i, m])[0, 1]))
        else:
            per_tf_r.append(0.0)
    f_mse = float(np.sum(((pred48 - true48) ** 2) * mask48) / np.sum(mask48))
    f_r2 = 1.0 - f_mse / (float(np.sum((true48**2) * mask48) / np.sum(mask48)) + 1e-8)
    return fold_i, fam_val, {
        "n_params": n_params,
        "mem_params": mem_params,
        "fold_seconds": fsec,
        "mse": f_mse,
        "r2": f_r2,
        "pearson_r": float(np.mean(per_tf_r)),
        "test_tf_indices": test_idx,
        "per_tf_mse": per_tf_mse,
        "per_tf_top50": per_tf_t50,
        "graph": graph.tolist(),
        "forecast_supported": True,
    }


def _dispatch_job(tagged_job: tuple[str, object]) -> tuple[str, object]:
    stage, payload = tagged_job
    if stage == "metr":
        return stage, _run_metr207_job(payload)
    if stage == "pg":
        return stage, _run_pg_job(payload)
    if stage == "reddit":
        return stage, _run_reddit_job(payload)
    return stage, _run_cd4t_job(payload)


def main() -> None:
    from concurrent.futures import as_completed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")

    # 1. Matched Full S=207 METR-LA (dispatch slowest cl_coupled_adaptive FIRST at t=0!)
    metr_jobs = [
        ("cl_coupled_adaptive", False),
        ("cl_memgrn", True),
        ("cl_memgrn", False),
        ("cl_local_const", True),
        ("cl_ltc_local", True),
        ("cl_depthwise_ssm", True),
        ("cl_selective_ssm", False),
        ("cl_cfc", False),
        ("cl_ltc_coupled", False),
        ("cl_perturb_ode", False),
        ("cl_latent_ode", False),
        ("dlinear", False),
        ("gru", False),
        ("transformer", False),
        ("persistence", False),
    ]

    # 2. Matched IEEE 39-Bus (12 worlds; BOTH hollow AND +self_mem for all 4 source-isolated families)
    pg_models = [
        ("cl_memgrn", True),
        ("cl_memgrn", False),
        ("cl_local_const", True),
        ("cl_local_const", False),
        ("cl_ltc_local", True),
        ("cl_ltc_local", False),
        ("cl_depthwise_ssm", True),
        ("cl_depthwise_ssm", False),
        ("cl_selective_ssm", False),
        ("cl_cfc", False),
        ("cl_coupled_adaptive", False),
        ("cl_ltc_coupled", False),
        ("cl_perturb_ode", False),
        ("cl_latent_ode", False),
    ]
    pg_jobs = [(7100 + w, m, sm) for w in range(12) for m, sm in pg_models]

    # 3. Strictly Non-Overlapping SNAP Reddit (S=24)
    red_jobs = [
        ("cl_memgrn", True),
        ("cl_memgrn", False),
        ("cl_local_const", True),
        ("cl_ltc_local", True),
        ("cl_depthwise_ssm", True),
        ("cl_selective_ssm", False),
        ("cl_cfc", False),
        ("cl_coupled_adaptive", False),
        ("cl_ltc_coupled", False),
        ("cl_perturb_ode", False),
        ("cl_latent_ode", False),
        ("dlinear", False),
        ("gru", False),
        ("transformer", False),
    ]

    # 4. Matched Primary Human CD4+ T-Cell Perturb-seq (GSE314342, S=128)
    cd4_data = fetch_and_prepare_marson_cd4t_perturbseq(RAW_ROOT / "marson_cd4t", 128, 128)
    folds = [list(range(f, len(cd4_data.ko_genes), 5)) for f in range(5)]
    cd4_models = [
        "cl_coupled_adaptive",
        "cl_memgrn",
        "cl_local_const",
        "cl_ltc_local",
        "cl_depthwise_ssm",
        "cl_selective_ssm",
        "cl_cfc",
        "cl_ltc_coupled",
        "cl_perturb_ode",
        "cl_latent_ode",
        "renge_graph_only",
        "persistence",
        "zero",
    ]
    cd4_jobs = [(f_i, mk, folds) for f_i in range(5) for mk in cd4_models]

    # Dispatch ALL 4 stages in ONE shared 24-worker pool (longest jobs first -> zero idle tail!)
    all_tagged = (
        [("metr", j) for j in metr_jobs]
        + [("cd4t", j) for j in cd4_jobs]
        + [("reddit", j) for j in red_jobs]
        + [("pg", j) for j in pg_jobs]
    )
    stage_totals = {
        "metr": len(metr_jobs),
        "cd4t": len(cd4_jobs),
        "reddit": len(red_jobs),
        "pg": len(pg_jobs),
    }
    stage_done = {k: 0 for k in stage_totals}
    metr_raw: dict[str, dict] = {}
    pg_raw: list[tuple[int, str, dict]] = []
    red_raw: dict[str, dict] = {}
    cd4_raw: list[tuple[int, str, dict]] = []

    print(
        f"[fair_closed_loop] Launching {len(all_tagged)} jobs across 24 workers "
        f"(metr={len(metr_jobs)}, cd4t={len(cd4_jobs)}, reddit={len(red_jobs)}, pg={len(pg_jobs)})",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=24, mp_context=ctx) as pool:
        futs = [pool.submit(_dispatch_job, tj) for tj in all_tagged]
        for fut in as_completed(futs):
            stage, res = fut.result()
            stage_done[stage] += 1
            if stage == "metr":
                k, d = res
                metr_raw[k] = d
                print(
                    f"[metr {stage_done['metr']}/{stage_totals['metr']}] {k}: "
                    f"mse={d['test_mse']:.4f}, mae={d['test_mae_mph']:.2f} mph ({d['runtime_seconds']:.1f}s)",
                    flush=True,
                )
            elif stage == "pg":
                pg_raw.append(res)
            elif stage == "reddit":
                k, d = res
                red_raw[k] = d
            elif stage == "cd4t":
                cd4_raw.append(res)
            if stage_done[stage] == stage_totals[stage]:
                print(
                    f"[STAGE COMPLETE] {stage} ({stage_totals[stage]} jobs) at t={time.time() - t0:.1f}s",
                    flush=True,
                )

    metr_eps = {k: np.asarray(v.pop("per_ep_mse"), dtype=np.float64) for k, v in metr_raw.items()}
    metr_holm = _two_sided_holm_stats(
        metr_eps["cl_memgrn_selfmem"], {k: v for k, v in metr_eps.items() if k != "cl_memgrn_selfmem"}
    )

    pg_by_k: dict[str, list[dict]] = {}
    for _s, k, r in pg_raw:
        pg_by_k.setdefault(k, []).append(r)
    pg_sum, pg_n2_vecs = {}, {}
    for k, lst in pg_by_k.items():
        n2 = np.array([x["double_n2_mse"] for x in lst])
        n1 = np.array([x["single_n1_mse"] for x in lst])
        ex = np.array([x["extrap_n2_mse"] for x in lst])
        au = np.array([x["graph_auroc"] for x in lst])
        ap = np.array([x["graph_ap"] for x in lst])
        pg_n2_vecs[k] = n2
        pg_sum[k] = {
            "n_params": lst[0]["n_params"],
            "mem_params": lst[0]["mem_params"],
            "single_n1_mse": float(np.mean(n1)),
            "single_n1_mse_se": float(np.std(n1, ddof=1) / np.sqrt(len(n1))),
            "double_n2_mse": float(np.mean(n2)),
            "double_n2_mse_se": float(np.std(n2, ddof=1) / np.sqrt(len(n2))),
            "extrap_n2_mse": float(np.mean(ex)),
            "graph_auroc": float(np.mean(au)),
            "graph_ap": float(np.mean(ap)),
        }
    pg_holm_self = _two_sided_holm_stats(
        pg_n2_vecs["cl_memgrn_selfmem"], {k: v for k, v in pg_n2_vecs.items() if k != "cl_memgrn_selfmem"}
    )
    pg_holm_hol = _two_sided_holm_stats(
        pg_n2_vecs["cl_memgrn_hollow"], {k: v for k, v in pg_n2_vecs.items() if k != "cl_memgrn_hollow"}
    )

    red_eps = {k: np.asarray(v.pop("per_nonoverlap_ep_mse"), dtype=np.float64) for k, v in red_raw.items()}
    red_holm = _two_sided_holm_stats(
        red_eps["cl_memgrn_selfmem"], {k: v for k, v in red_eps.items() if k != "cl_memgrn_selfmem"}
    )

    cd4_by_m: dict[str, list[dict]] = {m: [] for m in cd4_models}
    cd4_graphs: dict[str, list[np.ndarray]] = {m: [] for m in cd4_models}
    for _f, mk, r in cd4_raw:
        cd4_graphs[mk].append(np.asarray(r.pop("graph"), dtype=np.float64))
        cd4_by_m[mk].append(r)

    off_m = np.ones((128, len(cd4_data.genes)), dtype=bool)
    for i in range(128):
        off_m[i, i] = False
    col_lbl = cd4_data.collectri_edge_mask[off_m]
    col_br = float(np.mean(col_lbl))

    cd4_sum, cd4_tf_vecs = {}, {}
    for mk in cd4_models:
        fl = cd4_by_m[mk]
        mean_g = np.mean(np.stack(cd4_graphs[mk], axis=0), axis=0)
        if np.max(mean_g) > 1e-12:
            cn = mean_g / np.maximum(1e-8, np.max(mean_g, axis=0, keepdims=True))
            au_n, ap_n = _auroc_and_ap(cn[off_m], col_lbl)
            au_r, ap_r = _auroc_and_ap(mean_g[off_m], col_lbl)
        else:
            au_n, ap_n, au_r, ap_r = 0.5, col_br, 0.5, col_br

        entry = {
            "n_params": fl[0]["n_params"],
            "mem_params": fl[0]["mem_params"],
            "mean_fold_seconds": float(np.mean([x["fold_seconds"] for x in fl])),
            "collectri_colnorm_auroc": au_n,
            "collectri_colnorm_ap_over_random": float(ap_n / max(1e-8, col_br)),
            "collectri_raw_auroc": au_r,
            "collectri_raw_ap_over_random": float(ap_r / max(1e-8, col_br)),
            "forecast_supported": fl[0]["forecast_supported"],
        }
        if fl[0]["forecast_supported"]:
            v_mse = np.zeros(128, dtype=np.float64)
            v_t50 = np.zeros(128, dtype=np.float64)
            for fd in fl:
                for itf, vm, v50 in zip(fd["test_tf_indices"], fd["per_tf_mse"], fd["per_tf_top50"]):
                    v_mse[itf], v_t50[itf] = vm, v50
            cd4_tf_vecs[mk] = v_mse
            entry["unseen_tf_48h_mse"] = float(np.mean(v_mse))
            entry["unseen_tf_48h_mse_se"] = float(np.std(v_mse, ddof=1) / np.sqrt(128))
            entry["unseen_tf_48h_r2"] = float(np.mean([x["r2"] for x in fl]))
            entry["unseen_tf_48h_pearson_r"] = float(np.mean([x["pearson_r"] for x in fl]))
            entry["unseen_tf_top50_mse"] = float(np.mean(v_t50))
        cd4_sum[mk] = entry

    cd4_holm = _two_sided_holm_stats(
        cd4_tf_vecs["cl_memgrn"], {k: v for k, v in cd4_tf_vecs.items() if k != "cl_memgrn"}
    )

    elapsed = time.time() - t0
    design = {
        "benchmark": "fair_matched_closed_loop_suite",
        "matched_source_isolated_features": ["readout_rank", "use_cooperativity", "group_l21_scale", "use_self_memory"],
        "diagonal_free_hollow_cooperativity": True,
        "fixed_dt_independent_cfc": True,
        "fixed_latent_ode_direct_path": True,
        "reddit_test_protocol": "strictly_nonoverlapping_stride24_12_episodes",
        "cd4t_renge_protocol": "graph_recovery_only_not_unseen_ko_forecasting",
        "stats_test": "two_sided_wilcoxon_holm_bonferroni",
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "metr_la_s207": {"summary": metr_raw, "holm_two_sided_vs_cl_memgrn_selfmem": metr_holm},
        "powergrid_ieee39": {
            "summary": pg_sum,
            "holm_n2_vs_cl_memgrn_selfmem": pg_holm_self,
            "holm_n2_vs_cl_memgrn_hollow": pg_holm_hol,
        },
        "reddit_nonoverlap": {"summary": red_raw, "holm_two_sided_vs_cl_memgrn_selfmem": red_holm},
        "cd4t_gse314342": {"summary": cd4_sum, "holm_two_sided_vs_cl_memgrn": cd4_holm},
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 1),
                "metr_la_s207_top6": sorted(
                    [(k, round(v["test_mse"], 4), round(v["test_mae_mph"], 2), round(v["road_graph_raw_ap_over_random"], 2)) for k, v in metr_raw.items()],
                    key=lambda x: x[1],
                )[:7],
                "powergrid_top6": sorted(
                    [(k, round(v["double_n2_mse"], 5), round(v["single_n1_mse"], 5), round(v["graph_auroc"], 3)) for k, v in pg_sum.items()],
                    key=lambda x: x[1],
                )[:7],
                "reddit_nonoverlap_top6": sorted(
                    [(k, round(v["nonoverlap_test_mse"], 4), round(v["hyperlink_graph_auroc"], 3)) for k, v in red_raw.items()],
                    key=lambda x: x[1],
                )[:7],
                "cd4t_top6": sorted(
                    [(k, round(v["unseen_tf_48h_mse"], 4), round(v["collectri_colnorm_ap_over_random"], 2), round(v["collectri_raw_ap_over_random"], 2)) for k, v in cd4_sum.items() if v["forecast_supported"]],
                    key=lambda x: x[1],
                )[:7],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
