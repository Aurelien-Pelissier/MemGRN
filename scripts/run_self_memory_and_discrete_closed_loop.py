"""Cross-domain closed-loop upgrades: Diagonal Self-Memory, Discrete Baselines, & Full S=207 METR-LA.

Implements Upgrades #1 and #3 cleanly across all closed-loop non-biology networks:
  1. Official Full S = 207 METR-LA Highway Traffic Network (all 207 Los Angeles loop detectors,
     all 1,515 physical directed road edges from adj_mx.pkl, 7.09M raw observations):
     Compares Self-Memory Closed-Loop MemGRN against Self-Memory Local Constant, Source-Local LTC,
     Selective SSM (Mamba), CfC, Coupled LTC, Coupled Adaptive (O(S^2 R^2)), PerturbODE, Latent
     Neural ODE, DLinear, GRU, Temporal Transformer, and Persistence (13 models total!).
  2. Self-Memory Ablation + Discrete Baselines (DLinear, GRU, Transformer) on locked METR-LA (S=64),
     SNAP Reddit (S=24), and IEEE 39-Bus Power Grid (N-1 -> N-2, 12 worlds).
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
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    _init_closed_loop_tree,
    fit_closed_loop_network,
)
from memgrn.cross_domain_datasets import (  # noqa: E402
    load_metr_la_traffic,
    load_reddit_temporal_network,
)
from memgrn.discrete_baselines import (  # noqa: E402
    DiscreteModelFamily,
    fit_discrete_closed_loop,
)
from memgrn.powergrid import generate_ieee39_powergrid_world  # noqa: E402

OUT_DIR = ROOT / "results" / "closed_loop_upgrades"
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


def _run_metr_job(job: tuple[int, str, bool]) -> tuple[int, str, dict]:
    n_sensors, model_key, use_self_mem = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    max_tr = 120 if n_sensors == 207 else 180
    max_va = 40 if n_sensors == 207 else 60
    data = load_metr_la_traffic(
        RAW_ROOT / "metr_la",
        n_sensors=n_sensors,
        max_train_eps=max_tr,
        max_val_eps=max_va,
        max_test_eps=160,
    )
    n_g = n_sensors
    t0 = time.perf_counter()

    if model_key == "persistence":
        pred_fut = np.repeat(data.test_episodes[:, 11:12, :], 12, axis=1)
        n_params, mem_params = 0, 0
        graph = np.zeros((n_g, n_g), dtype=np.float64)
    elif model_key in ("dlinear", "gru", "transformer"):
        dfam = DiscreteModelFamily(model_key)
        h_dim = 64 if n_sensors == 207 else 48
        fitted_d = fit_discrete_closed_loop(
            family=dfam,
            train_hist=data.train_episodes[:, :12, :],
            train_fut=data.train_episodes[:, 12:, :],
            train_fut_mask=data.train_masks[:, 12:, :],
            val_hist=data.val_episodes[:, :12, :],
            val_fut=data.val_episodes[:, 12:, :],
            val_fut_mask=data.val_masks[:, 12:, :],
            hidden_dim=h_dim,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2),
            total_updates=500,
            lr_max=0.005,
        )
        pred_fut = fitted_d.predict_fn(data.test_episodes[:, :12, :])
        n_params = fitted_d.n_params
        mem_params = fitted_d.n_params
        graph = fitted_d.interaction_matrix
    else:
        fam = ClosedLoopModelFamily(model_key)

        def _pack(eps: np.ndarray, msk: np.ndarray):
            b = len(eps)
            ko = np.zeros((b, n_g), dtype=np.float64)
            y_obs = np.concatenate([np.zeros((b, 1, n_g)), eps], axis=1)
            w_obs = np.zeros_like(y_obs)
            w_obs[:, 13:, :] = msk[:, 12:, :]
            y_cl = np.zeros((24, b, n_g), dtype=np.float64)
            y_cl[:12] = np.transpose(eps[:, :12, :], (1, 0, 2))
            return ko, y_obs, w_obs, y_cl

        tr_ko, tr_y, tr_w, tr_cl = _pack(data.train_episodes, data.train_masks)
        va_ko, va_y, va_w, va_cl = _pack(data.val_episodes, data.val_masks)
        te_ko, _, _, te_cl = _pack(data.test_episodes, data.test_masks)
        basal = np.ones(n_g, dtype=np.float64)

        spec = ClosedLoopSpec(
            family=fam,
            n_genes=n_g,
            regulator_indices=tuple(range(n_g)),
            hollow=True,
            use_self_memory=use_self_mem,
            use_cooperativity=(fam == ClosedLoopModelFamily.MEMGRN),
            coop_rank=8,
            readout_rank=(
                24
                if (n_sensors == 207 and fam in (ClosedLoopModelFamily.MEMGRN, ClosedLoopModelFamily.LOCAL_CONST))
                else (
                    20
                    if fam in (ClosedLoopModelFamily.MEMGRN, ClosedLoopModelFamily.LOCAL_CONST)
                    else None
                )
            ),
            n_channels=3,
            hidden_width=5,
            module_dim=48 if n_sensors == 207 else 32,
            latent_dim=48 if n_sensors == 207 else 32,
            base_rates=(0.25, 1.0, 3.0),
        )
        fitted = fit_closed_loop_network(
            spec=spec,
            basal_levels=basal,
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
            group_l21_scale=(0.005 if fam == ClosedLoopModelFamily.MEMGRN else 0.0),
            total_updates=450 if n_sensors == 207 else 550,
            lr_max=0.006,
        )
        pred_fut = fitted.predict_ko_fn(te_ko, None, None, te_cl)[:, 13:, :]
        n_params = fitted.n_params
        tree0 = _init_closed_loop_tree(spec, seed=0)
        mem_params = int(sum(int(v.size) for k, v in tree0.items() if k not in READOUT_KEYS))
        graph = fitted.interaction_matrix

    runtime_s = time.perf_counter() - t0
    true_fut = data.test_episodes[:, 12:, :]
    msk_fut = data.test_masks[:, 12:, :]
    err2 = ((pred_fut - true_fut) ** 2) * msk_fut
    abse = np.abs(pred_fut - true_fut) * msk_fut * 20.0
    ep_mse = np.sum(err2, axis=(1, 2)) / np.maximum(1.0, np.sum(msk_fut, axis=(1, 2)))
    ep_mae = np.sum(abse, axis=(1, 2)) / np.maximum(1.0, np.sum(msk_fut, axis=(1, 2)))
    shock_m = data.test_shock_flags

    def _hor_mae(h_idx: int) -> float:
        return float(np.sum(abse[:, h_idx, :]) / max(1.0, np.sum(msk_fut[:, h_idx, :])))

    off = ~np.eye(n_g, dtype=bool)
    labels = data.adj_binary[off]
    base_rate = float(np.mean(labels))
    if graph is not None and np.max(graph[off]) > 1e-12:
        g_norm = graph / np.maximum(1e-8, np.max(graph, axis=0, keepdims=True))
        auroc_n, ap_n = _auroc_and_ap(g_norm[off], labels)
        auroc_r, ap_r = _auroc_and_ap(graph[off], labels)
    else:
        auroc_n, ap_n, auroc_r, ap_r = 0.5, base_rate, 0.5, base_rate

    label_key = f"{model_key}_selfmem" if (use_self_mem and model_key.startswith("cl_")) else model_key
    return (
        n_sensors,
        label_key,
        {
            "n_params": n_params,
            "mem_params": mem_params,
            "runtime_seconds": runtime_s,
            "test_mse": float(np.mean(ep_mse)),
            "test_mse_se": float(np.std(ep_mse, ddof=1) / np.sqrt(len(ep_mse))),
            "test_mae_mph": float(np.mean(ep_mae)),
            "mae_15min_mph": _hor_mae(2),
            "mae_30min_mph": _hor_mae(5),
            "mae_60min_mph": _hor_mae(11),
            "shockwave_mse": float(np.mean(ep_mse[shock_m])),
            "shockwave_mae_mph": float(np.mean(ep_mae[shock_m])),
            "road_graph_auroc": auroc_r,
            "road_graph_ap_over_random": float(ap_r / max(1e-8, base_rate)),
            "road_graph_colnorm_auroc": auroc_n,
            "road_graph_colnorm_ap_over_random": float(ap_n / max(1e-8, base_rate)),
            "per_ep_mse": ep_mse.tolist(),
        },
    )


def _run_reddit_job(job: tuple[str, bool]) -> tuple[str, dict]:
    model_key, use_self_mem = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    data = load_reddit_temporal_network(RAW_ROOT / "reddit_temporal", n_subreddits=24, episode_days=24)
    n_g = 24
    if model_key in ("dlinear", "gru", "transformer"):
        dfam = DiscreteModelFamily(model_key)
        fitted_d = fit_discrete_closed_loop(
            family=dfam,
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
        pred_fut = fitted_d.predict_fn(data.test_episodes[:, :12, :])
        n_params = fitted_d.n_params
        graph = fitted_d.interaction_matrix
        out_key = model_key
    else:
        fam = ClosedLoopModelFamily(model_key)

        def _pack(eps: np.ndarray):
            b = len(eps)
            ko = np.zeros((b, n_g), dtype=np.float64)
            y_obs = np.concatenate([np.zeros((b, 1, n_g)), eps], axis=1)
            w_obs = np.zeros_like(y_obs)
            w_obs[:, 13:, :] = 1.0
            y_cl = np.zeros((24, b, n_g), dtype=np.float64)
            y_cl[:12] = np.transpose(eps[:, :12, :], (1, 0, 2))
            return ko, y_obs, w_obs, y_cl

        tr_ko, tr_y, tr_w, tr_cl = _pack(data.train_episodes)
        va_ko, va_y, va_w, va_cl = _pack(data.val_episodes)
        te_ko, _, _, te_cl = _pack(data.test_episodes)
        spec = ClosedLoopSpec(
            family=fam,
            n_genes=n_g,
            regulator_indices=tuple(range(n_g)),
            hollow=True,
            use_self_memory=use_self_mem,
            use_cooperativity=(fam == ClosedLoopModelFamily.MEMGRN),
            coop_rank=6,
            n_channels=3,
            hidden_width=5,
            module_dim=24,
            latent_dim=24,
            base_rates=(0.25, 1.0, 3.0),
        )
        fitted = fit_closed_loop_network(
            spec=spec,
            basal_levels=np.ones(n_g, dtype=np.float64),
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
            group_l21_scale=(0.01 if fam == ClosedLoopModelFamily.MEMGRN else 0.0),
            total_updates=650,
            lr_max=0.006,
        )
        pred_fut = fitted.predict_ko_fn(te_ko, None, None, te_cl)[:, 13:, :]
        n_params = fitted.n_params
        graph = fitted.interaction_matrix
        out_key = f"{model_key}_selfmem"

    true_fut = data.test_episodes[:, 12:, :]
    ep_mse = np.mean((pred_fut - true_fut) ** 2, axis=(1, 2))
    test_mse = float(np.mean(ep_mse))
    test_var = float(np.mean(true_fut**2)) + 1e-8
    off = ~np.eye(n_g, dtype=bool)
    labels = data.test_adj_binary[off]
    base_rate = float(np.mean(labels))
    auroc_r, ap_r = _auroc_and_ap(graph[off], labels) if np.max(graph[off]) > 1e-12 else (0.5, base_rate)
    return (
        out_key,
        {
            "n_params": n_params,
            "test_mse": test_mse,
            "test_mse_se": float(np.std(ep_mse, ddof=1) / np.sqrt(len(ep_mse))),
            "test_r2": 1.0 - test_mse / test_var,
            "hyperlink_graph_auroc": auroc_r,
            "hyperlink_graph_ap_over_random": float(ap_r / max(1e-8, base_rate)),
            "per_ep_mse": ep_mse.tolist(),
        },
    )


def _run_pg_selfmem_job(job: tuple[int, str]) -> tuple[int, str, dict]:
    seed, fam_val = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    w = generate_ieee39_powergrid_world(seed)
    fam = ClosedLoopModelFamily(fam_val)
    spec = ClosedLoopSpec(
        family=fam,
        n_genes=10,
        regulator_indices=tuple(range(10)),
        hollow=True,
        use_self_memory=True,
        use_cooperativity=(fam == ClosedLoopModelFamily.MEMGRN),
        coop_rank=6,
        n_channels=3,
        hidden_width=5,
        module_dim=20,
        latent_dim=24,
        base_rates=(0.3, 1.0, 3.0),
    )
    obs_tr = tuple(range(0, 101, 5))
    obs_ext = tuple(range(0, 181, 5))
    fitted = fit_closed_loop_network(
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
        group_l21_scale=(0.01 if fam == ClosedLoopModelFamily.MEMGRN else 0.0),
        total_updates=800,
        lr_max=0.006,
    )

    def _eval_mask(kos: np.ndarray, n_t: int) -> np.ndarray:
        m = np.ones((len(kos), n_t, 10), dtype=np.float64)
        m[:, 0, :] = 0.0
        for b_i, kv in enumerate(kos):
            m[b_i, :, kv > 0.0] = 0.0
        return m

    pred_1 = fitted.predict_ko_fn(w.test_single_ko, None, None, None)
    m1 = _eval_mask(w.test_single_ko, len(obs_tr))
    pred_2 = fitted.predict_ko_fn(w.test_double_ko, None, None, None)
    m2 = _eval_mask(w.test_double_ko, len(obs_tr))

    from memgrn.baselines import build_closed_loop_jax_rollout
    import jax.numpy as jnp

    fwd_ext, _, _ = build_closed_loop_jax_rollout(
        spec, basal_levels=w.basal_load, dt=0.05, n_steps=180, obs_step_indices=obs_ext
    )
    pred_2e = np.asarray(
        fwd_ext(
            jnp.asarray(fitted.theta, dtype=jnp.float32),
            jnp.asarray(w.test_double_ko, dtype=jnp.float32),
            jnp.zeros((len(w.test_double_ko), 10), dtype=jnp.float32),
            None,
            None,
        ),
        dtype=np.float64,
    )
    m2e = _eval_mask(w.test_double_ko, len(obs_ext))
    m2e[:, : len(obs_tr), :] = 0.0
    off = ~np.eye(10, dtype=bool)
    auroc, ap = _auroc_and_ap(fitted.interaction_matrix[off], w.adj_binary[off])
    return (
        seed,
        f"{fam_val}_selfmem",
        {
            "n_params": fitted.n_params,
            "single_n1_mse": float(np.sum(((pred_1 - w.test_single_y) ** 2) * m1) / np.sum(m1)),
            "double_n2_mse": float(np.sum(((pred_2 - w.test_double_y) ** 2) * m2) / np.sum(m2)),
            "horizon_extrap_mse": float(
                np.sum(((pred_2e - w.test_double_y_extrap) ** 2) * m2e) / np.sum(m2e)
            ),
            "graph_auroc": auroc,
            "graph_ap": ap,
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metr207_models = [
        ("cl_memgrn", True),
        ("cl_local_const", True),
        ("cl_ltc_local", True),
        ("cl_selective_ssm", False),
        ("cl_cfc", False),
        ("cl_ltc_coupled", False),
        ("cl_coupled_adaptive", False),
        ("cl_perturb_ode", False),
        ("cl_latent_ode", False),
        ("dlinear", False),
        ("gru", False),
        ("transformer", False),
        ("persistence", False),
    ]
    metr64_extra = [
        ("cl_memgrn", True),
        ("cl_local_const", True),
        ("cl_ltc_local", True),
        ("dlinear", False),
        ("gru", False),
        ("transformer", False),
    ]
    reddit_extra = [
        ("cl_memgrn", True),
        ("cl_local_const", True),
        ("cl_ltc_local", True),
        ("dlinear", False),
        ("gru", False),
        ("transformer", False),
    ]

    design = {
        "benchmark": "closed_loop_selfmem_discrete_and_full207_metr_la",
        "metr_la_full207_models": [m for m, _ in metr207_models],
        "metr_la_s64_upgrades": [m for m, _ in metr64_extra],
        "reddit_s24_upgrades": [m for m, _ in reddit_extra],
        "powergrid_selfmem_seeds": list(range(700, 712)),
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=18, mp_context=ctx) as pool:
        f_metr = [pool.submit(_run_metr_job, (207, m, sm)) for m, sm in metr207_models]
        f_m64 = [pool.submit(_run_metr_job, (64, m, sm)) for m, sm in metr64_extra]
        f_red = [pool.submit(_run_reddit_job, (m, sm)) for m, sm in reddit_extra]
        f_pg = [
            pool.submit(_run_pg_selfmem_job, (s, fm))
            for s in range(700, 712)
            for fm in ("cl_memgrn", "cl_local_const")
        ]
        res_207 = [f.result() for f in f_metr]
        res_64 = [f.result() for f in f_m64]
        res_red = [f.result() for f in f_red]
        res_pg = [f.result() for f in f_pg]
    elapsed = time.time() - t0

    def _pack_metr(rows):
        summ, ep = {}, {}
        for _, k, d in rows:
            ep[k] = np.asarray(d.pop("per_ep_mse"), dtype=np.float64)
            summ[k] = d
        ref = ep["cl_memgrn_selfmem"]
        wilc = {}
        for k, v in ep.items():
            if k == "cl_memgrn_selfmem":
                continue
            p_v = float(wilcoxon(ref, v, alternative="less").pvalue) if np.any(ref != v) else 1.0
            wilc[k] = {
                "pct_reduction": float(100.0 * (1.0 - np.mean(ref) / max(1e-12, np.mean(v)))),
                "wins": int(np.sum(ref < v)),
                "total": len(ref),
                "wilcoxon_p": p_v,
            }
        return summ, wilc

    m207_sum, m207_wilc = _pack_metr(res_207)
    m64_sum, m64_wilc = _pack_metr(res_64)

    red_sum, red_ep = {}, {}
    for k, d in res_red:
        red_ep[k] = np.asarray(d.pop("per_ep_mse"), dtype=np.float64)
        red_sum[k] = d
    ref_r = red_ep["cl_memgrn_selfmem"]
    red_wilc = {
        k: {
            "pct_reduction": float(100.0 * (1.0 - np.mean(ref_r) / max(1e-12, np.mean(v)))),
            "wins": int(np.sum(ref_r < v)),
            "wilcoxon_p": float(wilcoxon(ref_r, v, alternative="less").pvalue),
        }
        for k, v in red_ep.items()
        if k != "cl_memgrn_selfmem"
    }

    pg_by: dict[str, list[dict]] = {"cl_memgrn_selfmem": [], "cl_local_const_selfmem": []}
    for _, k, d in res_pg:
        pg_by[k].append(d)
    pg_sum = {
        k: {
            "n_params": rows[0]["n_params"],
            "single_n1_mse": float(np.mean([r["single_n1_mse"] for r in rows])),
            "double_n2_mse": float(np.mean([r["double_n2_mse"] for r in rows])),
            "double_n2_mse_se": float(np.std([r["double_n2_mse"] for r in rows], ddof=1) / np.sqrt(len(rows))),
            "horizon_extrap_mse": float(np.mean([r["horizon_extrap_mse"] for r in rows])),
            "graph_auroc": float(np.mean([r["graph_auroc"] for r in rows])),
            "graph_ap": float(np.mean([r["graph_ap"] for r in rows])),
        }
        for k, rows in pg_by.items()
    }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "metr_la_full207": m207_sum,
        "metr_la_full207_wilcoxon_vs_memgrn": m207_wilc,
        "metr_la_s64_upgrades": m64_sum,
        "metr_la_s64_wilcoxon": m64_wilc,
        "reddit_s24_upgrades": red_sum,
        "reddit_s24_wilcoxon": red_wilc,
        "powergrid_selfmem": pg_sum,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": round(elapsed, 1), "metr_la_full207": m207_sum, "metr_la_s64_upgrades": m64_sum, "reddit_s24_upgrades": red_sum, "powergrid_selfmem": pg_sum}, indent=2))


if __name__ == "__main__":
    main()
