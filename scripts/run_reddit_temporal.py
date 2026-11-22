"""SNAP Temporal Reddit Hyperlink Network: Closed-Loop Social Cascade & Directed Graph Discovery.

Evaluates autonomous closed-loop dynamics on a real human temporal interaction network
(Kumar et al., WWW 2018; 571,927 timestamped directed subreddit-to-subreddit links, 2014-2017):
  - Top S = 24 interacting communities selected strictly on 2014-2015 training data.
  - Chronological split: Train 2014-2015 (118 episodes), Val H1 2016 (27), Test H2 2016-2017 (47).
  - Warm-starts internal memory over 12 observed days, then forecasts the next 12 days autonomously
    in closed loop (ZERO future inputs) across all 24 interacting communities.
  - Evaluates future cascade MSE/R^2 AND recovery of the true held-out 2016-2017 directed
    subreddit-to-subreddit hyperlink graph (143/552 directed pairs; AUROC & AP enrichment).
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
    fit_closed_loop_network,
)
from memgrn.cross_domain_datasets import load_reddit_temporal_network  # noqa: E402

OUT_DIR = ROOT / "results" / "reddit_temporal"
RAW_DIR = ROOT / "data" / "raw" / "reddit_temporal"

N_SUBS = 24
DT = 0.25
N_STEPS = 24
CLAMP_STEPS = 12
OBS_IDX = tuple(range(25))

MODELS: tuple[ClosedLoopModelFamily, ...] = (
    ClosedLoopModelFamily.MEMGRN,
    ClosedLoopModelFamily.LOCAL_CONST,
    ClosedLoopModelFamily.LTC_LOCAL,
    ClosedLoopModelFamily.SELECTIVE_SSM,
    ClosedLoopModelFamily.CFC,
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


def _run_model(fam_val: str) -> tuple[str, dict]:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    data = load_reddit_temporal_network(RAW_DIR, n_subreddits=N_SUBS, episode_days=24)
    n_g = N_SUBS

    def _pack(eps: np.ndarray):
        b = len(eps)
        ko = np.zeros((b, n_g), dtype=np.float64)
        y_obs = np.concatenate([np.zeros((b, 1, n_g)), eps], axis=1)
        w_obs = np.zeros_like(y_obs)
        w_obs[:, 13:, :] = 1.0  # score strictly on future 12 autonomous days
        y_cl = np.zeros((N_STEPS, b, n_g), dtype=np.float64)
        y_cl[:CLAMP_STEPS] = np.transpose(eps[:, :CLAMP_STEPS, :], (1, 0, 2))
        return ko, y_obs, w_obs, y_cl

    tr_ko, tr_y, tr_w, tr_cl = _pack(data.train_episodes)
    va_ko, va_y, va_w, va_cl = _pack(data.val_episodes)
    te_ko, te_y, te_w, te_cl = _pack(data.test_episodes)
    basal = np.ones(n_g, dtype=np.float64)

    if fam_val == "persistence":
        pred_fut = np.repeat(data.test_episodes[:, 11:12, :], 12, axis=1)
        n_params = 0
        graph = np.zeros((n_g, n_g), dtype=np.float64)
    else:
        fam = ClosedLoopModelFamily(fam_val)
        spec = ClosedLoopSpec(
            family=fam,
            n_genes=n_g,
            regulator_indices=tuple(range(n_g)),
            hollow=True,
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
            basal_levels=basal,
            train_ko=tr_ko,
            train_y=tr_y,
            train_weights=tr_w,
            train_y_clamp=tr_cl,
            val_ko=va_ko,
            val_y=va_y,
            val_weights=va_w,
            val_y_clamp=va_cl,
            dt=DT,
            n_steps=N_STEPS,
            obs_step_indices=OBS_IDX,
            clamp_steps=CLAMP_STEPS,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2, 5e-2),
            group_l21_scale=(0.01 if fam == ClosedLoopModelFamily.MEMGRN else 0.0),
            total_updates=650,
            lr_max=0.006,
        )
        pred_fut = fitted.predict_ko_fn(te_ko, None, None, te_cl)[:, 13:, :]
        n_params = fitted.n_params
        graph = fitted.interaction_matrix

    true_fut = data.test_episodes[:, 12:, :]
    ep_mse = np.mean((pred_fut - true_fut) ** 2, axis=(1, 2))
    test_mse = float(np.mean(ep_mse))
    test_var = float(np.mean(true_fut**2)) + 1e-8
    test_r2 = 1.0 - test_mse / test_var

    off = ~np.eye(n_g, dtype=bool)
    labels = data.test_adj_binary[off]
    base_rate = float(np.mean(labels))
    if np.max(graph[off]) > 1e-12:
        auroc_r, ap_r = _auroc_and_ap(graph[off], labels)
        g_norm = graph / np.maximum(1e-8, np.max(graph, axis=0, keepdims=True))
        auroc_n, ap_n = _auroc_and_ap(g_norm[off], labels)
    else:
        auroc_r, ap_r, auroc_n, ap_n = 0.5, base_rate, 0.5, base_rate

    return (
        fam_val,
        {
            "n_params": n_params,
            "test_mse": test_mse,
            "test_mse_se": float(np.std(ep_mse, ddof=1) / np.sqrt(len(ep_mse))),
            "test_r2": test_r2,
            "hyperlink_graph_auroc": auroc_r,
            "hyperlink_graph_ap": ap_r,
            "hyperlink_graph_ap_over_random": float(ap_r / max(1e-8, base_rate)),
            "hyperlink_colnorm_auroc": auroc_n,
            "per_ep_mse": ep_mse.tolist(),
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_reddit_temporal_network(RAW_DIR, n_subreddits=N_SUBS, episode_days=24)
    all_models = [*[m.value for m in MODELS], "persistence"]

    design = {
        "benchmark": "snap_reddit_temporal_community_cascades_s24",
        "source": "Kumar et al. (WWW 2018) SNAP soc-redditHyperlinks-title.tsv (571,927 links)",
        "n_subreddits": N_SUBS,
        "subreddits": list(data.subreddits),
        "n_train_eps": int(len(data.train_episodes)),
        "n_val_eps": int(len(data.val_episodes)),
        "n_test_eps": int(len(data.test_episodes)),
        "n_heldout_directed_edges": int(np.sum(data.test_adj_binary)),
        "models": all_models,
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=10, mp_context=ctx) as pool:
        raw_out = list(pool.map(_run_model, all_models))
    elapsed = time.time() - t0

    ep_map, summary = {}, {}
    for mk, res in raw_out:
        ep_map[mk] = np.asarray(res.pop("per_ep_mse"), dtype=np.float64)
        summary[mk] = res

    mem_v = ep_map["cl_memgrn"]
    wilc = {}
    for mk in all_models:
        if mk == "cl_memgrn":
            continue
        oth = ep_map[mk]
        p_v = float(wilcoxon(mem_v, oth, alternative="less").pvalue) if np.any(mem_v != oth) else 1.0
        wilc[mk] = {
            "pct_reduction": float(100.0 * (1.0 - np.mean(mem_v) / max(1e-12, np.mean(oth)))),
            "wins": int(np.sum(mem_v < oth)),
            "total": len(mem_v),
            "wilcoxon_p": p_v,
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "wilcoxon_vs_memgrn": wilc,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": round(elapsed, 1), "summary": summary, "wilcoxon": wilc}, indent=2))


if __name__ == "__main__":
    main()
