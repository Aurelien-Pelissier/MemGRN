"""METR-LA Real Highway Traffic Sensor Network: Closed-Loop Congestion & Road-Graph Discovery.

Evaluates autonomous closed-loop continuous-time dynamics at real urban infrastructure scale
(S = 64 connected Los Angeles highway loop detectors, 622 physical directed road edges from
Li et al., ICLR 2018 adj_mx.pkl, 7.09M raw observations, strict official Mar-May/May-Jun/Jun
chronological split):
  - Warm-starts internal memory over 1 hour of observed traffic speeds (12 x 5-min steps),
    then rolls out autonomously in closed loop for 1 full hour (12 x 5-min steps = 15m, 30m,
    and 60m horizons, ZERO future sensor inputs, missing speed == 0 masked strictly per DCRNN).
  - Task 1: All-test 15-min, 30-min, and 60-min closed-loop MSE & MAE (mph) vs. Persistence
    and all 9 closed-loop architectures (MemGRN, Local Constant, Source-Local LTC, Selective
    SSM/Mamba, CfC, Coupled Adaptive, Coupled LTC, PerturbODE, Latent ODE).
  - Task 2: Localized Congestion-Shockwave episodes (93/160 test windows where isolated jams
    break normal corridor-wide rush-hour correlation).
  - Task 3: Physical Directed Highway Road-Graph Recovery (AUROC & AP enrichment against the
    622 ground-truth highway edges in adj_mx.pkl) + real-world O(SR) vs. O(S^2 R^2) scaling!
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
from memgrn.cross_domain_datasets import load_metr_la_traffic  # noqa: E402

OUT_DIR = ROOT / "results" / "metr_la_traffic"
RAW_DIR = ROOT / "data" / "raw" / "metr_la"

N_SENSORS = 64
DT = 0.25  # 0.25 -> 1 step per 5-min tick (in 20-min units): 12 clamp steps + 12 autonomous steps = 24
N_STEPS = 24
CLAMP_STEPS = 12
OBS_IDX = tuple(range(25))  # 0 (zero init) + 1..12 (hist) + 13..24 (future 5m..60m)

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

    data = load_metr_la_traffic(RAW_DIR, n_sensors=N_SENSORS)
    n_g = N_SENSORS

    def _pack(eps: np.ndarray, msk: np.ndarray):
        b = len(eps)
        ko = np.zeros((b, n_g), dtype=np.float64)
        y_obs = np.concatenate([np.zeros((b, 1, n_g)), eps], axis=1)  # (B, 25, S)
        w_obs = np.zeros_like(y_obs)
        w_obs[:, 13:, :] = msk[:, 12:, :]  # score strictly on 12 future autonomous 5-min steps
        y_cl = np.zeros((N_STEPS, b, n_g), dtype=np.float64)
        y_cl[:CLAMP_STEPS] = np.transpose(eps[:, :CLAMP_STEPS, :], (1, 0, 2))
        return ko, y_obs, w_obs, y_cl

    tr_ko, tr_y, tr_w, tr_cl = _pack(data.train_episodes, data.train_masks)
    va_ko, va_y, va_w, va_cl = _pack(data.val_episodes, data.val_masks)
    te_ko, te_y, te_w, te_cl = _pack(data.test_episodes, data.test_masks)
    basal = np.ones(n_g, dtype=np.float64)

    t0 = time.perf_counter()
    if fam_val == "persistence":
        pred_fut = np.repeat(data.test_episodes[:, 11:12, :], 12, axis=1)
        n_params, mem_params = 0, 0
        graph = np.zeros((n_g, n_g), dtype=np.float64)
    else:
        fam = ClosedLoopModelFamily(fam_val)
        spec = ClosedLoopSpec(
            family=fam,
            n_genes=n_g,
            regulator_indices=tuple(range(n_g)),
            hollow=True,
            use_cooperativity=(fam == ClosedLoopModelFamily.MEMGRN),
            coop_rank=8,
            readout_rank=(
                20
                if fam in (ClosedLoopModelFamily.MEMGRN, ClosedLoopModelFamily.LOCAL_CONST)
                else None
            ),
            n_channels=3,
            hidden_width=5,
            module_dim=32,
            latent_dim=32,
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
            reg_grid=(1e-4, 1e-3, 1e-2),
            group_l21_scale=(0.005 if fam == ClosedLoopModelFamily.MEMGRN else 0.0),
            total_updates=550,
            lr_max=0.006,
        )
        pred_all = fitted.predict_ko_fn(te_ko, None, None, te_cl)
        pred_fut = pred_all[:, 13:, :]
        n_params = fitted.n_params
        tree0 = _init_closed_loop_tree(spec, seed=0)
        readout_keys = {
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
        }
        mem_params = int(sum(int(v.size) for k, v in tree0.items() if k not in readout_keys))
        graph = fitted.interaction_matrix
    runtime_s = time.perf_counter() - t0

    true_fut = data.test_episodes[:, 12:, :]
    msk_fut = data.test_masks[:, 12:, :]
    err2 = ((pred_fut - true_fut) ** 2) * msk_fut
    abse = np.abs(pred_fut - true_fut) * msk_fut * 20.0  # convert normalized congestion back to mph!

    ep_mse = np.sum(err2, axis=(1, 2)) / np.maximum(1.0, np.sum(msk_fut, axis=(1, 2)))
    ep_mae_mph = np.sum(abse, axis=(1, 2)) / np.maximum(1.0, np.sum(msk_fut, axis=(1, 2)))
    shock_m = data.test_shock_flags

    def _hor_mae(h_idx: int) -> float:
        return float(
            np.sum(abse[:, h_idx, :]) / max(1.0, np.sum(msk_fut[:, h_idx, :]))
        )

    off = ~np.eye(n_g, dtype=bool)
    labels = data.adj_binary[off]
    base_rate = float(np.mean(labels))
    if np.max(graph[off]) > 1e-12:
        g_norm = graph / np.maximum(1e-8, np.max(graph, axis=0, keepdims=True))
        auroc_n, ap_n = _auroc_and_ap(g_norm[off], labels)
        auroc_r, ap_r = _auroc_and_ap(graph[off], labels)
    else:
        auroc_n, ap_n, auroc_r, ap_r = 0.5, base_rate, 0.5, base_rate

    return (
        fam_val,
        {
            "n_params": n_params,
            "mem_params": mem_params,
            "runtime_seconds": runtime_s,
            "test_mse": float(np.mean(ep_mse)),
            "test_mse_se": float(np.std(ep_mse, ddof=1) / np.sqrt(len(ep_mse))),
            "test_mae_mph": float(np.mean(ep_mae_mph)),
            "mae_15min_mph": _hor_mae(2),
            "mae_30min_mph": _hor_mae(5),
            "mae_60min_mph": _hor_mae(11),
            "shockwave_mse": float(np.mean(ep_mse[shock_m])),
            "shockwave_mse_se": float(
                np.std(ep_mse[shock_m], ddof=1) / np.sqrt(np.sum(shock_m))
            ),
            "shockwave_mae_mph": float(np.mean(ep_mae_mph[shock_m])),
            "road_graph_auroc": auroc_r,
            "road_graph_ap": ap_r,
            "road_graph_ap_over_random": float(ap_r / max(1e-8, base_rate)),
            "road_graph_colnorm_auroc": auroc_n,
            "road_graph_colnorm_ap_over_random": float(ap_n / max(1e-8, base_rate)),
            "per_ep_mse": ep_mse.tolist(),
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_metr_la_traffic(RAW_DIR, n_sensors=N_SENSORS)
    all_models = [*[m.value for m in MODELS], "persistence"]

    design = {
        "benchmark": "metr_la_closed_loop_traffic_s64",
        "source": "Li et al. (ICLR 2018) METR-LA Highway Loop Detectors + adj_mx.pkl",
        "n_sensors": N_SENSORS,
        "n_physical_road_edges": int(np.sum(data.adj_binary)),
        "n_train_eps": int(len(data.train_episodes)),
        "n_val_eps": int(len(data.val_episodes)),
        "n_test_eps": int(len(data.test_episodes)),
        "n_test_shockwaves": int(np.sum(data.test_shock_flags)),
        "total_raw_observations": data.total_raw_observations,
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

    ep_map: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    for mk, res in raw_out:
        ep_map[mk] = np.asarray(res.pop("per_ep_mse"), dtype=np.float64)
        summary[mk] = res

    mem_v = ep_map["cl_memgrn"]
    shock_m = data.test_shock_flags
    wilcoxon_all: dict[str, dict] = {}
    for mk in all_models:
        if mk == "cl_memgrn":
            continue
        oth = ep_map[mk]
        p_all = float(wilcoxon(mem_v, oth, alternative="less").pvalue) if np.any(mem_v != oth) else 1.0
        p_shk = (
            float(wilcoxon(mem_v[shock_m], oth[shock_m], alternative="less").pvalue)
            if np.any(mem_v[shock_m] != oth[shock_m])
            else 1.0
        )
        wilcoxon_all[mk] = {
            "pct_reduction_all": float(100.0 * (1.0 - np.mean(mem_v) / max(1e-12, np.mean(oth)))),
            "wins_all": int(np.sum(mem_v < oth)),
            "wilcoxon_p_all": p_all,
            "pct_reduction_shock": float(
                100.0 * (1.0 - np.mean(mem_v[shock_m]) / max(1e-12, np.mean(oth[shock_m])))
            ),
            "wilcoxon_p_shock": p_shk,
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "wilcoxon_vs_memgrn": wilcoxon_all,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": round(elapsed, 1), "summary": summary, "wilcoxon": wilcoxon_all}, indent=2))


if __name__ == "__main__":
    main()
