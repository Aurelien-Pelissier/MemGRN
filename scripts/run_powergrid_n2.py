"""IEEE 39-Bus New England Power Grid: Closed-Loop N-1 -> N-2 Combinatorial Contingencies.

Non-biological physical infrastructure test of Theorem 1(i)-(ii):
  - Ground truth: 12 operating-point worlds on the exact published IEEE 39-Bus New England Test
    System (10 synchronous generators, 39 buses, 46 transmission lines, exact Kron-reduced
    electrical coupling matrix B_Kron) with saturable asymmetric turbine-governor / thermal
    hysteresis and pairwise nonlinear corridor overload.
  - Training: strictly WT + SINGLE-generator N-1 trips (ko=1.0) & partial deratings (ko=0.55)
    over [0, 5] s (ZERO double outages seen during training!).
  - Zero-shot test evaluation:
      1. Unseen single N-1 contingencies on [0, 5] s;
      2. Zero-shot combinatorial N-2 double contingencies (12 unseen pairs per world = 144 pairs);
      3. Out-of-horizon [5, 9] s post-contingency recovery extrapolation;
      4. Physical Kron-reduced electrical corridor graph recovery (AUROC & AP).
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
from memgrn.powergrid import generate_ieee39_powergrid_world  # noqa: E402

OUT_DIR = ROOT / "results" / "powergrid_n2"

MODELS: tuple[ClosedLoopModelFamily, ...] = (
    ClosedLoopModelFamily.MEMGRN,
    ClosedLoopModelFamily.LOCAL_CONST,
    ClosedLoopModelFamily.LTC_LOCAL,
    ClosedLoopModelFamily.COUPLED_ADAPTIVE,
    ClosedLoopModelFamily.LTC_COUPLED,
    ClosedLoopModelFamily.SELECTIVE_SSM,
    ClosedLoopModelFamily.CFC,
    ClosedLoopModelFamily.PERTURB_ODE,
    ClosedLoopModelFamily.LATENT_ODE,
)

N_NODES = 10
DT = 0.05
N_STEPS_TR = 100  # 5.0 s
N_STEPS_EXT = 180  # 9.0 s
OBS_TR = tuple(range(0, N_STEPS_TR + 1, 5))
OBS_EXT = tuple(range(0, N_STEPS_EXT + 1, 5))
WORLD_SEEDS = tuple(range(700, 712))  # 12 operating-point worlds


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


def _run_world_model(job: tuple[int, str]) -> tuple[int, str, dict]:
    seed, fam_val = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    w = generate_ieee39_powergrid_world(seed)
    fam = ClosedLoopModelFamily(fam_val)
    spec = ClosedLoopSpec(
        family=fam,
        n_genes=N_NODES,
        regulator_indices=tuple(range(N_NODES)),
        hollow=True,
        use_cooperativity=(fam == ClosedLoopModelFamily.MEMGRN),
        coop_rank=6,
        n_channels=3,
        hidden_width=5,
        module_dim=20,
        latent_dim=24,
        base_rates=(0.3, 1.0, 3.0),
    )

    fitted = fit_closed_loop_network(
        spec=spec,
        basal_levels=w.basal_load,
        train_ko=w.train_ko,
        train_y=w.train_y,
        train_weights=w.train_weights,
        val_ko=w.val_ko,
        val_y=w.val_y,
        val_weights=w.val_weights,
        dt=DT,
        n_steps=N_STEPS_TR,
        obs_step_indices=OBS_TR,
        seeds=(101, 202),
        reg_grid=(1e-4, 1e-3, 5e-3, 2e-2),
        l1_graph_ratio=0.25,
        group_l21_scale=(0.01 if fam == ClosedLoopModelFamily.MEMGRN else 0.0),
        total_updates=800,
        lr_max=0.006,
    )

    # Mask tripped generators from downstream MSE evaluation
    def _eval_mask(kos: np.ndarray, n_t: int) -> np.ndarray:
        m = np.ones((len(kos), n_t, N_NODES), dtype=np.float64)
        m[:, 0, :] = 0.0
        for b_i, kv in enumerate(kos):
            m[b_i, :, kv > 0.0] = 0.0
        return m

    pred_1 = fitted.predict_ko_fn(w.test_single_ko, None, None, None)
    m1 = _eval_mask(w.test_single_ko, len(OBS_TR))
    mse_1 = float(np.sum(((pred_1 - w.test_single_y) ** 2) * m1) / np.sum(m1))

    pred_2 = fitted.predict_ko_fn(w.test_double_ko, None, None, None)
    m2 = _eval_mask(w.test_double_ko, len(OBS_TR))
    mse_2 = float(np.sum(((pred_2 - w.test_double_y) ** 2) * m2) / np.sum(m2))

    # Out-of-horizon [5, 9] s extrapolation on N-2 double contingencies
    from memgrn.baselines import build_closed_loop_jax_rollout
    import jax.numpy as jnp

    fwd_ext, _, _ = build_closed_loop_jax_rollout(
        spec, basal_levels=w.basal_load, dt=DT, n_steps=N_STEPS_EXT, obs_step_indices=OBS_EXT
    )
    th_j = jnp.asarray(fitted.theta, dtype=jnp.float32)
    y0_2 = jnp.zeros((len(w.test_double_ko), N_NODES), dtype=jnp.float32)
    pred_2_ext = np.asarray(
        fwd_ext(th_j, jnp.asarray(w.test_double_ko, dtype=jnp.float32), y0_2, None, None),
        dtype=np.float64,
    )
    m2e = _eval_mask(w.test_double_ko, len(OBS_EXT))
    m2e[:, : len(OBS_TR), :] = 0.0  # strictly score [5, 9] s extrapolation tail
    mse_ext = float(np.sum(((pred_2_ext - w.test_double_y_extrap) ** 2) * m2e) / np.sum(m2e))

    off = ~np.eye(N_NODES, dtype=bool)
    graph = fitted.interaction_matrix
    if np.max(graph[off]) > 1e-12:
        auroc, ap = _auroc_and_ap(graph[off], w.adj_binary[off])
    else:
        auroc, ap = 0.5, float(np.mean(w.adj_binary[off]))

    return (
        seed,
        fam_val,
        {
            "n_params": fitted.n_params,
            "single_n1_mse": mse_1,
            "double_n2_mse": mse_2,
            "horizon_extrap_mse": mse_ext,
            "graph_auroc": auroc,
            "graph_ap": ap,
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    design = {
        "benchmark": "ieee39_powergrid_n1_to_n2_closed_loop",
        "system": "IEEE 39-Bus New England Test System (10 generators, 39 buses, 46 lines)",
        "world_seeds": list(WORLD_SEEDS),
        "models": [m.value for m in MODELS],
        "n_nodes": N_NODES,
        "dt": DT,
        "train_horizon_s": 5.0,
        "extrap_horizon_s": 9.0,
        "double_contingencies_in_training": 0,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    jobs = [(s, m.value) for s in WORLD_SEEDS for m in MODELS]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=24, mp_context=ctx) as pool:
        results = list(pool.map(_run_world_model, jobs))
    elapsed = time.time() - t0

    by_model: dict[str, list[dict]] = {m.value: [] for m in MODELS}
    for _, mk, res in results:
        by_model[mk].append(res)

    n_w = len(WORLD_SEEDS)
    summary: dict[str, dict] = {}
    for mk, rows in by_model.items():
        summary[mk] = {
            "n_params": rows[0]["n_params"],
            "single_n1_mse": float(np.mean([r["single_n1_mse"] for r in rows])),
            "single_n1_mse_se": float(np.std([r["single_n1_mse"] for r in rows], ddof=1) / np.sqrt(n_w)),
            "double_n2_mse": float(np.mean([r["double_n2_mse"] for r in rows])),
            "double_n2_mse_se": float(np.std([r["double_n2_mse"] for r in rows], ddof=1) / np.sqrt(n_w)),
            "horizon_extrap_mse": float(np.mean([r["horizon_extrap_mse"] for r in rows])),
            "horizon_extrap_mse_se": float(np.std([r["horizon_extrap_mse"] for r in rows], ddof=1) / np.sqrt(n_w)),
            "graph_auroc": float(np.mean([r["graph_auroc"] for r in rows])),
            "graph_ap": float(np.mean([r["graph_ap"] for r in rows])),
        }

    mem_n2 = np.array([r["double_n2_mse"] for r in by_model["cl_memgrn"]])
    wilcoxon_n2: dict[str, dict] = {}
    for mk, rows in by_model.items():
        if mk == "cl_memgrn":
            continue
        oth = np.array([r["double_n2_mse"] for r in rows])
        p_val = float(wilcoxon(mem_n2, oth, alternative="less").pvalue)
        wilcoxon_n2[mk] = {
            "pct_reduction": float(100.0 * (1.0 - np.mean(mem_n2) / max(1e-12, np.mean(oth)))),
            "wins_12": int(np.sum(mem_n2 < oth)),
            "wilcoxon_p": p_val,
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "wilcoxon_n2_vs_memgrn": wilcoxon_n2,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": round(elapsed, 1), "summary": summary, "wilcoxon": wilcoxon_n2}, indent=2))


if __name__ == "__main__":
    main()
