"""Closed-Loop Cyclic Feedback & Zero-Shot Combinatorial Double-Knockout Benchmark.

Directly stress-tests Theorem 1(ii) on fresh out-of-class 12-gene cyclic gene regulatory
networks containing genuine feedback motifs (3-gene repressilators, mutual repression toggles,
incoherent feed-forward loops, saturable chromatin buffering, and cooperative dual-TF enhancers).

Protocol (12 independent cyclic worlds):
  - Ground truth: 12 genes (8 regulators + 4 downstream targets) coupled by hollow cyclic
    regulatory networks with saturable Michaelis-Menten / chromatin-buffering memory and
    pairwise enhancer cooperativity.
  - Training: strictly SINGLE-gene knockouts (WT + 6 train single KOs, 2 val single KOs)
    observed on t in [0, 5.0] h.
  - Zero-shot test evaluation (never seen during training):
      1. Unseen single knockouts (2 held-out single KOs per world) on t in [0, 5.0] h.
      2. Zero-shot combinatorial DOUBLE knockouts (KO_i + KO_k, 12 pairs per world) on [0, 5.0] h.
      3. Out-of-horizon autonomous extrapolation on t in [5.0, 9.0] h (unseen single + double KOs).
      4. Direct causal graph recovery (AUROC & AP) on the true hollow cyclic adjacency matrix.
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
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    fit_closed_loop_network,
)

OUT_DIR = ROOT / "results" / "cyclic_double_ko"

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

N_GENES = 12
N_REGS = 8
DT = 0.125
N_STEPS_TR = 40  # t = 5.0 h
N_STEPS_EXT = 72  # t = 9.0 h
OBS_STRIDE = 4  # every 0.5 h -> 11 train knots (0..5h), 19 full knots (0..9h)


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


def _generate_cyclic_world(seed: int) -> dict:
    """Build an out-of-class 12-gene cyclic chromatin-buffered regulatory network."""
    rng = np.random.default_rng(seed)

    # True hollow cyclic graph among 8 regulators (0..7) -> 12 genes (0..11)
    # Contains:
    #   - 3-node repressilator cycle: 0 -> 1 -> 2 -> 0 (negative feedback loop)
    #   - Mutual repression toggle: 3 <-> 4 (bistable feedback)
    #   - Incoherent feed-forward + feedback cycle: 5 -> 6 -> 7 -> 5
    #   - Cross-module sparse directed edges + downstream targets 8..11
    w_direct = np.zeros((N_REGS, N_GENES), dtype=np.float64)
    w_chrom = np.zeros((N_REGS, N_GENES), dtype=np.float64)
    adj_mask = np.zeros((N_REGS, N_GENES), dtype=bool)

    def _add_edge(src: int, tgt: int, sign: float, d_amp: float, c_amp: float) -> None:
        if src == tgt:
            return
        adj_mask[src, tgt] = True
        w_direct[src, tgt] = sign * d_amp
        w_chrom[src, tgt] = sign * c_amp

    # Repressilator cycle 0 -| 1 -| 2 -| 0
    for u, v in ((0, 1), (1, 2), (2, 0)):
        _add_edge(u, v, -1.0, rng.uniform(0.20, 0.35), rng.uniform(0.95, 1.35))
    # Mutual toggle 3 -| 4, 4 -| 3 + activation from 0 -> 3, 2 -> 4
    _add_edge(3, 4, -1.0, rng.uniform(0.15, 0.30), rng.uniform(0.90, 1.25))
    _add_edge(4, 3, -1.0, rng.uniform(0.15, 0.30), rng.uniform(0.90, 1.25))
    _add_edge(0, 3, 1.0, rng.uniform(0.18, 0.32), rng.uniform(0.75, 1.10))
    _add_edge(2, 4, 1.0, rng.uniform(0.18, 0.32), rng.uniform(0.75, 1.10))
    # Feedback loop 5 -> 6 -> 7 -| 5 + cross-links
    _add_edge(5, 6, 1.0, rng.uniform(0.18, 0.32), rng.uniform(0.85, 1.25))
    _add_edge(6, 7, 1.0, rng.uniform(0.18, 0.32), rng.uniform(0.85, 1.25))
    _add_edge(7, 5, -1.0, rng.uniform(0.18, 0.32), rng.uniform(0.90, 1.30))
    _add_edge(1, 5, 1.0, rng.uniform(0.15, 0.28), rng.uniform(0.70, 1.05))
    _add_edge(3, 6, -1.0, rng.uniform(0.15, 0.28), rng.uniform(0.70, 1.05))
    _add_edge(4, 7, 1.0, rng.uniform(0.15, 0.28), rng.uniform(0.70, 1.05))

    # Sparse downstream regulatory edges into genes 8..11 (2-3 parents each)
    for tgt in range(N_REGS, N_GENES):
        parents = rng.choice(N_REGS, size=3, replace=False)
        for p in parents:
            sgn = float(rng.choice([-1.0, 1.0]))
            _add_edge(int(p), tgt, sgn, rng.uniform(0.15, 0.30), rng.uniform(0.85, 1.30))

    # Cooperative pairwise enhancer interactions (genuine non-additive double-KO epistasis!)
    coop_triples: list[tuple[int, int, int, float]] = []
    for tgt in range(N_GENES):
        cand_parents = np.where(adj_mask[:, tgt])[0]
        if len(cand_parents) >= 2:
            p1, p2 = int(cand_parents[0]), int(cand_parents[1])
            coop_triples.append((p1, p2, tgt, float(rng.choice([-1.0, 1.0]) * rng.uniform(0.65, 1.15))))
    # Also add cross-module cooperative edges into targets 8..11
    for tgt in range(N_REGS, N_GENES):
        p1, p2 = int(rng.integers(0, 4)), int(rng.integers(4, 8))
        coop_triples.append((p1, p2, tgt, float(rng.choice([-1.0, 1.0]) * rng.uniform(0.55, 0.95))))

    # Out-of-class saturable chromatin buffering + nonlinear Hill-like uptake/clearance per regulator
    k_on = rng.uniform(1.6, 3.2, size=N_REGS)
    k_off = rng.uniform(0.22, 0.65, size=N_REGS)  # strong chromatin-retention asymmetry!
    k_sat = rng.uniform(0.35, 0.75, size=N_REGS)
    k_clear = rng.uniform(0.35, 0.95, size=N_REGS)
    decay = rng.uniform(0.55, 0.95, size=N_GENES)
    basal = rng.uniform(0.85, 1.35, size=N_GENES)

    def _sim_ko(ko_vec: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        ko_reg = ko_vec[:N_REGS]

        def _rhs(_t: float, s: np.ndarray) -> np.ndarray:
            c1 = s[:N_REGS]
            c2 = s[N_REGS : 2 * N_REGS]
            y = s[2 * N_REGS :]
            g_reg = (1.0 - ko_reg) * y[:N_REGS] - ko_reg * basal[:N_REGS]
            act = np.tanh(g_reg)
            # Out-of-class two-stage saturable buffering ODE (Michaelis-Menten clearance!)
            up_rate = k_on / (1.0 + np.abs(c1) / k_sat)
            dn_rate = k_off / (1.0 + 1.8 * np.abs(c2))
            dc1 = np.where(act * (act - c1) >= 0.0, up_rate, dn_rate) * (act - c1)
            dc2 = (k_clear / (1.0 + 1.5 * np.abs(c2))) * (c1 - c2)
            prod = act @ w_direct + c2 @ w_chrom
            for p1, p2, tgt, gam in coop_triples:
                prod[tgt] += gam * (c2[p1] * c2[p2]) / (1.0 + 0.5 * abs(c2[p1] * c2[p2]))
            dy = prod - decay * y
            return np.concatenate([dc1, dc2, dy])

        sol = solve_ivp(
            _rhs,
            [float(t_eval[0]), float(t_eval[-1])],
            np.zeros(2 * N_REGS + N_GENES, dtype=np.float64),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        return sol.y[2 * N_REGS :].T  # (T, N_GENES)

    t_full = np.arange(0, N_STEPS_EXT + 1, OBS_STRIDE, dtype=np.float64) * DT  # 19 knots: 0..9h
    # Single knockouts: 0..7 (8 regulators) + WT
    # Hold out 2 unseen single KOs per world (rotated by seed) + 2 val single KOs + 4 train single KOs + 6 noisy replicates?
    # Better: use 12 single KOs across 8 regs with 2 partial/full KD strengths or 8 full single KOs + 4 partial KDs
    # Wait: even cleaner & 100% biologically standard:
    # Train on 6 single KOs (full KO) + 6 partial KDs (50% knockdown of those same 6 TFs -> 12 trajectories) + WT,
    # Validate on 2 held-out single KOs? No, to test UNSEEN single KOs we need 2 unseen single KOs strictly in test!
    # So split the 8 regulators into: 5 train regs (5 full KO + 5 partial 0.5 KD = 10), 1 val reg (+ 1 partial = 2),
    # 2 strictly unseen test regs (2 full single KOs), PLUS 12 strictly unseen combinatorial DOUBLE KOs (pairs of train+test regs)!
    perm = rng.permutation(N_REGS)
    tr_regs = perm[:5]
    va_regs = perm[5:6]
    te_single_regs = perm[6:8]

    def _make_ko_mask(indices: list[tuple[int, float]]) -> np.ndarray:
        m = np.zeros(N_GENES, dtype=np.float64)
        for idx, strength in indices:
            m[idx] = strength
        return m

    tr_kos = [np.zeros(N_GENES, dtype=np.float64)]
    for r in tr_regs:
        tr_kos.append(_make_ko_mask([(int(r), 1.0)]))
        tr_kos.append(_make_ko_mask([(int(r), 0.55)]))
    # Add 4 double KOs? NO! ZERO double KOs in training — train STRICTLY on single-gene perturbations!
    va_kos = [_make_ko_mask([(int(va_regs[0]), 1.0)]), _make_ko_mask([(int(va_regs[0]), 0.55)])]
    te_single_kos = [_make_ko_mask([(int(r), 1.0)]) for r in te_single_regs]

    # 12 zero-shot combinatorial double KOs (never seen during training!):
    # 8 pairs among training regulators (testing pure combinatorial epistasis!) + 4 pairs involving unseen regulators
    double_pairs: list[tuple[int, int]] = []
    for i in range(len(tr_regs)):
        for j in range(i + 1, len(tr_regs)):
            double_pairs.append((int(tr_regs[i]), int(tr_regs[j])))
    double_pairs = double_pairs[:8]
    for r_u in te_single_regs:
        double_pairs.append((int(r_u), int(tr_regs[0])))
        double_pairs.append((int(r_u), int(tr_regs[1])))
    te_double_kos = [_make_ko_mask([(p1, 1.0), (p2, 1.0)]) for p1, p2 in double_pairs]

    tr_y_clean = np.stack([_sim_ko(m, t_full[:11]) for m in tr_kos], axis=0)
    va_y_clean = np.stack([_sim_ko(m, t_full[:11]) for m in va_kos], axis=0)
    te_single_full = np.stack([_sim_ko(m, t_full) for m in te_single_kos], axis=0)
    te_double_full = np.stack([_sim_ko(m, t_full) for m in te_double_kos], axis=0)

    # Add realistic observation noise to train/val only; evaluate against clean test truth
    noise_sd = 0.015
    tr_y = tr_y_clean + rng.normal(0.0, noise_sd, size=tr_y_clean.shape)
    tr_y[:, 0, :] = 0.0
    va_y = va_y_clean + rng.normal(0.0, noise_sd, size=va_y_clean.shape)
    va_y[:, 0, :] = 0.0

    # Mask knocked-out genes from MSE evaluation (predicting non-edited downstream targets)
    def _eval_mask(kos: list[np.ndarray], n_t: int) -> np.ndarray:
        m = np.ones((len(kos), n_t, N_GENES), dtype=np.float64)
        m[:, 0, :] = 0.0
        for b_i, kv in enumerate(kos):
            m[b_i, :, kv > 0.0] = 0.0
        return m

    return {
        "seed": seed,
        "basal": basal,
        "adj_mask": adj_mask,
        "tr_ko": np.stack(tr_kos, axis=0),
        "tr_y": tr_y,
        "tr_w": _eval_mask(tr_kos, 11),
        "va_ko": np.stack(va_kos, axis=0),
        "va_y": va_y,
        "va_w": _eval_mask(va_kos, 11),
        "te_single_ko": np.stack(te_single_kos, axis=0),
        "te_single_y": te_single_full,
        "te_single_w": _eval_mask(te_single_kos, 19),
        "te_double_ko": np.stack(te_double_kos, axis=0),
        "te_double_y": te_double_full,
        "te_double_w": _eval_mask(te_double_kos, 19),
    }


def _run_one_world(seed: int) -> dict:
    world = _generate_cyclic_world(seed)
    reg_tuple = tuple(range(N_REGS))
    obs_tr = tuple(range(0, N_STEPS_TR + 1, OBS_STRIDE))
    obs_ext = tuple(range(0, N_STEPS_EXT + 1, OBS_STRIDE))

    res_models: dict[str, dict] = {}
    for fam in MODELS:
        spec = ClosedLoopSpec(
            family=fam,
            n_genes=N_GENES,
            regulator_indices=reg_tuple,
            hollow=True,
            use_cooperativity=(fam == ClosedLoopModelFamily.MEMGRN),
            coop_rank=4,
            n_channels=3,
            hidden_width=5,
            module_dim=14,
            latent_dim=18,
            rate_min=0.05,
            rate_max=8.0,
        )
        use_grp = 0.12 if fam in (ClosedLoopModelFamily.MEMGRN, ClosedLoopModelFamily.LOCAL_CONST) else 0.0
        fitted = fit_closed_loop_network(
            spec=spec,
            basal_levels=world["basal"],
            train_ko=world["tr_ko"],
            train_y=world["tr_y"],
            train_weights=world["tr_w"],
            val_ko=world["va_ko"],
            val_y=world["va_y"],
            val_weights=world["va_w"],
            dt=DT,
            n_steps=N_STEPS_TR,
            obs_step_indices=obs_tr,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 5e-3),
            group_l21_scale=use_grp,
            total_updates=900,
            lr_max=0.008,
            gate_shrinkage=1.5,
            hysteresis_shrinkage=1.5,
        )

        # Build extended-horizon rollout [0, 9.0h] with the exact learned parameters
        from memgrn.baselines import build_closed_loop_jax_rollout
        import jax.numpy as jnp

        ext_fwd, _, _ = build_closed_loop_jax_rollout(
            spec,
            basal_levels=world["basal"],
            dt=DT,
            n_steps=N_STEPS_EXT,
            obs_step_indices=obs_ext,
        )
        th_j = jnp.asarray(fitted.theta, dtype=jnp.float32)

        def _pred_ext(ko_arr: np.ndarray) -> np.ndarray:
            y0 = jnp.zeros((ko_arr.shape[0], N_GENES), dtype=jnp.float32)
            return np.asarray(ext_fwd(th_j, jnp.asarray(ko_arr, dtype=jnp.float32), y0), dtype=np.float64)

        p_single = _pred_ext(world["te_single_ko"])
        p_double = _pred_ext(world["te_double_ko"])

        def _masked_mse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
            return float(np.sum(((pred - true) ** 2) * mask) / np.maximum(1.0, np.sum(mask)))

        single_mse = _masked_mse(p_single[:, 1:11, :], world["te_single_y"][:, 1:11, :], world["te_single_w"][:, 1:11, :])
        double_mse = _masked_mse(p_double[:, 1:11, :], world["te_double_y"][:, 1:11, :], world["te_double_w"][:, 1:11, :])
        horizon_mse = _masked_mse(
            np.concatenate([p_single[:, 11:, :], p_double[:, 11:, :]], axis=0),
            np.concatenate([world["te_single_y"][:, 11:, :], world["te_double_y"][:, 11:, :]], axis=0),
            np.concatenate([world["te_single_w"][:, 11:, :], world["te_double_w"][:, 11:, :]], axis=0),
        )

        # Causal graph discovery on off-diagonal regulator->target pairs
        g_mat = fitted.interaction_matrix[:N_REGS, :]
        off_mask = np.ones((N_REGS, N_GENES), dtype=bool)
        for i in range(N_REGS):
            off_mask[i, i] = False
        auroc, ap = _auroc_and_ap(g_mat[off_mask], world["adj_mask"][off_mask])

        res_models[fam.value] = {
            "n_params": fitted.n_params,
            "unseen_single_ko_mse": single_mse,
            "double_ko_mse": double_mse,
            "horizon_extrap_mse": horizon_mse,
            "graph_auroc": auroc,
            "graph_ap": ap,
        }
    return {"seed": seed, "models": res_models}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(7001, 7013))  # 12 fresh cyclic worlds
    design = {
        "benchmark": "closed_loop_cyclic_feedback_and_zero_shot_double_ko",
        "n_genes": N_GENES,
        "n_regulators": N_REGS,
        "seeds": seeds,
        "train_horizon_h": N_STEPS_TR * DT,
        "extrap_horizon_h": N_STEPS_EXT * DT,
        "models": [m.value for m in MODELS],
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as pool:
        worlds = list(pool.map(_run_one_world, seeds))
    elapsed = time.time() - t0

    summary: dict[str, dict] = {}
    for fam in MODELS:
        mk = fam.value
        entry: dict[str, float | int] = {"n_params": worlds[0]["models"][mk]["n_params"]}
        for k_m in (
            "unseen_single_ko_mse",
            "double_ko_mse",
            "horizon_extrap_mse",
            "graph_auroc",
            "graph_ap",
        ):
            arr = np.array([w["models"][mk][k_m] for w in worlds], dtype=np.float64)
            entry[k_m] = float(np.mean(arr))
            entry[f"{k_m}_se"] = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        summary[mk] = entry

    # Paired Wilcoxon tests vs cl_memgrn on double_ko_mse
    wilcoxon_vs_memgrn: dict[str, dict] = {}
    mem_d = np.array([w["models"]["cl_memgrn"]["double_ko_mse"] for w in worlds])
    for fam in MODELS:
        mk = fam.value
        if mk == "cl_memgrn":
            continue
        oth_d = np.array([w["models"][mk]["double_ko_mse"] for w in worlds])
        stat = wilcoxon(mem_d, oth_d, alternative="less")
        wilcoxon_vs_memgrn[mk] = {
            "pct_reduction": float(100.0 * (1.0 - np.mean(mem_d) / np.mean(oth_d))),
            "wins": int(np.sum(mem_d < oth_d)),
            "n": len(worlds),
            "wilcoxon_p": float(stat.pvalue),
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "wilcoxon_double_ko_vs_memgrn": wilcoxon_vs_memgrn,
        "worlds": worlds,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": round(elapsed, 1), "summary": summary, "wilcoxon": wilcoxon_vs_memgrn}, indent=2))


if __name__ == "__main__":
    main()
