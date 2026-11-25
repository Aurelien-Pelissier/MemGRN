"""Biological Rate Phase-Portrait Interpretability & Genome-Scale S=8..512 Scaling Benchmark.

Produces two publication-grade mechanistic & engineering audits:
  1. Part A (Yeast Msn2 Learned Chromatin-Memory Phase Portraits):
     Trains unified decoupled-rate MemGRN per native yeast promoter (ALD3, DCS2, DDR2, HXK1,
     RTN2, SIP18, TKL2) on Hansen & Zechner (2021) single pulses and extracts the exact learned
     state-dependent rise/recovery rate landscapes lambda_up(a, h) and lambda_down(a, h),
     comparing the learned memory-retention slowdown ratio against Hansen & O'Shea's (2013)
     measured promoter chromatin-remodeling classes (Fast nucleosome-free: DCS2, HXK1 vs.
     Slow nucleosome-remodeled: SIP18, ALD3, DDR2, TKL2, RTN2).
  2. Part B (Genome-Scale S = 8 -> 512 TF / N = 32 -> 2048 Gene Scaling Benchmark):
     Measures compiled JAX RK4 forward+grad wall-clock latency (ms/step) and memory-module
     parameter count across S in {8, 16, 32, 64, 128, 256, 512} regulators comparing:
       - MemGRN (O(SR) source-isolated memory + low-rank/sparse enhancer cooperativity)
       - Coupled Adaptive (O(S^2 R^2) coupled memory state mixing)
       - Coupled LTC (O(S^2 R^2) dense liquid time-constant dynamics)
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    DynamicalModelFamily,
    ModelSpec,
    build_closed_loop_jax_rollout,
    build_jax_rollout,
    extract_memgrn_up_down,
    fit_dynamical_model,
)
from memgrn.datasets import (  # noqa: E402
    MSN2_PROMOTERS,
    build_rk4_stage_grid,
    load_archived_json,
    load_msn2_population_trace,
)

OUT_DIR = ROOT / "results" / "interpretability_and_scaling"
RAW_MSN2 = ROOT / "data" / "raw" / "msn2"

# Hansen & O'Shea (2013) Molecular Systems Biology chromatin-remodeling kinetics classification
PROMOTER_CHROMATIN_CLASS = {
    "DCS2": "Fast (low remodeling)",
    "HXK1": "Fast (low remodeling)",
    "RTN2": "Intermediate / fast-switching",
    "DDR2": "Slow (chromatin-remodeled)",
    "TKL2": "Slow (chromatin-remodeled)",
    "ALD3": "Slow (chromatin-remodeled)",
    "SIP18": "Slow (high-threshold remodeled)",
}


def run_yeast_rate_portraits() -> dict:
    """Extract learned asymmetric (lambda_up, lambda_down) phase portraits across 7 yeast promoters."""
    stage3 = ROOT / "results" / "stage3"
    rep_design = load_archived_json(stage3, "msn2_repeated_pulses/design.json")
    schedules = rep_design["schedules"]
    last_off = [s["n"] * 5.0 + (s["n"] - 1) * s["gap"] for s in schedules]

    with np.load(stage3 / "neural_rate_msn2_transfer" / "training.npz") as tr_f:
        train_y_obs = np.asarray(tr_f["y"], dtype=np.float64)  # (16, 24, 7) at 5..120 min
        scales = np.asarray(tr_f["scales"], dtype=np.float64)
        tf_single = np.asarray(tr_f["tf"], dtype=np.float64)  # (17, 49) at 0..120 min
        obs_min = np.asarray(tr_f["minutes"], dtype=np.float64)

    with np.load(stage3 / "msn2_repeated_pulses" / "predictions.npz") as rep_f:
        rep_drivers = np.asarray(rep_f["drivers"], dtype=np.float64)  # (10, 49)

    rep_truth = (
        np.stack(
            [
                np.stack(
                    [
                        load_msn2_population_trace(
                            RAW_MSN2 / f"{g}_{s['id']}_size.mat", "YFP_molecules", obs_min
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
    )  # (10, 24, 7)

    substeps = 4
    dt_hours = 2.5 / (60.0 * substeps)
    obs_stride = 2 * substeps  # 5.0 min
    single_stage = build_rk4_stage_grid(tf_single[:16], substeps)  # (192, 16, 1, 3)
    rep_stage = build_rk4_stage_grid(rep_drivers, substeps)  # (192, 10, 1, 3)

    single_y_full = np.concatenate(
        [np.zeros((16, 1, 7), dtype=np.float64), train_y_obs], axis=1
    )  # (16, 25, 7)
    val_idx = [1, 6, 10, 15]
    tr_idx = [i for i in range(16) if i not in val_idx]
    train_u = single_stage[:, tr_idx]
    val_u = single_stage[:, val_idx]

    h_grid = np.linspace(0.0, 0.9, 19)
    portraits: dict[str, dict] = {}

    for g_i, gname in enumerate(MSN2_PROMOTERS):
        tr_y = single_y_full[tr_idx, :, g_i : g_i + 1]
        va_y = single_y_full[val_idx, :, g_i : g_i + 1]
        tr_m = np.ones_like(tr_y)
        tr_m[:, 0, :] = 0.0
        va_m = np.ones_like(va_y)
        va_m[:, 0, :] = 0.0

        spec = ModelSpec(
            family=DynamicalModelFamily.MEMGRN,
            n_sources=1,
            n_targets=1,
            n_channels=3,
            hidden_width=4,
            rate_min=0.05,
            rate_max=14.0,
        )
        fitted = fit_dynamical_model(
            spec=spec,
            train_stage_u=train_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=val_u,
            val_y=va_y,
            val_mask=va_m,
            dt=dt_hours,
            obs_stride=obs_stride,
            seeds=(101, 202),
            reg_grid=(1e-5, 1e-4, 1e-3),
            total_updates=1200,
            lr_max=0.012,
        )
        _, unpack, _ = build_jax_rollout(spec, dt=dt_hours, obs_stride=obs_stride)
        p = {k: np.asarray(v, dtype=np.float64) for k, v in unpack(jnp.asarray(fitted.theta)).items()}

        # Compute weighted channel-averaged effective loading (a=0.8) & washout clearance (a=0.0) rates vs h
        b_abs = np.abs(p["delayed"][0, :, 0])  # (3,)
        ch_w = b_abs / (np.sum(b_abs) + 1e-12)

        def _eval_rates(a_val: float) -> tuple[np.ndarray, np.ndarray]:
            r_up, r_dn, v_up, v_dn = extract_memgrn_up_down(p)
            lam_up_curve = np.zeros_like(h_grid)
            lam_dn_curve = np.zeros_like(h_grid)
            for idx_h, hv in enumerate(h_grid):
                zv = 0.5 * (a_val + hv)
                hid = np.tanh(
                    p["bias"]
                    + a_val * p["in_proj"][0]
                    + zv * p["in_proj"][1]
                    + hv * p["in_proj"][2]
                )
                l_up = spec.rate_min + (spec.rate_max - spec.rate_min) / (
                    1.0 + np.exp(-(r_up[0] + v_up[0] @ hid))
                )
                l_dn = spec.rate_min + (spec.rate_max - spec.rate_min) / (
                    1.0 + np.exp(-(r_dn[0] + v_dn[0] @ hid))
                )
                lam_up_curve[idx_h] = float(ch_w @ l_up)
                lam_dn_curve[idx_h] = float(ch_w @ l_dn)
            return lam_up_curve, lam_dn_curve

        lam_up_on, _ = _eval_rates(0.8)
        _, lam_dn_off = _eval_rates(0.0)
        pred_te = fitted.predict_fn(rep_stage, None)[:, 1:, 0]  # (10, 24)
        truth_g = rep_truth[:, :, g_i]
        post_mse = float(
            np.mean(
                [
                    np.mean((pred_te[i, obs_min > stop] - truth_g[i, obs_min > stop]) ** 2)
                    for i, stop in enumerate(last_off)
                ]
            )
        )

        slowdown_ratio = float(lam_dn_off[0] / max(1e-6, lam_dn_off[14]))
        asym_ratio = float(np.mean(lam_up_on) / max(1e-6, np.mean(lam_dn_off)))
        effective_tau_washout_min = float(60.0 / max(1e-6, float(np.mean(lam_dn_off[6:15]))))

        portraits[gname] = {
            "chromatin_class": PROMOTER_CHROMATIN_CLASS[gname],
            "ten_train_post_mse": post_mse,
            "channel_weights": ch_w.tolist(),
            "h_grid": h_grid.tolist(),
            "lam_up_on_h": lam_up_on.tolist(),
            "lam_dn_off_h": lam_dn_off.tolist(),
            "washout_slowdown_ratio": slowdown_ratio,
            "asymmetry_rise_over_recovery": asym_ratio,
            "effective_washout_tau_min": effective_tau_washout_min,
        }
    return portraits


def run_genome_scale_benchmark() -> list[dict]:
    """Benchmark compiled JAX forward+grad latency and memory-module parameter scaling S=8..512."""
    s_values = (8, 16, 32, 64, 128, 256, 512)
    batch = 4
    n_steps = 20
    dt = 0.25
    obs_idx = (0, 5, 10, 15, 20)
    rows = []

    for s_tf in s_values:
        n_genes = 4 * s_tf  # up to 2,048 genes at S=512 TFs!
        reg_idx = tuple(range(s_tf))
        basal = np.ones(n_genes, dtype=np.float32)
        rng_sc = np.random.default_rng(1000 + s_tf)
        ko_np = np.zeros((batch, n_genes), dtype=np.float32)
        for b_i in range(batch):
            ko_np[b_i, b_i % s_tf] = 1.0
        ko = jnp.asarray(ko_np)
        y0 = jnp.asarray(rng_sc.normal(0.0, 0.25, size=(batch, n_genes)).astype(np.float32))
        target = jnp.asarray(
            rng_sc.normal(0.0, 0.25, size=(batch, len(obs_idx), n_genes)).astype(np.float32)
        )

        row = {"n_tfs": s_tf, "n_genes": n_genes}

        # Measure MemGRN (O(SR) memory + low-rank cooperative Gamma), Coupled Adaptive (O(S^2 R^2)),
        # and Coupled LTC (O(S^2 R^2) with matched hidden memory state M = S*R)
        configs = [
            (
                "memgrn",
                ClosedLoopSpec(
                    family=ClosedLoopModelFamily.MEMGRN,
                    n_genes=n_genes,
                    regulator_indices=reg_idx,
                    use_cooperativity=True,
                    coop_rank=8,
                    n_channels=3,
                    hidden_width=4,
                ),
            ),
            (
                "coupled_adaptive",
                ClosedLoopSpec(
                    family=ClosedLoopModelFamily.COUPLED_ADAPTIVE,
                    n_genes=n_genes,
                    regulator_indices=reg_idx,
                    n_channels=3,
                    hidden_width=4,
                ),
            ),
            (
                "coupled_ltc",
                ClosedLoopSpec(
                    family=ClosedLoopModelFamily.LTC_COUPLED,
                    n_genes=n_genes,
                    regulator_indices=reg_idx,
                    latent_dim=min(3 * s_tf, 768),
                ),
            ),
        ]

        for label, spec in configs:
            fwd, unpack, flat0 = build_closed_loop_jax_rollout(
                spec, basal_levels=basal, dt=dt, n_steps=n_steps, obs_step_indices=obs_idx
            )
            p_tree = unpack(jnp.asarray(flat0))
            readout_keys = {"direct", "delayed", "readout", "decay", "in_gain", "coop_u", "coop_v", "coop_w"}
            mem_params = int(sum(int(v.size) for k, v in p_tree.items() if k not in readout_keys))
            total_params = int(flat0.size)

            th = jnp.asarray(flat0)

            @jax.jit
            def loss_and_grad(theta_v):
                def _l(t_in):
                    pred = fwd(t_in, ko, y0)
                    return jnp.mean((pred - target) ** 2)

                return jax.value_and_grad(_l)(theta_v)

            # Warmup JIT compile
            val, g = loss_and_grad(th)
            g.block_until_ready()

            # Time 8 forward+backward steps
            n_reps = 6 if s_tf >= 256 else 12
            t_start = time.perf_counter()
            for _ in range(n_reps):
                val, g = loss_and_grad(th)
                g.block_until_ready()
            ms_per_step = 1000.0 * (time.perf_counter() - t_start) / n_reps

            row[f"{label}_mem_params"] = mem_params
            row[f"{label}_total_params"] = total_params
            row[f"{label}_ms_per_step"] = float(ms_per_step)

        rows.append(row)
        print(
            f"S={s_tf:3d} (N={n_genes:4d}): "
            f"MemGRN {row['memgrn_ms_per_step']:.1f}ms (mem_p={row['memgrn_mem_params']}) | "
            f"Cpl-Adapt {row['coupled_adaptive_ms_per_step']:.1f}ms (mem_p={row['coupled_adaptive_mem_params']}) | "
            f"Cpl-LTC {row['coupled_ltc_ms_per_step']:.1f}ms (mem_p={row['coupled_ltc_mem_params']})"
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    design = {
        "benchmark": "yeast_rate_portraits_and_genome_scaling",
        "promoters": list(PROMOTER_CHROMATIN_CLASS.keys()),
        "scaling_s_values": [8, 16, 32, 64, 128, 256, 512],
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    portraits = run_yeast_rate_portraits()
    scaling = run_genome_scale_benchmark()
    elapsed = time.time() - t0

    fast_taus = [portraits[g]["effective_washout_tau_min"] for g in ("DCS2", "HXK1")]
    slow_taus = [portraits[g]["effective_washout_tau_min"] for g in ("SIP18", "ALD3", "DDR2", "TKL2")]

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "yeast_portraits": portraits,
        "chromatin_summary": {
            "fast_nucleosome_free_mean_tau_min": float(np.mean(fast_taus)),
            "slow_chromatin_remodeled_mean_tau_min": float(np.mean(slow_taus)),
            "slow_over_fast_tau_ratio": float(np.mean(slow_taus) / max(1e-6, float(np.mean(fast_taus)))),
        },
        "scaling_benchmark": scaling,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["chromatin_summary"], indent=2))


if __name__ == "__main__":
    main()
