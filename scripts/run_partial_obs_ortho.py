"""Ground-Truth Partially Observed Multi-TF Cofactor Benchmark & Source-Local LTC Separation.

Two targeted controlled benchmarks directly testing our core theoretical claims:
  1. Part A (Ground-Truth Partially Observed 6-of-8 Multi-TF Benchmark for MemGRN-Ortho):
     Generates 12 out-of-class 8-TF biochemical worlds (6 additive + 6 cooperative), hides
     2 correlated TFs (observing only 6 of 8 TFs), and evaluates:
       - Local Constant (6 TFs)
       - MemGRN (6 TFs, observed only)
       - MemGRN + Unprojected Cofactor (6 TFs + unconstrained joint latent cofactor -> weight theft!)
       - MemGRN-Ortho (6 TFs + Proposition 1(iii) L2-orthogonalized cofactor)
       - PerturbODE (6 TFs)
       - Latent Neural ODE (6 TFs)
     Measures BOTH held-out OOD intervention MSE AND exact ground-truth attribution RMSE /
     causal parent Edge AP on the 6 observed TFs!
  2. Part B (Parameter-Matched Empirical Separation vs. Source-Local LTC, Theorem 1(iii)):
     Directly compares MemGRN vs. Source-Local LTC at matched parameter count on:
       (i) Second-order post-shutoff delayed-peak targets (short pulses of varying duration);
       (ii) Steady-state causal weight recovery across a 4x probe-amplitude shift.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from memgrn.baselines import (  # noqa: E402
    DynamicalModelFamily,
    ModelSpec,
    counterfactual_source_attribution,
    fit_dynamical_model,
    fit_two_stage_memgrn_ortho,
)
from memgrn.multitf_benchmark import generate_multitf_world  # noqa: E402
from run_multitf_factorial import _auroc_and_ap  # noqa: E402

OUT_DIR = ROOT / "results" / "partial_obs_and_ltc_sep"


def _fit_unprojected_memgrn_cofactor(
    base_memgrn,
    train_u: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    val_u: np.ndarray,
    val_y: np.ndarray,
    val_mask: np.ndarray,
    dt: float,
    obs_stride: int,
):
    """Ablation: unconstrained joint MemGRN + latent cofactor WITHOUT orthogonal projection.

    Directly models an unprojected latent state competing with observed TF channels on
    correlated inputs, demonstrating observed-TF weight theft vs. Proposition 1(iii) MemGRN-Ortho.
    """
    lat_spec = ModelSpec(
        family=DynamicalModelFamily.LATENT_ODE,
        n_sources=base_memgrn.spec.n_sources,
        n_targets=base_memgrn.spec.n_targets,
        latent_dim=14,
    )
    lat_fit = fit_dynamical_model(
        spec=lat_spec,
        train_stage_u=train_u,
        train_y=train_y,
        train_mask=train_mask,
        val_stage_u=val_u,
        val_y=val_y,
        val_mask=val_mask,
        dt=dt,
        obs_stride=obs_stride,
        seeds=(31, 47),
        reg_grid=(1e-4, 1e-3, 1e-2),
        total_updates=1000,
        lr_max=0.009,
    )
    va_m = base_memgrn.predict_fn(val_u, None).ravel()
    va_l = lat_fit.predict_fn(val_u, None).ravel()
    va_t = val_y.ravel()
    design_m = np.stack([va_m, va_l], axis=1)
    weights, _, _, _ = np.linalg.lstsq(design_m, va_t, rcond=None)
    w_m, w_l = float(weights[0]), float(weights[1])

    def predict_unproj(stage_u: np.ndarray, y0: np.ndarray | None = None) -> np.ndarray:
        return w_m * base_memgrn.predict_fn(stage_u, y0) + w_l * lat_fit.predict_fn(stage_u, y0)

    return predict_unproj, base_memgrn.n_params + lat_fit.n_params


def _run_partial_obs_world(job: tuple[int, str]) -> dict:
    """Hide 2 of 8 correlated TFs (indices 2 and 5) and score 6-observed-TF ground-truth recovery."""
    seed, regime = job
    world = generate_multitf_world(seed=seed, regime=regime, n_sources=8, n_targets=24)

    # Hide TF 2 (in module 0={0,1,2}) and TF 5 (in module 1={3,4,5}); observe 6 TFs!
    obs_tf_idx = [0, 1, 3, 4, 6, 7]
    tr_u6 = world.train_stage_u[:, :, obs_tf_idx, :]
    va_u6 = world.val_stage_u[:, :, obs_tf_idx, :]
    te_u6 = world.test_stage_u[:, :, obs_tf_idx, :]
    tr_y, va_y, te_y = world.train_y, world.val_y, world.test_y

    tr_mask = np.ones_like(tr_y)
    tr_mask[:, 0, :] = 0.0
    va_mask = np.ones_like(va_y)
    va_mask[:, 0, :] = 0.0

    # Exact ground-truth counterfactual intervention effects for the 6 OBSERVED TFs
    true_eff_6 = world.test_true_effects[obs_tf_idx]  # (6, B, T, G)
    true_edge_6 = world.parent_mask[obs_tf_idx, :]  # (6, G)

    base_fams = (
        DynamicalModelFamily.LOCAL_CONST,
        DynamicalModelFamily.MEMGRN,
        DynamicalModelFamily.PERTURB_ODE,
        DynamicalModelFamily.LATENT_ODE,
    )
    fitted_map = {}
    for fam in base_fams:
        spec = ModelSpec(
            family=fam,
            n_sources=6,
            n_targets=world.n_targets,
            n_channels=3,
            hidden_width=5,
            latent_dim=14,
            module_dim=14,
        )
        fitted_map[fam.value] = fit_dynamical_model(
            spec=spec,
            train_stage_u=tr_u6,
            train_y=tr_y,
            train_mask=tr_mask,
            val_stage_u=va_u6,
            val_y=va_y,
            val_mask=va_mask,
            dt=world.dt,
            obs_stride=world.obs_stride,
            seeds=(31, 47),
            reg_grid=(1e-4, 1e-3, 1e-2),
            total_updates=1000,
            lr_max=0.009,
        )

    # Fit MemGRN-Ortho (Proposition 1(iii)) and unprojected cofactor ablation
    ortho_fit = fit_two_stage_memgrn_ortho(
        base_memgrn=fitted_map["memgrn"],
        train_stage_u=tr_u6,
        train_y=tr_y,
        train_mask=tr_mask,
        val_stage_u=va_u6,
        val_y=va_y,
        val_mask=va_mask,
        dt=world.dt,
        obs_stride=world.obs_stride,
        latent_dim=14,
        seeds=(31, 47),
        reg_grid=(1e-4, 1e-3, 1e-2),
        total_updates=1000,
        lr_max=0.009,
    )
    unproj_pred, unproj_nparams = _fit_unprojected_memgrn_cofactor(
        fitted_map["memgrn"], tr_u6, tr_y, tr_mask, va_u6, va_y, va_mask, world.dt, world.obs_stride
    )

    models_eval = {
        "local_const": (fitted_map["local_const"].predict_fn, fitted_map["local_const"].predict_fn, fitted_map["local_const"].n_params),
        "memgrn": (fitted_map["memgrn"].predict_fn, fitted_map["memgrn"].predict_fn, fitted_map["memgrn"].n_params),
        "memgrn_unproj": (unproj_pred, unproj_pred, unproj_nparams),
        "memgrn_ortho": (ortho_fit.predict_fn, ortho_fit.predict_observed_only_fn, ortho_fit.n_params),
        "perturb_ode": (fitted_map["perturb_ode"].predict_fn, fitted_map["perturb_ode"].predict_fn, fitted_map["perturb_ode"].n_params),
        "latent_ode": (fitted_map["latent_ode"].predict_fn, fitted_map["latent_ode"].predict_fn, fitted_map["latent_ode"].n_params),
    }

    pure_eff = counterfactual_source_attribution(fitted_map["memgrn"].predict_fn, te_u6, None)
    pure_norm = float(np.linalg.norm(pure_eff)) + 1e-12

    res_world = {}
    for mk, (pred_fn, attr_fn, n_p) in models_eval.items():
        y_hat = pred_fn(te_u6, None)
        test_mse = float(np.mean((y_hat[:, 1:, :] - te_y[:, 1:, :]) ** 2))
        eff_hat = counterfactual_source_attribution(attr_fn, te_u6, None)
        attr_rmse = float(np.sqrt(np.mean((eff_hat[:, :, 1:, :] - true_eff_6[:, :, 1:, :]) ** 2)))
        edge_scores = np.sqrt(np.mean(eff_hat[:, :, 1:, :] ** 2, axis=(1, 2)))
        auroc, ap = _auroc_and_ap(edge_scores.ravel(), true_edge_6.ravel())
        drift_vs_pure = float(np.linalg.norm(eff_hat - pure_eff) / pure_norm)
        res_world[mk] = {
            "n_params": n_p,
            "test_mse": test_mse,
            "gt_obs_attr_rmse": attr_rmse,
            "gt_obs_edge_auroc": auroc,
            "gt_obs_edge_ap": ap,
            "obs_attr_drift_vs_memgrn": drift_vs_pure,
        }
    return {"seed": seed, "regime": regime, "models": res_world}


def _run_ltc_separation_experiment() -> dict:
    """Part B: Parameter-matched MemGRN vs. Source-Local LTC on Theorem 1(iii) tasks.

    Task B1 (Delayed Post-Shutoff Peak Recovery): 12 targets driven by 2-stage chromatin
      loading + clearance (post-shutoff delayed peak after short pulses), 100% parameter-matched.
    Task B2 (Amplitude-Shift Steady-State Weight Recovery): Train on low/mid amplitudes,
      probe steady-state gain at 3x amplitude to measure kinetic-readout entanglement.
    """
    rng = np.random.default_rng(20261121)
    n_src, n_tgt = 4, 12
    dt = 0.05
    n_steps = 120  # 6.0 h
    obs_stride = 4  # 31 obs points every 0.2 h

    # Ground-truth two-stage chromatin-delay plant (strict post-shutoff peaking after short pulses!)
    d_true = np.zeros((n_src, n_tgt), dtype=np.float64)
    b_true = np.zeros((n_src, n_tgt), dtype=np.float64)
    for j in range(n_tgt):
        p1, p2 = j % n_src, (j + 1) % n_src
        b_true[p1, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 1.5)
        b_true[p2, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.1)
    delta_true = rng.uniform(0.8, 1.5, size=n_tgt)
    lam_load = np.array([3.2, 2.5, 4.0, 2.8])
    lam_clear_base = np.array([0.45, 0.65, 0.55, 0.75])

    def _sim_delay_plant(stage_u: np.ndarray) -> np.ndarray:
        batch = stage_u.shape[1]
        z = np.zeros((batch, n_src))
        h = np.zeros((batch, n_src))
        y = np.zeros((batch, n_tgt))
        ys = [y.copy()]
        for st in range(n_steps):
            a0 = np.tanh(stage_u[st, :, :, 0])
            am = np.tanh(stage_u[st, :, :, 1])
            a1 = np.tanh(stage_u[st, :, :, 2])

            def _rhs(zs, hs, ys_, av):
                lu = lam_load[None] / (1.0 + 0.6 * np.abs(hs))
                ld = lam_clear_base[None] / (1.0 + 1.2 * np.abs(hs))
                return lu * (av - zs), ld * (zs - hs), av @ d_true + hs @ b_true - delta_true[None] * ys_

            k1 = _rhs(z, h, y, a0)
            k2 = _rhs(z + 0.5 * dt * k1[0], h + 0.5 * dt * k1[1], y + 0.5 * dt * k1[2], am)
            k3 = _rhs(z + 0.5 * dt * k2[0], h + 0.5 * dt * k2[1], y + 0.5 * dt * k2[2], am)
            k4 = _rhs(z + dt * k3[0], h + dt * k3[1], y + dt * k3[2], a1)
            z = z + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            h = h + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            y = y + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            if (st + 1) % obs_stride == 0:
                ys.append(y.copy())
        return np.stack(ys, axis=1)

    t0 = np.arange(n_steps) * dt
    tm = (np.arange(n_steps) + 0.5) * dt
    t1 = (np.arange(n_steps) + 1.0) * dt
    t3 = np.stack([t0, tm, t1], axis=-1)

    def _make_pulses(n_b: int, dur_range: tuple[float, float], amp_range: tuple[float, float], seed_p: int):
        r = np.random.default_rng(seed_p)
        u = np.zeros((n_steps, n_b, n_src, 3), dtype=np.float64)
        for b in range(n_b):
            for s in range(n_src):
                if r.random() < 0.75:
                    dur = r.uniform(*dur_range)
                    amp = r.uniform(*amp_range)
                    t_on = r.uniform(0.05, 0.35)
                    u[:, b, s, :] = amp * ((t3 >= t_on) & (t3 <= t_on + dur)).astype(np.float64)
        return u

    tr_u = _make_pulses(24, (0.4, 2.5), (0.3, 1.2), 11)
    va_u = _make_pulses(8, (0.4, 2.5), (0.3, 1.2), 22)
    # Test B1: Short pulses (0.25-0.6h) where post-shutoff peaking dominates!
    te_peak_u = _make_pulses(16, (0.25, 0.55), (0.6, 1.5), 33)

    # Test B2: Sustained single-TF step clamps at low (0.25) vs high (1.20) amplitudes
    # to test steady-state effective weight consistency W_eff(high) vs W_eff(low)
    clamp_low_u = np.zeros((n_steps, n_src, n_src, 3), dtype=np.float64)
    clamp_high_u = np.zeros((n_steps, n_src, n_src, 3), dtype=np.float64)
    for s in range(n_src):
        clamp_low_u[:, s, s, :] = 0.25
        clamp_high_u[:, s, s, :] = 1.20

    tr_y = _sim_delay_plant(tr_u) + rng.normal(0.0, 0.008, size=(24, 31, n_tgt))
    va_y = _sim_delay_plant(va_u) + rng.normal(0.0, 0.008, size=(8, 31, n_tgt))
    tr_y[:, 0, :] = 0.0
    va_y[:, 0, :] = 0.0
    te_peak_y = _sim_delay_plant(te_peak_u)
    te_low_y = _sim_delay_plant(clamp_low_u)
    te_high_y = _sim_delay_plant(clamp_high_u)

    tr_m = np.ones_like(tr_y)
    tr_m[:, 0, :] = 0.0
    va_m = np.ones_like(va_y)
    va_m[:, 0, :] = 0.0

    # Parameter-matched specs: MemGRN (n_channels=2, hidden_width=4, additive) vs Source-Local LTC (n_channels=3)
    specs = {
        "memgrn_matched": ModelSpec(
            family=DynamicalModelFamily.MEMGRN_ADDITIVE,
            n_sources=n_src,
            n_targets=n_tgt,
            n_channels=2,
            hidden_width=4,
        ),
        "ltc_local_matched": ModelSpec(
            family=DynamicalModelFamily.LTC_LOCAL,
            n_sources=n_src,
            n_targets=n_tgt,
            n_channels=3,
        ),
        "local_const_matched": ModelSpec(
            family=DynamicalModelFamily.LOCAL_CONST,
            n_sources=n_src,
            n_targets=n_tgt,
            n_channels=3,
        ),
    }

    res_b = {}
    for name, sp in specs.items():
        fit = fit_dynamical_model(
            spec=sp,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=dt,
            obs_stride=obs_stride,
            seeds=(17, 29, 43),
            reg_grid=(1e-5, 1e-4, 1e-3),
            total_updates=1400,
            lr_max=0.01,
        )
        p_peak = fit.predict_fn(te_peak_u, None)
        p_low = fit.predict_fn(clamp_low_u, None)
        p_high = fit.predict_fn(clamp_high_u, None)

        # Post-shutoff peak error (t >= 0.8 h, index 4:)
        peak_mse = float(np.mean((p_peak[:, 4:, :] - te_peak_y[:, 4:, :]) ** 2))
        # Steady-state normalized effective weight matrix W_eff(alpha) = y(T) / tanh(alpha)
        w_eff_low = p_low[:, -1, :] / np.tanh(0.25)
        w_eff_high = p_high[:, -1, :] / np.tanh(1.20)
        w_true_low = te_low_y[:, -1, :] / np.tanh(0.25)
        w_true_high = te_high_y[:, -1, :] / np.tanh(1.20)

        # Amplitude-invariance consistency ||W_eff(high) - W_eff(low)|| / ||W_true||
        amp_invar_err = float(
            np.linalg.norm(w_eff_high - w_eff_low) / (np.linalg.norm(w_true_high) + 1e-12)
        )
        ss_weight_err = float(
            0.5
            * (
                np.linalg.norm(w_eff_low - w_true_low) / np.linalg.norm(w_true_low)
                + np.linalg.norm(w_eff_high - w_true_high) / np.linalg.norm(w_true_high)
            )
        )
        res_b[name] = {
            "n_params": fit.n_params,
            "delayed_peak_post_mse": peak_mse,
            "steady_state_amp_invariance_error": amp_invar_err,
            "steady_state_weight_rel_error": ss_weight_err,
        }
    return res_b


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        *[(100 + i, "additive") for i in range(6)],
        *[(200 + i, "cooperative") for i in range(6)],
    ]
    design = {
        "benchmark": "partial_obs_6of8_ortho_and_local_ltc_separation",
        "part_a_worlds": len(jobs),
        "observed_tfs": [0, 1, 3, 4, 6, 7],
        "hidden_tfs": [2, 5],
        "part_a_models": ["local_const", "memgrn", "memgrn_unproj", "memgrn_ortho", "perturb_ode", "latent_ode"],
        "part_b_models": ["memgrn_matched", "ltc_local_matched", "local_const_matched"],
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as pool:
        part_a_raw = list(pool.map(_run_partial_obs_world, jobs))
    part_b = _run_ltc_separation_experiment()
    elapsed = time.time() - t0

    mk_list = design["part_a_models"]
    part_a_summary = {}
    for mk in mk_list:
        part_a_summary[mk] = {
            "n_params": part_a_raw[0]["models"][mk]["n_params"],
            "test_mse": float(np.mean([w["models"][mk]["test_mse"] for w in part_a_raw])),
            "gt_obs_attr_rmse": float(np.mean([w["models"][mk]["gt_obs_attr_rmse"] for w in part_a_raw])),
            "gt_obs_edge_auroc": float(np.mean([w["models"][mk]["gt_obs_edge_auroc"] for w in part_a_raw])),
            "gt_obs_edge_ap": float(np.mean([w["models"][mk]["gt_obs_edge_ap"] for w in part_a_raw])),
            "obs_attr_drift_vs_memgrn": float(
                np.mean([w["models"][mk]["obs_attr_drift_vs_memgrn"] for w in part_a_raw])
            ),
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "part_a_partial_obs_6of8_summary": part_a_summary,
        "part_a_per_world": part_a_raw,
        "part_b_ltc_separation": part_b,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": elapsed, "part_a": part_a_summary, "part_b": part_b}, indent=2))


if __name__ == "__main__":
    main()
