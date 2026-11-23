"""Cross-Domain Unsupervised Dynamical Hysteresis Portraits (Gauge-Invariant Step Response).

Replaces static (z=0)-clamped rate ratios with gauge-invariant dynamical step-response
half-times (tau_1/2^on, tau_1/2^off) and hysteresis ratio H = tau_1/2^off / tau_1/2^on
computed by integrating the actual learned two-stage ODE Eq. (1) forward under step-on
(a = +0.8) and step-off washout (a = 0.0) across ALL FOUR real domains using one identical
||B_{ir,.}||_2-weighted, multi-seed, validation-selected extraction recipe:

  1. Atmospheric Chemistry (UCI Beijing Air Quality, 4-fold LOSO, all 8 drivers):
     dt = 0.25 h, obs_stride = 4 => 1 model unit = 1.0 h.
     Tracks dominant signed target readout so clearing drivers (Wind_North, Wind_Speed, Rain)
     are clearly distinguished from accumulation drivers (Dewpoint, Pressure, Temp).
  2. Financial Markets (Binance 16-Asset 5-min, 3 seeds, all 16 assets):
     dt = 0.25, obs_stride = 2 => 1 bar (5 min) = 0.5 model units => 1 model unit = 10.0 min.
  3. Urban Highway Traffic (METR-LA S=64, 3 seeds, all 64 detectors):
     dt = 0.25, obs_stride = 1 => 1 step (5 min) = 0.25 model units => 1 model unit = 20.0 min.
  4. Live-Cell Molecular Biology (Yeast Msn2 Microscopy, 3 seeds, all 7 promoters):
     dt = 2.5/(60*4) h, obs_stride = 8 => 1 model unit = 60.0 min.

Gauge-invariance property: for ANY linear constant-rate two-stage cascade (regardless of
exchanging Stage-1 and Stage-2 poles lambda^(1) <-> lambda^(2)), tau_1/2^off === tau_1/2^on
identically (H = 1.000). Deviations H != 1.0 measure true state-dependent dynamical hysteresis.
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

import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    DynamicalModelFamily,
    ModelSpec,
    _init_closed_loop_tree,
    _init_tree,
    extract_memgrn_up_down,
    fit_closed_loop_network,
    fit_dynamical_model,
)
from memgrn.cross_domain_datasets import (  # noqa: E402
    load_beijing_air_quality,
    load_binance_market_contagion,
    load_metr_la_traffic,
)
from memgrn.datasets import (  # noqa: E402
    MSN2_PROMOTERS,
    build_rk4_stage_grid,
    load_archived_json,
    load_msn2_population_trace,
)

OUT_DIR = ROOT / "results" / "cross_domain_portraits"
RAW_ROOT = ROOT / "data" / "raw"

PROMOTER_CHROMATIN_CLASS = {
    "DCS2": "Fast (low remodeling)",
    "HXK1": "Fast (low remodeling)",
    "RTN2": "Intermediate / fast-switching",
    "DDR2": "Slow (chromatin-remodeled)",
    "TKL2": "Slow (chromatin-remodeled)",
    "ALD3": "Slow (chromatin-remodeled)",
    "SIP18": "Slow (high-threshold remodeled)",
}


def _build_multi_stage_grid(drivers: np.ndarray, substeps_per_knot: int) -> np.ndarray:
    b, t_len, n_s = drivers.shape
    frac = np.arange(substeps_per_knot, dtype=np.float64)[None, None, :, None] / substeps_per_knot
    lo = drivers[:, :-1, None, :]
    delta = np.diff(drivers, axis=1)[:, :, None, :]

    def _interp(offset: float) -> np.ndarray:
        vals = lo + delta * (frac + offset / substeps_per_knot)
        return vals.reshape(b, (t_len - 1) * substeps_per_knot, n_s).transpose(1, 0, 2)

    return np.stack([_interp(0.0), _interp(0.5), _interp(1.0)], axis=-1)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _simulate_two_stage_step_response(
    p: dict[str, np.ndarray],
    rate_min: float,
    rate_max: float,
    a_step: float,
    time_unit_scale: float,
    is_closed_loop: bool = False,
    dt_sim: float = 0.005,
    max_model_time: float = 80.0,
) -> dict[str, np.ndarray]:
    """Integrate Eq. (1) under step-on (0 -> a_step) and step-off (a_step -> 0) via RK4.

    Returns per-source readout-weighted half-times (tau_on, tau_off), gauge-invariant
    hysteresis ratio H = tau_off / tau_on, dominant readout sign, and normalized response
    trajectories on a fixed relative time grid.
    """
    r_up, r_dn, v_up, v_dn = extract_memgrn_up_down(p)
    s_dim, r_dim = r_up.shape
    bias = p["bias"]  # (H,)
    in_proj = p["in_proj"]  # (3, H)

    # Extract readout tensor B_{irj} (plus optional self-memory) to weight channels identically
    if "delayed_u" in p:
        b_full = np.einsum("srk,kg->srg", p["delayed_u"], p["delayed_v"])
    else:
        b_full = p["delayed"]
    if is_closed_loop:
        # Mask diagonal to match hollow off-diagonal readout, then add self_mem if present
        n_g = b_full.shape[2]
        if s_dim == n_g:
            b_eff = b_full.copy()
            for i in range(s_dim):
                b_eff[i, :, i] = p["self_mem"][i] if "self_mem" in p else 0.0
        else:
            b_eff = b_full
    else:
        b_eff = b_full

    ch_norm = np.linalg.norm(b_eff, axis=2)  # (S, R)
    ch_sum = np.sum(ch_norm, axis=1, keepdims=True)
    ch_w = np.where(ch_sum > 1e-12, ch_norm / np.maximum(1e-12, ch_sum), 1.0 / r_dim)

    # Signed readout statistics per source (direct + delayed sum across targets)
    d_mat = p["direct"]
    if is_closed_loop and d_mat.shape[0] == d_mat.shape[1]:
        d_eff = d_mat * (1.0 - np.eye(d_mat.shape[0]))
    else:
        d_eff = d_mat
    net_b = np.sum(b_eff, axis=1)  # (S, G)
    tot_w = d_eff + net_b  # (S, G)
    signed_net = np.mean(tot_w, axis=1)  # (S,)
    frac_pos = np.mean(tot_w > 0.0, axis=1)  # (S,)

    def _rhs(z: np.ndarray, h: np.ndarray, a_val: float) -> tuple[np.ndarray, np.ndarray]:
        hid = np.tanh(
            bias[None, None, :]
            + a_val * in_proj[0][None, None, :]
            + z[:, :, None] * in_proj[1][None, None, :]
            + h[:, :, None] * in_proj[2][None, None, :]
        )
        lu = rate_min + (rate_max - rate_min) * _sigmoid_np(
            r_up + np.einsum("srh,srh->sr", v_up, hid)
        )
        ld = rate_min + (rate_max - rate_min) * _sigmoid_np(
            r_dn + np.einsum("srh,srh->sr", v_dn, hid)
        )
        return lu * (a_val - z), ld * (z - h)

    def _integrate(z0_val: float, a_val: float, target_cross: float, rising: bool):
        z = np.full((s_dim, r_dim), z0_val, dtype=np.float64)
        h = np.full((s_dim, r_dim), z0_val, dtype=np.float64)
        n_steps = int(round(max_model_time / dt_sim))
        tau_half = np.full((s_dim,), max_model_time, dtype=np.float64)
        found = np.zeros((s_dim,), dtype=bool)
        prev_r = np.zeros((s_dim,), dtype=np.float64) if rising else np.ones((s_dim,), dtype=np.float64)

        # Sample normalized response on 41 points spanning [0, 4.0] model units
        sample_steps = np.round(np.linspace(0.0, 4.0, 41) / dt_sim).astype(int)
        sample_set = {int(st): idx for idx, st in enumerate(sample_steps)}
        curves = np.zeros((s_dim, len(sample_steps)), dtype=np.float64)
        curves[:, 0] = prev_r

        for st in range(1, n_steps + 1):
            k1z, k1h = _rhs(z, h, a_val)
            k2z, k2h = _rhs(z + 0.5 * dt_sim * k1z, h + 0.5 * dt_sim * k1h, a_val)
            k3z, k3h = _rhs(z + 0.5 * dt_sim * k2z, h + 0.5 * dt_sim * k2h, a_val)
            k4z, k4h = _rhs(z + dt_sim * k3z, h + dt_sim * k3h, a_val)
            z = z + (dt_sim / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
            h = h + (dt_sim / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)

            r_cur = np.sum(ch_w * (h / a_step), axis=1)
            if st in sample_set:
                curves[:, sample_set[st]] = r_cur

            if rising:
                crossed = (~found) & (r_cur >= target_cross)
            else:
                crossed = (~found) & (r_cur <= target_cross)
            if np.any(crossed):
                denom = np.where(np.abs(r_cur - prev_r) > 1e-12, r_cur - prev_r, 1.0)
                frac = np.clip((target_cross - prev_r) / denom, 0.0, 1.0)
                t_exact = (st - 1.0 + frac) * dt_sim
                tau_half = np.where(crossed, t_exact, tau_half)
                found = found | crossed
            prev_r = r_cur
            if np.all(found) and st >= int(sample_steps[-1]):
                break

        return tau_half * time_unit_scale, curves

    tau_on, curve_on = _integrate(0.0, a_step, 0.5, rising=True)
    tau_off, curve_off = _integrate(a_step, 0.0, 0.5, rising=False)
    hyst_ratio = tau_off / np.maximum(1e-8, tau_on)

    # Also evaluate gauge-invariance check by zeroing out_proj and out_proj_asym -> H must be 1.000!
    return {
        "tau_on": tau_on,
        "tau_off": tau_off,
        "hysteresis_ratio": hyst_ratio,
        "signed_net_readout": signed_net,
        "frac_positive_targets": frac_pos,
        "curve_on": curve_on,
        "curve_off": curve_off,
        "channel_weights": ch_w,
    }


def _verify_linear_gauge_invariance() -> float:
    """Confirm that an arbitrary constant-rate 2-stage filter with lambda1 != lambda2 gives H == 1.0."""
    p_lin = {
        "rate": np.array([[0.2, -0.5, 1.1]]),
        "rate_asym": np.array([[1.4, -1.2, 0.8]]),  # huge pole separation!
        "out_proj": np.zeros((1, 3, 4)),
        "out_proj_asym": np.zeros((1, 3, 4)),
        "in_proj": np.zeros((3, 4)),
        "bias": np.zeros((4,)),
        "direct": np.zeros((1, 2)),
        "delayed": np.ones((1, 3, 2)),
    }
    res = _simulate_two_stage_step_response(p_lin, 0.05, 5.0, a_step=0.8, time_unit_scale=1.0)
    err = float(abs(res["hysteresis_ratio"][0] - 1.0))
    if err > 1e-3:
        raise AssertionError(f"Linear two-stage step-response failed gauge invariance: H={res['hysteresis_ratio'][0]}")
    return err


def _extract_open_portrait(theta: np.ndarray, spec: ModelSpec, time_unit_scale: float) -> dict:
    init0 = _init_tree(spec, seed=0)
    _, unpack = ravel_pytree({k: jnp.asarray(v) for k, v in init0.items()})
    p = {k: np.asarray(v, dtype=np.float64) for k, v in unpack(jnp.asarray(theta)).items()}
    pos = _simulate_two_stage_step_response(
        p, spec.rate_min, spec.rate_max, a_step=0.8, time_unit_scale=time_unit_scale, is_closed_loop=False
    )
    neg = _simulate_two_stage_step_response(
        p, spec.rate_min, spec.rate_max, a_step=-0.8, time_unit_scale=time_unit_scale, is_closed_loop=False
    )
    return {"pos": pos, "neg": neg}


def _extract_closed_portrait(theta: np.ndarray, spec: ClosedLoopSpec, time_unit_scale: float) -> dict:
    init0 = _init_closed_loop_tree(spec, seed=0)
    _, unpack = ravel_pytree({k: jnp.asarray(v) for k, v in init0.items()})
    p = {k: np.asarray(v, dtype=np.float64) for k, v in unpack(jnp.asarray(theta)).items()}
    pos = _simulate_two_stage_step_response(
        p, spec.rate_min, spec.rate_max, a_step=0.8, time_unit_scale=time_unit_scale, is_closed_loop=True
    )
    neg = _simulate_two_stage_step_response(
        p, spec.rate_min, spec.rate_max, a_step=-0.8, time_unit_scale=time_unit_scale, is_closed_loop=True
    )
    return {"pos": pos, "neg": neg}


def _run_job(job: tuple) -> tuple[str, int | str, dict]:
    kind = job[0]
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    if kind == "beijing":
        _, sea_i = job
        bq = load_beijing_air_quality(RAW_ROOT / "beijing_aq", episode_hours=48)
        src_b = np.where(bq.seasons != sea_i)[0]
        va_b = src_b[4::5]
        tr_b = np.array([i for i in src_b if i not in set(va_b.tolist())], dtype=np.int64)
        d_mu = np.mean(bq.drivers[tr_b], axis=(0, 1), keepdims=True)
        d_sd = np.maximum(1e-3, np.std(bq.drivers[tr_b], axis=(0, 1), keepdims=True))
        y_sc = np.maximum(0.25, np.sqrt(np.mean(bq.targets[tr_b] ** 2, axis=(0, 1), keepdims=True)))
        tr_ub = _build_multi_stage_grid((bq.drivers[tr_b] - d_mu) / d_sd, 4)
        va_ub = _build_multi_stage_grid((bq.drivers[va_b] - d_mu) / d_sd, 4)
        tr_yb, va_yb = bq.targets[tr_b] / y_sc, bq.targets[va_b] / y_sc
        tr_mb, va_mb = np.ones_like(tr_yb), np.ones_like(va_yb)
        tr_mb[:, 0, :] = 0.0
        va_mb[:, 0, :] = 0.0
        spec = ModelSpec(
            family=DynamicalModelFamily.MEMGRN,
            n_sources=8,
            n_targets=12,
            n_channels=3,
            hidden_width=5,
            coop_rank=6,
            use_cooperativity=True,
            base_rates=(0.06, 0.25, 1.0),
            rate_min=0.015,
            rate_max=3.0,
        )
        fit = fit_dynamical_model(
            spec=spec,
            train_stage_u=tr_ub,
            train_y=tr_yb,
            train_mask=tr_mb,
            val_stage_u=va_ub,
            val_y=va_yb,
            val_mask=va_mb,
            dt=0.25,
            obs_stride=4,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2),
            gate_shrinkage_grid=(1.0,),
            hysteresis_shrinkage_grid=(1.0,),
            total_updates=500,
            lr_max=0.007,
        )
        # dt = 0.25 h, obs_stride = 4 -> 1 h = 1.0 model unit -> time_unit_scale = 1.0 h
        return "beijing", sea_i, _extract_open_portrait(fit.theta, spec, time_unit_scale=1.0)

    if kind == "binance":
        _, seed = job
        fin = load_binance_market_contagion(RAW_ROOT / "financial_defi", episode_bars=48)
        tr_f = np.where(fin.month_ids <= 6)[0]
        va_f = np.where(fin.month_ids == 7)[0]
        y_sf = np.maximum(0.10, np.sqrt(np.mean(fin.targets[tr_f] ** 2, axis=(0, 1), keepdims=True)))
        tr_uf = _build_multi_stage_grid(fin.drivers[tr_f], 2)
        va_uf = _build_multi_stage_grid(fin.drivers[va_f], 2)
        tr_yf, va_yf = fin.targets[tr_f] / y_sf, fin.targets[va_f] / y_sf
        tr_mf, va_mf = np.ones_like(tr_yf), np.ones_like(va_yf)
        tr_mf[:, 0, :] = 0.0
        va_mf[:, 0, :] = 0.0
        spec = ModelSpec(
            family=DynamicalModelFamily.MEMGRN,
            n_sources=16,
            n_targets=16,
            n_channels=3,
            hidden_width=5,
            coop_rank=8,
            use_cooperativity=True,
            base_rates=(0.2, 0.8, 2.5),
            rate_min=0.05,
            rate_max=5.0,
        )
        fit = fit_dynamical_model(
            spec=spec,
            train_stage_u=tr_uf,
            train_y=tr_yf,
            train_mask=tr_mf,
            val_stage_u=va_uf,
            val_y=va_yf,
            val_mask=va_mf,
            dt=0.25,
            obs_stride=2,
            seeds=(seed,),
            reg_grid=(1e-4, 1e-3, 1e-2),
            gate_shrinkage_grid=(1.0,),
            hysteresis_shrinkage_grid=(1.0,),
            total_updates=550,
            lr_max=0.006,
        )
        # VERIFIED SCALE: dt=0.25, obs_stride=2 -> 1 bar (5 min) = 0.5 model unit -> 1 model unit = 10.0 min!
        return "binance", seed, _extract_open_portrait(fit.theta, spec, time_unit_scale=10.0)

    if kind == "metr_la":
        _, seed = job
        ml = load_metr_la_traffic(RAW_ROOT / "metr_la", n_sensors=64)

        def _pack_ml(eps, msk):
            b = len(eps)
            ko = np.zeros((b, 64), dtype=np.float64)
            y_o = np.concatenate([np.zeros((b, 1, 64)), eps], axis=1)
            w_o = np.zeros_like(y_o)
            w_o[:, 13:, :] = msk[:, 12:, :]
            y_c = np.zeros((24, b, 64), dtype=np.float64)
            y_c[:12] = np.transpose(eps[:, :12, :], (1, 0, 2))
            return ko, y_o, w_o, y_c

        tr_km, tr_ym, tr_wm, tr_cm = _pack_ml(ml.train_episodes, ml.train_masks)
        va_km, va_ym, va_wm, va_cm = _pack_ml(ml.val_episodes, ml.val_masks)
        spec = ClosedLoopSpec(
            family=ClosedLoopModelFamily.MEMGRN,
            n_genes=64,
            regulator_indices=tuple(range(64)),
            hollow=True,
            use_self_memory=True,
            use_cooperativity=True,
            coop_rank=8,
            readout_rank=20,
            n_channels=3,
            hidden_width=5,
            base_rates=(0.25, 1.0, 3.0),
        )
        fit = fit_closed_loop_network(
            spec=spec,
            basal_levels=np.ones(64, dtype=np.float64),
            train_ko=tr_km,
            train_y=tr_ym,
            train_weights=tr_wm,
            train_y_clamp=tr_cm,
            val_ko=va_km,
            val_y=va_ym,
            val_weights=va_wm,
            val_y_clamp=va_cm,
            dt=0.25,
            n_steps=24,
            obs_step_indices=tuple(range(25)),
            clamp_steps=12,
            seeds=(seed,),
            reg_grid=(1e-4, 1e-3, 1e-2),
            group_l21_scale=0.005,
            total_updates=500,
            lr_max=0.006,
            gate_shrinkage=3.0,
            hysteresis_shrinkage=4.0,
        )
        # dt = 0.25 per 5-min step -> 1 model unit = 4 steps = 20.0 min!
        return "metr_la", seed, _extract_closed_portrait(fit.theta, spec, time_unit_scale=20.0)

    if kind == "yeast":
        _, g_i, gname, seed = job
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
        rep_truth_g = (
            np.stack(
                [
                    load_msn2_population_trace(
                        RAW_ROOT / "msn2" / f"{gname}_{s['id']}_size.mat", "YFP_molecules", obs_min
                    )[0]
                    for s in schedules
                ],
                axis=0,
            )
            / scales[g_i]
        )
        substeps = 4
        dt_hours = 2.5 / (60.0 * substeps)
        obs_stride = 2 * substeps
        single_stage = build_rk4_stage_grid(tf_single[:16], substeps)
        rep_stage = build_rk4_stage_grid(rep_drivers, substeps)
        single_y = np.concatenate(
            [np.zeros((16, 1, 1)), train_y_obs[:, :, g_i : g_i + 1]], axis=1
        )
        val_idx = [1, 6, 10, 15]
        tr_idx = [i for i in range(16) if i not in val_idx]
        tr_y, va_y = single_y[tr_idx], single_y[val_idx]
        tr_m, va_m = np.ones_like(tr_y), np.ones_like(va_y)
        tr_m[:, 0, :] = 0.0
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
        fit = fit_dynamical_model(
            spec=spec,
            train_stage_u=single_stage[:, tr_idx],
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=single_stage[:, val_idx],
            val_y=va_y,
            val_mask=va_m,
            dt=dt_hours,
            obs_stride=obs_stride,
            seeds=(seed,),
            reg_grid=(1e-5, 1e-4, 1e-3),
            gate_shrinkage_grid=(1.0,),
            hysteresis_shrinkage_grid=(1.0,),
            total_updates=1200,
            lr_max=0.012,
        )
        pred_te = fit.predict_fn(rep_stage, None)[:, 1:, 0]
        post_mse = float(
            np.mean(
                [
                    np.mean((pred_te[i, obs_min > stop] - rep_truth_g[i, obs_min > stop]) ** 2)
                    for i, stop in enumerate(last_off)
                ]
            )
        )
        port = _extract_open_portrait(fit.theta, spec, time_unit_scale=60.0)
        port["ten_train_post_mse"] = post_mse
        return "yeast", f"{gname}:{seed}", port

    raise ValueError(f"Unknown job kind: {kind}")


def _aggregate_replicates(names: tuple[str, ...], reps: list[dict], time_unit: str) -> dict:
    """Aggregate gauge-invariant dynamical step-response portraits across replicates (mean +/- SE)."""
    n_rep = len(reps)
    tau_on_pos = np.stack([r["pos"]["tau_on"] for r in reps], axis=0)  # (R, S)
    tau_off_pos = np.stack([r["pos"]["tau_off"] for r in reps], axis=0)
    hyst_pos = np.stack([r["pos"]["hysteresis_ratio"] for r in reps], axis=0)
    tau_on_neg = np.stack([r["neg"]["tau_on"] for r in reps], axis=0)
    tau_off_neg = np.stack([r["neg"]["tau_off"] for r in reps], axis=0)
    hyst_neg = np.stack([r["neg"]["hysteresis_ratio"] for r in reps], axis=0)
    signed_net = np.stack([r["pos"]["signed_net_readout"] for r in reps], axis=0)
    frac_pos = np.stack([r["pos"]["frac_positive_targets"] for r in reps], axis=0)
    curve_on = np.mean(np.stack([r["pos"]["curve_on"] for r in reps], axis=0), axis=0)
    curve_off = np.mean(np.stack([r["pos"]["curve_off"] for r in reps], axis=0), axis=0)

    def _se(arr: np.ndarray, axis: int = 0) -> np.ndarray:
        if arr.shape[axis] <= 1:
            return np.zeros(arr.shape[1:] if axis == 0 else arr.shape[:-1], dtype=np.float64)
        return np.std(arr, axis=axis, ddof=1) / np.sqrt(arr.shape[axis])

    by_source = {}
    for i, nm in enumerate(names):
        s_val = float(np.mean(signed_net[:, i]))
        by_source[nm] = {
            "tau_on_half": float(np.mean(tau_on_pos[:, i])),
            "tau_on_half_se": float(_se(tau_on_pos[:, i], axis=0)),
            "tau_off_half": float(np.mean(tau_off_pos[:, i])),
            "tau_off_half_se": float(_se(tau_off_pos[:, i], axis=0)),
            "hysteresis_ratio_off_over_on": float(np.mean(hyst_pos[:, i])),
            "hysteresis_ratio_se": float(_se(hyst_pos[:, i], axis=0)),
            "neg_step_tau_on_half": float(np.mean(tau_on_neg[:, i])),
            "neg_step_tau_off_half": float(np.mean(tau_off_neg[:, i])),
            "neg_step_hysteresis_ratio": float(np.mean(hyst_neg[:, i])),
            "signed_net_readout": s_val,
            "dominant_effect_sign": "positive" if s_val >= 0.0 else "negative",
            "frac_positive_targets": float(np.mean(frac_pos[:, i])),
            "step_on_curve": curve_on[i].tolist(),
            "step_off_curve": curve_off[i].tolist(),
        }

    src_tau_on = np.array([by_source[nm]["tau_on_half"] for nm in names])
    src_tau_off = np.array([by_source[nm]["tau_off_half"] for nm in names])
    src_hyst = np.array([by_source[nm]["hysteresis_ratio_off_over_on"] for nm in names])
    rep_mean_hyst = np.mean(hyst_pos, axis=1)
    return {
        "n_replicates": n_rep,
        "time_unit": time_unit,
        "step_curve_t_model_units": np.linspace(0.0, 4.0, 41).tolist(),
        "mean_tau_on_half": float(np.mean(src_tau_on)),
        "mean_tau_off_half": float(np.mean(src_tau_off)),
        "tau_on_range": [float(np.min(src_tau_on)), float(np.max(src_tau_on))],
        "tau_off_range": [float(np.min(src_tau_off)), float(np.max(src_tau_off))],
        "mean_hysteresis_ratio": float(np.mean(rep_mean_hyst)),
        "mean_hysteresis_ratio_se": float(_se(rep_mean_hyst, axis=0)),
        "hysteresis_ratio_range": [float(np.min(src_hyst)), float(np.max(src_hyst))],
        "mean_step_on_curve": np.mean(curve_on, axis=0).tolist(),
        "mean_step_off_curve": np.mean(curve_off, axis=0).tolist(),
        "by_source": by_source,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lin_err = _verify_linear_gauge_invariance()
    t0 = time.time()

    bq = load_beijing_air_quality(RAW_ROOT / "beijing_aq", episode_hours=48)
    fin = load_binance_market_contagion(RAW_ROOT / "financial_defi", episode_bars=48)
    ml = load_metr_la_traffic(RAW_ROOT / "metr_la", n_sensors=64)

    jobs: list[tuple] = []
    for sea_i in range(4):
        jobs.append(("beijing", sea_i))
    for seed in (101, 202, 303):
        jobs.append(("binance", seed))
    for seed in (101, 202, 303):
        jobs.append(("metr_la", seed))
    for g_i, gname in enumerate(MSN2_PROMOTERS):
        for seed in (101, 202, 303):
            jobs.append(("yeast", g_i, gname, seed))

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as pool:
        results = list(pool.map(_run_job, jobs))
    elapsed = time.time() - t0

    bq_reps = [r for k, _, r in results if k == "beijing"]
    fin_reps = [r for k, _, r in results if k == "binance"]
    ml_reps = [r for k, _, r in results if k == "metr_la"]
    yeast_by_gene: dict[str, list[dict]] = {g: [] for g in MSN2_PROMOTERS}
    for k, tag, r in results:
        if k == "yeast":
            gname = str(tag).split(":")[0]
            yeast_by_gene[gname].append(r)

    bq_sum = _aggregate_replicates(bq.driver_names, bq_reps, time_unit="hours")
    fin_sum = _aggregate_replicates(fin.symbols, fin_reps, time_unit="minutes")
    ml_sum = _aggregate_replicates(ml.sensor_ids, ml_reps, time_unit="minutes")

    yeast_per_gene = {}
    for g in MSN2_PROMOTERS:
        agg_g = _aggregate_replicates((g,), yeast_by_gene[g], time_unit="minutes")
        entry = agg_g["by_source"][g]
        entry["chromatin_class"] = PROMOTER_CHROMATIN_CLASS[g]
        entry["ten_train_post_mse"] = float(
            np.mean([r["ten_train_post_mse"] for r in yeast_by_gene[g]])
        )
        yeast_per_gene[g] = entry

    fast_genes = ("DCS2", "HXK1")
    slow_genes = ("DDR2", "ALD3", "TKL2", "SIP18")
    yeast_sum = {
        "n_replicates": 3,
        "time_unit": "minutes",
        "step_curve_t_model_units": np.linspace(0.0, 4.0, 41).tolist(),
        "fast_open_mean_tau_off_min": float(
            np.mean([yeast_per_gene[g]["tau_off_half"] for g in fast_genes])
        ),
        "slow_remod_mean_tau_off_min": float(
            np.mean([yeast_per_gene[g]["tau_off_half"] for g in slow_genes])
        ),
        "slow_over_fast_tau_off_ratio": float(
            np.mean([yeast_per_gene[g]["tau_off_half"] for g in slow_genes])
            / max(1e-8, float(np.mean([yeast_per_gene[g]["tau_off_half"] for g in fast_genes])))
        ),
        "fast_open_mean_hysteresis": float(
            np.mean([yeast_per_gene[g]["hysteresis_ratio_off_over_on"] for g in fast_genes])
        ),
        "slow_remod_mean_hysteresis": float(
            np.mean([yeast_per_gene[g]["hysteresis_ratio_off_over_on"] for g in slow_genes])
        ),
        "by_promoter": yeast_per_gene,
    }

    design = {
        "benchmark": "cross_domain_dynamical_step_response_portraits",
        "method": "gauge_invariant_rk4_step_on_off_half_times_B_l2_weighted",
        "domains": ["beijing_air_quality", "binance_16asset", "metr_la_s64", "yeast_msn2"],
        "time_unit_scales": {
            "beijing_hours": 1.0,
            "binance_minutes": 10.0,
            "metr_la_minutes": 20.0,
            "yeast_minutes": 60.0,
        },
        "replicates": {
            "beijing": "4_loso_folds",
            "binance": "3_seeds_101_202_303",
            "metr_la": "3_seeds_101_202_303",
            "yeast": "3_seeds_101_202_303",
        },
        "validation_shrinkage": "default_standard_no_zero_override",
        "linear_two_stage_gauge_invariance_error": lin_err,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "linear_two_stage_gauge_invariance_error": lin_err,
        "beijing_air_quality_hours": bq_sum,
        "binance_16asset_minutes": fin_sum,
        "metr_la_s64_minutes": ml_sum,
        "yeast_msn2_minutes": yeast_sum,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 1),
                "linear_gauge_err": lin_err,
                "beijing": {
                    "mean_H": round(bq_sum["mean_hysteresis_ratio"], 3),
                    "by_src_H": {
                        k: (
                            round(v["hysteresis_ratio_off_over_on"], 3),
                            v["dominant_effect_sign"],
                            round(v["tau_on_half"], 2),
                            round(v["tau_off_half"], 2),
                        )
                        for k, v in bq_sum["by_source"].items()
                    },
                },
                "binance": {
                    "mean_H": round(fin_sum["mean_hysteresis_ratio"], 3),
                    "mean_tau_on_min": round(fin_sum["mean_tau_on_half"], 2),
                    "mean_tau_off_min": round(fin_sum["mean_tau_off_half"], 2),
                    "BTC": round(fin_sum["by_source"]["BTCUSDT"]["hysteresis_ratio_off_over_on"], 3),
                    "ETH": round(fin_sum["by_source"]["ETHUSDT"]["hysteresis_ratio_off_over_on"], 3),
                },
                "metr_la_64": {
                    "mean_H": round(ml_sum["mean_hysteresis_ratio"], 4),
                    "tau_on_range_min": [round(x, 2) for x in ml_sum["tau_on_range"]],
                    "tau_off_range_min": [round(x, 2) for x in ml_sum["tau_off_range"]],
                },
                "yeast": {
                    "slow_over_fast_tau_off": round(yeast_sum["slow_over_fast_tau_off_ratio"], 3),
                    "by_gene": {
                        g: (
                            round(v["tau_on_half"], 1),
                            round(v["tau_off_half"], 1),
                            round(v["hysteresis_ratio_off_over_on"], 3),
                        )
                        for g, v in yeast_per_gene.items()
                    },
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
