"""Cross-Domain Unsupervised Hysteresis Mechanism Discovery (Upgrade #4).

Extracts learned loading (lambda_up) vs. clearance (lambda_down) rate landscapes, rise/recovery
asymmetry ratios, and effective memory timescales from source-isolated MemGRN across FOUR real
physical & biological domains without any mechanistic supervision:
  1. Atmospheric Chemistry (UCI Beijing Air Quality): cold-front northerly wind / rain vs.
     stagnant boundary-layer loading across 12 stations.
  2. Urban Highway Traffic (METR-LA S=64): congestion shock-loading vs. nonlinear queue clearance
     across 64 Los Angeles highway loop detectors.
  3. Financial Markets (Binance 16-Asset 5-min): sell-side liquidation order-flow shock vs. slow
     nonlinear volatility/liquidity replenishment (recovering the financial leverage effect).
  4. Live-Cell Molecular Biology (Yeast Msn2 Microscopy): fast nucleosome-free (HXK1/DCS2) vs.
     slow chromatin-remodeled (ALD3/TKL2/SIP18) promoters (archived portraits).
"""

import hashlib
import json
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
    fit_closed_loop_network,
    fit_dynamical_model,
)
from memgrn.cross_domain_datasets import (  # noqa: E402
    load_beijing_air_quality,
    load_binance_market_contagion,
    load_metr_la_traffic,
)

OUT_DIR = ROOT / "results" / "cross_domain_portraits"
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


def _eval_open_rates(
    theta: np.ndarray, spec: ModelSpec, h_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate learned loading rate lambda_up(a=0.8, z=h, h) and clearance rate lambda_dn(a=0, z=0, h)."""
    init0 = _init_tree(spec, seed=0)
    _, unpack = ravel_pytree({k: jnp.asarray(v) for k, v in init0.items()})
    p = {k: np.asarray(v, dtype=np.float64) for k, v in unpack(jnp.asarray(theta)).items()}
    r_min, r_max = spec.rate_min, spec.rate_max
    r_up = p["rate"] + p["rate_asym"]  # (S, R)
    r_dn = p["rate"] - p["rate_asym"]
    v_up = p["out_proj"] + p["out_proj_asym"]  # (S, R, H)
    v_dn = p["out_proj"] - p["out_proj_asym"]

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    # Loading regime: active shock a = +0.8, z = h
    lam_up_grid = np.zeros((spec.n_sources, len(h_grid)), dtype=np.float64)
    # Clearance/washout regime: post-shock a = 0.0, z = 0.0, memory = h
    lam_dn_grid = np.zeros((spec.n_sources, len(h_grid)), dtype=np.float64)
    for idx_h, hv in enumerate(h_grid):
        hid_on = np.tanh(
            p["bias"] + 0.8 * p["in_proj"][0] + hv * p["in_proj"][1] + hv * p["in_proj"][2]
        )
        hid_off = np.tanh(p["bias"] + hv * p["in_proj"][2])
        lu = r_min + (r_max - r_min) * _sigmoid(r_up + np.einsum("srh,h->sr", v_up, hid_on))
        ld = r_min + (r_max - r_min) * _sigmoid(r_dn + np.einsum("srh,h->sr", v_dn, hid_off))
        lam_up_grid[:, idx_h] = np.mean(lu, axis=1)
        lam_dn_grid[:, idx_h] = np.mean(ld, axis=1)
    return lam_up_grid, lam_dn_grid


def _eval_closed_rates(
    theta: np.ndarray, spec: ClosedLoopSpec, h_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    init0 = _init_closed_loop_tree(spec, seed=0)
    _, unpack = ravel_pytree({k: jnp.asarray(v) for k, v in init0.items()})
    p = {k: np.asarray(v, dtype=np.float64) for k, v in unpack(jnp.asarray(theta)).items()}
    r_min, r_max = spec.rate_min, spec.rate_max
    r_up = p["rate"] + p["rate_asym"]
    r_dn = p["rate"] - p["rate_asym"]
    v_up = p["out_proj"] + p["out_proj_asym"]
    v_dn = p["out_proj"] - p["out_proj_asym"]
    s_reg = r_up.shape[0]

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    lam_up_grid = np.zeros((s_reg, len(h_grid)), dtype=np.float64)
    lam_dn_grid = np.zeros((s_reg, len(h_grid)), dtype=np.float64)
    for idx_h, hv in enumerate(h_grid):
        hid_on = np.tanh(
            p["bias"] + 0.8 * p["in_proj"][0] + hv * p["in_proj"][1] + hv * p["in_proj"][2]
        )
        hid_off = np.tanh(p["bias"] + hv * p["in_proj"][2])
        lu = r_min + (r_max - r_min) * _sigmoid(r_up + np.einsum("srh,h->sr", v_up, hid_on))
        ld = r_min + (r_max - r_min) * _sigmoid(r_dn + np.einsum("srh,h->sr", v_dn, hid_off))
        lam_up_grid[:, idx_h] = np.mean(lu, axis=1)
        lam_dn_grid[:, idx_h] = np.mean(ld, axis=1)
    return lam_up_grid, lam_dn_grid


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h_grid = np.linspace(0.0, 0.9, 19)
    t0 = time.time()

    # 1. UCI Beijing Air Quality (train on Spring+Summer+Autumn, val on 20% source)
    bq = load_beijing_air_quality(RAW_ROOT / "beijing_aq", episode_hours=48)
    src_b = np.where(bq.seasons != 3)[0]
    va_b = src_b[4::5]
    tr_b = np.array([i for i in src_b if i not in set(va_b.tolist())], dtype=np.int64)
    d_mu = np.mean(bq.drivers[tr_b], axis=(0, 1), keepdims=True)
    d_sd = np.maximum(1e-3, np.std(bq.drivers[tr_b], axis=(0, 1), keepdims=True))
    y_sc = np.maximum(0.25, np.sqrt(np.mean(bq.targets[tr_b] ** 2, axis=(0, 1), keepdims=True)))
    tr_ub = _build_multi_stage_grid((bq.drivers[tr_b] - d_mu) / d_sd, 2)
    va_ub = _build_multi_stage_grid((bq.drivers[va_b] - d_mu) / d_sd, 2)
    tr_yb, va_yb = bq.targets[tr_b] / y_sc, bq.targets[va_b] / y_sc
    tr_mb, va_mb = np.ones_like(tr_yb), np.ones_like(va_yb)
    tr_mb[:, 0, :] = 0.0
    va_mb[:, 0, :] = 0.0
    spec_bq = ModelSpec(
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
    fit_bq = fit_dynamical_model(
        spec=spec_bq,
        train_stage_u=tr_ub,
        train_y=tr_yb,
        train_mask=tr_mb,
        val_stage_u=va_ub,
        val_y=va_yb,
        val_mask=va_mb,
        dt=0.5,
        obs_stride=2,
        seeds=(101,),
        reg_grid=(1e-4, 1e-3),
        gate_shrinkage_grid=(0.5,),
        hysteresis_shrinkage_grid=(0.0,),
        total_updates=550,
        lr_max=0.007,
    )
    bq_up, bq_dn = _eval_open_rates(fit_bq.theta, spec_bq, h_grid)

    # 2. Binance 16-Asset Financial Contagion (May-Jun train, Jul val)
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
    spec_fin = ModelSpec(
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
    fit_fin = fit_dynamical_model(
        spec=spec_fin,
        train_stage_u=tr_uf,
        train_y=tr_yf,
        train_mask=tr_mf,
        val_stage_u=va_uf,
        val_y=va_yf,
        val_mask=va_mf,
        dt=0.25,
        obs_stride=2,
        seeds=(101,),
        reg_grid=(1e-4, 1e-3),
        gate_shrinkage_grid=(0.5,),
        hysteresis_shrinkage_grid=(0.0,),
        total_updates=550,
        lr_max=0.006,
    )
    fin_up, fin_dn = _eval_open_rates(fit_fin.theta, spec_fin, h_grid)

    # 3. METR-LA Highway Traffic (S=64, self-memory closed-loop MemGRN)
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
    spec_ml = ClosedLoopSpec(
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
    fit_ml = fit_closed_loop_network(
        spec=spec_ml,
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
        seeds=(101,),
        reg_grid=(1e-4, 1e-3),
        group_l21_scale=0.005,
        total_updates=450,
        lr_max=0.006,
        gate_shrinkage=0.5,
        hysteresis_shrinkage=0.0,
    )
    ml_up, ml_dn = _eval_closed_rates(fit_ml.theta, spec_ml, h_grid)

    # 4. Archived Yeast Chromatin Portraits for unified 4-domain comparison
    yeast = json.loads(
        (ROOT / "results" / "interpretability_and_scaling" / "scores.json").read_text()
    )["yeast_portraits"]

    elapsed = time.time() - t0
    design = {
        "benchmark": "cross_domain_learned_hysteresis_portraits",
        "domains": ["beijing_air_quality", "binance_16asset", "metr_la_s64", "yeast_msn2"],
        "h_grid": h_grid.tolist(),
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    def _summarize(names, lu, ld, time_unit_scale: float):
        # time_unit_scale converts 1/rate into physical minutes or hours
        per_src = {}
        for i, nm in enumerate(names):
            u_m = float(np.mean(lu[i]))
            d_m = float(np.mean(ld[i]))
            per_src[nm] = {
                "mean_lam_up": u_m,
                "mean_lam_dn": d_m,
                "asymmetry_up_over_dn": u_m / max(1e-8, d_m),
                "tau_up": time_unit_scale / max(1e-8, u_m),
                "tau_dn": time_unit_scale / max(1e-8, d_m),
                "lam_up_h": lu[i].tolist(),
                "lam_dn_h": ld[i].tolist(),
            }
        all_u = np.mean(lu, axis=0)
        all_d = np.mean(ld, axis=0)
        return {
            "mean_lam_up_h": all_u.tolist(),
            "mean_lam_dn_h": all_d.tolist(),
            "mean_asymmetry_up_over_dn": float(np.mean(all_u) / max(1e-8, np.mean(all_d))),
            "mean_tau_up": float(time_unit_scale / max(1e-8, np.mean(all_u))),
            "mean_tau_dn": float(time_unit_scale / max(1e-8, np.mean(all_d))),
            "by_source": per_src,
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "h_grid": h_grid.tolist(),
        "beijing_air_quality_hours": _summarize(bq.driver_names, bq_up, bq_dn, time_unit_scale=1.0),
        "binance_16asset_minutes": _summarize(fin.symbols, fin_up, fin_dn, time_unit_scale=20.0),
        "metr_la_s64_minutes": _summarize(ml.sensor_ids[:10], ml_up[:10], ml_dn[:10], time_unit_scale=20.0),
        "metr_la_all64_summary": {
            "mean_lam_up_h": np.mean(ml_up, axis=0).tolist(),
            "mean_lam_dn_h": np.mean(ml_dn, axis=0).tolist(),
            "mean_asymmetry_up_over_dn": float(np.mean(ml_up) / max(1e-8, np.mean(ml_dn))),
            "mean_tau_up_min": float(20.0 / max(1e-8, np.mean(ml_up))),
            "mean_tau_dn_min": float(20.0 / max(1e-8, np.mean(ml_dn))),
        },
        "yeast_chromatin_summary": {
            "fast_open_asymmetry": float(
                np.mean([yeast[g]["asymmetry_rise_over_recovery"] for g in ("HXK1", "DCS2")])
            ),
            "slow_remod_asymmetry": float(
                np.mean([yeast[g]["asymmetry_rise_over_recovery"] for g in ("DDR2", "ALD3", "TKL2", "SIP18")])
            ),
        },
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 1),
                "beijing_mean_asym": round(out["beijing_air_quality_hours"]["mean_asymmetry_up_over_dn"], 3),
                "beijing_windN_asym": round(out["beijing_air_quality_hours"]["by_source"]["Wind_North"]["asymmetry_up_over_dn"], 3),
                "beijing_rain_asym": round(out["beijing_air_quality_hours"]["by_source"]["Rain"]["asymmetry_up_over_dn"], 3),
                "binance_mean_asym": round(out["binance_16asset_minutes"]["mean_asymmetry_up_over_dn"], 3),
                "binance_tau_up_min": round(out["binance_16asset_minutes"]["mean_tau_up"], 1),
                "binance_tau_dn_min": round(out["binance_16asset_minutes"]["mean_tau_dn"], 1),
                "metr_la_mean_asym": round(out["metr_la_all64_summary"]["mean_asymmetry_up_over_dn"], 3),
                "metr_la_tau_up_min": round(out["metr_la_all64_summary"]["mean_tau_up_min"], 1),
                "metr_la_tau_dn_min": round(out["metr_la_all64_summary"]["mean_tau_dn_min"], 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
