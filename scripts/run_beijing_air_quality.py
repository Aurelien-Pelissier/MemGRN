"""UCI Beijing Multi-Site Air Quality Benchmark: Leave-One-Season-Out Atmospheric Hysteresis.

Evaluates real-world geophysical loading/washout hysteresis and cross-season correlation shift
on Zhang et al. (Proc. R. Soc. A, 2017): 420,768 station-hours across 12 interacting Beijing
air-quality monitoring stations (2013-2017), 506 clean 48-hour episodes:
  - Inputs (S = 8 physical meteorological drivers): Northerly wind advection, Easterly wind
    advection, Wind speed, Rain, Temp, Pressure, Dewpoint, Diurnal boundary-layer phase.
  - Targets (N = 12 Beijing monitoring stations): 48-hour log1p(PM2.5) accumulation & washout.
  - Task 1 (4-Fold Leave-One-Season-Out Zero-Shot Regime Shift):
      Hold out an entire meteorological season (Spring, Summer, Autumn, Winter); train/validate
      strictly on the other 3 seasons; evaluate zero-shot MSE, R^2, and severe washout-storm MSE
      across all 12 stations and all 506 episodes.
  - Task 2 (Partial Observability / MemGRN-Ortho on Unmeasured Synoptic Pressure & Dewpoint):
      Hold out Pressure + Dewpoint (6 observed meteorological drivers) and verify OOD error gain
      with exact 0.0% observed-driver attribution drift vs. unprojected/Latent ODE.
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
    DynamicalModelFamily,
    ModelSpec,
    counterfactual_source_attribution,
    fit_dynamical_model,
    fit_two_stage_memgrn_ortho,
)
from memgrn.cross_domain_datasets import load_beijing_air_quality  # noqa: E402


def _build_multi_stage_grid(drivers: np.ndarray, substeps_per_knot: int) -> np.ndarray:
    """Interpolate (B, T, S) multi-source drivers onto RK4 stage grid (n_steps, B, S, 3)."""
    b, t_len, n_s = drivers.shape
    frac = np.arange(substeps_per_knot, dtype=np.float64)[None, None, :, None] / substeps_per_knot
    lo = drivers[:, :-1, None, :]
    delta = np.diff(drivers, axis=1)[:, :, None, :]

    def _interp(offset: float) -> np.ndarray:
        vals = lo + delta * (frac + offset / substeps_per_knot)
        return vals.reshape(b, (t_len - 1) * substeps_per_knot, n_s).transpose(1, 0, 2)

    return np.stack([_interp(0.0), _interp(0.5), _interp(1.0)], axis=-1)

OUT_DIR = ROOT / "results" / "beijing_air_quality"
RAW_DIR = ROOT / "data" / "raw" / "beijing_aq"

SEASON_NAMES = ("Spring", "Summer", "Autumn", "Winter")
DT_HOURS = 0.25  # 0.25 h substeps -> 4 substeps per hourly observation (47*4 = 188 steps)
OBS_STRIDE = 4

OPEN_MODELS: tuple[tuple[str, DynamicalModelFamily, int | None, bool], ...] = (
    ("memgrn_ortho", DynamicalModelFamily.MEMGRN_ORTHO, 6, True),
    ("memgrn", DynamicalModelFamily.MEMGRN, 6, True),
    ("memgrn_additive", DynamicalModelFamily.MEMGRN_ADDITIVE, None, False),
    ("local_const", DynamicalModelFamily.LOCAL_CONST, None, False),
    ("ltc_local", DynamicalModelFamily.LTC_LOCAL, None, False),
    ("coupled_adaptive", DynamicalModelFamily.COUPLED_ADAPTIVE, None, False),
    ("ltc_coupled", DynamicalModelFamily.LTC_COUPLED, None, False),
    ("selective_ssm", DynamicalModelFamily.SELECTIVE_SSM, None, False),
    ("cfc", DynamicalModelFamily.CFC, None, False),
    ("neural_cde", DynamicalModelFamily.NEURAL_CDE, None, False),
    ("perturb_ode", DynamicalModelFamily.PERTURB_ODE, None, False),
    ("latent_ode", DynamicalModelFamily.LATENT_ODE, None, False),
)


def _run_season_job(job: tuple[int, str, str, int | None, bool, bool]) -> tuple[int, str, bool, dict]:
    sea_i, mk, fam_val, coop_rank, use_coop, partial_obs = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    data = load_beijing_air_quality(RAW_DIR, episode_hours=48)
    test_idx = np.where(data.seasons == sea_i)[0]
    src_idx = np.where(data.seasons != sea_i)[0]
    # Stratified chronological validation: every 5th source episode (zero season leakage)
    val_idx = src_idx[4::5]
    train_idx = np.array([i for i in src_idx if i not in set(val_idx.tolist())], dtype=np.int64)

    # Partial observability hides Pressure (idx 5) & Dewpoint (idx 6) -> 6 observed drivers
    drv_cols = [0, 1, 2, 3, 4, 7] if partial_obs else list(range(8))
    n_s = len(drv_cols)
    n_t = len(data.stations)

    # Standardize drivers & target scale strictly on training seasons
    tr_d = data.drivers[train_idx][:, :, drv_cols]
    d_mu = np.mean(tr_d, axis=(0, 1), keepdims=True)
    d_sd = np.maximum(1e-3, np.std(tr_d, axis=(0, 1), keepdims=True))

    tr_y_raw = data.targets[train_idx]
    y_sc = np.maximum(0.25, np.sqrt(np.mean(tr_y_raw**2, axis=(0, 1), keepdims=True)))

    def _prep(idxs: np.ndarray):
        u_obs = (data.drivers[idxs][:, :, drv_cols] - d_mu) / d_sd
        u_stage = _build_multi_stage_grid(u_obs, substeps_per_knot=OBS_STRIDE)
        y_n = data.targets[idxs] / y_sc
        m_n = np.ones_like(y_n)
        m_n[:, 0, :] = 0.0
        return u_stage, y_n, m_n

    tr_u, tr_y, tr_m = _prep(train_idx)
    va_u, va_y, va_m = _prep(val_idx)
    te_u, te_y, te_m = _prep(test_idx)

    fam = DynamicalModelFamily(fam_val)
    if fam == DynamicalModelFamily.MEMGRN_ORTHO:
        base_spec = ModelSpec(
            family=DynamicalModelFamily.MEMGRN,
            n_sources=n_s,
            n_targets=n_t,
            n_channels=3,
            hidden_width=5,
            coop_rank=coop_rank,
            use_cooperativity=True,
            base_rates=(0.06, 0.25, 1.0),
            rate_min=0.015,
            rate_max=3.0,
        )
        base_fit = fit_dynamical_model(
            spec=base_spec,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=DT_HOURS,
            obs_stride=OBS_STRIDE,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2),
            gate_shrinkage_grid=(0.5, 1.5),
            hysteresis_shrinkage_grid=(0.0, 1.0),
            total_updates=500,
            lr_max=0.007,
        )
        fitted = fit_two_stage_memgrn_ortho(
            base_memgrn=base_fit,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=DT_HOURS,
            obs_stride=OBS_STRIDE,
            latent_dim=12,
            seeds=(101, 202),
            reg_grid=(1e-3, 1e-2, 5e-2),
            total_updates=400,
            lr_max=0.007,
        )
    else:
        spec = ModelSpec(
            family=fam,
            n_sources=n_s,
            n_targets=n_t,
            n_channels=3,
            hidden_width=5,
            latent_dim=24,
            module_dim=24,
            coop_rank=coop_rank,
            use_cooperativity=use_coop,
            base_rates=(0.06, 0.25, 1.0),
            rate_min=0.015,
            rate_max=3.0,
        )
        fitted = fit_dynamical_model(
            spec=spec,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=DT_HOURS,
            obs_stride=OBS_STRIDE,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2),
            gate_shrinkage_grid=(0.5, 1.5),
            hysteresis_shrinkage_grid=(0.0, 1.0),
            total_updates=500,
            lr_max=0.007,
        )

    pred = fitted.predict_fn(te_u, None)
    e2 = (pred[:, 1:, :] - te_y[:, 1:, :]) ** 2
    ep_mse = np.mean(e2, axis=(1, 2))
    wash_m = data.washout_flags[test_idx]
    fold_mse = float(np.mean(ep_mse))
    fold_var = float(np.mean(te_y[:, 1:, :] ** 2)) + 1e-8
    fold_r2 = 1.0 - fold_mse / fold_var
    wash_mse = float(np.mean(ep_mse[wash_m])) if np.any(wash_m) else fold_mse

    # Counterfactual attribution on first 16 test episodes (for Ortho zero-drift audit)
    attr_fn = (
        fitted.predict_observed_only_fn
        if fitted.predict_observed_only_fn is not None
        else fitted.predict_fn
    )
    attr = counterfactual_source_attribution(attr_fn, te_u[:, :16])
    attr_mag = np.sqrt(np.mean(attr[:, :, 1:, :] ** 2, axis=(1, 2)))  # (S, N)

    return (
        sea_i,
        mk,
        partial_obs,
        {
            "n_params": fitted.n_params,
            "test_indices": test_idx.tolist(),
            "per_ep_mse": ep_mse.tolist(),
            "fold_mse": fold_mse,
            "fold_r2": fold_r2,
            "washout_mse": wash_mse,
            "attr_mag": attr_mag.tolist(),
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_beijing_air_quality(RAW_DIR, episode_hours=48)

    design = {
        "benchmark": "uci_beijing_multisite_air_quality_loso",
        "source": "Zhang et al., Proc. R. Soc. A (2017) — UCI Beijing Multi-Site Air Quality (12 stations)",
        "total_station_hours": data.total_station_hours,
        "n_episodes": int(len(data.seasons)),
        "n_washout_storms": int(np.sum(data.washout_flags)),
        "seasons": list(SEASON_NAMES),
        "n_drivers_full": 8,
        "n_drivers_partial": 6,
        "hidden_drivers_partial": ["Pressure", "Dewpoint"],
        "models_full": [m[0] for m in OPEN_MODELS],
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    jobs = [(s_i, mk, fam.value, cr, uc, False) for s_i in range(4) for mk, fam, cr, uc in OPEN_MODELS]
    # Also run partial-observability (6 observed drivers) for key Ortho comparison models
    po_models = [m for m in OPEN_MODELS if m[0] in ("memgrn_ortho", "memgrn", "local_const", "perturb_ode", "latent_ode")]
    jobs += [(s_i, mk, fam.value, cr, uc, True) for s_i in range(4) for mk, fam, cr, uc in po_models]

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=24, mp_context=ctx) as pool:
        raw_out = list(pool.map(_run_season_job, jobs))
    elapsed = time.time() - t0

    def _agg(is_partial: bool, model_keys: list[str]) -> tuple[dict, dict]:
        by_m: dict[str, list[dict]] = {m: [] for m in model_keys}
        for _, mk, po_f, res in raw_out:
            if po_f == is_partial and mk in by_m:
                by_m[mk].append(res)
        n_ep = len(data.seasons)
        ep_map: dict[str, np.ndarray] = {}
        attr_map: dict[str, np.ndarray] = {}
        summ: dict[str, dict] = {}
        for mk in model_keys:
            fl = by_m[mk]
            vec = np.zeros(n_ep, dtype=np.float64)
            for fd in fl:
                for idx_e, val_e in zip(fd["test_indices"], fd["per_ep_mse"]):
                    vec[idx_e] = val_e
            ep_map[mk] = vec
            attr_map[mk] = np.mean([np.asarray(fd["attr_mag"]) for fd in fl], axis=0)
            summ[mk] = {
                "n_params": fl[0]["n_params"],
                "loso_mse": float(np.mean(vec)),
                "loso_mse_se": float(np.std(vec, ddof=1) / np.sqrt(n_ep)),
                "loso_r2": float(np.mean([fd["fold_r2"] for fd in fl])),
                "washout_storm_mse": float(np.mean(vec[data.washout_flags])),
                "washout_storm_mse_se": float(
                    np.std(vec[data.washout_flags], ddof=1) / np.sqrt(np.sum(data.washout_flags))
                ),
            }
        # Attribution drift relative to standalone memgrn
        ref_a = attr_map["memgrn"]
        ref_n = float(np.linalg.norm(ref_a)) + 1e-12
        for mk in model_keys:
            summ[mk]["obs_driver_attr_drift_vs_memgrn"] = float(
                np.linalg.norm(attr_map[mk] - ref_a) / ref_n
            )

        ref_m = ep_map["memgrn_ortho" if is_partial else "memgrn"]
        wilc: dict[str, dict] = {}
        ref_name = "memgrn_ortho" if is_partial else "memgrn"
        for mk in model_keys:
            if mk == ref_name:
                continue
            oth = ep_map[mk]
            p_v = float(wilcoxon(ref_m, oth, alternative="less").pvalue) if np.any(ref_m != oth) else 1.0
            wilc[mk] = {
                "pct_reduction": float(100.0 * (1.0 - np.mean(ref_m) / max(1e-12, np.mean(oth)))),
                "wins": int(np.sum(ref_m < oth)),
                "total": n_ep,
                "wilcoxon_p": p_v,
            }
        return summ, wilc

    full_sum, full_wilc = _agg(False, [m[0] for m in OPEN_MODELS])
    po_sum, po_wilc = _agg(True, [m[0] for m in po_models])

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "full_8driver_loso": full_sum,
        "full_wilcoxon_vs_memgrn": full_wilc,
        "partial_6driver_ortho": po_sum,
        "partial_wilcoxon_vs_memgrn_ortho": po_wilc,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 1),
                "full_8driver_loso": full_sum,
                "partial_6driver_ortho": po_sum,
                "full_wilcoxon": full_wilc,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
