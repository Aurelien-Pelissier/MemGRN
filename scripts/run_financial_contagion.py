"""Financial Market Multi-Asset Order-Flow & Volatility Contagion (Binance Vision, 16 Assets).

Evaluates asymmetric liquidity/volatility hysteresis (leverage effect: sharp sell-side shock
loading vs. slow nonlinear replenishment), correlation-breaking idiosyncratic asset shocks,
and unmeasured macro-factor partial observability (MemGRN-Ortho) on 566,784 5-minute bars
across 16 interacting liquid crypto markets (May-August 2024; 737 strict causal 4-hour episodes):
  - Chronological split (zero leakage):
      * Train: May-June 2024 (366 episodes)
      * Val:   July 2024 (186 episodes)
      * Test:  August 2024 (185 episodes, including the Aug 5 yen-carry liquidation flash crash
               and 60+ idiosyncratic single-asset shock episodes!)
  - Task 1 (Full 16-Asset Contagion & Idiosyncratic Shock Generalization):
      Evaluate all 12 models (MemGRN-Ortho, MemGRN, MemGRN Gamma=0, Local Constant, Source-Local
      LTC, Coupled Adaptive, Coupled LTC, Selective SSM/Mamba, CfC, Neural CDE, PerturbODE,
      Latent ODE) on August OOD MSE, R^2, idiosyncratic-shock MSE, and self-vs.-cross attribution
      specificity (ratio of own-asset counterfactual effect vs. off-diagonal cross-talk).
  - Task 2 (Partial Observability: Unmeasured BTC & ETH Macro Factors -> MemGRN-Ortho):
      Hide BTCUSDT and ETHUSDT (train/test using only the 14 altcoins as observed drivers),
      testing Proposition 1(iii)'s OOD error reduction + exact 0.0% observed-asset drift!
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
from memgrn.cross_domain_datasets import load_binance_market_contagion  # noqa: E402


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

OUT_DIR = ROOT / "results" / "financial_contagion"
RAW_DIR = ROOT / "data" / "raw" / "financial_defi"

DT = 0.25  # 2 substeps per 5-min bar (in 20-min units)
OBS_STRIDE = 2

OPEN_MODELS: tuple[tuple[str, DynamicalModelFamily, int | None, bool], ...] = (
    ("memgrn_ortho", DynamicalModelFamily.MEMGRN_ORTHO, 8, True),
    ("memgrn", DynamicalModelFamily.MEMGRN, 8, True),
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


def _run_fin_job(job: tuple[str, str, int | None, bool, bool]) -> tuple[str, bool, dict]:
    mk, fam_val, coop_rank, use_coop, hide_btc_eth = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    data = load_binance_market_contagion(RAW_DIR, episode_bars=48)
    tr_idx = np.where(data.month_ids <= 6)[0]
    va_idx = np.where(data.month_ids == 7)[0]
    te_idx = np.where(data.month_ids == 8)[0]

    # Partial observability: hide BTC (idx 0) & ETH (idx 1), predict all 14 altcoins from 14 altcoin drivers
    src_cols = list(range(2, 16)) if hide_btc_eth else list(range(16))
    tgt_cols = list(range(2, 16)) if hide_btc_eth else list(range(16))
    n_s, n_t = len(src_cols), len(tgt_cols)

    tr_y_raw = data.targets[tr_idx][:, :, tgt_cols]
    y_sc = np.maximum(0.10, np.sqrt(np.mean(tr_y_raw**2, axis=(0, 1), keepdims=True)))

    def _prep(idxs: np.ndarray):
        u_obs = data.drivers[idxs][:, :, src_cols]
        u_stage = _build_multi_stage_grid(u_obs, substeps_per_knot=OBS_STRIDE)
        y_n = data.targets[idxs][:, :, tgt_cols] / y_sc
        m_n = np.ones_like(y_n)
        m_n[:, 0, :] = 0.0
        return u_stage, y_n, m_n

    tr_u, tr_y, tr_m = _prep(tr_idx)
    va_u, va_y, va_m = _prep(va_idx)
    te_u, te_y, te_m = _prep(te_idx)

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
            base_rates=(0.2, 0.8, 2.5),
            rate_min=0.05,
            rate_max=5.0,
        )
        base_fit = fit_dynamical_model(
            spec=base_spec,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=DT,
            obs_stride=OBS_STRIDE,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2),
            gate_shrinkage_grid=(0.5, 1.5),
            hysteresis_shrinkage_grid=(0.0, 1.0),
            total_updates=550,
            lr_max=0.006,
        )
        fitted = fit_two_stage_memgrn_ortho(
            base_memgrn=base_fit,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=DT,
            obs_stride=OBS_STRIDE,
            latent_dim=16,
            seeds=(101, 202),
            reg_grid=(1e-3, 1e-2, 5e-2),
            total_updates=400,
            lr_max=0.006,
        )
    else:
        spec = ModelSpec(
            family=fam,
            n_sources=n_s,
            n_targets=n_t,
            n_channels=3,
            hidden_width=5,
            latent_dim=32,
            module_dim=32,
            coop_rank=coop_rank,
            use_cooperativity=use_coop,
            base_rates=(0.2, 0.8, 2.5),
            rate_min=0.05,
            rate_max=5.0,
        )
        fitted = fit_dynamical_model(
            spec=spec,
            train_stage_u=tr_u,
            train_y=tr_y,
            train_mask=tr_m,
            val_stage_u=va_u,
            val_y=va_y,
            val_mask=va_m,
            dt=DT,
            obs_stride=OBS_STRIDE,
            seeds=(101, 202),
            reg_grid=(1e-4, 1e-3, 1e-2),
            gate_shrinkage_grid=(0.5, 1.5),
            hysteresis_shrinkage_grid=(0.0, 1.0),
            total_updates=550,
            lr_max=0.006,
        )

    pred = fitted.predict_fn(te_u, None)
    e2 = (pred[:, 1:, :] - te_y[:, 1:, :]) ** 2
    ep_mse = np.mean(e2, axis=(1, 2))
    idio_m = data.idiosyncratic_flags[te_idx]
    all_mse = float(np.mean(ep_mse))
    all_var = float(np.mean(te_y[:, 1:, :] ** 2)) + 1e-8
    all_r2 = 1.0 - all_mse / all_var
    idio_mse = float(np.mean(ep_mse[idio_m]))

    # Counterfactual attribution matrix (S x N) across first 24 idiosyncratic test episodes
    idio_pos = np.where(idio_m)[0][:24]
    attr_fn = (
        fitted.predict_observed_only_fn
        if fitted.predict_observed_only_fn is not None
        else fitted.predict_fn
    )
    attr = counterfactual_source_attribution(attr_fn, te_u[:, idio_pos])
    attr_mag = np.sqrt(np.mean(attr[:, :, 1:, :] ** 2, axis=(1, 2)))  # (S, N)
    diag_mean = float(np.mean(np.diag(attr_mag)))
    off_mean = float(np.mean(attr_mag[~np.eye(n_s, dtype=bool)]))
    self_specificity = diag_mean / max(1e-8, off_mean)

    return (
        mk,
        hide_btc_eth,
        {
            "n_params": fitted.n_params,
            "aug_test_mse": all_mse,
            "aug_test_mse_se": float(np.std(ep_mse, ddof=1) / np.sqrt(len(ep_mse))),
            "aug_test_r2": all_r2,
            "idio_shock_mse": idio_mse,
            "idio_shock_mse_se": float(np.std(ep_mse[idio_m], ddof=1) / np.sqrt(np.sum(idio_m))),
            "self_vs_cross_specificity": self_specificity,
            "attr_mag": attr_mag.tolist(),
            "per_ep_mse": ep_mse.tolist(),
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_binance_market_contagion(RAW_DIR, episode_bars=48)
    te_idx = np.where(data.month_ids == 8)[0]
    idio_m = data.idiosyncratic_flags[te_idx]

    design = {
        "benchmark": "binance_16asset_financial_volatility_contagion",
        "source": "Binance Vision official 5-minute spot klines (May-Aug 2024, 16 major USDT pairs)",
        "symbols": list(data.symbols),
        "total_5min_bars": data.total_5min_bars,
        "n_train_eps_may_jun": int(np.sum(data.month_ids <= 6)),
        "n_val_eps_jul": int(np.sum(data.month_ids == 7)),
        "n_test_eps_aug": int(len(te_idx)),
        "n_test_idiosyncratic": int(np.sum(idio_m)),
        "models_full": [m[0] for m in OPEN_MODELS],
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    jobs = [(mk, fam.value, cr, uc, False) for mk, fam, cr, uc in OPEN_MODELS]
    po_models = [m for m in OPEN_MODELS if m[0] in ("memgrn_ortho", "memgrn", "local_const", "perturb_ode", "latent_ode")]
    jobs += [(mk, fam.value, cr, uc, True) for mk, fam, cr, uc in po_models]

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=17, mp_context=ctx) as pool:
        raw_out = list(pool.map(_run_fin_job, jobs))
    elapsed = time.time() - t0

    def _agg(is_po: bool, keys: list[str]) -> tuple[dict, dict]:
        summ, ep_map, attr_map = {}, {}, {}
        for mk, po_f, res in raw_out:
            if po_f == is_po and mk in keys:
                ep_map[mk] = np.asarray(res.pop("per_ep_mse"), dtype=np.float64)
                attr_map[mk] = np.asarray(res.pop("attr_mag"), dtype=np.float64)
                summ[mk] = res
        ref_a = attr_map["memgrn"]
        ref_n = float(np.linalg.norm(ref_a)) + 1e-12
        for mk in keys:
            summ[mk]["obs_asset_attr_drift_vs_memgrn"] = float(
                np.linalg.norm(attr_map[mk] - ref_a) / ref_n
            )
        ref_k = "memgrn_ortho" if is_po else "memgrn"
        ref_v = ep_map[ref_k]
        wilc = {}
        for mk in keys:
            if mk == ref_k:
                continue
            oth = ep_map[mk]
            p_a = float(wilcoxon(ref_v, oth, alternative="less").pvalue) if np.any(ref_v != oth) else 1.0
            p_i = (
                float(wilcoxon(ref_v[idio_m], oth[idio_m], alternative="less").pvalue)
                if np.any(ref_v[idio_m] != oth[idio_m])
                else 1.0
            )
            wilc[mk] = {
                "pct_reduction_all": float(100.0 * (1.0 - np.mean(ref_v) / max(1e-12, np.mean(oth)))),
                "wins_all": int(np.sum(ref_v < oth)),
                "wilcoxon_p_all": p_a,
                "pct_reduction_idio": float(
                    100.0 * (1.0 - np.mean(ref_v[idio_m]) / max(1e-12, np.mean(oth[idio_m])))
                ),
                "wilcoxon_p_idio": p_i,
            }
        return summ, wilc

    full_sum, full_wilc = _agg(False, [m[0] for m in OPEN_MODELS])
    po_sum, po_wilc = _agg(True, [m[0] for m in po_models])

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "full_16asset": full_sum,
        "full_wilcoxon_vs_memgrn": full_wilc,
        "partial_14alt_hidden_btc_eth": po_sum,
        "partial_wilcoxon_vs_memgrn_ortho": po_wilc,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 1),
                "full_16asset": full_sum,
                "partial_14alt": po_sum,
                "full_wilcoxon": full_wilc,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
