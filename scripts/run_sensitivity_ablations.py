"""Sensitivity & Robustness Ablations: Channel Count R, Measurement Noise, & Irregular Dropout.

Runs locked robustness checks on 6 out-of-class 8-TF cooperative worlds comparing MemGRN
against Source-Local LTC and Coupled Adaptive across:
  1. Channel bank size R in {1, 2, 3, 5}.
  2. Observation noise sigma in {0.015, 0.05, 0.10, 0.15}.
  3. Irregular random training time-point dropout in {0%, 25%, 50%}.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    DynamicalModelFamily,
    ModelSpec,
    counterfactual_source_attribution,
    fit_dynamical_model,
)
from memgrn.multitf_benchmark import generate_multitf_world  # noqa: E402

OUT_DIR = ROOT / "results" / "sensitivity_ablations"


def _eval_case(job: tuple[str, str, int, str, int, float, float]) -> dict:
    axis, label, seed, fam_val, n_ch, extra_noise, drop_frac = job
    world = generate_multitf_world(seed=seed, regime="cooperative", n_sources=8, n_targets=24)
    rng = np.random.default_rng(seed * 17 + int(1000 * extra_noise) + int(100 * drop_frac))

    tr_y = world.train_y.copy()
    va_y = world.val_y.copy()
    if extra_noise > 0.015:
        add_sd = float(np.sqrt(extra_noise**2 - 0.015**2))
        tr_y[:, 1:, :] += rng.normal(0.0, add_sd, size=tr_y[:, 1:, :].shape)
        va_y[:, 1:, :] += rng.normal(0.0, add_sd, size=va_y[:, 1:, :].shape)

    tr_mask = np.ones_like(tr_y)
    tr_mask[:, 0, :] = 0.0
    if drop_frac > 0.0:
        keep = (rng.random(size=tr_y.shape[:2]) >= drop_frac).astype(np.float64)
        keep[:, 0] = 0.0
        # Guarantee at least 3 observed knots per schedule
        for b_i in range(keep.shape[0]):
            if np.sum(keep[b_i]) < 3:
                keep[b_i, [2, 6, 10]] = 1.0
        tr_mask = tr_mask * keep[:, :, None]

    va_mask = np.ones_like(va_y)
    va_mask[:, 0, :] = 0.0

    spec = ModelSpec(
        family=DynamicalModelFamily(fam_val),
        n_sources=world.n_sources,
        n_targets=world.n_targets,
        n_channels=n_ch,
        hidden_width=6,
        module_dim=14,
    )
    fitted = fit_dynamical_model(
        spec=spec,
        train_stage_u=world.train_stage_u,
        train_y=tr_y,
        train_mask=tr_mask,
        val_stage_u=world.val_stage_u,
        val_y=va_y,
        val_mask=va_mask,
        dt=world.dt,
        obs_stride=world.obs_stride,
        seeds=(101,),
        reg_grid=(1e-5, 1e-4, 1e-3),
        total_updates=900,
        lr_max=0.008,
    )
    pred = fitted.predict_fn(world.test_stage_u, None)
    test_mse = float(np.mean((pred[:, 1:, :] - world.test_y[:, 1:, :]) ** 2))
    est_eff = counterfactual_source_attribution(fitted.predict_fn, world.test_stage_u, None)
    attr_rmse = float(np.sqrt(np.mean((est_eff[:, :, 1:, :] - world.test_true_effects[:, :, 1:, :]) ** 2)))
    return {
        "axis": axis,
        "label": label,
        "seed": seed,
        "model": fam_val,
        "n_params": fitted.n_params,
        "test_mse": test_mse,
        "attribution_rmse": attr_rmse,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(6001, 6007))  # 6 cooperative worlds
    jobs: list[tuple[str, str, int, str, int, float, float]] = []

    # Axis 1: Channel count R in {1, 2, 3, 5} for MemGRN
    for r_ch in (1, 2, 3, 5):
        for s in seeds:
            jobs.append(("channels", f"R={r_ch}", s, "memgrn", r_ch, 0.015, 0.0))

    # Axis 2: Noise robustness sigma in {0.015, 0.05, 0.10, 0.15}
    for sig in (0.015, 0.05, 0.10, 0.15):
        for fam in ("memgrn", "ltc_local", "coupled_adaptive"):
            for s in seeds:
                jobs.append(("noise", f"sigma={sig:.3f}", s, fam, 3, sig, 0.0))

    # Axis 3: Irregular random time-point dropout in {0%, 25%, 50%}
    for dr in (0.0, 0.25, 0.50):
        for fam in ("memgrn", "ltc_local", "coupled_adaptive"):
            for s in seeds:
                jobs.append(("dropout", f"drop={int(100*dr)}%", s, fam, 3, 0.015, dr))

    design = {"benchmark": "sensitivity_and_robustness_ablations", "seeds": seeds, "n_jobs": len(jobs)}
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=24, mp_context=ctx) as pool:
        rows = list(pool.map(_eval_case, jobs))
    elapsed = time.time() - t0

    summary: dict[str, dict[str, dict]] = {"channels": {}, "noise": {}, "dropout": {}}
    for axis in ("channels", "noise", "dropout"):
        sub_a = [r for r in rows if r["axis"] == axis]
        keys = sorted({(r["label"], r["model"]) for r in sub_a})
        for lbl, mk in keys:
            grp = [r for r in sub_a if r["label"] == lbl and r["model"] == mk]
            mses = np.array([r["test_mse"] for r in grp])
            atrs = np.array([r["attribution_rmse"] for r in grp])
            summary[axis].setdefault(lbl, {})[mk] = {
                "n_params": grp[0]["n_params"],
                "test_mse": float(np.mean(mses)),
                "test_mse_se": float(np.std(mses, ddof=1) / np.sqrt(len(mses))),
                "attribution_rmse": float(np.mean(atrs)),
                "attribution_rmse_se": float(np.std(atrs, ddof=1) / np.sqrt(len(atrs))),
            }

    out = {"design_sha256": design_sha, "elapsed_seconds": elapsed, "summary": summary, "rows": rows}
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": round(elapsed, 1), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
