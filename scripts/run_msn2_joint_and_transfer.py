"""Fair Joint-Training, Symmetric Cross-Readout 2x2, Shared-Filter Promoter Transfer,
and Iso-Expression Non-Markovian Audit on Live-Cell Yeast Msn2 Microscopy.

Addresses reviewer fairness and network-transfer questions under a strictly source-locked
protocol (16 single-pulse schedules only for training, CV hyperparameter selection, and
readout/rate refitting; zero-shot evaluation on 30-min holdout and 10 pulse trains).
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_env_var] = "1"

import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    DynamicalModelFamily,
    ModelSpec,
    build_jax_rollout,
    fit_dynamical_model,
)
from memgrn.datasets import (  # noqa: E402
    MSN2_PROMOTERS,
    build_rk4_stage_grid,
    fetch_msn2_raw,
    load_archived_json,
    load_msn2_population_trace,
)
from memgrn.dynamics import (  # noqa: E402
    MemGRNConfig,
    MemGRNParams,
    RateInputMode,
    rollout_memgrn_rk4,
)
from memgrn.training import (  # noqa: E402
    FrozenReadoutParams,
    fit_frozen_readout_rates,
)

OUT_DIR = ROOT / "results" / "msn2_joint_and_cross_readout"
RAW_MSN2 = ROOT / "data" / "raw" / "msn2"


def _post_shutoff_train_mse(
    pred: np.ndarray,
    truth: np.ndarray,
    obs_min: np.ndarray,
    last_off: list[float],
) -> tuple[float, float, list[float]]:
    """Compute primary and first-40-min post-shutoff MSE across 10 repeated-pulse trains."""
    err = pred - truth
    full_rows = []
    f40_rows = []
    for i, stop in enumerate(last_off):
        m_full = obs_min > stop
        m_40 = (obs_min > stop) & (obs_min <= stop + 40.0)
        full_rows.append(np.mean(err[i, m_full] ** 2, axis=0))
        f40_rows.append(np.mean(err[i, m_40] ** 2, axis=0))
    full_arr = np.stack(full_rows, axis=0)  # (10, G)
    f40_arr = np.stack(f40_rows, axis=0)
    return (
        float(np.mean(full_arr)),
        float(np.mean(f40_arr)),
        np.mean(full_arr, axis=1).tolist(),
    )


def _unpack_memgrn_flat(spec: ModelSpec, theta: np.ndarray) -> dict[str, np.ndarray]:
    _, unpack, _ = build_jax_rollout(spec, dt=2.5 / (60.0 * 4), obs_stride=8)
    tree = unpack(theta)
    return {k: np.asarray(v, dtype=np.float64) for k, v in tree.items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage3 = ROOT / "results" / "stage3"

    rep_design = load_archived_json(stage3, "msn2_repeated_pulses/design.json")
    schedules = rep_design["schedules"]
    last_off = [s["n"] * 5.0 + (s["n"] - 1) * s["gap"] for s in schedules]

    # Fetch and SHA-256 verify raw Msn2 microscopy files from pinned manifests
    pinned_shas = {
        **load_archived_json(stage3, "neural_rate_msn2_transfer/inputs.json"),
        **load_archived_json(stage3, "msn2_repeated_pulses/inputs.json"),
    }
    manifest = fetch_msn2_raw(
        RAW_MSN2,
        include_repeated=True,
        repeated_schedules=schedules,
        expected_sha256=pinned_shas,
        only_held_and_repeated=True,
    )

    with np.load(stage3 / "neural_rate_msn2_transfer" / "training.npz") as tr_f:
        train_y_obs = np.asarray(tr_f["y"], dtype=np.float64)  # (16, 24, 7) at 5..120 min
        scales = np.asarray(tr_f["scales"], dtype=np.float64)
        tf_single = np.asarray(tr_f["tf"], dtype=np.float64)  # (17, 49) at 0..120 min
        train_pulses = [tuple(int(x) for x in row) for row in tr_f["train"]]
        obs_min = np.asarray(tr_f["minutes"], dtype=np.float64)

    with np.load(stage3 / "msn2_repeated_pulses" / "predictions.npz") as rep_f:
        rep_drivers = np.asarray(rep_f["drivers"], dtype=np.float64)  # (10, 49)
        archived_intact = np.asarray(rep_f["neural_rate_gw"], dtype=np.float64)

    # Load held-out 30-min single pulse and 10 repeated-pulse ground-truth YFP matrices
    held_traces = [
        load_msn2_population_trace(
            RAW_MSN2 / f"{g}_DM_30min_690nM_size.mat", "YFP_molecules", obs_min
        )[0]
        for g in MSN2_PROMOTERS
    ]
    held_y_obs = np.stack(held_traces, axis=1) / scales[None, :]  # (24, 7)

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

    # Sanity-check exact match with archived repeated-pulse score (0.02856998)
    arch_primary, _, _ = _post_shutoff_train_mse(archived_intact, rep_truth, obs_min, last_off)
    if abs(arch_primary - 0.02856998185676145) > 1e-6:
        raise AssertionError(f"Repeated-pulse ground-truth mismatch: {arch_primary}")

    design = {
        "benchmark": "msn2_joint_cross_readout_and_promoter_transfer",
        "train_schedules": 16,
        "val_split": "4_single_pulses_stratified_duration_dose_holdout",
        "held_single_pulse": [30, 690],
        "zero_shot_pulse_trains": [s["id"] for s in schedules],
        "models_joint": [
            "local_const",
            "memgrn",
            "ltc_local",
            "perturb_ode",
            "latent_ode",
        ],
        "cross_readout_sources": ["local_const", "memgrn"],
        "cross_readout_rate_modes": ["constant", "activity", "memory", "both"],
        "raw_file_count": len(manifest),
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    substeps = 4
    dt = 2.5 / (60.0 * substeps)
    obs_stride = 2 * substeps  # 5.0 min per observation

    single_stage_all = build_rk4_stage_grid(tf_single[:16], substeps_per_knot=substeps)
    held_stage = build_rk4_stage_grid(tf_single[16:17], substeps_per_knot=substeps)
    rep_stage = build_rk4_stage_grid(rep_drivers, substeps_per_knot=substeps)

    # Prepend t=0 zeros to match (batch, 25, 7) convention of fit_dynamical_model
    single_y_full = np.concatenate(
        [np.zeros((16, 1, 7), dtype=np.float64), train_y_obs], axis=1
    )
    # Stratified source validation indices across the 4x4 (duration x dose) grid (1 per duration & dose!)
    val_indices = [1, 6, 8, 15]  # (10m,275), (20m,690), (40m,100), (50m,3000)
    tr_indices = [i for i in range(16) if i not in val_indices]

    # Weight post-shutoff recovery frames 2x alongside full pulse trajectory on source splits
    single_weights = np.ones_like(single_y_full)
    single_weights[:, 0, :] = 0.0
    for i, (dur_min, _dose) in enumerate(train_pulses):
        post_mask = obs_min > dur_min
        single_weights[i, 1:, :][post_mask] = 2.0

    # =========================================================================
    # Part 1: Fair Joint Training from Scratch with Source-Only Validation
    # =========================================================================
    joint_families = (
        DynamicalModelFamily.LOCAL_CONST,
        DynamicalModelFamily.MEMGRN,
        DynamicalModelFamily.LTC_LOCAL,
        DynamicalModelFamily.PERTURB_ODE,
        DynamicalModelFamily.LATENT_ODE,
    )
    joint_results: dict[str, dict] = {}
    fitted_joint_models = {}

    for fam in joint_families:
        spec = ModelSpec(
            family=fam,
            n_sources=1,
            n_targets=7,
            n_channels=3,
            hidden_width=4,
            latent_dim=14,
            module_dim=14,
        )
        fitted = fit_dynamical_model(
            spec=spec,
            train_stage_u=single_stage_all[:, tr_indices, :, :],
            train_y=single_y_full[tr_indices],
            train_mask=single_weights[tr_indices],
            val_stage_u=single_stage_all[:, val_indices, :, :],
            val_y=single_y_full[val_indices],
            val_mask=single_weights[val_indices],
            dt=dt,
            obs_stride=obs_stride,
            seeds=(416, 819, 1207),
            reg_grid=(1e-6, 1e-5, 1e-4),
            total_updates=1800,
            lr_max=0.009,
        )
        fitted_joint_models[fam.value] = fitted

        pred_30 = fitted.predict_fn(held_stage, None)[0, 1:, :]  # (24, 7)
        held_post_mse = float(np.mean((pred_30[obs_min > 30.0] - held_y_obs[obs_min > 30.0]) ** 2))

        pred_rep = fitted.predict_fn(rep_stage, None)[:, 1:, :]  # (10, 24, 7)
        rep_mse, rep_f40, rep_by_sched = _post_shutoff_train_mse(
            pred_rep, rep_truth, obs_min, last_off
        )
        no_sip_idx = [i for i, g in enumerate(MSN2_PROMOTERS) if g != "SIP18"]
        rep_nosip, _, _ = _post_shutoff_train_mse(
            pred_rep[:, :, no_sip_idx], rep_truth[:, :, no_sip_idx], obs_min, last_off
        )

        joint_results[fam.value] = {
            "n_params": fitted.n_params,
            "val_loss": fitted.best_val_loss,
            "chosen_reg": fitted.chosen_reg,
            "held_30min_post_mse": held_post_mse,
            "ten_trains_post_mse": rep_mse,
            "ten_trains_first40_mse": rep_f40,
            "ten_trains_no_sip18_mse": rep_nosip,
            "by_schedule": rep_by_sched,
        }

    # =========================================================================
    # Part 2: Symmetric 2x2 Cross-Readout Sensitivity (Constant-W vs MemGRN-W)
    # =========================================================================
    fine_sub = 8
    dt_fine = 2.5 / (60.0 * fine_sub)
    tanh_train_stages = np.tanh(build_rk4_stage_grid(tf_single[:16], substeps_per_knot=fine_sub)[:, :, 0, :])
    rep_fine_stages = build_rk4_stage_grid(rep_drivers, substeps_per_knot=fine_sub)

    loss_w_single = np.zeros((16, 24), dtype=np.float64)
    for i, (dur_m, _d) in enumerate(train_pulses):
        mask_p = obs_min > dur_m
        loss_w_single[i] = 0.25 + 0.75 * mask_p.astype(np.float64)
    loss_w_single /= np.sum(loss_w_single)

    cross_readout: dict[str, dict[str, float]] = {}
    for readout_origin in ("local_const", "memgrn"):
        origin_model = fitted_joint_models[readout_origin]
        unpacked = _unpack_memgrn_flat(origin_model.spec, origin_model.theta)
        origin_scores: dict[str, float] = {}

        for mode_str in ("constant", "activity", "memory", "both"):
            mode = RateInputMode(mode_str)
            per_gene_mses = []
            for g_i in range(len(MSN2_PROMOTERS)):
                froz = FrozenReadoutParams(
                    direct=float(unpacked["direct"][0, g_i]),
                    delayed=unpacked["delayed"][0, :, g_i],
                    decay=float(0.1 + np.logaddexp(0.0, unpacked["decay"][g_i])),
                )
                fit_res = fit_frozen_readout_rates(
                    mode=mode,
                    frozen=froz,
                    tanh_stages=tanh_train_stages,
                    targets=train_y_obs[:, :, g_i],
                    loss_weights=loss_w_single,
                    seed=416 + g_i,
                    total_updates=1600,
                    substeps_per_knot=fine_sub,
                )
                # Evaluate fitted rate law on 10 unseen repeated-pulse trains
                th_r = fit_res.best_theta
                if mode == RateInputMode.CONSTANT:
                    r_p = MemGRNParams(
                        direct_weights=np.array([[froz.direct]]),
                        delayed_weights=froz.delayed.reshape(1, 3, 1),
                        relaxation=np.array([froz.decay]),
                        rate_logits=th_r.reshape(1, 3).astype(np.float64),
                    )
                else:
                    n_in = 1 if mode == RateInputMode.ACTIVITY else (2 if mode == RateInputMode.MEMORY else 3)
                    idx = 0
                    b_v = th_r[idx : idx + 4].astype(np.float64)
                    idx += 4
                    in_v = th_r[idx : idx + n_in * 4].reshape(n_in, 4).astype(np.float64)
                    idx += n_in * 4
                    out_v = th_r[idx : idx + 12].reshape(1, 3, 4).astype(np.float64)
                    idx += 12
                    rt_v = th_r[idx : idx + 3].reshape(1, 3).astype(np.float64)
                    r_p = MemGRNParams(
                        direct_weights=np.array([[froz.direct]]),
                        delayed_weights=froz.delayed.reshape(1, 3, 1),
                        relaxation=np.array([froz.decay]),
                        rate_logits=rt_v,
                        input_proj=in_v,
                        hidden_bias=b_v,
                        output_proj=out_v,
                    )
                cfg_g = MemGRNConfig(
                    n_sources=1, n_targets=1, n_channels=3, hidden_width=4, rate_mode=mode
                )
                _, _, y_sim = rollout_memgrn_rk4(rep_fine_stages, dt_fine, r_p, cfg_g)
                pred_g = y_sim[16::16, :, 0].T[:, :, None]
                mse_g, _, _ = _post_shutoff_train_mse(
                    pred_g, rep_truth[:, :, g_i : g_i + 1], obs_min, last_off
                )
                per_gene_mses.append(mse_g)
            origin_scores[mode_str] = float(np.mean(per_gene_mses))
        cross_readout[readout_origin] = origin_scores

    # =========================================================================
    # Part 3: Leave-One-Promoter-Out Shared Msn2 Filter Transfer (7 folds)
    # =========================================================================
    shared_transfer_by_gene: dict[str, dict[str, float]] = {}
    for held_g_idx, held_gene in enumerate(MSN2_PROMOTERS):
        src_genes = [i for i in range(7) if i != held_g_idx]
        spec_6 = ModelSpec(
            family=DynamicalModelFamily.MEMGRN,
            n_sources=1,
            n_targets=6,
            n_channels=3,
            hidden_width=4,
        )
        fit_6 = fit_dynamical_model(
            spec=spec_6,
            train_stage_u=single_stage_all[:, tr_indices, :, :],
            train_y=single_y_full[tr_indices][:, :, src_genes],
            train_mask=single_weights[tr_indices][:, :, src_genes],
            val_stage_u=single_stage_all[:, val_indices, :, :],
            val_y=single_y_full[val_indices][:, :, src_genes],
            val_mask=single_weights[val_indices][:, :, src_genes],
            dt=dt,
            obs_stride=obs_stride,
            seeds=(416, 819),
            reg_grid=(1e-5, 1e-4),
            total_updates=1400,
        )
        p6 = _unpack_memgrn_flat(spec_6, fit_6.theta)

        # Roll out the frozen shared 3-channel Msn2 filter bank (z, h) on single & repeated pulses
        dummy_cfg = MemGRNConfig(n_sources=1, n_targets=1, n_channels=3, hidden_width=4)
        dummy_params = MemGRNParams(
            direct_weights=np.zeros((1, 1)),
            delayed_weights=np.zeros((1, 3, 1)),
            relaxation=np.array([0.5]),
            rate_logits=p6["rate"],
            input_proj=p6["in_proj"],
            hidden_bias=p6["bias"],
            output_proj=p6["out_proj"],
        )
        _, h_single, _ = rollout_memgrn_rk4(single_stage_all, dt, dummy_params, dummy_cfg)
        _, h_rep, _ = rollout_memgrn_rk4(rep_stage, dt, dummy_params, dummy_cfg)
        a_single = np.tanh(single_stage_all[:, :, 0, :])
        a_rep = np.tanh(rep_stage[:, :, 0, :])

        # Fit ONLY the 5 readout scalars (D, B_1, B_2, B_3, log_decay) on held-out promoter's single pulses!
        y_target_single = train_y_obs[:, :, held_g_idx]  # (16, 24)

        def _integrate_linear_readout(
            a_st: np.ndarray, h_traj: np.ndarray, w_vec: np.ndarray, dec: float
        ) -> np.ndarray:
            # Exact step-wise RK4 integration of dy/dt = d*a + h @ b - dec*y
            d_w = w_vec[0]
            b_w = w_vec[1:4]
            n_st, b_sz = a_st.shape[:2]
            y_cur = np.zeros((b_sz,), dtype=np.float64)
            out_y = []
            for st in range(n_st):
                hm = 0.5 * (h_traj[st, :, 0, :] + h_traj[st + 1, :, 0, :])
                p0 = d_w * a_st[st, :, 0] + h_traj[st, :, 0, :] @ b_w
                pm = d_w * a_st[st, :, 1] + hm @ b_w
                p1 = d_w * a_st[st, :, 2] + h_traj[st + 1, :, 0, :] @ b_w
                ky1 = p0 - dec * y_cur
                ky2 = pm - dec * (y_cur + 0.5 * dt * ky1)
                ky3 = pm - dec * (y_cur + 0.5 * dt * ky2)
                ky4 = p1 - dec * (y_cur + dt * ky3)
                y_cur = y_cur + (dt / 6.0) * (ky1 + 2 * ky2 + 2 * ky3 + ky4)
                if (st + 1) % obs_stride == 0:
                    out_y.append(y_cur.copy())
            return np.stack(out_y, axis=1)

        def _resid(x_opt: np.ndarray) -> np.ndarray:
            dec = 0.1 + np.logaddexp(0.0, x_opt[4])
            pred = _integrate_linear_readout(a_single, h_single, x_opt[:4], dec)
            return ((pred - y_target_single) * np.sqrt(single_weights[:, 1:, held_g_idx])).ravel()

        lsq = least_squares(_resid, np.zeros(5, dtype=np.float64), method="trf")
        dec_fit = float(0.1 + np.logaddexp(0.0, lsq.x[4]))
        pred_rep_shared = _integrate_linear_readout(a_rep, h_rep, lsq.x[:4], dec_fit)[:, :, None]

        # Compare against Constant-filter LOPO transfer (fixing constant rates from 6 promoters)
        spec_6c = ModelSpec(
            family=DynamicalModelFamily.LOCAL_CONST,
            n_sources=1,
            n_targets=6,
            n_channels=3,
        )
        fit_6c = fit_dynamical_model(
            spec=spec_6c,
            train_stage_u=single_stage_all[:, tr_indices, :, :],
            train_y=single_y_full[tr_indices][:, :, src_genes],
            train_mask=single_weights[tr_indices][:, :, src_genes],
            val_stage_u=single_stage_all[:, val_indices, :, :],
            val_y=single_y_full[val_indices][:, :, src_genes],
            val_mask=single_weights[val_indices][:, :, src_genes],
            dt=dt,
            obs_stride=obs_stride,
            seeds=(416, 819),
            reg_grid=(1e-5, 1e-4),
            total_updates=1400,
        )
        p6c = _unpack_memgrn_flat(spec_6c, fit_6c.theta)
        cfg_c = MemGRNConfig(
            n_sources=1, n_targets=1, n_channels=3, rate_mode=RateInputMode.CONSTANT
        )
        params_c = MemGRNParams(
            direct_weights=np.zeros((1, 1)),
            delayed_weights=np.zeros((1, 3, 1)),
            relaxation=np.array([0.5]),
            rate_logits=p6c["rate"],
        )
        _, h_single_c, _ = rollout_memgrn_rk4(single_stage_all, dt, params_c, cfg_c)
        _, h_rep_c, _ = rollout_memgrn_rk4(rep_stage, dt, params_c, cfg_c)
        lsq_c = least_squares(
            lambda x_o: (
                (
                    _integrate_linear_readout(
                        a_single, h_single_c, x_o[:4], float(0.1 + np.logaddexp(0.0, x_o[4]))
                    )
                    - y_target_single
                )
                * np.sqrt(single_weights[:, 1:, held_g_idx])
            ).ravel(),
            np.zeros(5, dtype=np.float64),
            method="trf",
        )
        pred_rep_const_shared = _integrate_linear_readout(
            a_rep, h_rep_c, lsq_c.x[:4], float(0.1 + np.logaddexp(0.0, lsq_c.x[4]))
        )[:, :, None]

        mse_adapt_lopo, _, _ = _post_shutoff_train_mse(
            pred_rep_shared, rep_truth[:, :, held_g_idx : held_g_idx + 1], obs_min, last_off
        )
        mse_const_lopo, _, _ = _post_shutoff_train_mse(
            pred_rep_const_shared, rep_truth[:, :, held_g_idx : held_g_idx + 1], obs_min, last_off
        )
        shared_transfer_by_gene[held_gene] = {
            "shared_memgrn_filter_mse": mse_adapt_lopo,
            "shared_const_filter_mse": mse_const_lopo,
        }

    # =========================================================================
    # Part 4: Empirical Iso-Expression Non-Markovian Audit on Real Yeast Pulses
    # =========================================================================
    # Compare pairs of single pulses with different durations/doses whose measured YFP at
    # pulse shutoff matches closely (|y_1(T_1) - y_2(T_2)| < 0.12), and compare post-washout
    # trajectories predicted by Joint MemGRN vs. Joint State-Only PerturbODE!
    pred_single_mem = fitted_joint_models["memgrn"].predict_fn(single_stage_all, None)[:, 1:, :]
    pred_single_pode = fitted_joint_models["perturb_ode"].predict_fn(single_stage_all, None)[:, 1:, :]

    iso_pairs = []
    for g_i, g_name in enumerate(MSN2_PROMOTERS):
        for i in range(16):
            dur_i, dose_i = train_pulses[i]
            idx_off_i = int(np.where(obs_min == dur_i)[0][0])
            y_off_i = float(train_y_obs[i, idx_off_i, g_i])
            if y_off_i < 0.35:
                continue
            for k in range(i + 1, 16):
                dur_k, dose_k = train_pulses[k]
                if dur_i == dur_k:
                    continue
                idx_off_k = int(np.where(obs_min == dur_k)[0][0])
                y_off_k = float(train_y_obs[k, idx_off_k, g_i])
                if abs(y_off_i - y_off_k) <= 0.10:
                    # Compare post-shutoff 40-min trajectory divergence (8 frames = 40 min)
                    tail_i = train_y_obs[i, idx_off_i + 1 : idx_off_i + 9, g_i]
                    tail_k = train_y_obs[k, idx_off_k + 1 : idx_off_k + 9, g_i]
                    if len(tail_i) == 8 and len(tail_k) == 8:
                        emp_div = float(np.mean((tail_i - tail_k) ** 2))
                        mem_mse = 0.5 * float(
                            np.mean((pred_single_mem[i, idx_off_i + 1 : idx_off_i + 9, g_i] - tail_i) ** 2)
                            + np.mean((pred_single_mem[k, idx_off_k + 1 : idx_off_k + 9, g_i] - tail_k) ** 2)
                        )
                        pode_mse = 0.5 * float(
                            np.mean((pred_single_pode[i, idx_off_i + 1 : idx_off_i + 9, g_i] - tail_i) ** 2)
                            + np.mean((pred_single_pode[k, idx_off_k + 1 : idx_off_k + 9, g_i] - tail_k) ** 2)
                        )
                        iso_pairs.append(
                            {
                                "promoter": g_name,
                                "pulse_1": [dur_i, dose_i],
                                "pulse_2": [dur_k, dose_k],
                                "washout_y_gap": abs(y_off_i - y_off_k),
                                "empirical_tail_divergence_mse": emp_div,
                                "memgrn_pair_tail_mse": mem_mse,
                                "perturb_ode_pair_tail_mse": pode_mse,
                            }
                        )

    elapsed = time.time() - t0
    lopo_adapt_mean = float(
        np.mean([v["shared_memgrn_filter_mse"] for v in shared_transfer_by_gene.values()])
    )
    lopo_const_mean = float(
        np.mean([v["shared_const_filter_mse"] for v in shared_transfer_by_gene.values()])
    )

    payload = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "joint_training": joint_results,
        "cross_readout_2x2": cross_readout,
        "shared_filter_lopo_transfer": {
            "mean_memgrn_shared": lopo_adapt_mean,
            "mean_const_shared": lopo_const_mean,
            "pct_reduction": 100.0 * (1.0 - lopo_adapt_mean / lopo_const_mean),
            "wins_memgrn": int(
                sum(
                    v["shared_memgrn_filter_mse"] < v["shared_const_filter_mse"]
                    for v in shared_transfer_by_gene.values()
                )
            ),
            "by_promoter": shared_transfer_by_gene,
        },
        "iso_expression_audit": {
            "n_matched_pulse_pairs": len(iso_pairs),
            "mean_washout_y_gap": float(np.mean([p["washout_y_gap"] for p in iso_pairs])),
            "mean_empirical_tail_divergence_mse": float(
                np.mean([p["empirical_tail_divergence_mse"] for p in iso_pairs])
            ),
            "mean_memgrn_tail_mse": float(np.mean([p["memgrn_pair_tail_mse"] for p in iso_pairs])),
            "mean_perturb_ode_tail_mse": float(
                np.mean([p["perturb_ode_pair_tail_mse"] for p in iso_pairs])
            ),
            "pairs": iso_pairs,
        },
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "design_sha256": design_sha,
                "elapsed_s": round(elapsed, 1),
                "joint_ten_trains_mse": {
                    k: round(v["ten_trains_post_mse"], 5) for k, v in joint_results.items()
                },
                "cross_readout_2x2": cross_readout,
                "shared_lopo_mean": {
                    "memgrn": round(lopo_adapt_mean, 5),
                    "const": round(lopo_const_mean, 5),
                },
                "iso_expression": {
                    "n_pairs": len(iso_pairs),
                    "memgrn_mse": round(payload["iso_expression_audit"]["mean_memgrn_tail_mse"], 5),
                    "perturb_ode_mse": round(
                        payload["iso_expression_audit"]["mean_perturb_ode_tail_mse"], 5
                    ),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
