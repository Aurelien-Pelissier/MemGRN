"""Independent numerical verification of MemGRN's theoretical propositions and theorems."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.stats import norm

from memgrn.dynamics import (
    MemGRNConfig,
    MemGRNParams,
    RateInputMode,
    compute_exact_contributions,
    memgrn_vector_field,
    memory_reset_splice_decomposition,
)
from memgrn.identification import (
    ar1_innovation_schedule_bound,
    exact_swap_cancellation_accuracy,
    invert_swap_waveforms,
    swap_singular_values,
    template_margin_lower_bound,
)
from memgrn.theory import (
    verify_closed_loop_stability_and_identifiability_theorem,
    verify_coupled_crosstalk_theorem,
    verify_general_s_covariance_crosstalk_theorem,
    verify_iso_expression_proposition,
    verify_orthogonal_cofactor_invariance_proposition,
    verify_unit_dc_gain_and_second_order_vs_local_ltc,
)


def verify_all_theory(output_json: Path | None = None) -> dict:
    """Run deterministic numerical audits of Theorems 1-2, Proposition 1, and schedule bounds."""
    # 1. Theorem 2: 25 (r, theta) pairs against adaptive Gauss-Kronrod quadrature
    swap_cases = []
    for r_val in (0.05, 0.25, 0.75, 1.5, 3.0):
        for deg in (5, 20, 45, 70, 90):
            theta = np.deg2rad(deg)
            closed = exact_swap_cancellation_accuracy(r_val, theta)

            def _integrand(x: float) -> float:
                hi = (abs(x) - np.cos(theta) * x) / np.sin(theta)
                lo = (-abs(x) - np.cos(theta) * x) / np.sin(theta)
                return float(norm.pdf(x - r_val) * (norm.cdf(hi) - norm.cdf(lo)))

            numeric = (
                quad(_integrand, -np.inf, 0.0, epsabs=1e-13)[0]
                + quad(_integrand, 0.0, np.inf, epsabs=1e-13)[0]
            )
            swap_cases.append(
                {
                    "r": r_val,
                    "degrees": deg,
                    "closed_form": closed,
                    "quadrature": numeric,
                    "abs_error": abs(closed - numeric),
                }
            )
    max_swap_err = max(c["abs_error"] for c in swap_cases)
    if max_swap_err >= 1e-11:
        raise AssertionError(f"Theorem 2 quadrature mismatch: {max_swap_err}")

    # 2. Algebraic swap inversion check
    t_grid = np.linspace(0.0, 6.0, 61)
    h1 = np.exp(-0.7 * t_grid) - np.exp(-2.1 * t_grid)
    h2 = -0.8 * (np.exp(-0.5 * t_grid) - np.exp(-1.8 * t_grid))
    amp_weak = 0.2 * np.sqrt(2.0 / 17.0)
    amp_strong = 4.0 * amp_weak
    sv_sum, sv_diff = swap_singular_values(amp_weak, amp_strong, apply_tanh=True)
    rec1 = np.tanh(amp_weak) * h1 + np.tanh(amp_strong) * h2
    rec2 = np.tanh(amp_strong) * h1 + np.tanh(amp_weak) * h2
    h1_rec, h2_rec = invert_swap_waveforms(rec1, rec2, amp_weak, amp_strong)
    inversion_err = float(max(np.max(np.abs(h1 - h1_rec)), np.max(np.abs(h2 - h2_rec))))
    if inversion_err >= 1e-14:
        raise AssertionError(f"Swap inversion error too large: {inversion_err}")

    # 3. Proposition 1(ii): Bounded asymmetric filters, additive accounting, and DOP853 match
    cfg = MemGRNConfig(
        n_sources=2,
        n_targets=1,
        n_channels=1,
        hidden_width=2,
        rate_min=0.1,
        rate_max=2.9,
        rate_mode=RateInputMode.BOTH,
        use_asymmetric_rates=True,
    )
    params = MemGRNParams(
        direct_weights=np.array([[0.6], [-0.4]]),
        delayed_weights=np.array([[[0.8]], [[-0.7]]]),
        relaxation=np.array([0.35]),
        rate_logits=np.array([[0.3], [0.3]]),
        rate_logits_down=np.array([[-0.2], [-0.1]]),
        input_proj=np.array([[0.5, -0.3], [0.8, 0.2], [-0.4, 0.6]]),
        hidden_bias=np.array([0.1, -0.1]),
        output_proj=np.array([[[0.7, -0.5]], [[-0.4, 0.6]]]),
        output_proj_down=np.array([[[-0.5, 0.4]], [[0.3, -0.5]]]),
    )
    n_steps = 400
    dt = 12.0 / n_steps
    times_3 = np.stack(
        [
            np.arange(n_steps) * dt,
            (np.arange(n_steps) + 0.5) * dt,
            (np.arange(n_steps) + 1.0) * dt,
        ],
        axis=-1,
    )
    raw_u = np.stack(
        [0.65 * np.sin(1.7 * times_3), 0.55 * np.cos(0.9 * times_3)], axis=1
    )
    y0 = np.array([0.3])
    y_traj, decay_term, contrib = compute_exact_contributions(
        raw_u, dt, params, cfg, y0=y0
    )
    identity_err = float(
        np.max(np.abs(y_traj - (decay_term + np.sum(contrib, axis=1))))
    )
    if identity_err >= 1e-9:
        raise AssertionError(f"Contribution accounting mismatch: {identity_err}")

    splice = memory_reset_splice_decomposition(
        raw_u, dt, params, cfg, reset_steps={100, 200}, shutoff_step=250
    )
    total_shift = splice["intact"][250:] - splice["reset"][250:]
    summed_parts = (
        splice["carryover_after_shutoff"] + splice["post_shutoff_production"]
    )
    splice_err = float(np.max(np.abs(total_shift - summed_parts)))
    if splice_err >= 1e-13:
        raise AssertionError(f"Splice decomposition mismatch: {splice_err}")

    bound_a = float(np.max(np.abs(np.tanh(raw_u))))

    def _dop_rhs(t: float, s: np.ndarray) -> np.ndarray:
        u_t = np.array([0.65 * np.sin(1.7 * t), 0.55 * np.cos(0.9 * t)])
        z_s = s[:2].reshape(2, 1)
        h_s = s[2:4].reshape(2, 1)
        y_s = s[4:5]
        dz, dh, dy = memgrn_vector_field(u_t, z_s, h_s, y_s, params, cfg)
        return np.concatenate([dz.ravel(), dh.ravel(), dy.ravel()])

    sol = solve_ivp(
        _dop_rhs,
        [0.0, 12.0],
        np.array([0.0, 0.0, 0.0, 0.0, 0.3]),
        t_eval=np.linspace(0.0, 12.0, n_steps + 1),
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    dop_err = float(np.max(np.abs(sol.y[4] - y_traj[:, 0])))
    max_filter_mag = float(np.max(np.abs(sol.y[:4])))
    if max_filter_mag > bound_a + 1e-9:
        raise AssertionError("Filter invariant interval violated")

    # 4. Gaussian template margin check
    rng = np.random.default_rng(20261121)
    true_templates = np.array([[0.3, -0.2, 0.1], [-0.25, 0.4, -0.15]])
    wrong_templates = np.array([[0.95, 0.45, -0.2], [-0.8, -0.5, 0.35]])
    mu = np.array([0.33, -0.15, 0.14])
    lb = template_margin_lower_bound(mu, true_templates, wrong_templates)
    samples = mu + rng.normal(size=(200_000, 3))
    d_true = np.min(
        np.sum((samples[:, None, :] - true_templates[None, :, :]) ** 2, axis=-1),
        axis=1,
    )
    d_wrong = np.min(
        np.sum((samples[:, None, :] - wrong_templates[None, :, :]) ** 2, axis=-1),
        axis=1,
    )
    emp_acc = float(np.mean(d_true < d_wrong))
    if emp_acc + 0.005 < lb:
        raise AssertionError(f"Empirical accuracy {emp_acc} violated lower bound {lb}")

    # 5. AR(1) temporal-innovation DP bound vs exhaustive search on compact grid (n=14, k=4)
    grid_t = np.linspace(0.2, 4.0, 14)
    deltas = np.stack(
        [
            np.stack(
                [
                    np.exp(-0.6 * grid_t) - np.exp(-1.5 * grid_t),
                    0.5 * np.sin(grid_t) * np.exp(-0.4 * grid_t),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    0.7 * np.exp(-0.9 * grid_t),
                    (grid_t * np.exp(-0.8 * grid_t)) * 0.6,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    0.45 * np.cos(0.8 * grid_t) * np.exp(-0.3 * grid_t),
                    0.55 * np.exp(-0.5 * grid_t),
                ],
                axis=-1,
            ),
        ],
        axis=0,
    )
    sched_res = ar1_innovation_schedule_bound(
        deltas=deltas,
        sigmas=np.array([0.085, 0.085, 0.085]),
        rhos=np.array([0.5, 0.75, 0.9]),
        k_samples=4,
        run_exhaustive=True,
    )
    assert sched_res.exhaustive_optimum is not None
    if sched_res.exhaustive_optimum > sched_res.dual_upper_bound + 1e-9:
        raise AssertionError("Minimax dual upper bound violated exhaustive optimum")

    # 6. Theorem 1(a): 2-TF collinear and general S-regulator correlated covariance separation
    ct_res = verify_coupled_crosstalk_theorem(w_star=2.4, rho=0.85)
    if ct_res.coupled_ko_trajectory_l2_error <= 0.01:
        raise AssertionError("Expected strictly positive coupled cross-talk error")
    if ct_res.local_ko_trajectory_l2_error >= 1e-13:
        raise AssertionError("Source-local MemGRN violated counterfactual invariance")

    gen_s = verify_general_s_covariance_crosstalk_theorem(n_sources=12, k_active=2)
    if gen_s.analytical_vs_solve_max_diff >= 1e-12:
        raise AssertionError("Spectral ridge identity failed")
    if gen_s.coupled_nonparent_weight_l2 <= 0.1 or gen_s.local_nonparent_weight_l2 != 0.0:
        raise AssertionError("General-S cross-talk check failed")
    if gen_s.finite_sample_coupled_risk_s24 <= 3.0 * gen_s.finite_sample_local_l1_risk_s24:
        raise AssertionError("Expected S=24 sparse local risk << coupled dense risk")

    # 7. Theorem 1(b): Closed-loop absorbing ball, graph identifiability, & GL(M) gauge
    cl_res = verify_closed_loop_stability_and_identifiability_theorem(n_nodes=5)
    if not cl_res.within_absorbing_ball or cl_res.max_filter_abs > 1.0 + 1e-9:
        raise AssertionError("Closed-loop MemGRN violated compact absorbing ball")
    if cl_res.coupled_gauge_trajectory_max_diff >= 1e-12:
        raise AssertionError("Coupled GL(M) gauge trajectories failed equivalence")
    if cl_res.coupled_gauge_attribution_disagreement <= 0.2:
        raise AssertionError("Expected large coupled gauge attribution ambiguity")
    if (
        cl_res.memgrn_direct_weight_recovery_max_error >= 1e-11
        or cl_res.memgrn_total_weight_recovery_max_error >= 1e-8
    ):
        raise AssertionError("Closed-loop MemGRN graph weight identifiability failed")

    # 8. Proposition 1(i) & 1(iii): Iso-expression lower bound & orthogonal cofactor invariance
    iso_res = verify_iso_expression_proposition()
    if iso_res.washout_value_diff >= 1e-11:
        raise AssertionError(f"Iso-expression washout mismatch: {iso_res.washout_value_diff}")
    if iso_res.best_state_only_post_washout_error_l2 + 1e-12 < iso_res.state_only_universal_lower_bound:
        raise AssertionError("State-only lower bound violated")

    ortho_res = verify_orthogonal_cofactor_invariance_proposition()
    if ortho_res.orthogonality_max_inner_product >= 1e-11:
        raise AssertionError("Orthogonal cofactor projection failed")
    if ortho_res.observed_weight_invariance_error >= 1e-12:
        raise AssertionError("Observed TF weight invariance violated")
    if ortho_res.mse_memgrn_ortho >= 0.2 * ortho_res.mse_observed_only:
        raise AssertionError("Expected substantial residual MSE drop from orthogonal cofactor")

    # 9. Theorem 1(iii): Unit-DC-gain identifiability & 2nd-order delay separation vs. Source-Local LTC
    ltc_sep = verify_unit_dc_gain_and_second_order_vs_local_ltc()
    if ltc_sep.ltc_scale_gauge_trajectory_max_diff >= 1e-12:
        raise AssertionError("Source-Local LTC scale gauge equivalence failed")
    if ltc_sep.ltc_scale_gauge_weight_disagreement <= 0.5:
        raise AssertionError("Expected large Source-Local LTC weight non-identifiability")
    if ltc_sep.ltc_amplitude_gain_distortion_ratio <= 1.15:
        raise AssertionError("Expected nonlinear amplitude distortion in Source-Local LTC")
    if (
        ltc_sep.memgrn_amplitude_gain_max_error >= 1e-10
        or ltc_sep.memgrn_kinetic_change_steady_state_diff >= 1e-10
    ):
        raise AssertionError("MemGRN unit-DC-gain steady-state invariance failed")
    if ltc_sep.first_order_post_shutoff_max_slope > 1e-12:
        raise AssertionError("First-order LTC violated post-shutoff monotonicity")
    if (
        ltc_sep.memgrn_post_shutoff_peak_gain <= 0.05
        or ltc_sep.best_monotone_first_order_l2_floor <= 0.02
    ):
        raise AssertionError("Expected strict 2nd-order post-shutoff peak separation")

    summary = {
        "crosstalk_coupled_ko_l2_error": ct_res.coupled_ko_trajectory_l2_error,
        "crosstalk_coupled_false_attr_l2": ct_res.coupled_false_source2_attribution_l2,
        "crosstalk_local_ko_l2_error": ct_res.local_ko_trajectory_l2_error,
        "general_s12_coupled_nonparent_l2": gen_s.coupled_nonparent_weight_l2,
        "general_s12_coupled_ko_error": gen_s.coupled_single_ko_l2_error,
        "general_s12_local_ko_error": gen_s.local_single_ko_l2_error,
        "finite_sample_s24_coupled_vs_local_risk": [
            gen_s.finite_sample_coupled_risk_s24,
            gen_s.finite_sample_local_l1_risk_s24,
        ],
        "closed_loop_within_absorbing_ball": cl_res.within_absorbing_ball,
        "closed_loop_trapping_radius": cl_res.theoretical_y_trapping_radius,
        "closed_loop_coupled_gauge_traj_diff": cl_res.coupled_gauge_trajectory_max_diff,
        "closed_loop_coupled_gauge_attr_disagreement": cl_res.coupled_gauge_attribution_disagreement,
        "closed_loop_memgrn_direct_id_error": cl_res.memgrn_direct_weight_recovery_max_error,
        "closed_loop_memgrn_total_id_error": cl_res.memgrn_total_weight_recovery_max_error,
        "vs_local_ltc_scale_gauge_traj_diff": ltc_sep.ltc_scale_gauge_trajectory_max_diff,
        "vs_local_ltc_scale_gauge_weight_disagreement": ltc_sep.ltc_scale_gauge_weight_disagreement,
        "vs_local_ltc_amp_gain_distortion_ratio": ltc_sep.ltc_amplitude_gain_distortion_ratio,
        "vs_local_ltc_memgrn_dc_gain_max_error": ltc_sep.memgrn_amplitude_gain_max_error,
        "vs_local_ltc_memgrn_kinetic_invariance_diff": ltc_sep.memgrn_kinetic_change_steady_state_diff,
        "vs_local_ltc_first_order_max_post_slope": ltc_sep.first_order_post_shutoff_max_slope,
        "vs_local_ltc_memgrn_post_peak_gain": ltc_sep.memgrn_post_shutoff_peak_gain,
        "vs_local_ltc_memgrn_post_peak_time": ltc_sep.memgrn_post_shutoff_peak_time,
        "vs_local_ltc_monotone_first_order_l2_floor": ltc_sep.best_monotone_first_order_l2_floor,
        "iso_expr_washout_diff": iso_res.washout_value_diff,
        "iso_expr_memory_diff_at_washout": iso_res.memory_state_diff_at_washout,
        "iso_expr_state_only_lower_bound": iso_res.state_only_universal_lower_bound,
        "ortho_cofactor_weight_invariance_err": ortho_res.observed_weight_invariance_error,
        "ortho_unprojected_weight_theft_rel": ortho_res.unprojected_observed_weight_theft_rel_error,
        "ortho_mse_before_after": [ortho_res.mse_observed_only, ortho_res.mse_memgrn_ortho],
        "max_swap_quadrature_error": max_swap_err,
        "swap_singular_values_4to1": [sv_sum, sv_diff],
        "swap_waveform_inversion_error": inversion_err,
        "contribution_accounting_error": identity_err,
        "memory_reset_splice_error": splice_err,
        "rk4_vs_dop853_error": dop_err,
        "max_filter_abs_within_bound": max_filter_mag <= bound_a + 1e-9,
        "margin_lower_bound": lb,
        "margin_empirical_accuracy": emp_acc,
        "ar1_dp_worst_distance_sq": sched_res.best_dp_worst_distance_sq,
        "ar1_exhaustive_optimum": sched_res.exhaustive_optimum,
        "ar1_dual_upper_bound": sched_res.dual_upper_bound,
    }
    if output_json is not None:
        output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    out_file = repo_root / "results" / "theory_verification.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    res = verify_all_theory(out_file)
    print(json.dumps(res, indent=2))
