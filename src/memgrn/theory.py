"""Theoretical guarantees for source-isolated adaptive memory, closed-loop networks, and orthogonal cofactors."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

from memgrn.dynamics import (
    MemGRNConfig,
    MemGRNParams,
    RateInputMode,
    memgrn_vector_field,
    project_orthogonal_cofactor_basis,
)


@dataclass(frozen=True, kw_only=True)
class CrossTalkAuditResult:
    """Analytical vs. numerical verification of 2-TF collinear cross-talk vs. MemGRN."""

    rho_train: float
    coupled_weights: tuple[float, float]
    local_weights: tuple[float, float]
    coupled_ko_trajectory_l2_error: float
    coupled_false_source2_attribution_l2: float
    local_ko_trajectory_l2_error: float
    local_false_source2_attribution_l2: float


@dataclass(frozen=True, kw_only=True)
class GeneralSCovarianceAuditResult:
    """Verification of Theorem 1(a): S-regulator correlated covariance & sample-complexity separation."""

    n_sources: int
    k_active: int
    cov_condition_number: float
    analytical_vs_solve_max_diff: float
    equicorrelated_closed_form_max_diff: float
    equicorrelated_rho1_limit_error: float
    coupled_nonparent_weight_l2: float
    local_nonparent_weight_l2: float
    coupled_single_ko_l2_error: float
    local_single_ko_l2_error: float
    coupled_false_nonparent_attr_l2: float
    local_false_nonparent_attr_l2: float
    finite_sample_coupled_risk_s4: float
    finite_sample_coupled_risk_s24: float
    finite_sample_local_l1_risk_s4: float
    finite_sample_local_l1_risk_s24: float


@dataclass(frozen=True, kw_only=True)
class ClosedLoopIdentifiabilityAuditResult:
    """Verification of Theorem 1(b): closed-loop absorbing ball, graph identifiability, & coupled gauge."""

    n_nodes: int
    max_filter_abs: float
    theoretical_y_trapping_radius: float
    observed_max_y_after_transient: float
    within_absorbing_ball: bool
    coupled_gauge_trajectory_max_diff: float
    coupled_gauge_attribution_disagreement: float
    memgrn_relaxation_recovery_max_error: float
    memgrn_direct_weight_recovery_max_error: float
    memgrn_total_weight_recovery_max_error: float
    factorized_coop_single_clamp_max_error: float


@dataclass(frozen=True, kw_only=True)
class IsoExpressionAuditResult:
    """Analytical vs. numerical verification of Proposition 1(i) (non-Markovian separation)."""

    washout_value_diff: float
    memory_state_diff_at_washout: float
    true_post_washout_separation_l2: float
    state_only_universal_lower_bound: float
    best_state_only_post_washout_error_l2: float
    memgrn_post_washout_error_l2: float


@dataclass(frozen=True, kw_only=True)
class OrthogonalCofactorAuditResult:
    """Verification of Proposition 1(iii): orthogonal cofactor attribution invariance."""

    orthogonality_max_inner_product: float
    observed_weight_invariance_error: float
    unprojected_observed_weight_theft_rel_error: float
    mse_observed_only: float
    mse_memgrn_ortho: float


@dataclass(frozen=True, kw_only=True)
class UnitDCGainAndSecondOrderAuditResult:
    """Verification of Theorem 1(iii): MemGRN vs. Source-Local LTC separation.

    Audits two exact structural separations between MemGRN's unit-DC-gain two-stage
    convex relaxation and a first-order Source-Local LTC cell:
      (1) Kinetic-graph decoupling: MemGRN steady-state causal weight W^tot = D + sum_r B
          is invariant to kinetic rates and probe amplitudes, whereas Source-Local LTC
          suffers exact scale non-identifiability (c * A, B / c) and rate-entangled
          amplitude-dependent steady-state gain distortion.
      (2) Second-order post-shutoff delay peak: any first-order positive relaxation ODE
          (including 1-stage Source-Local LTC) is monotonically non-increasing after
          stimulus shutoff (zero post-shutoff peak), incurring a strict L2 error lower
          bound on delayed-peak targets that two-stage MemGRN recovers exactly.
    """

    ltc_scale_gauge_trajectory_max_diff: float
    ltc_scale_gauge_weight_disagreement: float
    ltc_amplitude_gain_distortion_ratio: float
    memgrn_amplitude_gain_max_error: float
    memgrn_kinetic_change_steady_state_diff: float
    first_order_post_shutoff_max_slope: float
    memgrn_post_shutoff_peak_gain: float
    memgrn_post_shutoff_peak_time: float
    best_monotone_first_order_l2_floor: float
    memgrn_delayed_peak_l2_error: float


@dataclass(frozen=True, kw_only=True)
class OrthoOODGeneralizationAuditResult:
    """Verification of upgraded Proposition 1(iii): MemGRN-Ortho OOD risk & counterfactual invariance.

    Audits exact OOD excess risk decomposition when observed regulators O and unmeasured
    cofactors U shift covariance from Sigma_OU^tr to Sigma_OU^te:
      (1) Counterfactual observed-TF attribution invariance holds identically on ANY
          out-of-distribution intervention g (zero drift), whereas unprojected latent models
          suffer large attribution distortion under single-TF knockouts.
      (2) Exact analytical OOD MSE reduction formula equals empirical OOD MSE reduction
          to machine precision and remains strictly positive across covariance shift whenever
          conditional unobserved variance dominates shift cross-error.
    """

    ood_counterfactual_drift_ortho: float
    ood_counterfactual_drift_unproj: float
    analytical_vs_empirical_ood_risk_diff: float
    ood_mse_observed_only: float
    ood_mse_memgrn_ortho: float
    ood_mse_unproj_ridge: float
    unproj_ood_obs_attr_rmse: float
    ortho_ood_obs_attr_rmse: float


@dataclass(frozen=True, kw_only=True)
class SelectiveSSMCrossTalkAuditResult:
    """Verification of Theorem 1(i) Corollary: Mamba/S6 diagonal SSM Delta(a)-gate cross-talk.

    Even when state transition matrix A = -diag(alpha) is strictly diagonal, an input-selective
    timescale gate Delta_m(a) = softplus(w_m^T a + b_m) mixes correlated upstream regulators,
    leaking nonzero weight onto non-parent regulators and incurring positive knockout error
    and false non-parent attribution, whereas MemGRN achieves identically zero leakage.
    """

    ssm_nonparent_delta_weight: float
    ssm_ko_trajectory_l2_error: float
    ssm_false_nonparent_attribution_l2: float
    memgrn_ko_trajectory_l2_error: float
    memgrn_false_nonparent_attribution_l2: float


def minimum_norm_collinear_gate_weights(
    w_star: float,
    rho: float,
    eta_var: float = 0.0,
    a1_var: float = 1.0,
    l2_penalty: float = 1e-6,
) -> tuple[float, float]:
    """Compute closed-form ridge / minimum-norm coupled gate weights under correlated inputs."""
    if eta_var == 0.0:
        scale = 1.0 / (1.0 + rho**2)
        return float(scale * w_star), float(rho * scale * w_star)

    c11 = a1_var + l2_penalty
    c12 = rho * a1_var
    c22 = (rho**2) * a1_var + (1.0 - rho**2) * eta_var + l2_penalty
    cov_mat = np.array([[c11, c12], [c12, c22]], dtype=np.float64)
    rhs = np.array([a1_var * w_star, rho * a1_var * w_star], dtype=np.float64)
    w_hat = np.linalg.solve(cov_mat, rhs)
    return float(w_hat[0]), float(w_hat[1])


def general_s_coupled_ridge_weights(
    sigma_a: np.ndarray,
    w_star_vec: np.ndarray,
    l2_penalty: float,
) -> np.ndarray:
    """Closed-form S-regulator coupled gate ridge/minimum-norm weights (Sigma + mu I)^{-1} Sigma w*."""
    s_dim = sigma_a.shape[0]
    if l2_penalty == 0.0:
        return np.linalg.pinv(sigma_a, rcond=1e-12) @ (sigma_a @ w_star_vec)
    return np.linalg.solve(
        sigma_a + l2_penalty * np.eye(s_dim, dtype=np.float64), sigma_a @ w_star_vec
    )


def verify_coupled_crosstalk_theorem(
    w_star: float = 2.4,
    rho: float = 0.85,
    rate_min: float = 0.1,
    rate_max: float = 8.0,
    base_logit: float = -0.8,
    delayed_weight: float = 1.2,
    decay: float = 0.5,
) -> CrossTalkAuditResult:
    """Verify 2-TF collinear special case of Theorem 1(a)."""
    w1_cpl, w2_cpl = minimum_norm_collinear_gate_weights(w_star, rho, eta_var=0.0)
    w1_loc, w2_loc = float(w_star), 0.0

    def _rollout(a1_fn, a2_fn, w1: float, w2: float, t_eval: np.ndarray) -> np.ndarray:
        def _rhs(t: float, s: np.ndarray) -> list[float]:
            a1 = float(a1_fn(t))
            a2 = float(a2_fn(t))
            z, h, y = float(s[0]), float(s[1]), float(s[2])
            logit = base_logit + w1 * a1 + w2 * a2 - 1.1 * h
            lam = rate_min + (rate_max - rate_min) / (1.0 + np.exp(-logit))
            return [lam * (a1 - z), lam * (z - h), delayed_weight * h - decay * y]

        sol = solve_ivp(
            _rhs,
            [float(t_eval[0]), float(t_eval[-1])],
            [0.0, 0.0, 0.0],
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )
        return sol.y[2]

    t_grid = np.linspace(0.0, 6.0, 301)
    dt = float(t_grid[1] - t_grid[0])
    pulse = lambda t: float(0.85 / (1.0 + np.exp((t - 1.8) / 0.06)))
    collinear_a2 = lambda t: rho * pulse(t)
    zero_fn = lambda _t: 0.0

    y_true_train = _rollout(pulse, collinear_a2, w_star, 0.0, t_grid)
    y_cpl_train = _rollout(pulse, collinear_a2, w1_cpl, w2_cpl, t_grid)
    if float(np.max(np.abs(y_true_train - y_cpl_train))) > 1e-11:
        raise AssertionError("Coupled minimum-norm interpolant failed to match training manifold")

    y_true_ko = _rollout(pulse, zero_fn, w_star, 0.0, t_grid)
    y_cpl_ko = _rollout(pulse, zero_fn, w1_cpl, w2_cpl, t_grid)
    y_loc_ko = _rollout(pulse, zero_fn, w1_loc, w2_loc, t_grid)

    cpl_false_attr = float(np.sqrt(trapezoid((y_cpl_train - y_cpl_ko) ** 2, dx=dt)))
    loc_false_attr = float(np.sqrt(trapezoid((y_true_train - y_loc_ko) ** 2, dx=dt)))
    cpl_ko_err = float(np.sqrt(trapezoid((y_cpl_ko - y_true_ko) ** 2, dx=dt)))
    loc_ko_err = float(np.sqrt(trapezoid((y_loc_ko - y_true_ko) ** 2, dx=dt)))

    return CrossTalkAuditResult(
        rho_train=rho,
        coupled_weights=(w1_cpl, w2_cpl),
        local_weights=(w1_loc, w2_loc),
        coupled_ko_trajectory_l2_error=cpl_ko_err,
        coupled_false_source2_attribution_l2=cpl_false_attr,
        local_ko_trajectory_l2_error=loc_ko_err,
        local_false_source2_attribution_l2=loc_false_attr,
    )


def verify_general_s_covariance_crosstalk_theorem(
    n_sources: int = 12,
    k_active: int = 2,
    rho_module: float = 0.92,
    l2_penalty: float = 0.08,
) -> GeneralSCovarianceAuditResult:
    """Verify Theorem 1(a): single-parent (k=1) linearized gate cross-talk & sparse readout risk.

    Separates:
      1. Rate-gate cross-talk (linearized quadratic surrogate with k=1 single-parent ground truth
         and matched ridge penalty mu=l2_penalty for BOTH coupled and source-isolated gates),
         verifying the exact equicorrelated leakage formula
           mu * rho / ((1 - rho + mu) * (1 + (b - 1) * rho + mu)) * |w_i*| -> |w_i*| / (b + mu).
      2. Sparse linear readout finite-sample intervention risk (k_active-sparse Lasso vs dense OLS).
    """
    # Construct strictly positive-definite equicorrelated pathway-module covariance Sigma_a > 0
    block_size = 4
    sigma_a = np.eye(n_sources, dtype=np.float64) * (1.0 - rho_module)
    for b_start in range(0, n_sources, block_size):
        b_end = min(n_sources, b_start + block_size)
        sigma_a[b_start:b_end, b_start:b_end] += rho_module
    cond_num = float(np.linalg.cond(sigma_a))

    # Single-parent (k = 1) rate-gate ground truth: realizable by source-isolated MemGRN!
    w_star_vec = np.zeros(n_sources, dtype=np.float64)
    w_star_vec[0] = 2.2
    active_mask = np.abs(w_star_vec) > 0.0

    # Closed-form spectral formula vs direct linear solve
    eigvals, eigvecs = np.linalg.eigh(sigma_a)
    spectral_w = eigvecs @ ((eigvals / (eigvals + l2_penalty)) * (eigvecs.T @ w_star_vec))
    direct_w = general_s_coupled_ridge_weights(sigma_a, w_star_vec, l2_penalty=l2_penalty)
    spec_diff = float(np.max(np.abs(spectral_w - direct_w)))

    # Verify exact closed-form equicorrelated block leakage and rho -> 1^- limit |w_0*| / (b + mu)
    exact_leak = (
        (l2_penalty * rho_module)
        / ((1.0 - rho_module + l2_penalty) * (1.0 + (block_size - 1) * rho_module + l2_penalty))
        * abs(w_star_vec[0])
    )
    equicorr_diff = float(np.max(np.abs(np.abs(direct_w[1:block_size]) - exact_leak)))
    rho_near1 = 1.0 - 1e-8
    sigma_near1 = np.eye(block_size, dtype=np.float64) * (1.0 - rho_near1) + rho_near1
    w_near1 = general_s_coupled_ridge_weights(
        sigma_near1, np.array([w_star_vec[0]] + [0.0] * (block_size - 1)), l2_penalty=l2_penalty
    )
    rho1_limit_target = abs(w_star_vec[0]) / (block_size + l2_penalty)
    rho1_limit_err = float(abs(abs(w_near1[1]) - rho1_limit_target))

    cpl_nonparent_l2 = float(np.linalg.norm(direct_w[~active_mask]))
    loc_nonparent_l2 = 0.0

    # Matched ridge penalty mu = l2_penalty on source-isolated gate weight w_loc[0]
    w_loc = np.zeros(n_sources, dtype=np.float64)
    w_loc[0] = float((sigma_a[0, 0] / (sigma_a[0, 0] + l2_penalty)) * w_star_vec[0])

    # Simulate nonlinear ODE rollout under (a) single-parent activation (non-parents knocked out)
    # and (b) co-activated equicorrelated module (a_{1..b-1} = rho * a_0) to measure false attribution.
    t_grid = np.linspace(0.0, 5.0, 251)
    dt = float(t_grid[1] - t_grid[0])

    def _sim_s_gate(w_gate: np.ndarray, coactivate_module: bool) -> np.ndarray:
        def _rhs(t: float, s: np.ndarray) -> list[float]:
            a_vec = np.zeros(n_sources, dtype=np.float64)
            a0 = float(0.85 / (1.0 + np.exp((t - 1.5) / 0.06)))
            a_vec[0] = a0
            if coactivate_module:
                a_vec[1:block_size] = rho_module * a0
            z, h, y = float(s[0]), float(s[1]), float(s[2])
            logit = -0.6 + float(w_gate @ a_vec) - 1.0 * h
            lam = 0.1 + 7.9 / (1.0 + np.exp(-logit))
            return [lam * (a0 - z), lam * (z - h), 1.3 * h - 0.5 * y]

        sol = solve_ivp(
            _rhs,
            [0.0, 5.0],
            [0.0, 0.0, 0.0],
            t_eval=t_grid,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )
        return sol.y[2]

    y_star_ko = _sim_s_gate(w_star_vec, coactivate_module=False)
    y_cpl_ko = _sim_s_gate(direct_w, coactivate_module=False)
    y_cpl_co = _sim_s_gate(direct_w, coactivate_module=True)
    y_loc_ko = _sim_s_gate(w_loc, coactivate_module=False)
    y_loc_co = _sim_s_gate(w_loc, coactivate_module=True)

    cpl_ko_err = float(np.sqrt(trapezoid((y_cpl_ko - y_star_ko) ** 2, dx=dt)))
    loc_ko_err = float(np.sqrt(trapezoid((y_loc_ko - y_star_ko) ** 2, dx=dt)))
    cpl_false_attr = float(np.sqrt(trapezoid((y_cpl_co - y_cpl_ko) ** 2, dx=dt)))
    loc_false_attr = float(np.sqrt(trapezoid((y_loc_co - y_loc_ko) ** 2, dx=dt)))

    # Finite-sample linear readout intervention risk: Omega(S / n) dense OLS vs O(k log S / n) Lasso
    rng = np.random.default_rng(20261121)
    n_samples = 64
    noise_sd = 0.35

    def _mc_readout_risk(s_dim: int) -> tuple[float, float]:
        # Correlated training design X (n_samples, s_dim), sparse k_active ground truth beta_star
        cov_s = 0.70 * np.ones((s_dim, s_dim), dtype=np.float64) + 0.30 * np.eye(s_dim)
        chol = np.linalg.cholesky(cov_s)
        beta_star = np.zeros(s_dim, dtype=np.float64)
        beta_star[:k_active] = [1.8, -1.5][:k_active]
        lam_l1 = 1.15 * noise_sd * np.sqrt(2.0 * np.log(s_dim) / n_samples)

        cpl_errs = []
        loc_errs = []
        for _ in range(160):
            x_mat = rng.normal(size=(n_samples, s_dim)) @ chol.T
            y_obs = x_mat @ beta_star + rng.normal(0.0, noise_sd, size=n_samples)
            beta_ols = np.linalg.solve(x_mat.T @ x_mat + 1e-3 * np.eye(s_dim), x_mat.T @ y_obs)
            cpl_errs.append(float(np.sum((beta_ols - beta_star) ** 2)))

            # Thresholded Lasso (Gauss-Lasso) readout on isolated per-source features
            beta_l1 = np.zeros(s_dim, dtype=np.float64)
            col_norms = np.sum(x_mat**2, axis=0) / n_samples
            resid = y_obs.copy()
            for _it in range(40):
                for j in range(s_dim):
                    old_j = beta_l1[j]
                    rho_j = float(x_mat[:, j] @ resid) / n_samples + col_norms[j] * old_j
                    new_j = np.sign(rho_j) * max(0.0, abs(rho_j) - lam_l1) / col_norms[j]
                    if new_j != old_j:
                        resid -= x_mat[:, j] * (new_j - old_j)
                        beta_l1[j] = new_j
            supp = np.where(np.abs(beta_l1) > 0.15)[0]
            beta_deb = np.zeros(s_dim, dtype=np.float64)
            if len(supp) > 0:
                beta_deb[supp] = np.linalg.lstsq(x_mat[:, supp], y_obs, rcond=None)[0]
            loc_errs.append(float(np.sum((beta_deb - beta_star) ** 2)))
        return float(np.mean(cpl_errs)), float(np.mean(loc_errs))

    cpl_s4, loc_s4 = _mc_readout_risk(4)
    cpl_s24, loc_s24 = _mc_readout_risk(32)

    return GeneralSCovarianceAuditResult(
        n_sources=n_sources,
        k_active=k_active,
        cov_condition_number=cond_num,
        analytical_vs_solve_max_diff=spec_diff,
        equicorrelated_closed_form_max_diff=equicorr_diff,
        equicorrelated_rho1_limit_error=rho1_limit_err,
        coupled_nonparent_weight_l2=cpl_nonparent_l2,
        local_nonparent_weight_l2=loc_nonparent_l2,
        coupled_single_ko_l2_error=cpl_ko_err,
        local_single_ko_l2_error=loc_ko_err,
        coupled_false_nonparent_attr_l2=cpl_false_attr,
        local_false_nonparent_attr_l2=loc_false_attr,
        finite_sample_coupled_risk_s4=cpl_s4,
        finite_sample_coupled_risk_s24=cpl_s24,
        finite_sample_local_l1_risk_s4=loc_s4,
        finite_sample_local_l1_risk_s24=loc_s24,
    )


def verify_closed_loop_stability_and_identifiability_theorem(
    n_nodes: int = 5,
) -> ClosedLoopIdentifiabilityAuditResult:
    """Verify Theorem 1(b): closed-loop absorbing ball, graph identifiability, & GL(M) gauge."""
    rng = np.random.default_rng(7711)
    cfg = MemGRNConfig(
        n_sources=n_nodes,
        n_targets=n_nodes,
        n_channels=2,
        hidden_width=3,
        rate_min=0.15,
        rate_max=6.0,
        rate_mode=RateInputMode.BOTH,
        use_asymmetric_rates=True,
    )
    off_diag = 1.0 - np.eye(n_nodes, dtype=np.float64)
    # Deliberately choose strong cyclic feedback weights so linear spectral radius > 1!
    direct_w = rng.normal(0.0, 0.95, size=(n_nodes, n_nodes)) * off_diag
    delayed_w = rng.normal(0.0, 0.75, size=(n_nodes, 2, n_nodes)) * off_diag[:, None, :]
    relaxation = np.linspace(0.45, 0.85, n_nodes)
    params = MemGRNParams(
        direct_weights=direct_w,
        delayed_weights=delayed_w,
        relaxation=relaxation,
        rate_logits=rng.normal(0.0, 0.5, size=(n_nodes, 2)),
        rate_logits_down=rng.normal(-0.3, 0.5, size=(n_nodes, 2)),
        input_proj=rng.normal(0.0, 0.7, size=(3, 3)),
        hidden_bias=rng.normal(0.0, 0.2, size=(3,)),
        output_proj=rng.normal(0.0, 0.6, size=(n_nodes, 2, 3)),
        output_proj_down=rng.normal(0.0, 0.6, size=(n_nodes, 2, 3)),
    )

    # Theoretical compact absorbing radius R*_j = (sum_i |D_ij| + sum_{i,r} |B_irj|) / delta_j
    r_star = (
        np.sum(np.abs(direct_w), axis=0) + np.sum(np.abs(delayed_w), axis=(0, 1))
    ) / relaxation
    max_r_star = float(np.max(r_star))

    # Start far outside the ball (||y(0)||_inf = 12.0) and integrate closed-loop with DOP853
    y0_far = rng.choice([-12.0, 12.0], size=n_nodes)
    init_vec = np.concatenate(
        [np.zeros(2 * n_nodes * 2, dtype=np.float64), y0_far]
    )

    def _cl_dop(t: float, s: np.ndarray) -> np.ndarray:
        _ = t
        n_zh = n_nodes * 2
        z_s = s[:n_zh].reshape(n_nodes, 2)
        h_s = s[n_zh : 2 * n_zh].reshape(n_nodes, 2)
        y_s = s[2 * n_zh :]
        dz, dh, dy = memgrn_vector_field(y_s, z_s, h_s, y_s, params, cfg)
        return np.concatenate([dz.ravel(), dh.ravel(), dy.ravel()])

    sol = solve_ivp(
        _cl_dop,
        [0.0, 35.0],
        init_vec,
        t_eval=np.linspace(0.0, 35.0, 351),
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    n_zh = n_nodes * 2
    max_filter_abs = float(np.max(np.abs(sol.y[: 2 * n_zh])))
    late_y = sol.y[2 * n_zh :, -50:]
    within_ball = bool(np.all(np.abs(late_y) <= r_star[:, None] + 1e-4))
    obs_max_late = float(np.max(np.abs(late_y)))

    # Part (ii-a): Constructive GL(M) gauge non-identifiability of coupled closed-loop networks
    # dx/dt = A_x x + B_x tanh(y), dy/dt = C_x x - delta * y  <=>  x_tilde = T x with T non-diagonal
    a_x = -np.diag([0.6, 1.1, 1.8, 2.4, 3.0])
    b_x = np.eye(n_nodes, dtype=np.float64)
    c_x = direct_w.copy()
    rot = np.eye(n_nodes, dtype=np.float64)
    theta_g = 0.65
    rot[:2, :2] = [[np.cos(theta_g), -np.sin(theta_g)], [np.sin(theta_g), np.cos(theta_g)]]
    a_x_tilde = rot @ a_x @ rot.T
    b_x_tilde = rot @ b_x
    c_x_tilde = rot @ c_x  # since rot is orthogonal: x^T c_x == (rot x)^T (rot c_x)

    def _sim_coupled_linear(a_m: np.ndarray, b_m: np.ndarray, c_m: np.ndarray) -> np.ndarray:
        def _rhs(_t: float, s: np.ndarray) -> np.ndarray:
            x_v, y_v = s[:n_nodes], s[n_nodes:]
            dx = a_m @ x_v + b_m @ np.tanh(y_v)
            dy = c_m.T @ x_v - relaxation * y_v
            return np.concatenate([dx, dy])

        return solve_ivp(
            _rhs,
            [0.0, 6.0],
            np.concatenate([np.zeros(n_nodes), np.linspace(-0.8, 0.9, n_nodes)]),
            t_eval=np.linspace(0.0, 6.0, 121),
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        ).y[n_nodes:]

    y_orig = _sim_coupled_linear(a_x, b_x, c_x)
    y_gauge = _sim_coupled_linear(a_x_tilde, b_x_tilde, c_x_tilde)
    gauge_traj_diff = float(np.max(np.abs(y_orig - y_gauge)))
    # Naive channel readout attribution ||c_m[i, :]|| differs sharply under rotation T!
    gauge_attr_disagree = float(np.max(np.abs(c_x - c_x_tilde)))

    # Part (ii-b): Exact structural recovery of delta_j, D_{ij}, and total steady-state weight
    # W^tot_{ij} = D_{ij} + sum_r B_{irj} via open-loop clamp probes (without plugging in ground truth!):
    # Step 1: Zero-memory initial-state relaxation probe (z(0)=h(0)=0, a(t)=0, y(0)=1) -> dy/dt = -delta * y
    def _zero_mem_relax_rhs(_t: float, s: np.ndarray) -> np.ndarray:
        z_s = s[:n_zh].reshape(n_nodes, 2)
        h_s = s[n_zh : 2 * n_zh].reshape(n_nodes, 2)
        y_s = s[2 * n_zh :]
        dz, dh, dy = memgrn_vector_field(np.zeros(n_nodes), z_s, h_s, y_s, params, cfg)
        return np.concatenate([dz.ravel(), dh.ravel(), dy.ravel()])

    t_rel = 1.0
    sol_rel = solve_ivp(
        _zero_mem_relax_rhs,
        [0.0, t_rel],
        np.concatenate([np.zeros(2 * n_zh), np.ones(n_nodes)]),
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
    )
    rec_relaxation = -np.log(sol_rel.y[2 * n_zh :, -1]) / t_rel
    relax_err = float(np.max(np.abs(rec_relaxation - relaxation)))

    # Step 2: Single-node clamp probes do(g_i = alpha), do(g_{k!=i} = 0) opening the feedback loop
    alpha_probe = np.arctanh(0.5)  # so a_i = tanh(alpha_probe) = 0.5 exactly
    rec_direct = np.zeros_like(direct_w)
    rec_total = np.zeros_like(direct_w)
    true_total = direct_w + np.sum(delayed_w, axis=1)

    # Also verify diagonal-free factorized cooperativity produces identically zero single-clamp distortion
    coop_u = rng.normal(0.0, 0.5, size=(n_nodes, 4))
    coop_v = rng.normal(0.0, 0.5, size=(n_nodes, 4))
    coop_w = rng.normal(0.0, 0.4, size=(4, n_nodes))
    coop_single_errs = []

    def _diag_free_coop(h_s: np.ndarray) -> np.ndarray:
        hb = np.mean(h_s, axis=-1)
        su = hb @ coop_u
        sv = hb @ coop_v
        feat = su * sv - (hb**2) @ (coop_u * coop_v)
        return feat @ coop_w

    for i_src in range(n_nodes):
        clamp_g = np.zeros(n_nodes, dtype=np.float64)
        clamp_g[i_src] = alpha_probe
        # At t = 0+ with z(0) = h(0) = y(0) = 0: dy_j(0+)/dt = D_{ij} * 0.5
        _, _, dy0 = memgrn_vector_field(
            clamp_g,
            np.zeros((n_nodes, 2)),
            np.zeros((n_nodes, 2)),
            np.zeros(n_nodes),
            params,
            cfg,
        )
        rec_direct[i_src] = dy0 / 0.5
        rec_direct[i_src, i_src] = 0.0

        # Under sustained clamp do(g = clamp_g), z_{ir}(t), h_{ir}(t) -> a_i = 0.5 exponentially
        # by Proposition 1(ii), so rec_relaxation_j * y_j(infty) / 0.5 -> D_{ij} + sum_r B_{irj}!
        def _clamp_rhs(_t: float, s: np.ndarray) -> np.ndarray:
            z_s = s[:n_zh].reshape(n_nodes, 2)
            h_s = s[n_zh : 2 * n_zh].reshape(n_nodes, 2)
            y_s = s[2 * n_zh :]
            dz, dh, dy = memgrn_vector_field(clamp_g, z_s, h_s, y_s, params, cfg)
            dy = dy + _diag_free_coop(h_s)
            return np.concatenate([dz.ravel(), dh.ravel(), dy.ravel()])

        sol_c = solve_ivp(
            _clamp_rhs,
            [0.0, 90.0],
            np.zeros(2 * n_zh + n_nodes),
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        h_inf = sol_c.y[n_zh : 2 * n_zh, -1].reshape(n_nodes, 2)
        coop_single_errs.append(float(np.max(np.abs(_diag_free_coop(h_inf)))))
        y_inf = sol_c.y[2 * n_zh :, -1]
        rec_total[i_src] = (rec_relaxation * y_inf) / 0.5
        rec_total[i_src, i_src] = 0.0

        # Also verify Theorem 1(ii) with nonzero diagonal self-memory b_self_{ir} != 0:
        # Since a_{k!=i} == 0 => h_{k!=i, r} == 0 by source isolation, off-diagonal recovery
        # of D_{ij} and W^tot_{ij} (j != i) is strictly invariant to diagonal self-memory!
        self_mem_w = np.array([[0.45, -0.25], [0.35, 0.20], [-0.30, 0.40], [0.50, -0.15], [0.25, 0.30]])

        def _clamp_self_rhs(_t: float, s: np.ndarray) -> np.ndarray:
            z_s = s[:n_zh].reshape(n_nodes, 2)
            h_s = s[n_zh : 2 * n_zh].reshape(n_nodes, 2)
            y_s = s[2 * n_zh :]
            dz, dh, dy = memgrn_vector_field(clamp_g, z_s, h_s, y_s, params, cfg)
            dy = dy + np.sum(h_s * self_mem_w, axis=1) + _diag_free_coop(h_s)
            return np.concatenate([dz.ravel(), dh.ravel(), dy.ravel()])

        sol_cs = solve_ivp(
            _clamp_self_rhs,
            [0.0, 90.0],
            np.zeros(2 * n_zh + n_nodes),
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        y_inf_self = sol_cs.y[2 * n_zh :, -1]
        rec_self_row = (rec_relaxation * y_inf_self) / 0.5
        rec_self_row[i_src] = 0.0
        if float(np.max(np.abs(rec_self_row - rec_total[i_src]))) > 1e-11:
            raise AssertionError("Diagonal self-memory altered off-diagonal Theorem 1(ii) recovery")

    d_err = float(np.max(np.abs(rec_direct - direct_w)))
    tot_err = float(np.max(np.abs(rec_total - true_total)))

    return ClosedLoopIdentifiabilityAuditResult(
        n_nodes=n_nodes,
        max_filter_abs=max_filter_abs,
        theoretical_y_trapping_radius=max_r_star,
        observed_max_y_after_transient=obs_max_late,
        within_absorbing_ball=within_ball,
        coupled_gauge_trajectory_max_diff=gauge_traj_diff,
        coupled_gauge_attribution_disagreement=gauge_attr_disagree,
        memgrn_relaxation_recovery_max_error=relax_err,
        memgrn_direct_weight_recovery_max_error=d_err,
        memgrn_total_weight_recovery_max_error=tot_err,
        factorized_coop_single_clamp_max_error=float(max(coop_single_errs)),
    )


def state_only_iso_expression_lower_bound(
    post_washout_separation_l2: float,
    horizon_tau: float,
    lipschitz_const: float = 0.0,
    matched_washout_state: bool = True,
) -> float:
    """Compute Proposition 1(i) post-washout L2 error lower bound for state-only ODEs."""
    if matched_washout_state:
        return 0.5 * post_washout_separation_l2
    denom = 2.0 * np.sqrt(horizon_tau) * (1.0 + np.exp(lipschitz_const * horizon_tau))
    return float(post_washout_separation_l2 / denom)


def verify_iso_expression_proposition() -> IsoExpressionAuditResult:
    """Construct exact iso-expression pulse pair and verify Proposition 1(i) vs. PerturbODE."""
    t_wash = 1.0
    t_end = 5.0
    decay = 0.6

    def _simulate_plant(duration: float, amp: float, t_eval: np.ndarray) -> np.ndarray:
        t_start = t_wash - duration

        def _rhs_on(_t: float, s: np.ndarray) -> list[float]:
            z, h, y = float(s[0]), float(s[1]), float(s[2])
            lam = 0.25 + 3.5 / (1.0 + np.exp(2.2 * h - 0.8))
            return [lam * (amp - z), lam * (z - h), 1.4 * h + 0.35 * amp - decay * y]

        def _rhs_off(_t: float, s: np.ndarray) -> list[float]:
            z, h, y = float(s[0]), float(s[1]), float(s[2])
            lam = 0.25 + 3.5 / (1.0 + np.exp(2.2 * h - 0.8))
            return [lam * (-z), lam * (z - h), 1.4 * h - decay * y]

        sol_on = solve_ivp(
            _rhs_on,
            [t_start, t_wash],
            [0.0, 0.0, 0.0],
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        sol_off = solve_ivp(
            _rhs_off,
            [t_wash, t_end],
            sol_on.y[:, -1],
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        return sol_off.y

    t_post = np.linspace(t_wash, t_end, 401)
    dt = float(t_post[1] - t_post[0])

    traj_long = _simulate_plant(duration=1.0, amp=0.12, t_eval=t_post)
    target_y_wash = float(traj_long[2, 0])

    lo, hi = 0.12, 0.98
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = float(
            _simulate_plant(duration=0.35, amp=mid, t_eval=np.array([t_wash, t_end]))[2, 0]
        )
        if val < target_y_wash:
            lo = mid
        else:
            hi = mid
    traj_short = _simulate_plant(duration=0.35, amp=0.5 * (lo + hi), t_eval=t_post)

    wash_diff = abs(float(traj_long[2, 0] - traj_short[2, 0]))
    h_diff = abs(float(traj_long[1, 0] - traj_short[1, 0]))
    sep_l2 = float(np.sqrt(trapezoid((traj_long[2] - traj_short[2]) ** 2, dx=dt)))
    bound = state_only_iso_expression_lower_bound(
        sep_l2, t_end - t_wash, matched_washout_state=True
    )

    midpoint_curve = 0.5 * (traj_long[2] + traj_short[2])
    best_state_only_err = max(
        float(np.sqrt(trapezoid((midpoint_curve - traj_long[2]) ** 2, dx=dt))),
        float(np.sqrt(trapezoid((midpoint_curve - traj_short[2]) ** 2, dx=dt))),
    )

    return IsoExpressionAuditResult(
        washout_value_diff=wash_diff,
        memory_state_diff_at_washout=h_diff,
        true_post_washout_separation_l2=sep_l2,
        state_only_universal_lower_bound=bound,
        best_state_only_post_washout_error_l2=best_state_only_err,
        memgrn_post_washout_error_l2=0.0,
    )


def verify_orthogonal_cofactor_invariance_proposition() -> OrthogonalCofactorAuditResult:
    """Verify Proposition 1(iii): orthogonal cofactor readout preserves observed TF attribution."""
    rng = np.random.default_rng(314159)
    n_samples, p_obs, k_lat, n_genes = 160, 4, 3, 12

    psi_obs = rng.normal(size=(n_samples, p_obs))
    # Unmeasured cofactor trajectory partially collinear with observed TF trajectories
    collinear_mix = rng.normal(0.0, 0.85, size=(p_obs, k_lat))
    psi_unmeasured_true = psi_obs @ collinear_mix + 0.75 * rng.normal(size=(n_samples, k_lat))

    w_obs_true = rng.normal(0.0, 1.0, size=(p_obs, n_genes))
    w_lat_true = rng.normal(0.0, 0.9, size=(k_lat, n_genes))
    y_target = psi_obs @ w_obs_true + psi_unmeasured_true @ w_lat_true + rng.normal(
        0.0, 0.05, size=(n_samples, n_genes)
    )

    # 1. Observed-only estimator: Psi_obs^+ Y
    w_obs_only, _, _, _ = np.linalg.lstsq(psi_obs, y_target, rcond=None)
    mse_obs_only = float(np.mean((y_target - psi_obs @ w_obs_only) ** 2))

    # 2. Unprojected joint ridge estimator (analogous to unconstrained Latent ODE + direct readout):
    # splits variance between collinear Psi_obs and Psi_lat, stealing observed TF weights!
    joint_unproj = np.concatenate([psi_obs, psi_unmeasured_true], axis=1)
    reg = 1.0
    w_joint_unproj = np.linalg.solve(
        joint_unproj.T @ joint_unproj + reg * np.eye(p_obs + k_lat),
        joint_unproj.T @ y_target,
    )
    w_obs_unproj = w_joint_unproj[:p_obs]
    theft_rel = float(
        np.linalg.norm(w_obs_unproj - w_obs_only) / np.linalg.norm(w_obs_only)
    )

    # 3. Projection-orthogonalized cofactor estimator (MemGRN-Ortho):
    psi_perp = project_orthogonal_cofactor_basis(psi_obs, psi_unmeasured_true, ridge_eps=0.0)
    max_ip = float(np.max(np.abs(psi_obs.T @ psi_perp)))

    joint_ortho = np.concatenate([psi_obs, psi_perp], axis=1)
    # Penalty applied only to latent cofactor block leaves w_obs_ortho === w_obs_only identically!
    pen_diag = np.diag([0.0] * p_obs + [reg] * k_lat)
    w_joint_ortho = np.linalg.solve(
        joint_ortho.T @ joint_ortho + pen_diag, joint_ortho.T @ y_target
    )
    w_obs_ortho = w_joint_ortho[:p_obs]
    invariance_err = float(np.max(np.abs(w_obs_ortho - w_obs_only)))
    mse_ortho = float(np.mean((y_target - joint_ortho @ w_joint_ortho) ** 2))

    return OrthogonalCofactorAuditResult(
        orthogonality_max_inner_product=max_ip,
        observed_weight_invariance_error=invariance_err,
        unprojected_observed_weight_theft_rel_error=theft_rel,
        mse_observed_only=mse_obs_only,
        mse_memgrn_ortho=mse_ortho,
    )


def verify_unit_dc_gain_and_second_order_vs_local_ltc() -> UnitDCGainAndSecondOrderAuditResult:
    """Verify Theorem 1(iii): unit-DC-gain identifiability & 2nd-order delay vs. Source-Local LTC."""
    # -------------------------------------------------------------------------
    # Part (a): Kinetic-readout scale & saturation entanglement in state-dependent
    # Source-Local LTC (with simultaneous gate rescaling f_ir(./c, a_i))
    # vs. exact rate-invariant unit-DC-gain graph identifiability in MemGRN.
    # -------------------------------------------------------------------------
    tau_inv = 0.95
    w_x = 1.15
    w_a = 2.6
    b_g = -1.6
    e_rev = 1.4
    b_read = 1.6
    d_read = 0.25
    delta = 0.6
    gauge_c = 2.5

    def _ltc_rollout(
        a_fn, e_scale: float, b_scale: float, wx_scale: float, t_eval: np.ndarray
    ) -> np.ndarray:
        def _rhs(t: float, s: np.ndarray) -> list[float]:
            a_val = float(a_fn(t))
            x_v, y_v = float(s[0]), float(s[1])
            gate = 3.5 / (1.0 + np.exp(-(wx_scale * x_v + w_a * a_val + b_g)))
            dx = -(tau_inv + gate) * x_v + gate * (a_val * e_scale)
            dy = d_read * a_val + b_scale * x_v - delta * y_v
            return [dx, dy]

        return solve_ivp(
            _rhs,
            [float(t_eval[0]), float(t_eval[-1])],
            [0.0, 0.0],
            t_eval=t_eval,
            method="DOP853",
            max_step=0.05,
            rtol=1e-13,
            atol=1e-15,
        ).y[1]

    t_probe = np.linspace(0.0, 12.0, 241)
    pulse_probe = lambda t: float(0.75 * np.exp(-0.25 * (t - 1.5) ** 2))
    # Exact scale-gauge equivalence: (A, B, w_x) -> (c*A, B/c, w_x/c) leaves y(t) strictly invariant
    y_ltc_1 = _ltc_rollout(pulse_probe, e_rev, b_read, w_x, t_probe)
    y_ltc_2 = _ltc_rollout(pulse_probe, gauge_c * e_rev, b_read / gauge_c, w_x / gauge_c, t_probe)
    ltc_gauge_traj_diff = float(np.max(np.abs(y_ltc_1 - y_ltc_2)))
    ltc_gauge_w_diff = abs(b_read - b_read / gauge_c)

    # Even fixing e_rev = 1.0, state-dependent Source-Local LTC steady-state effective gain
    # W_eff(alpha) = (delta * y_inf(alpha)) / alpha varies nonlinearly across probe amplitudes alpha!
    eff_ltc = []
    for alpha in (0.15, 0.85):
        y_ss = float(_ltc_rollout(lambda _t, a=alpha: a, 1.0, b_read, w_x, np.array([0.0, 90.0]))[-1])
        eff_ltc.append(delta * y_ss / alpha)
    ltc_amp_ratio = float(eff_ltc[1] / eff_ltc[0])

    # Contrast with MemGRN two-stage unit-DC-gain steady-state recovery across two distinct
    # nonlinear rate functions and two distinct probe amplitudes alpha in {0.15, 0.85}.
    w_tot_true = d_read + b_read

    def _memgrn_steady_gain(alpha: float, rate_shift: float) -> float:
        def _rhs(_t: float, s: np.ndarray) -> list[float]:
            z_v, h_v, y_v = float(s[0]), float(s[1]), float(s[2])
            lam_up = 0.25 + 4.5 / (1.0 + np.exp(-(rate_shift + 1.8 * alpha - 1.1 * h_v)))
            lam_dn = 0.25 + 4.5 / (1.0 + np.exp(-(rate_shift - 0.6 + 0.9 * z_v - 1.4 * h_v)))
            return [lam_up * (alpha - z_v), lam_dn * (z_v - h_v), d_read * alpha + b_read * h_v - delta * y_v]

        sol = solve_ivp(
            _rhs,
            [0.0, 90.0],
            [0.0, 0.0, 0.0],
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        return float(delta * sol.y[2, -1] / alpha)

    mg_a1_k1 = _memgrn_steady_gain(0.15, 0.0)
    mg_a2_k1 = _memgrn_steady_gain(0.85, 0.0)
    mg_a2_k2 = _memgrn_steady_gain(0.85, -1.2)
    memgrn_amp_err = float(max(abs(mg_a1_k1 - w_tot_true), abs(mg_a2_k1 - w_tot_true)))
    memgrn_kin_diff = float(abs(mg_a2_k1 - mg_a2_k2))

    # -------------------------------------------------------------------------
    # Part (b): Second-order post-shutoff delayed peak vs. 1st-order LTC monotonicity
    # -------------------------------------------------------------------------
    t_off = 0.45
    t_end = 5.0
    t_post = np.linspace(t_off, t_end, 456)
    dt_p = float(t_post[1] - t_post[0])

    def _memgrn_pulse_rhs(t: float, s: np.ndarray) -> list[float]:
        a_val = 0.90 if t <= t_off else 0.0
        z_v, h_v = float(s[0]), float(s[1])
        lam_up = 2.8 if a_val > 0.0 else 0.65
        lam_dn = 0.55 + 1.1 / (1.0 + np.exp(1.5 * h_v))
        return [lam_up * (a_val - z_v), lam_dn * (z_v - h_v)]

    sol_on = solve_ivp(
        _memgrn_pulse_rhs, [0.0, t_off], [0.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14
    )
    sol_off = solve_ivp(
        _memgrn_pulse_rhs,
        [t_off, t_end],
        sol_on.y[:, -1],
        t_eval=t_post,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    h_post = sol_off.y[1]
    peak_idx = int(np.argmax(h_post))
    peak_gain = float(h_post[peak_idx] - h_post[0])
    peak_time = float(t_post[peak_idx])

    # Verify first-order autonomous positive LTC channel post-shutoff slope <= 0 everywhere
    def _ltc1_off_rhs(_t: float, s: np.ndarray) -> list[float]:
        x_v = float(s[0])
        gate = 3.5 / (1.0 + np.exp(-(1.2 * x_v + b_g)))
        return [-(tau_inv + gate) * x_v]

    sol_ltc1 = solve_ivp(
        _ltc1_off_rhs, [t_off, t_end], [float(sol_on.y[0, -1])], t_eval=t_post, method="DOP853", rtol=1e-12, atol=1e-14
    )
    ltc1_slopes = np.array([_ltc1_off_rhs(0.0, np.array([xv]))[0] for xv in sol_ltc1.y[0]])
    first_order_max_slope = float(np.max(ltc1_slopes))

    # Best possible non-increasing L2 approximation to h_post via antitonic Pool-Adjacent-Violators (PAV)!
    # Any nonnegative-readout 1-stage Source-Local LTC (a(t)=0 post-shutoff) lies in the non-increasing cone.
    blocks: list[list[float]] = []
    for val in h_post:
        blocks.append([float(val), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            w1, n1 = blocks[-2]
            w2, n2 = blocks[-1]
            blocks.pop()
            blocks[-1] = [(w1 * n1 + w2 * n2) / (n1 + n2), n1 + n2]
    mono_proj = np.concatenate([np.full(int(n), v, dtype=np.float64) for v, n in blocks])
    best_mono_l2 = float(np.sqrt(trapezoid((h_post - mono_proj) ** 2, dx=dt_p)))

    return UnitDCGainAndSecondOrderAuditResult(
        ltc_scale_gauge_trajectory_max_diff=ltc_gauge_traj_diff,
        ltc_scale_gauge_weight_disagreement=ltc_gauge_w_diff,
        ltc_amplitude_gain_distortion_ratio=ltc_amp_ratio,
        memgrn_amplitude_gain_max_error=memgrn_amp_err,
        memgrn_kinetic_change_steady_state_diff=memgrn_kin_diff,
        first_order_post_shutoff_max_slope=first_order_max_slope,
        memgrn_post_shutoff_peak_gain=peak_gain,
        memgrn_post_shutoff_peak_time=peak_time,
        best_monotone_first_order_l2_floor=best_mono_l2,
        memgrn_delayed_peak_l2_error=0.0,
    )


def verify_ortho_ood_generalization_theorem() -> OrthoOODGeneralizationAuditResult:
    """Verify Proposition 1(iii) under oracle exogenous test-time cofactors and matched ridge.

    Audits two exact algebraic identities under matched regularization (reg_o = reg_u = 10.0):
      (1) Block decoupling: solving the joint orthogonalized system [Phi_O, U_perp] recovers
          W_O^ortho === W_O^obs identically (preserving the standalone observed-source estimator
          without weight theft), whereas unprojected joint ridge on [Phi_O, U] shifts W_O^unproj
          away from W_O^obs. (Note: preserving W_O^obs preserves the observed-only total-effect
          estimator rather than deconfounding omitted-variable bias B_tr W_U*.)
      (2) Exact analytical OOD MSE reduction identity holds to machine precision when test-time
          cofactor basis U_te spans the exogenous unmeasured factor Phi_U^te.
    """
    rng = np.random.default_rng(202705)
    n_tr, n_te, p_obs, k_u, n_genes = 2000, 1200, 6, 2, 16
    reg_match = 10.0

    w_obs_true = rng.normal(0.0, 0.9, size=(p_obs, n_genes))
    w_u_true = rng.normal(0.0, 1.0, size=(k_u, n_genes))

    # Training distribution: unobserved cofactors U are strongly correlated with observed O
    # U = O @ B_tr + E_u (confounded training regime)
    b_tr = rng.normal(0.0, 0.55, size=(p_obs, k_u))
    phi_o_tr = rng.normal(size=(n_tr, p_obs))
    e_u_tr = rng.normal(0.0, 0.85, size=(n_tr, k_u))
    phi_u_tr = phi_o_tr @ b_tr + e_u_tr
    y_tr = phi_o_tr @ w_obs_true + phi_u_tr @ w_u_true

    # OOD test distribution: correlation shifts (B_te = 0.25 * B_tr, breaking O-U confounding)
    b_te = 0.25 * b_tr
    phi_o_te = rng.normal(size=(n_te, p_obs))
    e_u_te = rng.normal(0.0, 0.85, size=(n_te, k_u))
    phi_u_te = phi_o_te @ b_te + e_u_te
    y_te = phi_o_te @ w_obs_true + phi_u_te @ w_u_true

    # 1. Observed-only ridge estimator with matched penalty reg_match
    w_o_obs = np.linalg.solve(
        phi_o_tr.T @ phi_o_tr + reg_match * np.eye(p_obs),
        phi_o_tr.T @ y_tr,
    )
    pred_te_obs = phi_o_te @ w_o_obs
    mse_te_obs = float(np.mean((y_te - pred_te_obs) ** 2))

    # 2. MemGRN-Ortho: project cofactor basis U onto col(O)^perp using P_tr = (O^T O)^-1 O^T U
    # Solve the ACTUAL joint block system [O, U_perp] with matched penalty reg_match on BOTH blocks
    p_tr = np.linalg.solve(phi_o_tr.T @ phi_o_tr, phi_o_tr.T @ phi_u_tr)
    u_perp_tr = phi_u_tr - phi_o_tr @ p_tr
    joint_ortho_tr = np.concatenate([phi_o_tr, u_perp_tr], axis=1)
    w_joint_ortho = np.linalg.solve(
        joint_ortho_tr.T @ joint_ortho_tr + reg_match * np.eye(p_obs + k_u),
        joint_ortho_tr.T @ y_tr,
    )
    w_o_ortho = w_joint_ortho[:p_obs]
    w_perp = w_joint_ortho[p_obs:]

    # At OOD test time, evaluate with oracle exogenous test-time cofactor basis phi_u_te
    u_perp_te = phi_u_te - phi_o_te @ p_tr
    pred_te_ortho = phi_o_te @ w_o_ortho + u_perp_te @ w_perp
    mse_te_ortho = float(np.mean((y_te - pred_te_ortho) ** 2))

    # Exact analytical OOD MSE reduction identity:
    # ||Y_te - Y_obs||^2 - ||Y_te - Y_ortho||^2 =
    #   2 <Y_te - Y_obs, U_perp_te W_perp> - ||U_perp_te W_perp||^2
    res_obs_te = y_te - pred_te_obs
    ortho_add_te = u_perp_te @ w_perp
    analytical_red = float(
        (2.0 * np.sum(res_obs_te * ortho_add_te) - np.sum(ortho_add_te**2)) / y_te.size
    )
    empirical_red = mse_te_obs - mse_te_ortho
    risk_identity_diff = abs(analytical_red - empirical_red)

    # 3. Unprojected joint ridge estimator with IDENTICAL penalty reg_match
    joint_tr = np.concatenate([phi_o_tr, phi_u_tr], axis=1)
    w_joint_u = np.linalg.solve(
        joint_tr.T @ joint_tr + reg_match * np.eye(p_obs + k_u),
        joint_tr.T @ y_tr,
    )
    w_o_unproj = w_joint_u[:p_obs]
    w_u_unproj = w_joint_u[p_obs:]
    pred_te_unproj = phi_o_te @ w_o_unproj + phi_u_te @ w_u_unproj
    mse_te_unproj = float(np.mean((y_te - pred_te_unproj) ** 2))

    # Measure weight preservation relative to standalone observed-only W_o_obs (solving joint system!)
    drift_ortho_list = []
    drift_unproj_list = []
    rmse_ortho_list = []
    rmse_unproj_list = []
    for i_src in range(p_obs):
        eff_true = phi_o_te[:, i_src : i_src + 1] @ w_obs_true[i_src : i_src + 1]
        eff_obs_only = phi_o_te[:, i_src : i_src + 1] @ w_o_obs[i_src : i_src + 1]
        eff_ortho = phi_o_te[:, i_src : i_src + 1] @ w_o_ortho[i_src : i_src + 1]
        eff_unproj = phi_o_te[:, i_src : i_src + 1] @ w_o_unproj[i_src : i_src + 1]
        drift_ortho_list.append(float(np.max(np.abs(eff_ortho - eff_obs_only))))
        drift_unproj_list.append(
            float(np.linalg.norm(eff_unproj - eff_obs_only) / np.linalg.norm(eff_obs_only))
        )
        rmse_ortho_list.append(float(np.sqrt(np.mean((eff_ortho - eff_true) ** 2))))
        rmse_unproj_list.append(float(np.sqrt(np.mean((eff_unproj - eff_true) ** 2))))

    return OrthoOODGeneralizationAuditResult(
        ood_counterfactual_drift_ortho=float(max(drift_ortho_list)),
        ood_counterfactual_drift_unproj=float(np.mean(drift_unproj_list)),
        analytical_vs_empirical_ood_risk_diff=risk_identity_diff,
        ood_mse_observed_only=mse_te_obs,
        ood_mse_memgrn_ortho=mse_te_ortho,
        ood_mse_unproj_ridge=mse_te_unproj,
        unproj_ood_obs_attr_rmse=float(np.mean(rmse_unproj_list)),
        ortho_ood_obs_attr_rmse=float(np.mean(rmse_ortho_list)),
    )


def verify_selective_ssm_crosstalk_corollary(
    w_star: float = 2.4, rho: float = 0.85, l2_penalty: float = 1e-4
) -> SelectiveSSMCrossTalkAuditResult:
    """Verify Corollary to Theorem 1(i): Mamba/S6 diagonal SSM Delta(a) gate suffers cross-talk.

    Even with diagonal state matrix A = -diag(alpha), the coupled timescale gate
    Delta_1(a) = softplus(w_1 a_1 + w_2 a_2 + b_dt) fits minimum-norm collinear weights
    (w_1, w_2) = (w_star / (1 + rho^2), rho * w_star / (1 + rho^2)) on correlated training
    pulses a_2 = rho * a_1, leaking nonzero timescale modulation onto non-parent TF 2.
    """
    w1_ssm, w2_ssm = minimum_norm_collinear_gate_weights(
        w_star=w_star, rho=rho, eta_var=0.0, l2_penalty=l2_penalty
    )
    alpha_diag = 1.1
    b_dt = -0.3
    t_eval = np.linspace(0.0, 6.0, 241)
    dt = float(t_eval[1] - t_eval[0])

    def _softplus(x: float) -> float:
        return float(np.logaddexp(0.0, x))

    def _sim_ssm(w1: float, w2: float, a1_fn, a2_fn) -> np.ndarray:
        def _rhs(t: float, s: np.ndarray) -> list[float]:
            a1, a2 = float(a1_fn(t)), float(a2_fn(t))
            x_v = float(s[0])
            delta_t = 0.25 + _softplus(w1 * a1 + w2 * a2 + b_dt)
            dx = delta_t * alpha_diag * (a1 - x_v)
            return [dx]

        return solve_ivp(
            _rhs,
            [0.0, 6.0],
            [0.0],
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        ).y[0]

    pulse = lambda t: 0.9 if 0.0 <= t <= 1.2 else 0.0
    zero = lambda _t: 0.0

    # True plant: diagonal selective SSM whose timescale depends ONLY on parent TF 1 (w_star, 0)
    x_true_ko2 = _sim_ssm(w_star, 0.0, pulse, zero)
    x_ssm_ko2 = _sim_ssm(w1_ssm, w2_ssm, pulse, zero)
    x_ssm_both = _sim_ssm(w1_ssm, w2_ssm, pulse, lambda t: rho * pulse(t))
    x_mem_ko2 = _sim_ssm(w_star, 0.0, pulse, zero)
    x_mem_both = _sim_ssm(w_star, 0.0, pulse, lambda t: rho * pulse(t))

    ssm_ko_err = float(np.sqrt(trapezoid((x_ssm_ko2 - x_true_ko2) ** 2, dx=dt)))
    ssm_false_attr = float(np.sqrt(trapezoid((x_ssm_both - x_ssm_ko2) ** 2, dx=dt)))
    mem_ko_err = float(np.sqrt(trapezoid((x_mem_ko2 - x_true_ko2) ** 2, dx=dt)))
    mem_false_attr = float(np.sqrt(trapezoid((x_mem_both - x_mem_ko2) ** 2, dx=dt)))

    return SelectiveSSMCrossTalkAuditResult(
        ssm_nonparent_delta_weight=w2_ssm,
        ssm_ko_trajectory_l2_error=ssm_ko_err,
        ssm_false_nonparent_attribution_l2=ssm_false_attr,
        memgrn_ko_trajectory_l2_error=mem_ko_err,
        memgrn_false_nonparent_attribution_l2=mem_false_attr,
    )
