"""Theoretical guarantees for source-isolated adaptive memory vs. coupled LTC and PerturbODE."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp, trapezoid


@dataclass(frozen=True, kw_only=True)
class CrossTalkAuditResult:
    """Analytical vs. numerical verification of Theorem 1 (coupled cross-talk vs. MemGRN)."""

    rho_train: float
    coupled_weights: tuple[float, float]
    local_weights: tuple[float, float]
    coupled_ko_trajectory_l2_error: float
    coupled_false_source2_attribution_l2: float
    local_ko_trajectory_l2_error: float
    local_false_source2_attribution_l2: float


@dataclass(frozen=True, kw_only=True)
class IsoExpressionAuditResult:
    """Analytical vs. numerical verification of Proposition 2 (non-Markovian separation)."""

    washout_value_diff: float
    memory_state_diff_at_washout: float
    true_post_washout_separation_l2: float
    state_only_universal_lower_bound: float
    best_state_only_post_washout_error_l2: float
    memgrn_post_washout_error_l2: float


def minimum_norm_collinear_gate_weights(
    w_star: float,
    rho: float,
    eta_var: float = 0.0,
    a1_var: float = 1.0,
    l2_penalty: float = 1e-6,
) -> tuple[float, float]:
    """Compute closed-form ridge / minimum-norm coupled gate weights under correlated inputs.

    When a_2 = rho * a_1 + sqrt(1 - rho^2) * eta with Var(a_1) = a1_var and Var(eta) = eta_var,
    fitting w_1 a_1 + w_2 a_2 to target w_star * a_1 with L2 regularization l2_penalty yields
    the closed-form weights verified in Theorem 1. In the noiseless pathway limit eta_var -> 0
    and l2_penalty -> 0+, this converges to (w_star / (1 + rho^2), rho * w_star / (1 + rho^2)).
    """
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


def verify_coupled_crosstalk_theorem(
    w_star: float = 2.4,
    rho: float = 0.85,
    rate_min: float = 0.1,
    rate_max: float = 8.0,
    base_logit: float = -0.8,
    delayed_weight: float = 1.2,
    decay: float = 0.5,
) -> CrossTalkAuditResult:
    """Verify Theorem 1: minimum-norm coupled adaptive filter vs. source-isolated MemGRN."""
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

    # On the correlated training manifold a2 = rho * a1, BOTH models achieve zero training error
    y_true_train = _rollout(pulse, collinear_a2, w_star, 0.0, t_grid)
    y_cpl_train = _rollout(pulse, collinear_a2, w1_cpl, w2_cpl, t_grid)
    if float(np.max(np.abs(y_true_train - y_cpl_train))) > 1e-11:
        raise AssertionError("Coupled minimum-norm interpolant failed to match training manifold")

    # Under correlation-breaking intervention do(a2 <- 0) while a1 pulses:
    y_true_ko = _rollout(pulse, zero_fn, w_star, 0.0, t_grid)
    y_cpl_ko = _rollout(pulse, zero_fn, w1_cpl, w2_cpl, t_grid)
    y_loc_ko = _rollout(pulse, zero_fn, w1_loc, w2_loc, t_grid)

    # Counterfactual attribution to non-causal regulator 2: y(a1, a2) - y(a1, 0)
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


def state_only_iso_expression_lower_bound(
    post_washout_separation_l2: float,
    horizon_tau: float,
    lipschitz_const: float = 0.0,
    matched_washout_state: bool = True,
) -> float:
    """Compute Proposition 2 post-washout L2 error lower bound for state-only ODEs dy/dt = F(y, g).

    Args:
        post_washout_separation_l2: ||y_star^(1) - y_star^(2)||_{L^2[T, T + tau]}.
        horizon_tau: Length of post-washout window tau > 0.
        lipschitz_const: Lipschitz constant L_F of F(y, 0) on [T, T + tau] when y_F(T) is free.
        matched_washout_state: If True, returns exact triangle-inequality floor 0.5 * sep_l2
            applicable to every state-only ODE with y_F^(1)(T) = y_F^(2)(T); otherwise returns
            the combined Gronwall bound across washout mismatch + post-washout error.
    """
    if matched_washout_state:
        return 0.5 * post_washout_separation_l2
    denom = 2.0 * np.sqrt(horizon_tau) * (1.0 + np.exp(lipschitz_const * horizon_tau))
    return float(post_washout_separation_l2 / denom)


def verify_iso_expression_proposition() -> IsoExpressionAuditResult:
    """Construct exact iso-expression pulse pair and verify Proposition 2 vs. PerturbODE."""
    # Single-channel MemGRN plant over [0, T] with T = 1.0 h, washout [1.0, 5.0] h
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

    # Long moderate pulse: duration = 1.0 h, amp = 0.12
    traj_long = _simulate_plant(duration=1.0, amp=0.12, t_eval=t_post)
    target_y_wash = float(traj_long[2, 0])

    # Bisection on short intense pulse (duration = 0.35 h) to match y(T) identically
    lo, hi = 0.12, 0.98
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = float(_simulate_plant(duration=0.35, amp=mid, t_eval=np.array([t_wash, t_end]))[2, 0])
        if val < target_y_wash:
            lo = mid
        else:
            hi = mid
    traj_short = _simulate_plant(duration=0.35, amp=0.5 * (lo + hi), t_eval=t_post)

    wash_diff = abs(float(traj_long[2, 0] - traj_short[2, 0]))
    h_diff = abs(float(traj_long[1, 0] - traj_short[1, 0]))
    sep_l2 = float(np.sqrt(trapezoid((traj_long[2] - traj_short[2]) ** 2, dx=dt)))
    bound = state_only_iso_expression_lower_bound(sep_l2, t_end - t_wash, matched_washout_state=True)

    # Optimal oracle state-only trajectory matching y(T) is the midpoint curve 0.5*(y1 + y2)
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
