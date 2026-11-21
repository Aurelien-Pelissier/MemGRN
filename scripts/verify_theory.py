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


def verify_all_theory(output_json: Path | None = None) -> dict:
    """Run deterministic numerical audits of Proposition 1, Theorem 1, and schedule bounds."""
    # 1. Theorem 1: 25 (r, theta) pairs against adaptive Gauss-Kronrod quadrature
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
        raise AssertionError(f"Theorem 1 quadrature mismatch: {max_swap_err}")

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

    # 3. Proposition 1: Bounded filters, additive accounting, and DOP853 comparison
    cfg = MemGRNConfig(
        n_sources=2,
        n_targets=1,
        n_channels=1,
        hidden_width=2,
        rate_min=0.1,
        rate_max=2.9,
        rate_mode=RateInputMode.BOTH,
    )
    params = MemGRNParams(
        direct_weights=np.array([[0.6], [-0.4]]),
        delayed_weights=np.array([[[0.8]], [[-0.7]]]),
        relaxation=np.array([0.35]),
        rate_logits=np.array([[0.3], [0.3]]),
        input_proj=np.array([[0.5, -0.3], [0.8, 0.2], [-0.4, 0.6]]),
        hidden_bias=np.array([0.1, -0.1]),
        output_proj=np.array([[[0.7, -0.5]], [[-0.4, 0.6]]]),
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

    # Splice decomposition check
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

    # Compare against adaptive SciPy DOP853 solver on constant-input boundedness
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

    summary = {
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
    res = verify_all_theory(
        Path(__file__).resolve().parents[1] / "results" / "theory_verification.json"
    )
    print(json.dumps(res, indent=2))
