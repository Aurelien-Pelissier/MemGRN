"""Counterbalanced perturbation design, identifiability bounds, and schedule certification."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import beta


@dataclass(frozen=True, kw_only=True)
class ScheduleBoundResult:
    """Primal-dual minimax schedule evaluation under AR(1) temporal innovations."""

    best_dp_schedule: tuple[int, ...]
    best_dp_worst_distance_sq: float
    dual_upper_bound: float
    exhaustive_optimum: float | None = None


def swap_singular_values(amp_a: float, amp_b: float, apply_tanh: bool = True) -> tuple[float, float]:
    """Return singular values (|alpha + beta|, |alpha - beta|) of the 2x2 swap mixing matrix."""
    alpha = float(np.tanh(amp_a)) if apply_tanh else float(amp_a)
    beta = float(np.tanh(amp_b)) if apply_tanh else float(amp_b)
    return abs(alpha + beta), abs(alpha - beta)


def invert_swap_waveforms(
    record_1: np.ndarray,
    record_2: np.ndarray,
    amp_a: float,
    amp_b: float,
    apply_tanh: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Algebraically recover unit-activity source waveforms (h_i, h_k) from swapped records."""
    alpha = float(np.tanh(amp_a)) if apply_tanh else float(amp_a)
    beta = float(np.tanh(amp_b)) if apply_tanh else float(amp_b)
    if np.isclose(abs(alpha - beta), 0.0) or np.isclose(abs(alpha + beta), 0.0):
        raise ValueError(f"Degenerate swap strengths alpha={alpha}, beta={beta}")
    sum_wave = (record_1 + record_2) / (alpha + beta)
    diff_wave = (record_1 - record_2) / (alpha - beta)
    return 0.5 * (sum_wave + diff_wave), 0.5 * (sum_wave - diff_wave)


def exact_swap_cancellation_accuracy(norm_r: float, angle_rad: float) -> float:
    """Closed-form Bayes and nearest-template accuracy for two cancelling pairs (Theorem 1).

    Args:
        norm_r: Whitened difference-channel norm r >= 0.
        angle_rad: Acute angle theta in [0, pi/2] between difference-channel directions.

    Returns:
        Exact probability of recovering the true unordered source pair.
    """
    if norm_r < 0.0:
        raise ValueError(f"norm_r must be nonnegative, got {norm_r}")
    u_prob = float(ndtr(norm_r * np.sin(0.5 * angle_rad)))
    v_prob = float(ndtr(norm_r * np.cos(0.5 * angle_rad)))
    return 0.5 + 2.0 * (u_prob - 0.5) * (v_prob - 0.5)


def template_margin_lower_bound(
    true_mean: np.ndarray,
    true_templates: np.ndarray,
    wrong_templates: np.ndarray,
) -> float:
    """Directional Gaussian lower bound on unordered-pair recovery with template error.

    Args:
        true_mean: Whitened true two-record response vector mu of shape (dim,).
        true_templates: Fitted templates for the two orientations of the true pair,
            shape (2, dim).
        wrong_templates: Fitted templates for rival unordered pairs and orientations,
            shape (n_rivals, dim).

    Returns:
        Conservative lower bound in [0, 1] maximized over the two true-pair anchors.
    """
    best_bound = 0.0
    for anchor in true_templates:
        diffs = wrong_templates - anchor[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        if np.any(dists <= 1e-12):
            continue
        normals = diffs / dists[:, None]
        projected_err = normals @ (true_mean - anchor)
        error_sum = float(np.sum(ndtr(-0.5 * dists + projected_err)))
        best_bound = max(best_bound, max(0.0, 1.0 - error_sum))
    return best_bound


def ar1_innovation_schedule_bound(
    deltas: np.ndarray,
    sigmas: np.ndarray,
    rhos: np.ndarray,
    k_samples: int,
    run_exhaustive: bool = False,
) -> ScheduleBoundResult:
    """Compute primal dynamic-program schedule and minimax dual bound for AR(1) traces.

    Args:
        deltas: Array of rival trace-difference waveforms, shape (n_scenarios, n_grid, 2).
        sigmas: Per-scenario marginal SDs, shape (n_scenarios,).
        rhos: Per-scenario lag-1 correlations in [0, 1), shape (n_scenarios,).
        k_samples: Number of observation times to select.
        run_exhaustive: Whether to also enumerate all C(n_grid, k_samples) schedules
            when n_grid <= 30 and k_samples <= 5.

    Returns:
        ScheduleBoundResult with feasible DP schedule, worst-case distance, and dual bound.
    """
    n_scen, n_grid, _ = deltas.shape
    norm_sq = np.sum(deltas**2, axis=-1)
    first_reward = norm_sq / (sigmas[:, None] ** 2)

    trans_reward = np.zeros((n_scen, n_grid, n_grid), dtype=np.float64)
    for i in range(n_grid):
        for j in range(i + 1, n_grid):
            lag = j - i
            rho_lag = rhos[:, None] ** lag
            innov = deltas[:, j, :] - rho_lag * deltas[:, i, :]
            denom = (sigmas**2) * (1.0 - rhos ** (2 * lag))
            trans_reward[:, i, j] = np.sum(innov**2, axis=-1) / denom

    def _dp_for_weights(weights: np.ndarray) -> tuple[float, tuple[int, ...]]:
        a_w = weights @ first_reward
        b_w = np.tensordot(weights, trans_reward, axes=(0, 0))
        val = np.full((k_samples, n_grid), -np.inf, dtype=np.float64)
        parent = np.full((k_samples, n_grid), -1, dtype=np.int64)
        val[0] = a_w
        for step in range(1, k_samples):
            for j in range(step, n_grid):
                scores = val[step - 1, :j] + b_w[:j, j]
                best_i = int(np.argmax(scores))
                val[step, j] = scores[best_i]
                parent[step, j] = best_i
        last_j = int(np.argmax(val[k_samples - 1]))
        best_val = float(val[k_samples - 1, last_j])
        sched = [last_j]
        cur = last_j
        for step in range(k_samples - 1, 0, -1):
            cur = int(parent[step, cur])
            sched.append(cur)
        sched.reverse()
        return best_val, tuple(sched)

    def _eval_schedule(sched: tuple[int, ...]) -> np.ndarray:
        total = first_reward[:, sched[0]].copy()
        for step in range(1, k_samples):
            total += trans_reward[:, sched[step - 1], sched[step]]
        return total

    # Subgradient / L-BFGS optimization on simplex via softmax parameterization
    best_primal_val = -np.inf
    best_primal_sched: tuple[int, ...] = tuple(range(k_samples))
    best_dual = np.inf

    def _dual_obj(logits: np.ndarray) -> float:
        nonlocal best_primal_val, best_primal_sched, best_dual
        shifted = logits - np.max(logits)
        w = np.exp(shifted)
        w /= np.sum(w)
        u_val, sched = _dp_for_weights(w)
        scen_vals = _eval_schedule(sched)
        worst = float(np.min(scen_vals))
        if worst > best_primal_val:
            best_primal_val = worst
            best_primal_sched = sched
        if u_val < best_dual:
            best_dual = u_val
        return u_val

    res = minimize(
        _dual_obj,
        np.zeros(n_scen, dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": 1500, "xatol": 1e-9, "fatol": 1e-9},
    )
    _dual_obj(res.x)

    exhaustive_opt: float | None = None
    if run_exhaustive:
        exhaustive_opt = max(
            float(np.min(_eval_schedule(comb)))
            for comb in combinations(range(n_grid), k_samples)
        )
        if exhaustive_opt > best_primal_val:
            # Find exact maximizing schedule
            for comb in combinations(range(n_grid), k_samples):
                val = float(np.min(_eval_schedule(comb)))
                if val >= exhaustive_opt - 1e-15:
                    best_primal_val = val
                    best_primal_sched = comb
                    break

    return ScheduleBoundResult(
        best_dp_schedule=best_primal_sched,
        best_dp_worst_distance_sq=best_primal_val,
        dual_upper_bound=best_dual,
        exhaustive_optimum=exhaustive_opt,
    )


def decode_swapped_pairs(
    whitened_templates: np.ndarray,
    whitened_observations: np.ndarray,
    tie_priorities: np.ndarray | None = None,
) -> np.ndarray:
    """Decode unordered parent pairs from concatenated whitened two-record observations.

    Args:
        whitened_templates: Candidate templates of shape (n_pairs, 2, dim), where axis 1
            indexes the two consistent role assignments across the two swapped records.
        whitened_observations: Observed concatenated whitened records of shape (n_trials, dim).
        tie_priorities: Optional uniform random tie-breakers of shape (n_trials, n_pairs).

    Returns:
        Integer array of chosen pair indices of shape (n_trials,).
    """
    n_pairs, _, dim = whitened_templates.shape
    flat_templates = whitened_templates.reshape(2 * n_pairs, dim)
    sq_norms = np.sum(flat_templates**2, axis=1)
    scores = sq_norms[None, :] - 2.0 * (whitened_observations @ flat_templates.T)
    pair_losses = np.minimum(scores[:, 0::2], scores[:, 1::2])
    if tie_priorities is None:
        return np.argmin(pair_losses, axis=1)
    tied = pair_losses <= (np.min(pair_losses, axis=1, keepdims=True) + 1e-10)
    return np.argmin(np.where(tied, tie_priorities, 2.0), axis=1)


def _clopper_pearson_interval(k: int, n: int, alpha: float) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(0.5 * alpha, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - 0.5 * alpha, k + 1, n - k))
    return lo, hi


def paired_schedule_certificate(
    wins_p_only: int,
    wins_q_only: int,
    n_trials: int,
    alpha: float = 0.05,
    resolved_fraction: float = 1.0,
) -> tuple[float, float]:
    """Compute exact Clopper-Pearson paired schedule accuracy-difference certificate."""
    discordant = wins_p_only + wins_q_only
    d_lo, d_hi = _clopper_pearson_interval(discordant, n_trials, 0.5 * alpha)
    w_lo, w_hi = _clopper_pearson_interval(wins_p_only, discordant, 0.5 * alpha)
    corners = [
        d_val * (2.0 * w_val - 1.0)
        for d_val in (d_lo, d_hi)
        for w_val in (w_lo, w_hi)
    ]
    lo, hi = min(corners), max(corners)
    unres = 1.0 - resolved_fraction
    return resolved_fraction * lo - unres, resolved_fraction * hi + unres
