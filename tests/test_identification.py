"""Unit tests for strength-swap identification, margin bounds, and schedule certificates."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from memgrn.identification import (
    ar1_innovation_schedule_bound,
    decode_swapped_pairs,
    exact_swap_cancellation_accuracy,
    invert_swap_waveforms,
    paired_schedule_certificate,
    swap_singular_values,
    template_margin_lower_bound,
)


@pytest.mark.parametrize(
    ("r_val", "deg"),
    [(0.1, 15), (0.5, 30), (1.2, 45), (2.5, 75), (1.0, 90)],
)
def test_theorem1_exact_cancellation_matches_quadrature(r_val: float, deg: int) -> None:
    theta = np.deg2rad(deg)
    closed = exact_swap_cancellation_accuracy(r_val, theta)

    def _integrand(x: float) -> float:
        hi = (abs(x) - np.cos(theta) * x) / np.sin(theta)
        lo = (-abs(x) - np.cos(theta) * x) / np.sin(theta)
        return float(norm.pdf(x - r_val) * (norm.cdf(hi) - norm.cdf(lo)))

    num = (
        quad(_integrand, -np.inf, 0.0, epsabs=1e-12)[0]
        + quad(_integrand, 0.0, np.inf, epsabs=1e-12)[0]
    )
    assert abs(closed - num) < 1e-11


def test_theorem1_parallel_directions_give_chance() -> None:
    assert abs(exact_swap_cancellation_accuracy(2.0, 0.0) - 0.5) < 1e-15


def test_swap_inversion_and_singular_values() -> None:
    a, b = 0.2, 0.8
    sv_plus, sv_minus = swap_singular_values(a, b, apply_tanh=True)
    assert sv_plus > sv_minus > 0.0

    t = np.linspace(0.0, 4.0, 40)
    h1 = np.sin(t) * np.exp(-0.4 * t)
    h2 = -np.cos(t) * np.exp(-0.7 * t)
    r1 = np.tanh(a) * h1 + np.tanh(b) * h2
    r2 = np.tanh(b) * h1 + np.tanh(a) * h2
    h1_hat, h2_hat = invert_swap_waveforms(r1, r2, a, b)
    np.testing.assert_allclose(h1_hat, h1, atol=1e-14)
    np.testing.assert_allclose(h2_hat, h2, atol=1e-14)


def test_ar1_schedule_dp_and_certificate() -> None:
    t = np.linspace(0.2, 2.4, 10)
    deltas = np.stack(
        [
            np.stack([np.exp(-0.5 * t), 0.4 * np.exp(-1.1 * t)], axis=-1),
            np.stack([0.6 * np.exp(-0.8 * t), np.sin(t) * np.exp(-0.6 * t)], axis=-1),
        ],
        axis=0,
    )
    res = ar1_innovation_schedule_bound(
        deltas,
        sigmas=np.array([0.08, 0.08]),
        rhos=np.array([0.5, 0.8]),
        k_samples=3,
        run_exhaustive=True,
    )
    assert res.exhaustive_optimum is not None
    assert res.best_dp_worst_distance_sq <= res.exhaustive_optimum + 1e-12
    assert res.exhaustive_optimum <= res.dual_upper_bound + 1e-9

    lo, hi = paired_schedule_certificate(80877, 70626, 524288, alpha=0.015)
    assert 0.015 < lo < hi < 0.025


def test_margin_bound_and_decoder() -> None:
    templates = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    obs = np.array([[0.9, 0.05], [0.02, -0.95]])
    preds = decode_swapped_pairs(templates, obs)
    np.testing.assert_array_equal(preds, np.array([0, 1]))

    lb = template_margin_lower_bound(
        true_mean=np.array([1.0, 0.0]),
        true_templates=templates[0],
        wrong_templates=templates[1],
    )
    assert 0.5 < lb < 1.0
