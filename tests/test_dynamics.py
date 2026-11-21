"""Unit tests for MemGRN dynamics, boundedness, and exact response accounting."""

import numpy as np
import pytest

from memgrn.dynamics import (
    MemGRNConfig,
    RateInputMode,
    adaptive_rate_law,
    compute_exact_contributions,
    initialize_memgrn_params,
    memory_reset_splice_decomposition,
    rollout_memgrn_rk4,
)


@pytest.mark.parametrize(
    "mode",
    [
        RateInputMode.CONSTANT,
        RateInputMode.ACTIVITY,
        RateInputMode.MEMORY,
        RateInputMode.BOTH,
    ],
)
def test_rate_bounds_and_filter_invariant_interval(mode: RateInputMode) -> None:
    cfg = MemGRNConfig(
        n_sources=3,
        n_targets=2,
        n_channels=3,
        rate_min=0.05,
        rate_max=14.0,
        rate_mode=mode,
    )
    params = initialize_memgrn_params(cfg, seed=42)
    rng = np.random.default_rng(7)
    raw_u = rng.normal(0.0, 1.5, size=(120, 3, 3))
    amp_bound = float(np.max(np.abs(np.tanh(raw_u))))

    rates = adaptive_rate_law(
        raw_u[:, :, 0],
        np.tanh(raw_u[:, :, 0]),
        np.zeros((120, 3, 3)),
        np.zeros((120, 3, 3)),
        params,
        cfg,
    )
    assert np.all(rates >= cfg.rate_min)
    assert np.all(rates <= cfg.rate_max)

    z_traj, h_traj, y_traj = rollout_memgrn_rk4(raw_u, dt=0.02, params=params, config=cfg)
    assert z_traj.shape == (121, 3, 3)
    assert h_traj.shape == (121, 3, 3)
    assert y_traj.shape == (121, 2)
    assert float(np.max(np.abs(z_traj))) <= amp_bound + 1e-9
    assert float(np.max(np.abs(h_traj))) <= amp_bound + 1e-9


def test_exact_additive_source_contributions_and_removal() -> None:
    cfg = MemGRNConfig(
        n_sources=3,
        n_targets=2,
        n_channels=3,
        rate_mode=RateInputMode.BOTH,
    )
    params = initialize_memgrn_params(cfg, seed=19)
    rng = np.random.default_rng(23)
    # Give non-zero output_proj so rate adaptation is strongly nonlinear
    params = type(params)(
        direct_weights=rng.normal(0.0, 0.5, size=(3, 2)),
        delayed_weights=rng.normal(0.0, 0.5, size=(3, 3, 2)),
        relaxation=np.array([0.4, 0.8]),
        rate_logits=params.rate_logits,
        input_proj=params.input_proj,
        hidden_bias=params.hidden_bias,
        output_proj=rng.normal(0.0, 0.7, size=(3, 3, 4)),
    )
    t_steps = 150
    dt = 0.025
    t3 = np.stack(
        [
            np.arange(t_steps) * dt,
            (np.arange(t_steps) + 0.5) * dt,
            (np.arange(t_steps) + 1.0) * dt,
        ],
        axis=-1,
    )
    raw_u = np.stack(
        [np.sin(1.2 * t3), 0.7 * np.cos(0.8 * t3), -0.5 * np.sin(2.1 * t3)], axis=1
    )
    y0 = np.array([0.25, -0.15])
    y_full, decay_term, contrib = compute_exact_contributions(
        raw_u, dt, params, cfg, y0=y0
    )
    np.testing.assert_allclose(
        y_full, decay_term + np.sum(contrib, axis=1), rtol=1e-9, atol=1e-10
    )

    # Zeroing source 0 must remove exactly contrib[:, 0, :] without perturbing sources 1, 2
    raw_zero_0 = raw_u.copy()
    raw_zero_0[:, 0, :] = 0.0
    _, _, y_no_0 = rollout_memgrn_rk4(raw_zero_0, dt, params, cfg, y0=y0)
    np.testing.assert_allclose(
        y_full - y_no_0, contrib[:, 0, :], rtol=1e-11, atol=1e-12
    )


def test_memory_reset_splice_identity() -> None:
    cfg = MemGRNConfig(n_sources=1, n_targets=2, n_channels=3, rate_mode=RateInputMode.BOTH)
    params = initialize_memgrn_params(cfg, seed=101)
    rng = np.random.default_rng(102)
    raw_u = np.abs(rng.normal(0.0, 0.9, size=(120, 1, 3)))
    raw_u[80:, :, :] = 0.0
    decomp = memory_reset_splice_decomposition(
        raw_u, dt=0.02, params=params, config=cfg, reset_steps={30, 60}, shutoff_step=80
    )
    np.testing.assert_allclose(
        decomp["intact"][80:] - decomp["reset"][80:],
        decomp["carryover_after_shutoff"] + decomp["post_shutoff_production"],
        rtol=1e-12,
        atol=1e-13,
    )
