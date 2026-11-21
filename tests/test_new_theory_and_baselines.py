"""Unit tests for new architectural baselines, MemGRN-Int, and theoretical bounds."""

import numpy as np
import pytest

from memgrn.baselines import (
    DynamicalModelFamily,
    ModelSpec,
    build_jax_rollout,
    counterfactual_source_attribution,
)
from memgrn.dynamics import (
    MemGRNConfig,
    RateInputMode,
    initialize_memgrn_params,
    rollout_memgrn_rk4,
)
from memgrn.theory import (
    minimum_norm_collinear_gate_weights,
    verify_coupled_crosstalk_theorem,
    verify_iso_expression_proposition,
)


def test_crosstalk_theorem_and_iso_expression_proposition() -> None:
    w1, w2 = minimum_norm_collinear_gate_weights(w_star=2.0, rho=0.8, eta_var=0.0)
    assert np.isclose(w1 + 0.8 * w2, 2.0, atol=1e-12)
    assert np.isclose(w2 / w1, 0.8, atol=1e-12)

    ct = verify_coupled_crosstalk_theorem(w_star=2.4, rho=0.85)
    assert ct.coupled_ko_trajectory_l2_error > 0.02
    assert ct.coupled_false_source2_attribution_l2 > 0.02
    assert ct.local_ko_trajectory_l2_error < 1e-13
    assert ct.local_false_source2_attribution_l2 < 1e-13

    iso = verify_iso_expression_proposition()
    assert iso.washout_value_diff < 1e-11
    assert iso.memory_state_diff_at_washout > 0.02
    assert np.isclose(
        iso.best_state_only_post_washout_error_l2,
        iso.state_only_universal_lower_bound,
        rtol=1e-9,
    )


def test_memgrn_int_source_isolation_and_interaction_readout() -> None:
    cfg = MemGRNConfig(
        n_sources=3,
        n_targets=2,
        n_channels=3,
        rate_mode=RateInputMode.BOTH,
        use_pairwise_interactions=True,
    )
    params = initialize_memgrn_params(cfg, seed=7)
    assert params.interaction_weights is not None
    assert params.interaction_weights.shape == (3, 2)

    rng = np.random.default_rng(13)
    params = type(params)(
        direct_weights=params.direct_weights,
        delayed_weights=params.delayed_weights,
        relaxation=params.relaxation,
        rate_logits=params.rate_logits,
        input_proj=params.input_proj,
        hidden_bias=params.hidden_bias,
        output_proj=rng.normal(0.0, 0.5, size=(3, 3, 4)),
        interaction_weights=np.array([[1.2, -0.8], [0.0, 0.0], [0.0, 0.0]]),
    )
    u_a = rng.normal(0.0, 1.0, size=(40, 3, 3))
    u_b = u_a.copy()
    u_b[:, 2, :] = 0.0  # Perturbing source 2 must not alter z_0, h_0 or z_1, h_1!
    z_a, h_a, _ = rollout_memgrn_rk4(u_a, dt=0.05, params=params, config=cfg)
    z_b, h_b, _ = rollout_memgrn_rk4(u_b, dt=0.05, params=params, config=cfg)
    np.testing.assert_allclose(z_a[:, :2, :], z_b[:, :2, :], atol=1e-14)
    np.testing.assert_allclose(h_a[:, :2, :], h_b[:, :2, :], atol=1e-14)


@pytest.mark.parametrize("family", list(DynamicalModelFamily))
def test_all_jax_dynamical_model_families_zero_equilibrium(family: DynamicalModelFamily) -> None:
    spec = ModelSpec(family=family, n_sources=4, n_targets=3, n_channels=2, hidden_width=4)
    fwd, _, th0 = build_jax_rollout(spec, dt=0.05, obs_stride=4)
    zero_u = np.zeros((16, 2, 4, 3), dtype=np.float32)
    zero_y0 = np.zeros((2, 3), dtype=np.float32)
    out = np.asarray(fwd(th0, zero_u, zero_y0))
    assert out.shape == (2, 5, 3)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)

    # Check counterfactual attribution shape on non-zero input
    rng = np.random.default_rng(5)
    u_nz = rng.normal(0.0, 0.5, size=(16, 2, 4, 3)).astype(np.float32)
    effects = counterfactual_source_attribution(
        lambda u_in, y_in: np.asarray(
            fwd(
                th0,
                u_in.astype(np.float32),
                np.zeros((u_in.shape[1], 3), dtype=np.float32) if y_in is None else y_in,
            )
        ),
        u_nz,
        None,
    )
    assert effects.shape == (4, 2, 5, 3)
