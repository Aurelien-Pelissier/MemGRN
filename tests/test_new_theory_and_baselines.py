"""Unit tests for unified MemGRN, closed-loop networks, MemGRN-Ortho, and theory."""

import numpy as np
import pytest

from memgrn.baselines import (
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    DynamicalModelFamily,
    ModelSpec,
    build_closed_loop_jax_rollout,
    build_jax_rollout,
    counterfactual_source_attribution,
)
from memgrn.dynamics import (
    MemGRNConfig,
    RateInputMode,
    initialize_memgrn_params,
    rollout_closed_loop_memgrn_rk4,
    rollout_memgrn_rk4,
)
from memgrn.theory import (
    minimum_norm_collinear_gate_weights,
    verify_closed_loop_stability_and_identifiability_theorem,
    verify_coupled_crosstalk_theorem,
    verify_general_s_covariance_crosstalk_theorem,
    verify_iso_expression_proposition,
    verify_orthogonal_cofactor_invariance_proposition,
)


def test_crosstalk_general_s_and_iso_expression_theorems() -> None:
    w1, w2 = minimum_norm_collinear_gate_weights(w_star=2.0, rho=0.8, eta_var=0.0)
    assert np.isclose(w1 + 0.8 * w2, 2.0, atol=1e-12)
    assert np.isclose(w2 / w1, 0.8, atol=1e-12)

    ct = verify_coupled_crosstalk_theorem(w_star=2.4, rho=0.85)
    assert ct.coupled_ko_trajectory_l2_error > 0.02
    assert ct.local_ko_trajectory_l2_error < 1e-13

    gen_s = verify_general_s_covariance_crosstalk_theorem(n_sources=12, k_active=2)
    assert gen_s.analytical_vs_solve_max_diff < 1e-12
    assert gen_s.coupled_nonparent_weight_l2 > 0.2
    assert gen_s.local_nonparent_weight_l2 == 0.0
    assert gen_s.finite_sample_coupled_risk_s24 > 4.0 * gen_s.finite_sample_local_l1_risk_s24

    iso = verify_iso_expression_proposition()
    assert iso.washout_value_diff < 1e-11
    assert iso.memory_state_diff_at_washout > 0.02
    assert np.isclose(
        iso.best_state_only_post_washout_error_l2,
        iso.state_only_universal_lower_bound,
        rtol=1e-9,
    )


def test_closed_loop_identifiability_and_orthogonal_cofactor_theorems() -> None:
    cl = verify_closed_loop_stability_and_identifiability_theorem(n_nodes=5)
    assert cl.within_absorbing_ball
    assert cl.max_filter_abs <= 1.0 + 1e-9
    assert cl.coupled_gauge_trajectory_max_diff < 1e-12
    assert cl.coupled_gauge_attribution_disagreement > 0.25
    assert cl.memgrn_direct_weight_recovery_max_error < 1e-11
    assert cl.memgrn_total_weight_recovery_max_error < 1e-8

    ortho = verify_orthogonal_cofactor_invariance_proposition()
    assert ortho.orthogonality_max_inner_product < 1e-12
    assert ortho.observed_weight_invariance_error < 1e-12
    assert ortho.unprojected_observed_weight_theft_rel_error > 0.15
    assert ortho.mse_memgrn_ortho < 0.1 * ortho.mse_observed_only


def test_asymmetric_source_isolation_and_closed_loop_numpy_rollout() -> None:
    cfg = MemGRNConfig(
        n_sources=3,
        n_targets=3,
        n_channels=3,
        rate_mode=RateInputMode.BOTH,
        use_pairwise_interactions=True,
        use_asymmetric_rates=True,
    )
    params = initialize_memgrn_params(cfg, seed=7)
    rng = np.random.default_rng(13)
    params = type(params)(
        direct_weights=params.direct_weights,
        delayed_weights=params.delayed_weights,
        relaxation=params.relaxation,
        rate_logits=params.rate_logits,
        rate_logits_down=params.rate_logits_down,
        input_proj=params.input_proj,
        hidden_bias=params.hidden_bias,
        output_proj=rng.normal(0.0, 0.5, size=(3, 3, 4)),
        output_proj_down=rng.normal(0.0, 0.5, size=(3, 3, 4)),
        interaction_weights=np.array([[1.2, -0.8, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    u_a = rng.normal(0.0, 1.0, size=(30, 3, 3))
    u_b = u_a.copy()
    u_b[:, 2, :] = 0.0
    z_a, h_a, _ = rollout_memgrn_rk4(u_a, dt=0.05, params=params, config=cfg)
    z_b, h_b, _ = rollout_memgrn_rk4(u_b, dt=0.05, params=params, config=cfg)
    np.testing.assert_allclose(z_a[:, :2, :], z_b[:, :2, :], atol=1e-14)
    np.testing.assert_allclose(h_a[:, :2, :], h_b[:, :2, :], atol=1e-14)

    z_cl, h_cl, y_cl = rollout_closed_loop_memgrn_rk4(
        n_steps=20,
        dt=0.1,
        params=params,
        config=cfg,
        ko_mask=np.array([1.0, 0.0, 0.0]),
        basal_levels=np.array([1.2, 0.8, 1.0]),
    )
    assert y_cl.shape == (21, 3)
    assert np.max(np.abs(z_cl)) <= 1.0 + 1e-9
    assert np.max(np.abs(h_cl)) <= 1.0 + 1e-9


@pytest.mark.parametrize("family", list(DynamicalModelFamily))
def test_all_jax_dynamical_model_families_zero_equilibrium(family: DynamicalModelFamily) -> None:
    spec = ModelSpec(family=family, n_sources=4, n_targets=3, n_channels=2, hidden_width=4)
    fwd, _, th0 = build_jax_rollout(spec, dt=0.05, obs_stride=4)
    zero_u = np.zeros((16, 2, 4, 3), dtype=np.float32)
    zero_y0 = np.zeros((2, 3), dtype=np.float32)
    out = np.asarray(fwd(th0, zero_u, zero_y0))
    assert out.shape == (2, 5, 3)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)

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


@pytest.mark.parametrize("cl_family", list(ClosedLoopModelFamily))
def test_all_closed_loop_jax_families_wildtype_equilibrium(
    cl_family: ClosedLoopModelFamily,
) -> None:
    spec = ClosedLoopSpec(family=cl_family, n_genes=6, n_channels=2, hidden_width=4, latent_dim=8)
    fwd, _, th0 = build_closed_loop_jax_rollout(
        spec,
        basal_levels=np.ones(6, dtype=np.float32),
        dt=0.25,
        n_steps=12,
        obs_step_indices=(0, 4, 8, 12),
    )
    wt_ko = np.zeros((2, 6), dtype=np.float32)
    wt_y0 = np.zeros((2, 6), dtype=np.float32)
    out_wt = np.asarray(fwd(th0, wt_ko, wt_y0))
    assert out_wt.shape == (2, 4, 6)
    np.testing.assert_allclose(out_wt, 0.0, atol=1e-6)
