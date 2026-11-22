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
    verify_unit_dc_gain_and_second_order_vs_local_ltc,
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

    ltc_sep = verify_unit_dc_gain_and_second_order_vs_local_ltc()
    assert ltc_sep.ltc_scale_gauge_trajectory_max_diff < 1e-12
    assert ltc_sep.ltc_scale_gauge_weight_disagreement > 0.5
    assert ltc_sep.ltc_amplitude_gain_distortion_ratio > 1.2
    assert ltc_sep.memgrn_amplitude_gain_max_error < 1e-10
    assert ltc_sep.memgrn_kinetic_change_steady_state_diff < 1e-10
    assert ltc_sep.first_order_post_shutoff_max_slope <= 0.0
    assert ltc_sep.memgrn_post_shutoff_peak_gain > 0.1
    assert ltc_sep.best_monotone_first_order_l2_floor > 0.03


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


def test_closed_loop_external_pulse_and_cooperativity_zero_equilibrium() -> None:
    spec = ClosedLoopSpec(
        family=ClosedLoopModelFamily.MEMGRN,
        n_genes=6,
        regulator_indices=(0, 1, 2),
        n_ext_inputs=3,
        use_cooperativity=True,
        coop_rank=4,
        n_channels=2,
        hidden_width=4,
    )
    fwd, _, th0 = build_closed_loop_jax_rollout(
        spec,
        basal_levels=np.ones(6, dtype=np.float32),
        dt=0.25,
        n_steps=12,
        obs_step_indices=(0, 4, 8, 12),
    )
    wt_ko = np.zeros((2, 6), dtype=np.float32)
    wt_y0 = np.zeros((2, 6), dtype=np.float32)
    zero_u = np.zeros((12, 2, 3), dtype=np.float32)
    out_zero = np.asarray(fwd(th0, wt_ko, wt_y0, zero_u, None))
    np.testing.assert_allclose(out_zero, 0.0, atol=1e-6)

    pulse_u = np.zeros((12, 2, 3), dtype=np.float32)
    pulse_u[:6, 0, 0] = 1.0
    out_pulse = np.asarray(fwd(th0, wt_ko, wt_y0, pulse_u, None))
    assert out_pulse.shape == (2, 4, 6)
    assert np.all(np.isfinite(out_pulse))
    assert float(np.max(np.abs(out_pulse[0]))) > 1e-4


def test_ortho_ood_generalization_and_selective_ssm_corollary() -> None:
    from memgrn.theory import (
        verify_ortho_ood_generalization_theorem,
        verify_selective_ssm_crosstalk_corollary,
    )

    ortho_ood = verify_ortho_ood_generalization_theorem()
    assert ortho_ood.ood_counterfactual_drift_ortho < 1e-13
    assert ortho_ood.analytical_vs_empirical_ood_risk_diff < 1e-12
    assert ortho_ood.ood_mse_memgrn_ortho < 0.25 * ortho_ood.ood_mse_observed_only
    assert ortho_ood.ood_counterfactual_drift_unproj > 0.1

    ssm_ct = verify_selective_ssm_crosstalk_corollary()
    assert ssm_ct.ssm_nonparent_delta_weight > 0.5
    assert ssm_ct.ssm_ko_trajectory_l2_error > 0.05
    assert ssm_ct.memgrn_ko_trajectory_l2_error < 1e-13


def test_factorized_coop_and_hysteresis_extraction() -> None:
    from memgrn.baselines import extract_memgrn_up_down

    spec = ModelSpec(
        family=DynamicalModelFamily.MEMGRN,
        n_sources=6,
        n_targets=10,
        n_channels=3,
        coop_rank=4,
        use_cooperativity=True,
    )
    fwd, unpack, th0 = build_jax_rollout(spec, dt=0.2, obs_stride=2)
    p = {k: np.asarray(v) for k, v in unpack(th0).items()}
    r_up, r_dn, v_up, v_dn = extract_memgrn_up_down(p)
    np.testing.assert_allclose(r_up, r_dn, atol=1e-7)
    np.testing.assert_allclose(v_up, v_dn, atol=1e-7)
    u_zero = np.zeros((8, 2, 6, 3), dtype=np.float32)
    y0 = np.zeros((2, 10), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(fwd(th0, u_zero, y0)), 0.0, atol=1e-6)


def test_closed_loop_diagonal_self_memory_and_hollow_graph() -> None:
    from memgrn.baselines import fit_closed_loop_network

    rng = np.random.default_rng(17)
    b, n_g = 4, 5
    y = rng.normal(0.0, 0.5, (b, 13, n_g))
    w = np.ones_like(y)
    w[:, :7, :] = 0.0
    ko = np.zeros((b, n_g))
    y_cl = np.zeros((12, b, n_g))
    y_cl[:6] = y[:, :6, :].transpose(1, 0, 2)
    spec = ClosedLoopSpec(
        family=ClosedLoopModelFamily.MEMGRN,
        n_genes=n_g,
        regulator_indices=tuple(range(n_g)),
        hollow=True,
        use_self_memory=True,
        n_channels=2,
        hidden_width=4,
    )
    fwd, _, th0 = build_closed_loop_jax_rollout(
        spec, basal_levels=np.ones(n_g, dtype=np.float32), dt=0.25, n_steps=12, obs_step_indices=tuple(range(13))
    )
    out_wt = np.asarray(fwd(th0, np.zeros((2, n_g), dtype=np.float32), np.zeros((2, n_g), dtype=np.float32)))
    np.testing.assert_allclose(out_wt, 0.0, atol=1e-6)

    fitted = fit_closed_loop_network(
        spec=spec,
        basal_levels=np.ones(n_g),
        train_ko=ko,
        train_y=y,
        train_weights=w,
        train_y_clamp=y_cl,
        val_ko=ko,
        val_y=y,
        val_weights=w,
        val_y_clamp=y_cl,
        dt=0.25,
        n_steps=12,
        obs_step_indices=tuple(range(13)),
        clamp_steps=6,
        seeds=(0,),
        reg_grid=(1e-3,),
        total_updates=5,
    )
    assert np.isfinite(fitted.best_val_loss)
    assert float(np.max(np.abs(np.diag(fitted.interaction_matrix)))) == 0.0


def test_per_target_memgrn_ortho_zero_drift_and_discrete_baselines() -> None:
    from memgrn.baselines import fit_dynamical_model, fit_two_stage_memgrn_ortho
    from memgrn.discrete_baselines import (
        DiscreteModelFamily,
        discrete_counterfactual_attribution,
        fit_discrete_closed_loop,
        fit_discrete_open_loop,
    )

    rng = np.random.default_rng(23)
    u = rng.normal(0.0, 0.7, (6, 12, 3))
    y = rng.normal(0.0, 0.7, (6, 12, 4))
    m = np.ones_like(y)
    m[:, 0, :] = 0.0
    u_stage = np.zeros((11, 6, 3, 3))
    for k in range(3):
        u_stage[:, :, :, k] = u[:, :-1, :].transpose(1, 0, 2)

    base_spec = ModelSpec(
        family=DynamicalModelFamily.MEMGRN,
        n_sources=3,
        n_targets=4,
        n_channels=2,
        hidden_width=4,
        coop_rank=3,
        use_cooperativity=True,
    )
    base_fit = fit_dynamical_model(
        spec=base_spec,
        train_stage_u=u_stage[:, :4],
        train_y=y[:4],
        train_mask=m[:4],
        val_stage_u=u_stage[:, 4:],
        val_y=y[4:],
        val_mask=m[4:],
        dt=0.5,
        obs_stride=1,
        seeds=(0,),
        reg_grid=(1e-3,),
        total_updates=5,
    )
    ortho_fit = fit_two_stage_memgrn_ortho(
        base_memgrn=base_fit,
        train_stage_u=u_stage[:, :4],
        train_y=y[:4],
        train_mask=m[:4],
        val_stage_u=u_stage[:, 4:],
        val_y=y[4:],
        val_mask=m[4:],
        dt=0.5,
        obs_stride=1,
        latent_dim=10,
        seeds=(0,),
        reg_grid=(1e-4, 1e-3),
        total_updates=5,
        residual_families=(DynamicalModelFamily.LATENT_ODE, DynamicalModelFamily.CFC),
        per_target_blend=True,
    )
    a0 = counterfactual_source_attribution(base_fit.predict_fn, u_stage[:, 4:])
    a1 = counterfactual_source_attribution(ortho_fit.predict_observed_only_fn, u_stage[:, 4:])
    np.testing.assert_allclose(a0, a1, atol=1e-12)

    for fam in (DiscreteModelFamily.DLINEAR, DiscreteModelFamily.GRU, DiscreteModelFamily.TRANSFORMER):
        ol = fit_discrete_open_loop(
            family=fam,
            train_u=u[:4],
            train_y=y[:4],
            train_mask=m[:4],
            val_u=u[4:],
            val_y=y[4:],
            val_mask=m[4:],
            hidden_dim=12,
            seeds=(0,),
            reg_grid=(1e-3,),
            total_updates=4,
        )
        assert np.isfinite(ol.best_val_loss)
        attr = discrete_counterfactual_attribution(ol.predict_fn, u[4:])
        assert attr.shape == (3, 2, 12, 4)
        cl = fit_discrete_closed_loop(
            family=fam,
            train_hist=y[:4, :6, :],
            train_fut=y[:4, 6:, :],
            train_fut_mask=np.ones((4, 6, 4)),
            val_hist=y[4:, :6, :],
            val_fut=y[4:, 6:, :],
            val_fut_mask=np.ones((2, 6, 4)),
            hidden_dim=12,
            seeds=(0,),
            reg_grid=(1e-3,),
            total_updates=4,
        )
        assert np.isfinite(cl.best_val_loss)
        assert cl.interaction_matrix.shape == (4, 4)
        assert float(np.max(np.abs(np.diag(cl.interaction_matrix)))) == 0.0
