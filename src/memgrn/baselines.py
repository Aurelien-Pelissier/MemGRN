"""Unified JAX open-loop, orthogonal-cofactor, and closed-loop dynamical models.

Implements matched continuous-time RK4 architectures sharing data splits, clipped cosine-Adam
optimization, validation selection, and counterfactual attribution across both open-loop
(regulator-driven) and closed-loop (autonomous gene-network) settings:
  - local_const:      Source-isolated constant-rate filter cascade + explicit readout
  - memgrn:           Unified source-isolated asymmetric rise/recovery filter + sparse cooperative readout
  - memgrn_additive:  Additive-readout ablation of unified MemGRN (Gamma = 0)
  - memgrn_sym:       Symmetric-rate ablation of MemGRN (lambda_up == lambda_down)
  - memgrn_ortho:     MemGRN + L2(P_train)-orthogonalized low-rank latent cofactor module
  - coupled_const:    Source-coupled constant-rate filter cascade + explicit readout
  - coupled_adaptive: Source-coupled adaptive-rate filter cascade + explicit readout
  - ltc_coupled:      Liquid Time-Constant Network (Hasani et al., 2021), coupled state
  - ltc_local:        Source-isolated Liquid Time-Constant Network ablation
  - perturb_ode:      PerturbODE (Lin et al., 2025) module-structured state-only ODE
  - latent_ode:       Unconstrained Latent Neural ODE (Rubanova et al., 2019)
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


class DynamicalModelFamily(StrEnum):
    """Supported continuous-time open-loop / conditional dynamical architectures."""

    LOCAL_CONST = "local_const"
    MEMGRN = "memgrn"
    MEMGRN_ADDITIVE = "memgrn_additive"
    MEMGRN_SYM = "memgrn_sym"
    MEMGRN_ORTHO = "memgrn_ortho"
    COUPLED_CONST = "coupled_const"
    COUPLED_ADAPTIVE = "coupled_adaptive"
    LTC_COUPLED = "ltc_coupled"
    LTC_LOCAL = "ltc_local"
    PERTURB_ODE = "perturb_ode"
    LATENT_ODE = "latent_ode"


class ClosedLoopModelFamily(StrEnum):
    """Supported autonomous closed-loop gene regulatory network architectures."""

    MEMGRN = "cl_memgrn"
    LOCAL_CONST = "cl_local_const"
    COUPLED_ADAPTIVE = "cl_coupled_adaptive"
    LTC_COUPLED = "cl_ltc_coupled"
    LTC_LOCAL = "cl_ltc_local"
    PERTURB_ODE = "cl_perturb_ode"
    LATENT_ODE = "cl_latent_ode"


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    """Hyperparameter and dimension specification for an open-loop dynamical model."""

    family: DynamicalModelFamily
    n_sources: int
    n_targets: int
    n_channels: int = 3
    hidden_width: int = 6
    latent_dim: int = 16
    ortho_dim: int = 6
    module_dim: int = 12
    rate_min: float = 0.05
    rate_max: float = 14.0


@dataclass(frozen=True, kw_only=True)
class ClosedLoopSpec:
    """Specification for an autonomous N-gene closed-loop dynamical network."""

    family: ClosedLoopModelFamily
    n_genes: int
    regulator_indices: tuple[int, ...] | None = None
    n_channels: int = 3
    hidden_width: int = 6
    module_dim: int = 20
    latent_dim: int = 20
    rate_min: float = 0.08
    rate_max: float = 6.0


@dataclass(frozen=True, kw_only=True)
class TrainedModel:
    """Fitted open-loop model parameters, diagnostics, and compiled predictor."""

    spec: ModelSpec
    theta: np.ndarray
    n_params: int
    best_val_loss: float
    chosen_reg: float
    chosen_seed: int
    predict_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    predict_observed_only_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray] | None = None


@dataclass(frozen=True, kw_only=True)
class TrainedClosedLoopModel:
    """Fitted autonomous closed-loop network model, predictor, and inferred causal graph."""

    spec: ClosedLoopSpec
    theta: np.ndarray
    n_params: int
    best_val_loss: float
    chosen_reg: float
    chosen_seed: int
    predict_ko_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    interaction_matrix: np.ndarray


def _init_tree(spec: ModelSpec, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    s_dim, g_dim, r_dim = spec.n_sources, spec.n_targets, spec.n_channels
    h_dim = spec.hidden_width

    base_rates = np.resize(np.array([0.45, 1.2, 3.2], dtype=np.float32), r_dim)
    base_logits = np.log((base_rates - spec.rate_min) / (spec.rate_max - base_rates))

    fam = spec.family
    if fam in (
        DynamicalModelFamily.LOCAL_CONST,
        DynamicalModelFamily.MEMGRN,
        DynamicalModelFamily.MEMGRN_ADDITIVE,
        DynamicalModelFamily.MEMGRN_SYM,
        DynamicalModelFamily.MEMGRN_ORTHO,
    ):
        tree: dict[str, np.ndarray] = {
            "direct": rng.normal(0.0, 0.02, size=(s_dim, g_dim)).astype(np.float32),
            "delayed": rng.normal(0.0, 0.02, size=(s_dim, r_dim, g_dim)).astype(np.float32),
            "decay": np.zeros((g_dim,), dtype=np.float32),
            "rate": np.broadcast_to(base_logits, (s_dim, r_dim)).copy(),
        }
        if fam != DynamicalModelFamily.LOCAL_CONST:
            tree["in_proj"] = rng.normal(0.0, 0.7, size=(3, h_dim)).astype(np.float32)
            tree["bias"] = rng.normal(0.0, 0.25, size=(h_dim,)).astype(np.float32)
            tree["out_proj"] = np.zeros((s_dim, r_dim, h_dim), dtype=np.float32)
            if fam != DynamicalModelFamily.MEMGRN_SYM:
                tree["rate_dn"] = np.broadcast_to(base_logits, (s_dim, r_dim)).copy()
                tree["out_proj_dn"] = np.zeros((s_dim, r_dim, h_dim), dtype=np.float32)
        if fam == DynamicalModelFamily.MEMGRN and s_dim >= 2:
            n_pairs = s_dim * (s_dim - 1) // 2
            tree["interact"] = np.zeros((n_pairs, g_dim), dtype=np.float32)
        if fam == DynamicalModelFamily.MEMGRN_ORTHO:
            k_dim = spec.ortho_dim
            # Pulse-clock + weak input driven latent cofactor ODE
            tree["c_rate"] = np.linspace(-0.8, 0.8, k_dim).astype(np.float32)
            tree["c_ws"] = rng.normal(0.0, 0.25 / np.sqrt(k_dim), size=(k_dim, k_dim)).astype(
                np.float32
            )
            tree["c_wu"] = rng.normal(0.0, 0.5, size=(s_dim + 1, k_dim)).astype(np.float32)
            tree["c_bias"] = rng.normal(0.0, 0.15, size=(k_dim,)).astype(np.float32)
            tree["c_readout"] = rng.normal(0.0, 0.04, size=(k_dim, g_dim)).astype(np.float32)
        return tree

    if fam in (DynamicalModelFamily.COUPLED_CONST, DynamicalModelFamily.COUPLED_ADAPTIVE):
        m_dim = s_dim * r_dim
        eye_mix = np.repeat(np.eye(s_dim, dtype=np.float32), r_dim, axis=1)
        mix_init = eye_mix + rng.normal(0.0, 0.08, size=(s_dim, m_dim)).astype(np.float32)
        tree = {
            "mix_in": mix_init,
            "direct": rng.normal(0.0, 0.02, size=(s_dim, g_dim)).astype(np.float32),
            "delayed": rng.normal(0.0, 0.02, size=(m_dim, g_dim)).astype(np.float32),
            "decay": np.zeros((g_dim,), dtype=np.float32),
            "rate": np.tile(base_logits, s_dim).astype(np.float32),
        }
        if fam == DynamicalModelFamily.COUPLED_ADAPTIVE:
            cpl_w = max(h_dim, 8)
            tree["in_a"] = rng.normal(0.0, 0.5 / np.sqrt(s_dim), size=(s_dim, cpl_w)).astype(
                np.float32
            )
            tree["in_zh"] = rng.normal(
                0.0, 0.5 / np.sqrt(m_dim), size=(2 * m_dim, cpl_w)
            ).astype(np.float32)
            tree["bias"] = rng.normal(0.0, 0.2, size=(cpl_w,)).astype(np.float32)
            tree["out_proj"] = np.zeros((cpl_w, m_dim), dtype=np.float32)
        return tree

    if fam == DynamicalModelFamily.LTC_COUPLED:
        m_dim = s_dim * r_dim
        return {
            "tau_log": np.tile(base_logits, s_dim).astype(np.float32),
            "w_x": rng.normal(0.0, 0.35 / np.sqrt(m_dim), size=(m_dim, m_dim)).astype(
                np.float32
            ),
            "w_a": rng.normal(0.0, 0.6 / np.sqrt(s_dim), size=(s_dim, m_dim)).astype(
                np.float32
            ),
            "b_gate": rng.normal(0.0, 0.2, size=(m_dim,)).astype(np.float32),
            "e_rev": rng.normal(0.0, 0.6, size=(m_dim,)).astype(np.float32),
            "direct": rng.normal(0.0, 0.02, size=(s_dim, g_dim)).astype(np.float32),
            "readout": rng.normal(0.0, 0.03, size=(m_dim, g_dim)).astype(np.float32),
            "decay": np.zeros((g_dim,), dtype=np.float32),
        }

    if fam == DynamicalModelFamily.LTC_LOCAL:
        return {
            "tau_log": np.broadcast_to(base_logits, (s_dim, r_dim)).copy(),
            "w_x": rng.normal(0.0, 0.35, size=(s_dim, r_dim, r_dim)).astype(np.float32),
            "w_a": rng.normal(0.0, 0.7, size=(s_dim, r_dim)).astype(np.float32),
            "b_gate": rng.normal(0.0, 0.2, size=(s_dim, r_dim)).astype(np.float32),
            "e_rev": np.ones((s_dim, r_dim), dtype=np.float32),
            "direct": rng.normal(0.0, 0.02, size=(s_dim, g_dim)).astype(np.float32),
            "delayed": rng.normal(0.0, 0.02, size=(s_dim, r_dim, g_dim)).astype(np.float32),
            "decay": np.zeros((g_dim,), dtype=np.float32),
        }

    if fam == DynamicalModelFamily.PERTURB_ODE:
        mod_d = spec.module_dim
        return {
            "w_enc": rng.normal(0.0, 0.15 / np.sqrt(g_dim), size=(g_dim, mod_d)).astype(
                np.float32
            ),
            "b_mod": rng.normal(0.0, 0.1, size=(mod_d,)).astype(np.float32),
            "w_dec": np.zeros((mod_d, g_dim), dtype=np.float32),
            "direct": rng.normal(0.0, 0.03, size=(s_dim, g_dim)).astype(np.float32),
            "decay": np.zeros((g_dim,), dtype=np.float32),
        }

    if fam == DynamicalModelFamily.LATENT_ODE:
        lat_d = spec.latent_dim
        return {
            "rate": np.zeros((lat_d,), dtype=np.float32),
            "w_s": rng.normal(0.0, 0.25 / np.sqrt(lat_d), size=(lat_d, lat_d)).astype(
                np.float32
            ),
            "w_a": rng.normal(0.0, 0.5 / np.sqrt(s_dim), size=(s_dim, lat_d)).astype(
                np.float32
            ),
            "bias": rng.normal(0.0, 0.15, size=(lat_d,)).astype(np.float32),
            "w_dyn": rng.normal(0.0, 0.15 / np.sqrt(lat_d), size=(lat_d, lat_d)).astype(
                np.float32
            ),
            "readout": rng.normal(0.0, 0.08, size=(lat_d, g_dim)).astype(np.float32),
            "direct": rng.normal(0.0, 0.02, size=(s_dim, g_dim)).astype(np.float32),
            "decay": np.zeros((g_dim,), dtype=np.float32),
        }

    raise ValueError(f"Unsupported model family: {fam}")


def build_jax_rollout(
    spec: ModelSpec,
    dt: float,
    obs_stride: int,
    stimulus_window_steps: int | None = None,
    return_ortho_handles: bool = False,
):
    """Build JIT-compiled batched RK4 rollout for any DynamicalModelFamily."""
    template = _init_tree(spec, seed=0)
    template_j = {k: jnp.asarray(v) for k, v in template.items()}
    flat0, unpack = ravel_pytree(template_j)
    s_dim, g_dim, r_dim = spec.n_sources, spec.n_targets, spec.n_channels
    idx_i, idx_k = (
        np.triu_indices(s_dim, k=1) if s_dim >= 2 else (np.array([], dtype=int), np.array([], dtype=int))
    )
    fam = spec.family

    if fam == DynamicalModelFamily.MEMGRN_ORTHO:
        k_dim = spec.ortho_dim
        obs_feat_dim = s_dim * (1 + r_dim)

        def trace_features(theta: jnp.ndarray, stage_u: jnp.ndarray, y0: jnp.ndarray):
            """Integrate observed source-isolated channels AND latent cofactor state."""
            p = unpack(theta)
            decay = 0.1 + jnp.logaddexp(0.0, p["decay"])
            c_decay = 0.12 + jnp.logaddexp(0.0, p["c_rate"])
            rest_c = jnp.tanh(p["c_bias"])
            batch = stage_u.shape[1]
            n_steps = stage_u.shape[0]
            win_steps = stimulus_window_steps if stimulus_window_steps is not None else max(1, n_steps // 16)
            step_idx = jnp.arange(n_steps, dtype=theta.dtype)
            stim_amp = jnp.tanh(5.0 * jnp.max(jnp.abs(stage_u), axis=(0, 2, 3)))[None, :, None]
            pulse_clock_b = (step_idx < win_steps).astype(theta.dtype)[:, None, None] * stim_amp

            init_state = (
                jnp.zeros((batch, s_dim, r_dim), dtype=theta.dtype),
                jnp.zeros((batch, s_dim, r_dim), dtype=theta.dtype),
                jnp.zeros((batch, k_dim), dtype=theta.dtype),
                y0,
                jnp.zeros((batch, obs_feat_dim, g_dim), dtype=theta.dtype),
                jnp.zeros((batch, k_dim, g_dim), dtype=theta.dtype),
            )

            def rhs(state, inputs):
                z, h, psi, y_obs, phi_int, psi_int = state
                raw_u, clk = inputs
                act = jnp.tanh(raw_u)
                feat_a = jnp.broadcast_to(act[:, :, None], z.shape)
                hid = (
                    p["bias"]
                    + feat_a[..., None] * p["in_proj"][0]
                    + z[..., None] * p["in_proj"][1]
                    + h[..., None] * p["in_proj"][2]
                )
                th_hid = jnp.tanh(hid)
                logits_up = p["rate"][None] + jnp.sum(th_hid * p["out_proj"][None], axis=-1)
                logits_dn = p["rate_dn"][None] + jnp.sum(th_hid * p["out_proj_dn"][None], axis=-1)
                lam_up = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(logits_up)
                lam_dn = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(logits_dn)
                dz = lam_up * (act[:, :, None] - z)
                dh = lam_dn * (z - h)

                c_in = jnp.concatenate([clk, act], axis=-1)
                dpsi = -c_decay[None] * psi + (
                    jnp.tanh(psi @ p["c_ws"] + c_in @ p["c_wu"] + p["c_bias"]) - rest_c
                )

                obs_vec = jnp.concatenate([act, h.reshape(batch, s_dim * r_dim)], axis=-1)
                prod_obs = act @ p["direct"] + h.reshape(batch, s_dim * r_dim) @ p[
                    "delayed"
                ].reshape(s_dim * r_dim, g_dim)
                dy_obs = prod_obs - decay[None] * y_obs
                # Filtered leaky basis integrals for exact decay-matched orthogonal projection
                dphi_int = obs_vec[:, :, None] - decay[None, None, :] * phi_int
                dpsi_int = psi[:, :, None] - decay[None, None, :] * psi_int
                return dz, dh, dpsi, dy_obs, dphi_int, dpsi_int

            def step_fn(carry, step_pack):
                u_step, clk_step = step_pack
                u0, um, u1 = u_step[..., 0], u_step[..., 1], u_step[..., 2]
                k1 = rhs(carry, (u0, clk_step))
                k2 = rhs(jax.tree.map(lambda c, d: c + 0.5 * dt * d, carry, k1), (um, clk_step))
                k3 = rhs(jax.tree.map(lambda c, d: c + 0.5 * dt * d, carry, k2), (um, clk_step))
                k4 = rhs(jax.tree.map(lambda c, d: c + dt * d, carry, k3), (u1, clk_step))
                nxt = jax.tree.map(
                    lambda c, d1, d2, d3, d4: c + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4),
                    carry,
                    k1,
                    k2,
                    k3,
                    k4,
                )
                return nxt, (nxt[3], nxt[4], nxt[5])

            _, (ys_obs, phi_all, psi_all) = jax.lax.scan(
                step_fn, init_state, (stage_u, pulse_clock_b)
            )
            ys_s = jnp.transpose(
                jnp.concatenate([y0[None], ys_obs[obs_stride - 1 :: obs_stride]], axis=0),
                (1, 0, 2),
            )
            phi_s = jnp.transpose(
                jnp.concatenate(
                    [
                        jnp.zeros((1, batch, obs_feat_dim, g_dim), dtype=theta.dtype),
                        phi_all[obs_stride - 1 :: obs_stride],
                    ],
                    axis=0,
                ),
                (1, 0, 2, 3),
            )
            psi_s = jnp.transpose(
                jnp.concatenate(
                    [
                        jnp.zeros((1, batch, k_dim, g_dim), dtype=theta.dtype),
                        psi_all[obs_stride - 1 :: obs_stride],
                    ],
                    axis=0,
                ),
                (1, 0, 2, 3),
            )
            return ys_s, phi_s, psi_s

        def compute_proj_op(phi_s: jnp.ndarray, psi_s: jnp.ndarray) -> jnp.ndarray:
            """Compute ridge-stabilized L2(P_train) projection onto observed TF span."""
            phi_g = jnp.transpose(phi_s, (3, 0, 1, 2)).reshape(g_dim, -1, obs_feat_dim)
            psi_g = jnp.transpose(psi_s, (3, 0, 1, 2)).reshape(g_dim, -1, k_dim)
            gram_raw = jnp.einsum("gmp,gmq->gpq", phi_g, phi_g)
            diag_scale = 0.25 * jnp.mean(
                jnp.diagonal(gram_raw, axis1=-2, axis2=-1), axis=-1, keepdims=True
            )[..., None] + 1e-2
            gram = gram_raw + diag_scale * jnp.eye(obs_feat_dim, dtype=phi_s.dtype)[None]
            rhs_m = jnp.einsum("gmp,gmk->gpk", phi_g, psi_g)
            return jnp.linalg.solve(gram, rhs_m)

        def forward_with_proj(
            theta: jnp.ndarray,
            stage_u: jnp.ndarray,
            y0: jnp.ndarray,
            proj_op: jnp.ndarray | None = None,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            p = unpack(theta)
            ys_obs, phi_s, psi_s = trace_features(theta, stage_u, y0)
            p_mat = compute_proj_op(phi_s, psi_s) if proj_op is None else proj_op
            # Orthogonalized cofactor basis per target gene: psi_perp = psi_s - phi_s @ p_mat
            psi_perp = psi_s - jnp.einsum("btpg,gpk->btkg", phi_s, p_mat)
            y_ortho_add = jnp.einsum("btkg,kg->btg", psi_perp, p["c_readout"])
            return ys_obs + y_ortho_add, ys_obs, p_mat

        def forward(theta: jnp.ndarray, stage_u: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
            tot, _, _ = forward_with_proj(theta, stage_u, y0, None)
            return tot

        if return_ortho_handles:
            return (
                jax.jit(forward),
                unpack,
                np.asarray(flat0, dtype=np.float32),
                jax.jit(forward_with_proj),
            )
        return jax.jit(forward), unpack, np.asarray(flat0, dtype=np.float32)

    def forward(theta: jnp.ndarray, stage_u: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
        p = unpack(theta)
        decay = 0.1 + jnp.logaddexp(0.0, p["decay"])
        batch = stage_u.shape[1]

        if fam in (
            DynamicalModelFamily.LOCAL_CONST,
            DynamicalModelFamily.MEMGRN,
            DynamicalModelFamily.MEMGRN_ADDITIVE,
            DynamicalModelFamily.MEMGRN_SYM,
        ):
            init_state = (
                jnp.zeros((batch, s_dim, r_dim), dtype=theta.dtype),
                jnp.zeros((batch, s_dim, r_dim), dtype=theta.dtype),
                y0,
            )

            def rhs(state, raw_u):
                z, h, y = state
                act = jnp.tanh(raw_u)
                if fam == DynamicalModelFamily.LOCAL_CONST:
                    lam_up = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                        p["rate"][None, :, :]
                    )
                    lam_dn = lam_up
                else:
                    feat_a = jnp.broadcast_to(act[:, :, None], z.shape)
                    hid = (
                        p["bias"]
                        + feat_a[..., None] * p["in_proj"][0]
                        + z[..., None] * p["in_proj"][1]
                        + h[..., None] * p["in_proj"][2]
                    )
                    th_hid = jnp.tanh(hid)
                    logits_up = p["rate"][None, :, :] + jnp.sum(
                        th_hid * p["out_proj"][None], axis=-1
                    )
                    lam_up = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                        logits_up
                    )
                    if fam == DynamicalModelFamily.MEMGRN_SYM:
                        lam_dn = lam_up
                    else:
                        logits_dn = p["rate_dn"][None, :, :] + jnp.sum(
                            th_hid * p["out_proj_dn"][None], axis=-1
                        )
                        lam_dn = spec.rate_min + (
                            spec.rate_max - spec.rate_min
                        ) * jax.nn.sigmoid(logits_dn)

                dz = lam_up * (act[:, :, None] - z)
                dh = lam_dn * (z - h)
                prod = act @ p["direct"] + h.reshape(batch, s_dim * r_dim) @ p[
                    "delayed"
                ].reshape(s_dim * r_dim, g_dim)
                if fam == DynamicalModelFamily.MEMGRN and s_dim >= 2 and "interact" in p:
                    mean_h = jnp.mean(h, axis=-1)
                    pairs = mean_h[:, idx_i] * mean_h[:, idx_k]
                    prod = prod + pairs @ p["interact"]
                return dz, dh, prod - decay * y

        elif fam in (
            DynamicalModelFamily.COUPLED_CONST,
            DynamicalModelFamily.COUPLED_ADAPTIVE,
        ):
            m_dim = s_dim * r_dim
            init_state = (
                jnp.zeros((batch, m_dim), dtype=theta.dtype),
                jnp.zeros((batch, m_dim), dtype=theta.dtype),
                y0,
            )

            def rhs(state, raw_u):
                z, h, y = state
                act = jnp.tanh(raw_u)
                drive = act @ p["mix_in"]
                logits = p["rate"][None, :]
                if fam == DynamicalModelFamily.COUPLED_ADAPTIVE:
                    zh = jnp.concatenate([z, h], axis=-1)
                    hid = jnp.tanh(act @ p["in_a"] + zh @ p["in_zh"] + p["bias"])
                    logits = logits + hid @ p["out_proj"]
                lam = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(logits)
                dz = lam * (drive - z)
                dh = lam * (z - h)
                prod = act @ p["direct"] + h @ p["delayed"]
                return dz, dh, prod - decay * y

        elif fam == DynamicalModelFamily.LTC_COUPLED:
            m_dim = s_dim * r_dim
            init_state = (jnp.zeros((batch, m_dim), dtype=theta.dtype), y0)
            rest_gate = jax.nn.sigmoid(p["b_gate"])

            def rhs(state, raw_u):
                x, y = state
                act = jnp.tanh(raw_u)
                base_tau_inv = spec.rate_min + (
                    spec.rate_max - spec.rate_min
                ) * jax.nn.sigmoid(p["tau_log"])
                gate = (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                    x @ p["w_x"] + act @ p["w_a"] + p["b_gate"]
                )
                eq_shift = (spec.rate_max - spec.rate_min) * rest_gate * p["e_rev"]
                dx = -(base_tau_inv + gate) * x + gate * p["e_rev"] - eq_shift
                prod = act @ p["direct"] + x @ p["readout"]
                return dx, prod - decay * y

        elif fam == DynamicalModelFamily.LTC_LOCAL:
            init_state = (jnp.zeros((batch, s_dim, r_dim), dtype=theta.dtype), y0)

            def rhs(state, raw_u):
                x, y = state
                act = jnp.tanh(raw_u)
                base_tau_inv = spec.rate_min + (
                    spec.rate_max - spec.rate_min
                ) * jax.nn.sigmoid(p["tau_log"])
                x_mix = jnp.einsum("bsr,srk->bsk", x, p["w_x"])
                gate = (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                    x_mix + act[:, :, None] * p["w_a"][None] + p["b_gate"][None]
                )
                dx = -(base_tau_inv[None] + gate) * x + gate * (
                    act[:, :, None] * p["e_rev"][None]
                )
                prod = act @ p["direct"] + x.reshape(batch, s_dim * r_dim) @ p[
                    "delayed"
                ].reshape(s_dim * r_dim, g_dim)
                return dx, prod - decay * y

        elif fam == DynamicalModelFamily.PERTURB_ODE:
            init_state = (y0,)
            rest_mod = jnp.tanh(p["b_mod"])

            def rhs(state, raw_u):
                (y,) = state
                act = jnp.tanh(raw_u)
                mod = jnp.tanh(y @ p["w_enc"] + p["b_mod"]) - rest_mod
                dy = mod @ p["w_dec"] + act @ p["direct"] - decay * y
                return (dy,)

        else:  # LATENT_ODE
            lat_d = spec.latent_dim
            init_state = (jnp.zeros((batch, lat_d), dtype=theta.dtype), y0)
            rest_hid = jnp.tanh(p["bias"])

            def rhs(state, raw_u):
                z, y = state
                act = jnp.tanh(raw_u)
                lat_decay = 0.1 + jnp.logaddexp(0.0, p["rate"])
                hid = jnp.tanh(z @ p["w_s"] + act @ p["w_a"] + p["bias"]) - rest_hid
                dz = -lat_decay * z + hid @ p["w_dyn"]
                dy = act @ p["direct"] + z @ p["readout"] - decay * y
                return dz, dy

        def step_fn(carry, u_step):
            u0 = u_step[..., 0]
            um = u_step[..., 1]
            u1 = u_step[..., 2]
            k1 = rhs(carry, u0)
            k2 = rhs(jax.tree.map(lambda c, d: c + 0.5 * dt * d, carry, k1), um)
            k3 = rhs(jax.tree.map(lambda c, d: c + 0.5 * dt * d, carry, k2), um)
            k4 = rhs(jax.tree.map(lambda c, d: c + dt * d, carry, k3), u1)
            nxt = jax.tree.map(
                lambda c, d1, d2, d3, d4: c + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4),
                carry,
                k1,
                k2,
                k3,
                k4,
            )
            return nxt, nxt[-1]

        _, ys = jax.lax.scan(step_fn, init_state, stage_u)
        sampled = ys[obs_stride - 1 :: obs_stride]
        with_init = jnp.concatenate([y0[None, :, :], sampled], axis=0)
        return jnp.transpose(with_init, (1, 0, 2))

    return jax.jit(forward), unpack, np.asarray(flat0, dtype=np.float32)


def fit_dynamical_model(
    spec: ModelSpec,
    train_stage_u: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    val_stage_u: np.ndarray,
    val_y: np.ndarray,
    val_mask: np.ndarray,
    dt: float,
    obs_stride: int,
    seeds: tuple[int, ...] = (101, 202),
    reg_grid: tuple[float, ...] = (1e-5, 1e-4, 1e-3),
    total_updates: int = 1200,
    lr_max: float = 0.008,
    l1_interact_scale: float = 5.0,
    gate_shrinkage_grid: tuple[float, ...] = (1.0,),
    stimulus_window_steps: int | None = None,
) -> TrainedModel:
    """Train an open-loop dynamical model with validation-locked regularization & multi-start."""
    built = build_jax_rollout(
        spec,
        dt=dt,
        obs_stride=obs_stride,
        stimulus_window_steps=stimulus_window_steps,
        return_ortho_handles=(spec.family == DynamicalModelFamily.MEMGRN_ORTHO),
    )
    if spec.family == DynamicalModelFamily.MEMGRN_ORTHO:
        forward, unpack, flat_template, forward_with_proj = built
    else:
        forward, unpack, flat_template = built
        forward_with_proj = None

    tr_u = jnp.asarray(train_stage_u, dtype=jnp.float32)
    tr_y = jnp.asarray(train_y, dtype=jnp.float32)
    tr_y0 = tr_y[:, 0, :]
    tr_w = jnp.asarray(train_mask, dtype=jnp.float32)
    tr_denom = jnp.maximum(1.0, jnp.sum(tr_w))

    va_u = jnp.asarray(val_stage_u, dtype=jnp.float32)
    va_y = jnp.asarray(val_y, dtype=jnp.float32)
    va_y0 = va_y[:, 0, :]
    va_w = jnp.asarray(val_mask, dtype=jnp.float32)
    va_denom = jnp.maximum(1.0, jnp.sum(va_w))

    def train_mse(theta: jnp.ndarray) -> jnp.ndarray:
        pred = forward(theta, tr_u, tr_y0)
        return jnp.sum(((pred - tr_y) ** 2) * tr_w) / tr_denom

    def val_mse(theta: jnp.ndarray) -> jnp.ndarray:
        if forward_with_proj is not None:
            _, _, tr_proj = forward_with_proj(theta, tr_u, tr_y0, None)
            pred_va, _, _ = forward_with_proj(theta, va_u, va_y0, tr_proj)
        else:
            pred_va = forward(theta, va_u, va_y0)
        return jnp.sum(((pred_va - va_y) ** 2) * va_w) / va_denom

    def penalty(theta: jnp.ndarray, gate_scale: jnp.ndarray) -> jnp.ndarray:
        p = unpack(theta)
        gate_keys = {"out_proj", "out_proj_dn", "in_proj"}
        skip_keys = {"decay", "rate", "rate_dn", "tau_log", "c_rate"}
        l2 = jnp.asarray(0.0, dtype=theta.dtype)
        for k, v in p.items():
            if k in skip_keys:
                continue
            mult = gate_scale if k in gate_keys else 1.0
            l2 = l2 + mult * jnp.sum(v**2)
        if "interact" in p:
            l2 = l2 + l1_interact_scale * jnp.sum(jnp.abs(p["interact"]))
        if "rate_dn" in p:
            l2 = l2 + 0.25 * gate_scale * jnp.sum((p["rate_dn"] - p["rate"]) ** 2)
        return l2

    @jax.jit
    def run_single(
        theta0: jnp.ndarray, reg_val: jnp.ndarray, gate_scale: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def obj(th: jnp.ndarray) -> jnp.ndarray:
            return train_mse(th) + reg_val * penalty(th, gate_scale)

        val_every = max(1, total_updates // 12)
        n_chunks = total_updates // val_every

        def chunk_step(carry, chunk_idx):
            th, m, v, best_th, best_val = carry

            def inner_body(k, inner_carry):
                th_i, m_i, v_i = inner_carry
                step = chunk_idx * val_every + k + 1
                g = jax.grad(obj)(th_i)
                g = g * jnp.minimum(1.0, 4.0 / (jnp.linalg.norm(g) + 1e-9))
                m_i = 0.9 * m_i + 0.1 * g
                v_i = 0.999 * v_i + 0.001 * (g * g)
                lr = 0.0006 + (lr_max - 0.0006) * 0.5 * (
                    1.0 + jnp.cos(jnp.pi * step / total_updates)
                )
                m_hat = m_i / (1.0 - 0.9**step)
                v_hat = v_i / (1.0 - 0.999**step)
                th_i = th_i - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
                return th_i, m_i, v_i

            th, m, v = jax.lax.fori_loop(0, val_every, inner_body, (th, m, v))
            v_loss = val_mse(th)
            better = v_loss < best_val
            best_th = jnp.where(better, th, best_th)
            best_val = jnp.where(better, v_loss, best_val)
            return (th, m, v, best_th, best_val), v_loss

        init_val = val_mse(theta0)
        init_carry = (
            theta0,
            jnp.zeros_like(theta0),
            jnp.zeros_like(theta0),
            theta0,
            init_val,
        )
        (_, _, _, best_th, best_val), _ = jax.lax.scan(
            chunk_step, init_carry, jnp.arange(n_chunks)
        )
        return best_th, best_val

    best_overall_val = float("inf")
    best_theta_np = flat_template.copy()
    best_reg = reg_grid[0]
    best_seed = seeds[0]

    g_grid = (
        gate_shrinkage_grid
        if spec.family
        in (
            DynamicalModelFamily.MEMGRN,
            DynamicalModelFamily.MEMGRN_ADDITIVE,
            DynamicalModelFamily.MEMGRN_SYM,
            DynamicalModelFamily.MEMGRN_ORTHO,
        )
        else (1.0,)
    )

    for seed in seeds:
        init_dict = _init_tree(spec, seed=seed)
        th0 = ravel_pytree({k: jnp.asarray(v) for k, v in init_dict.items()})[0]
        for reg in reg_grid:
            for g_sc in g_grid:
                th_opt, val_l = run_single(
                    th0,
                    jnp.asarray(reg, dtype=jnp.float32),
                    jnp.asarray(g_sc, dtype=jnp.float32),
                )
                v_float = float(val_l)
                th_np = np.asarray(th_opt, dtype=np.float32)
                if np.all(np.isfinite(th_np)) and v_float < best_overall_val:
                    best_overall_val = v_float
                    best_theta_np = th_np
                    best_reg = reg
                    best_seed = seed

    th_j = jnp.asarray(best_theta_np, dtype=jnp.float32)

    if forward_with_proj is not None:
        _, _, frozen_tr_proj = forward_with_proj(th_j, tr_u, tr_y0, None)

        def predict_fn(stage_u_np: np.ndarray, y0_np: np.ndarray | None = None) -> np.ndarray:
            batch = stage_u_np.shape[1]
            y0_arr = (
                np.zeros((batch, spec.n_targets), dtype=np.float32)
                if y0_np is None
                else np.asarray(y0_np, dtype=np.float32)
            )
            tot, _, _ = forward_with_proj(
                th_j, jnp.asarray(stage_u_np, dtype=jnp.float32), jnp.asarray(y0_arr), frozen_tr_proj
            )
            return np.asarray(tot, dtype=np.float64)

        def predict_obs_fn(stage_u_np: np.ndarray, y0_np: np.ndarray | None = None) -> np.ndarray:
            batch = stage_u_np.shape[1]
            y0_arr = (
                np.zeros((batch, spec.n_targets), dtype=np.float32)
                if y0_np is None
                else np.asarray(y0_np, dtype=np.float32)
            )
            _, obs_only, _ = forward_with_proj(
                th_j, jnp.asarray(stage_u_np, dtype=jnp.float32), jnp.asarray(y0_arr), frozen_tr_proj
            )
            return np.asarray(obs_only, dtype=np.float64)

        return TrainedModel(
            spec=spec,
            theta=best_theta_np,
            n_params=int(best_theta_np.size),
            best_val_loss=best_overall_val,
            chosen_reg=best_reg,
            chosen_seed=best_seed,
            predict_fn=predict_fn,
            predict_observed_only_fn=predict_obs_fn,
        )

    def predict_fn(stage_u_np: np.ndarray, y0_np: np.ndarray | None = None) -> np.ndarray:
        batch = stage_u_np.shape[1]
        y0_arr = (
            np.zeros((batch, spec.n_targets), dtype=np.float32)
            if y0_np is None
            else np.asarray(y0_np, dtype=np.float32)
        )
        out = forward(th_j, jnp.asarray(stage_u_np, dtype=jnp.float32), jnp.asarray(y0_arr))
        return np.asarray(out, dtype=np.float64)

    return TrainedModel(
        spec=spec,
        theta=best_theta_np,
        n_params=int(best_theta_np.size),
        best_val_loss=best_overall_val,
        chosen_reg=best_reg,
        chosen_seed=best_seed,
        predict_fn=predict_fn,
    )


def counterfactual_source_attribution(
    predict_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    stage_u: np.ndarray,
    y0: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate common model-agnostic input-intervention effects Delta_{i -> j}(t)."""
    base_pred = predict_fn(stage_u, y0)
    n_sources = stage_u.shape[2]
    effects = []
    for src_idx in range(n_sources):
        ablated = stage_u.copy()
        ablated[:, :, src_idx, :] = 0.0
        effects.append(base_pred - predict_fn(ablated, y0))
    return np.stack(effects, axis=0)


def fit_two_stage_memgrn_ortho(
    base_memgrn: TrainedModel,
    train_stage_u: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    val_stage_u: np.ndarray,
    val_y: np.ndarray,
    val_mask: np.ndarray,
    dt: float,
    obs_stride: int,
    latent_dim: int = 12,
    seeds: tuple[int, ...] = (416, 819),
    reg_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 5e-2),
    total_updates: int = 1200,
    lr_max: float = 0.007,
) -> TrainedModel:
    """Fit Proposition 1(iii) two-stage orthogonalized cofactor residual atop frozen MemGRN.

    Stage 1 holds the fitted source-isolated MemGRN observed-TF predictor y_obs(t; g) strictly
    fixed (guaranteeing exact zero observed-TF attribution drift). Stage 2 fits a low-rank
    latent cofactor ODE psi(t) to the orthogonalized residual r = y - y_obs, projecting out
    collinear components along y_obs on the source training split.
    """
    tr_obs = base_memgrn.predict_fn(train_stage_u, None)
    va_obs = base_memgrn.predict_fn(val_stage_u, None)

    tr_resid = train_y - tr_obs
    va_resid = val_y - va_obs
    tr_resid[:, 0, :] = 0.0
    va_resid[:, 0, :] = 0.0

    resid_spec = ModelSpec(
        family=DynamicalModelFamily.LATENT_ODE,
        n_sources=base_memgrn.spec.n_sources,
        n_targets=base_memgrn.spec.n_targets,
        latent_dim=latent_dim,
    )
    resid_fit = fit_dynamical_model(
        spec=resid_spec,
        train_stage_u=train_stage_u,
        train_y=tr_resid,
        train_mask=train_mask,
        val_stage_u=val_stage_u,
        val_y=va_resid,
        val_mask=val_mask,
        dt=dt,
        obs_stride=obs_stride,
        seeds=seeds,
        reg_grid=reg_grid,
        total_updates=total_updates,
        lr_max=lr_max,
    )

    # Compute per-gene scalar orthogonal projection alpha_g = <y_obs_g, r_hat_g>_train / (||y_obs_g||^2 + eps)
    r_tr_hat = resid_fit.predict_fn(train_stage_u, None)
    w_tr = train_mask
    num = np.sum(w_tr * tr_obs * r_tr_hat, axis=(0, 1))
    den = np.sum(w_tr * (tr_obs**2), axis=(0, 1)) + 1.0
    alpha_proj = num / den  # (n_targets,)

    # Validation guard: shrink residual weight lambda_blend in [0, 1] on source validation split
    r_va_perp = resid_fit.predict_fn(val_stage_u, None) - alpha_proj[None, None, :] * va_obs
    blend_num = float(np.sum(val_mask * va_resid * r_va_perp))
    blend_den = float(np.sum(val_mask * (r_va_perp**2))) + 1e-8
    blend = float(np.clip(blend_num / blend_den, 0.0, 1.0))

    def predict_total(stage_u_np: np.ndarray, y0_np: np.ndarray | None = None) -> np.ndarray:
        y_o = base_memgrn.predict_fn(stage_u_np, y0_np)
        r_raw = resid_fit.predict_fn(stage_u_np, None)
        r_perp = r_raw - alpha_proj[None, None, :] * y_o
        return y_o + blend * r_perp

    ortho_spec = ModelSpec(
        family=DynamicalModelFamily.MEMGRN_ORTHO,
        n_sources=base_memgrn.spec.n_sources,
        n_targets=base_memgrn.spec.n_targets,
        n_channels=base_memgrn.spec.n_channels,
        hidden_width=base_memgrn.spec.hidden_width,
        ortho_dim=latent_dim,
    )
    va_tot = predict_total(val_stage_u, None)
    va_mse = float(np.sum(((va_tot - val_y) ** 2) * val_mask) / np.maximum(1.0, np.sum(val_mask)))

    return TrainedModel(
        spec=ortho_spec,
        theta=np.concatenate([base_memgrn.theta, resid_fit.theta]),
        n_params=base_memgrn.n_params + resid_fit.n_params,
        best_val_loss=va_mse,
        chosen_reg=resid_fit.chosen_reg,
        chosen_seed=resid_fit.chosen_seed,
        predict_fn=predict_total,
        predict_observed_only_fn=base_memgrn.predict_fn,
    )


# =============================================================================
# Autonomous Closed-Loop Gene Regulatory Network Suite (No Future TF Traces)
# =============================================================================


def _init_closed_loop_tree(spec: ClosedLoopSpec, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_g, r_dim, h_dim = spec.n_genes, spec.n_channels, spec.hidden_width
    s_reg = n_g if spec.regulator_indices is None else len(spec.regulator_indices)
    base_rates = np.resize(np.array([0.35, 1.0, 2.8], dtype=np.float32), r_dim)
    base_logits = np.log((base_rates - spec.rate_min) / (spec.rate_max - base_rates))
    fam = spec.family

    if fam in (ClosedLoopModelFamily.LOCAL_CONST, ClosedLoopModelFamily.MEMGRN):
        tree: dict[str, np.ndarray] = {
            "direct": rng.normal(0.0, 0.015, size=(s_reg, n_g)).astype(np.float32),
            "delayed": rng.normal(0.0, 0.015, size=(s_reg, r_dim, n_g)).astype(np.float32),
            "decay": np.zeros((n_g,), dtype=np.float32),
            "in_gain": np.zeros((s_reg,), dtype=np.float32),
            "rate": np.broadcast_to(base_logits, (s_reg, r_dim)).copy(),
        }
        if fam == ClosedLoopModelFamily.MEMGRN:
            tree["rate_dn"] = np.broadcast_to(base_logits, (s_reg, r_dim)).copy()
            tree["in_proj"] = rng.normal(0.0, 0.6, size=(3, h_dim)).astype(np.float32)
            tree["bias"] = rng.normal(0.0, 0.2, size=(h_dim,)).astype(np.float32)
            tree["out_proj"] = np.zeros((s_reg, r_dim, h_dim), dtype=np.float32)
            tree["out_proj_dn"] = np.zeros((s_reg, r_dim, h_dim), dtype=np.float32)
        return tree

    if fam == ClosedLoopModelFamily.COUPLED_ADAPTIVE:
        m_dim = s_reg * r_dim
        eye_mix = np.repeat(np.eye(s_reg, dtype=np.float32), r_dim, axis=1)
        return {
            "mix_in": eye_mix + rng.normal(0.0, 0.05, size=(s_reg, m_dim)).astype(np.float32),
            "direct": rng.normal(0.0, 0.015, size=(s_reg, n_g)).astype(np.float32),
            "delayed": rng.normal(0.0, 0.015, size=(m_dim, n_g)).astype(np.float32),
            "decay": np.zeros((n_g,), dtype=np.float32),
            "in_gain": np.zeros((s_reg,), dtype=np.float32),
            "rate": np.tile(base_logits, s_reg).astype(np.float32),
            "in_a": rng.normal(0.0, 0.4 / np.sqrt(s_reg), size=(s_reg, h_dim)).astype(np.float32),
            "in_zh": rng.normal(0.0, 0.4 / np.sqrt(m_dim), size=(2 * m_dim, h_dim)).astype(
                np.float32
            ),
            "bias": rng.normal(0.0, 0.2, size=(h_dim,)).astype(np.float32),
            "out_proj": np.zeros((h_dim, m_dim), dtype=np.float32),
        }

    if fam == ClosedLoopModelFamily.LTC_COUPLED:
        m_dim = spec.latent_dim
        return {
            "tau_log": np.zeros((m_dim,), dtype=np.float32),
            "w_x": rng.normal(0.0, 0.3 / np.sqrt(m_dim), size=(m_dim, m_dim)).astype(np.float32),
            "w_a": rng.normal(0.0, 0.4 / np.sqrt(s_reg), size=(s_reg, m_dim)).astype(np.float32),
            "b_gate": rng.normal(0.0, 0.15, size=(m_dim,)).astype(np.float32),
            "e_rev": rng.normal(0.0, 0.5, size=(m_dim,)).astype(np.float32),
            "direct": rng.normal(0.0, 0.015, size=(s_reg, n_g)).astype(np.float32),
            "readout": rng.normal(0.0, 0.02, size=(m_dim, n_g)).astype(np.float32),
            "decay": np.zeros((n_g,), dtype=np.float32),
            "in_gain": np.zeros((s_reg,), dtype=np.float32),
        }

    if fam == ClosedLoopModelFamily.LTC_LOCAL:
        return {
            "tau_log": np.broadcast_to(base_logits, (s_reg, r_dim)).copy(),
            "w_x": rng.normal(0.0, 0.3, size=(s_reg, r_dim, r_dim)).astype(np.float32),
            "w_a": rng.normal(0.0, 0.6, size=(s_reg, r_dim)).astype(np.float32),
            "b_gate": rng.normal(0.0, 0.15, size=(s_reg, r_dim)).astype(np.float32),
            "e_rev": np.ones((s_reg, r_dim), dtype=np.float32),
            "direct": rng.normal(0.0, 0.015, size=(s_reg, n_g)).astype(np.float32),
            "delayed": rng.normal(0.0, 0.015, size=(s_reg, r_dim, n_g)).astype(np.float32),
            "decay": np.zeros((n_g,), dtype=np.float32),
            "in_gain": np.zeros((s_reg,), dtype=np.float32),
        }

    if fam == ClosedLoopModelFamily.PERTURB_ODE:
        mod_d = spec.module_dim
        return {
            "w_enc": rng.normal(0.0, 0.15 / np.sqrt(s_reg), size=(s_reg, mod_d)).astype(np.float32),
            "b_mod": rng.normal(0.0, 0.1, size=(mod_d,)).astype(np.float32),
            "w_dec": np.zeros((mod_d, n_g), dtype=np.float32),
            "direct": rng.normal(0.0, 0.015, size=(s_reg, n_g)).astype(np.float32),
            "decay": np.zeros((n_g,), dtype=np.float32),
            "in_gain": np.zeros((s_reg,), dtype=np.float32),
        }

    lat_d = spec.latent_dim
    return {
        "rate": np.zeros((lat_d,), dtype=np.float32),
        "w_s": rng.normal(0.0, 0.25 / np.sqrt(lat_d), size=(lat_d, lat_d)).astype(np.float32),
        "w_a": rng.normal(0.0, 0.4 / np.sqrt(s_reg), size=(s_reg, lat_d)).astype(np.float32),
        "bias": rng.normal(0.0, 0.15, size=(lat_d,)).astype(np.float32),
        "w_dyn": rng.normal(0.0, 0.15 / np.sqrt(lat_d), size=(lat_d, lat_d)).astype(np.float32),
        "readout": rng.normal(0.0, 0.03, size=(lat_d, n_g)).astype(np.float32),
        "decay": np.zeros((n_g,), dtype=np.float32),
        "in_gain": np.zeros((s_reg,), dtype=np.float32),
    }


def build_closed_loop_jax_rollout(
    spec: ClosedLoopSpec,
    basal_levels: np.ndarray,
    dt: float,
    n_steps: int,
    obs_step_indices: tuple[int, ...],
    clamp_steps: int = 0,
):
    """Build JIT-compiled autonomous closed-loop CRISPR knockout network rollout.

    If clamp_steps > 0 and y0 is provided, steps 0..clamp_steps-1 linearly warm-start
    intracellular memory along the ramp 0 -> y0 before releasing y(t) at step clamp_steps
    from y0 for fully autonomous closed-loop integration over the remaining horizon.
    """
    template = _init_closed_loop_tree(spec, seed=0)
    template_j = {k: jnp.asarray(v) for k, v in template.items()}
    flat0, unpack = ravel_pytree(template_j)
    n_g, r_dim = spec.n_genes, spec.n_channels
    reg_idx_np = (
        np.arange(n_g, dtype=np.int32)
        if spec.regulator_indices is None
        else np.asarray(spec.regulator_indices, dtype=np.int32)
    )
    s_reg = len(reg_idx_np)
    reg_idx_j = jnp.asarray(reg_idx_np, dtype=jnp.int32)

    hollow_np = np.ones((s_reg, n_g), dtype=np.float32)
    for r_pos, g_pos in enumerate(reg_idx_np):
        hollow_np[r_pos, g_pos] = 0.0
    off_diag = jnp.asarray(hollow_np)

    basal_reg_j = jnp.asarray(basal_levels[reg_idx_np], dtype=jnp.float32)
    obs_idx_j = jnp.asarray(obs_step_indices, dtype=jnp.int32)
    fam = spec.family

    def forward(theta: jnp.ndarray, ko_mask: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
        p = unpack(theta)
        decay = 0.15 + jnp.logaddexp(0.0, p["decay"])
        gain = 0.35 + jax.nn.softplus(p["in_gain"])
        batch = ko_mask.shape[0]
        ko_reg = ko_mask[:, reg_idx_j]

        def _gene_activity(y_state: jnp.ndarray) -> jnp.ndarray:
            y_reg = y_state[:, reg_idx_j]
            raw_g = (1.0 - ko_reg) * (gain[None] * y_reg) - ko_reg * (gain[None] * basal_reg_j[None])
            return jnp.tanh(raw_g)

        y_start = jnp.zeros_like(y0) if clamp_steps > 0 else y0

        if fam in (ClosedLoopModelFamily.LOCAL_CONST, ClosedLoopModelFamily.MEMGRN):
            d_hol = p["direct"] * off_diag
            b_hol = p["delayed"] * off_diag[:, None, :]
            init_state = (
                jnp.zeros((batch, s_reg, r_dim), dtype=theta.dtype),
                jnp.zeros((batch, s_reg, r_dim), dtype=theta.dtype),
                y_start,
            )

            def rhs(state):
                z, h, y = state
                act = _gene_activity(y)
                if fam == ClosedLoopModelFamily.LOCAL_CONST:
                    lam_up = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                        p["rate"][None]
                    )
                    lam_dn = lam_up
                else:
                    feat_a = jnp.broadcast_to(act[:, :, None], z.shape)
                    hid = (
                        p["bias"]
                        + feat_a[..., None] * p["in_proj"][0]
                        + z[..., None] * p["in_proj"][1]
                        + h[..., None] * p["in_proj"][2]
                    )
                    th_h = jnp.tanh(hid)
                    lam_up = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                        p["rate"][None] + jnp.sum(th_h * p["out_proj"][None], axis=-1)
                    )
                    lam_dn = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                        p["rate_dn"][None] + jnp.sum(th_h * p["out_proj_dn"][None], axis=-1)
                    )
                dz = lam_up * (act[:, :, None] - z)
                dh = lam_dn * (z - h)
                prod = act @ d_hol + h.reshape(batch, s_reg * r_dim) @ b_hol.reshape(
                    s_reg * r_dim, n_g
                )
                return dz, dh, prod - decay[None] * y

        elif fam == ClosedLoopModelFamily.COUPLED_ADAPTIVE:
            m_dim = s_reg * r_dim
            d_hol = p["direct"] * off_diag
            init_state = (
                jnp.zeros((batch, m_dim), dtype=theta.dtype),
                jnp.zeros((batch, m_dim), dtype=theta.dtype),
                y_start,
            )

            def rhs(state):
                z, h, y = state
                act = _gene_activity(y)
                drive = act @ p["mix_in"]
                zh = jnp.concatenate([z, h], axis=-1)
                hid = jnp.tanh(act @ p["in_a"] + zh @ p["in_zh"] + p["bias"])
                lam = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                    p["rate"][None] + hid @ p["out_proj"]
                )
                dz = lam * (drive - z)
                dh = lam * (z - h)
                return dz, dh, act @ d_hol + h @ p["delayed"] - decay[None] * y

        elif fam == ClosedLoopModelFamily.LTC_COUPLED:
            m_dim = spec.latent_dim
            d_hol = p["direct"] * off_diag
            init_state = (jnp.zeros((batch, m_dim), dtype=theta.dtype), y_start)
            rest_gate = jax.nn.sigmoid(p["b_gate"])

            def rhs(state):
                x, y = state
                act = _gene_activity(y)
                base_tau = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                    p["tau_log"]
                )
                gate = (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                    x @ p["w_x"] + act @ p["w_a"] + p["b_gate"]
                )
                eq_shift = (spec.rate_max - spec.rate_min) * rest_gate * p["e_rev"]
                dx = -(base_tau[None] + gate) * x + gate * p["e_rev"][None] - eq_shift[None]
                return dx, act @ d_hol + x @ p["readout"] - decay[None] * y

        elif fam == ClosedLoopModelFamily.LTC_LOCAL:
            d_hol = p["direct"] * off_diag
            b_hol = p["delayed"] * off_diag[:, None, :]
            init_state = (jnp.zeros((batch, s_reg, r_dim), dtype=theta.dtype), y_start)

            def rhs(state):
                x, y = state
                act = _gene_activity(y)
                base_tau = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                    p["tau_log"]
                )
                x_mix = jnp.einsum("bnr,nrk->bnk", x, p["w_x"])
                gate = (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(
                    x_mix + act[:, :, None] * p["w_a"][None] + p["b_gate"][None]
                )
                dx = -(base_tau[None] + gate) * x + gate * (
                    act[:, :, None] * p["e_rev"][None]
                )
                prod = act @ d_hol + x.reshape(batch, s_reg * r_dim) @ b_hol.reshape(
                    s_reg * r_dim, n_g
                )
                return dx, prod - decay[None] * y

        elif fam == ClosedLoopModelFamily.PERTURB_ODE:
            d_hol = p["direct"] * off_diag
            init_state = (y_start,)
            rest_mod = jnp.tanh(p["b_mod"])

            def rhs(state):
                (y,) = state
                act = _gene_activity(y)
                mod = jnp.tanh(act @ p["w_enc"] + p["b_mod"]) - rest_mod
                dy = mod @ p["w_dec"] + act @ d_hol - decay[None] * y
                return (dy,)

        else:  # LATENT_ODE
            lat_d = spec.latent_dim
            init_state = (jnp.zeros((batch, lat_d), dtype=theta.dtype), y_start)
            rest_hid = jnp.tanh(p["bias"])

            def rhs(state):
                z, y = state
                act = _gene_activity(y)
                lat_dec = 0.15 + jnp.logaddexp(0.0, p["rate"])
                hid = jnp.tanh(z @ p["w_s"] + act @ p["w_a"] + p["bias"]) - rest_hid
                dz = -lat_dec[None] * z + hid @ p["w_dyn"]
                return dz, z @ p["readout"] - decay[None] * y

        def step_fn(carry, step_idx):
            k1 = rhs(carry)
            k2 = rhs(jax.tree.map(lambda c, d: c + 0.5 * dt * d, carry, k1))
            k3 = rhs(jax.tree.map(lambda c, d: c + 0.5 * dt * d, carry, k2))
            k4 = rhs(jax.tree.map(lambda c, d: c + dt * d, carry, k3))
            nxt = jax.tree.map(
                lambda c, d1, d2, d3, d4: c + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4),
                carry,
                k1,
                k2,
                k3,
                k4,
            )
            if clamp_steps > 0:
                frac = (step_idx + 1).astype(theta.dtype) / float(clamp_steps)
                y_ramp = frac * y0
                y_next = jnp.where(step_idx < clamp_steps, y_ramp, nxt[-1])
                nxt = (*nxt[:-1], y_next)
            return nxt, nxt[-1]

        _, ys = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
        with_init = jnp.concatenate([y_start[None], ys], axis=0)
        sampled = with_init[obs_idx_j]
        return jnp.transpose(sampled, (1, 0, 2))

    return jax.jit(forward), unpack, np.asarray(flat0, dtype=np.float32)


def fit_closed_loop_network(
    spec: ClosedLoopSpec,
    basal_levels: np.ndarray,
    train_ko: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    val_ko: np.ndarray,
    val_y: np.ndarray,
    val_weights: np.ndarray,
    dt: float,
    n_steps: int,
    obs_step_indices: tuple[int, ...],
    train_y0: np.ndarray | None = None,
    val_y0: np.ndarray | None = None,
    clamp_steps: int = 0,
    seeds: tuple[int, ...] = (101, 202),
    reg_grid: tuple[float, ...] = (1e-3, 5e-3, 2e-2),
    l1_graph_ratio: float = 0.25,
    total_updates: int = 900,
    lr_max: float = 0.006,
) -> TrainedClosedLoopModel:
    """Fit a closed-loop N-gene network with validation regularization selection."""
    forward, unpack, flat_template = build_closed_loop_jax_rollout(
        spec,
        basal_levels=basal_levels,
        dt=dt,
        n_steps=n_steps,
        obs_step_indices=obs_step_indices,
        clamp_steps=clamp_steps,
    )

    tr_ko = jnp.asarray(train_ko, dtype=jnp.float32)
    tr_y = jnp.asarray(train_y, dtype=jnp.float32)
    tr_y0 = (
        jnp.zeros((train_ko.shape[0], spec.n_genes), dtype=jnp.float32)
        if train_y0 is None
        else jnp.asarray(train_y0, dtype=jnp.float32)
    )
    tr_w = jnp.asarray(train_weights, dtype=jnp.float32)
    tr_den = jnp.maximum(1.0, jnp.sum(tr_w))

    va_ko = jnp.asarray(val_ko, dtype=jnp.float32)
    va_y = jnp.asarray(val_y, dtype=jnp.float32)
    va_y0 = (
        jnp.zeros((val_ko.shape[0], spec.n_genes), dtype=jnp.float32)
        if val_y0 is None
        else jnp.asarray(val_y0, dtype=jnp.float32)
    )
    va_w = jnp.asarray(val_weights, dtype=jnp.float32)
    va_den = jnp.maximum(1.0, jnp.sum(va_w))

    def data_loss(theta, ko_b, y_b, y0_b, w_b, den_b):
        pred = forward(theta, ko_b, y0_b)
        return jnp.sum(((pred - y_b) ** 2) * w_b) / den_b

    def reg_penalty(theta):
        p = unpack(theta)
        skip = {"decay", "rate", "rate_dn", "tau_log", "in_gain"}
        l2 = sum(jnp.sum(v**2) for k, v in p.items() if k not in skip)
        l1 = jnp.asarray(0.0, dtype=theta.dtype)
        if "direct" in p:
            l1 = l1 + jnp.sum(jnp.abs(p["direct"]))
        if "delayed" in p:
            l1 = l1 + jnp.sum(jnp.abs(p["delayed"])) / float(spec.n_channels)
        if "out_proj" in p:
            l2 = l2 + 3.0 * jnp.sum(p["out_proj"] ** 2)
        if "out_proj_dn" in p:
            l2 = l2 + 3.0 * jnp.sum(p["out_proj_dn"] ** 2)
        return (1.0 - l1_graph_ratio) * l2 + l1_graph_ratio * 0.2 * l1

    @jax.jit
    def run_single(theta0: jnp.ndarray, reg_val: jnp.ndarray):
        def obj(th):
            return data_loss(th, tr_ko, tr_y, tr_y0, tr_w, tr_den) + reg_val * reg_penalty(th)

        val_every = max(1, total_updates // 12)
        n_chunks = total_updates // val_every

        def chunk_step(carry, chunk_idx):
            th, m, v, best_th, best_val = carry

            def inner(k, ic):
                th_i, m_i, v_i = ic
                step = chunk_idx * val_every + k + 1
                g = jax.grad(obj)(th_i)
                g = g * jnp.minimum(1.0, 4.0 / (jnp.linalg.norm(g) + 1e-9))
                m_i = 0.9 * m_i + 0.1 * g
                v_i = 0.999 * v_i + 0.001 * (g * g)
                lr = 0.0005 + (lr_max - 0.0005) * 0.5 * (
                    1.0 + jnp.cos(jnp.pi * step / total_updates)
                )
                m_hat = m_i / (1.0 - 0.9**step)
                v_hat = v_i / (1.0 - 0.999**step)
                th_i = th_i - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
                return th_i, m_i, v_i

            th, m, v = jax.lax.fori_loop(0, val_every, inner, (th, m, v))
            v_l = data_loss(th, va_ko, va_y, va_y0, va_w, va_den)
            better = v_l < best_val
            return (
                th,
                m,
                v,
                jnp.where(better, th, best_th),
                jnp.where(better, v_l, best_val),
            ), v_l

        init_v = data_loss(theta0, va_ko, va_y, va_y0, va_w, va_den)
        (_, _, _, best_th, best_val), _ = jax.lax.scan(
            chunk_step,
            (theta0, jnp.zeros_like(theta0), jnp.zeros_like(theta0), theta0, init_v),
            jnp.arange(n_chunks),
        )
        return best_th, best_val

    best_overall = float("inf")
    best_th_np = flat_template.copy()
    best_reg = reg_grid[0]
    best_seed = seeds[0]

    for seed in seeds:
        th0 = ravel_pytree({k: jnp.asarray(v) for k, v in _init_closed_loop_tree(spec, seed).items()})[0]
        for reg in reg_grid:
            th_opt, v_l = run_single(th0, jnp.asarray(reg, dtype=jnp.float32))
            vf = float(v_l)
            th_arr = np.asarray(th_opt, dtype=np.float32)
            if np.all(np.isfinite(th_arr)) and vf < best_overall:
                best_overall = vf
                best_th_np = th_arr
                best_reg = reg
                best_seed = seed

    th_j = jnp.asarray(best_th_np, dtype=jnp.float32)

    def predict_ko_fn(ko_np: np.ndarray, y0_np: np.ndarray | None = None) -> np.ndarray:
        ko_arr = np.asarray(ko_np, dtype=np.float32)
        unbatched = ko_arr.ndim == 1
        if unbatched:
            ko_arr = ko_arr[None, :]
        y0_arr = (
            np.zeros((ko_arr.shape[0], spec.n_genes), dtype=np.float32)
            if y0_np is None
            else np.asarray(y0_np, dtype=np.float32)
        )
        pred = np.asarray(forward(th_j, jnp.asarray(ko_arr), jnp.asarray(y0_arr)), dtype=np.float64)
        return pred[0] if unbatched else pred

    # Common early-time counterfactual single-TF knockout sensitivity across all architectures
    reg_idx_np = (
        np.arange(spec.n_genes, dtype=np.int64)
        if spec.regulator_indices is None
        else np.asarray(spec.regulator_indices, dtype=np.int64)
    )
    early_fwd, _, _ = build_closed_loop_jax_rollout(
        spec,
        basal_levels=basal_levels,
        dt=dt,
        n_steps=max(4, n_steps // 4),
        obs_step_indices=(1, 2, 4),
        clamp_steps=0,
    )
    ko_probe = np.zeros((len(reg_idx_np), spec.n_genes), dtype=np.float32)
    for r_pos, g_pos in enumerate(reg_idx_np):
        ko_probe[r_pos, g_pos] = 1.0
    zero_y0_p = jnp.zeros((len(reg_idx_np), spec.n_genes), dtype=jnp.float32)
    wt_early = np.asarray(
        early_fwd(th_j, jnp.zeros((1, spec.n_genes), dtype=jnp.float32), jnp.zeros((1, spec.n_genes), dtype=jnp.float32)),
        dtype=np.float64,
    )[0]
    ko_early = np.asarray(early_fwd(th_j, jnp.asarray(ko_probe), zero_y0_p), dtype=np.float64)
    sens = np.sqrt(np.mean((ko_early - wt_early[None]) ** 2, axis=1))
    graph_mat = np.zeros((spec.n_genes, spec.n_genes), dtype=np.float64)
    for r_pos, g_pos in enumerate(reg_idx_np):
        sens[r_pos, g_pos] = 0.0
        graph_mat[g_pos] = sens[r_pos]

    return TrainedClosedLoopModel(
        spec=spec,
        theta=best_th_np,
        n_params=int(best_th_np.size),
        best_val_loss=best_overall,
        chosen_reg=best_reg,
        chosen_seed=best_seed,
        predict_ko_fn=predict_ko_fn,
        interaction_matrix=graph_mat,
    )
