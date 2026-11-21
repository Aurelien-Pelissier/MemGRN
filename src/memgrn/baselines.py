"""Unified JAX dynamical baselines for 2x2 factorial memory-topology and external comparisons.

Implements matched continuous-time RK4 models sharing data splits, clipped cosine-Adam
optimization, validation selection, and counterfactual input-intervention attribution:
  1. local_const:      Source-isolated constant-rate filter cascade + explicit readout
  2. memgrn:           Source-isolated adaptive-rate filter cascade + explicit readout
  3. memgrn_int:       Source-isolated adaptive-rate filter cascade + sparse pairwise readout
  4. coupled_const:    Source-coupled constant-rate filter cascade + explicit readout
  5. coupled_adaptive: Source-coupled adaptive-rate filter cascade + explicit readout
  6. ltc_coupled:      Liquid Time-Constant Network (Hasani et al., 2021), coupled state
  7. ltc_local:        Source-isolated Liquid Time-Constant Network ablation
  8. perturb_ode:      PerturbODE (Lin et al., 2025) module-structured state-only ODE
  9. latent_ode:       Unconstrained Latent Neural ODE (Rubanova et al., 2019)
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


class DynamicalModelFamily(StrEnum):
    """Supported continuous-time dynamical architectures."""

    LOCAL_CONST = "local_const"
    MEMGRN = "memgrn"
    MEMGRN_INT = "memgrn_int"
    COUPLED_CONST = "coupled_const"
    COUPLED_ADAPTIVE = "coupled_adaptive"
    LTC_COUPLED = "ltc_coupled"
    LTC_LOCAL = "ltc_local"
    PERTURB_ODE = "perturb_ode"
    LATENT_ODE = "latent_ode"


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    """Hyperparameter and dimension specification for a dynamical model."""

    family: DynamicalModelFamily
    n_sources: int
    n_targets: int
    n_channels: int = 3
    hidden_width: int = 6
    latent_dim: int = 16
    module_dim: int = 12
    rate_min: float = 0.05
    rate_max: float = 14.0


@dataclass(frozen=True, kw_only=True)
class TrainedModel:
    """Fitted model parameters, validation diagnostics, and compiled predictor."""

    spec: ModelSpec
    theta: np.ndarray
    n_params: int
    best_val_loss: float
    chosen_reg: float
    chosen_seed: int
    predict_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray]


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
        DynamicalModelFamily.MEMGRN_INT,
    ):
        tree: dict[str, np.ndarray] = {
            "direct": rng.normal(0.0, 0.02, size=(s_dim, g_dim)).astype(np.float32),
            "delayed": rng.normal(0.0, 0.02, size=(s_dim, r_dim, g_dim)).astype(np.float32),
            "decay": np.zeros((g_dim,), dtype=np.float32),
            "rate": np.broadcast_to(base_logits, (s_dim, r_dim)).copy(),
        }
        if fam in (DynamicalModelFamily.MEMGRN, DynamicalModelFamily.MEMGRN_INT):
            tree["in_proj"] = rng.normal(0.0, 0.7, size=(3, h_dim)).astype(np.float32)
            tree["bias"] = rng.normal(0.0, 0.25, size=(h_dim,)).astype(np.float32)
            tree["out_proj"] = np.zeros((s_dim, r_dim, h_dim), dtype=np.float32)
        if fam == DynamicalModelFamily.MEMGRN_INT:
            n_pairs = s_dim * (s_dim - 1) // 2
            tree["interact"] = np.zeros((n_pairs, g_dim), dtype=np.float32)
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
):
    """Build JIT-compiled batched RK4 rollout for any DynamicalModelFamily.

    Args:
        spec: Model architecture specification.
        dt: RK4 integration step in hours.
        obs_stride: Number of RK4 steps between recorded observation knots (including t=0).

    Returns:
        Tuple (forward_fn, unpack_fn, template_vec) where forward_fn(theta, stage_u, y0)
        takes stage_u of shape (n_steps, batch, n_sources, 3) and y0 of shape
        (batch, n_targets), returning predicted trajectories of shape
        (batch, n_obs, n_targets) including t=0 at index 0.
    """
    template = _init_tree(spec, seed=0)
    template_j = {k: jnp.asarray(v) for k, v in template.items()}
    flat0, unpack = ravel_pytree(template_j)
    s_dim, g_dim, r_dim = spec.n_sources, spec.n_targets, spec.n_channels
    idx_i, idx_k = np.triu_indices(s_dim, k=1) if s_dim >= 2 else (np.array([]), np.array([]))
    fam = spec.family

    def forward(theta: jnp.ndarray, stage_u: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
        p = unpack(theta)
        decay = 0.1 + jnp.logaddexp(0.0, p["decay"])
        batch = stage_u.shape[1]

        if fam in (
            DynamicalModelFamily.LOCAL_CONST,
            DynamicalModelFamily.MEMGRN,
            DynamicalModelFamily.MEMGRN_INT,
        ):
            init_state = (
                jnp.zeros((batch, s_dim, r_dim), dtype=theta.dtype),
                jnp.zeros((batch, s_dim, r_dim), dtype=theta.dtype),
                y0,
            )

            def rhs(state, raw_u):
                z, h, y = state
                act = jnp.tanh(raw_u)
                logits = p["rate"][None, :, :]
                if fam in (DynamicalModelFamily.MEMGRN, DynamicalModelFamily.MEMGRN_INT):
                    feat_a = jnp.broadcast_to(act[:, :, None], z.shape)
                    hid = (
                        p["bias"]
                        + feat_a[..., None] * p["in_proj"][0]
                        + z[..., None] * p["in_proj"][1]
                        + h[..., None] * p["in_proj"][2]
                    )
                    logits = logits + jnp.sum(jnp.tanh(hid) * p["out_proj"][None], axis=-1)
                lam = spec.rate_min + (spec.rate_max - spec.rate_min) * jax.nn.sigmoid(logits)
                dz = lam * (act[:, :, None] - z)
                dh = lam * (z - h)
                prod = act @ p["direct"] + h.reshape(batch, s_dim * r_dim) @ p[
                    "delayed"
                ].reshape(s_dim * r_dim, g_dim)
                if fam == DynamicalModelFamily.MEMGRN_INT and s_dim >= 2:
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
                # Subtract equilibrium bias so x=0 is preserved at zero input a=0
                eq_shift = (spec.rate_max - spec.rate_min) * rest_gate * p["e_rev"]
                dx = -(base_tau_inv + gate) * x + gate * p["e_rev"] - eq_shift
                prod = act @ p["direct"] + x @ p["readout"]
                return dx, prod - decay * y

        elif fam == DynamicalModelFamily.LTC_LOCAL:
            init_state = (jnp.zeros((batch, s_dim, r_dim), dtype=theta.dtype), y0)
            rest_gate = jax.nn.sigmoid(p["b_gate"])

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
                eq_shift = (spec.rate_max - spec.rate_min) * rest_gate * p["e_rev"]
                dx = -(base_tau_inv[None] + gate) * x + gate * (
                    act[:, :, None] * p["e_rev"][None]
                )
                _ = eq_shift
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
) -> TrainedModel:
    """Train a dynamical model with validation-locked regularization & multi-start selection."""
    forward, unpack, flat_template = build_jax_rollout(spec, dt=dt, obs_stride=obs_stride)

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

    def data_mse(theta: jnp.ndarray, u_arr: jnp.ndarray, y_arr: jnp.ndarray, y0_arr: jnp.ndarray, w_arr: jnp.ndarray, denom: jnp.ndarray) -> jnp.ndarray:
        pred = forward(theta, u_arr, y0_arr)
        return jnp.sum(((pred - y_arr) ** 2) * w_arr) / denom

    def penalty(theta: jnp.ndarray) -> jnp.ndarray:
        p = unpack(theta)
        l2 = sum(jnp.sum(v**2) for k, v in p.items() if k not in ("decay", "rate", "tau_log"))
        if spec.family == DynamicalModelFamily.MEMGRN_INT and "interact" in p:
            l2 = l2 + l1_interact_scale * jnp.sum(jnp.abs(p["interact"]))
        return l2

    @jax.jit
    def run_single(theta0: jnp.ndarray, reg_val: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def obj(th: jnp.ndarray) -> jnp.ndarray:
            return data_mse(th, tr_u, tr_y, tr_y0, tr_w, tr_denom) + reg_val * penalty(th)

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
            v_loss = data_mse(th, va_u, va_y, va_y0, va_w, va_denom)
            better = v_loss < best_val
            best_th = jnp.where(better, th, best_th)
            best_val = jnp.where(better, v_loss, best_val)
            return (th, m, v, best_th, best_val), v_loss

        init_val = data_mse(theta0, va_u, va_y, va_y0, va_w, va_denom)
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

    for seed in seeds:
        init_dict = _init_tree(spec, seed=seed)
        th0 = ravel_pytree({k: jnp.asarray(v) for k, v in init_dict.items()})[0]
        for reg in reg_grid:
            th_opt, val_l = run_single(th0, jnp.asarray(reg, dtype=jnp.float32))
            v_float = float(val_l)
            th_np = np.asarray(th_opt, dtype=np.float32)
            if np.all(np.isfinite(th_np)) and v_float < best_overall_val:
                best_overall_val = v_float
                best_theta_np = th_np
                best_reg = reg
                best_seed = seed

    th_j = jnp.asarray(best_theta_np, dtype=jnp.float32)

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
    """Evaluate common model-agnostic input-intervention effects Delta_{i -> j}(t).

    For each regulator i in {0, ..., S - 1}, computes hat{y}(t; g) - hat{y}(t; g_{-i})
    where g_{-i} sets regulator i's trajectory to zero while preserving all other inputs.

    Returns:
        Array of shape (n_sources, batch, n_obs, n_targets).
    """
    base_pred = predict_fn(stage_u, y0)
    n_sources = stage_u.shape[2]
    effects = []
    for src_idx in range(n_sources):
        ablated = stage_u.copy()
        ablated[:, :, src_idx, :] = 0.0
        effects.append(base_pred - predict_fn(ablated, y0))
    return np.stack(effects, axis=0)
