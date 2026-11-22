"""Standard discrete-time sequence baselines (DLinear, GRU, Temporal Transformer) in pure JAX.

Provides matched discrete-time deep learning comparators for both:
  1. Open-loop driven sequence forecasting: u[0..T-1] (S drivers) -> y[0..T-1] (N targets),
     with counterfactual source attribution Delta_{i -> j}(t) = y(u) - y(u_{-i}).
  2. Closed-loop autonomous network forecasting: observed history y[0..H-1] (S nodes) ->
     autonomous future rollout y[H..H+F-1] (S nodes), plus finite-difference sensitivity
     interaction graphs for unsupervised physical/social topology discovery.

All baselines share clipped cosine-Adam optimization, multi-seed + validation weight-decay
selection, and deterministic CPU execution with the continuous-time suite in ``baselines.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


class DiscreteModelFamily(StrEnum):
    """Standard discrete-time deep learning time-series architectures."""

    DLINEAR = "dlinear"
    GRU = "gru"
    TRANSFORMER = "transformer"


@dataclass(frozen=True, kw_only=True)
class TrainedDiscreteModel:
    """Fitted discrete-time sequence model with predictor and optional interaction graph."""

    family: DiscreteModelFamily
    n_params: int
    best_val_loss: float
    chosen_reg: float
    chosen_seed: int
    predict_fn: Callable[[np.ndarray], np.ndarray]
    interaction_matrix: np.ndarray | None = None


def _cosine_adam_train(
    loss_fn: Callable,
    val_fn: Callable,
    theta0: jnp.ndarray,
    total_updates: int,
    lr_max: float,
    val_every: int = 25,
) -> tuple[np.ndarray, float]:
    n_chunks = max(1, total_updates // val_every)

    @jax.jit
    def _run(th_init: jnp.ndarray):
        def chunk_step(carry, chunk_idx):
            th, m, v, best_th, best_val = carry

            def inner_body(k_i, inner_carry):
                th_i, m_i, v_i = inner_carry
                step = chunk_idx * val_every + k_i + 1
                _, grads = jax.value_and_grad(loss_fn)(th_i)
                g_norm = jnp.linalg.norm(grads)
                grads = jnp.where(g_norm > 5.0, grads * (5.0 / (g_norm + 1e-8)), grads)
                m_i = 0.9 * m_i + 0.1 * grads
                v_i = 0.999 * v_i + 0.001 * (grads**2)
                m_hat = m_i / (1.0 - 0.9**step)
                v_hat = v_i / (1.0 - 0.999**step)
                prog = jnp.minimum(1.0, step / float(total_updates))
                lr = lr_max * (0.05 + 0.475 * (1.0 + jnp.cos(jnp.pi * prog)))
                th_i = th_i - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
                return th_i, m_i, v_i

            th, m, v = jax.lax.fori_loop(0, val_every, inner_body, (th, m, v))
            v_l = val_fn(th)
            better = jnp.isfinite(v_l) & (v_l < best_val)
            best_th = jnp.where(better, th, best_th)
            best_val = jnp.where(better, v_l, best_val)
            return (th, m, v, best_th, best_val), v_l

        init_v = val_fn(th_init)
        carry0 = (th_init, jnp.zeros_like(th_init), jnp.zeros_like(th_init), th_init, init_v)
        (_, _, _, best_th, best_val), _ = jax.lax.scan(chunk_step, carry0, jnp.arange(n_chunks))
        return best_th, best_val

    b_th, b_val = _run(theta0)
    return np.asarray(b_th, dtype=np.float32), float(b_val)


# =============================================================================
# 1. Open-Loop Driven Discrete-Time Baselines: u (B, T, S) -> y (B, T, N)
# =============================================================================


def _init_open_discrete(
    family: DiscreteModelFamily,
    n_sources: int,
    n_targets: int,
    seq_len: int,
    hidden_dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if family == DiscreteModelFamily.DLINEAR:
        # Causal lower-triangular temporal decomposition + linear channel projection (trend + seasonal)
        return {
            "w_trend_t": rng.normal(0.0, 0.15 / np.sqrt(seq_len), size=(seq_len, seq_len)).astype(np.float32),
            "w_seas_t": rng.normal(0.0, 0.15 / np.sqrt(seq_len), size=(seq_len, seq_len)).astype(np.float32),
            "w_trend_c": rng.normal(0.0, 0.25 / np.sqrt(n_sources), size=(n_sources, n_targets)).astype(np.float32),
            "w_seas_c": rng.normal(0.0, 0.25 / np.sqrt(n_sources), size=(n_sources, n_targets)).astype(np.float32),
            "bias": np.zeros((n_targets,), dtype=np.float32),
        }
    if family == DiscreteModelFamily.GRU:
        d = hidden_dim
        s_in = n_sources
        return {
            "w_z": rng.normal(0.0, 0.35 / np.sqrt(s_in + d), size=(s_in + d, d)).astype(np.float32),
            "b_z": np.zeros((d,), dtype=np.float32),
            "w_r": rng.normal(0.0, 0.35 / np.sqrt(s_in + d), size=(s_in + d, d)).astype(np.float32),
            "b_r": np.zeros((d,), dtype=np.float32),
            "w_h": rng.normal(0.0, 0.35 / np.sqrt(s_in + d), size=(s_in + d, d)).astype(np.float32),
            "b_h": np.zeros((d,), dtype=np.float32),
            "w_out": rng.normal(0.0, 0.25 / np.sqrt(d), size=(d, n_targets)).astype(np.float32),
            "b_out": np.zeros((n_targets,), dtype=np.float32),
        }
    # TRANSFORMER: 1-layer causal pre-norm self-attention + FFN with positional embeddings
    d = hidden_dim
    return {
        "w_in": rng.normal(0.0, 0.4 / np.sqrt(n_sources), size=(n_sources, d)).astype(np.float32),
        "b_in": np.zeros((d,), dtype=np.float32),
        "pos": rng.normal(0.0, 0.05, size=(seq_len, d)).astype(np.float32),
        "w_q": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, d)).astype(np.float32),
        "w_k": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, d)).astype(np.float32),
        "w_v": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, d)).astype(np.float32),
        "w_o": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, d)).astype(np.float32),
        "ff1": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, 2 * d)).astype(np.float32),
        "bf1": np.zeros((2 * d,), dtype=np.float32),
        "ff2": rng.normal(0.0, 0.35 / np.sqrt(2 * d), size=(2 * d, d)).astype(np.float32),
        "bf2": np.zeros((d,), dtype=np.float32),
        "w_out": rng.normal(0.0, 0.25 / np.sqrt(d), size=(d, n_targets)).astype(np.float32),
        "b_out": np.zeros((n_targets,), dtype=np.float32),
    }


def _causal_moving_avg(x: jnp.ndarray, kernel: int = 5) -> jnp.ndarray:
    """Strictly causal moving average along axis=1 for DLinear decomposition."""
    pad = jnp.repeat(x[:, :1, :], kernel - 1, axis=1)
    xp = jnp.concatenate([pad, x], axis=1)
    cs = jnp.cumsum(xp, axis=1)
    zeros = jnp.zeros_like(x[:, :1, :])
    cs_pad = jnp.concatenate([zeros, cs], axis=1)
    return (cs_pad[:, kernel:, :] - cs_pad[:, :-kernel, :]) / float(kernel)


def fit_discrete_open_loop(
    family: DiscreteModelFamily,
    train_u: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    val_u: np.ndarray,
    val_y: np.ndarray,
    val_mask: np.ndarray,
    hidden_dim: int = 32,
    seeds: tuple[int, ...] = (101, 202),
    reg_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2),
    total_updates: int = 600,
    lr_max: float = 0.005,
) -> TrainedDiscreteModel:
    """Fit a causal discrete-time open-loop baseline (DLinear, GRU, or Transformer)."""
    _, seq_len, n_sources = train_u.shape
    n_targets = train_y.shape[2]

    init0 = _init_open_discrete(family, n_sources, n_targets, seq_len, hidden_dim, seed=0)
    _, unpack = ravel_pytree({k: jnp.asarray(v) for k, v in init0.items()})
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32))

    def forward(theta: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        p = unpack(theta)
        b_sz = u.shape[0]
        if family == DiscreteModelFamily.DLINEAR:
            tr = _causal_moving_avg(u, kernel=5)
            se = u - tr
            wt = p["w_trend_t"] * causal_mask
            ws = p["w_seas_t"] * causal_mask
            # Causal temporal mix (B, T, S) then channel readout -> (B, T, N)
            tr_t = jnp.einsum("tk,bks->bts", wt, tr)
            se_t = jnp.einsum("tk,bks->bts", ws, se)
            out = tr_t @ p["w_trend_c"] + se_t @ p["w_seas_c"] + p["bias"]
            return out

        if family == DiscreteModelFamily.GRU:
            h0 = jnp.zeros((b_sz, hidden_dim), dtype=theta.dtype)

            def step(h, u_t):
                xh = jnp.concatenate([u_t, h], axis=-1)
                z = jax.nn.sigmoid(xh @ p["w_z"] + p["b_z"])
                r = jax.nn.sigmoid(xh @ p["w_r"] + p["b_r"])
                xrh = jnp.concatenate([u_t, r * h], axis=-1)
                h_cand = jnp.tanh(xrh @ p["w_h"] + p["b_h"])
                h_nxt = (1.0 - z) * h + z * h_cand
                y_t = h_nxt @ p["w_out"] + p["b_out"]
                return h_nxt, y_t

            _, ys = jax.lax.scan(step, h0, jnp.transpose(u, (1, 0, 2)))
            out = jnp.transpose(ys, (1, 0, 2))
            return out

        # Causal Temporal Transformer (4 heads)
        x = u @ p["w_in"] + p["b_in"] + p["pos"][None, :, :]
        # Pre-LayerNorm
        x_n = (x - jnp.mean(x, axis=-1, keepdims=True)) / (
            jnp.std(x, axis=-1, keepdims=True) + 1e-5
        )
        n_heads = 4
        d_h = hidden_dim // n_heads
        q = (x_n @ p["w_q"]).reshape(b_sz, seq_len, n_heads, d_h).transpose(0, 2, 1, 3)
        k = (x_n @ p["w_k"]).reshape(b_sz, seq_len, n_heads, d_h).transpose(0, 2, 1, 3)
        v = (x_n @ p["w_v"]).reshape(b_sz, seq_len, n_heads, d_h).transpose(0, 2, 1, 3)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(float(d_h))
        logits = jnp.where(causal_mask[None, None, :, :] > 0.5, logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", attn, v).transpose(0, 2, 1, 3).reshape(
            b_sz, seq_len, hidden_dim
        )
        x = x + ctx @ p["w_o"]
        x_n2 = (x - jnp.mean(x, axis=-1, keepdims=True)) / (
            jnp.std(x, axis=-1, keepdims=True) + 1e-5
        )
        x = x + jax.nn.gelu(x_n2 @ p["ff1"] + p["bf1"]) @ p["ff2"] + p["bf2"]
        out = x @ p["w_out"] + p["b_out"]
        return out

    tr_u_j = jnp.asarray(train_u, dtype=jnp.float32)
    tr_y_j = jnp.asarray(train_y, dtype=jnp.float32)
    tr_m_j = jnp.asarray(train_mask, dtype=jnp.float32)
    va_u_j = jnp.asarray(val_u, dtype=jnp.float32)
    va_y_j = jnp.asarray(val_y, dtype=jnp.float32)
    va_m_j = jnp.asarray(val_mask, dtype=jnp.float32)
    tr_den = jnp.maximum(1.0, jnp.sum(tr_m_j))
    va_den = jnp.maximum(1.0, jnp.sum(va_m_j))

    best_val = float("inf")
    best_th = None
    best_reg = reg_grid[0]
    best_seed = seeds[0]

    for seed in seeds:
        init_d = _init_open_discrete(family, n_sources, n_targets, seq_len, hidden_dim, seed)
        th0 = ravel_pytree({k: jnp.asarray(v) for k, v in init_d.items()})[0]
        for reg in reg_grid:
            reg_j = jnp.asarray(reg, dtype=jnp.float32)

            def loss_fn(th):
                pred = forward(th, tr_u_j)
                mse = jnp.sum(((pred - tr_y_j) ** 2) * tr_m_j) / tr_den
                return mse + reg_j * jnp.mean(th**2)

            def val_fn(th):
                pred = forward(th, va_u_j)
                return jnp.sum(((pred - va_y_j) ** 2) * va_m_j) / va_den

            th_opt, v_l = _cosine_adam_train(loss_fn, val_fn, th0, total_updates, lr_max)
            if np.all(np.isfinite(th_opt)) and v_l < best_val:
                best_val = v_l
                best_th = th_opt
                best_reg = reg
                best_seed = seed

    assert best_th is not None
    th_best_j = jnp.asarray(best_th, dtype=jnp.float32)
    fwd_jit = jax.jit(forward)

    def predict_fn(u_np: np.ndarray) -> np.ndarray:
        return np.asarray(fwd_jit(th_best_j, jnp.asarray(u_np, dtype=jnp.float32)), dtype=np.float64)

    return TrainedDiscreteModel(
        family=family,
        n_params=int(best_th.size),
        best_val_loss=best_val,
        chosen_reg=best_reg,
        chosen_seed=best_seed,
        predict_fn=predict_fn,
    )


def discrete_counterfactual_attribution(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    u_obs: np.ndarray,
) -> np.ndarray:
    """Compute Delta_{i -> j}(t) = y(u) - y(u_{-i}) for a discrete-time open-loop model."""
    base = predict_fn(u_obs)
    n_s = u_obs.shape[2]
    effs = []
    for i in range(n_s):
        abl = u_obs.copy()
        abl[:, :, i] = 0.0
        effs.append(base - predict_fn(abl))
    return np.stack(effs, axis=0)


# =============================================================================
# 2. Closed-Loop Discrete-Time Baselines: hist (B, H, S) -> fut (B, F, S)
# =============================================================================


def _init_closed_discrete(
    family: DiscreteModelFamily,
    n_nodes: int,
    hist_len: int,
    fut_len: int,
    hidden_dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if family == DiscreteModelFamily.DLINEAR:
        # Standard channel-independent DLinear + optional low-rank cross-node residual
        # (matches Zeng et al. 2023 DLinear decomposition + lightweight cross-node interaction)
        k_r = min(24, n_nodes)
        return {
            "w_trend": rng.normal(0.0, 0.2 / np.sqrt(hist_len), size=(hist_len, fut_len)).astype(np.float32),
            "b_trend": np.zeros((fut_len,), dtype=np.float32),
            "w_seas": rng.normal(0.0, 0.2 / np.sqrt(hist_len), size=(hist_len, fut_len)).astype(np.float32),
            "b_seas": np.zeros((fut_len,), dtype=np.float32),
            "cross_u": rng.normal(0.0, 0.1 / np.sqrt(n_nodes), size=(n_nodes, k_r)).astype(np.float32),
            "cross_v": np.zeros((k_r, n_nodes), dtype=np.float32),
        }
    if family == DiscreteModelFamily.GRU:
        d = hidden_dim
        return {
            "w_z": rng.normal(0.0, 0.3 / np.sqrt(n_nodes + d), size=(n_nodes + d, d)).astype(np.float32),
            "b_z": np.zeros((d,), dtype=np.float32),
            "w_r": rng.normal(0.0, 0.3 / np.sqrt(n_nodes + d), size=(n_nodes + d, d)).astype(np.float32),
            "b_r": np.zeros((d,), dtype=np.float32),
            "w_h": rng.normal(0.0, 0.3 / np.sqrt(n_nodes + d), size=(n_nodes + d, d)).astype(np.float32),
            "b_h": np.zeros((d,), dtype=np.float32),
            "w_out": rng.normal(0.0, 0.2 / np.sqrt(d), size=(d, n_nodes)).astype(np.float32),
            "b_out": np.zeros((n_nodes,), dtype=np.float32),
        }
    # TRANSFORMER: history encoder with self-attention -> multi-horizon projection
    d = hidden_dim
    return {
        "w_in": rng.normal(0.0, 0.35 / np.sqrt(n_nodes), size=(n_nodes, d)).astype(np.float32),
        "b_in": np.zeros((d,), dtype=np.float32),
        "pos": rng.normal(0.0, 0.05, size=(hist_len, d)).astype(np.float32),
        "w_q": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, d)).astype(np.float32),
        "w_k": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, d)).astype(np.float32),
        "w_v": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, d)).astype(np.float32),
        "w_o": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, d)).astype(np.float32),
        "ff1": rng.normal(0.0, 0.35 / np.sqrt(d), size=(d, 2 * d)).astype(np.float32),
        "bf1": np.zeros((2 * d,), dtype=np.float32),
        "ff2": rng.normal(0.0, 0.35 / np.sqrt(2 * d), size=(2 * d, d)).astype(np.float32),
        "bf2": np.zeros((d,), dtype=np.float32),
        "w_t": rng.normal(0.0, 0.2 / np.sqrt(hist_len), size=(hist_len, fut_len)).astype(np.float32),
        "w_out": rng.normal(0.0, 0.2 / np.sqrt(d), size=(d, n_nodes)).astype(np.float32),
        "b_out": np.zeros((n_nodes,), dtype=np.float32),
    }


def fit_discrete_closed_loop(
    family: DiscreteModelFamily,
    train_hist: np.ndarray,
    train_fut: np.ndarray,
    train_fut_mask: np.ndarray,
    val_hist: np.ndarray,
    val_fut: np.ndarray,
    val_fut_mask: np.ndarray,
    hidden_dim: int = 48,
    seeds: tuple[int, ...] = (101, 202),
    reg_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2),
    total_updates: int = 550,
    lr_max: float = 0.005,
) -> TrainedDiscreteModel:
    """Fit discrete-time multi-step closed-loop network predictor (DLinear, GRU, Transformer)."""
    _, hist_len, n_nodes = train_hist.shape
    fut_len = train_fut.shape[1]

    init0 = _init_closed_discrete(family, n_nodes, hist_len, fut_len, hidden_dim, seed=0)
    _, unpack = ravel_pytree({k: jnp.asarray(v) for k, v in init0.items()})

    def forward(theta: jnp.ndarray, hist: jnp.ndarray) -> jnp.ndarray:
        p = unpack(theta)
        b_sz = hist.shape[0]
        if family == DiscreteModelFamily.DLINEAR:
            tr = _causal_moving_avg(hist, kernel=3)
            se = hist - tr
            y_tr = jnp.einsum("bhn,hf->bfn", tr, p["w_trend"]) + p["b_trend"][None, :, None]
            y_se = jnp.einsum("bhn,hf->bfn", se, p["w_seas"]) + p["b_seas"][None, :, None]
            base = y_tr + y_se
            cross = (base @ p["cross_u"]) @ p["cross_v"]
            return base + cross

        if family == DiscreteModelFamily.GRU:
            h0 = jnp.zeros((b_sz, hidden_dim), dtype=theta.dtype)

            def enc_step(h, x_t):
                xh = jnp.concatenate([x_t, h], axis=-1)
                z = jax.nn.sigmoid(xh @ p["w_z"] + p["b_z"])
                r = jax.nn.sigmoid(xh @ p["w_r"] + p["b_r"])
                xrh = jnp.concatenate([x_t, r * h], axis=-1)
                hc = jnp.tanh(xrh @ p["w_h"] + p["b_h"])
                return (1.0 - z) * h + z * hc, None

            h_enc, _ = jax.lax.scan(enc_step, h0, jnp.transpose(hist, (1, 0, 2)))

            def dec_step(carry, _):
                h, y_prev = carry
                xh = jnp.concatenate([y_prev, h], axis=-1)
                z = jax.nn.sigmoid(xh @ p["w_z"] + p["b_z"])
                r = jax.nn.sigmoid(xh @ p["w_r"] + p["b_r"])
                xrh = jnp.concatenate([y_prev, r * h], axis=-1)
                hc = jnp.tanh(xrh @ p["w_h"] + p["b_h"])
                h_nxt = (1.0 - z) * h + z * hc
                y_nxt = h_nxt @ p["w_out"] + p["b_out"]
                return (h_nxt, y_nxt), y_nxt

            _, ys = jax.lax.scan(dec_step, (h_enc, hist[:, -1, :]), None, length=fut_len)
            return jnp.transpose(ys, (1, 0, 2))

        # Temporal Transformer encoder -> multi-step horizon readout
        x = hist @ p["w_in"] + p["b_in"] + p["pos"][None, :, :]
        x_n = (x - jnp.mean(x, axis=-1, keepdims=True)) / (
            jnp.std(x, axis=-1, keepdims=True) + 1e-5
        )
        n_heads = 4
        d_h = hidden_dim // n_heads
        q = (x_n @ p["w_q"]).reshape(b_sz, hist_len, n_heads, d_h).transpose(0, 2, 1, 3)
        k = (x_n @ p["w_k"]).reshape(b_sz, hist_len, n_heads, d_h).transpose(0, 2, 1, 3)
        v = (x_n @ p["w_v"]).reshape(b_sz, hist_len, n_heads, d_h).transpose(0, 2, 1, 3)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(float(d_h))
        attn = jax.nn.softmax(logits, axis=-1)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", attn, v).transpose(0, 2, 1, 3).reshape(
            b_sz, hist_len, hidden_dim
        )
        x = x + ctx @ p["w_o"]
        x_n2 = (x - jnp.mean(x, axis=-1, keepdims=True)) / (
            jnp.std(x, axis=-1, keepdims=True) + 1e-5
        )
        x = x + jax.nn.gelu(x_n2 @ p["ff1"] + p["bf1"]) @ p["ff2"] + p["bf2"]
        # Temporal projection (hist -> fut) then node readout
        xf = jnp.einsum("bhd,hf->bfd", x, p["w_t"])
        return xf @ p["w_out"] + p["b_out"]

    tr_h_j = jnp.asarray(train_hist, dtype=jnp.float32)
    tr_f_j = jnp.asarray(train_fut, dtype=jnp.float32)
    tr_m_j = jnp.asarray(train_fut_mask, dtype=jnp.float32)
    va_h_j = jnp.asarray(val_hist, dtype=jnp.float32)
    va_f_j = jnp.asarray(val_fut, dtype=jnp.float32)
    va_m_j = jnp.asarray(val_fut_mask, dtype=jnp.float32)
    tr_den = jnp.maximum(1.0, jnp.sum(tr_m_j))
    va_den = jnp.maximum(1.0, jnp.sum(va_m_j))

    best_val = float("inf")
    best_th = None
    best_reg = reg_grid[0]
    best_seed = seeds[0]

    for seed in seeds:
        init_d = _init_closed_discrete(family, n_nodes, hist_len, fut_len, hidden_dim, seed)
        th0 = ravel_pytree({k: jnp.asarray(v) for k, v in init_d.items()})[0]
        for reg in reg_grid:
            reg_j = jnp.asarray(reg, dtype=jnp.float32)

            def loss_fn(th):
                pred = forward(th, tr_h_j)
                mse = jnp.sum(((pred - tr_f_j) ** 2) * tr_m_j) / tr_den
                return mse + reg_j * jnp.mean(th**2)

            def val_fn(th):
                pred = forward(th, va_h_j)
                return jnp.sum(((pred - va_f_j) ** 2) * va_m_j) / va_den

            th_opt, v_l = _cosine_adam_train(loss_fn, val_fn, th0, total_updates, lr_max)
            if np.all(np.isfinite(th_opt)) and v_l < best_val:
                best_val = v_l
                best_th = th_opt
                best_reg = reg
                best_seed = seed

    assert best_th is not None
    th_best_j = jnp.asarray(best_th, dtype=jnp.float32)
    fwd_jit = jax.jit(forward)

    # Extract directed source->target sensitivity graph via input perturbation of each node i
    base_h = np.zeros((1, hist_len, n_nodes), dtype=np.float32)
    y0_b = np.asarray(fwd_jit(th_best_j, jnp.asarray(base_h)), dtype=np.float64)[0]
    pert_batch = np.zeros((n_nodes, hist_len, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        pert_batch[i, :, i] = 0.5
    yp_b = np.asarray(fwd_jit(th_best_j, jnp.asarray(pert_batch)), dtype=np.float64)
    # RMS early-horizon (first 4 steps) cross-node response magnitude
    graph = np.sqrt(np.mean((yp_b[:, : min(4, fut_len), :] - y0_b[None, : min(4, fut_len), :]) ** 2, axis=1))
    np.fill_diagonal(graph, 0.0)

    def predict_fn(hist_np: np.ndarray) -> np.ndarray:
        return np.asarray(fwd_jit(th_best_j, jnp.asarray(hist_np, dtype=jnp.float32)), dtype=np.float64)

    return TrainedDiscreteModel(
        family=family,
        n_params=int(best_th.size),
        best_val_loss=best_val,
        chosen_reg=best_reg,
        chosen_seed=best_seed,
        predict_fn=predict_fn,
        interaction_matrix=graph,
    )
