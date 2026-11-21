"""Differentiable JAX training routines for joint MemGRN and frozen-readout ablations."""

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from memgrn.dynamics import RateInputMode


@dataclass(frozen=True, kw_only=True)
class FrozenReadoutParams:
    """Frozen explicit readout and relaxation parameters for a single target."""

    direct: float
    delayed: np.ndarray
    decay: float


@dataclass(frozen=True, kw_only=True)
class RateFitResult:
    """Optimization outcome for a frozen-readout rate-law fit."""

    chosen_start: int
    source_losses: list[float]
    best_theta: np.ndarray
    all_thetas: np.ndarray


def _active_rows(mode: RateInputMode) -> tuple[int, ...]:
    if mode == RateInputMode.CONSTANT:
        return ()
    if mode == RateInputMode.ACTIVITY:
        return (0,)
    if mode == RateInputMode.MEMORY:
        return (1, 2)
    return (0, 1, 2)


def make_rate_starts(
    mode: RateInputMode,
    seed: int,
    archived_rate_params: dict[str, np.ndarray] | None = None,
    n_channels: int = 3,
    hidden_width: int = 4,
) -> list[np.ndarray]:
    """Construct the three standard rate initializations (archived, untrained, dispersed)."""
    rows = _active_rows(mode)
    rng_init = np.random.default_rng(seed)
    base_rates = np.array([0.45, 1.2, 3.2], dtype=np.float32)[:n_channels]
    base_logits = np.log((base_rates - 0.05) / (14.0 - base_rates))

    untrained: dict[str, np.ndarray] = {"rate": base_logits.copy()}
    if rows:
        full_in = rng_init.normal(0.0, 0.85, size=(3, hidden_width)).astype(np.float32)
        untrained["input"] = full_in[list(rows)]
        untrained["bias"] = rng_init.normal(0.0, 0.28, size=(hidden_width,)).astype(
            np.float32
        )
        untrained["output"] = np.zeros((n_channels, hidden_width), dtype=np.float32)

    rng_disp = np.random.default_rng(seed + 1033)
    dispersed: dict[str, np.ndarray] = {
        "rate": (
            np.log((3.0 - 0.05) / (14.0 - 3.0))
            + rng_disp.normal(0.0, 0.6, size=(n_channels,))
        ).astype(np.float32)
    }
    if rows:
        full_in_d = rng_disp.normal(0.0, 0.85, size=(3, hidden_width)).astype(
            np.float32
        )
        dispersed["input"] = full_in_d[list(rows)]
        dispersed["bias"] = rng_disp.normal(0.0, 0.28, size=(hidden_width,)).astype(
            np.float32
        )
        dispersed["output"] = rng_disp.normal(
            0.0, 0.12, size=(n_channels, hidden_width)
        ).astype(np.float32)

    if archived_rate_params is not None:
        arch: dict[str, np.ndarray] = {
            "rate": np.asarray(archived_rate_params["rate"], dtype=np.float32)
        }
        if rows:
            arch["input"] = np.asarray(
                archived_rate_params["input"][list(rows)], dtype=np.float32
            )
            arch["bias"] = np.asarray(archived_rate_params["bias"], dtype=np.float32)
            arch["output"] = np.asarray(
                archived_rate_params["output"], dtype=np.float32
            )
    else:
        arch = {k: v.copy() for k, v in untrained.items()}

    return [ravel_pytree(d)[0] for d in (arch, untrained, dispersed)]


@lru_cache(maxsize=16)
def build_jax_frozen_readout_solver(
    mode: RateInputMode,
    substeps_per_knot: int = 8,
    n_channels: int = 3,
    hidden_width: int = 4,
):
    """Compile JAX RK4 forward pass and clipped cosine-Adam trainer for frozen readout."""
    template_starts = make_rate_starts(
        mode, seed=0, n_channels=n_channels, hidden_width=hidden_width
    )
    rows = _active_rows(mode)
    template_dict: dict[str, jnp.ndarray] = {
        "rate": jnp.zeros((n_channels,), dtype=jnp.float32)
    }
    if rows:
        template_dict["input"] = jnp.zeros(
            (len(rows), hidden_width), dtype=jnp.float32
        )
        template_dict["bias"] = jnp.zeros((hidden_width,), dtype=jnp.float32)
        template_dict["output"] = jnp.zeros(
            (n_channels, hidden_width), dtype=jnp.float32
        )
    _, unpack = ravel_pytree(template_dict)
    _ = template_starts  # ensure identical tree layout
    dt = 2.5 / (60.0 * substeps_per_knot)

    def forward(
        theta: jnp.ndarray,
        frozen_tuple: tuple[float, jnp.ndarray, float],
        tanh_stages: jnp.ndarray,
    ) -> jnp.ndarray:
        p = unpack(theta)
        direct, mix, decay = frozen_tuple

        def rhs(state, a_val):
            z, h, y = state
            logits = p["rate"]
            if rows:
                v = p["bias"]
                for k, row_idx in enumerate(rows):
                    feat = (a_val[:, None], z, h)[row_idx]
                    v = v + feat[..., None] * p["input"][k]
                logits = logits + jnp.sum(jnp.tanh(v) * p["output"], axis=-1)
            rate = 0.05 + 13.95 * jax.nn.sigmoid(logits)
            return (
                rate * (a_val[:, None] - z),
                rate * (z - h),
                a_val * direct + h @ mix - decay * y,
            )

        def advance(state, step_inputs):
            a0, am, a1 = step_inputs.T
            k1 = rhs(state, a0)
            k2 = rhs(jax.tree.map(lambda x, d: x + 0.5 * dt * d, state, k1), am)
            k3 = rhs(jax.tree.map(lambda x, d: x + 0.5 * dt * d, state, k2), am)
            k4 = rhs(jax.tree.map(lambda x, d: x + dt * d, state, k3), a1)
            nxt = jax.tree.map(
                lambda x, d1, d2, d3, d4: x + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4),
                state,
                k1,
                k2,
                k3,
                k4,
            )
            return nxt, nxt[-1]

        n_sched = tanh_stages.shape[1]
        zeros_zh = jnp.zeros((n_sched, n_channels), dtype=theta.dtype)
        zeros_y = jnp.zeros((n_sched,), dtype=theta.dtype)
        _, ys = jax.lax.scan(advance, (zeros_zh, zeros_zh, zeros_y), tanh_stages)
        return ys[2 * substeps_per_knot - 1 :: 2 * substeps_per_knot].T

    return jax.jit(forward), unpack


def fit_frozen_readout_rates(
    mode: RateInputMode,
    frozen: FrozenReadoutParams,
    tanh_stages: np.ndarray,
    targets: np.ndarray,
    loss_weights: np.ndarray,
    seed: int,
    archived_rate_params: dict[str, np.ndarray] | None = None,
    total_updates: int = 2400,
    substeps_per_knot: int = 8,
) -> RateFitResult:
    """Optimize local filter rate parameters on source schedules with frozen (D, B, delta)."""
    forward, _ = build_jax_frozen_readout_solver(mode, substeps_per_knot)
    frozen_tuple = (
        float(frozen.direct),
        jnp.asarray(frozen.delayed, dtype=jnp.float32),
        float(frozen.decay),
    )
    stages_j = jnp.asarray(tanh_stages, dtype=jnp.float32)
    targets_j = jnp.asarray(targets, dtype=jnp.float32)
    weights_j = jnp.asarray(loss_weights, dtype=jnp.float32)

    def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
        pred = forward(theta, frozen_tuple, stages_j)
        return jnp.sum(((pred - targets_j) ** 2) * weights_j)

    @jax.jit
    def train_loop(theta0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def body(i: int, carry):
            th, m, v = carry
            grad = jax.grad(loss_fn)(th)
            grad = grad * jnp.minimum(1.0, 4.0 / (jnp.linalg.norm(grad) + 1e-9))
            step = i + 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad * grad)
            lr = 0.0008 + 0.0082 * 0.5 * (1.0 + jnp.cos(jnp.pi * step / total_updates))
            m_hat = m / (1.0 - 0.9**step)
            v_hat = v / (1.0 - 0.999**step)
            th = th - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            return th, m, v

        final_th, _, _ = jax.lax.fori_loop(
            0,
            total_updates,
            body,
            (theta0, jnp.zeros_like(theta0), jnp.zeros_like(theta0)),
        )
        return final_th, loss_fn(final_th)

    starts = make_rate_starts(
        mode, seed=seed, archived_rate_params=archived_rate_params
    )
    fitted = []
    losses = []
    for th0 in starts:
        th_opt, l_val = train_loop(th0)
        th_np = np.asarray(th_opt, dtype=np.float32)
        if not np.all(np.isfinite(th_np)):
            raise ArithmeticError(f"Non-finite parameters in {mode} seed={seed}")
        fitted.append(th_np)
        losses.append(float(l_val))

    chosen = int(np.argmin(losses))
    return RateFitResult(
        chosen_start=chosen,
        source_losses=losses,
        best_theta=fitted[chosen],
        all_thetas=np.stack(fitted, axis=0),
    )
