"""Source-local adaptive memory dynamics, explicit readouts, and response accounting."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.special import expit

_DEFAULT_RATE_INIT = (0.45, 1.2, 3.2)


class RateInputMode(StrEnum):
    """State variables permitted to modulate source-local filter rates."""

    CONSTANT = "constant"
    ACTIVITY = "activity"
    MEMORY = "memory"
    BOTH = "both"


@dataclass(frozen=True, kw_only=True)
class MemGRNConfig:
    """Architectural configuration for MemGRN."""

    n_sources: int
    n_targets: int
    n_channels: int = 3
    hidden_width: int = 4
    rate_min: float = 0.05
    rate_max: float = 14.0
    rate_mode: RateInputMode = RateInputMode.BOTH
    use_activity_switch: bool = False
    switch_threshold: float = 0.1
    use_pairwise_interactions: bool = False


@dataclass(frozen=True, kw_only=True)
class MemGRNParams:
    """Explicit readout and source-local kinetic parameters for MemGRN.

    Attributes:
        direct_weights: Immediate source-to-target weights D of shape (n_sources, n_targets).
        delayed_weights: Delayed channel-to-target weights B of shape
            (n_sources, n_channels, n_targets).
        relaxation: Positive target relaxation rates delta of shape (n_targets,).
        rate_logits: Baseline rate logits ell of shape (n_sources, n_channels).
        switch_offsets: Optional discrete activity-switch offsets s of shape
            (n_sources, n_channels), or None when disabled.
        input_proj: Shared local projection A of shape (n_active_inputs, hidden_width),
            or None for constant-rate variants.
        hidden_bias: Hidden bias b of shape (hidden_width,), or None for constant rates.
        output_proj: Channel-specific output weights v of shape
            (n_sources, n_channels, hidden_width), or None for constant rates.
        interaction_weights: Optional pairwise source-memory interaction weights Gamma of shape
            (n_sources * (n_sources - 1) // 2, n_targets) for MemGRN-Int enhancer cooperativity.
    """

    direct_weights: np.ndarray
    delayed_weights: np.ndarray
    relaxation: np.ndarray
    rate_logits: np.ndarray
    switch_offsets: np.ndarray | None = None
    input_proj: np.ndarray | None = None
    hidden_bias: np.ndarray | None = None
    output_proj: np.ndarray | None = None
    interaction_weights: np.ndarray | None = None


def initialize_memgrn_params(
    config: MemGRNConfig,
    seed: int = 0,
) -> MemGRNParams:
    """Initialize MemGRN parameters with standard channel timescales."""
    rng = np.random.default_rng(seed)
    base_rates = np.resize(
        np.asarray(_DEFAULT_RATE_INIT, dtype=np.float64), config.n_channels
    )
    clipped = np.clip(base_rates, config.rate_min + 1e-3, config.rate_max - 1e-3)
    logits_1d = np.log(
        (clipped - config.rate_min) / (config.rate_max - clipped)
    )
    rate_logits = np.broadcast_to(
        logits_1d, (config.n_sources, config.n_channels)
    ).copy()

    direct = rng.normal(0.0, 0.015, size=(config.n_sources, config.n_targets))
    delayed = rng.normal(
        0.0, 0.015, size=(config.n_sources, config.n_channels, config.n_targets)
    )
    relaxation = np.full(config.n_targets, 0.1 + np.logaddexp(0.0, 0.0))

    switch_offsets = (
        np.zeros((config.n_sources, config.n_channels), dtype=np.float64)
        if config.use_activity_switch
        else None
    )

    n_pairs = config.n_sources * (config.n_sources - 1) // 2
    inter_w = (
        np.zeros((n_pairs, config.n_targets), dtype=np.float64)
        if config.use_pairwise_interactions and n_pairs > 0
        else None
    )

    n_inputs = _active_input_indices(config.rate_mode)
    if not n_inputs:
        return MemGRNParams(
            direct_weights=direct,
            delayed_weights=delayed,
            relaxation=relaxation,
            rate_logits=rate_logits,
            switch_offsets=switch_offsets,
            interaction_weights=inter_w,
        )

    input_proj = rng.normal(0.0, 0.85, size=(len(n_inputs), config.hidden_width))
    hidden_bias = rng.normal(0.0, 0.28, size=(config.hidden_width,))
    output_proj = np.zeros(
        (config.n_sources, config.n_channels, config.hidden_width), dtype=np.float64
    )
    return MemGRNParams(
        direct_weights=direct,
        delayed_weights=delayed,
        relaxation=relaxation,
        rate_logits=rate_logits,
        switch_offsets=switch_offsets,
        input_proj=input_proj,
        hidden_bias=hidden_bias,
        output_proj=output_proj,
        interaction_weights=inter_w,
    )


def _active_input_indices(mode: RateInputMode) -> tuple[int, ...]:
    if mode == RateInputMode.CONSTANT:
        return ()
    if mode == RateInputMode.ACTIVITY:
        return (0,)
    if mode == RateInputMode.MEMORY:
        return (1, 2)
    return (0, 1, 2)


def adaptive_rate_law(
    raw_input: np.ndarray,
    activity: np.ndarray,
    z_state: np.ndarray,
    h_state: np.ndarray,
    params: MemGRNParams,
    config: MemGRNConfig,
) -> np.ndarray:
    """Compute bounded source-local channel rates lambda_{ir}(a_i, z_{ir}, h_{ir}).

    Args:
        raw_input: Uncompressed TF input g_i of shape (..., n_sources).
        activity: Saturated TF activity a_i = tanh(g_i) of shape (..., n_sources).
        z_state: Upstream filter state z_{ir} of shape (..., n_sources, n_channels).
        h_state: Delayed filter state h_{ir} of shape (..., n_sources, n_channels).
        params: Model parameters.
        config: Architectural configuration.

    Returns:
        Positive bounded filter rates in [rate_min, rate_max] of shape
        (..., n_sources, n_channels).
    """
    logits = np.broadcast_to(params.rate_logits, z_state.shape).copy()
    if config.use_activity_switch and params.switch_offsets is not None:
        gate = (np.abs(raw_input) > config.switch_threshold)[..., None].astype(
            np.float64
        )
        logits = logits + gate * params.switch_offsets

    indices = _active_input_indices(config.rate_mode)
    if indices and params.input_proj is not None and params.output_proj is not None:
        features = (
            np.broadcast_to(activity[..., None], z_state.shape),
            z_state,
            h_state,
        )
        hidden = np.broadcast_to(
            params.hidden_bias, (*z_state.shape, config.hidden_width)
        ).copy()
        for proj_row, feat_idx in enumerate(indices):
            hidden = hidden + features[feat_idx][..., None] * params.input_proj[proj_row]
        logits = logits + np.sum(np.tanh(hidden) * params.output_proj, axis=-1)

    return config.rate_min + (config.rate_max - config.rate_min) * expit(logits)


def memgrn_vector_field(
    raw_input: np.ndarray,
    z_state: np.ndarray,
    h_state: np.ndarray,
    y_state: np.ndarray,
    params: MemGRNParams,
    config: MemGRNConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate (dz/dt, dh/dt, dy/dt) for MemGRN."""
    activity = np.tanh(raw_input)
    rates = adaptive_rate_law(raw_input, activity, z_state, h_state, params, config)
    dz = rates * (activity[..., None] - z_state)
    dh = rates * (z_state - h_state)

    flat_h = h_state.reshape(*h_state.shape[:-2], config.n_sources * config.n_channels)
    flat_b = params.delayed_weights.reshape(
        config.n_sources * config.n_channels, config.n_targets
    )
    production = activity @ params.direct_weights + flat_h @ flat_b
    if (
        config.use_pairwise_interactions
        and params.interaction_weights is not None
        and config.n_sources >= 2
    ):
        mean_h = np.mean(h_state, axis=-1)
        idx_i, idx_k = np.triu_indices(config.n_sources, k=1)
        pair_feats = mean_h[..., idx_i] * mean_h[..., idx_k]
        production = production + pair_feats @ params.interaction_weights
    dy = production - params.relaxation * y_state
    return dz, dh, dy


def rollout_memgrn_rk4(
    stage_inputs: np.ndarray,
    dt: float,
    params: MemGRNParams,
    config: MemGRNConfig,
    y0: np.ndarray | None = None,
    z0: np.ndarray | None = None,
    h0: np.ndarray | None = None,
    reset_steps: set[int] | None = None,
    splice_step: int | None = None,
    splice_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate MemGRN with classical fourth-order Runge-Kutta (RK4).

    Args:
        stage_inputs: Input values at (start, midpoint, end) of each step, shape
            (n_steps, batch, n_sources, 3) or (n_steps, n_sources, 3).
        dt: Integration time step.
        params: MemGRN parameters.
        config: MemGRN configuration.
        y0: Optional initial target state.
        z0: Optional initial upstream filter state.
        h0: Optional initial delayed filter state.
        reset_steps: Optional step indices at which filter states (z, h) reset to zero
            while preserving target state y.
        splice_step: Optional step index at which target state y is replaced by splice_y.
        splice_y: Replacement target state used when splice_step is reached.

    Returns:
        Tuple (z_traj, h_traj, y_traj) including initial state at index 0, with length
        n_steps + 1 along axis 0.
    """
    unbatched = stage_inputs.ndim == 3
    if unbatched:
        stage_inputs = stage_inputs[:, None, :, :]
    n_steps, batch, _, _ = stage_inputs.shape

    z = (
        np.zeros((batch, config.n_sources, config.n_channels), dtype=np.float64)
        if z0 is None
        else np.broadcast_to(
            np.asarray(z0, dtype=np.float64),
            (batch, config.n_sources, config.n_channels),
        ).copy()
    )
    h = (
        np.zeros((batch, config.n_sources, config.n_channels), dtype=np.float64)
        if h0 is None
        else np.broadcast_to(
            np.asarray(h0, dtype=np.float64),
            (batch, config.n_sources, config.n_channels),
        ).copy()
    )
    y = (
        np.zeros((batch, config.n_targets), dtype=np.float64)
        if y0 is None
        else np.broadcast_to(
            np.asarray(y0, dtype=np.float64), (batch, config.n_targets)
        ).copy()
    )

    z_hist = [z.copy()]
    h_hist = [h.copy()]
    y_hist = [y.copy()]
    resets = reset_steps or set()

    for step in range(n_steps):
        if step in resets:
            z.fill(0.0)
            h.fill(0.0)
        if splice_step is not None and step == splice_step and splice_y is not None:
            y = np.broadcast_to(splice_y, y.shape).copy()

        u0 = stage_inputs[step, ..., 0]
        um = stage_inputs[step, ..., 1]
        u1 = stage_inputs[step, ..., 2]

        kz1, kh1, ky1 = memgrn_vector_field(u0, z, h, y, params, config)
        kz2, kh2, ky2 = memgrn_vector_field(
            um, z + 0.5 * dt * kz1, h + 0.5 * dt * kh1, y + 0.5 * dt * ky1, params, config
        )
        kz3, kh3, ky3 = memgrn_vector_field(
            um, z + 0.5 * dt * kz2, h + 0.5 * dt * kh2, y + 0.5 * dt * ky2, params, config
        )
        kz4, kh4, ky4 = memgrn_vector_field(
            u1, z + dt * kz3, h + dt * kh3, y + dt * ky3, params, config
        )

        z = z + (dt / 6.0) * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4)
        h = h + (dt / 6.0) * (kh1 + 2.0 * kh2 + 2.0 * kh3 + kh4)
        y = y + (dt / 6.0) * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4)

        z_hist.append(z.copy())
        h_hist.append(h.copy())
        y_hist.append(y.copy())

    z_out = np.stack(z_hist, axis=0)
    h_out = np.stack(h_hist, axis=0)
    y_out = np.stack(y_hist, axis=0)
    if unbatched:
        return z_out[:, 0], h_out[:, 0], y_out[:, 0]
    return z_out, h_out, y_out


def rollout_memgrn_midpoint(
    midpoint_inputs: np.ndarray,
    dt: float,
    params: MemGRNParams,
    config: MemGRNConfig,
    y0: np.ndarray | None = None,
    reset_steps: set[int] | None = None,
    splice_step: int | None = None,
    splice_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate MemGRN with explicit second-order midpoint steps."""
    unbatched = midpoint_inputs.ndim == 3
    if unbatched:
        midpoint_inputs = midpoint_inputs[:, None, :, :]
    n_steps, batch, _, _ = midpoint_inputs.shape

    z = np.zeros((batch, config.n_sources, config.n_channels), dtype=np.float64)
    h = np.zeros((batch, config.n_sources, config.n_channels), dtype=np.float64)
    y = (
        np.zeros((batch, config.n_targets), dtype=np.float64)
        if y0 is None
        else np.broadcast_to(
            np.asarray(y0, dtype=np.float64), (batch, config.n_targets)
        ).copy()
    )

    z_hist = [z.copy()]
    h_hist = [h.copy()]
    y_hist = [y.copy()]
    resets = reset_steps or set()

    for step in range(n_steps):
        if step in resets:
            z.fill(0.0)
            h.fill(0.0)
        if splice_step is not None and step == splice_step and splice_y is not None:
            y = np.broadcast_to(splice_y, y.shape).copy()

        u_start = midpoint_inputs[step, ..., 0]
        u_mid = midpoint_inputs[step, ..., 1]
        kz1, kh1, ky1 = memgrn_vector_field(u_start, z, h, y, params, config)
        kz2, kh2, ky2 = memgrn_vector_field(
            u_mid,
            z + 0.5 * dt * kz1,
            h + 0.5 * dt * kh1,
            y + 0.5 * dt * ky1,
            params,
            config,
        )
        z = z + dt * kz2
        h = h + dt * kh2
        y = y + dt * ky2
        z_hist.append(z.copy())
        h_hist.append(h.copy())
        y_hist.append(y.copy())

    z_out = np.stack(z_hist, axis=0)
    h_out = np.stack(h_hist, axis=0)
    y_out = np.stack(y_hist, axis=0)
    if unbatched:
        return z_out[:, 0], h_out[:, 0], y_out[:, 0]
    return z_out, h_out, y_out


def compute_exact_contributions(
    stage_inputs: np.ndarray,
    dt: float,
    params: MemGRNParams,
    config: MemGRNConfig,
    y0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Proposition 1 source-specific integrated contributions C_{ij}(t).

    Integrates each source's linear contribution ODE dC_{ij}/dt = p_{ij}(t) - delta_j C_{ij}
    alongside the joint trajectory so that y_j(t) = exp(-delta_j t) y_j(0) + sum_i C_{ij}(t).

    Returns:
        Tuple (y_traj, initial_decay_term, contributions) where contributions has shape
        (n_steps + 1, n_sources, n_targets).
    """
    if stage_inputs.ndim != 3:
        raise ValueError("stage_inputs must have shape (n_steps, n_sources, 3)")
    n_steps = stage_inputs.shape[0]
    y_init = (
        np.zeros(config.n_targets, dtype=np.float64)
        if y0 is None
        else np.asarray(y0, dtype=np.float64).copy()
    )

    z = np.zeros((config.n_sources, config.n_channels), dtype=np.float64)
    h = np.zeros((config.n_sources, config.n_channels), dtype=np.float64)
    y = y_init.copy()
    contrib = np.zeros((config.n_sources, config.n_targets), dtype=np.float64)

    def _augmented_rhs(
        u_val: np.ndarray,
        z_val: np.ndarray,
        h_val: np.ndarray,
        y_val: np.ndarray,
        c_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dz, dh, dy = memgrn_vector_field(u_val, z_val, h_val, y_val, params, config)
        act = np.tanh(u_val)
        per_source_prod = act[:, None] * params.direct_weights + np.einsum(
            "ir,irj->ij", h_val, params.delayed_weights
        )
        dc = per_source_prod - params.relaxation[None, :] * c_val
        return dz, dh, dy, dc

    y_hist = [y.copy()]
    c_hist = [contrib.copy()]
    for step in range(n_steps):
        u0, um, u1 = (
            stage_inputs[step, :, 0],
            stage_inputs[step, :, 1],
            stage_inputs[step, :, 2],
        )
        kz1, kh1, ky1, kc1 = _augmented_rhs(u0, z, h, y, contrib)
        kz2, kh2, ky2, kc2 = _augmented_rhs(
            um,
            z + 0.5 * dt * kz1,
            h + 0.5 * dt * kh1,
            y + 0.5 * dt * ky1,
            contrib + 0.5 * dt * kc1,
        )
        kz3, kh3, ky3, kc3 = _augmented_rhs(
            um,
            z + 0.5 * dt * kz2,
            h + 0.5 * dt * kh2,
            y + 0.5 * dt * ky2,
            contrib + 0.5 * dt * kc2,
        )
        kz4, kh4, ky4, kc4 = _augmented_rhs(
            u1, z + dt * kz3, h + dt * kh3, y + dt * ky3, contrib + dt * kc3
        )
        z = z + (dt / 6.0) * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4)
        h = h + (dt / 6.0) * (kh1 + 2.0 * kh2 + 2.0 * kh3 + kh4)
        y = y + (dt / 6.0) * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4)
        contrib = contrib + (dt / 6.0) * (kc1 + 2.0 * kc2 + 2.0 * kc3 + kc4)
        y_hist.append(y.copy())
        c_hist.append(contrib.copy())

    times = np.arange(n_steps + 1, dtype=np.float64) * dt
    decay_term = np.exp(-np.outer(times, params.relaxation)) * y_init[None, :]
    return np.stack(y_hist, axis=0), decay_term, np.stack(c_hist, axis=0)


def memory_reset_splice_decomposition(
    stage_inputs: np.ndarray,
    dt: float,
    params: MemGRNParams,
    config: MemGRNConfig,
    reset_steps: set[int],
    shutoff_step: int,
) -> dict[str, np.ndarray]:
    """Decompose intact-minus-reset recovery into shutoff carryover and post-shutoff production."""
    _, _, y_intact = rollout_memgrn_rk4(stage_inputs, dt, params, config)
    _, _, y_reset = rollout_memgrn_rk4(
        stage_inputs, dt, params, config, reset_steps=reset_steps
    )
    _, _, y_spliced = rollout_memgrn_rk4(
        stage_inputs,
        dt,
        params,
        config,
        reset_steps=reset_steps,
        splice_step=shutoff_step,
        splice_y=y_intact[shutoff_step],
    )
    carryover = y_spliced[shutoff_step:] - y_reset[shutoff_step:]
    post_production = y_intact[shutoff_step:] - y_spliced[shutoff_step:]
    return {
        "intact": y_intact,
        "reset": y_reset,
        "spliced": y_spliced,
        "carryover_after_shutoff": carryover,
        "post_shutoff_production": post_production,
    }


def rollout_latent_ode(
    midpoint_inputs: np.ndarray,
    dt: float,
    rate_logits: np.ndarray,
    w_state: np.ndarray,
    w_input: np.ndarray,
    bias: np.ndarray,
    w_output: np.ndarray,
    w_direct: np.ndarray,
    n_targets: int,
) -> np.ndarray:
    """Integrate the matched deterministic input-driven Latent Neural ODE baseline."""
    unbatched = midpoint_inputs.ndim == 3
    if unbatched:
        midpoint_inputs = midpoint_inputs[:, None, :, :]
    n_steps, batch, _, _ = midpoint_inputs.shape
    state_dim = rate_logits.shape[0]
    decay = 0.1 + np.logaddexp(0.0, rate_logits)
    rest_hidden = np.tanh(bias)

    def _field(s: np.ndarray, raw: np.ndarray) -> np.ndarray:
        act = np.tanh(raw)
        hidden = np.tanh(s @ w_state + act @ w_input + bias) - rest_hidden
        return -decay * s + hidden @ w_output + act @ w_direct

    s = np.zeros((batch, state_dim), dtype=np.float64)
    obs = [s[:, -n_targets:].copy()]
    for step in range(n_steps):
        u0 = midpoint_inputs[step, ..., 0]
        um = midpoint_inputs[step, ..., 1]
        s_mid = s + 0.5 * dt * _field(s, u0)
        s = s + dt * _field(s_mid, um)
        obs.append(s[:, -n_targets:].copy())
    out = np.stack(obs, axis=0)
    return out[:, 0] if unbatched else out
