"""Controlled synthetic dynamical network generators for MemGRN benchmarks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class SyntheticNetwork:
    """Synthetic six-node network with planted two-parent regulatory wiring."""

    seed: int
    parents: list[tuple[int, int]]
    rate: np.ndarray
    mix: np.ndarray
    direct: np.ndarray
    decay: np.ndarray
    kon: np.ndarray
    koff: np.ndarray
    boost: np.ndarray
    buffer_capacity: np.ndarray | None = None
    buffer_binding: np.ndarray | None = None
    buffer_release: np.ndarray | None = None


def build_synthetic_network(seed: int, with_buffer: bool = False) -> SyntheticNetwork:
    """Construct deterministic six-source, six-target network with fast/slow/biphasic channels."""
    rng_topo = np.random.default_rng([seed, 409])
    parents: list[tuple[int, int]] = []
    kinds_list: list[tuple[int, int]] = []
    weights_list: list[tuple[float, float]] = []

    kinetic_pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    rng_topo.shuffle(kinetic_pairs)
    opposing_flags = [True, True, True, False, False, False]
    rng_topo.shuffle(opposing_flags)

    for j in range(6):
        eligible = [i for i in range(6) if i != j]
        chosen = tuple(sorted(rng_topo.choice(eligible, size=2, replace=False).tolist()))
        parents.append((chosen[0], chosen[1]))
        kinds_list.append(kinetic_pairs[j])
        mag1, mag2 = rng_topo.uniform(0.55, 0.95, size=2)
        sign1 = float(rng_topo.choice([-1.0, 1.0]))
        sign2 = -sign1 if opposing_flags[j] else sign1
        weights_list.append((sign1 * float(mag1), sign2 * float(mag2)))

    decay = rng_topo.uniform(0.35, 0.75, size=6)

    rng_kin = np.random.default_rng([seed, 813])
    rates = rng_kin.uniform([0.55, 1.35, 3.0], [0.85, 1.85, 4.0], size=(6, 3))
    mix = np.zeros((6, 3, 6), dtype=np.float64)
    direct = np.zeros((6, 6), dtype=np.float64)
    for j, (pair, kpair, wpair) in enumerate(zip(parents, kinds_list, weights_list)):
        for src, kind, weight in zip(pair, kpair, wpair):
            if kind == 0:
                mix[src, 2, j] = weight
            elif kind == 1:
                mix[src, 0, j] = weight
            else:
                mix[src, 2, j] = weight
                mix[src, 0, j] = -1.25 * weight
            direct[src, j] = rng_kin.uniform(0.05, 0.12) * weight

    cap = bind = rel = None
    if with_buffer:
        rng_buf = np.random.default_rng([seed, 655317])
        cap = rng_buf.uniform(0.45, 1.0, size=(6, 3))
        bind = rng_buf.uniform(2.5, 5.5, size=(6, 3))
        rel = rng_buf.uniform(0.22, 0.6, size=(6, 3))

    return SyntheticNetwork(
        seed=seed,
        parents=parents,
        rate=rates,
        mix=mix.reshape(18, 6),
        direct=direct,
        decay=decay,
        kon=rng_kin.uniform(1.5, 3.0, size=6),
        koff=rng_kin.uniform(0.4, 0.9, size=6),
        boost=rng_kin.uniform(2.0, 3.5, size=6),
        buffer_capacity=cap,
        buffer_binding=bind,
        buffer_release=rel,
    )


def build_pulse_inputs(
    rows: np.ndarray,
    n_steps: int = 48,
    n_sources: int = 6,
) -> np.ndarray:
    """Build piecewise-constant dual-source pulse inputs from protocol specification rows.

    Each row specifies (duration_steps, strong_amp, strength_ratio, source_weak, source_strong).
    """
    duration, amp, ratio, src_i, src_k = rows.T
    source = np.zeros((len(rows), n_sources), dtype=np.float64)
    idx = np.arange(len(rows))
    source[idx, src_i.astype(int)] = amp / ratio
    source[idx, src_k.astype(int)] = amp
    active = np.arange(n_steps)[None, :, None] < duration[:, None, None]
    return active * source[:, None, :]


def simulate_occupancy_network(
    net: SyntheticNetwork,
    rows: np.ndarray,
    asymmetric: bool = True,
    dt: float = 0.025,
    step_interval: float = 0.2,
) -> np.ndarray:
    """Integrate constant or occupancy-gated mediator kinetics with RK4."""
    raw = build_pulse_inputs(rows)
    n_batch, n_steps, _ = raw.shape
    substeps = round(step_interval / dt)
    state = np.zeros((n_batch, 48), dtype=np.float64)
    outputs = []

    def _field(s: np.ndarray, a: np.ndarray) -> np.ndarray:
        z = s[:, :18].reshape(-1, 6, 3)
        h = s[:, 18:36].reshape(-1, 6, 3)
        b = s[:, 36:42]
        rate = net.rate[None, :, :]
        if asymmetric:
            scale = 0.42 + net.boost[None, :] * b + 0.65 * a
            rate = rate * scale[:, :, None]
        dz = rate * (a[:, :, None] - z)
        dh = rate * (z - h)
        db = net.kon[None, :] * a * (1.0 - b) - net.koff[None, :] * b
        dy = -net.decay[None, :] * s[:, 42:] + a @ net.direct + s[:, 18:36] @ net.mix
        return np.column_stack([dz.reshape(n_batch, 18), dh.reshape(n_batch, 18), db, dy])

    for step in range(n_steps):
        act = np.tanh(raw[:, step])
        for _ in range(substeps):
            k1 = _field(state, act)
            k2 = _field(state + 0.5 * dt * k1, act)
            k3 = _field(state + 0.5 * dt * k2, act)
            k4 = _field(state + dt * k3, act)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        outputs.append(state[:, 42:].copy())
    return np.stack(outputs, axis=1)


def simulate_buffered_network(
    net: SyntheticNetwork,
    rows: np.ndarray,
    dt: float = 0.025,
    step_interval: float = 0.2,
) -> np.ndarray:
    """Integrate out-of-class reversible mass-action mediator buffering with RK4."""
    if (
        net.buffer_capacity is None
        or net.buffer_binding is None
        or net.buffer_release is None
    ):
        raise ValueError("SyntheticNetwork must be initialized with with_buffer=True")

    raw = build_pulse_inputs(rows)
    n_batch, n_steps, _ = raw.shape
    substeps = round(step_interval / dt)
    state = np.zeros((n_batch, 60), dtype=np.float64)
    outputs = []

    def _field(s: np.ndarray, a: np.ndarray) -> np.ndarray:
        z = s[:, :18].reshape(-1, 6, 3)
        b = s[:, 18:36].reshape(-1, 6, 3)
        h = s[:, 36:54].reshape(-1, 6, 3)
        flux = net.buffer_binding * z * (net.buffer_capacity - b) - net.buffer_release * b
        dz = net.rate * (a[:, :, None] - z) - flux
        dh = net.rate * (z - h)
        dy = -net.decay * s[:, 54:] + a @ net.direct + s[:, 36:54] @ net.mix
        return np.column_stack(
            [
                dz.reshape(n_batch, 18),
                flux.reshape(n_batch, 18),
                dh.reshape(n_batch, 18),
                dy,
            ]
        )

    for step in range(n_steps):
        act = np.tanh(raw[:, step])
        for _ in range(substeps):
            k1 = _field(state, act)
            k2 = _field(state + 0.5 * dt * k1, act)
            k3 = _field(state + 0.5 * dt * k2, act)
            k4 = _field(state + dt * k3, act)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        outputs.append(state[:, 54:].copy())
    return np.stack(outputs, axis=1)
