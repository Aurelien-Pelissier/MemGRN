"""Out-of-class multi-TF correlation-shift benchmark with additive & cooperative regimes."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class MultiTFWorld:
    """Generated multi-TF biochemical system and locked train/val/test intervention splits."""

    seed: int
    regime: str
    n_sources: int
    n_targets: int
    dt: float
    obs_stride: int
    obs_times: np.ndarray
    parent_mask: np.ndarray
    train_stage_u: np.ndarray
    train_y: np.ndarray
    val_stage_u: np.ndarray
    val_y: np.ndarray
    test_stage_u: np.ndarray
    test_y: np.ndarray
    test_true_effects: np.ndarray
    test_categories: tuple[str, ...]


def _build_piecewise_stage_grid(
    knots: np.ndarray,
    substeps: int = 4,
) -> np.ndarray:
    """Convert knot inputs (batch, n_knots, n_sources) into RK4 stage grid (n_steps, batch, S, 3)."""
    lo = knots[:, :-1, :, None]
    delta = np.diff(knots, axis=1)[:, :, :, None]
    frac = np.arange(substeps, dtype=np.float64) / substeps

    def _at(offset: float) -> np.ndarray:
        vals = lo + delta * (frac + offset / substeps)
        # (batch, n_intervals, S, substeps) -> (batch, n_intervals * substeps, S)
        trans = np.transpose(vals, (0, 1, 3, 2)).reshape(knots.shape[0], -1, knots.shape[2])
        return np.transpose(trans, (1, 0, 2))

    return np.stack([_at(0.0), _at(0.5), _at(1.0)], axis=-1)


def generate_multitf_world(
    seed: int,
    regime: str = "additive",
    n_sources: int = 8,
    n_targets: int = 24,
    obs_noise_sd: float = 0.015,
) -> MultiTFWorld:
    """Generate an independent out-of-class 8-TF, 24-target network with correlation shift.

    Pathway structure during training/validation:
      - Module 0: TFs {0, 1, 2} co-activated with correlation ~0.88
      - Module 1: TFs {3, 4, 5} co-activated with correlation ~0.88
      - Module 2: TFs {6, 7}    co-activated with correlation ~0.88
    Each target has 2 true causal parents (often drawn from distinct or shared modules).
    Test splits break training correlations via:
      1. single_ko_in_module (8 schedules): stimulate a module with 1 member knocked out
      2. phase_lag_shift     (8 schedules): asynchronous relative onset delays between TFs
      3. cross_module_combo  (8 schedules): novel 2-TF cross-module asymmetric pulses
    """
    if regime not in ("additive", "cooperative"):
        raise ValueError(f"Unknown regime {regime}")
    rng = np.random.default_rng(seed)

    # Planted sparse causal wiring: each target has 2 causal parents
    parent_mask = np.zeros((n_sources, n_targets), dtype=bool)
    parents_list: list[tuple[int, int]] = []
    weights_direct = np.zeros((n_sources, n_targets), dtype=np.float64)
    weights_fast = np.zeros((n_sources, n_targets), dtype=np.float64)
    weights_slow = np.zeros((n_sources, n_targets), dtype=np.float64)
    coop_gain = np.zeros((n_targets,), dtype=np.float64)

    for j in range(n_targets):
        # Pick 1 parent from Module (j % 3) and 1 parent from any other TF so correlated
        # co-regulators create genuine confounding for source-mixing models!
        mod_id = j % 3
        mod_members = [[0, 1, 2], [3, 4, 5], [6, 7]][mod_id]
        p1 = int(rng.choice(mod_members))
        rest = [i for i in range(n_sources) if i != p1]
        p2 = int(rng.choice(rest))
        pair = (min(p1, p2), max(p1, p2))
        parents_list.append(pair)
        parent_mask[pair[0], j] = True
        parent_mask[pair[1], j] = True

        for p_idx in pair:
            sgn = float(rng.choice([-1.0, 1.0]))
            mag = float(rng.uniform(0.65, 1.15))
            kind = int(rng.integers(0, 3))
            weights_direct[p_idx, j] = 0.08 * sgn * mag
            if kind == 0:
                weights_fast[p_idx, j] = sgn * mag
            elif kind == 1:
                weights_slow[p_idx, j] = sgn * mag
            else:
                weights_fast[p_idx, j] = sgn * mag
                weights_slow[p_idx, j] = -0.85 * sgn * mag

        if regime == "cooperative" and (j % 2 == 0):
            coop_gain[j] = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.95, 1.45))

    # Out-of-class source biochemical kinetics: occupancy-gated chromatin + finite-pool buffer
    base_fast = rng.uniform(2.2, 3.8, size=(n_sources,))
    base_slow = rng.uniform(0.35, 0.75, size=(n_sources,))
    k_on = rng.uniform(1.8, 3.2, size=(n_sources,))
    k_off = rng.uniform(0.25, 0.55, size=(n_sources,))
    occ_boost = rng.uniform(1.8, 3.0, size=(n_sources,))
    buf_cap = rng.uniform(0.50, 0.90, size=(n_sources,))
    buf_bind = rng.uniform(2.5, 4.5, size=(n_sources,))
    buf_rel = rng.uniform(0.20, 0.45, size=(n_sources,))
    target_decay = rng.uniform(0.40, 0.85, size=(n_targets,))

    n_knots = 25  # 0.0 to 6.0 hours every 0.25 h
    knot_dt = 0.25
    substeps = 4
    dt = knot_dt / substeps
    obs_times = np.arange(n_knots, dtype=np.float64) * knot_dt
    modules = ([0, 1, 2], [3, 4, 5], [6, 7])

    def _smooth_pulse(t_arr: np.ndarray, t_on: float, t_off: float, amp: float) -> np.ndarray:
        rise = 1.0 / (1.0 + np.exp(-(t_arr - t_on) / 0.08))
        fall = 1.0 / (1.0 + np.exp((t_arr - t_off) / 0.08))
        return amp * rise * fall

    def _make_correlated_schedules(n_sched: int, split_rng: np.random.Generator) -> np.ndarray:
        knots = np.zeros((n_sched, n_knots, n_sources), dtype=np.float64)
        for s_i in range(n_sched):
            active_mods = split_rng.choice(3, size=int(split_rng.choice([1, 2])), replace=False)
            for m_id in active_mods:
                dur = float(split_rng.choice([0.75, 1.25, 2.0, 2.75]))
                t_on = float(split_rng.choice([0.0, 0.25, 0.5]))
                base_amp = float(split_rng.uniform(0.45, 1.45))
                # Optional second pulse on 35% of training schedules
                two_pulse = bool(split_rng.random() < 0.35)
                for tf in modules[int(m_id)]:
                    jitter = float(split_rng.normal(1.0, 0.12))
                    amp_tf = max(0.1, base_amp * jitter)
                    trace = _smooth_pulse(obs_times, t_on, t_on + dur, amp_tf)
                    if two_pulse:
                        trace = trace + _smooth_pulse(
                            obs_times, t_on + dur + 0.5, t_on + 2 * dur + 0.5, 0.85 * amp_tf
                        )
                    knots[s_i, :, tf] = trace
        return knots

    train_knots = _make_correlated_schedules(28, np.random.default_rng([seed, 11]))
    val_knots = _make_correlated_schedules(8, np.random.default_rng([seed, 22]))

    # Locked test interventions breaking training TF correlations
    rng_test = np.random.default_rng([seed, 33])
    test_knots_list: list[np.ndarray] = []
    test_cats: list[str] = []

    # Category 1: Single-TF knockout inside an activated pathway module (8 schedules)
    for k_i in range(8):
        row = np.zeros((n_knots, n_sources), dtype=np.float64)
        m_id = k_i % 3
        dur = float(rng_test.choice([1.0, 1.75, 2.5]))
        amp = float(rng_test.uniform(0.75, 1.35))
        ko_tf = modules[m_id][k_i % len(modules[m_id])]
        for tf in modules[m_id]:
            if tf != ko_tf:
                row[:, tf] = _smooth_pulse(obs_times, 0.25, 0.25 + dur, amp)
        test_knots_list.append(row)
        test_cats.append("single_ko_in_module")

    # Category 2: Relative pulse-timing / phase-lag shift between co-regulators (8 schedules)
    for k_i in range(8):
        row = np.zeros((n_knots, n_sources), dtype=np.float64)
        p1, p2 = parents_list[k_i % n_targets]
        lag = float(rng_test.choice([0.75, 1.5, 2.25]))
        dur = float(rng_test.choice([1.0, 1.5]))
        amp1 = float(rng_test.uniform(0.7, 1.3))
        amp2 = float(rng_test.uniform(0.7, 1.3))
        row[:, p1] = _smooth_pulse(obs_times, 0.25, 0.25 + dur, amp1)
        row[:, p2] = _smooth_pulse(obs_times, 0.25 + lag, 0.25 + lag + dur, amp2)
        test_knots_list.append(row)
        test_cats.append("phase_lag_shift")

    # Category 3: Cross-module asymmetric 3:1 combinations (8 schedules)
    for k_i in range(8):
        row = np.zeros((n_knots, n_sources), dtype=np.float64)
        m_a = k_i % 3
        m_b = (k_i + 1) % 3
        tf_a = int(rng_test.choice(modules[m_a]))
        tf_b = int(rng_test.choice(modules[m_b]))
        dur = float(rng_test.choice([1.25, 2.25]))
        strong = float(rng_test.uniform(0.9, 1.4))
        row[:, tf_a] = _smooth_pulse(obs_times, 0.25, 0.25 + dur, strong)
        row[:, tf_b] = _smooth_pulse(obs_times, 0.25, 0.25 + dur, strong / 3.0)
        test_knots_list.append(row)
        test_cats.append("cross_module_combo")

    test_knots = np.stack(test_knots_list, axis=0)

    def _simulate_exact(stage_u: np.ndarray) -> np.ndarray:
        n_steps, batch, _, _ = stage_u.shape
        zf = np.zeros((batch, n_sources), dtype=np.float64)
        hf = np.zeros((batch, n_sources), dtype=np.float64)
        zs = np.zeros((batch, n_sources), dtype=np.float64)
        hs = np.zeros((batch, n_sources), dtype=np.float64)
        occ = np.zeros((batch, n_sources), dtype=np.float64)
        buf = np.zeros((batch, n_sources), dtype=np.float64)
        y = np.zeros((batch, n_targets), dtype=np.float64)

        def _rhs(s_tup, raw_u):
            zf_c, hf_c, zs_c, hs_c, occ_c, buf_c, y_c = s_tup
            act = np.tanh(raw_u)
            pos_a = np.abs(act)
            d_occ = k_on * pos_a * (1.0 - occ_c) - k_off * occ_c
            # Occupancy slows post-washout chromatin closing while accelerating initial opening
            rf = base_fast * (0.45 + 0.85 * pos_a) / (1.0 + occ_boost * occ_c)
            rs = base_slow * (0.40 + 0.90 * occ_c)
            flux = buf_bind * zs_c * (buf_cap - buf_c) - buf_rel * buf_c
            dzf = rf * (act - zf_c)
            dhf = rf * (zf_c - hf_c)
            dzs = rs * (act - zs_c) - flux
            dhs = rs * (zs_c - hs_c)

            prod = act @ weights_direct + hf_c @ weights_fast + hs_c @ weights_slow
            if regime == "cooperative":
                eff_signal = 0.5 * (hf_c + hs_c)
                coop_terms = np.zeros_like(prod)
                for j_idx, (pa, pb) in enumerate(parents_list):
                    if coop_gain[j_idx] != 0.0:
                        ca = eff_signal[:, pa]
                        cb = eff_signal[:, pb]
                        denom = 1.0 + np.abs(ca) + np.abs(cb) + 1.5 * np.abs(ca * cb)
                        coop_terms[:, j_idx] = coop_gain[j_idx] * (ca * cb) / denom
                prod = prod + coop_terms
            dy = prod - target_decay * y_c
            return dzf, dhf, dzs, dhs, d_occ, flux, dy

        obs = [y.copy()]
        for step in range(n_steps):
            u0, um, u1 = stage_u[step, ..., 0], stage_u[step, ..., 1], stage_u[step, ..., 2]
            cur = (zf, hf, zs, hs, occ, buf, y)
            k1 = _rhs(cur, u0)
            k2 = _rhs(tuple(c + 0.5 * dt * d for c, d in zip(cur, k1)), um)
            k3 = _rhs(tuple(c + 0.5 * dt * d for c, d in zip(cur, k2)), um)
            k4 = _rhs(tuple(c + dt * d for c, d in zip(cur, k3)), u1)
            zf, hf, zs, hs, occ, buf, y = tuple(
                c + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
                for c, d1, d2, d3, d4 in zip(cur, k1, k2, k3, k4)
            )
            if (step + 1) % substeps == 0:
                obs.append(y.copy())
        return np.transpose(np.stack(obs, axis=0), (1, 0, 2))

    train_stage = _build_piecewise_stage_grid(train_knots, substeps=substeps)
    val_stage = _build_piecewise_stage_grid(val_knots, substeps=substeps)
    test_stage = _build_piecewise_stage_grid(test_knots, substeps=substeps)

    train_clean = _simulate_exact(train_stage)
    val_clean = _simulate_exact(val_stage)
    test_clean = _simulate_exact(test_stage)

    # Training-only RMS target normalization (strictly leakage-free)
    scales = np.maximum(0.1, np.sqrt(np.mean(train_clean**2, axis=(0, 1))))
    rng_noise = np.random.default_rng([seed, 99])
    train_y = (train_clean / scales) + rng_noise.normal(0.0, obs_noise_sd, size=train_clean.shape)
    train_y[:, 0, :] = 0.0
    val_y = (val_clean / scales) + rng_noise.normal(0.0, obs_noise_sd, size=val_clean.shape)
    val_y[:, 0, :] = 0.0
    test_y = test_clean / scales

    # Ground-truth counterfactual single-source removal effects on test interventions
    true_effects = []
    for src_i in range(n_sources):
        abl = test_stage.copy()
        abl[:, :, src_i, :] = 0.0
        true_effects.append((test_clean - _simulate_exact(abl)) / scales)
    test_effects = np.stack(true_effects, axis=0)

    return MultiTFWorld(
        seed=seed,
        regime=regime,
        n_sources=n_sources,
        n_targets=n_targets,
        dt=dt,
        obs_stride=substeps,
        obs_times=obs_times,
        parent_mask=parent_mask,
        train_stage_u=train_stage,
        train_y=train_y,
        val_stage_u=val_stage,
        val_y=val_y,
        test_stage_u=test_stage,
        test_y=test_y,
        test_true_effects=test_effects,
        test_categories=tuple(test_cats),
    )
