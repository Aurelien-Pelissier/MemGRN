"""IEEE 39-Bus New England Power Grid closed-loop N-1 -> N-2 combinatorial contingency simulator.

Implements the published IEEE 39-Bus New England Test System (Athay et al., IEEE Trans. PAS 1979;
Pai, Energy Function Analysis for Power System Stability, 1989): 39 buses, 10 synchronous
generators (buses 30..39), and 46 transmission lines with exact per-unit reactances X_l.

Applies exact Kron reduction to obtain the physical 10 x 10 generator electrical coupling
admittance matrix B_Kron, then simulates out-of-class nonlinear closed-loop power-grid
frequency/stress dynamics with saturable asymmetric turbine-governor / thermal-limiter hysteresis:
  - Fast overload stress accumulation during line/generator contingencies (lambda_up fast)
  - Slow nonlinear spinning-reserve & thermal relaxation (lambda_down slow, state-dependent)
  - Pairwise nonlinear power-transfer rerouting coupling across connected transmission corridors

Training supplies ONLY normal load disturbances and single N-1 generator contingencies over
[0, 5] s; test evaluates zero-shot on:
  1. Unseen single N-1 contingencies;
  2. Zero-shot combinatorial N-2 double contingencies (ko_i + ko_k, never seen in training);
  3. Out-of-horizon [5, 9] s post-contingency recovery extrapolation;
  4. Physical Kron-reduced electrical corridor graph recovery (AUROC & AP).
"""

from dataclasses import dataclass

import numpy as np

# Standard IEEE 39-Bus New England Test System: 46 transmission lines & transformers
# Each tuple is (from_bus, to_bus, reactance_pu) with 1-based bus numbering (1..39).
# Generators sit at buses 30..39 (10 synchronous machines).
IEEE39_BRANCHES: tuple[tuple[int, int, float], ...] = (
    (1, 2, 0.0411),
    (1, 39, 0.0250),
    (2, 3, 0.0151),
    (2, 25, 0.0086),
    (3, 4, 0.0213),
    (3, 18, 0.0133),
    (4, 5, 0.0128),
    (4, 14, 0.0129),
    (5, 6, 0.0026),
    (5, 8, 0.0112),
    (6, 7, 0.0092),
    (6, 11, 0.0082),
    (7, 8, 0.0046),
    (8, 9, 0.0363),
    (9, 39, 0.0250),
    (10, 11, 0.0043),
    (10, 13, 0.0043),
    (13, 14, 0.0101),
    (14, 15, 0.0217),
    (15, 16, 0.0094),
    (16, 17, 0.0089),
    (16, 19, 0.0195),
    (16, 21, 0.0135),
    (16, 24, 0.0059),
    (17, 18, 0.0082),
    (17, 27, 0.0173),
    (21, 22, 0.0140),
    (22, 23, 0.0096),
    (23, 24, 0.0350),
    (25, 26, 0.0323),
    (26, 27, 0.0147),
    (26, 28, 0.0474),
    (26, 29, 0.0625),
    (28, 29, 0.0151),
    (12, 11, 0.0435),
    (12, 13, 0.0435),
    (6, 31, 0.0250),
    (10, 32, 0.0200),
    (19, 33, 0.0142),
    (20, 34, 0.0180),
    (22, 35, 0.0143),
    (23, 36, 0.0272),
    (25, 37, 0.0232),
    (2, 30, 0.0181),
    (29, 38, 0.0156),
    (19, 20, 0.0138),
)

# Published generator inertia constants H_i (seconds) for generators G1..G10 (buses 30..39)
IEEE39_GEN_INERTIAS: tuple[float, ...] = (
    42.0,
    30.3,
    35.8,
    28.6,
    26.0,
    34.8,
    26.4,
    24.3,
    34.5,
    500.0,  # Bus 39 represents aggregated interconnection to rest of US/Canada grid
)

IEEE39_GEN_NAMES: tuple[str, ...] = tuple(f"G{i+1}_Bus{30+i}" for i in range(10))


def compute_ieee39_kron_admittance() -> tuple[np.ndarray, np.ndarray]:
    """Compute exact Kron-reduced 10x10 generator coupling matrix from IEEE 39-bus topology.

    Returns ``(B_norm, adj_binary)`` where ``B_norm`` has zero diagonal and positive off-diagonal
    normalized electrical susceptances, and ``adj_binary`` marks the strong physical transmission
    corridors (top 35% of Kron-reduced electrical susceptances).
    """
    n_bus = 39
    lap = np.zeros((n_bus, n_bus), dtype=np.float64)
    for u, v, x_pu in IEEE39_BRANCHES:
        b = 1.0 / x_pu
        ui, vi = u - 1, v - 1
        lap[ui, ui] += b
        lap[vi, vi] += b
        lap[ui, vi] -= b
        lap[vi, ui] -= b

    # Partition into 10 generator buses (29..38 in 0-based) and 29 interior load buses (0..28)
    g_idx = np.arange(29, 39)
    l_idx = np.arange(0, 29)
    b_gg = lap[np.ix_(g_idx, g_idx)]
    b_gl = lap[np.ix_(g_idx, l_idx)]
    b_ll = lap[np.ix_(l_idx, l_idx)]
    # Exact Kron reduction: B_red = B_gg - B_gl @ B_ll^{-1} @ B_gl^T
    b_red = b_gg - b_gl @ np.linalg.solve(b_ll, b_gl.T)
    coupling = np.maximum(0.0, -b_red)
    np.fill_diagonal(coupling, 0.0)
    # Normalize so median positive corridor weight ~ 1.0
    b_norm = coupling / max(1e-8, float(np.max(coupling)))
    off_vals = b_norm[~np.eye(10, dtype=bool)]
    thresh = float(np.percentile(off_vals, 65))
    adj_bin = b_norm >= thresh
    np.fill_diagonal(adj_bin, False)
    return b_norm, adj_bin


@dataclass(frozen=True, kw_only=True)
class PowerGridWorld:
    """One IEEE 39-bus operating-point world with single N-1 train and zero-shot N-2 test."""

    world_seed: int
    n_nodes: int  # 10 synchronous generators
    adj_binary: np.ndarray  # (10, 10) physical Kron electrical corridors
    coupling_weights: np.ndarray  # (10, 10)
    basal_load: np.ndarray  # (10,)
    dt: float
    train_ko: np.ndarray
    train_u_ext: np.ndarray
    train_y: np.ndarray
    train_weights: np.ndarray
    val_ko: np.ndarray
    val_u_ext: np.ndarray
    val_y: np.ndarray
    val_weights: np.ndarray
    test_single_ko: np.ndarray
    test_single_u_ext: np.ndarray
    test_single_y: np.ndarray
    test_double_ko: np.ndarray
    test_double_u_ext: np.ndarray
    test_double_y: np.ndarray
    test_double_y_extrap: np.ndarray


def generate_ieee39_powergrid_world(seed: int) -> PowerGridWorld:
    """Generate one out-of-class nonlinear IEEE 39-bus N-1 -> N-2 contingency world."""
    rng = np.random.default_rng(seed)
    b_norm, adj_bin = compute_ieee39_kron_admittance()
    n_g = 10

    # Operating-point dispatch heterogeneity across the 12 worlds
    inertias = np.array(IEEE39_GEN_INERTIAS, dtype=np.float64)
    inv_m = 1.0 / np.log1p(inertias)  # normalized inverse inertia (Bus 39 stiffest)
    inv_m = inv_m / np.mean(inv_m)

    # Signed physical rerouting/oscillatory couplings strictly supported on true IEEE39 corridors
    # plus weak global electrical admittance tail
    w_direct = (0.35 + 0.25 * rng.random((n_g, n_g))) * b_norm * adj_bin
    w_delayed = (1.15 + 0.55 * rng.random((n_g, n_g))) * b_norm * adj_bin
    # Secondary negative damping/governor counter-reaction on a subset of strong corridors
    signs = np.where(rng.random((n_g, n_g)) < 0.72, 1.0, -0.85)
    w_direct = w_direct * signs
    w_delayed = w_delayed * signs
    np.fill_diagonal(w_direct, 0.0)
    np.fill_diagonal(w_delayed, 0.0)

    # Pairwise nonlinear rerouting stress when TWO corridors simultaneously saturate (N-2 epistasis!)
    coop_triples: list[tuple[int, int, int, float]] = []
    for tgt in range(n_g):
        nbrs = np.where(adj_bin[:, tgt])[0]
        if len(nbrs) >= 2:
            for a_i in range(len(nbrs)):
                for b_i in range(a_i + 1, min(len(nbrs), a_i + 3)):
                    # Simultaneous stress on two adjacent generators amplifies overload nonlinearly
                    coef = float(rng.uniform(0.65, 1.25))
                    coop_triples.append((int(nbrs[a_i]), int(nbrs[b_i]), tgt, coef))

    # Out-of-class saturable asymmetric governor / thermal hysteresis per generator:
    # Fast overload buildup (k_on) vs slow nonlinear valve-limited recovery (k_off(c))
    k_on = rng.uniform(2.8, 4.8, size=n_g)
    k_off_fast = rng.uniform(0.35, 0.65, size=n_g)
    k_off_locked = rng.uniform(0.08, 0.18, size=n_g)  # thermal/valve saturation lock-in!
    c_half = rng.uniform(0.28, 0.45, size=n_g)
    damping = rng.uniform(0.55, 0.95, size=n_g) * inv_m
    basal = rng.uniform(0.55, 0.95, size=n_g)

    dt = 0.05  # 0.05 s substeps -> 100 steps = 5.0 s train, 180 steps = 9.0 s extrap
    obs_train = tuple(range(0, 101, 5))  # 21 observations on [0, 5] s
    obs_extrap = tuple(range(0, 181, 5))  # 37 observations on [0, 9] s

    def _simulate(
        ko_vec: np.ndarray, u_ext_traj: np.ndarray, n_steps: int, obs_idx: tuple[int, ...]
    ) -> np.ndarray:
        """Simulate out-of-class saturable nonlinear swing + governor hysteresis ODE via RK4."""
        y = np.zeros(n_g, dtype=np.float64)  # frequency/stress deviation
        q = np.zeros(n_g, dtype=np.float64)  # fast turbine stress buffer
        c = np.zeros(n_g, dtype=np.float64)  # slow thermal / governor-saturation memory

        def _rhs(state: tuple[np.ndarray, np.ndarray, np.ndarray], u_t: np.ndarray):
            ys, qs, cs = state
            # Generator drive: tripped generator (ko_i=1) injects negative power step -basal_i
            drv = (1.0 - ko_vec) * np.tanh(1.1 * ys) - ko_vec * basal + u_t
            act = np.tanh(drv)
            # Out-of-class Michaelis-Menten / Hill saturable asymmetric clearance
            load_rate = k_on * (0.4 + 0.6 * np.abs(act))
            clear_rate = k_off_locked + (k_off_fast - k_off_locked) / (
                1.0 + (np.abs(cs) / c_half) ** 2
            )
            dq = load_rate * (act - qs)
            dc = clear_rate * (qs - cs)
            # Direct electrical swing + delayed governor rerouting + nonlinear N-2 overload
            prod = act @ w_direct + np.sin(0.95 * cs) @ w_delayed
            for i_s, k_s, j_t, cf in coop_triples:
                prod[j_t] += cf * np.tanh(1.4 * cs[i_s] * cs[k_s])
            dy = inv_m * prod - damping * ys + 0.55 * u_t
            # Tripped generator stays isolated at its trip deficit
            dy = dy * (1.0 - ko_vec) - 2.0 * ko_vec * (ys + basal)
            return dy, dq, dc

        traj = np.zeros((n_steps + 1, n_g), dtype=np.float64)
        traj[0] = y
        for step in range(n_steps):
            u0 = u_ext_traj[step]
            u1 = u_ext_traj[min(step + 1, n_steps - 1)]
            um = 0.5 * (u0 + u1)
            k1 = _rhs((y, q, c), u0)
            k2 = _rhs((y + 0.5 * dt * k1[0], q + 0.5 * dt * k1[1], c + 0.5 * dt * k1[2]), um)
            k3 = _rhs((y + 0.5 * dt * k2[0], q + 0.5 * dt * k2[1], c + 0.5 * dt * k2[2]), um)
            k4 = _rhs((y + dt * k3[0], q + dt * k3[1], c + dt * k3[2]), u1)
            y = y + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            q = q + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            c = c + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            traj[step + 1] = y
        return traj[list(obs_idx)]

    def _make_pulse(n_steps: int, amp: float, t0: float, dur: float, nodes: list[int]) -> np.ndarray:
        u = np.zeros((n_steps, n_g), dtype=np.float64)
        t_grid = np.arange(n_steps) * dt
        mask = (t_grid >= t0) & (t_grid <= t0 + dur)
        for nd in nodes:
            u[mask, nd] += amp
        return u

    # Split 10 generators into 6 train N-1 contingencies, 2 val N-1, 2 unseen test N-1
    perm = rng.permutation(n_g).tolist()
    tr_gens, va_gens, te_gens = perm[:6], perm[6:8], perm[8:10]

    # Training: WT + 6 full N-1 generator trips (ko=1.0) + 6 partial unit deratings (ko=0.55)
    # ZERO double contingencies (N-2) in training!
    tr_ko_l, tr_u_l, tr_y_l, tr_w_l = [], [], [], []
    u_zero = np.zeros((100, n_g), dtype=np.float64)
    # Baseline normal operation
    ko_wt = np.zeros(n_g)
    tr_ko_l.append(ko_wt)
    tr_u_l.append(u_zero)
    y_wt = _simulate(ko_wt, u_zero, 100, obs_train) + rng.normal(0.0, 0.012, size=(len(obs_train), n_g))
    y_wt[0] = 0.0
    tr_y_l.append(y_wt)
    w_wt = np.ones((len(obs_train), n_g))
    w_wt[0] = 0.0
    tr_w_l.append(w_wt)

    for g in tr_gens:
        for strength in (1.0, 0.55):
            ko = np.zeros(n_g)
            ko[g] = strength
            y = _simulate(ko, u_zero, 100, obs_train) + rng.normal(0.0, 0.012, size=(len(obs_train), n_g))
            y[0] = 0.0
            w = np.ones((len(obs_train), n_g))
            w[0] = 0.0
            w[:, g] = 0.0
            tr_ko_l.append(ko)
            tr_u_l.append(u_zero)
            tr_y_l.append(y)
            tr_w_l.append(w)

    # Validation: 2 held-out single N-1 trips + 2 partial deratings
    va_ko_l, va_u_l, va_y_l, va_w_l = [], [], [], []
    for g in va_gens:
        for strength in (1.0, 0.55):
            ko = np.zeros(n_g)
            ko[g] = strength
            y = _simulate(ko, u_zero, 100, obs_train) + rng.normal(0.0, 0.012, size=(len(obs_train), n_g))
            y[0] = 0.0
            w = np.ones((len(obs_train), n_g))
            w[0] = 0.0
            w[:, g] = 0.0
            va_ko_l.append(ko)
            va_u_l.append(u_zero)
            va_y_l.append(y)
            va_w_l.append(w)

    # Test 1: Unseen single N-1 contingencies
    ts_ko_l, ts_u_l, ts_y_l = [], [], []
    for g in te_gens:
        ko = np.zeros(n_g)
        ko[g] = 1.0
        ts_ko_l.append(ko)
        ts_u_l.append(u_zero)
        ts_y_l.append(_simulate(ko, u_zero, 100, obs_train))

    # Test 2: Zero-shot combinatorial N-2 double contingencies (12 unseen pairs!) + [5, 9] s extrap
    # 8 pairs among training generators (pure combinatorial epistasis) + 4 pairs with unseen generators
    chosen_pairs: list[tuple[int, int]] = []
    for i_a in range(len(tr_gens)):
        for i_b in range(i_a + 1, len(tr_gens)):
            chosen_pairs.append((int(tr_gens[i_a]), int(tr_gens[i_b])))
    rng.shuffle(chosen_pairs)
    chosen_pairs = chosen_pairs[:8]
    for g_u in te_gens:
        chosen_pairs.append((int(g_u), int(tr_gens[0])))
        chosen_pairs.append((int(g_u), int(tr_gens[1])))
    td_ko_l, td_u_l, td_y_l, td_ye_l = [], [], [], []
    u180 = np.zeros((180, n_g), dtype=np.float64)
    for i_p, k_p in chosen_pairs:
        ko = np.zeros(n_g)
        ko[i_p] = 1.0
        ko[k_p] = 1.0
        ye = _simulate(ko, u180, 180, obs_extrap)
        td_ko_l.append(ko)
        td_u_l.append(u180)
        td_y_l.append(ye[: len(obs_train)])
        td_ye_l.append(ye)

    return PowerGridWorld(
        world_seed=seed,
        n_nodes=n_g,
        adj_binary=adj_bin,
        coupling_weights=b_norm,
        basal_load=basal,
        dt=dt,
        train_ko=np.stack(tr_ko_l),
        train_u_ext=np.stack(tr_u_l),
        train_y=np.stack(tr_y_l),
        train_weights=np.stack(tr_w_l),
        val_ko=np.stack(va_ko_l),
        val_u_ext=np.stack(va_u_l),
        val_y=np.stack(va_y_l),
        val_weights=np.stack(va_w_l),
        test_single_ko=np.stack(ts_ko_l),
        test_single_u_ext=np.stack(ts_u_l),
        test_single_y=np.stack(ts_y_l),
        test_double_ko=np.stack(td_ko_l),
        test_double_u_ext=np.stack(td_u_l),
        test_double_y=np.stack(td_y_l),
        test_double_y_extrap=np.stack(td_ye_l),
    )
