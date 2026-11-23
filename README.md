# MemGRN: Learning Interacting Network Dynamics with Explicit Weights and Adaptive Memory

Clean, standalone, and reproducible reference implementation for **MemGRN** (including closed-loop autonomous network dynamics and **MemGRN-Ortho**), accompanying the ICLR manuscript *MemGRN: Learning Interacting Network Dynamics with Explicit Weights and Adaptive Memory*.

## Overview

**MemGRN** enforces **source-isolated adaptive memory** across both **autonomous closed-loop interacting networks** and **regulator-driven dynamical systems** spanning **geophysical/atmospheric chemistry, urban highway traffic, electric power grids, financial order-flow contagion, temporal social networks, and molecular/cellular gene regulation**: each source or node $i$ evolves its own private, bounded, decoupled loading/clearance memory channels $(z_{ir}(t), h_{ir}(t))$ with $(\lambda^{\uparrow}_{ir}, \lambda^{\downarrow}_{ir}) \in [\lambda_{\min}, \lambda_{\max}]^2$ and $\partial(z_{ir}, h_{ir}) / \partial g_k \equiv 0$ ($k \neq i$), feeding an explicit signed interaction readout ($D_{ij}, B_{irj}$, plus $\ell_1$-sparse pairwise cooperativity $\Gamma_{ikj}\bar{h}_i\bar{h}_k$):

1. **$S$-Source Separation, Closed-Loop Identifiability, & Unit-DC-Gain Separation vs. Source-Local LTC (Theorem 1)**: Proves coupled adaptive gates (including diagonal selective SSM / Mamba timescale projections) incur cross-source leakage across correlated covariances $\Sigma_a$ and $GL(M)$ closed-loop gauge ambiguity, whereas hollow MemGRN is globally trapped in a compact absorbing ball, uniquely identifies signed direct and steady-state graphs, and separates from block-diagonal first-order LTCs via exact unit-DC-gain decoupling and second-order post-shutoff peaking.
2. **Non-Markovian Separation & Orthogonal Cofactor Invariance (Proposition 1)**: Establishes a universal Grönwall error floor for state-only perturbation ODEs (`PerturbODE`) on iso-state washout pairs, and proves that $L^2(\mathbb{P}_{\mathrm{train}})$-orthogonalized cofactor modules (**MemGRN-Ortho**) absorb unmeasured parallel pathways without stealing observed source attributions.
3. **Counterbalanced $4{:}1$ Strength-Swap Identifiability (Theorem 2)**: Resolves opposing-parent readout cancellation via asymmetric dual-pulse designs and $\mathrm{AR}(1)$ innovation dynamic programming.

## Repository Structure

```text
MemGRN/
├── src/memgrn/
│   ├── dynamics.py               # Unified asymmetric MemGRN, closed-loop network RK4, & orthogonal cofactor projection
│   ├── baselines.py              # Open-loop & closed-loop JAX suites: MemGRN (+self-mem), MemGRN-Ortho, LTC, Mamba, CfC, CDE, PerturbODE, Latent ODE
│   ├── discrete_baselines.py     # Pure-JAX discrete-time deep learning comparators: DLinear, GRU, and causal Temporal Transformer
│   ├── theory.py                 # General-S covariance, Selective-SSM corollary, closed-loop ID, vs. Local LTC, iso-state, & Ortho OOD audits
│   ├── identification.py         # 4:1 strength-swap algebra, Theorem 2 closed form, margin & AR(1) DP bounds
│   ├── multitf_benchmark.py      # Out-of-class 8-source, 24-target additive & cooperative correlation-shift generator
│   ├── simulators.py             # Occupancy-gated & reversible mass-action buffer synthetic generators
│   ├── datasets.py               # Hash-pinned hiPSC GSE213069, CD4+ T-cell GSE314342, EMT GSE147405, Msn2, Macrophage, & NF-kB loaders
│   ├── cross_domain_datasets.py  # Hash-pinned UCI Beijing Air Quality, METR-LA Traffic, Binance 16-Asset, & SNAP Reddit loaders
│   ├── powergrid.py              # Published IEEE 39-Bus New England Power Grid (Kron admittance & N-1 -> N-2 simulator)
│   └── training.py               # Differentiable JAX RK4 frozen-readout rate trainers
├── scripts/
│   ├── verify_theory.py                  # Deterministic numerical verification of all theorems and propositions
│   ├── run_fair_open_loop_suite.py       # Equal-budget matched open-loop suite (Beijing u/Delta-u/PO6, Multi-TF, 6/8 PO + MEMGRN_UNPROJ, EMT, Binance, Yeast, S=8..512)
│   ├── run_fair_closed_loop_suite.py     # Matched closed-loop suite (S=207 METR-LA, IEEE-39 hollow+selfmem, non-overlapping Reddit, S=128 CD4+ T-cell)
│   ├── run_cross_domain_portraits.py     # Gauge-invariant RK4 step-response dynamical hysteresis portraits (Beijing, Binance, METR-LA ctrl, Yeast)
│   ├── run_beijing_air_quality.py        # Legacy standalone UCI Beijing Air Quality script
│   ├── run_metr_la_traffic.py            # Legacy standalone METR-LA S=64 script
│   ├── run_powergrid_n2.py               # Legacy standalone IEEE 39-Bus script
│   ├── run_financial_contagion.py        # Legacy standalone Binance 16-Asset script
│   ├── run_reddit_temporal.py            # Legacy standalone SNAP Reddit script
│   ├── run_cd4t_closed_loop.py           # Legacy standalone CD4+ T-cell script
│   ├── run_hipsc_closed_loop.py          # Closed-loop longitudinal hiPSC CRISPR Perturb-seq (GSE213069, 14,945 cells)
│   ├── run_cyclic_double_ko.py           # Cyclic closed-loop feedback & zero-shot combinatorial double-KO benchmark (12 worlds)
│   ├── run_sensitivity_ablations.py      # Channel count R, noise sigma, & irregular time-point dropout ablations
│   ├── run_msn2_joint_and_transfer.py    # Cross-readout 2x2 & LOPO shared-filter transfer on live-cell yeast Msn2
│   ├── run_macrophage_multitf.py         # 4-TF mammalian macrophage 14-stimulus, 477-gene benchmark
│   ├── run_nfkb_ortho.py                 # MemGRN-Ortho unmeasured-cofactor benchmark on 177-gene fibroblast washout
│   ├── export_trajectory_examples.py     # Deterministic continuous-time trajectory export for mechanism figures
│   ├── reproduce_all.py                  # Complete SHA-256 design-lock & claim-ledger audit + live double-precision ODE replay
│   └── build_paper_assets.py             # Rebuilds all 8 vector figures (PDF/SVG/PNG) and 19 LaTeX tables + claim_ledger.json
├── results/                  # Pre-locked SHA-256 design manifests and score archives
└── tests/                    # Pytest unit, theoretical, and end-to-end reproducibility tests
```

## Quick Start & Reproducibility

```bash
pip install -e ".[dev]"
pytest -q
python scripts/verify_theory.py
python scripts/reproduce_all.py
python scripts/build_paper_assets.py
```

To re-run the experimental suites from scratch:

```bash
python scripts/run_beijing_air_quality.py
python scripts/run_metr_la_traffic.py
python scripts/run_powergrid_n2.py
python scripts/run_financial_contagion.py
python scripts/run_reddit_temporal.py
python scripts/run_self_memory_and_discrete_closed_loop.py
python scripts/run_open_loop_upgrades.py
python scripts/run_cross_domain_portraits.py
python scripts/run_cd4t_closed_loop.py
python scripts/run_hipsc_closed_loop.py
python scripts/run_cyclic_double_ko.py
python scripts/run_emt_withdrawal_forecast.py
python scripts/run_multitf_factorial.py
python scripts/run_partial_obs_ortho.py
python scripts/run_sensitivity_ablations.py
python scripts/run_msn2_joint_and_transfer.py
python scripts/run_interpretability_and_scaling.py
python scripts/run_macrophage_multitf.py
python scripts/run_nfkb_ortho.py
python scripts/export_trajectory_examples.py
```

## License

Released under the [MIT License](LICENSE).
