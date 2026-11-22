# MemGRN: Learning Regulatory Dynamics with Explicit Weights and Adaptive Memory

Clean, standalone, and reproducible reference implementation for **MemGRN** (including closed-loop autonomous network dynamics and **MemGRN-Ortho**), accompanying the ICLR manuscript *MemGRN: Learning Regulatory Dynamics with Explicit Weights and Adaptive Memory*.

## Overview

**MemGRN** enforces **source-isolated adaptive memory** across both **autonomous closed-loop gene networks** (predicting longitudinal CRISPR knockouts from $t = 0$ without supplied future TF traces) and **regulator-driven systems**: each gene or transcription factor $i$ evolves its own private, bounded, decoupled rise/recovery memory channels $(z_{ir}(t), h_{ir}(t))$ with $(\lambda^{\uparrow}_{ir}, \lambda^{\downarrow}_{ir}) \in [\lambda_{\min}, \lambda_{\max}]^2$ and $\partial(z_{ir}, h_{ir}) / \partial g_k \equiv 0$ ($k \neq i$), feeding an explicit signed enhancer readout ($D_{ij}, B_{irj}$, plus $\ell_1$-sparse pairwise cooperativity $\Gamma_{ikj}\bar{h}_i\bar{h}_k$):

1. **$S$-Regulator Separation, Closed-Loop Identifiability, & Unit-DC-Gain Separation vs. Source-Local LTC (Theorem 1)**: Proves coupled adaptive gates incur cross-regulator leakage across correlated pathway covariances $\Sigma_a$ and $GL(M)$ closed-loop gauge ambiguity, whereas hollow MemGRN is globally trapped in a compact absorbing ball, uniquely identifies signed direct and steady-state graphs, and separates from block-diagonal first-order LTCs via exact unit-DC-gain decoupling and second-order post-shutoff peaking.
2. **Non-Markovian Separation & Orthogonal Cofactor Invariance (Proposition 1)**: Establishes a universal Grönwall error floor for state-only perturbation ODEs (`PerturbODE`) on iso-expression washout pairs, and proves that $L^2(\mathbb{P}_{\mathrm{train}})$-orthogonalized cofactor modules (**MemGRN-Ortho**) absorb unmeasured parallel pathways without stealing observed TF attributions.
3. **Counterbalanced $4{:}1$ Strength-Swap Identifiability (Theorem 2)**: Resolves opposing-parent readout cancellation via asymmetric dual-pulse designs and $\mathrm{AR}(1)$ innovation dynamic programming.

## Repository Structure

```text
MemGRN/
├── src/memgrn/
│   ├── dynamics.py           # Unified asymmetric MemGRN, closed-loop network RK4, & orthogonal cofactor projection
│   ├── baselines.py          # Open-loop & closed-loop JAX suites: MemGRN, MemGRN-Ortho, LTC, Selective SSM, CfC, Neural CDE, PerturbODE, Latent ODE
│   ├── theory.py             # General-S covariance, Selective-SSM corollary, closed-loop ID, vs. Local LTC, iso-expr., & Ortho OOD audits
│   ├── identification.py     # 4:1 strength-swap algebra, Theorem 2 closed form, margin & AR(1) DP bounds
│   ├── multitf_benchmark.py  # Out-of-class 8-TF, 24-target additive & cooperative correlation-shift generator
│   ├── simulators.py         # Occupancy-gated & reversible mass-action buffer synthetic generators
│   ├── datasets.py           # Hash-pinned hiPSC GSE213069 + ChIP-Atlas, EMT GSE147405, Msn2, Macrophage, & NF-kB loaders
│   └── training.py           # Differentiable JAX RK4 frozen-readout rate trainers
├── scripts/
│   ├── verify_theory.py                  # Deterministic numerical verification of all theorems and propositions
│   ├── run_hipsc_closed_loop.py          # Closed-loop longitudinal hiPSC CRISPR Perturb-seq (GSE213069, 14,945 cells)
│   ├── run_cyclic_double_ko.py           # Cyclic closed-loop feedback & zero-shot combinatorial double-KO benchmark (12 worlds)
│   ├── run_emt_withdrawal_forecast.py    # Zero-shot human cancer EMT 3-day withdrawal benchmark (GSE147405, 53,290 cells)
│   ├── run_multitf_factorial.py          # Locked 2x2 multi-TF correlation-shift benchmark (24 worlds, 11 models)
│   ├── run_partial_obs_ortho.py          # Ground-truth 6-of-8 MemGRN-Ortho & parameter-matched Local-LTC separation
│   ├── run_sensitivity_ablations.py      # Channel count R, noise sigma, & irregular time-point dropout ablations
│   ├── run_msn2_joint_and_transfer.py    # Source-CV joint training, cross-readout 2x2, & LOPO shared-filter transfer
│   ├── run_interpretability_and_scaling.py # Yeast chromatin-memory phase portraits & S=8..512 (2,048-gene) scaling
│   ├── run_macrophage_multitf.py         # 4-TF mammalian macrophage 14-stimulus, 477-gene benchmark
│   ├── run_nfkb_ortho.py                 # MemGRN-Ortho unmeasured-cofactor benchmark on 177-gene fibroblast washout
│   ├── export_trajectory_examples.py     # Deterministic continuous-time trajectory export for mechanism figures
│   ├── reproduce_all.py                  # SHA-256 receipt audit + live double-precision ODE trajectory replay
│   └── build_paper_assets.py             # Rebuilds all vector figures (PDF/SVG/PNG) and LaTeX tables
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
