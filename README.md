# MemGRN: Learning Regulatory Dynamics with Explicit Weights and Adaptive Memory

Clean, standalone, and reproducible reference implementation for **MemGRN** and **MemGRN-Int**, accompanying the ICLR manuscript *MemGRN: Learning Regulatory Dynamics with Explicit Weights and Adaptive Memory*.

## Overview

**MemGRN** enforces **source-isolated adaptive memory**: each measured transcription factor (TF) $g_i(t)$ evolves its own local, bounded, state-dependent filter bank $(z_{ir}(t), h_{ir}(t))$ with $\partial(z_{ir}, h_{ir}) / \partial g_k \equiv 0$ ($k \neq i$), feeding an explicit target accumulation--relaxation readout $\dot{y}_j = p_j(t) - \delta_j y_j(t)$ (plus optional sparse pairwise enhancer cooperativity $\Gamma_{ikj}\bar{h}_i\bar{h}_k$ in **MemGRN-Int**):

1. **Counterfactual intervention invariance over coupled LTCs (Theorem 1)**: Prevents spurious cross-regulator rate leakage when test-time interventions break training-time TF correlations.
2. **Non-Markovian separation over state-only PerturbODEs (Proposition 1)**: Resolves iso-expression pulse pairs that reach identical $y(T)$ at stimulus shutoff via distinct intracellular memory histories.
3. **Counterbalanced $4{:}1$ strength-swap identifiability (Theorem 2)**: Breaks symmetric two-parent cancellation in dual-source perturbation assays.

## Repository Structure

```text
MemGRN/
├── src/memgrn/
│   ├── dynamics.py           # Source-local MemGRN & MemGRN-Int ODEs, RK4/midpoint solvers, splice accounting
│   ├── baselines.py          # Unified JAX RK4 suite: Local-Const, Coupled-Const, Coupled-Adaptive, LTC, PerturbODE, Latent ODE
│   ├── theory.py             # Coupled cross-talk theorem & iso-expression PerturbODE lower-bound verifiers
│   ├── identification.py     # 4:1 strength-swap algebra, Theorem 2 closed form, margin & AR(1) DP bounds
│   ├── multitf_benchmark.py  # Out-of-class 8-TF, 24-target additive & cooperative correlation-shift generator
│   ├── simulators.py         # Occupancy-gated & reversible mass-action buffer synthetic generators
│   ├── datasets.py           # Hash-pinned Msn2 (Hansen & Zechner 2021), 4-TF Macrophage (Cheng 2017 / Sheu 2023), & NF-kB loaders
│   └── training.py           # Differentiable JAX RK4 frozen-readout rate trainers
├── scripts/
│   ├── verify_theory.py              # Deterministic numerical verification of all theorems and propositions
│   ├── run_multitf_factorial.py      # Locked 2x2 multi-TF correlation-shift benchmark (24 worlds)
│   ├── run_msn2_joint_and_transfer.py # Source-CV joint training, cross-readout 2x2, & LOPO shared-filter transfer
│   ├── run_macrophage_multitf.py     # 4-TF mammalian macrophage 14-stimulus, 477-gene benchmark
│   ├── reproduce_all.py              # SHA-256 receipt audit + live double-precision ODE trajectory replay
│   └── build_paper_assets.py         # Rebuilds all vector figures (PDF/SVG/PNG) and LaTeX tables
├── results/                  # Pre-locked design hashes and score archives
└── tests/                    # 27 Pytest unit, theoretical, and end-to-end reproducibility tests
```

## Quick Start & Reproducibility

```bash
pip install -e ".[dev]"
pytest -q
python scripts/verify_theory.py
python scripts/reproduce_all.py
python scripts/build_paper_assets.py
```

To re-run the three full experimental suites from scratch:

```bash
python scripts/run_multitf_factorial.py
python scripts/run_msn2_joint_and_transfer.py
python scripts/run_macrophage_multitf.py
```

## License

Released under the [MIT License](LICENSE).
