# MemGRN: Learning Regulatory Dynamics with Explicit Weights and Adaptive Memory

Clean, standalone, and reproducible reference implementation for **MemGRN**, accompanying the manuscript *MemGRN: Learning Regulatory Dynamics with Explicit Weights and Adaptive Memory*.

## Overview

**MemGRN** separates *which* transcription factor (TF) contributes to a predicted transcriptional response from *how rapidly* that contribution accumulates and relaxes over time:

1. **Source-local adaptive memory cascades**: Each measured TF trajectory $g_i(t)$ is transformed via $a_i(t) = \tanh(g_i(t))$ and routed through $R = 3$ source-isolated two-stage filter channels $(z_{ir}, h_{ir})$ governed by bounded local neural rates $\lambda_{ir}(a_i, z_{ir}, h_{ir}) \in [\lambda_{\min}, \lambda_{\max}]$.
2. **Explicit linear accumulation--relaxation readout**: Immediate ($D_{ij}$) and delayed ($B_{irj}$) signed regulatory weights combine with first-order target relaxation $\dot{y}_j = p_j(t) - \delta_j y_j(t)$, enabling exact variation-of-constants source accounting and frozen-readout kinetic interventions.
3. **Counterbalanced perturbation design**: Algebraic and Gaussian-detection guarantees characterize when two-record $4{:}1$ strength-swapped pulses resolve cancelling multi-parent contributions that remain indistinguishable under repeated stimulation.

## Repository Structure

```text
MemGRN/
├── src/memgrn/
│   ├── dynamics.py         # Bounded source-local filters, RK4/midpoint solvers, exact accounting & splice decomposition
│   ├── identification.py   # 4:1 strength-swap algebra, Theorem 1 exact discrimination, margin & AR(1) DP schedule bounds
│   ├── simulators.py       # Occupancy-gated & out-of-class reversible mass-action buffer synthetic generators
│   ├── datasets.py         # Pinned, hash-verified Hansen & Zechner (2021) Msn2 and Sen et al. (2020) NF-kB loaders
│   └── training.py         # Differentiable JAX RK4 forward passes and cosine-decayed clipped Adam trainers
├── scripts/
│   ├── verify_theory.py    # Independent numerical checks of Proposition 1, Theorem 1, margin bounds & AR(1) DP
│   ├── reproduce_all.py    # SHA-256 receipt audit + live double-precision ODE trajectory replay on frozen checkpoints
│   └── build_paper_assets.py # Rebuilds all vector figures (PDF/SVG/PNG) and LaTeX tables from verified archives
├── results/stage3/         # Lightweight SHA-256-fingerprinted score, design, and Msn2 checkpoint archives
├── tests/                  # Pytest unit and end-to-end reproducibility suite
├── pyproject.toml
└── requirements.txt
```

## Quick Start & Reproducibility

### 1. Installation

```bash
pip install -e ".[dev]"
```

### 2. Run Unit & Reproducibility Tests

```bash
pytest -q
```

### 3. Verify Theoretical Statements Numerically

Checks invariant filter bounds, variation-of-constants additive response accounting, post-shutoff splice decomposition, 25-point Gauss--Kronrod quadrature agreement for Theorem 1 (`< 1e-12`), Gaussian margin lower bounds, and exact minimax AR(1) schedule dynamic programming:

```bash
python scripts/verify_theory.py
```

### 4. Replay Trajectories & Audit SHA-256 Provenance

Verifies SHA-256 digests across archived experimental runs and independently replays all 48 selected six-promoter frozen-readout RK4 trajectories plus single-pulse 30-minute Msn2 transfer predictions in double precision (`max discrepancy < 1.5e-6` standardized reporter units):

```bash
python scripts/reproduce_all.py
```

### 5. Regenerate All Paper Figures & Tables

```bash
python scripts/build_paper_assets.py
```

## External Public Datasets

All biological benchmarks reuse publicly released measurements pinned by commit SHA and SHA-256 file hashes (`memgrn.datasets`):

- **Yeast Msn2 single-cell live microscopy**: Hansen & Zechner (*Molecular Systems Biology*, 2021; [Zenodo 2755026](https://zenodo.org/records/2755026), `zechnerlab/PromoterManifest` commit `9d22cc6f5d06f85543be2cfbdbb13fb7e57c1c70`).
- **Mammalian NF-$\kappa$B EMSA & RNA-seq washout**: Sen et al. (*Cell Systems*, 2020; `biomystery/duration_decode_manuscript` commit `37be90e56f6e4a9c726990c09e0adc114dcc8efc`).

## License

Released under the [MIT License](LICENSE).
