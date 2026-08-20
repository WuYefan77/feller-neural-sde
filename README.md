# Feller Neural SDE

Reusable Python components from the project
[*Noise-Induced Transitions and Coherence Resonance in a 5D Conductance-Based Neuronal Model*](https://arxiv.org/abs/2605.04088).

The repository implements selected public components of the numerical workflow: a five-dimensional conductance-based CA1 pyramidal-neuron model, bounded state-dependent noise on the slow M-current gate, burst statistics, restricted low-noise Arrhenius analysis and parameter-robustness experiments.

## Model

The slow M-current gating variable follows

$$
dz = \frac{z_\infty(V)-z}{\tau_z}\,dt + \sigma_z\sqrt{z(1-z)}\,dW_t.
$$

The state-dependent diffusion vanishes at the physical boundaries. The implementation uses a full-truncation semi-implicit Euler scheme to control numerical boundary violations while retaining the geometry of the diffusion term.

The voltage and remaining gates form a Hodgkin–Huxley-type five-dimensional conductance model based on the CA1 pyramidal-neuron formulation of Golomb et al. (2006). Numba-compiled kernels support parallel Monte Carlo trials.

## Installation

```bash
git clone https://github.com/WuYefan77/feller-neural-sde.git
cd feller-neural-sde
python -m pip install -e .
```

## Python API

```python
import numpy as np

from feller_neural_sde import (
    compute_batch_stats,
    get_deterministic_steady_state,
    simulate_feller,
)

rng = np.random.default_rng(42)
dt = 0.01
n_trials = 10
n_steps = 100_000
i_app = 0.39

initial_state = get_deterministic_steady_state(i_app, dt=dt)
noise = rng.normal(0.0, np.sqrt(dt), size=(n_trials, n_steps))
voltage = simulate_feller(
    initial_state,
    noise,
    dt,
    sigma_z=0.01,
    i_app=i_app,
)
cv, burst_rate = compute_batch_stats(
    voltage,
    dt,
    burn_in_steps=10_000,
)
```

Wiener increments are generated outside the solver, which makes random seeds and common-random-number comparisons explicit. Model perturbations such as `g_m` are passed as function arguments rather than mutable globals.

## Included experiments

```text
experiments/
├── run_heatmap.py       burst-rate and CV phase diagrams
├── run_knockout.py      state-dependent versus Gaussian controls
├── run_kramers.py       restricted low-noise Arrhenius-like fits
└── run_robustness.py    M-current conductance sensitivity
```

Run the lightweight example with:

```bash
python demo.py
```

The experiment scripts save generated arrays and figures under `data/`, which is excluded from version control.

## Numerical scope

This public repository is a compact, reusable subset of the research code rather than a complete reproduction archive. It focuses on the numerical mechanisms needed to inspect the model and rerun the included analyses.

The accompanying study reports model-specific coherence resonance, restricted Arrhenius-like scaling in a deep subthreshold regime, noise-accelerated firing under strong state-dependent noise and qualitative differences from clipped additive-Gaussian controls. See [arXiv:2605.04088](https://arxiv.org/abs/2605.04088) for the complete analysis and interpretation.

## Author

Yefan Wu, University of Sydney
