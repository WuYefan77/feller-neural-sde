# Feller Neural SDE: Bounded Multiplicative Noise in a 5D Cortical Pacemaker

Companion code for the paper *"Noise-accelerated Kramers Escape and Coherence Resonance in a 5D Neural Manifold"* (under review, Physical Review E).

## Model

A 5D Hodgkin-Huxley-type conductance-based model of a CA1 pyramidal neuron (Golomb et al., 2006) with state-dependent multiplicative noise on the slow M-type potassium gating variable:

$$dz = \frac{z_\infty(V) - z}{\tau_z} dt + \sigma_z \sqrt{z(1-z)} dW_t$$

The multiplicative term $\sqrt{z(1-z)}$ enforces **Feller boundary conditions**: noise vanishes at the physical limits $z=0$ and $z=1$, preserving the biophysical probability domain $[0,1]$.

## Numerical Scheme

Full-truncation semi-implicit Euler scheme (Lord et al., 2010; Higham, 2005) with:
- Forward Euler for membrane potential $V$
- Semi-implicit for all gating variables $\{h, n, b, z\}$
- Full truncation on diffusion term, no artificial absorbing boundaries

Accelerated with **Numba** (parallel `prange` for multi-trial simulations).

## Repository Structure

```
src/
  model.py                5D model, Feller/Gauss/Gauss-matched solvers, stats
experiments/
  run_knockout.py         Fig 12: Feller vs Gauss knockout (all 4 regimes)
  run_kramers.py          Figs 4,8: Dense Arrhenius/Kramers analysis
  run_heatmap.py          Fig 3:  Global CV + rate phase diagram (25x18)
  run_robustness.py       Fig 11: Conductance perturbations (+/-50%)
demo.py                   Minimal single-point simulation
```

## Usage

```bash
pip install -r requirements.txt

# Quick demo (single noise point)
python demo.py

# Full experiments (generate figures in data/)
python experiments/run_knockout.py
python experiments/run_kramers.py
python experiments/run_heatmap.py
python experiments/run_robustness.py
```

Typical runtime on modern CPU (~10 cores): ~15-20 minutes for full Kramers + heatmap experiments.

## Key Finding

Replacing Feller noise with unbounded Gaussian noise (clipped to [0,1]) **completely abolishes bursting** in all 4 dynamical regimes. This knockout demonstrates that the *geometry* of the boundary constraint, not merely the amplitude of fluctuations, is required for the noise-accelerated bursting mechanism.

## Contact

Yefan Wu, `wuyefan718@gmail.com`
