import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.model import solver_feller, compute_batch_stats, get_deterministic_steady_state


def main():
    print("=" * 70)
    print("Minimal Demo: Single-point stochastic simulation")
    print("=" * 70)

    np.random.seed(42)

    Iapp = 0.39
    sigma = 0.01
    n_trials = 10
    dt = 0.01
    t_total = 5000.0
    n_steps = int(t_total / dt)
    burn_in_steps = 50000

    y0_seed = get_deterministic_steady_state(Iapp, t_end=3000.0, dt=0.01)
    noise = np.random.normal(0, np.sqrt(dt), (n_trials, n_steps))

    v_hist = solver_feller(y0_seed, noise, dt, sigma, Iapp, n_steps, n_trials)
    cv_array, rate_array = compute_batch_stats(v_hist, dt, burn_in_steps, n_trials)
    valid = rate_array[rate_array > 0.01]

    print(f"Feller noise at Iapp={Iapp}, sigma={sigma}:")
    print(f"  Mean rate: {np.mean(valid):.3f} Hz")
    print(f"  SEM rate: {np.std(valid)/np.sqrt(len(valid)):.3f}")
    print(f"  Valid trials: {len(valid)}/{n_trials}")


if __name__ == "__main__":
    main()
