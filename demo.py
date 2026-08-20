"""Minimal example for the public Feller neural SDE package."""

import numpy as np

from feller_neural_sde import (
    compute_batch_stats,
    get_deterministic_steady_state,
    simulate_feller,
)


def main() -> None:
    rng = np.random.default_rng(42)
    i_app = 0.39
    sigma_z = 0.01
    n_trials = 10
    dt = 0.01
    t_total = 5000.0
    n_steps = int(t_total / dt)
    burn_in_steps = 50_000

    initial_state = get_deterministic_steady_state(
        i_app,
        t_end=3000.0,
        dt=dt,
    )
    noise = rng.normal(0.0, np.sqrt(dt), size=(n_trials, n_steps))
    voltage = simulate_feller(initial_state, noise, dt, sigma_z, i_app)
    cv, rate = compute_batch_stats(voltage, dt, burn_in_steps)
    valid = rate > 0.01

    print(f"Feller-type noise at I_app={i_app}, sigma_z={sigma_z}")
    if np.any(valid):
        print(f"  Mean burst rate: {np.mean(rate[valid]):.3f} Hz")
        print(f"  Mean burst CV:   {np.nanmean(cv[valid]):.3f}")
    else:
        print("  No trials met the burst-rate threshold")
    print(f"  Valid trials:    {np.count_nonzero(valid)}/{n_trials}")


if __name__ == "__main__":
    main()
