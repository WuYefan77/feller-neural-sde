"""Generate the global burst-rate and CV phase diagram."""

from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from feller_neural_sde import (
    compute_batch_stats,
    get_deterministic_steady_state,
    simulate_feller,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def run_heatmap(i_app_grid, sigma_grid, n_trials, dt, t_total, burn_in_steps, rng):
    n_steps = int(t_total / dt)
    cv_map = np.full((len(i_app_grid), len(sigma_grid)), np.nan)
    rate_map = np.full((len(i_app_grid), len(sigma_grid)), np.nan)

    for i_index, i_app in enumerate(i_app_grid):
        initial_state = get_deterministic_steady_state(i_app, t_end=3000.0, dt=dt)
        for sigma_index, sigma_z in enumerate(sigma_grid):
            noise = rng.normal(0.0, np.sqrt(dt), size=(n_trials, n_steps))
            voltage = simulate_feller(initial_state, noise, dt, sigma_z, i_app)
            cv, rate = compute_batch_stats(voltage, dt, burn_in_steps)
            valid = rate > 0.01
            rate_map[i_index, sigma_index] = np.mean(rate[valid]) if np.any(valid) else 0.0
            cv_map[i_index, sigma_index] = np.mean(cv[valid]) if np.any(valid) else np.nan

    return cv_map, rate_map


def plot_heatmap(i_app_grid, sigma_grid, cv_map, rate_map, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    cv_image = axes[0].pcolormesh(
        sigma_grid,
        i_app_grid,
        cv_map,
        shading="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1.5,
    )
    figure.colorbar(cv_image, ax=axes[0], label="CV")
    axes[0].set(xscale="log", xlabel=r"$\sigma_z$", ylabel=r"$I_{app}$")
    axes[0].set_title("Coefficient of variation")

    rate_image = axes[1].pcolormesh(
        sigma_grid,
        i_app_grid,
        rate_map,
        shading="auto",
        cmap="inferno",
        vmin=0,
        vmax=8,
    )
    figure.colorbar(rate_image, ax=axes[1], label="Burst rate (Hz)")
    axes[1].set(xscale="log", xlabel=r"$\sigma_z$", ylabel=r"$I_{app}$")
    axes[1].set_title("Burst rate")
    figure.tight_layout()
    figure.savefig(output_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    figure.savefig(output_dir / "heatmap.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(456)
    i_app_grid = np.linspace(0.30, 0.50, 25)
    sigma_grid = np.logspace(-3, 0, 18)
    n_trials = 50
    dt = 0.01
    t_total = 5000.0
    burn_in_steps = 50_000

    OUTPUT_DIR.mkdir(exist_ok=True)
    cv_map, rate_map = run_heatmap(
        i_app_grid,
        sigma_grid,
        n_trials,
        dt,
        t_total,
        burn_in_steps,
        rng,
    )
    plot_heatmap(i_app_grid, sigma_grid, cv_map, rate_map, OUTPUT_DIR)
    np.savez(
        OUTPUT_DIR / "heatmap.npz",
        i_app_grid=i_app_grid,
        sigma_grid=sigma_grid,
        cv_map=cv_map,
        rate_map=rate_map,
    )
    print(f"Completed in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
