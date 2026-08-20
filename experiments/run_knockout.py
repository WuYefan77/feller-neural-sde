"""Compare state-dependent diffusion with clipped Gaussian controls."""

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
    simulate_gaussian,
    simulate_matched_gaussian,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def run_regime(i_app, sigma_values, n_trials, dt, t_total, burn_in_steps, rng):
    n_steps = int(t_total / dt)
    initial_state = get_deterministic_steady_state(i_app, t_end=3000.0, dt=dt)
    solvers = {
        "feller": simulate_feller,
        "gaussian": simulate_gaussian,
        "gaussian_matched": simulate_matched_gaussian,
    }
    results = {name: {"rate": [], "sem": []} for name in solvers}

    for sigma_z in sigma_values:
        noise = rng.normal(0.0, np.sqrt(dt), size=(n_trials, n_steps))
        for name, solver in solvers.items():
            voltage = solver(initial_state, noise, dt, sigma_z, i_app)
            _, rate = compute_batch_stats(voltage, dt, burn_in_steps)
            valid = rate[rate > 0.01]
            results[name]["rate"].append(np.mean(valid) if len(valid) else 0.0)
            results[name]["sem"].append(
                np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
            )

    for values in results.values():
        for statistic in values:
            values[statistic] = np.asarray(values[statistic])
    return results


def plot_knockout(regimes, sigma_values, all_results, output_dir):
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    styles = {
        "feller": ("Feller-type", "o-", "#d62728"),
        "gaussian": ("Gaussian (clip)", "s--", "#1f77b4"),
        "gaussian_matched": ("Gaussian (matched)", "^:", "#2ca02c"),
    }

    for axis, (i_app, label) in zip(axes.flat, regimes):
        for name, (display, marker, colour) in styles.items():
            values = all_results[i_app][name]
            axis.errorbar(
                sigma_values,
                values["rate"],
                yerr=values["sem"],
                fmt=marker,
                color=colour,
                capsize=3,
                label=display,
            )
        axis.set(xscale="log", xlabel=r"$\sigma_z$", ylabel="Burst rate (Hz)")
        axis.set_title(f"{label}\n($I_{{app}}={i_app}$)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output_dir / "feller_gaussian_4quad.png", dpi=150, bbox_inches="tight")
    figure.savefig(output_dir / "feller_gaussian_4quad.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(42)
    regimes = [
        (0.35, "Regime I: quiescent"),
        (0.39, "Regime IIa: subcritical"),
        (0.3955, "Regime IIb: critical"),
        (0.45, "Regime III: pacemaker"),
    ]
    sigma_values = np.asarray([0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    n_trials = 50
    dt = 0.01
    t_total = 5000.0
    burn_in_steps = 50_000

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {
        i_app: run_regime(
            i_app,
            sigma_values,
            n_trials,
            dt,
            t_total,
            burn_in_steps,
            rng,
        )
        for i_app, _ in regimes
    }
    plot_knockout(regimes, sigma_values, results, OUTPUT_DIR)

    saved = {"sigma": sigma_values}
    for i_app, regime_results in results.items():
        for name, values in regime_results.items():
            saved[f"{i_app}_{name}_rate"] = values["rate"]
            saved[f"{i_app}_{name}_sem"] = values["sem"]
    np.savez(OUTPUT_DIR / "feller_gaussian_4quad.npz", **saved)
    print(f"Completed in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
