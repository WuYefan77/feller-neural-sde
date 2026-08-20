"""Evaluate restricted low-noise Arrhenius-like scaling."""

from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from feller_neural_sde import (
    compute_batch_stats,
    fit_arrhenius,
    get_deterministic_steady_state,
    simulate_feller,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def run_rates(i_app, sigma_values, n_trials, dt, t_total, burn_in_steps, rng):
    n_steps = int(t_total / dt)
    initial_state = get_deterministic_steady_state(i_app, t_end=3000.0, dt=dt)
    rates = []
    sems = []

    for sigma_z in sigma_values:
        noise = rng.normal(0.0, np.sqrt(dt), size=(n_trials, n_steps))
        voltage = simulate_feller(initial_state, noise, dt, sigma_z, i_app)
        _, rate = compute_batch_stats(voltage, dt, burn_in_steps)
        valid = rate[rate > 0.01]
        rates.append(np.mean(valid) if len(valid) else 0.0)
        sems.append(np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0)
    return np.asarray(rates), np.asarray(sems)


def plot_arrhenius(i_app, sigma_values, rates, sems, output_dir):
    fit = fit_arrhenius(sigma_values, rates, max_sigma=0.1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].errorbar(
        sigma_values,
        rates,
        yerr=sems,
        fmt="o-",
        color="#d62728",
        capsize=3,
    )
    axes[0].set(xscale="log", xlabel=r"$\sigma_z$", ylabel="Burst rate (Hz)")
    axes[0].set_title(f"$I_{{app}}={i_app}$")
    axes[0].grid(True, alpha=0.3)

    x_values = 1.0 / fit.sigma**2
    y_values = np.log(fit.rate)
    x_line = np.linspace(np.min(x_values), np.max(x_values), 100)
    axes[1].scatter(x_values, y_values, color="#1f77b4", s=24)
    axes[1].plot(
        x_line,
        -fit.barrier * x_line + fit.intercept,
        "--",
        color="red",
        label=rf"$R^2={fit.r_squared:.2f}$, $\Delta U_{{eff}}={fit.barrier:.2e}$",
    )
    axes[1].set(xlabel=r"$1/\sigma_z^2$", ylabel=r"$\log(\mathrm{rate})$")
    axes[1].set_title(f"Restricted low-noise fit ({len(fit.sigma)} points)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    stem = f"arrhenius_{i_app}".replace(".", "")
    figure.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)
    return fit


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(123)
    sigma_values = np.logspace(-3, 0, 20)
    n_trials = 50
    dt = 0.01
    t_total = 5000.0
    burn_in_steps = 50_000
    # The accompanying study limits Arrhenius-like interpretation to the
    # deep-subthreshold, low-noise regime.
    regimes = [(0.35, "deep subthreshold")]

    OUTPUT_DIR.mkdir(exist_ok=True)
    saved = {"sigma": sigma_values}
    for i_app, label in regimes:
        rates, sems = run_rates(
            i_app,
            sigma_values,
            n_trials,
            dt,
            t_total,
            burn_in_steps,
            rng,
        )
        fit = plot_arrhenius(i_app, sigma_values, rates, sems, OUTPUT_DIR)
        saved[f"{i_app}_rate"] = rates
        saved[f"{i_app}_sem"] = sems
        print(
            f"{label}: R^2={fit.r_squared:.3f}, "
            f"effective barrier={fit.barrier:.3e}, n={len(fit.sigma)}"
        )

    np.savez(OUTPUT_DIR / "arrhenius_rates.npz", **saved)
    print(f"Completed in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
