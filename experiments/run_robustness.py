"""Evaluate sensitivity to M-current conductance perturbations."""

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


def run_robustness(
    i_app,
    sigma_values,
    g_m_values,
    n_trials,
    dt,
    t_total,
    burn_in_steps,
    rng,
):
    n_steps = int(t_total / dt)
    initial_states = {
        g_m: get_deterministic_steady_state(
            i_app,
            g_m=g_m,
            t_end=3000.0,
            dt=dt,
        )
        for g_m in g_m_values
    }
    results = {g_m: {"rate": [], "sem": []} for g_m in g_m_values}

    for sigma_z in sigma_values:
        # Common random numbers isolate the conductance comparison.
        noise = rng.normal(0.0, np.sqrt(dt), size=(n_trials, n_steps))
        for g_m in g_m_values:
            voltage = simulate_feller(
                initial_states[g_m],
                noise,
                dt,
                sigma_z,
                i_app,
                g_m=g_m,
            )
            _, rate = compute_batch_stats(voltage, dt, burn_in_steps)
            valid = rate[rate > 0.01]
            results[g_m]["rate"].append(np.mean(valid) if len(valid) else 0.0)
            results[g_m]["sem"].append(
                np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
            )

    for values in results.values():
        for statistic in values:
            values[statistic] = np.asarray(values[statistic])
    return results


def plot_robustness(sigma_values, results, output_dir):
    figure, axis = plt.subplots(figsize=(8, 5))
    for g_m in sorted(results):
        values = results[g_m]
        axis.errorbar(
            sigma_values,
            values["rate"],
            yerr=values["sem"],
            fmt="o-",
            capsize=3,
            label=rf"$g_M={g_m}$",
        )
    axis.set(xscale="log", xlabel=r"$\sigma_z$", ylabel="Burst rate (Hz)")
    axis.set_title(r"M-current conductance sensitivity ($I_{app}=0.45$)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "robustness.png", dpi=150, bbox_inches="tight")
    figure.savefig(output_dir / "robustness.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(789)
    sigma_values = np.logspace(-3, 0, 20)
    g_m_values = [0.5, 1.0, 1.5]
    i_app = 0.45
    n_trials = 50
    dt = 0.01
    t_total = 5000.0
    burn_in_steps = 50_000

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = run_robustness(
        i_app,
        sigma_values,
        g_m_values,
        n_trials,
        dt,
        t_total,
        burn_in_steps,
        rng,
    )
    plot_robustness(sigma_values, results, OUTPUT_DIR)

    saved = {"sigma": sigma_values}
    for g_m, values in results.items():
        saved[f"{g_m}_rate"] = values["rate"]
        saved[f"{g_m}_sem"] = values["sem"]
    np.savez(OUTPUT_DIR / "robustness.npz", **saved)
    print(f"Completed in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
