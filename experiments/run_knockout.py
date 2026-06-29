import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from src.model import (
    solver_feller, solver_gauss, solver_gauss_matched,
    compute_batch_stats, get_deterministic_steady_state
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


def run_regime(Iapp, sigma_values, n_trials, dt, t_total, burn_in_steps):
    n_steps = int(t_total / dt)
    y0_seed = get_deterministic_steady_state(Iapp, t_end=3000.0, dt=0.01)

    noise_types = [
        ('feller', solver_feller),
        ('gauss', solver_gauss),
        ('gauss_matched', solver_gauss_matched),
    ]
    regime_data = {name: {'rate': [], 'sem': []} for name, _ in noise_types}

    for sigma in sigma_values:
        noise = np.random.normal(0, np.sqrt(dt), (n_trials, n_steps))

        for key, solver in noise_types:
            v_hist = solver(y0_seed, noise, dt, sigma, Iapp, n_steps, n_trials)
            cv_array, rate_array = compute_batch_stats(v_hist, dt, burn_in_steps, n_trials)
            valid = rate_array[rate_array > 0.01]
            r_mean = np.mean(valid) if len(valid) > 0 else 0.0
            r_sem = np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
            regime_data[key]['rate'].append(r_mean)
            regime_data[key]['sem'].append(r_sem)

        print(f"  sigma={sigma:.3f}: Feller={regime_data['feller']['rate'][-1]:.2f} | "
              f"Gauss={regime_data['gauss']['rate'][-1]:.2f}")

    for key in regime_data:
        for stat in regime_data[key]:
            regime_data[key][stat] = np.array(regime_data[key][stat])

    return regime_data


def plot_knockout(regimes, sigma_arr, all_results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = {'feller': '#d62728', 'gauss': '#1f77b4', 'gauss_matched': '#2ca02c'}
    markers = {'feller': 'o-', 'gauss': 's--', 'gauss_matched': '^:'}
    display = {'feller': 'Feller', 'gauss': 'Gauss (clip)', 'gauss_matched': 'Gauss (matched)'}

    for idx, (Iapp, label) in enumerate(regimes):
        ax = axes[idx]
        for key in ['feller', 'gauss', 'gauss_matched']:
            rates = all_results[Iapp][key]['rate']
            sems = all_results[Iapp][key]['sem']
            ax.errorbar(sigma_arr, rates, yerr=sems,
                        fmt=markers[key], color=colors[key],
                        markersize=7, capsize=3, lw=2, label=display[key])
        ax.set_xscale('log')
        ax.set_xlabel(r'$\sigma_z$')
        ax.set_ylabel('Burst Rate (Hz)')
        ax.set_title(f'{label}\n(I_app={Iapp})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(r'Feller vs Gauss: Knockout Experiment (n=50 trials/point)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out_dir, 'feller_gauss_4quad.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'feller_gauss_4quad.pdf'), bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("Experiment: Feller vs Gauss 4-Quadrant Knockout")
    print("=" * 70)

    start_time = time.time()
    np.random.seed(42)

    regimes = [
        (0.35, "Regime I: Quiescent"),
        (0.39, "Regime IIa: Sub-critical"),
        (0.3955, "Regime IIb: Critical"),
        (0.45, "Regime III: Pacemaker"),
    ]

    sigma_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    n_trials = 50
    dt = 0.01
    t_total = 5000.0
    burn_in_steps = 50000

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for Iapp, label in regimes:
        print(f"\n{'='*50}")
        print(f"  {label} (I_app = {Iapp})")
        print(f"{'='*50}")
        all_results[Iapp] = run_regime(Iapp, sigma_values, n_trials, dt, t_total, burn_in_steps)

    save_dict = {'sigma': sigma_values}
    for Iapp, data in all_results.items():
        for key in data:
            save_dict[f'{Iapp}_{key}_rate'] = data[key]['rate']
            save_dict[f'{Iapp}_{key}_sem'] = data[key]['sem']
    np.savez(os.path.join(OUTPUT_DIR, 'feller_gauss_4quad.npz'), **save_dict)

    plot_knockout(regimes, np.array(sigma_values), all_results, OUTPUT_DIR)

    print(f"\nKnockout summary:")
    for Iapp, label in regimes:
        gauss_rates = all_results[Iapp]['gauss']['rate']
        feller_rates = all_results[Iapp]['feller']['rate']
        print(f"  I_app={Iapp}: Feller={feller_rates.tolist()}")
        print(f"             Gauss ={gauss_rates.tolist()}")

    elapsed = (time.time() - start_time) / 60
    print(f"\nTotal time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
