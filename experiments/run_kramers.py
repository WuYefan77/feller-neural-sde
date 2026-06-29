import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as ss
import time

from src.model import (
    solver_feller, compute_batch_stats, get_deterministic_steady_state
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


def kramers_fit(rates, sigmas, low_sigma_idx):
    x = 1.0 / sigmas[low_sigma_idx] ** 2
    y = np.log(rates[low_sigma_idx])
    slope, intercept, r_val_lo, _, _ = ss.linregress(x, y)
    barrier = -slope
    w = 1.0 / (ss.linregress(x, y)[4] ** 2 + 1e-9)
    return barrier, r_val_lo ** 2


def run_kramers(Iapp, sigma_values, n_trials, dt, t_total, burn_in_steps):
    n_steps = int(t_total / dt)
    y0_seed = get_deterministic_steady_state(Iapp, t_end=3000.0, dt=0.01)

    rates = []
    sems = []
    for sigma in sigma_values:
        noise = np.random.normal(0, np.sqrt(dt), (n_trials, n_steps))
        v_hist = solver_feller(y0_seed, noise, dt, sigma, Iapp, n_steps, n_trials)
        cv_array, rate_array = compute_batch_stats(v_hist, dt, burn_in_steps, n_trials)
        valid = rate_array[rate_array > 0.01]
        r_mean = np.mean(valid) if len(valid) > 0 else 0.0
        r_sem = np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
        rates.append(r_mean)
        sems.append(r_sem)
        print(f"  sigma={sigma:.4f}: rate={r_mean:.3f} +/- {r_sem:.3f}")

    return np.array(rates), np.array(sems)


def plot_kramers(Iapp, sigma_arr, rates, sems, lo_idx, out_dir):
    sig_low = sigma_arr[lo_idx]
    rate_low = rates[lo_idx]
    sem_low = sems[lo_idx]

    x_lo = 1.0 / sig_low ** 2
    y_lo = np.log(rate_low)
    w_lo = 1.0 / (sem_low / (rate_low + 1e-9)) ** 2
    slope, intercept, r_lo, pval, stderr = ss.linregress(x_lo, y_lo)
    barrier = -slope

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].errorbar(sigma_arr, rates, yerr=sems, fmt='o-', color='#d62728',
                     markersize=6, capsize=3, lw=1.8)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\sigma_z$')
    axes[0].set_ylabel('Burst Rate (Hz)')
    axes[0].set_title(f'I_app = {Iapp} (n=50 trials)')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(1.0 / sigma_arr ** 2, np.log(rates), color='#1f77b4', s=18)
    xs = np.linspace(x_lo[0], x_lo[-1], 100)
    axes[1].plot(xs, slope * xs + intercept, '--', color='red', alpha=0.7,
                 label=f'$R^2={r_lo**2:.2f}$, $\\Delta U_{{eff}}={barrier:.2e}$')
    axes[1].set_xlabel(r'$1/\sigma_z^2$')
    axes[1].set_ylabel(r'$\ln(\text{rate})$')
    axes[1].set_title(f'Arrhenius (low-$\\sigma$, {len(sig_low)} pts)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Kramers Analysis (I_app = {Iapp})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f'kramers_{Iapp}'.replace('.', '')
    fig.savefig(os.path.join(out_dir, f'{fname}.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()

    return r_lo ** 2, barrier


def main():
    print("=" * 70)
    print("Experiment: Kramers Escape Analysis (Regimes I and III)")
    print("=" * 70)

    start_time = time.time()
    np.random.seed(123)

    sigma_dense = np.logspace(-3, 0, 20)
    n_trials = 50
    dt = 0.01
    t_total = 5000.0
    burn_in_steps = 50000
    low_mask = sigma_dense <= 0.1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    regimes = [(0.35, 'Regime I'), (0.45, 'Regime III')]
    for Iapp, label in regimes:
        print(f"\n{label} (I_app = {Iapp})")
        rates, sems = run_kramers(Iapp, sigma_dense, n_trials, dt, t_total, burn_in_steps)
        r2, barrier = plot_kramers(Iapp, sigma_dense, rates, sems, low_mask, OUTPUT_DIR)
        print(f"  Arrhenius R² = {r2:.3f}, barrier = {barrier:.2e}")

    elapsed = (time.time() - start_time) / 60
    print(f"\nTotal time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
