import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from src.model import solver_feller, compute_batch_stats, get_deterministic_steady_state, gM

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


def run_robustness(Iapp, sigma_values, gM_values, n_trials, dt, t_total, burn_in_steps):
    import src.model as m
    n_steps = int(t_total / dt)
    y0_seed = get_deterministic_steady_state(Iapp, t_end=3000.0, dt=0.01)

    results = {}
    for gM_val in gM_values:
        m.gM = gM_val
        rates = []
        sems = []
        for sigma in sigma_values:
            noise = np.random.normal(0, np.sqrt(dt), (n_trials, n_steps))
            v_hist = solver_feller(y0_seed, noise, dt, sigma, Iapp, n_steps, n_trials)
            _, rate_array = compute_batch_stats(v_hist, dt, burn_in_steps, n_trials)
            valid = rate_array[rate_array > 0.01]
            r_mean = np.mean(valid) if len(valid) > 0 else 0.0
            r_sem = np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
            rates.append(r_mean)
            sems.append(r_sem)
        results[gM_val] = {'rate': np.array(rates), 'sem': np.array(sems)}
        print(f"  gM={gM_val}: peak rate = {max(rates):.2f}")

    m.gM = 1.0
    return results


def plot_robustness(sigma_arr, results, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for gM_val in sorted(results.keys()):
        r = results[gM_val]
        ax.errorbar(sigma_arr, r['rate'], yerr=r['sem'],
                    fmt='o-', capsize=3, lw=1.8, label=f'gM={gM_val}')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma_z$')
    ax.set_ylabel('Burst Rate (Hz)')
    ax.set_title('Biological Robustness (I_app=0.45, gM varied +/- 50%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'robustness.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'robustness.pdf'), bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("Experiment: Biological Robustness (gM variations)")
    print("=" * 70)

    start_time = time.time()
    np.random.seed(789)

    sigma_values = np.logspace(-3, 0, 20)
    gM_values = [0.5, 1.0, 1.5]
    Iapp = 0.45
    n_trials = 50
    dt = 0.01
    t_total = 5000.0
    burn_in_steps = 50000

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = run_robustness(Iapp, sigma_values, gM_values, n_trials, dt, t_total, burn_in_steps)
    plot_robustness(sigma_values, results, OUTPUT_DIR)

    elapsed = (time.time() - start_time) / 60
    print(f"\nTotal time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
