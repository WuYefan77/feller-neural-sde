import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from src.model import solver_feller, compute_batch_stats, get_deterministic_steady_state

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


def run_heatmap(iapp_grid, sigma_grid, n_trials, dt, t_total, burn_in_steps):
    n_steps = int(t_total / dt)
    cv_map = np.full((len(iapp_grid), len(sigma_grid)), np.nan)
    rate_map = np.full((len(iapp_grid), len(sigma_grid)), np.nan)

    for ii, Iapp in enumerate(iapp_grid):
        y0_seed = get_deterministic_steady_state(Iapp, t_end=3000.0, dt=0.01)
        for jj, sigma in enumerate(sigma_grid):
            noise = np.random.normal(0, np.sqrt(dt), (n_trials, n_steps))
            v_hist = solver_feller(y0_seed, noise, dt, sigma, Iapp, n_steps, n_trials)
            cv_array, rate_array = compute_batch_stats(v_hist, dt, burn_in_steps, n_trials)
            valid_rate = rate_array[rate_array > 0.01]
            valid_cv = cv_array[rate_array > 0.01]
            rate_map[ii, jj] = np.mean(valid_rate) if len(valid_rate) > 0 else 0.0
            cv_map[ii, jj] = np.mean(valid_cv) if len(valid_cv) > 0 else np.nan
            if jj % 5 == 0 and ii == 0:
                print(f"  I_app={Iapp:.3f}, sigma={sigma:.4f}: rate={rate_map[ii,jj]:.2f}, cv={cv_map[ii,jj]:.3f}" if not np.isnan(cv_map[ii,jj]) else "")

    return cv_map, rate_map


def plot_heatmap(iapp_grid, sigma_grid, cv_map, rate_map, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im0 = axes[0].pcolormesh(sigma_grid, iapp_grid, cv_map, shading='auto',
                              cmap='RdYlGn_r', vmin=0, vmax=1.5)
    fig.colorbar(im0, ax=axes[0], label='CV')
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\sigma_z$')
    axes[0].set_ylabel(r'$I_{app}$')
    axes[0].set_title('Coefficient of Variation (CV)')

    im1 = axes[1].pcolormesh(sigma_grid, iapp_grid, rate_map, shading='auto',
                              cmap='inferno', vmin=0, vmax=8)
    fig.colorbar(im1, ax=axes[1], label='Burst Rate (Hz)')
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$\sigma_z$')
    axes[1].set_ylabel(r'$I_{app}$')
    axes[1].set_title('Burst Rate (Hz)')

    fig.suptitle(f'Global Phase Diagram (25x18 grid, n=50 trials/point)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out_dir, 'heatmap.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'heatmap.pdf'), bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("Experiment: Global Phase Diagram (CV + Rate heatmap)")
    print("=" * 70)

    start_time = time.time()
    np.random.seed(456)

    iapp_grid = np.linspace(0.30, 0.50, 25)
    sigma_grid = np.logspace(-3, 0, 18)
    n_trials = 50
    dt = 0.01
    t_total = 5000.0
    burn_in_steps = 50000

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cv_map, rate_map = run_heatmap(iapp_grid, sigma_grid, n_trials, dt, t_total, burn_in_steps)
    plot_heatmap(iapp_grid, sigma_grid, cv_map, rate_map, OUTPUT_DIR)

    np.savez(os.path.join(OUTPUT_DIR, 'heatmap.npz'),
             iapp_grid=iapp_grid, sigma_grid=sigma_grid,
             cv_map=cv_map, rate_map=rate_map)

    elapsed = (time.time() - start_time) / 60
    print(f"\nTotal time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
