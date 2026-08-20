"""Spike, burst and batch summary statistics."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _find_peaks(voltage_trace, height, min_distance):
    peaks = np.empty(len(voltage_trace), dtype=np.int64)
    count = 0
    last_peak = -min_distance

    for index in range(1, len(voltage_trace) - 1):
        is_peak = (
            voltage_trace[index] > height
            and voltage_trace[index] > voltage_trace[index - 1]
            and voltage_trace[index] > voltage_trace[index + 1]
        )
        if is_peak and index - last_peak >= min_distance:
            peaks[count] = index
            count += 1
            last_peak = index
    return peaks[:count]


@njit(cache=True)
def _detect_burst_starts(spike_times, isi_threshold):
    if len(spike_times) < 2:
        return spike_times[:0]

    starts = np.empty(len(spike_times), dtype=np.float64)
    starts[0] = spike_times[0]
    count = 1
    for index in range(1, len(spike_times)):
        if spike_times[index] - spike_times[index - 1] > isi_threshold:
            starts[count] = spike_times[index]
            count += 1
    return starts[:count]


@njit(cache=True)
def _trial_stats_kernel(
    voltage_trace,
    dt,
    spike_height,
    min_distance_steps,
    isi_threshold,
):
    spike_indices = _find_peaks(voltage_trace, spike_height, min_distance_steps)
    spike_times = spike_indices.astype(np.float64) * dt
    burst_starts = _detect_burst_starts(spike_times, isi_threshold)
    n_bursts = len(burst_starts)

    if n_bursts < 3:
        return np.nan, 0.0

    intervals = np.diff(burst_starts)
    mean_interval = np.mean(intervals)
    cv = np.std(intervals) / mean_interval if mean_interval > 0.0 else np.nan
    analysis_time_seconds = len(voltage_trace) * dt / 1000.0
    rate = n_bursts / analysis_time_seconds if analysis_time_seconds > 0.0 else 0.0
    return cv, rate


def compute_trial_stats(
    voltage_trace,
    dt: float,
    burn_in_steps: int = 0,
    *,
    spike_height: float = -20.0,
    min_distance_steps: int = 2,
    isi_threshold: float = 40.0,
) -> tuple[float, float]:
    """Compute burst CV and rate after discarding the burn-in segment."""
    trace = np.asarray(voltage_trace, dtype=np.float64)
    if trace.ndim != 1 or len(trace) < 3 or not np.all(np.isfinite(trace)):
        raise ValueError("voltage_trace must be a finite one-dimensional array")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not isinstance(burn_in_steps, (int, np.integer)) or not 0 <= burn_in_steps < len(trace):
        raise ValueError("burn_in_steps must index a proper prefix of the trace")
    if not isinstance(min_distance_steps, (int, np.integer)) or min_distance_steps < 1:
        raise ValueError("min_distance_steps must be a positive integer")
    if not np.isfinite(isi_threshold) or isi_threshold <= 0.0:
        raise ValueError("isi_threshold must be finite and positive")

    analysed_trace = trace[burn_in_steps:]
    cv, rate = _trial_stats_kernel(
        analysed_trace,
        float(dt),
        float(spike_height),
        int(min_distance_steps),
        float(isi_threshold),
    )
    return float(cv), float(rate)


@njit(parallel=True, cache=True)
def _batch_stats_kernel(
    voltage_history,
    dt,
    burn_in_steps,
    spike_height,
    min_distance_steps,
    isi_threshold,
):
    n_trials = voltage_history.shape[0]
    cv_values = np.empty(n_trials)
    rate_values = np.empty(n_trials)
    for trial in prange(n_trials):
        cv, rate = _trial_stats_kernel(
            voltage_history[trial, burn_in_steps:],
            dt,
            spike_height,
            min_distance_steps,
            isi_threshold,
        )
        cv_values[trial] = cv
        rate_values[trial] = rate
    return cv_values, rate_values


def compute_batch_stats(
    voltage_history,
    dt: float,
    burn_in_steps: int = 0,
    *,
    spike_height: float = -20.0,
    min_distance_steps: int = 2,
    isi_threshold: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute trial-wise burst statistics after a shared burn-in period."""
    history = np.asarray(voltage_history, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] < 1 or history.shape[1] < 3:
        raise ValueError("voltage_history must have shape (n_trials, n_steps)")
    if not np.all(np.isfinite(history)):
        raise ValueError("voltage_history must contain only finite values")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not isinstance(burn_in_steps, (int, np.integer)) or not 0 <= burn_in_steps < history.shape[1]:
        raise ValueError("burn_in_steps must index a proper prefix of each trace")
    return _batch_stats_kernel(
        history,
        float(dt),
        int(burn_in_steps),
        float(spike_height),
        int(min_distance_steps),
        float(isi_threshold),
    )
