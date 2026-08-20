"""Five-dimensional conductance-based neuronal model and SDE solvers."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


C_M = 1.0
V_NA = 55.0
V_K = -90.0
V_L = -70.0
G_NA = 35.0
G_KDR = 6.0
G_A = 1.4
G_L = 0.05
G_NAP = 0.25
TAU_B = 15.0
TAU_Z = 75.0


@njit(cache=True)
def _logistic(argument: float) -> float:
    argument = max(min(argument, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(argument))


@njit(cache=True)
def m_inf(voltage: float) -> float:
    return _logistic(-(voltage + 30.0) / 9.5)


@njit(cache=True)
def h_inf(voltage: float) -> float:
    return _logistic((voltage + 45.0) / 7.0)


@njit(cache=True)
def n_inf(voltage: float) -> float:
    return _logistic(-(voltage + 35.0) / 10.0)


@njit(cache=True)
def a_inf(voltage: float) -> float:
    return _logistic(-(voltage + 50.0) / 20.0)


@njit(cache=True)
def b_inf(voltage: float) -> float:
    return _logistic((voltage + 80.0) / 6.0)


@njit(cache=True)
def z_inf(voltage: float) -> float:
    return _logistic(-(voltage + 39.0) / 5.0)


@njit(cache=True)
def p_inf(voltage: float) -> float:
    return _logistic(-(voltage + 47.0) / 3.0)


@njit(cache=True)
def tau_h(voltage: float) -> float:
    return 0.1 + 0.75 * _logistic(-(voltage + 40.5) / 6.0)


@njit(cache=True)
def tau_n(voltage: float) -> float:
    return 0.1 + 0.5 * _logistic(-(voltage + 27.0) / 15.0)


@njit(cache=True)
def _advance_deterministic_gates(
    h: float,
    n: float,
    b: float,
    h_target: float,
    n_target: float,
    b_target: float,
    tau_h_value: float,
    tau_n_value: float,
    dt: float,
) -> tuple[float, float, float]:
    h = (h + h_target / tau_h_value * dt) / (1.0 + dt / tau_h_value)
    n = (n + n_target / tau_n_value * dt) / (1.0 + dt / tau_n_value)
    b = (b + b_target / TAU_B * dt) / (1.0 + dt / TAU_B)
    return h, n, b


@njit(parallel=True, cache=True)
def _simulate_kernel(
    initial_state: np.ndarray,
    noise: np.ndarray,
    dt: float,
    sigma_z: float,
    i_app: float,
    g_m: float,
    noise_mode: int,
) -> np.ndarray:
    n_trials, n_steps = noise.shape
    voltage_history = np.empty((n_trials, n_steps), dtype=np.float64)

    for trial in prange(n_trials):
        voltage, h, n, b, z = initial_state

        for step in range(n_steps):
            voltage_history[trial, step] = voltage

            m_value = m_inf(voltage)
            h_target = h_inf(voltage)
            n_target = n_inf(voltage)
            a_value = a_inf(voltage)
            b_target = b_inf(voltage)
            z_target = z_inf(voltage)
            p_value = p_inf(voltage)
            tau_h_value = tau_h(voltage)
            tau_n_value = tau_n(voltage)

            i_na = G_NA * m_value**3 * h * (voltage - V_NA)
            i_nap = G_NAP * p_value * (voltage - V_NA)
            i_kdr = G_KDR * n**4 * (voltage - V_K)
            i_a = G_A * a_value**3 * b * (voltage - V_K)
            i_m = g_m * z * (voltage - V_K)
            i_l = G_L * (voltage - V_L)

            d_voltage = (-i_na - i_nap - i_kdr - i_a - i_m - i_l + i_app) / C_M
            voltage += d_voltage * dt

            h, n, b = _advance_deterministic_gates(
                h,
                n,
                b,
                h_target,
                n_target,
                b_target,
                tau_h_value,
                tau_n_value,
                dt,
            )

            if noise_mode == 0:
                z_effective = min(max(z, 0.0), 1.0)
                diffusion = sigma_z * np.sqrt(z_effective * (1.0 - z_effective))
            else:
                diffusion = sigma_z

            z = (
                z
                + z_target / TAU_Z * dt
                + diffusion * noise[trial, step]
            ) / (1.0 + dt / TAU_Z)

            if noise_mode == 1:
                z = min(max(z, 0.0), 1.0)

    return voltage_history


def _validate_simulation_inputs(
    initial_state,
    noise,
    dt: float,
    sigma_z: float,
    i_app: float,
    g_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    state = np.asarray(initial_state, dtype=np.float64)
    increments = np.asarray(noise, dtype=np.float64)

    if state.shape != (5,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain five finite values")
    if increments.ndim != 2 or increments.shape[0] < 1 or increments.shape[1] < 1:
        raise ValueError("noise must have shape (n_trials, n_steps)")
    if not np.all(np.isfinite(increments)):
        raise ValueError("noise must contain only finite values")

    dt = float(dt)
    sigma_z = float(sigma_z)
    i_app = float(i_app)
    g_m = float(g_m)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(sigma_z) or sigma_z < 0.0:
        raise ValueError("sigma_z must be finite and non-negative")
    if not np.isfinite(i_app):
        raise ValueError("i_app must be finite")
    if not np.isfinite(g_m) or g_m < 0.0:
        raise ValueError("g_m must be finite and non-negative")
    return state, increments, dt, sigma_z, i_app, g_m


def simulate_feller(
    initial_state,
    noise,
    dt: float,
    sigma_z: float,
    i_app: float,
    *,
    g_m: float = 1.0,
) -> np.ndarray:
    """Simulate bounded state-dependent diffusion on the M-current gate.

    ``noise`` must contain pre-generated Wiener increments with variance ``dt``.
    The diffusion coefficient is evaluated using full truncation of the gate.
    """
    values = _validate_simulation_inputs(initial_state, noise, dt, sigma_z, i_app, g_m)
    return _simulate_kernel(*values, noise_mode=0)


def simulate_gaussian(
    initial_state,
    noise,
    dt: float,
    sigma_z: float,
    i_app: float,
    *,
    g_m: float = 1.0,
) -> np.ndarray:
    """Simulate additive Gaussian gate noise with post-step clipping."""
    values = _validate_simulation_inputs(initial_state, noise, dt, sigma_z, i_app, g_m)
    return _simulate_kernel(*values, noise_mode=1)


def simulate_matched_gaussian(
    initial_state,
    noise,
    dt: float,
    sigma_z: float,
    i_app: float,
    *,
    g_m: float = 1.0,
) -> np.ndarray:
    """Simulate the amplitude-matched Gaussian control used in the study."""
    return simulate_gaussian(
        initial_state,
        noise,
        dt,
        0.5 * sigma_z,
        i_app,
        g_m=g_m,
    )


@njit(cache=True)
def _integrate_deterministic(
    initial_state: np.ndarray,
    dt: float,
    i_app: float,
    g_m: float,
    n_steps: int,
) -> np.ndarray:
    voltage, h, n, b, z = initial_state

    for _ in range(n_steps):
        m_value = m_inf(voltage)
        h_target = h_inf(voltage)
        n_target = n_inf(voltage)
        a_value = a_inf(voltage)
        b_target = b_inf(voltage)
        z_target = z_inf(voltage)
        p_value = p_inf(voltage)
        tau_h_value = tau_h(voltage)
        tau_n_value = tau_n(voltage)

        i_na = G_NA * m_value**3 * h * (voltage - V_NA)
        i_nap = G_NAP * p_value * (voltage - V_NA)
        i_kdr = G_KDR * n**4 * (voltage - V_K)
        i_a = G_A * a_value**3 * b * (voltage - V_K)
        i_m = g_m * z * (voltage - V_K)
        i_l = G_L * (voltage - V_L)

        voltage += (-i_na - i_nap - i_kdr - i_a - i_m - i_l + i_app) / C_M * dt
        h += (h_target - h) / tau_h_value * dt
        n += (n_target - n) / tau_n_value * dt
        b += (b_target - b) / TAU_B * dt
        z += (z_target - z) / TAU_Z * dt

    return np.array([voltage, h, n, b, z])


def get_deterministic_steady_state(
    i_app: float,
    *,
    g_m: float = 1.0,
    t_end: float = 2000.0,
    dt: float = 0.01,
) -> np.ndarray:
    """Approximate a deterministic initial state by forward integration."""
    if not np.isfinite(i_app):
        raise ValueError("i_app must be finite")
    if not np.isfinite(g_m) or g_m < 0.0:
        raise ValueError("g_m must be finite and non-negative")
    if not np.isfinite(t_end) or t_end <= 0.0 or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("t_end and dt must be finite and positive")

    voltage_rest = -70.0
    initial_state = np.array(
        [
            voltage_rest,
            h_inf(voltage_rest),
            n_inf(voltage_rest),
            b_inf(voltage_rest),
            z_inf(voltage_rest),
        ],
        dtype=np.float64,
    )
    n_steps = int(t_end / dt)
    return _integrate_deterministic(initial_state, dt, i_app, g_m, n_steps)
