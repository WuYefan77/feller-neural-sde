import numpy as np
from numba import njit, prange

C_m = 1.0
VNa = 55.0
VK = -90.0
VL = -70.0
gNa = 35.0
gKdr = 6.0
gA = 1.4
gM = 1.0
gL = 0.05
gNaP = 0.25
tau_b = 15.0
tau_z = 75.0


@njit(cache=True)
def m_inf(V):
    arg = -(V + 30.0) / 9.5
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(arg))


@njit(cache=True)
def h_inf(V):
    arg = (V + 45.0) / 7.0
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(arg))


@njit(cache=True)
def n_inf(V):
    arg = -(V + 35.0) / 10.0
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(arg))


@njit(cache=True)
def a_inf(V):
    arg = -(V + 50.0) / 20.0
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(arg))


@njit(cache=True)
def b_inf(V):
    arg = (V + 80.0) / 6.0
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(arg))


@njit(cache=True)
def z_inf(V):
    arg = -(V + 39.0) / 5.0
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(arg))


@njit(cache=True)
def p_inf(V):
    arg = -(V + 47.0) / 3.0
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(arg))


@njit(cache=True)
def tau_h(V):
    arg = -(V + 40.5) / 6.0
    arg = max(min(arg, 500.0), -500.0)
    return 0.1 + 0.75 / (1.0 + np.exp(arg))


@njit(cache=True)
def tau_n(V):
    arg = -(V + 27.0) / 15.0
    arg = max(min(arg, 500.0), -500.0)
    return 0.1 + 0.5 / (1.0 + np.exp(arg))


@njit(parallel=True, cache=True)
def solver_feller(y0, noise, dt, sigma_z, Iapp, n_steps, n_trials):
    v_history = np.empty((n_trials, n_steps))

    for j in prange(n_trials):
        V = y0[0]
        h = y0[1]
        n = y0[2]
        b = y0[3]
        z = y0[4]

        for i in range(n_steps):
            v_history[j, i] = V

            m_val = m_inf(V)
            h_inf_v = h_inf(V)
            n_inf_v = n_inf(V)
            a_val = a_inf(V)
            b_inf_v = b_inf(V)
            z_inf_v = z_inf(V)
            p_val = p_inf(V)
            tau_h_v = tau_h(V)
            tau_n_v = tau_n(V)

            INa = gNa * m_val**3 * h * (V - VNa)
            INaP = gNaP * p_val * (V - VNa)
            IKdr = gKdr * n**4 * (V - VK)
            IA = gA * a_val**3 * b * (V - VK)
            IM = gM * z * (V - VK)
            IL = gL * (V - VL)

            dV = (-INa - INaP - IKdr - IA - IM - IL + Iapp) / C_m
            V = V + dV * dt

            z_eff = min(max(z, 0.0), 1.0)
            diff_z = sigma_z * np.sqrt(z_eff * (1.0 - z_eff))

            h = (h + h_inf_v / tau_h_v * dt) / (1.0 + dt / tau_h_v)
            n = (n + n_inf_v / tau_n_v * dt) / (1.0 + dt / tau_n_v)
            b = (b + b_inf_v / tau_b * dt) / (1.0 + dt / tau_b)
            z = (z + z_inf_v / tau_z * dt + diff_z * noise[j, i]) / (1.0 + dt / tau_z)

    return v_history


@njit(parallel=True, cache=True)
def solver_gauss(y0, noise, dt, sigma_z, Iapp, n_steps, n_trials):
    v_history = np.empty((n_trials, n_steps))

    for j in prange(n_trials):
        V = y0[0]
        h = y0[1]
        n = y0[2]
        b = y0[3]
        z = y0[4]

        for i in range(n_steps):
            v_history[j, i] = V

            m_val = m_inf(V)
            h_inf_v = h_inf(V)
            n_inf_v = n_inf(V)
            a_val = a_inf(V)
            b_inf_v = b_inf(V)
            z_inf_v = z_inf(V)
            p_val = p_inf(V)
            tau_h_v = tau_h(V)
            tau_n_v = tau_n(V)

            INa = gNa * m_val**3 * h * (V - VNa)
            INaP = gNaP * p_val * (V - VNa)
            IKdr = gKdr * n**4 * (V - VK)
            IA = gA * a_val**3 * b * (V - VK)
            IM = gM * z * (V - VK)
            IL = gL * (V - VL)

            dV = (-INa - INaP - IKdr - IA - IM - IL + Iapp) / C_m
            V = V + dV * dt

            h = (h + h_inf_v / tau_h_v * dt) / (1.0 + dt / tau_h_v)
            n = (n + n_inf_v / tau_n_v * dt) / (1.0 + dt / tau_n_v)
            b = (b + b_inf_v / tau_b * dt) / (1.0 + dt / tau_b)
            z = (z + z_inf_v / tau_z * dt + sigma_z * noise[j, i]) / (1.0 + dt / tau_z)
            z = min(max(z, 0.0), 1.0)

    return v_history


@njit(parallel=True, cache=True)
def solver_gauss_matched(y0, noise, dt, sigma_z, Iapp, n_steps, n_trials):
    sigma_eff = sigma_z * 0.5
    return solver_gauss(y0, noise, dt, sigma_eff, Iapp, n_steps, n_trials)


@njit(cache=True)
def find_peaks_numba(v_trace, height, min_distance):
    n = len(v_trace)
    peaks = np.empty(n // 10, dtype=np.int64)
    count = 0
    last_peak = -min_distance

    for i in range(1, n - 1):
        if v_trace[i] > height and v_trace[i] > v_trace[i - 1] and v_trace[i] > v_trace[i + 1]:
            if i - last_peak >= min_distance:
                if count < len(peaks):
                    peaks[count] = i
                    count += 1
                last_peak = i

    return peaks[:count]


@njit(cache=True)
def detect_bursts_numba(spike_times, isi_threshold):
    n_spikes = len(spike_times)
    if n_spikes < 2:
        return spike_times[0:0]

    burst_starts = np.empty(n_spikes, dtype=np.float64)
    burst_starts[0] = spike_times[0]
    count = 1

    for i in range(1, n_spikes):
        if (spike_times[i] - spike_times[i - 1]) > isi_threshold:
            burst_starts[count] = spike_times[i]
            count += 1

    return burst_starts[:count]


@njit(cache=True)
def compute_trial_stats(v_trace, dt, burn_in_steps, sigma_z, Iapp_app):
    spike_indices = find_peaks_numba(v_trace, -20.0, 2)
    n_spikes = len(spike_indices)

    spike_times = np.empty(n_spikes, dtype=np.float64)
    for i in range(n_spikes):
        spike_times[i] = spike_indices[i] * dt

    burst_starts = detect_bursts_numba(spike_times, 40.0)
    n_bursts = len(burst_starts)

    if n_bursts < 3:
        return np.nan, 0.0

    ibis = np.empty(n_bursts - 1, dtype=np.float64)
    for i in range(n_bursts - 1):
        ibis[i] = burst_starts[i + 1] - burst_starts[i]

    analysis_time = (len(v_trace) - burn_in_steps) * dt / 1000.0
    rate = n_bursts / analysis_time if analysis_time > 0 else 0.0

    mean_ibi = 0.0
    for i in range(len(ibis)):
        mean_ibi += ibis[i]
    mean_ibi /= len(ibis)

    var_ibi = 0.0
    for i in range(len(ibis)):
        var_ibi += (ibis[i] - mean_ibi) ** 2
    var_ibi /= len(ibis)

    cv = np.sqrt(var_ibi) / mean_ibi if mean_ibi > 0 else np.nan

    return cv, rate


@njit(parallel=True, cache=True)
def compute_batch_stats(v_hist, dt, burn_in_steps, n_trials):
    cv_array = np.empty(n_trials)
    rate_array = np.empty(n_trials)

    for j in prange(n_trials):
        cv, rate = compute_trial_stats(v_hist[j], dt, burn_in_steps, 0.0, 0.0)
        cv_array[j] = cv
        rate_array[j] = rate

    return cv_array, rate_array


@njit(cache=True)
def ode_integrate(y0, dt, Iapp_val, n_steps):
    V = y0[0]
    h = y0[1]
    n = y0[2]
    b = y0[3]
    z = y0[4]

    for i in range(n_steps):
        m_val = m_inf(V)
        h_inf_v = h_inf(V)
        n_inf_v = n_inf(V)
        a_val = a_inf(V)
        b_inf_v = b_inf(V)
        z_inf_v = z_inf(V)
        p_val = p_inf(V)
        tau_h_v = tau_h(V)
        tau_n_v = tau_n(V)

        INa = gNa * m_val**3 * h * (V - VNa)
        INaP = gNaP * p_val * (V - VNa)
        IKdr = gKdr * n**4 * (V - VK)
        IA = gA * a_val**3 * b * (V - VK)
        IM = gM * z * (V - VK)
        IL = gL * (V - VL)

        dV = (-INa - INaP - IKdr - IA - IM - IL + Iapp_val) / C_m
        dh = (h_inf_v - h) / tau_h_v
        dn = (n_inf_v - n) / tau_n_v
        db = (b_inf_v - b) / tau_b
        dz = (z_inf_v - z) / tau_z

        V = V + dV * dt
        h = h + dh * dt
        n = n + dn * dt
        b = b + db * dt
        z = z + dz * dt

    result = np.empty(5)
    result[0] = V
    result[1] = h
    result[2] = n
    result[3] = b
    result[4] = z
    return result


def get_deterministic_steady_state(Iapp, t_end=2000.0, dt=0.01):
    V_rest = -70.0
    h_rest = 1.0 / (1.0 + np.exp((V_rest + 45.0) / 7.0))
    n_rest = 1.0 / (1.0 + np.exp(-(V_rest + 35.0) / 10.0))
    b_rest = 1.0 / (1.0 + np.exp((V_rest + 80.0) / 6.0))
    z_rest = 1.0 / (1.0 + np.exp(-(V_rest + 39.0) / 5.0))

    y0 = np.array([V_rest, h_rest, n_rest, b_rest, z_rest])
    n_steps = int(t_end / dt)
    y_final = ode_integrate(y0, dt, Iapp, n_steps)

    print(f"  ODE {t_end:.0f}ms at Iapp={Iapp}: V={y_final[0]:.2f}, h={y_final[1]:.4f}, "
          f"n={y_final[2]:.4f}, b={y_final[3]:.4f}, z={y_final[4]:.4f}")
    return y_final
