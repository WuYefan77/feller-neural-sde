import numpy as np

from feller_neural_sde.model import simulate_feller


def test_conductance_is_an_explicit_solver_parameter():
    state = np.array([-60.0, 0.5, 0.5, 0.5, 0.5])
    noise = np.zeros((1, 4))

    low = simulate_feller(state, noise, 0.01, 0.0, 0.39, g_m=0.5)
    high = simulate_feller(state, noise, 0.01, 0.0, 0.39, g_m=1.5)

    assert not np.array_equal(low, high)
