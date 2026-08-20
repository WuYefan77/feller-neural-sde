import numpy as np

from feller_neural_sde.analysis import fit_arrhenius


def test_arrhenius_fit_ignores_nonpositive_rates():
    sigma = np.array([0.02, 0.03, 0.04, 0.05, 0.2])
    barrier = 0.001
    rate = np.exp(2.0 - barrier / sigma**2)
    rate[0] = 0.0

    result = fit_arrhenius(sigma, rate, max_sigma=0.1)

    assert len(result.sigma) == 3
    assert np.isclose(result.barrier, barrier)
    assert np.isclose(result.r_squared, 1.0)
