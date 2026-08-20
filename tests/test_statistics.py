import numpy as np

from feller_neural_sde.statistics import compute_trial_stats


def test_burn_in_events_are_excluded():
    trace = np.full(1_000, -70.0)
    trace[[10, 200, 400]] = 10.0

    _, full_rate = compute_trial_stats(trace, dt=1.0, burn_in_steps=0)
    cv_after_burn, rate_after_burn = compute_trial_stats(
        trace,
        dt=1.0,
        burn_in_steps=500,
    )

    assert full_rate > 0.0
    assert rate_after_burn == 0.0
    assert np.isnan(cv_after_burn)
