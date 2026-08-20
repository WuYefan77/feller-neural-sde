"""Small analysis utilities shared by the experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ArrheniusFit:
    """Result of an unweighted low-noise Arrhenius regression."""

    barrier: float
    intercept: float
    r_squared: float
    sigma: np.ndarray
    rate: np.ndarray


def fit_arrhenius(sigmas, rates, *, max_sigma: float = 0.1) -> ArrheniusFit:
    """Fit ``log(rate)`` against ``1 / sigma**2`` on valid low-noise points."""
    sigma = np.asarray(sigmas, dtype=float)
    rate = np.asarray(rates, dtype=float)
    if sigma.ndim != 1 or rate.ndim != 1 or sigma.shape != rate.shape:
        raise ValueError("sigmas and rates must be equal-length vectors")
    if not np.isfinite(max_sigma) or max_sigma <= 0.0:
        raise ValueError("max_sigma must be finite and positive")

    valid = (
        np.isfinite(sigma)
        & np.isfinite(rate)
        & (sigma > 0.0)
        & (sigma <= max_sigma)
        & (rate > 0.0)
    )
    selected_sigma = sigma[valid]
    selected_rate = rate[valid]
    if len(selected_sigma) < 3:
        raise ValueError("at least three finite positive low-noise rates are required")

    x_values = 1.0 / selected_sigma**2
    y_values = np.log(selected_rate)
    regression = stats.linregress(x_values, y_values)
    return ArrheniusFit(
        barrier=float(-regression.slope),
        intercept=float(regression.intercept),
        r_squared=float(regression.rvalue**2),
        sigma=selected_sigma,
        rate=selected_rate,
    )
