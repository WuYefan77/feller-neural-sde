"""Reusable simulation tools for a five-dimensional neuronal SDE."""

from .analysis import ArrheniusFit, fit_arrhenius
from .model import (
    get_deterministic_steady_state,
    simulate_feller,
    simulate_gaussian,
    simulate_matched_gaussian,
)
from .statistics import compute_batch_stats, compute_trial_stats

__all__ = [
    "ArrheniusFit",
    "compute_batch_stats",
    "compute_trial_stats",
    "fit_arrhenius",
    "get_deterministic_steady_state",
    "simulate_feller",
    "simulate_gaussian",
    "simulate_matched_gaussian",
]

__version__ = "0.1.0"
