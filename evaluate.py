"""Backward-compatible re-exports.

The experiment platform now uses:

- ``metrics.py`` for metric computation
- ``experiment_runner.py`` for running experiments

This module remains only as a thin compatibility layer.
"""

from experiment_runner import run_single_experiment
from metrics import (
    ResultBundle,
    compute_classification_metrics,
    compute_curve_bundle,
    compute_memory_metrics,
    compute_reject_metrics,
    compute_update_metrics,
)

__all__ = [
    "ResultBundle",
    "compute_classification_metrics",
    "compute_curve_bundle",
    "compute_memory_metrics",
    "compute_reject_metrics",
    "compute_update_metrics",
    "run_single_experiment",
]
