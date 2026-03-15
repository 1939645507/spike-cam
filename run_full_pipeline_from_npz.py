"""Deprecated wrapper for the new config-driven experiment runner.

Use ``python run_experiment.py --config configs/baseline_top10_ae16.json``
for the new workflow.
"""

from __future__ import annotations

from config import load_config
from experiment_runner import run_experiment_suite


def main() -> None:
    """Run the default baseline config."""

    suite = load_config("configs/baseline_top10_ae16.json")
    run_experiment_suite(suite)


if __name__ == "__main__":
    main()
