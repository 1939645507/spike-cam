"""Spike CAM experiment platform.

This package exposes the primary building blocks used by the project:

- configuration loading
- waveform / encoded dataset helpers
- encoder interfaces
- CAM core
- experiment runner
"""

from config import load_config
from dataio import EncodedDataset, WaveformDataset
from experiment_runner import run_experiment_suite

__all__ = [
    "EncodedDataset",
    "WaveformDataset",
    "load_config",
    "run_experiment_suite",
]
