"""Generate reproducible toy spike datasets saved in the project's raw `.npz` format.

中文说明
--------
这个脚本负责生成一批“适合 Spike CAM 研究”的 toy dataset，
并直接保存成和真实数据一致的数据契约：

- `raw_data`
- `spike_times`
- `spike_clusters`
- `fs`
- `n_channels`
- `duration_sec`

这样后面的整条 pipeline 可以完全复用，不需要为 toy 数据单独写特殊分支。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ensure_parent, json_ready, project_path


@dataclass
class ToyDatasetSpec:
    """Configuration for one synthetic dataset scenario."""

    name: str
    duration_sec: int
    num_units: int
    num_channels: int
    sampling_frequency: float = 30000.0
    firing_rate: float = 5.0
    average_peak_amplitude: float = -120.0
    extra_noise_std: float = 0.0
    drift_strength: float = 0.0
    drift_phase: float = 0.0
    description: str = ""
    seed: int = 42

    @property
    def npz_path(self) -> Path:
        return project_path("dataset", "toy", f"{self.name}.npz")

    @property
    def metadata_path(self) -> Path:
        return project_path("dataset", "toy", f"{self.name}.json")


TOY_SPECS: Dict[str, ToyDatasetSpec] = {
    "toy_easy_stable_u8_c8_60s": ToyDatasetSpec(
        name="toy_easy_stable_u8_c8_60s",
        duration_sec=60,
        num_units=8,
        num_channels=8,
        firing_rate=5.0,
        average_peak_amplitude=-120.0,
        extra_noise_std=0.0,
        drift_strength=0.0,
        description="Easy sanity-check toy dataset: 8 units, 8 channels, stable recording, no extra perturbation.",
        seed=11,
    ),
    "toy_dense_stable_u12_c16_75s": ToyDatasetSpec(
        name="toy_dense_stable_u12_c16_75s",
        duration_sec=75,
        num_units=12,
        num_channels=16,
        firing_rate=5.0,
        average_peak_amplitude=-115.0,
        extra_noise_std=2.0,
        drift_strength=0.0,
        description="Denser but still stable toy dataset: more units and channels, mild extra noise.",
        seed=22,
    ),
    "toy_drift_u12_c16_75s": ToyDatasetSpec(
        name="toy_drift_u12_c16_75s",
        duration_sec=75,
        num_units=12,
        num_channels=16,
        firing_rate=5.0,
        average_peak_amplitude=-115.0,
        extra_noise_std=4.0,
        drift_strength=0.28,
        drift_phase=0.8,
        description="Closed-set drift scenario: same scale as dense toy, but with extra noise and smooth channel-wise drift over time.",
        seed=33,
    ),
    "toy_open_u20_c24_90s": ToyDatasetSpec(
        name="toy_open_u20_c24_90s",
        duration_sec=90,
        num_units=20,
        num_channels=24,
        firing_rate=5.0,
        average_peak_amplitude=-110.0,
        extra_noise_std=5.5,
        drift_strength=0.22,
        drift_phase=1.4,
        description="Memory-limited open-set toy dataset: many units, more channels, moderate drift and extra noise.",
        seed=44,
    ),
}


def _collect_spikes(sorting) -> tuple[np.ndarray, np.ndarray]:
    """Collect all spike times and labels from one single-segment sorting object."""

    all_times: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for unit_id in sorting.unit_ids:
        spikes = np.asarray(sorting.get_unit_spike_train(unit_id, segment_index=0), dtype=np.int64)
        if spikes.size == 0:
            continue
        all_times.append(spikes)
        all_labels.append(np.full(spikes.shape[0], int(unit_id), dtype=np.int64))

    if not all_times:
        raise ValueError("Synthetic sorting contains no spikes")

    spike_times = np.concatenate(all_times, axis=0)
    spike_clusters = np.concatenate(all_labels, axis=0)
    order = np.argsort(spike_times, kind="mergesort")
    return spike_times[order], spike_clusters[order]


def _apply_extra_noise(raw_data: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0.0:
        return np.asarray(raw_data, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=float(std), size=np.asarray(raw_data).shape).astype(np.float32)
    return np.asarray(raw_data, dtype=np.float32) + noise


def _apply_channel_drift(raw_data: np.ndarray, strength: float, phase: float, rng: np.random.Generator) -> np.ndarray:
    """Apply a smooth time-varying, channel-wise gain drift.

    中文：
    这里故意做的是“连续的 channel gain 漂移”，而不是突然跳变。
    它会让后半段 spike 的空间 footprint 和前半段有系统差异，
    这样更容易观察动态更新是否真的有帮助。
    """

    if strength <= 0.0:
        return np.asarray(raw_data, dtype=np.float32)

    raw = np.asarray(raw_data, dtype=np.float32)
    num_samples, num_channels = raw.shape
    time = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)[:, None]
    base_pattern = rng.uniform(-1.0, 1.0, size=(1, num_channels)).astype(np.float32)
    harmonic = np.sin(np.linspace(0.0, 2.0 * np.pi, num_samples, dtype=np.float32)[:, None] + float(phase))
    gain = 1.0 + float(strength) * (0.65 * time + 0.35 * harmonic) * base_pattern
    return raw * gain.astype(np.float32)


def generate_one_dataset(spec: ToyDatasetSpec) -> Dict[str, object]:
    """Generate one toy dataset and save it to `dataset/toy/`."""

    import spikeinterface.full as si

    rng = np.random.default_rng(spec.seed)
    recording, sorting = si.toy_example(
        duration=int(spec.duration_sec),
        num_channels=int(spec.num_channels),
        num_units=int(spec.num_units),
        sampling_frequency=float(spec.sampling_frequency),
        num_segments=1,
        firing_rate=float(spec.firing_rate),
        average_peak_amplitude=float(spec.average_peak_amplitude),
        seed=int(spec.seed),
    )

    raw_data = np.asarray(recording.get_traces(segment_index=0), dtype=np.float32)
    raw_data = _apply_extra_noise(raw_data, spec.extra_noise_std, rng)
    raw_data = _apply_channel_drift(raw_data, spec.drift_strength, spec.drift_phase, rng)

    spike_times, spike_clusters = _collect_spikes(sorting)

    ensure_parent(spec.npz_path)
    np.savez_compressed(
        spec.npz_path,
        raw_data=raw_data.astype(np.float32),
        spike_times=spike_times.astype(np.int64),
        spike_clusters=spike_clusters.astype(np.int64),
        fs=np.asarray(float(spec.sampling_frequency), dtype=np.float32),
        n_channels=np.asarray(int(spec.num_channels), dtype=np.int64),
        duration_sec=np.asarray(float(spec.duration_sec), dtype=np.float32),
    )

    unique_labels, counts = np.unique(spike_clusters, return_counts=True)
    metadata = {
        **json_ready(asdict(spec)),
        "npz_path": str(spec.npz_path),
        "num_spikes": int(spike_times.shape[0]),
        "label_counts": {str(label): int(count) for label, count in zip(unique_labels, counts)},
        "mean_count_per_unit": float(np.mean(counts)),
        "std_count_per_unit": float(np.std(counts)),
    }
    ensure_parent(spec.metadata_path)
    spec.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def generate_named_datasets(names: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name in names:
        if name not in TOY_SPECS:
            raise KeyError(f"Unknown toy dataset name: {name}")
        rows.append(generate_one_dataset(TOY_SPECS[name]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy spike datasets for the Spike CAM platform.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Toy dataset name to generate. Can be passed multiple times. If omitted, generate all presets.",
    )
    args = parser.parse_args()

    names = args.dataset or list(TOY_SPECS.keys())
    rows = generate_named_datasets(names)
    for row in rows:
        print(f"[generated] {row['npz_path']}  spikes={row['num_spikes']}  units={row['num_units']}  channels={row['num_channels']}")


if __name__ == "__main__":
    main()
