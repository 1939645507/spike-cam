"""Data loading, waveform extraction, and dataset helper utilities.

This module defines the "data contract" used by the rest of the project:

- raw spike stream -> :class:`WaveformDataset`
- encoded bit stream -> :class:`EncodedDataset`

The main design goal is to preserve spike order and spike times so that
chronological / online CAM evaluation remains the default behavior.

中文说明
--------
这个文件负责把“原始数据”整理成后面所有模块都能稳定使用的数据格式。

最重要的两个数据对象是：

- ``WaveformDataset``: 原始 spike stream 提取出的 waveform 数据
- ``EncodedDataset``: waveform 经过 encoder 后得到的 bits 数据

这里有两个核心原则：

1. 尽量保留 ``spike_times`` 和原始顺序
2. 先把数据契约定清楚，后面 encoder / CAM / metrics 才能彻底解耦
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import (
    DatasetConfig,
    EvaluationConfig,
    PreprocessConfig,
    SubsetConfig,
    WaveformConfig,
    ensure_parent,
    json_ready,
    resolve_path,
)


REJECT_LABEL = -1


@dataclass
class WaveformDataset:
    """Waveforms extracted from a raw spike stream.

    Attributes
    ----------
    waveforms:
        Array of shape ``(N, feature_dim)``.
    labels:
        Ground-truth neuron / cluster ids for each spike.
    spike_times:
        Spike timestamps aligned with ``waveforms``.
    source_indices:
        Indices into the original ``spike_times`` / ``spike_clusters`` arrays.
    meta:
        Small JSON-serializable metadata dictionary.

    中文：
    这是“还没编码”的数据对象，包含 waveform、label 和 spike time。
    """

    waveforms: np.ndarray
    labels: np.ndarray
    spike_times: np.ndarray
    source_indices: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedDataset:
    """Cached encoded representation used by CAM experiments.

    中文：这是 CAM 实验真正直接消费的数据格式。
    你以后做大部分 CAM 对比实验时，通常都只需要这一个对象。
    """

    bits: np.ndarray
    labels: np.ndarray
    spike_times: np.ndarray
    source_indices: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def bit_width(self) -> int:
        """Return the number of bits per spike code.

        中文：每条 spike code 的 bit 数。
        """

        if self.bits.ndim != 2:
            raise ValueError(f"Encoded bits must be 2D, got {self.bits.shape}")
        return int(self.bits.shape[1])

    @property
    def num_spikes(self) -> int:
        """Return the number of encoded spikes.

        中文：编码后的 spike 总数。
        """

        return int(self.bits.shape[0])

    def save_npz(self, path: str | Path) -> Path:
        """Save the encoded dataset to disk.

        中文：把 encoded dataset 缓存到磁盘，避免每次都重新跑 encoder。
        """

        out_path = Path(path)
        ensure_parent(out_path)
        np.savez_compressed(
            out_path,
            bits=self.bits.astype(np.uint8),
            labels=self.labels.astype(np.int64),
            spike_times=self.spike_times.astype(np.int64),
            source_indices=self.source_indices.astype(np.int64),
            meta_json=json.dumps(json_ready(self.meta), sort_keys=True),
        )
        return out_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "EncodedDataset":
        """Load a previously cached encoded dataset.

        中文：读取已经缓存好的 bits 数据。
        """

        pack = np.load(Path(path), allow_pickle=False)
        meta_json = str(pack["meta_json"])
        return cls(
            bits=np.asarray(pack["bits"]).astype(np.uint8),
            labels=np.asarray(pack["labels"]).astype(np.int64),
            spike_times=np.asarray(pack["spike_times"]).astype(np.int64),
            source_indices=np.asarray(pack["source_indices"]).astype(np.int64),
            meta=json.loads(meta_json),
        )


def _bit_column_sort_key(column_name: str, prefix: str) -> tuple[int, str]:
    """Sort columns like bit0, bit1, ..., bit31 in numeric order."""

    suffix = str(column_name)[len(prefix) :]
    try:
        return int(suffix), str(column_name)
    except ValueError:
        return 10**9, str(column_name)


def load_encoded_csv_dataset(
    csv_path: str | Path,
    *,
    label_col: str = "unit_id",
    bit_prefix: str = "bit",
    spike_time_col: Optional[str] = None,
) -> EncodedDataset:
    """Load a previously exported encoded CSV into :class:`EncodedDataset`.

    中文：
    这个函数主要是为 `previous work/spike_pca_dataset.csv` 这一类旧资产准备的。

    旧 CSV 往往只保存：

    - `bit0`, `bit1`, ...
    - `unit_id`

    没有显式的 `spike_times`。这种情况下这里会退化为“按行号生成伪时间”，
    方便继续复用当前 CAM / metrics / report 框架。
    """

    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if label_col not in fieldnames:
            raise KeyError(f"Encoded CSV is missing label column: {label_col}")

        bit_columns = sorted(
            [name for name in fieldnames if str(name).startswith(bit_prefix)],
            key=lambda name: _bit_column_sort_key(name, bit_prefix),
        )
        if not bit_columns:
            raise KeyError(f"Encoded CSV has no bit columns with prefix '{bit_prefix}'")

        bit_rows: List[List[int]] = []
        labels: List[int] = []
        spike_times: List[int] = []
        source_indices: List[int] = []

        for row_index, row in enumerate(reader):
            labels.append(int(row[label_col]))
            bit_rows.append([int(float(row[col])) for col in bit_columns])
            if spike_time_col and spike_time_col in row and row[spike_time_col] != "":
                spike_times.append(int(float(row[spike_time_col])))
            else:
                spike_times.append(int(row_index))
            source_indices.append(int(row_index))

    label_array = np.asarray(labels, dtype=np.int64)
    unique_labels, counts = np.unique(label_array, return_counts=True)
    return EncodedDataset(
        bits=np.asarray(bit_rows, dtype=np.uint8),
        labels=label_array,
        spike_times=np.asarray(spike_times, dtype=np.int64),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        meta={
            "source_csv": str(csv_path),
            "label_col": label_col,
            "bit_prefix": bit_prefix,
            "spike_time_col": spike_time_col,
            "num_units": int(unique_labels.size),
            "label_counts": {str(label): int(count) for label, count in zip(unique_labels, counts)},
            "order_warning": "spike_times are synthetic row indices when no explicit spike_time_col is provided",
        },
    )


def _align_waveform(waveform: np.ndarray, mode: str, center_index: int) -> np.ndarray:
    """Shift a 1D waveform so its salient peak lands at ``center_index``.

    中文：把 spike waveform 的关键峰值对齐到统一中心位置。
    """

    if mode == "none":
        return waveform
    if mode == "min":
        peak_index = int(np.argmin(waveform))
    elif mode == "max":
        peak_index = int(np.argmax(waveform))
    else:
        raise ValueError(f"Unknown align_mode: {mode}")

    shift = center_index - peak_index
    if shift == 0:
        return waveform

    aligned = np.empty_like(waveform)
    if shift > 0:
        aligned[:shift] = waveform[0]
        aligned[shift:] = waveform[:-shift]
    else:
        shift = -shift
        aligned[-shift:] = waveform[-1]
        aligned[:-shift] = waveform[shift:]
    return aligned


def _alignment_profile(window: np.ndarray, mode: str, alignment_reference: str, anchor_channel: Optional[int] = None) -> np.ndarray:
    """Return a 1D alignment profile over time for one 2D spike window.

    中文：
    对多通道数据，默认用“全通道整体活动”找峰值位置，
    比先随便挑一个 channel 再对齐更稳一些。
    """

    if alignment_reference == "global_activity":
        if mode == "min":
            return np.max(-window, axis=1)
        if mode == "max":
            return np.max(window, axis=1)
        raise ValueError(f"Unknown align_mode: {mode}")
    if alignment_reference == "anchor_channel":
        if anchor_channel is None:
            raise ValueError("anchor_channel is required when alignment_reference='anchor_channel'")
        return np.asarray(window[:, anchor_channel], dtype=np.float32)
    raise ValueError(f"Unknown alignment_reference: {alignment_reference}")


def _align_window(
    window: np.ndarray,
    mode: str,
    center_index: int,
    alignment_reference: str,
    anchor_channel: Optional[int] = None,
) -> np.ndarray:
    """Shift a 2D spike window according to a chosen alignment reference.

    中文：
    多通道 waveform 不能每个通道各自对齐，否则会破坏跨通道时序关系。
    这里是把整块窗口作为一个整体平移。
    """

    if mode == "none":
        return window

    profile = _alignment_profile(window, mode, alignment_reference, anchor_channel=anchor_channel)
    if alignment_reference == "anchor_channel" and anchor_channel is not None:
        if mode == "min":
            peak_index = int(np.argmin(profile))
        elif mode == "max":
            peak_index = int(np.argmax(profile))
        else:
            raise ValueError(f"Unknown align_mode: {mode}")
    else:
        peak_index = int(np.argmax(profile))

    shift = center_index - peak_index
    if shift == 0:
        return window

    aligned = np.empty_like(window)
    if shift > 0:
        aligned[:shift, :] = window[[0], :]
        aligned[shift:, :] = window[:-shift, :]
    else:
        shift = -shift
        aligned[-shift:, :] = window[[-1], :]
        aligned[:-shift, :] = window[shift:, :]
    return aligned


def _channel_strengths(window: np.ndarray, cfg: WaveformConfig) -> np.ndarray:
    """Return one scalar strength per channel.

    中文：
    这里不再看整段窗口，而是默认只看 spike 中心附近的一小段局部窗口。
    这样可以尽量避免把时域上的无关噪声峰值当成“主通道”。
    """

    radius = max(0, int(cfg.selection_radius))
    start = max(0, int(cfg.center_index) - radius)
    stop = min(int(window.shape[0]), int(cfg.center_index) + radius + 1)
    local_window = window[start:stop, :]
    return np.max(np.abs(local_window), axis=0)


def _sort_channel_indices(indices: np.ndarray, strengths: np.ndarray, order_mode: str) -> np.ndarray:
    """Sort selected channel indices for stable flattening."""

    indices = np.asarray(indices, dtype=np.int64)
    if order_mode == "by_index":
        return np.sort(indices, kind="mergesort")
    if order_mode == "by_strength":
        local_strengths = strengths[indices]
        order = np.lexsort((indices, -local_strengths))
        return indices[order]
    raise ValueError(f"Unknown channel_order: {order_mode}")


def _select_channel_indices(window: np.ndarray, cfg: WaveformConfig) -> np.ndarray:
    """Select one or more channels from a ``(time, channel)`` spike window.

    中文：
    旧版默认只取单个 max-abs channel，这会把空间 footprint 丢得很厉害。
    新版默认支持 top-k channels，再 flatten 成一个更稳定的特征向量。
    """

    strengths = _channel_strengths(window, cfg)
    num_channels = int(window.shape[1])

    if cfg.channel_selection == "fixed":
        indices = np.asarray([int(cfg.fixed_channel)], dtype=np.int64)
    elif cfg.channel_selection == "max_abs":
        indices = np.asarray([int(np.argmax(strengths))], dtype=np.int64)
    elif cfg.channel_selection == "topk_max_abs":
        topk = max(1, min(int(cfg.topk_channels), num_channels))
        order = np.lexsort((np.arange(num_channels, dtype=np.int64), -strengths))
        indices = order[:topk]
    elif cfg.channel_selection == "all":
        indices = np.arange(num_channels, dtype=np.int64)
    else:
        raise ValueError(f"Unknown channel_selection: {cfg.channel_selection}")

    return _sort_channel_indices(indices, strengths, cfg.channel_order)


def _selected_channel_count(num_channels: int, cfg: WaveformConfig) -> int:
    """Return the fixed number of channels selected per spike."""

    if cfg.channel_selection in {"fixed", "max_abs"}:
        return 1
    if cfg.channel_selection == "topk_max_abs":
        return max(1, min(int(cfg.topk_channels), int(num_channels)))
    if cfg.channel_selection == "all":
        return int(num_channels)
    raise ValueError(f"Unknown channel_selection: {cfg.channel_selection}")


def _flatten_window(window: np.ndarray, channel_indices: np.ndarray, flatten_order: str) -> np.ndarray:
    """Flatten a selected multi-channel waveform window into 1D features."""

    selected = np.asarray(window[:, channel_indices], dtype=np.float32)
    if selected.ndim != 2:
        raise ValueError(f"Selected window must be 2D, got shape {selected.shape}")
    if flatten_order == "time_major":
        return selected.reshape(-1).astype(np.float32)
    if flatten_order == "channel_major":
        return selected.T.reshape(-1).astype(np.float32)
    raise ValueError(f"Unknown flatten_order: {flatten_order}")


def _load_sampling_rate(pack: np.lib.npyio.NpzFile) -> float:
    """Load the sampling rate from a raw dataset pack."""

    if "fs" not in pack:
        raise KeyError("Raw dataset is missing 'fs', required for signal preprocessing")
    return float(np.asarray(pack["fs"]).item())


def _apply_bandpass(recording: np.ndarray, sampling_rate_hz: float, cfg: PreprocessConfig) -> np.ndarray:
    """Apply a Butterworth bandpass filter to the continuous recording."""

    from scipy.signal import butter, sosfiltfilt

    nyquist = 0.5 * float(sampling_rate_hz)
    low = float(cfg.bandpass_low_hz) / nyquist
    high = float(cfg.bandpass_high_hz) / nyquist
    if not 0.0 < low < high < 1.0:
        raise ValueError(
            "Invalid bandpass frequencies: "
            f"low={cfg.bandpass_low_hz}, high={cfg.bandpass_high_hz}, fs={sampling_rate_hz}"
        )
    sos = butter(int(cfg.bandpass_order), [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, np.asarray(recording, dtype=np.float32), axis=0)
    return np.asarray(filtered, dtype=np.float32)


def _apply_common_reference(recording: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Apply common median or mean reference across channels."""

    if cfg.common_reference_mode == "median":
        reference = np.median(recording, axis=1, keepdims=True)
    elif cfg.common_reference_mode == "mean":
        reference = np.mean(recording, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown common_reference_mode: {cfg.common_reference_mode}")
    return np.asarray(recording - reference, dtype=np.float32)


def _fit_whitening_matrix(recording: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a channel-whitening transform from sampled frames.

    中文：
    whitening 这里做的是“通道维”的 decorrelation，
    不是对每条 waveform feature 单独做标准化。
    """

    num_frames = int(recording.shape[0])
    sample_count = max(1, min(int(cfg.whitening_num_samples), num_frames))
    sample_indices = np.linspace(0, num_frames - 1, num=sample_count, dtype=np.int64)
    sample = np.asarray(recording[sample_indices], dtype=np.float32)
    mean = sample.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = sample - mean
    covariance = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    inv_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, float(cfg.whitening_eps)))
    whitening_matrix = (eigenvectors * inv_sqrt) @ eigenvectors.T
    return mean.astype(np.float32), whitening_matrix.astype(np.float32)


def _apply_whitening(recording: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply channel whitening chunk by chunk."""

    mean, whitening_matrix = _fit_whitening_matrix(recording, cfg)
    whitened = np.empty_like(recording, dtype=np.float32)
    chunk_size = max(1, int(cfg.whitening_chunk_size))
    for start in range(0, int(recording.shape[0]), chunk_size):
        stop = min(int(recording.shape[0]), start + chunk_size)
        whitened[start:stop] = (recording[start:stop] - mean) @ whitening_matrix.T
    whitening_meta = {
        "num_samples": int(min(int(cfg.whitening_num_samples), int(recording.shape[0]))),
        "chunk_size": int(chunk_size),
        "eps": float(cfg.whitening_eps),
    }
    return whitened, whitening_meta


def preprocess_recording(raw_data: np.ndarray, sampling_rate_hz: float, cfg: PreprocessConfig) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply configurable preprocessing to a continuous recording.

    中文：
    这是新版 dataio 的关键增强点。

    顺序固定为：

    1. bandpass
    2. common reference
    3. optional whitening
    """

    recording = np.asarray(raw_data, dtype=np.float32)
    applied_steps: List[str] = []
    whitening_meta: Dict[str, Any] | None = None

    if cfg.bandpass_enable:
        recording = _apply_bandpass(recording, sampling_rate_hz, cfg)
        applied_steps.append("bandpass")
    if cfg.common_reference_enable:
        recording = _apply_common_reference(recording, cfg)
        applied_steps.append(f"common_reference_{cfg.common_reference_mode}")
    if cfg.whitening_enable:
        recording, whitening_meta = _apply_whitening(recording, cfg)
        applied_steps.append("whitening")

    return recording.astype(np.float32), {
        "applied_steps": applied_steps,
        "sampling_rate_hz": float(sampling_rate_hz),
        "bandpass_enable": bool(cfg.bandpass_enable),
        "bandpass_low_hz": float(cfg.bandpass_low_hz),
        "bandpass_high_hz": float(cfg.bandpass_high_hz),
        "bandpass_order": int(cfg.bandpass_order),
        "common_reference_enable": bool(cfg.common_reference_enable),
        "common_reference_mode": cfg.common_reference_mode,
        "whitening_enable": bool(cfg.whitening_enable),
        "whitening_meta": whitening_meta,
    }


def _label_order_by_count(unique_labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return labels ordered by descending count, then ascending label id.

    中文：
    当多个类计数相同时，也保持稳定、可复现的顺序，
    避免 top-k 因为排序实现细节而漂移。
    """

    unique_labels = np.asarray(unique_labels, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    return np.lexsort((unique_labels, -counts))


def select_labels_by_counts(unique_labels: np.ndarray, counts: np.ndarray, subset_cfg: SubsetConfig) -> np.ndarray:
    """Apply a subset rule to explicit ``(label, count)`` arrays.

    中文：
    有些时候我们并不是直接拿一串 labels 来选 top-k，
    而是已经有了某个阶段统计好的计数，例如：

    - waveform 下采样之前的原始类计数
    - warmup 段里的类计数
    - encoded stream 的类计数
    """

    unique_labels = np.asarray(unique_labels, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if unique_labels.shape != counts.shape:
        raise ValueError("unique_labels and counts must have the same shape")

    order = _label_order_by_count(unique_labels, counts)

    if subset_cfg.mode == "all":
        return unique_labels[order]
    if subset_cfg.mode == "topk":
        if subset_cfg.topk is None:
            raise ValueError("subset.mode='topk' requires subset.topk")
        keep = order[: int(subset_cfg.topk)]
        return unique_labels[keep]
    if subset_cfg.mode == "min_count":
        if subset_cfg.min_count is None:
            raise ValueError("subset.mode='min_count' requires subset.min_count")
        keep = order[counts[order] >= int(subset_cfg.min_count)]
        return unique_labels[keep]
    raise ValueError(f"Unknown subset mode: {subset_cfg.mode}")


def subset_by_label_rule(labels: np.ndarray, subset_cfg: SubsetConfig) -> np.ndarray:
    """Return the label ids kept by a subset rule.

    Parameters
    ----------
    labels:
        Label array after any boundary filtering.
    subset_cfg:
        One of:
        - ``all``
        - ``topk``
        - ``min_count``

    中文：
    真实数据常常是长尾分布，所以主实验经常不会直接用 all units。
    """

    labels = np.asarray(labels).astype(np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    return select_labels_by_counts(unique_labels, counts, subset_cfg)


def subset_by_count_map(label_counts: Dict[str, int] | Dict[int, int], subset_cfg: SubsetConfig) -> np.ndarray:
    """Apply a subset rule to a JSON-like ``label -> count`` mapping.

    中文：
    这个函数主要给 memory selection 用，
    因为我们经常会把“下采样前的类计数”存在 metadata 里。
    """

    if not label_counts:
        return np.asarray([], dtype=np.int64)
    labels = np.asarray([int(label) for label in label_counts.keys()], dtype=np.int64)
    counts = np.asarray([int(count) for count in label_counts.values()], dtype=np.int64)
    return select_labels_by_counts(labels, counts, subset_cfg)


def chronological_split(
    spike_times: np.ndarray,
    warmup_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a time-sorted spike stream into warmup and online parts.

    中文：这是主实验默认使用的切分方式。
    前一段时间用于初始化模板，后一段时间按在线流式方式评估。
    """

    spike_times = np.asarray(spike_times).astype(np.int64)
    if spike_times.ndim != 1:
        raise ValueError("spike_times must be 1D for chronological_split")
    if not 0.0 < warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in (0, 1)")

    order = np.argsort(spike_times, kind="mergesort")
    split_idx = max(1, min(len(order) - 1, int(np.floor(len(order) * warmup_ratio))))
    return order[:split_idx], order[split_idx:]


def random_split(
    num_samples: int,
    train_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optional non-chronological split for baseline comparisons only.

    中文：仅作为附加 baseline 使用，不是主评估方式。
    """

    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = np.arange(num_samples)
    rng.shuffle(order)
    split_idx = max(1, min(num_samples - 1, int(np.floor(num_samples * train_frac))))
    return order[:split_idx], order[split_idx:]


def _evenly_spaced_positions(length: int, target_size: int) -> np.ndarray:
    """Pick approximately uniform positions without breaking determinism."""

    if target_size >= length:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, num=target_size, dtype=int)).astype(np.int64)


def _downsample_indices(
    labels: np.ndarray,
    spike_times: np.ndarray,
    source_indices: np.ndarray,
    max_total_spikes: Optional[int],
    max_spikes_per_unit: Optional[int],
    selection_mode: str,
) -> np.ndarray:
    """Return a mask of rows kept after optional downsampling."""

    num_samples = labels.shape[0]
    keep_mask = np.ones(num_samples, dtype=bool)

    if max_spikes_per_unit is not None:
        keep_mask[:] = False
        for label in np.unique(labels):
            local = np.flatnonzero(labels == label)
            if selection_mode == "first":
                chosen = local[: int(max_spikes_per_unit)]
            else:
                positions = _evenly_spaced_positions(local.size, int(max_spikes_per_unit))
                chosen = local[positions]
            keep_mask[chosen] = True

    kept_indices = np.flatnonzero(keep_mask)
    if max_total_spikes is not None and kept_indices.size > int(max_total_spikes):
        if selection_mode == "first":
            kept_indices = kept_indices[: int(max_total_spikes)]
        else:
            positions = _evenly_spaced_positions(kept_indices.size, int(max_total_spikes))
            kept_indices = kept_indices[positions]

    final_mask = np.zeros(num_samples, dtype=bool)
    final_mask[kept_indices] = True
    return final_mask


def load_label_counts_before_sampling(dataset_cfg: DatasetConfig) -> Dict[str, int]:
    """Return label counts after boundary filtering and stream subseting.

    中文：
    这个函数不会提 waveform，只会快速统计：

    - 经过边界过滤后
    - 经过 `dataset.subset` 后
    - 但还没做 waveform downsampling 之前

    各个类还剩多少 spike。

    它主要用于 `cam.memory_subset.selection_source = pre_sampling`。
    """

    npz_path = resolve_path(dataset_cfg.npz_path)
    pack = np.load(npz_path, allow_pickle=False)

    required_keys = {"raw_data", "spike_times", "spike_clusters"}
    missing = sorted(required_keys - set(pack.files))
    if missing:
        raise KeyError(f"Raw dataset is missing keys: {missing}")

    raw_data = np.asarray(pack["raw_data"])
    spike_times = np.asarray(pack["spike_times"]).astype(np.int64)
    spike_clusters = np.asarray(pack["spike_clusters"]).astype(np.int64)

    waveform_cfg = dataset_cfg.waveform
    waveform_length = int(waveform_cfg.waveform_length)
    center_index = int(waveform_cfg.center_index)
    pre = center_index
    post = waveform_length - center_index

    valid_mask = (spike_times - pre >= 0) & (spike_times + post <= raw_data.shape[0])
    spike_times = spike_times[valid_mask]
    spike_clusters = spike_clusters[valid_mask]

    kept_labels = subset_by_label_rule(spike_clusters, dataset_cfg.subset)
    subset_mask = np.isin(spike_clusters, kept_labels)
    spike_times = spike_times[subset_mask]
    spike_clusters = spike_clusters[subset_mask]

    unique_labels, counts = np.unique(spike_clusters, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique_labels, counts)}


def load_waveform_dataset(dataset_cfg: DatasetConfig) -> WaveformDataset:
    """Load a raw ``.npz`` file and extract aligned spike waveforms.

    The function preserves chronological order by default. If sampling is
    applied, selection remains deterministic and approximately uniform in time.

    中文：
    这是 raw `.npz` -> `WaveformDataset` 的主入口。
    它会依次做：

    1. 读取 raw_data / spike_times / spike_clusters
    2. 去掉越界 spikes
    3. 按 subset 规则筛选 units
    4. 保时间顺序
    5. 可选下采样
    6. 提取每个 spike 的 waveform
    """

    npz_path = resolve_path(dataset_cfg.npz_path)
    pack = np.load(npz_path, allow_pickle=False)

    required_keys = {"raw_data", "spike_times", "spike_clusters"}
    missing = sorted(required_keys - set(pack.files))
    if missing:
        raise KeyError(f"Raw dataset is missing keys: {missing}")

    raw_data = np.asarray(pack["raw_data"])
    spike_times = np.asarray(pack["spike_times"]).astype(np.int64)
    spike_clusters = np.asarray(pack["spike_clusters"]).astype(np.int64)
    source_indices = np.arange(spike_times.shape[0], dtype=np.int64)

    sampling_rate_hz = _load_sampling_rate(pack) if (
        dataset_cfg.preprocess.bandpass_enable or dataset_cfg.preprocess.whitening_enable
    ) else float(np.asarray(pack["fs"]).item()) if "fs" in pack else None
    if sampling_rate_hz is not None:
        raw_data, preprocess_meta = preprocess_recording(raw_data, float(sampling_rate_hz), dataset_cfg.preprocess)
    else:
        raw_data = np.asarray(raw_data, dtype=np.float32)
        preprocess_meta = {
            "applied_steps": [],
            "sampling_rate_hz": None,
            "bandpass_enable": False,
            "common_reference_enable": False,
            "whitening_enable": False,
        }

    waveform_cfg = dataset_cfg.waveform
    waveform_length = int(waveform_cfg.waveform_length)
    center_index = int(waveform_cfg.center_index)
    pre = center_index
    post = waveform_length - center_index

    # 中文：无法完整截出 waveform 窗口的 spike 直接丢掉。
    # 这样后续每条 waveform 长度都一致。
    valid_mask = (spike_times - pre >= 0) & (spike_times + post <= raw_data.shape[0])
    spike_times = spike_times[valid_mask]
    spike_clusters = spike_clusters[valid_mask]
    source_indices = source_indices[valid_mask]

    kept_labels = subset_by_label_rule(spike_clusters, dataset_cfg.subset)
    subset_mask = np.isin(spike_clusters, kept_labels)
    spike_times = spike_times[subset_mask]
    spike_clusters = spike_clusters[subset_mask]
    source_indices = source_indices[subset_mask]

    # 中文：主实验默认强制按时间排序，保证后面 online evaluation 正确。
    if dataset_cfg.sort_by_time:
        order = np.argsort(spike_times, kind="mergesort")
        spike_times = spike_times[order]
        spike_clusters = spike_clusters[order]
        source_indices = source_indices[order]

    pre_sampling_labels = spike_clusters.copy()
    pre_sampling_unique, pre_sampling_counts = np.unique(pre_sampling_labels, return_counts=True)

    sampling_cfg = dataset_cfg.sampling
    keep_mask = _downsample_indices(
        labels=spike_clusters,
        spike_times=spike_times,
        source_indices=source_indices,
        max_total_spikes=sampling_cfg.max_total_spikes,
        max_spikes_per_unit=sampling_cfg.max_spikes_per_unit,
        selection_mode=sampling_cfg.selection_mode,
    )
    spike_times = spike_times[keep_mask]
    spike_clusters = spike_clusters[keep_mask]
    source_indices = source_indices[keep_mask]

    # 中文：下采样之后再排一次，防止采样破坏时间顺序。
    if dataset_cfg.sort_by_time:
        order = np.argsort(spike_times, kind="mergesort")
        spike_times = spike_times[order]
        spike_clusters = spike_clusters[order]
        source_indices = source_indices[order]

    selected_channel_count = _selected_channel_count(int(raw_data.shape[1]), waveform_cfg)
    feature_dim = int(waveform_length * selected_channel_count)
    waveforms = np.empty((spike_times.shape[0], feature_dim), dtype=np.float32)
    for i, spike_time in enumerate(spike_times):
        window = raw_data[spike_time - pre : spike_time + post, :]
        aligned_window = _align_window(
            window,
            waveform_cfg.align_mode,
            waveform_cfg.center_index,
            waveform_cfg.alignment_reference,
        )
        selected_channels = _select_channel_indices(aligned_window, waveform_cfg)
        waveforms[i] = _flatten_window(aligned_window, selected_channels, waveform_cfg.flatten_order)

    unique_labels, counts = np.unique(spike_clusters, return_counts=True)
    meta = {
        "source_npz": str(npz_path),
        "waveform_length": waveform_length,
        "feature_dim": int(feature_dim),
        "center_index": center_index,
        "align_mode": waveform_cfg.align_mode,
        "alignment_reference": waveform_cfg.alignment_reference,
        "channel_selection": waveform_cfg.channel_selection,
        "topk_channels": int(waveform_cfg.topk_channels),
        "selection_radius": int(waveform_cfg.selection_radius),
        "channel_order": waveform_cfg.channel_order,
        "flatten_order": waveform_cfg.flatten_order,
        "selected_channel_count": int(selected_channel_count),
        "subset_mode": dataset_cfg.subset.mode,
        "num_units": int(unique_labels.size),
        "label_counts": {str(label): int(count) for label, count in zip(unique_labels, counts)},
        "label_counts_before_sampling": {
            str(label): int(count) for label, count in zip(pre_sampling_unique, pre_sampling_counts)
        },
        "preprocess": preprocess_meta,
    }
    if "fs" in pack:
        meta["sampling_rate_hz"] = float(np.asarray(pack["fs"]).item())
    if "duration_sec" in pack:
        meta["duration_sec"] = float(np.asarray(pack["duration_sec"]).item())

    return WaveformDataset(
        waveforms=waveforms,
        labels=spike_clusters,
        spike_times=spike_times,
        source_indices=source_indices,
        meta=meta,
    )


__all__ = [
    "EncodedDataset",
    "REJECT_LABEL",
    "WaveformDataset",
    "chronological_split",
    "load_encoded_csv_dataset",
    "load_waveform_dataset",
    "load_label_counts_before_sampling",
    "random_split",
    "subset_by_label_rule",
    "subset_by_count_map",
    "select_labels_by_counts",
]
