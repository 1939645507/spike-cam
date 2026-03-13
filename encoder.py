from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import os
import sys
import numpy as np


@dataclass
class WaveformExtractConfig:
    """
    从 raw voltage + spike_times 提取波形的配置。
    """

    waveform_len: int = 79           # 每个 spike 的采样点数
    center_index: int = 39           # 对齐到的中心点索引
    align_mode: str = "max"          # "none" | "max" | "min"
    channel_select: str = "max_abs"  # "max_abs" | "fixed"
    fixed_channel: int = 0


def _align_1d(wf: np.ndarray, mode: str, center_index: int) -> np.ndarray:
    if mode == "none":
        return wf
    if mode == "max":
        peak = int(np.argmax(wf))
    elif mode == "min":
        peak = int(np.argmin(wf))
    else:
        raise ValueError(f"未知对齐模式: {mode}")

    shift = center_index - peak
    if shift == 0:
        return wf
    out = np.empty_like(wf)
    if shift > 0:
        out[:shift] = wf[0]
        out[shift:] = wf[:-shift]
    else:
        s = -shift
        out[-s:] = wf[-1]
        out[:-s] = wf[s:]
    return out


def extract_waveforms_from_npz(
    npz_path: str,
    cfg: WaveformExtractConfig,
    *,
    max_total_spikes: int | None = None,
    max_spikes_per_unit: int | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从类似 my_validation_subset_*.npz 的文件中抽取 spike 波形。

    约定 npz 里至少包含：
    - raw_data:      (samples, channels)
    - spike_times:   (N_spikes,)
    - spike_clusters:(N_spikes,)

    返回：
    - waveforms: (N, L) float32
    - labels:    (N,)   int64  （来自 spike_clusters）
    """
    pack = np.load(npz_path)
    raw = pack["raw_data"]
    spike_times = np.asarray(pack["spike_times"]).astype(np.int64)
    spike_clusters = np.asarray(pack["spike_clusters"]).astype(np.int64)

    n_samples, n_channels = raw.shape
    L = int(cfg.waveform_len)
    pre = int(cfg.center_index)
    post = L - pre

    # 过滤越界 spike
    valid = (spike_times - pre >= 0) & (spike_times + post <= n_samples)
    spike_times = spike_times[valid]
    spike_clusters = spike_clusters[valid]

    rng = np.random.default_rng(seed)

    # 控制每个 unit 的最大样本数
    if max_spikes_per_unit is not None:
        unit_ids = np.unique(spike_clusters)
        indices_all = []
        for u in unit_ids:
            idx = np.flatnonzero(spike_clusters == u)
            if idx.size == 0:
                continue
            if idx.size > max_spikes_per_unit:
                idx = rng.choice(idx, size=max_spikes_per_unit, replace=False)
            indices_all.append(idx)
        if indices_all:
            indices = np.concatenate(indices_all)
            spike_times = spike_times[indices]
            spike_clusters = spike_clusters[indices]

    # 控制总样本数
    if max_total_spikes is not None and spike_times.size > max_total_spikes:
        idx = rng.choice(np.arange(spike_times.size), size=max_total_spikes, replace=False)
        spike_times = spike_times[idx]
        spike_clusters = spike_clusters[idx]

    waveforms = np.empty((spike_times.size, L), dtype=np.float32)

    for i, t in enumerate(spike_times):
        window = raw[t - pre : t + post, :]  # (L, C)

        if cfg.channel_select == "fixed":
            ch = int(cfg.fixed_channel)
        elif cfg.channel_select == "max_abs":
            ch = int(np.argmax(np.max(np.abs(window), axis=0)))
        else:
            raise ValueError(f"未知 channel_select: {cfg.channel_select}")

        wf = window[:, ch].astype(np.float32)
        wf = _align_1d(wf, cfg.align_mode, cfg.center_index)
        waveforms[i] = wf

    return waveforms, spike_clusters


def encode_waveforms_with(
    waveforms: np.ndarray,
    labels: np.ndarray,
    encoder_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    通用 encoder 入口：你只需要提供一个函数，把 (waveforms, labels) -> (bits, labels)。

    - waveforms: (N, L) float
    - labels:    (N,)   int
    - encoder_fn: 用户自定义，内部可调用 Autoencoders-in-Spike-Sorting / 你的 AE / 其他方法

    返回：
    - bits:   (N, B) 0/1 ndarray
    - labels: (N,)   int ndarray  （与输入一一对应）
    """
    waveforms = np.asarray(waveforms)
    labels = np.asarray(labels).astype(int)
    bits, out_labels = encoder_fn(waveforms, labels)
    bits = np.asarray(bits).astype(int)
    out_labels = np.asarray(out_labels).astype(int)

    if out_labels.shape[0] != bits.shape[0]:
        raise ValueError("encoder_fn 输出的 bits 与 labels 行数不一致")
    return bits, out_labels


def autoencoder_bits_encoder(
    waveforms: np.ndarray,
    labels: np.ndarray,
    *,
    ae_type: str = "normal",
    code_size: int = 8,
    ae_layers: list[int] | None = None,
    scale: str = "minmax",
    nr_epochs: int = 50,
    output_activation: str = "tanh",
    loss_function: str = "mse",
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    shuffle: bool = True,
    no_noise: bool = False,
    verbose: int = 1,
    binarize_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Autoencoders-in-Spike-Sorting 里的 `run_autoencoder` 把波形编码成 bits。

    这是一个可以直接传给 `encode_waveforms_with` 的 encoder_fn：

    ```python
    bits, labels = encode_waveforms_with(waveforms, labels, autoencoder_bits_encoder)
    ```

    典型用法：
    - data_type = \"m0\"（表示直接用传入的矩阵 data）
    - ae_type  = \"normal\" / \"shallow\" / \"tied\" / \"contractive\" 等
    - code_size 决定 latent 维度，后续会二值化成对应长度的 bits
    """
    waveforms = np.asarray(waveforms, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)

    # 1) 把 Autoencoders-in-Spike-Sorting 目录加入 sys.path
    here = os.path.dirname(os.path.abspath(__file__))
    ae_root = os.path.join(here, "encoder", "Autoencoders-in-Spike-Sorting")
    if ae_root not in sys.path:
        sys.path.insert(0, ae_root)

    from ae_function import run_autoencoder  # type: ignore

    if ae_layers is None:
        # 默认网络层数，可以根据需要调整
        ae_layers = [70, 60, 50, 40, 30]

    import numpy as _np  # 局部导入，避免和上面别名冲突

    features, _, gt_labels = run_autoencoder(
        data_type="m0",              # 直接用矩阵 data
        simulation_number=None,
        data=waveforms,
        labels=None,
        gt_labels=labels,            # 作为 ground truth 传入，内部可能会 shuffle
        index=None,
        ae_type=ae_type,
        ae_layers=_np.array(ae_layers),
        code_size=code_size,
        output_activation=output_activation,
        loss_function=loss_function,
        scale=scale,
        shuff=shuffle,
        noNoise=no_noise,
        nr_epochs=nr_epochs,
        doPlot=False,
        savePlot=False,
        saveWeights=False,
        dropout=dropout,
        weight_init="glorot_uniform",
        learning_rate=learning_rate,
        verbose=verbose,
    )

    features = _np.asarray(features, dtype=_np.float32)
    gt_labels = _np.asarray(gt_labels, dtype=int)

    # 简单二值化：> threshold 视为 1，否则为 0
    bits = (features > binarize_threshold).astype(int)
    return bits, gt_labels


__all__ = [
    "WaveformExtractConfig",
    "extract_waveforms_from_npz",
    "encode_waveforms_with",
    "autoencoder_bits_encoder",
]

