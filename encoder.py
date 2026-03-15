"""Waveform-to-bits encoders and encoded-dataset statistics.

中文说明
--------
这个文件负责前端编码：

- 输入：waveform
- 输出：固定 bit 数的二值编码

虽然 encoder 不是本项目的主研究对象，但它决定了 CAM 看到的输入质量。
所以这里同时提供：

- PCA baseline
- 轻量 numpy AE
- 外部 AE 仓库适配接口
- encoded dataset 统计诊断

这样你可以把“encoder 质量”和“CAM 算法质量”分开分析。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np

from config import EncoderConfig, ensure_parent, json_ready, project_path, resolve_path, save_json
from dataio import EncodedDataset, WaveformDataset


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise divide with small-value protection."""

    return numerator / np.where(np.abs(denominator) < 1e-8, 1.0, denominator)


def fit_scaler(waveforms: np.ndarray, method: str) -> Dict[str, np.ndarray]:
    """Fit feature-wise scaling parameters.

    中文：先对 waveform 做 feature-wise scale，通常能让编码更稳定。
    """

    x = np.asarray(waveforms, dtype=np.float32)
    if method == "none":
        return {}
    if method == "zscore":
        return {"mean": x.mean(axis=0), "scale": x.std(axis=0)}
    if method == "maxabs":
        return {"scale": np.max(np.abs(x), axis=0)}
    if method == "minmax":
        return {"min": x.min(axis=0), "max": x.max(axis=0)}
    if method == "robust":
        q25 = np.percentile(x, 25, axis=0)
        q50 = np.percentile(x, 50, axis=0)
        q75 = np.percentile(x, 75, axis=0)
        return {"median": q50, "iqr": q75 - q25}
    raise ValueError(f"Unknown scaling method: {method}")


def apply_scaler(waveforms: np.ndarray, method: str, params: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply feature-wise scaling.

    中文：把训练时拟合得到的 scale 参数真正应用到 waveform 上。
    """

    x = np.asarray(waveforms, dtype=np.float32)
    if method == "none":
        return x
    if method == "zscore":
        return _safe_divide(x - params["mean"], params["scale"]).astype(np.float32)
    if method == "maxabs":
        return _safe_divide(x, params["scale"]).astype(np.float32)
    if method == "minmax":
        denom = np.where(np.abs(params["max"] - params["min"]) < 1e-8, 1.0, params["max"] - params["min"])
        return (2.0 * ((x - params["min"]) / denom) - 1.0).astype(np.float32)
    if method == "robust":
        return _safe_divide(x - params["median"], params["iqr"]).astype(np.float32)
    raise ValueError(f"Unknown scaling method: {method}")


@dataclass
class BinaryEncoder:
    """Base class for waveform encoders.

    中文：所有 encoder 的统一接口。
    """

    cfg: EncoderConfig
    seed: int

    def fit_transform(self, waveforms: np.ndarray) -> np.ndarray:
        """Fit the encoder and return binary codes."""

        raise NotImplementedError

    def transform(self, waveforms: np.ndarray) -> np.ndarray:
        """Transform waveforms into binary codes using fitted parameters."""

        raise NotImplementedError


@dataclass
class PCABinaryEncoder(BinaryEncoder):
    """Simple PCA baseline using numpy SVD.

    中文：最重要的 baseline 之一，用来和 AE 做对照。
    """

    scaler_params: Optional[Dict[str, np.ndarray]] = None
    components_: Optional[np.ndarray] = None
    mean_: Optional[np.ndarray] = None
    code_thresholds_: Optional[np.ndarray] = None

    def fit_transform(self, waveforms: np.ndarray) -> np.ndarray:
        x = np.asarray(waveforms, dtype=np.float32)
        self.scaler_params = fit_scaler(x, self.cfg.scale)
        x_scaled = apply_scaler(x, self.cfg.scale, self.scaler_params)
        self.mean_ = x_scaled.mean(axis=0)
        centered = x_scaled - self.mean_
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.cfg.code_size].astype(np.float32)
        codes = centered @ self.components_.T
        # 中文：
        # - median: 当前主线默认，用各维中位数让 bit 更均衡
        # - zero: 兼容 previous work 里最常见的 “PCA > 0” 二值化协议
        if self.cfg.binarize_mode == "median":
            self.code_thresholds_ = np.median(codes, axis=0).astype(np.float32) + float(self.cfg.binarize_threshold)
        elif self.cfg.binarize_mode == "zero":
            self.code_thresholds_ = np.zeros(self.cfg.code_size, dtype=np.float32) + float(self.cfg.binarize_threshold)
        else:
            raise ValueError(f"Unknown PCA binarize_mode: {self.cfg.binarize_mode}")
        return (codes >= self.code_thresholds_).astype(np.uint8)

    def transform(self, waveforms: np.ndarray) -> np.ndarray:
        if self.scaler_params is None or self.components_ is None or self.mean_ is None or self.code_thresholds_ is None:
            raise RuntimeError("PCA encoder must be fitted before calling transform")
        x = apply_scaler(waveforms, self.cfg.scale, self.scaler_params)
        codes = (x - self.mean_) @ self.components_.T
        return (codes >= self.code_thresholds_).astype(np.uint8)


@dataclass
class NumpyAutoencoderBinaryEncoder(BinaryEncoder):
    """Lightweight tanh autoencoder implemented with numpy only.

    This encoder is included so the project remains runnable in environments
    without TensorFlow / PyTorch. It is not meant to replace a stronger AE,
    but it is good enough for reproducible CAM-side experiments.

    中文：
    这是一个“可直接跑”的轻量 AE fallback。
    它不是为了追求最强 encoder，而是为了保证整个 CAM 平台在轻环境下也能跑通。
    """

    scaler_params: Optional[Dict[str, np.ndarray]] = None
    weights_: Optional[list[np.ndarray]] = None
    biases_: Optional[list[np.ndarray]] = None
    code_thresholds_: Optional[np.ndarray] = None

    def _network_sizes(self, input_dim: int) -> list[int]:
        # 中文：encoder 部分是 input -> hidden -> code，
        # decoder 部分按 hidden 反向对称展开。
        hidden = list(self.cfg.layers)
        return [input_dim, *hidden, self.cfg.code_size, *reversed(hidden), input_dim]

    def _initialize(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        sizes = self._network_sizes(input_dim)
        self.weights_ = []
        self.biases_ = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.weights_.append(rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32))
            self.biases_.append(np.zeros(out_dim, dtype=np.float32))

    def _forward(self, batch: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self.weights_ is None or self.biases_ is None:
            raise RuntimeError("Autoencoder has not been initialized")
        activations = [batch]
        pre_activations = []
        num_layers = len(self.weights_)
        for layer_idx, (weights, bias) in enumerate(zip(self.weights_, self.biases_)):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                z = np.clip(activations[-1] @ weights + bias, -20.0, 20.0)
            pre_activations.append(z)
            if layer_idx == num_layers - 1:
                a = z
            else:
                a = np.tanh(z)
            activations.append(a.astype(np.float32))
        return activations, pre_activations

    def _encode_scaled(self, waveforms_scaled: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.biases_ is None:
            raise RuntimeError("Autoencoder must be fitted before encoding")
        activations = np.clip(waveforms_scaled, -5.0, 5.0)
        encoder_layers = len(self.cfg.layers) + 1
        for layer_idx in range(encoder_layers):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                z = np.clip(activations @ self.weights_[layer_idx] + self.biases_[layer_idx], -20.0, 20.0)
            activations = np.tanh(z).astype(np.float32)
        return activations

    def fit_transform(self, waveforms: np.ndarray) -> np.ndarray:
        x = np.asarray(waveforms, dtype=np.float32)
        self.scaler_params = fit_scaler(x, self.cfg.scale)
        x_scaled = np.clip(apply_scaler(x, self.cfg.scale, self.scaler_params), -5.0, 5.0)
        self._initialize(x_scaled.shape[1])

        if self.weights_ is None or self.biases_ is None:
            raise RuntimeError("Autoencoder weights were not initialized")

        rng = np.random.default_rng(self.seed)
        batch_size = max(1, int(self.cfg.batch_size))
        num_samples = x_scaled.shape[0]
        learning_rate = float(self.cfg.learning_rate)

        # 中文：这里是很轻量的 full-batch / mini-batch numpy 训练逻辑，
        # 目标是稳定可跑，不是追求最先进 AE 性能。
        for epoch in range(int(self.cfg.epochs)):
            order = rng.permutation(num_samples)
            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                batch_indices = order[start : start + batch_size]
                batch = x_scaled[batch_indices]
                activations, _ = self._forward(batch)
                reconstruction = activations[-1]
                delta = np.clip((reconstruction - batch) / max(1, batch.shape[0]), -5.0, 5.0)
                epoch_loss += float(np.mean((reconstruction - batch) ** 2))

                for layer_idx in reversed(range(len(self.weights_))):
                    if layer_idx < len(self.weights_) - 1:
                        delta = np.clip(delta * (1.0 - activations[layer_idx + 1] ** 2), -5.0, 5.0)
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        grad_w = np.clip(activations[layer_idx].T @ delta, -5.0, 5.0)
                    grad_b = np.clip(delta.sum(axis=0), -5.0, 5.0)
                    if layer_idx > 0:
                        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                            next_delta = np.clip(delta @ self.weights_[layer_idx].T, -5.0, 5.0)
                    self.weights_[layer_idx] -= learning_rate * grad_w.astype(np.float32)
                    self.biases_[layer_idx] -= learning_rate * grad_b.astype(np.float32)
                    self.weights_[layer_idx] = np.clip(
                        np.nan_to_num(self.weights_[layer_idx], nan=0.0, posinf=3.0, neginf=-3.0),
                        -3.0,
                        3.0,
                    )
                    self.biases_[layer_idx] = np.clip(
                        np.nan_to_num(self.biases_[layer_idx], nan=0.0, posinf=3.0, neginf=-3.0),
                        -3.0,
                        3.0,
                    )
                    if layer_idx > 0:
                        delta = next_delta

            if self.cfg.verbose:
                mean_loss = epoch_loss / max(1, int(np.ceil(num_samples / batch_size)))
                print(f"[encoder][numpy-ae] epoch {epoch + 1:02d}/{self.cfg.epochs} loss={mean_loss:.6f}")

        codes = self._encode_scaled(x_scaled)
        self.code_thresholds_ = np.median(codes, axis=0).astype(np.float32) + float(self.cfg.binarize_threshold)
        return (codes >= self.code_thresholds_).astype(np.uint8)

    def transform(self, waveforms: np.ndarray) -> np.ndarray:
        if self.scaler_params is None or self.code_thresholds_ is None:
            raise RuntimeError("Autoencoder scaler parameters are not available")
        x_scaled = np.clip(apply_scaler(waveforms, self.cfg.scale, self.scaler_params), -5.0, 5.0)
        codes = self._encode_scaled(x_scaled)
        return (codes >= self.code_thresholds_).astype(np.uint8)


def _external_scale_params(waveforms: np.ndarray, scale: str) -> Dict[str, np.ndarray]:
    """Fit serializable scale parameters compatible with the external AE repo.

    中文：
    为了做到“训练一次、后续直接加载模型反复 encode”，
    不能只保存模型权重，还要把训练时的 scale 参数也保存下来。
    """

    x = np.asarray(waveforms, dtype=np.float32)
    if scale in {"minmax", "minmax_relu", "minmax_spp"}:
        return {
            "global_min": np.asarray(float(np.min(x)), dtype=np.float32),
            "global_max": np.asarray(float(np.max(x)), dtype=np.float32),
        }
    if scale == "scaler":
        return {
            "feature_min": np.min(x, axis=0).astype(np.float32),
            "feature_max": np.max(x, axis=0).astype(np.float32),
        }
    if scale in {"ignore_amplitude", "divide_amplitude", "scale_no_energy_loss"}:
        return {}
    if scale == "-1+1":
        return {
            "global_min": np.asarray(-1.0, dtype=np.float32),
            "global_max": np.asarray(1.0, dtype=np.float32),
        }
    raise ValueError(
        "Artifact-backed external AE currently supports scale in "
        "{'minmax', 'minmax_relu', 'minmax_spp', 'scaler', 'ignore_amplitude', "
        "'divide_amplitude', 'scale_no_energy_loss', '-1+1'}"
    )


def _external_apply_scale(waveforms: np.ndarray, scale: str, params: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply a saved external-AE scaling rule.

    中文：把训练时拟合的 external scale 参数稳定地应用到任意新 waveform 上。
    """

    x = np.asarray(waveforms, dtype=np.float32)
    if scale in {"minmax", "minmax_relu", "minmax_spp", "-1+1"}:
        min_peak = float(np.asarray(params["global_min"]).item())
        max_peak = float(np.asarray(params["global_max"]).item())
        denom = max(1e-8, max_peak - min_peak)
        scaled = (x - min_peak) / denom
        if scale == "minmax":
            return (scaled * 2.0 - 1.0).astype(np.float32)
        if scale == "minmax_spp":
            return (scaled * 4.0 - 3.0).astype(np.float32)
        return scaled.astype(np.float32)
    if scale == "scaler":
        min_peak = np.asarray(params["feature_min"], dtype=np.float32)
        max_peak = np.asarray(params["feature_max"], dtype=np.float32)
        denom = np.where(np.abs(max_peak - min_peak) < 1e-8, 1.0, max_peak - min_peak)
        return ((x - min_peak) / denom).astype(np.float32)
    if scale == "ignore_amplitude":
        row_min = np.min(x, axis=1, keepdims=True)
        row_max = np.max(x, axis=1, keepdims=True)
        denom = np.where(np.abs(row_max - row_min) < 1e-8, 1.0, row_max - row_min)
        return (((x - row_min) / denom) * 2.0 - 1.0).astype(np.float32)
    if scale == "divide_amplitude":
        amplitudes = np.max(x, axis=1, keepdims=True)
        amplitudes = np.where(np.abs(amplitudes) < 1e-8, 1.0, amplitudes)
        return (x / amplitudes).astype(np.float32)
    if scale == "scale_no_energy_loss":
        scaled = x.copy()
        row_min = np.min(scaled, axis=1, keepdims=True)
        row_max = np.max(scaled, axis=1, keepdims=True)
        neg_mask = scaled < 0
        pos_mask = scaled > 0
        scaled[neg_mask] = scaled[neg_mask] / np.where(np.abs(row_min.repeat(scaled.shape[1], axis=1)[neg_mask]) < 1e-8, 1.0, np.abs(row_min.repeat(scaled.shape[1], axis=1)[neg_mask]))
        scaled[pos_mask] = scaled[pos_mask] / np.where(np.abs(row_max.repeat(scaled.shape[1], axis=1)[pos_mask]) < 1e-8, 1.0, row_max.repeat(scaled.shape[1], axis=1)[pos_mask])
        return scaled.astype(np.float32)
    raise ValueError(f"Unsupported external artifact scale: {scale}")


def _external_artifact_paths(artifact_root: Path) -> Dict[str, Path]:
    """Return canonical file paths for one saved external AE artifact."""

    return {
        "root": artifact_root,
        "metadata": artifact_root / "metadata.json",
        "weights": artifact_root / "autoencoder.weights.h5",
        "thresholds": artifact_root / "code_thresholds.npy",
        "scale_params": artifact_root / "scale_params.npz",
    }


def _load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    """Load a simple ``key -> ndarray`` mapping from a compressed npz file."""

    pack = np.load(path, allow_pickle=False)
    return {key: np.asarray(pack[key]) for key in pack.files}


def train_external_autoencoder_artifact(
    waveforms: np.ndarray,
    cfg: EncoderConfig,
    *,
    seed: int,
    artifact_path: str | Path,
    resume: bool = False,
) -> Dict[str, Any]:
    """Train or resume one external AE artifact and save it to disk.

    中文：
    这是给独立训练脚本用的公共函数：

    - 可直接对一批 waveform 训练 external AE
    - 可选择从已有 artifact 继续训练
    - 会把权重、threshold、scale 参数和训练摘要一起保存
    """

    artifact_root = Path(artifact_path)
    encoder = ExternalAutoencoderBinaryEncoder(cfg=cfg, seed=seed)
    imports = encoder._prepare_external_repo()

    initial_weights_path: Optional[Path] = None
    if resume and artifact_root.exists():
        encoder._load_artifact(artifact_root)
        initial_weights_path = _external_artifact_paths(artifact_root)["weights"]
        if encoder.actual_scale_ is None or encoder.scale_params_ is None:
            raise RuntimeError("Cannot resume external AE training without saved scale parameters")
        actual_scale = encoder.actual_scale_
        scale_params = encoder.scale_params_
    else:
        actual_scale = encoder._resolve_external_scale()
        scale_params = _external_scale_params(waveforms, actual_scale)

    scaled_waveforms = _external_apply_scale(np.asarray(waveforms, dtype=np.float32), actual_scale, scale_params)
    train_bundle = imports["train_autoencoder_for_spikes"](
        scaled_waveforms,
        ae_type=cfg.ae_type,
        ae_layers=np.asarray(cfg.layers, dtype=int),
        code_size=cfg.code_size,
        output_activation="tanh",
        loss_function="mse",
        nr_epochs=int(cfg.epochs),
        learning_rate=float(cfg.learning_rate),
        verbose=int(cfg.verbose),
        dropout=0.0,
        weight_init="glorot_uniform",
        scale=actual_scale,
        apply_scale=False,
        initial_weights_path=str(initial_weights_path) if initial_weights_path is not None else None,
    )

    encoder.encoder_model_ = train_bundle["encoder"]
    encoder.autoencoder_model_ = train_bundle["autoencoder"]
    encoder.actual_scale_ = actual_scale
    encoder.scale_params_ = scale_params
    encoder.code_thresholds_ = np.median(np.asarray(train_bundle["features"], dtype=np.float32), axis=0).astype(np.float32)
    encoder._save_artifact(
        artifact_root,
        input_dim=np.asarray(waveforms, dtype=np.float32).shape[1],
        resolved_layers=list(train_bundle["resolved_layers"]),
        resolved_loss=str(train_bundle["resolved_loss"]),
        history=train_bundle.get("history"),
        training_waveforms=np.asarray(waveforms, dtype=np.float32),
        scaled_waveforms=scaled_waveforms,
        features=np.asarray(train_bundle["features"], dtype=np.float32),
    )

    summary = {
        "artifact_path": str(artifact_root),
        "resume": bool(resume),
        "actual_scale": actual_scale,
        "num_training_spikes": int(np.asarray(waveforms).shape[0]),
        "input_dim": int(np.asarray(waveforms).shape[1]),
        "code_size": int(cfg.code_size),
        "ae_type": cfg.ae_type,
        "resolved_layers": list(train_bundle["resolved_layers"]),
        "history": train_bundle.get("history"),
    }
    return summary


@dataclass
class ExternalAutoencoderBinaryEncoder(BinaryEncoder):
    """Adapter around the bundled ``Autoencoders-in-Spike-Sorting`` repo.

    中文：把外部 AE 仓库包装成统一接口，方便直接接入本实验平台。
    """

    code_thresholds_: Optional[np.ndarray] = None
    artifact_root_: Optional[Path] = None
    artifact_meta_: Optional[Dict[str, Any]] = None
    actual_scale_: Optional[str] = None
    scale_params_: Optional[Dict[str, np.ndarray]] = None
    encoder_model_: Any = None
    autoencoder_model_: Any = None

    def _resolve_external_scale(self) -> str:
        """Map platform scale names to values supported by the external repo."""

        supported = {
            "minmax",
            "minmax_relu",
            "minmax_spp",
            "-1+1",
            "scaler",
            "ignore_amplitude",
            "ignore_amplitude_add_amplitude",
            "add_energy",
            "divide_amplitude",
            "scale_no_energy_loss",
        }
        if self.cfg.scale in supported:
            return self.cfg.scale
        # 中文：平台默认常用 robust scale，但外部仓库并不支持，
        # 这里统一回退到它最常见、最稳妥的 minmax。
        return "minmax"

    def _prepare_external_repo(self):
        """Import the external repo helpers on demand.

        中文：只有在真正需要 external AE 时才导入 TensorFlow 相关依赖。
        """

        external_root = project_path("external", "Autoencoders-in-Spike-Sorting")
        if not external_root.exists():
            raise RuntimeError("external/Autoencoders-in-Spike-Sorting is missing")

        if importlib.util.find_spec("sklearn") is None:
            raise RuntimeError("scikit-learn is required for the external AE backend")
        if importlib.util.find_spec("tensorflow") is None:
            raise RuntimeError("tensorflow is required for the external AE backend")

        if str(external_root) not in sys.path:
            sys.path.insert(0, str(external_root))

        from ae_function import build_autoencoder_model_bundle, train_autoencoder_for_spikes  # type: ignore

        return {
            "build_autoencoder_model_bundle": build_autoencoder_model_bundle,
            "train_autoencoder_for_spikes": train_autoencoder_for_spikes,
        }

    def _artifact_root(self) -> Optional[Path]:
        if not self.cfg.artifact_path:
            return None
        return resolve_path(str(self.cfg.artifact_path))

    def _save_artifact(
        self,
        artifact_root: Path,
        *,
        input_dim: int,
        resolved_layers: list[int],
        resolved_loss: str,
        history: Optional[Dict[str, Any]],
        training_waveforms: np.ndarray,
        scaled_waveforms: np.ndarray,
        features: np.ndarray,
    ) -> None:
        """Persist a trained external AE so future runs can reuse it directly.

        中文：
        artifact 里保存：

        - Keras weights
        - bit 二值化 thresholds
        - scale 参数
        - 训练配置与一些诊断指标
        """

        if self.autoencoder_model_ is None or self.code_thresholds_ is None or self.actual_scale_ is None or self.scale_params_ is None:
            raise RuntimeError("External AE artifact cannot be saved before training finishes")

        paths = _external_artifact_paths(artifact_root)
        paths["root"].mkdir(parents=True, exist_ok=True)
        ensure_parent(paths["weights"])
        self.autoencoder_model_.save_weights(str(paths["weights"]))
        np.save(paths["thresholds"], np.asarray(self.code_thresholds_, dtype=np.float32))
        np.savez_compressed(paths["scale_params"], **{key: np.asarray(value) for key, value in self.scale_params_.items()})

        try:
            reconstructions = self.autoencoder_model_.predict(scaled_waveforms, verbose=0)
        except TypeError:
            reconstructions = self.autoencoder_model_.predict(scaled_waveforms)
        reconstruction_mse = float(np.mean((np.asarray(reconstructions, dtype=np.float32) - scaled_waveforms) ** 2))

        metadata = {
            "format_version": 1,
            "backend": "external",
            "ae_type": self.cfg.ae_type,
            "resolved_layers": list(resolved_layers),
            "resolved_loss": resolved_loss,
            "code_size": int(self.cfg.code_size),
            "output_activation": "tanh",
            "scale": self.actual_scale_,
            "input_dim": int(input_dim),
            "learning_rate": float(self.cfg.learning_rate),
            "epochs": int(self.cfg.epochs),
            "binarize_threshold": float(self.cfg.binarize_threshold),
            "num_training_spikes": int(training_waveforms.shape[0]),
            "reconstruction_mse": reconstruction_mse,
            "feature_mean": np.asarray(features.mean(axis=0), dtype=np.float32).tolist(),
            "feature_std": np.asarray(features.std(axis=0), dtype=np.float32).tolist(),
            "thresholds_preview": np.asarray(self.code_thresholds_[: min(8, self.code_thresholds_.shape[0])], dtype=np.float32).tolist(),
            "history": history,
        }
        save_json(paths["metadata"], metadata)
        self.artifact_root_ = artifact_root
        self.artifact_meta_ = metadata

    def _load_artifact(self, artifact_root: Path) -> None:
        """Load a previously trained external AE artifact.

        中文：从磁盘恢复 external AE，避免每次实验都重新训练。
        """

        paths = _external_artifact_paths(artifact_root)
        if not paths["metadata"].exists() or not paths["weights"].exists() or not paths["thresholds"].exists() or not paths["scale_params"].exists():
            raise RuntimeError(f"Incomplete external AE artifact at {artifact_root}")

        imports = self._prepare_external_repo()
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        bundle = imports["build_autoencoder_model_bundle"](
            ae_type=metadata["ae_type"],
            input_dim=int(metadata["input_dim"]),
            ae_layers=np.asarray(metadata["resolved_layers"], dtype=int),
            code_size=int(metadata["code_size"]),
            output_activation=str(metadata.get("output_activation", "tanh")),
            loss_function=str(metadata.get("resolved_loss", "mse")),
            dropout=0.0,
            weight_init="glorot_uniform",
        )

        autoencoder_model = bundle["autoencoder"]
        encoder_model = bundle["encoder"]
        autoencoder_model.load_weights(str(paths["weights"]))

        self.encoder_model_ = encoder_model
        self.autoencoder_model_ = autoencoder_model
        self.code_thresholds_ = np.asarray(np.load(paths["thresholds"]), dtype=np.float32)
        self.scale_params_ = _load_npz_dict(paths["scale_params"])
        self.actual_scale_ = str(metadata["scale"])
        self.artifact_root_ = artifact_root
        self.artifact_meta_ = metadata

    def _encode_features(self, waveforms: np.ndarray) -> np.ndarray:
        """Encode waveforms into continuous latent features."""

        if self.encoder_model_ is None or self.code_thresholds_ is None or self.actual_scale_ is None or self.scale_params_ is None:
            raise RuntimeError("External AE must be trained or loaded before encoding")
        scaled = _external_apply_scale(waveforms, self.actual_scale_, self.scale_params_)
        try:
            return np.asarray(self.encoder_model_.predict(scaled, verbose=0), dtype=np.float32)
        except TypeError:
            return np.asarray(self.encoder_model_.predict(scaled), dtype=np.float32)

    def fit_transform(self, waveforms: np.ndarray) -> np.ndarray:
        artifact_root = self._artifact_root()
        if (
            artifact_root is not None
            and bool(self.cfg.use_artifact)
            and artifact_root.exists()
            and not bool(self.cfg.force_retrain_artifact)
        ):
            self._load_artifact(artifact_root)
            features = self._encode_features(np.asarray(waveforms, dtype=np.float32))
            return (features >= self.code_thresholds_).astype(np.uint8)

        imports = self._prepare_external_repo()
        actual_scale = self._resolve_external_scale()
        training_waveforms = np.asarray(waveforms, dtype=np.float32)
        self.actual_scale_ = actual_scale
        self.scale_params_ = _external_scale_params(training_waveforms, actual_scale)
        scaled_waveforms = _external_apply_scale(training_waveforms, actual_scale, self.scale_params_)

        train_bundle = imports["train_autoencoder_for_spikes"](
            scaled_waveforms,
            ae_type=self.cfg.ae_type,
            ae_layers=np.asarray(self.cfg.layers, dtype=int),
            code_size=self.cfg.code_size,
            output_activation="tanh",
            loss_function="mse",
            nr_epochs=int(self.cfg.epochs),
            learning_rate=float(self.cfg.learning_rate),
            verbose=int(self.cfg.verbose),
            dropout=0.0,
            weight_init="glorot_uniform",
            scale=actual_scale,
            apply_scale=False,
        )
        self.encoder_model_ = train_bundle["encoder"]
        self.autoencoder_model_ = train_bundle["autoencoder"]
        features = np.asarray(train_bundle["features"], dtype=np.float32)
        self.code_thresholds_ = np.median(features, axis=0).astype(np.float32) + float(self.cfg.binarize_threshold)

        if artifact_root is not None and bool(self.cfg.save_artifact):
            self._save_artifact(
                artifact_root,
                input_dim=training_waveforms.shape[1],
                resolved_layers=list(train_bundle["resolved_layers"]),
                resolved_loss=str(train_bundle["resolved_loss"]),
                history=train_bundle.get("history"),
                training_waveforms=training_waveforms,
                scaled_waveforms=scaled_waveforms,
                features=features,
            )
        return (features >= self.code_thresholds_).astype(np.uint8)

    def transform(self, waveforms: np.ndarray) -> np.ndarray:
        features = self._encode_features(np.asarray(waveforms, dtype=np.float32))
        return (features >= self.code_thresholds_).astype(np.uint8)


def build_encoder(cfg: EncoderConfig, seed: int) -> BinaryEncoder:
    """Factory for binary encoders.

    中文：根据 config 创建实际使用的 encoder。
    """

    if cfg.method == "pca":
        return PCABinaryEncoder(cfg=cfg, seed=seed)
    if cfg.method != "ae":
        raise ValueError(f"Unknown encoder method: {cfg.method}")

    if cfg.backend == "numpy":
        return NumpyAutoencoderBinaryEncoder(cfg=cfg, seed=seed)
    if cfg.backend == "external":
        return ExternalAutoencoderBinaryEncoder(cfg=cfg, seed=seed)
    if cfg.backend == "auto":
        external_root = project_path("external", "Autoencoders-in-Spike-Sorting")
        sklearn_ok = importlib.util.find_spec("sklearn") is not None
        tensorflow_ok = importlib.util.find_spec("tensorflow") is not None
        if external_root.exists() and sklearn_ok and tensorflow_ok:
            return ExternalAutoencoderBinaryEncoder(cfg=cfg, seed=seed)
        return NumpyAutoencoderBinaryEncoder(cfg=cfg, seed=seed)
    raise ValueError(f"Unknown AE backend: {cfg.backend}")


def encode_waveform_dataset(
    dataset: WaveformDataset,
    cfg: EncoderConfig,
    seed: int,
) -> EncodedDataset:
    """Encode a waveform dataset into binary bits while preserving order.

    中文：这是 waveform -> encoded dataset 的主入口。
    """

    encoder = build_encoder(cfg, seed)
    bits = encoder.fit_transform(dataset.waveforms)
    resolved_backend = cfg.backend
    if isinstance(encoder, ExternalAutoencoderBinaryEncoder):
        resolved_backend = "external"
    elif isinstance(encoder, NumpyAutoencoderBinaryEncoder):
        resolved_backend = "numpy"
    elif isinstance(encoder, PCABinaryEncoder):
        resolved_backend = "pca"
    meta = {
        "encoder_method": cfg.method,
        "encoder_backend": resolved_backend,
        "encoder_impl": encoder.__class__.__name__,
        "code_size": int(cfg.code_size),
        "ae_type": cfg.ae_type,
        "scale": cfg.scale,
        "epochs": int(cfg.epochs),
        "layers": list(cfg.layers),
        "source_meta": json_ready(dataset.meta),
    }
    if isinstance(encoder, ExternalAutoencoderBinaryEncoder):
        meta["external_artifact_path"] = str(encoder.artifact_root_) if encoder.artifact_root_ is not None else None
        meta["external_actual_scale"] = encoder.actual_scale_
        if encoder.artifact_meta_ is not None:
            meta["external_artifact_meta"] = json_ready(encoder.artifact_meta_)
    return EncodedDataset(
        bits=np.asarray(bits, dtype=np.uint8),
        labels=np.asarray(dataset.labels, dtype=np.int64),
        spike_times=np.asarray(dataset.spike_times, dtype=np.int64),
        source_indices=np.asarray(dataset.source_indices, dtype=np.int64),
        meta=meta,
    )


def _sample_pair_hamming(
    bits: np.ndarray,
    labels: np.ndarray,
    *,
    same_class: bool,
    max_pairs: int,
    rng: np.random.Generator,
) -> float:
    """Estimate average Hamming distance for same-class or different-class pairs."""

    num_samples = bits.shape[0]
    if num_samples < 2:
        return float("nan")

    distances = []
    attempts = 0
    max_attempts = max_pairs * 20
    while len(distances) < max_pairs and attempts < max_attempts:
        i = int(rng.integers(0, num_samples))
        j = int(rng.integers(0, num_samples))
        if i == j:
            attempts += 1
            continue
        if same_class and labels[i] != labels[j]:
            attempts += 1
            continue
        if (not same_class) and labels[i] == labels[j]:
            attempts += 1
            continue
        distances.append(np.sum(np.bitwise_xor(bits[i], bits[j])))
        attempts += 1

    if not distances:
        return float("nan")
    return float(np.mean(distances))


def compute_bit_statistics(
    dataset: EncodedDataset,
    *,
    max_pairs: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute lightweight separability diagnostics for an encoded dataset.

    中文：
    这部分不是正式分类指标，而是 encoder 质量的诊断指标，
    比如：

    - 每位是否平衡
    - code 是否太重复
    - 同类 / 异类 Hamming 距离差距
    """

    bits = np.asarray(dataset.bits, dtype=np.uint8)
    labels = np.asarray(dataset.labels, dtype=np.int64)
    bit_means = bits.mean(axis=0)
    eps = 1e-8
    bit_entropy = -(bit_means * np.log2(bit_means + eps) + (1.0 - bit_means) * np.log2(1.0 - bit_means + eps))

    rng = np.random.default_rng(seed)
    intra = _sample_pair_hamming(bits, labels, same_class=True, max_pairs=max_pairs, rng=rng)
    inter = _sample_pair_hamming(bits, labels, same_class=False, max_pairs=max_pairs, rng=rng)

    unique_rows = np.unique(bits, axis=0)
    return {
        "num_spikes": int(bits.shape[0]),
        "bit_width": int(bits.shape[1]),
        "num_units": int(np.unique(labels).size),
        "unique_code_count": int(unique_rows.shape[0]),
        "unique_code_ratio": float(unique_rows.shape[0] / max(1, bits.shape[0])),
        "bit_mean_mean": float(bit_means.mean()),
        "bit_mean_std": float(bit_means.std()),
        "bit_entropy_mean": float(bit_entropy.mean()),
        "bit_entropy_std": float(bit_entropy.std()),
        "mean_intra_hamming": intra,
        "mean_inter_hamming": inter,
        "mean_hamming_gap": float(inter - intra) if np.isfinite(intra) and np.isfinite(inter) else float("nan"),
        "per_bit_mean": bit_means.tolist(),
        "per_bit_entropy": bit_entropy.tolist(),
    }


__all__ = [
    "BinaryEncoder",
    "ExternalAutoencoderBinaryEncoder",
    "NumpyAutoencoderBinaryEncoder",
    "PCABinaryEncoder",
    "apply_scaler",
    "build_encoder",
    "compute_bit_statistics",
    "encode_waveform_dataset",
    "fit_scaler",
]
