"""Experiment orchestration for the spike CAM platform.

中文说明
--------
这个文件是整个项目的“总调度器”。

它负责把前面各层真正串起来：

1. 读取 config
2. 准备 / 缓存 encoded dataset
3. 建立初始模板
4. 创建 CAM
5. 做 chronological online evaluation
6. 保存 metrics / predictions / curves / confusion

如果你把这个项目看成一条完整 pipeline，
那这个文件就是把所有模块连成实验闭环的地方。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from cam_core import CAM
from config import ExperimentConfig, ExperimentSuiteConfig, SubsetConfig, expand_variants, json_ready, resolve_path, save_json
from dataio import (
    EncodedDataset,
    REJECT_LABEL,
    WaveformDataset,
    chronological_split,
    load_label_counts_before_sampling,
    load_waveform_dataset,
    random_split,
    subset_by_count_map,
    subset_by_label_rule,
)
from encoder import compute_bit_statistics, encode_waveform_dataset
from match_strategies import HammingNearestMatch, MarginRejectMatch, Top2MarginMatch, WeightedHammingMatch
from metrics import (
    ResultBundle,
    compute_classification_metrics,
    compute_curve_bundle,
    compute_memory_metrics,
    compute_reject_metrics,
    compute_update_metrics,
    confusion_with_reject,
)
from templates import build_template_rows
from update_strategies import (
    ConfidenceWeightedUpdate,
    CooldownUpdate,
    CounterUpdate,
    DualTemplateUpdate,
    GrowingUpdate,
    MarginEmaUpdate,
    NoUpdate,
    ProbabilisticUpdate,
    Top2MarginUpdate,
)


def _safe_variant_name(name: str) -> str:
    """Create a filesystem-safe variant name.

    中文：把 variant 名字清洗成适合做目录名的形式。
    """

    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _cache_key(config: ExperimentConfig) -> str:
    """Hash dataset + encoder settings for encoded-dataset caching.

    中文：用数据配置和 encoder 配置生成缓存 key。
    """

    artifact_signature: Dict[str, Any] | None = None
    artifact_path = getattr(config.encoder, "artifact_path", None)
    if artifact_path:
        artifact_root = resolve_path(str(artifact_path))
        metadata_path = artifact_root / "metadata.json"
        thresholds_path = artifact_root / "code_thresholds.npy"
        artifact_signature = {
            "artifact_path": str(artifact_root),
            "metadata_exists": metadata_path.exists(),
            "thresholds_exists": thresholds_path.exists(),
        }
        if metadata_path.exists():
            artifact_signature["metadata_text"] = metadata_path.read_text(encoding="utf-8")
        if thresholds_path.exists():
            artifact_signature["thresholds_mtime_ns"] = thresholds_path.stat().st_mtime_ns

    relevant = {
        "dataset": json_ready(asdict(config.dataset)),
        "encoder": json_ready(asdict(config.encoder)),
        "seed": int(config.seed),
        "artifact_signature": artifact_signature,
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()[:12]


def encoded_cache_path(config: ExperimentConfig) -> Path:
    """Return the cache path for one experiment's encoded dataset.

    中文：计算某个实验对应的 encoded dataset 缓存路径。
    """

    cache_dir = resolve_path(config.encoder.cache_dir)
    dataset_stem = Path(config.dataset.npz_path).stem
    subset_mode = config.dataset.subset.mode
    key = _cache_key(config)
    filename = f"{dataset_stem}__{config.encoder.method}_{config.encoder.code_size}bit__{subset_mode}__{key}.npz"
    return cache_dir / filename


def resolve_result_root(results_dir: str) -> Path:
    """Resolve the suite result root, with a dated default under results/experiments.

    中文：
    如果配置里仍然写的是默认的 `results`，这里会自动把实际输出定向到：

    `results/experiments/<YYYY-MM-DD>/`

    这样普通实验不会继续把 `results/` 根目录塞满。
    如果用户显式指定了其他 `results_dir`，则尊重配置原样输出。
    """

    normalized = Path(results_dir).as_posix().rstrip("/")
    if normalized == "results":
        return resolve_path("results") / "experiments" / date.today().isoformat()
    return resolve_path(results_dir)


def prepare_encoded_dataset(config: ExperimentConfig) -> tuple[EncodedDataset, Path]:
    """Load or build the encoded dataset for one experiment config.

    中文：优先复用缓存；没有缓存时才从 raw 数据重新编码。
    """

    cache_path = encoded_cache_path(config)
    if config.encoder.reuse_cache and cache_path.exists() and not config.encoder.force_reencode:
        encoded = EncodedDataset.load_npz(cache_path)
        source_meta = encoded.meta.setdefault("source_meta", {})
        if "label_counts_before_sampling" not in source_meta:
            # 中文：旧缓存里没有 pre-sampling 计数时，自动补齐 metadata，
            # 否则 memory top-k 会被下采样后的“平票计数”误导。
            source_meta["label_counts_before_sampling"] = load_label_counts_before_sampling(config.dataset)
            encoded.save_npz(cache_path)
        return encoded, cache_path

    waveform_dataset = load_waveform_dataset(config.dataset)
    encoded = encode_waveform_dataset(waveform_dataset, config.encoder, seed=config.seed)
    encoded.meta["dataset_config"] = json_ready(asdict(config.dataset))
    encoded.meta["encoder_config"] = json_ready(asdict(config.encoder))
    encoded.save_npz(cache_path)
    return encoded, cache_path


def build_match_strategy(config: ExperimentConfig):
    """Instantiate the configured matcher.

    中文：根据 config 创建匹配策略对象。
    """

    method = config.matcher.method
    if method == "hamming_nearest":
        return HammingNearestMatch()
    if method == "weighted_hamming":
        return WeightedHammingMatch()
    if method == "margin_reject":
        return MarginRejectMatch(min_accept_margin=config.matcher.min_accept_margin)
    if method == "top2_margin":
        return Top2MarginMatch(min_margin=config.matcher.min_margin)
    raise ValueError(f"Unknown matcher method: {method}")


def build_update_strategy(config: ExperimentConfig):
    """Instantiate the configured update strategy.

    中文：根据 config 创建动态更新策略对象。
    """

    cfg = config.update
    if cfg.method == "none":
        return NoUpdate()
    if cfg.method == "counter":
        return CounterUpdate(max_confidence=cfg.max_confidence)
    if cfg.method == "margin_ema":
        return MarginEmaUpdate(alpha=cfg.alpha, margin_band=cfg.margin_band)
    if cfg.method == "confidence_weighted":
        return ConfidenceWeightedUpdate(
            lr=cfg.lr,
            max_conf=cfg.max_conf,
            min_weight=cfg.min_weight,
            flip_threshold=cfg.flip_threshold,
        )
    if cfg.method == "dual_template":
        return DualTemplateUpdate(alpha=cfg.alpha)
    if cfg.method == "probabilistic":
        return ProbabilisticUpdate(alpha=cfg.alpha)
    if cfg.method == "growing":
        return GrowingUpdate(split_threshold=cfg.split_threshold, allow_evict=cfg.allow_evict)
    if cfg.method == "cooldown":
        return CooldownUpdate(alpha=cfg.alpha, cooldown_steps=cfg.cooldown_steps)
    if cfg.method == "top2_margin":
        return Top2MarginUpdate(alpha=cfg.alpha, min_margin=cfg.min_margin)
    raise ValueError(f"Unknown update method: {cfg.method}")


def build_cam(config: ExperimentConfig, initial_template_count: int) -> CAM:
    """Construct one CAM instance for a concrete experiment.

    中文：按配置创建一个真正用于实验的 CAM 实例。
    """

    # 中文：如果没有显式给 capacity，就按初始化模板数量乘一个系数估算。
    if config.cam.capacity is not None:
        capacity = int(config.cam.capacity)
    else:
        capacity = int(max(initial_template_count, ceil(initial_template_count * config.cam.capacity_factor) + config.cam.extra_rows))

    match_strategy = build_match_strategy(config)
    update_strategy = build_update_strategy(config)
    return CAM(
        capacity=capacity,
        bit_width=int(config.encoder.code_size),
        match_strategy=match_strategy,
        update_strategy=update_strategy,
        eviction_policy=config.cam.eviction_policy,
    )


def _split_indices(encoded: EncodedDataset, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Choose warmup / online indices according to the evaluation mode.

    中文：决定 warmup 段和 online 段的划分方式。
    """

    if config.evaluation.mode == "chronological":
        return chronological_split(encoded.spike_times, config.evaluation.warmup_ratio)
    if config.evaluation.mode == "random_split":
        return random_split(encoded.num_spikes, config.evaluation.random_train_frac, config.seed)
    raise ValueError(f"Unknown evaluation mode: {config.evaluation.mode}")


def _memory_selection_rule(config: ExperimentConfig) -> Optional[SubsetConfig]:
    """Convert ``cam.memory_subset`` into a normal subset rule if needed.

    中文：
    ``same_as_stream`` 表示沿用旧逻辑，不额外限制 CAM 记忆类集合；
    否则就把 memory subset 转成普通 subset 规则来选“允许进 CAM 的类”。
    """

    mem_cfg = config.cam.memory_subset
    if mem_cfg.mode == "same_as_stream":
        return None
    return SubsetConfig(mode=mem_cfg.mode, topk=mem_cfg.topk, min_count=mem_cfg.min_count)


def select_memory_labels(
    encoded: EncodedDataset,
    warmup_labels: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Choose which labels are allowed to become CAM templates.

    中文：
    这是这次协议修正的关键函数：

    - stream 里可以有很多类
    - 但 CAM 初始模板只从 memory subset 里挑
    - 不在 memory subset 的类仍然会出现在测试流里，并且应该被 reject
    """

    rule = _memory_selection_rule(config)
    if rule is None:
        return np.unique(np.asarray(warmup_labels, dtype=np.int64))

    source = config.cam.memory_subset.selection_source
    if source == "warmup":
        return subset_by_label_rule(warmup_labels, rule).astype(np.int64)
    if source == "encoded":
        return subset_by_label_rule(encoded.labels, rule).astype(np.int64)
    if source == "pre_sampling":
        source_meta = encoded.meta.get("source_meta", {})
        label_counts = source_meta.get("label_counts_before_sampling") or source_meta.get("label_counts")
        if isinstance(label_counts, dict) and label_counts:
            return subset_by_count_map(label_counts, rule).astype(np.int64)
        return subset_by_label_rule(encoded.labels, rule).astype(np.int64)
    raise ValueError(f"Unknown cam.memory_subset.selection_source: {source}")


def run_single_experiment(config: ExperimentConfig, encoded: EncodedDataset, cache_path: Path) -> ResultBundle:
    """Run one concrete CAM experiment on one encoded dataset.

    中文：一次“具体实验变体”的核心执行函数。
    """

    warmup_idx, stream_idx = _split_indices(encoded, config)
    # 中文：warmup 段用于初始化模板，stream 段用于真正在线评估。
    warmup_bits = encoded.bits[warmup_idx]
    warmup_labels = encoded.labels[warmup_idx]

    target_memory_labels = select_memory_labels(encoded, warmup_labels, config)
    warmup_memory_mask = np.isin(warmup_labels, target_memory_labels)
    warmup_memory_bits = warmup_bits[warmup_memory_mask]
    warmup_memory_labels = warmup_labels[warmup_memory_mask]
    if warmup_memory_labels.size == 0:
        raise ValueError("No warmup spikes remain after applying cam.memory_subset")

    template_rows = build_template_rows(warmup_memory_bits, warmup_memory_labels, config.template_init)
    initial_template_count = int(template_rows.unit_ids.shape[0])
    cam = build_cam(config, initial_template_count=initial_template_count)
    cam.load_templates(template_rows)

    stream_bits = encoded.bits[stream_idx]
    stream_labels = encoded.labels[stream_idx]
    stream_times = encoded.spike_times[stream_idx]
    stream_sources = encoded.source_indices[stream_idx]

    num_steps = int(stream_bits.shape[0])
    y_true = np.empty(num_steps, dtype=np.int64)
    y_pred = np.empty(num_steps, dtype=np.int64)
    best_distance = np.empty(num_steps, dtype=np.float32)
    second_distance = np.empty(num_steps, dtype=np.float32)
    accepted = np.empty(num_steps, dtype=np.uint8)
    update_flags = np.empty(num_steps, dtype=np.uint8)
    wrong_update_flags = np.empty(num_steps, dtype=np.uint8)
    template_counts = np.empty(num_steps, dtype=np.int64)

    threshold = float(config.matcher.threshold)
    known_labels = np.unique(template_rows.unit_ids).astype(np.int64)
    missing_memory_labels = np.setdiff1d(np.unique(target_memory_labels.astype(np.int64)), known_labels, assume_unique=False)

    # 中文：这里是整个 chronological online evaluation 的核心循环。
    # 每条 spike 都按时间顺序进入 CAM。
    for step_index, (bits, true_label) in enumerate(zip(stream_bits, stream_labels)):
        step = cam.process(bits, threshold=threshold, step_index=step_index)
        predicted = step.predicted_id
        update_happened = bool(step.update.updated or step.update.allocated_row >= 0)

        target_id = None
        if step.update.updated_row >= 0:
            target_id = int(cam.neuron_ids[step.update.updated_row])
        elif step.match.best_id is not None:
            target_id = int(step.match.best_id)
        wrong_update = update_happened and target_id is not None and target_id != int(true_label)

        y_true[step_index] = int(true_label)
        y_pred[step_index] = int(predicted)
        best_distance[step_index] = float(step.match.best_distance)
        second_distance[step_index] = float(step.match.second_distance)
        accepted[step_index] = np.uint8(step.match.accepted)
        update_flags[step_index] = np.uint8(update_happened)
        wrong_update_flags[step_index] = np.uint8(wrong_update)
        template_counts[step_index] = int(step.template_count)

    confusion, confusion_labels = confusion_with_reject(y_true, y_pred)
    metrics = {}
    metrics.update(compute_classification_metrics(y_true, y_pred))
    metrics.update(compute_reject_metrics(y_true, y_pred, known_labels))
    metrics.update(compute_update_metrics(update_flags, wrong_update_flags))
    metrics.update(compute_memory_metrics(template_counts, cam.capacity, initial_template_count))
    metrics.update(
        {
            "warmup_size": int(warmup_idx.size),
            "warmup_memory_size": int(warmup_memory_labels.size),
            "warmup_ignored_size": int(warmup_labels.size - warmup_memory_labels.size),
            "stream_size": int(stream_idx.size),
            "known_label_count": int(known_labels.size),
            "target_memory_label_count": int(np.unique(target_memory_labels).size),
            "missing_memory_label_count": int(missing_memory_labels.size),
            "unknown_stream_fraction": float(np.mean(~np.isin(y_true, known_labels))) if y_true.size else 0.0,
            "memory_selection_mode": config.cam.memory_subset.mode,
            "memory_selection_source": config.cam.memory_subset.selection_source,
            "cache_path": str(cache_path),
        }
    )

    curves = compute_curve_bundle(
        y_true=y_true,
        y_pred=y_pred,
        update_flags=update_flags,
        wrong_update_flags=wrong_update_flags,
        template_counts=template_counts,
        window_size=config.evaluation.window_size,
    )

    predictions = {
        "y_true": y_true,
        "y_pred": y_pred,
        "spike_times": stream_times.astype(np.int64),
        "source_indices": stream_sources.astype(np.int64),
        "best_distance": best_distance,
        "second_distance": second_distance,
        "accepted": accepted,
        "updated": update_flags,
        "wrong_update": wrong_update_flags,
        "template_count": template_counts,
    }

    meta = {
        "experiment_name": config.experiment_name,
        "variant_name": config.variant_name,
        "variant_description": config.variant_description,
        "config": json_ready(asdict(config)),
        "encoded_dataset_meta": json_ready(encoded.meta),
        "template_meta": json_ready(template_rows.meta),
        "memory_selection": {
            "target_memory_labels": [int(label) for label in np.unique(target_memory_labels)],
            "loaded_memory_labels": [int(label) for label in known_labels],
            "missing_memory_labels": [int(label) for label in missing_memory_labels],
        },
    }

    return ResultBundle(
        experiment_name=config.experiment_name,
        variant_name=config.variant_name,
        metrics=metrics,
        predictions=predictions,
        confusion=confusion,
        confusion_labels=confusion_labels,
        curves=curves,
        meta=meta,
    )


def run_experiment_suite(
    suite: ExperimentSuiteConfig,
    *,
    selected_variants: Optional[Iterable[str]] = None,
    encode_only: bool = False,
) -> Dict[str, ResultBundle]:
    """Run one experiment suite and save its results.

    中文：执行一个 config 文件里的全部实验变体。
    """

    variant_filter = set(selected_variants or [])
    configs = expand_variants(suite)
    if variant_filter:
        configs = [cfg for cfg in configs if cfg.variant_name in variant_filter]
        if not configs:
            raise ValueError(f"No variants matched filter: {sorted(variant_filter)}")

    result_root = resolve_result_root(suite.results.results_dir) / suite.experiment_name
    result_root.mkdir(parents=True, exist_ok=True)
    save_json(result_root / "config.json", json_ready(asdict(suite)))

    bundles: Dict[str, ResultBundle] = {}
    summary: Dict[str, Any] = {}

    for config in configs:
        encoded, cache_path = prepare_encoded_dataset(config)
        variant_dir = result_root / "runs" / _safe_variant_name(config.variant_name)
        encoded_stats = compute_bit_statistics(encoded, seed=config.seed)
        save_json(variant_dir / "encoded_stats.json", encoded_stats)

        if encode_only:
            if config.results.copy_encoded_dataset:
                encoded.save_npz(variant_dir / "encoded_dataset.npz")
            summary[config.variant_name] = {"mode": "encode_only", "cache_path": str(cache_path), **encoded_stats}
            continue

        bundle = run_single_experiment(config, encoded, cache_path)
        bundle.save(
            variant_dir,
            save_predictions=config.results.save_predictions,
            save_curves=config.results.save_curves,
            save_confusion=config.results.save_confusion,
        )
        if config.results.copy_encoded_dataset:
            encoded.save_npz(variant_dir / "encoded_dataset.npz")

        bundles[config.variant_name] = bundle
        summary[config.variant_name] = bundle.metrics

    save_json(result_root / "summary.json", summary)
    return bundles


def run_experiment_suite_on_encoded_dataset(
    suite: ExperimentSuiteConfig,
    encoded: EncodedDataset,
    *,
    cache_path: Optional[Path] = None,
    selected_variants: Optional[Iterable[str]] = None,
    save_encoded_copy: bool = False,
) -> Dict[str, ResultBundle]:
    """Run one suite on an already encoded dataset.

    中文：
    这个入口主要服务两类场景：

    1. 旧项目里已经导出的 bit-level CSV / NPZ
    2. 研究对照时想固定同一份 encoded dataset，只比较 CAM 侧策略

    这样就不用强行把一切都走 raw-data -> waveform -> encoder 这条路径。
    """

    variant_filter = set(selected_variants or [])
    configs = expand_variants(suite)
    if variant_filter:
        configs = [cfg for cfg in configs if cfg.variant_name in variant_filter]
        if not configs:
            raise ValueError(f"No variants matched filter: {sorted(variant_filter)}")

    result_root = resolve_result_root(suite.results.results_dir) / suite.experiment_name
    result_root.mkdir(parents=True, exist_ok=True)
    save_json(result_root / "config.json", json_ready(asdict(suite)))

    bundles: Dict[str, ResultBundle] = {}
    summary: Dict[str, Any] = {}
    encoded_stats = compute_bit_statistics(encoded, seed=suite.seed)
    effective_cache_path = cache_path if cache_path is not None else (result_root / "provided_encoded_dataset.npz")

    for config in configs:
        variant_dir = result_root / "runs" / _safe_variant_name(config.variant_name)
        save_json(variant_dir / "encoded_stats.json", encoded_stats)

        bundle = run_single_experiment(config, encoded, effective_cache_path)
        bundle.save(
            variant_dir,
            save_predictions=config.results.save_predictions,
            save_curves=config.results.save_curves,
            save_confusion=config.results.save_confusion,
        )
        if save_encoded_copy or config.results.copy_encoded_dataset:
            encoded.save_npz(variant_dir / "encoded_dataset.npz")

        bundles[config.variant_name] = bundle
        summary[config.variant_name] = bundle.metrics

    save_json(result_root / "summary.json", summary)
    return bundles


__all__ = [
    "encoded_cache_path",
    "prepare_encoded_dataset",
    "resolve_result_root",
    "run_experiment_suite_on_encoded_dataset",
    "run_experiment_suite",
    "run_single_experiment",
]
