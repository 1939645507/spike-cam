"""Metrics, online curves, and result bundles for spike CAM experiments.

中文说明
--------
这个文件负责“怎么算实验结果”。

这里的重点是：

- 不只算最终 accuracy
- 要保留 reject 相关指标
- 要保留 online 过程曲线
- 要保留 memory / update 相关统计

这样后面你做论文图表时，数据基本都能直接从结果目录里取。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from config import ensure_parent, json_ready, save_json
from dataio import REJECT_LABEL


@dataclass
class ResultBundle:
    """All saved outputs for one concrete experiment variant.

    中文：一次实验变体运行后的完整结果包。
    """

    experiment_name: str
    variant_name: str
    metrics: Dict[str, Any]
    predictions: Dict[str, np.ndarray]
    confusion: np.ndarray
    confusion_labels: np.ndarray
    curves: Dict[str, np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, run_dir: Path, *, save_predictions: bool = True, save_curves: bool = True, save_confusion: bool = True) -> None:
        """Persist the result bundle to a directory.

        中文：统一把实验结果保存到规范目录结构中。
        """

        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "metrics.json", self.metrics)
        save_json(run_dir / "meta.json", self.meta)

        if save_predictions:
            np.savez_compressed(run_dir / "predictions.npz", **self.predictions)
        if save_curves:
            np.savez_compressed(run_dir / "curves.npz", **self.curves)
        if save_confusion:
            np.save(run_dir / "confusion.npy", self.confusion)
            np.save(run_dir / "confusion_labels.npy", self.confusion_labels)


def confusion_with_reject(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a square confusion matrix including the reject label.

    中文：把 reject 也作为一个特殊预测标签纳入 confusion matrix。
    """

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    label_array = np.asarray(labels, dtype=np.int64)
    label_to_index = {int(label): idx for idx, label in enumerate(label_array)}
    confusion = np.zeros((label_array.size, label_array.size), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        confusion[label_to_index[int(truth)], label_to_index[int(pred)]] += 1
    return confusion, label_array


def _per_class_prf(y_true: np.ndarray, y_pred: np.ndarray, class_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class precision / recall / F1 without external dependencies."""

    precision = np.zeros(class_labels.size, dtype=np.float64)
    recall = np.zeros(class_labels.size, dtype=np.float64)
    f1 = np.zeros(class_labels.size, dtype=np.float64)

    for idx, label in enumerate(class_labels):
        truth_mask = y_true == label
        pred_mask = y_pred == label
        tp = int(np.sum(truth_mask & pred_mask))
        fp = int(np.sum((~truth_mask) & pred_mask))
        fn = int(np.sum(truth_mask & (~pred_mask)))

        precision[idx] = tp / max(1, tp + fp)
        recall[idx] = tp / max(1, tp + fn)
        denom = precision[idx] + recall[idx]
        f1[idx] = 0.0 if denom <= 1e-12 else (2.0 * precision[idx] * recall[idx] / denom)
    return precision, recall, f1


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, macro-F1, and balanced accuracy.

    Rejects count as incorrect predictions for the true class.

    中文：这里的 accuracy 是严格口径，reject 直接算错，不会被排除。
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    class_labels = np.unique(y_true)
    _, recall, f1 = _per_class_prf(y_true, y_pred, class_labels)
    accepted = y_pred != REJECT_LABEL

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "accept_rate": float(np.mean(accepted)),
        "accepted_accuracy": float(np.mean(y_true[accepted] == y_pred[accepted])) if np.any(accepted) else 0.0,
        "macro_f1": float(np.mean(f1)) if f1.size else 0.0,
        "balanced_accuracy": float(np.mean(recall)) if recall.size else 0.0,
    }


def compute_reject_metrics(y_true: np.ndarray, y_pred: np.ndarray, known_labels: np.ndarray) -> Dict[str, float]:
    """Compute reject-related metrics.

    ``known_labels`` are the labels present in the initial CAM templates.

    中文：
    这里的 known / unknown 是相对于“初始模板集合”来说的，
    不是相对于整个数据集标签空间。
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    known = np.isin(y_true, np.asarray(known_labels, dtype=np.int64))
    rejected = y_pred == REJECT_LABEL

    false_reject = known & rejected
    false_accept = (~known) & (~rejected)

    return {
        "reject_rate": float(np.mean(rejected)),
        "false_reject_count": int(np.sum(false_reject)),
        "false_reject_rate": float(np.mean(false_reject)) if y_true.size else 0.0,
        "false_accept_count": int(np.sum(false_accept)),
        "false_accept_rate": float(np.mean(false_accept)) if y_true.size else 0.0,
    }


def compute_curve_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    update_flags: np.ndarray,
    wrong_update_flags: np.ndarray,
    template_counts: np.ndarray,
    *,
    window_size: int,
) -> Dict[str, np.ndarray]:
    """Build online curves saved for later plotting.

    中文：生成在线过程曲线，方便后面画图。
    """

    correct = (y_true == y_pred).astype(np.int64)
    cumulative_steps = np.arange(1, y_true.shape[0] + 1, dtype=np.int64)
    cumulative_accuracy = np.cumsum(correct) / cumulative_steps
    cumulative_updates = np.cumsum(update_flags.astype(np.int64))
    cumulative_wrong_updates = np.cumsum(wrong_update_flags.astype(np.int64))

    window_start = np.arange(0, y_true.shape[0], max(1, int(window_size)), dtype=np.int64)
    per_window_accuracy = np.zeros(window_start.size, dtype=np.float64)
    per_window_reject_rate = np.zeros(window_start.size, dtype=np.float64)
    per_window_updates = np.zeros(window_start.size, dtype=np.int64)
    per_window_wrong_updates = np.zeros(window_start.size, dtype=np.int64)

    for idx, start in enumerate(window_start):
        stop = min(y_true.shape[0], start + max(1, int(window_size)))
        window_slice = slice(start, stop)
        per_window_accuracy[idx] = float(np.mean(correct[window_slice]))
        per_window_reject_rate[idx] = float(np.mean(y_pred[window_slice] == REJECT_LABEL))
        per_window_updates[idx] = int(np.sum(update_flags[window_slice]))
        per_window_wrong_updates[idx] = int(np.sum(wrong_update_flags[window_slice]))

    return {
        "step_index": cumulative_steps,
        "cumulative_accuracy": cumulative_accuracy.astype(np.float32),
        "cumulative_updates": cumulative_updates.astype(np.int64),
        "cumulative_wrong_updates": cumulative_wrong_updates.astype(np.int64),
        "template_count": template_counts.astype(np.int64),
        "window_start_index": window_start,
        "per_window_accuracy": per_window_accuracy.astype(np.float32),
        "per_window_reject_rate": per_window_reject_rate.astype(np.float32),
        "per_window_updates": per_window_updates.astype(np.int64),
        "per_window_wrong_updates": per_window_wrong_updates.astype(np.int64),
    }


def compute_memory_metrics(
    template_counts: np.ndarray,
    capacity: int,
    initial_template_count: int,
) -> Dict[str, float]:
    """Summarize CAM memory usage over the experiment.

    中文：统计模板数量随时间变化和容量使用情况。
    """

    template_counts = np.asarray(template_counts, dtype=np.int64)
    final_count = int(template_counts[-1]) if template_counts.size else int(initial_template_count)
    max_count = int(template_counts.max()) if template_counts.size else int(initial_template_count)
    return {
        "initial_template_count": int(initial_template_count),
        "final_template_count": final_count,
        "max_template_count": max_count,
        "template_growth": int(final_count - initial_template_count),
        "used_rows_over_capacity": float(final_count / max(1, capacity)),
        "max_rows_over_capacity": float(max_count / max(1, capacity)),
    }


def compute_update_metrics(update_flags: np.ndarray, wrong_update_flags: np.ndarray) -> Dict[str, float]:
    """Summarize update counts.

    中文：统计一共更新了多少次，以及错误更新比例。
    """

    total_updates = int(np.sum(update_flags))
    wrong_updates = int(np.sum(wrong_update_flags))
    return {
        "update_count": total_updates,
        "wrong_update_count": wrong_updates,
        "wrong_update_rate": float(wrong_updates / max(1, total_updates)),
    }


__all__ = [
    "ResultBundle",
    "compute_classification_metrics",
    "compute_curve_bundle",
    "compute_memory_metrics",
    "compute_reject_metrics",
    "compute_update_metrics",
    "confusion_with_reject",
]
