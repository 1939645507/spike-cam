from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type

import numpy as np

from cam_core import CAM
from match_strategies import HammingNearestMatch
from templates import majority_vote_templates
from update_strategies import (
    CounterUpdate,
    DualTemplateUpdate,
    GrowingUpdate,
    MarginEmaUpdate,
    NoUpdate,
    ProbabilisticUpdate,
    ConfidenceWeightedUpdate,
)


@dataclass
class CamRunResult:
    name: str
    accuracy: float
    unit_ids: np.ndarray
    confusion: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    extra: Dict[str, Any]


def evaluate_cam_on_encoded_dataset(
    *,
    bits: np.ndarray,
    labels: np.ndarray,
    cam_variants: Sequence[Tuple[str, str]],
    train_frac: float = 0.7,
    threshold: int = 2,
    seed: int = 42,
) -> Dict[str, CamRunResult]:
    """
    在“已编码好的数据集（bits + id）”上评估多种 CAM 算法。

    参数：
    - bits:   (N, B) 0/1 ndarray
    - labels: (N,)   int ndarray
    - cam_variants:
        一个列表，里面的元素是 (name, mode)，mode 目前支持：
        - \"static\"      : 只用 HammingNearestMatch（无动态更新）
        - \"counter\"     : HammingNearestMatch + CounterUpdate
        - \"margin_ema\"  : HammingNearestMatch + MarginEmaUpdate
        - \"conf_weight\" : HammingNearestMatch + ConfidenceWeightedUpdate
        - \"dual\"        : HammingNearestMatch + DualTemplateUpdate
        - \"prob\"        : HammingNearestMatch + ProbabilisticUpdate
        - \"growing\"     : HammingNearestMatch + GrowingUpdate

      你以后可以把这个列表当成配置文件，随时加新模式。

    返回：
    - dict[name] -> CamRunResult
    """
    bits = np.asarray(bits).astype(int)
    labels = np.asarray(labels).astype(int)

    if bits.ndim != 2:
        raise ValueError(f"bits 必须是 2D (N,B)，现在是 {bits.shape}")
    if labels.shape[0] != bits.shape[0]:
        raise ValueError("bits 与 labels 行数不一致")

    rng = np.random.default_rng(seed)
    n_total = bits.shape[0]
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_train = int(np.floor(n_total * train_frac))
    idx_train = idx[:n_train]
    idx_test = idx[n_train:]

    bits_tr, y_tr = bits[idx_train], labels[idx_train]
    bits_te, y_te = bits[idx_test], labels[idx_test]

    unit_ids, templates = majority_vote_templates(bits_tr, y_tr)

    results: Dict[str, CamRunResult] = {}

    for name, mode in cam_variants:
        match = HammingNearestMatch()
        if mode == "static":
            update = NoUpdate(base_match=match)
        elif mode == "counter":
            update = CounterUpdate(base_match=match, max_confidence=10)
        elif mode == "margin_ema":
            update = MarginEmaUpdate(base_match=match, alpha=0.05)
        elif mode == "conf_weight":
            update = ConfidenceWeightedUpdate(base_match=match, lr=0.1, max_conf=5.0)
        elif mode == "dual":
            update = DualTemplateUpdate(base_match=match, alpha=0.1)
        elif mode == "prob":
            update = ProbabilisticUpdate(base_match=match, alpha=0.05, eps=1e-4)
        elif mode == "growing":
            update = GrowingUpdate(base_match=match, split_threshold=3)
        else:
            raise ValueError(f"未知 CAM 模式: {mode}")

        cap = max(len(unit_ids) * 2, len(unit_ids) + 4)
        bit_width = bits.shape[1]

        cam = CAM(
            capacity=cap,
            bit_width=bit_width,
            match_strategy=match,
            update_strategy=update,
        )
        cam.load_templates(unit_ids, templates)

        y_true: List[int] = []
        y_pred: List[int] = []

        for xb, yb in zip(bits_te, y_te):
            nid, row_idx, dist = cam.match_and_update(xb, threshold=threshold)
            pred = -1 if nid is None else int(nid)
            y_true.append(int(yb))
            y_pred.append(pred)

        y_true_arr = np.array(y_true, dtype=int)
        y_pred_arr = np.array(y_pred, dtype=int)
        valid = y_pred_arr != -1
        if np.any(valid):
            acc = float(np.mean(y_true_arr[valid] == y_pred_arr[valid]))
        else:
            acc = 0.0

        all_units = np.unique(np.concatenate([y_true_arr, y_pred_arr[y_pred_arr != -1]]))
        all_units = np.sort(all_units)
        u2i = {u: i for i, u in enumerate(all_units)}
        K = all_units.size
        confusion = np.zeros((K, K), dtype=int)
        for t, p in zip(y_true_arr[valid], y_pred_arr[valid]):
            confusion[u2i[t], u2i[p]] += 1

        results[name] = CamRunResult(
            name=name,
            accuracy=acc,
            unit_ids=all_units,
            confusion=confusion,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            extra={
                "mode": mode,
                "threshold": threshold,
                "train_frac": train_frac,
            },
        )

    return results


__all__ = [
    "CamRunResult",
    "evaluate_cam_on_encoded_dataset",
]

