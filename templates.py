from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def majority_vote_templates(bits: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    最常用的初始模板策略：对每个 unit，在训练集上做逐 bit 多数表决。

    输入：
    - bits:   (N, B) 0/1 ndarray
    - labels: (N,)   int ndarray

    输出：
    - unit_ids: (U,) int，排序后的所有 unit id
    - templates: (U, B) 0/1 ndarray，对应 unit 的模板
    """
    bits = np.asarray(bits).astype(int)
    labels = np.asarray(labels).astype(int)

    if bits.ndim != 2:
        raise ValueError(f"bits 必须是 2D (N,B)，现在是 {bits.shape}")
    if labels.shape[0] != bits.shape[0]:
        raise ValueError("bits 与 labels 行数不一致")

    unit_ids = np.unique(labels)
    templates = []

    for u in unit_ids:
        mask = labels == u
        if not np.any(mask):
            continue
        m = bits[mask]  # (n_u, B)
        fp = (m.mean(axis=0) >= 0.5).astype(int)
        templates.append(fp)

    if not templates:
        raise RuntimeError("没有任何 unit 可以生成模板（检查 labels 是否为空）")

    templates_arr = np.stack(templates, axis=0)
    return unit_ids.astype(int), templates_arr


def first_sample_templates(bits: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    另一种简单策略：每个 unit 取第一条样本的 bits 作为模板。
    """
    bits = np.asarray(bits).astype(int)
    labels = np.asarray(labels).astype(int)

    if bits.ndim != 2:
        raise ValueError(f"bits 必须是 2D (N,B)，现在是 {bits.shape}")
    if labels.shape[0] != bits.shape[0]:
        raise ValueError("bits 与 labels 行数不一致")

    unit_ids = np.unique(labels)
    templates = []
    for u in unit_ids:
        idx = np.flatnonzero(labels == u)
        if idx.size == 0:
            continue
        templates.append(bits[idx[0]])

    if not templates:
        raise RuntimeError("没有任何 unit 可以生成模板")

    templates_arr = np.stack(templates, axis=0)
    return unit_ids.astype(int), templates_arr


__all__ = [
    "majority_vote_templates",
    "first_sample_templates",
]

