"""Template initialization strategies for CAM rows.

中文说明
--------
这个文件负责 warmup 阶段“初始模板怎么选”。

这部分虽然不是你的主研究对象，但会强烈影响后面动态更新的起点。
所以它被单独拆出来，方便做 initialization ablation。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from cam_core import TemplateRows
from config import TemplateInitConfig


def _check_bits_labels(bits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate encoded bits / labels arrays.

    中文：统一检查输入维度，避免模板生成阶段悄悄出错。
    """

    bit_array = np.asarray(bits, dtype=np.uint8)
    label_array = np.asarray(labels, dtype=np.int64)
    if bit_array.ndim != 2:
        raise ValueError(f"bits must be 2D, got {bit_array.shape}")
    if label_array.shape[0] != bit_array.shape[0]:
        raise ValueError("bits and labels must have the same first dimension")
    return bit_array, label_array


def _bit_stability(unit_bits: np.ndarray) -> np.ndarray:
    """Return per-bit stability in ``[0.5, 1.0]``.

    中文：一个 bit 越稳定，说明这个 neuron 在该位上越一致。
    """

    prob_one = unit_bits.mean(axis=0)
    return np.maximum(prob_one, 1.0 - prob_one)


def majority_vote_templates(bits: np.ndarray, labels: np.ndarray) -> TemplateRows:
    """Build one majority-vote template per unit.

    中文：每个 unit 的每一位都取多数表决。
    """

    bit_array, label_array = _check_bits_labels(bits, labels)
    unit_ids = np.unique(label_array)

    templates: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    counts: List[int] = []

    for unit_id in unit_ids:
        unit_bits = bit_array[label_array == unit_id]
        templates.append((unit_bits.mean(axis=0) >= 0.5).astype(np.uint8))
        masks.append(np.ones(bit_array.shape[1], dtype=np.uint8))
        weights.append(np.ones(bit_array.shape[1], dtype=np.float32))
        counts.append(int(unit_bits.shape[0]))

    return TemplateRows(
        unit_ids=unit_ids.astype(np.int64),
        templates=np.stack(templates, axis=0),
        masks=np.stack(masks, axis=0),
        weights=np.stack(weights, axis=0),
        support_counts=np.asarray(counts, dtype=np.int64),
        meta={"init_method": "majority_vote"},
    )


def medoid_templates(bits: np.ndarray, labels: np.ndarray) -> TemplateRows:
    """Build one medoid template per unit.

    The medoid is a real sample whose average Hamming distance to other
    samples in the same unit is minimal.

    中文：
    medoid 和 majority vote 的区别在于：
    它选的是“真实存在的一条样本”，而不是逐 bit 拼出来的合成模板。
    """

    bit_array, label_array = _check_bits_labels(bits, labels)
    unit_ids = np.unique(label_array)

    templates: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    counts: List[int] = []

    for unit_id in unit_ids:
        unit_bits = bit_array[label_array == unit_id]
        if unit_bits.shape[0] == 1:
            medoid = unit_bits[0]
        else:
            diff = np.bitwise_xor(unit_bits[:, None, :], unit_bits[None, :, :]).sum(axis=2)
            medoid = unit_bits[np.argmin(diff.mean(axis=1))]
        templates.append(medoid.astype(np.uint8))
        masks.append(np.ones(bit_array.shape[1], dtype=np.uint8))
        weights.append(np.ones(bit_array.shape[1], dtype=np.float32))
        counts.append(int(unit_bits.shape[0]))

    return TemplateRows(
        unit_ids=unit_ids.astype(np.int64),
        templates=np.stack(templates, axis=0),
        masks=np.stack(masks, axis=0),
        weights=np.stack(weights, axis=0),
        support_counts=np.asarray(counts, dtype=np.int64),
        meta={"init_method": "medoid"},
    )


def stable_mask_templates(bits: np.ndarray, labels: np.ndarray, stability_threshold: float) -> TemplateRows:
    """Build majority-vote templates with unstable bits masked out.

    中文：对于不稳定的 bit，初始化时先 mask 掉，不参与匹配。
    """

    bit_array, label_array = _check_bits_labels(bits, labels)
    unit_ids = np.unique(label_array)

    templates: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    counts: List[int] = []

    for unit_id in unit_ids:
        unit_bits = bit_array[label_array == unit_id]
        prob_one = unit_bits.mean(axis=0)
        stability = np.maximum(prob_one, 1.0 - prob_one)
        template = (prob_one >= 0.5).astype(np.uint8)
        mask = (stability >= stability_threshold).astype(np.uint8)
        if not np.any(mask):
            # 中文：如果一个 unit 的所有 bit 都不够稳定，仍然退回到全位匹配，
            # 否则这个 unit 会完全无法参与匹配。
            mask = np.ones(bit_array.shape[1], dtype=np.uint8)
        templates.append(template)
        masks.append(mask)
        weights.append(stability.astype(np.float32))
        counts.append(int(unit_bits.shape[0]))

    return TemplateRows(
        unit_ids=unit_ids.astype(np.int64),
        templates=np.stack(templates, axis=0),
        masks=np.stack(masks, axis=0),
        weights=np.stack(weights, axis=0),
        support_counts=np.asarray(counts, dtype=np.int64),
        meta={
            "init_method": "stable_mask",
            "stability_threshold": float(stability_threshold),
        },
    )


def multi_template_templates(bits: np.ndarray, labels: np.ndarray, templates_per_unit: int) -> TemplateRows:
    """Build multiple templates per unit using greedy farthest-point seeding.

    This is a simple, lightweight alternative to full k-medoids. It is
    intended for exploratory experiments rather than as the primary default.

    中文：
    这是一个轻量版的“每个 neuron 多模板”初始化方法，
    比较适合 exploratory experiment，不是默认主方案。
    """

    bit_array, label_array = _check_bits_labels(bits, labels)
    if templates_per_unit <= 1:
        return majority_vote_templates(bit_array, label_array)

    unit_rows: List[int] = []
    templates: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    counts: List[int] = []

    for unit_id in np.unique(label_array):
        unit_bits = bit_array[label_array == unit_id]
        seed_indices = [0]
        while len(seed_indices) < min(templates_per_unit, unit_bits.shape[0]):
            chosen = unit_bits[np.array(seed_indices)]
            dist_to_chosen = np.bitwise_xor(unit_bits[:, None, :], chosen[None, :, :]).sum(axis=2)
            best_existing = dist_to_chosen.min(axis=1)
            next_idx = int(np.argmax(best_existing))
            if next_idx in seed_indices:
                break
            seed_indices.append(next_idx)

        chosen = unit_bits[np.array(seed_indices)]
        dist = np.bitwise_xor(unit_bits[:, None, :], chosen[None, :, :]).sum(axis=2)
        assignment = np.argmin(dist, axis=1)

        for cluster_idx in range(chosen.shape[0]):
            cluster_bits = unit_bits[assignment == cluster_idx]
            if cluster_bits.size == 0:
                cluster_bits = chosen[cluster_idx : cluster_idx + 1]
            template = (cluster_bits.mean(axis=0) >= 0.5).astype(np.uint8)
            stability = _bit_stability(cluster_bits).astype(np.float32)

            unit_rows.append(int(unit_id))
            templates.append(template)
            masks.append(np.ones(bit_array.shape[1], dtype=np.uint8))
            weights.append(stability)
            counts.append(int(cluster_bits.shape[0]))

    return TemplateRows(
        unit_ids=np.asarray(unit_rows, dtype=np.int64),
        templates=np.stack(templates, axis=0),
        masks=np.stack(masks, axis=0),
        weights=np.stack(weights, axis=0),
        support_counts=np.asarray(counts, dtype=np.int64),
        meta={
            "init_method": "multi_template",
            "templates_per_unit": int(templates_per_unit),
        },
    )


def build_template_rows(bits: np.ndarray, labels: np.ndarray, cfg: TemplateInitConfig) -> TemplateRows:
    """Factory function for initial template construction.

    中文：模板初始化总入口，runner 最终就是调用这里。
    """

    method = cfg.method
    if method == "majority_vote":
        return majority_vote_templates(bits, labels)
    if method == "medoid":
        return medoid_templates(bits, labels)
    if method == "stable_mask":
        return stable_mask_templates(bits, labels, cfg.stable_mask_threshold)
    if method == "multi_template":
        return multi_template_templates(bits, labels, cfg.multi_template_per_unit)
    raise ValueError(f"Unknown template init method: {method}")


__all__ = [
    "build_template_rows",
    "majority_vote_templates",
    "medoid_templates",
    "multi_template_templates",
    "stable_mask_templates",
]
