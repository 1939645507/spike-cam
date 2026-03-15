"""Matching strategies for the spike CAM platform.

中文说明
--------
这个文件只负责“怎么匹配”，不负责更新模板。

输入：
- 当前 CAM 状态
- 一条输入 bits
- threshold

输出：
- 最优匹配行
- 匹配到的 neuron id
- 距离 / 分数
- top-2 信息

把匹配和更新分开，是为了后面做更干净的算法对比。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cam_core import CAM, MatchResult, MatchStrategy


def _sorted_candidate_rows(cam: CAM, input_bits: np.ndarray, distance_fn) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate rows sorted by distance.

    中文：先对所有已使用模板行算距离，再按距离从小到大排序。
    """

    used = cam.used_row_indices
    if used.size == 0:
        return used, np.empty(0, dtype=np.float32)

    distances = np.array([distance_fn(row_idx) for row_idx in used], dtype=np.float32)
    order = np.argsort(distances, kind="mergesort")
    return used[order], distances[order]


@dataclass
class HammingNearestMatch(MatchStrategy):
    """Nearest-neighbor matching with masked Hamming distance.

    中文：最基本的最近邻匹配，也是很多对照实验的 baseline。
    """

    def match(self, cam: CAM, input_bits: np.ndarray, threshold: float) -> MatchResult:
        def distance_fn(row_idx: int) -> float:
            return float(
                CAM.hamming_distance(
                    input_bits,
                    cam.templates[row_idx],
                    mask=cam.masks[row_idx],
                )
            )

        rows, distances = _sorted_candidate_rows(cam, input_bits, distance_fn)
        if rows.size == 0:
            return MatchResult(
                best_id=None,
                matched_id=None,
                best_row=-1,
                best_distance=float("inf"),
                accepted=False,
                threshold=float(threshold),
            )

        best_row = int(rows[0])
        best_distance = float(distances[0])
        second_row = int(rows[1]) if rows.size > 1 else -1
        second_distance = float(distances[1]) if rows.size > 1 else float("inf")
        best_id = int(cam.neuron_ids[best_row])
        accepted = best_distance <= threshold

        return MatchResult(
            best_id=best_id,
            matched_id=best_id if accepted else None,
            best_row=best_row,
            best_distance=best_distance,
            accepted=accepted,
            threshold=float(threshold),
            second_row=second_row,
            second_distance=second_distance,
        )


@dataclass
class WeightedHammingMatch(MatchStrategy):
    """Nearest-neighbor matching using per-row bit weights.

    中文：和普通 Hamming 不同，这里不同 bit 的重要性可以不同。
    """

    def match(self, cam: CAM, input_bits: np.ndarray, threshold: float) -> MatchResult:
        def distance_fn(row_idx: int) -> float:
            return CAM.weighted_hamming_distance(
                input_bits,
                cam.templates[row_idx],
                mask=cam.masks[row_idx],
                weights=cam.match_weights[row_idx],
            )

        rows, distances = _sorted_candidate_rows(cam, input_bits, distance_fn)
        if rows.size == 0:
            return MatchResult(
                best_id=None,
                matched_id=None,
                best_row=-1,
                best_distance=float("inf"),
                accepted=False,
                threshold=float(threshold),
            )

        best_row = int(rows[0])
        best_distance = float(distances[0])
        second_row = int(rows[1]) if rows.size > 1 else -1
        second_distance = float(distances[1]) if rows.size > 1 else float("inf")
        best_id = int(cam.neuron_ids[best_row])
        accepted = best_distance <= threshold

        return MatchResult(
            best_id=best_id,
            matched_id=best_id if accepted else None,
            best_row=best_row,
            best_distance=best_distance,
            accepted=accepted,
            threshold=float(threshold),
            second_row=second_row,
            second_distance=second_distance,
        )


@dataclass
class MarginRejectMatch(MatchStrategy):
    """Reject matches that are too close to the distance threshold.

    中文：即使 best distance 已经低于 threshold，
    如果离 threshold 太近，也可以保守地拒识。
    """

    min_accept_margin: float = 1.0

    def match(self, cam: CAM, input_bits: np.ndarray, threshold: float) -> MatchResult:
        base = HammingNearestMatch().match(cam, input_bits, threshold)
        if not base.accepted:
            return base

        if base.acceptance_margin < self.min_accept_margin:
            base.accepted = False
            base.matched_id = None
            base.extras["reject_reason"] = "threshold_margin"
        return base


@dataclass
class Top2MarginMatch(MatchStrategy):
    """Reject matches whose best-vs-second-best gap is too small.

    中文：如果第一名和第二名差得不够开，就认为这次匹配不够确定。
    """

    min_margin: float = 2.0

    def match(self, cam: CAM, input_bits: np.ndarray, threshold: float) -> MatchResult:
        base = HammingNearestMatch().match(cam, input_bits, threshold)
        if not base.accepted:
            return base

        if base.top2_margin < self.min_margin:
            base.accepted = False
            base.matched_id = None
            base.extras["reject_reason"] = "top2_margin"
        return base


__all__ = [
    "HammingNearestMatch",
    "MarginRejectMatch",
    "Top2MarginMatch",
    "WeightedHammingMatch",
]
