"""Dynamic update strategies for CAM templates.

Each strategy receives the already-computed match result, which keeps the
matching and updating concerns cleanly separated.

中文说明
--------
这个文件是整个毕设最核心的算法区之一。

这里每个类都代表一种“模板动态更新策略”。
它们的共同特点是：

- 先使用 matcher 得到匹配结果
- 再决定是否更新模板
- 可以维护自己的内部 state

这样就能很方便地比较：

- 不更新 vs 更新
- 保守更新 vs 激进更新
- 单模板 vs 可塑模板
- 固定容量 vs 可增长模板
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cam_core import CAM, MatchResult, UpdateResult, UpdateStrategy


def _count_bit_changes(before: np.ndarray, after: np.ndarray) -> int:
    """Return how many bit positions changed.

    中文：统计一次更新里到底翻了多少个 bit。
    """

    return int(np.sum(np.asarray(before, dtype=np.uint8) != np.asarray(after, dtype=np.uint8)))


@dataclass
class NoUpdate(UpdateStrategy):
    """Control strategy that keeps templates fixed.

    中文：静态模板对照组。
    """

    def initialize_state(self, cam: CAM) -> None:
        return None

    def update(self, cam: CAM, input_bits: np.ndarray, match: MatchResult, step_index: int) -> UpdateResult:
        return UpdateResult(updated=False, reason="disabled")


@dataclass
class CounterUpdate(UpdateStrategy):
    """Bit-wise hysteresis update.

    A signed confidence counter is maintained for each bit:

    - repeated evidence for bit ``1`` pushes the counter positive
    - repeated evidence for bit ``0`` pushes the counter negative
    - the stored binary template follows the sign of the counter

    This makes template flips deliberate rather than instantaneous.

    中文：
    这是一个比较“稳”的更新策略。
    不会因为一次 mismatch 就立刻翻转模板位，而是要累计证据。
    """

    max_confidence: int = 12

    def initialize_state(self, cam: CAM) -> None:
        signed = np.where(cam.templates > 0, self.max_confidence // 2, -(self.max_confidence // 2))
        cam.state["counter_signed"] = signed.astype(np.int16)

    def update(self, cam: CAM, input_bits: np.ndarray, match: MatchResult, step_index: int) -> UpdateResult:
        if not match.accepted or match.best_row < 0:
            return UpdateResult(updated=False, reason="not_accepted")

        counters = cam.state["counter_signed"]
        row = match.best_row
        before = cam.templates[row].copy()

        signed_input = np.where(input_bits > 0, 1, -1).astype(np.int16)
        counters[row] = np.clip(counters[row] + signed_input, -self.max_confidence, self.max_confidence)
        cam.templates[row] = (counters[row] >= 0).astype(np.uint8)

        changed = _count_bit_changes(before, cam.templates[row])
        return UpdateResult(
            updated=changed > 0,
            updated_row=row,
            updated_bits=changed,
            reason="counter",
        )


@dataclass
class MarginEmaUpdate(UpdateStrategy):
    """EMA update applied only to matches close to the threshold boundary.

    中文：只对“接近 decision boundary”的样本做 EMA 更新。
    """

    alpha: float = 0.05
    margin_band: float = 1.0

    def initialize_state(self, cam: CAM) -> None:
        cam.state["ema_templates"] = cam.templates.astype(np.float32)

    def update(self, cam: CAM, input_bits: np.ndarray, match: MatchResult, step_index: int) -> UpdateResult:
        if not match.accepted or match.best_row < 0:
            return UpdateResult(updated=False, reason="not_accepted")
        if match.acceptance_margin > self.margin_band:
            return UpdateResult(updated=False, reason="outside_margin_band")

        ema = cam.state["ema_templates"]
        row = match.best_row
        before = cam.templates[row].copy()

        ema[row] = (1.0 - self.alpha) * ema[row] + self.alpha * input_bits.astype(np.float32)
        cam.templates[row] = (ema[row] >= 0.5).astype(np.uint8)

        changed = _count_bit_changes(before, cam.templates[row])
        return UpdateResult(
            updated=changed > 0,
            updated_row=row,
            updated_bits=changed,
            reason="margin_ema",
        )


@dataclass
class ConfidenceWeightedUpdate(UpdateStrategy):
    """Update both template bits and per-bit match weights.

    Matched bits gain confidence. Mismatched bits lose confidence and can flip
    once their confidence falls below ``flip_threshold``.

    中文：
    这个策略除了更新模板，还会更新每个位的可信度，
    因此既影响 update，也能影响 weighted matching。
    """

    lr: float = 0.15
    max_conf: float = 5.0
    min_weight: float = 0.25
    flip_threshold: float = 0.9

    def initialize_state(self, cam: CAM) -> None:
        cam.state["bit_confidence"] = np.ones((cam.capacity, cam.bit_width), dtype=np.float32)
        cam.match_weights[:] = 1.0

    def update(self, cam: CAM, input_bits: np.ndarray, match: MatchResult, step_index: int) -> UpdateResult:
        if not match.accepted or match.best_row < 0:
            return UpdateResult(updated=False, reason="not_accepted")

        confidence = cam.state["bit_confidence"]
        row = match.best_row
        before = cam.templates[row].copy()

        same = input_bits == cam.templates[row]
        confidence[row][same] += self.lr
        confidence[row][~same] -= self.lr
        confidence[row] = np.clip(confidence[row], self.min_weight, self.max_conf)
        cam.match_weights[row] = confidence[row]

        flip_mask = (~same) & (confidence[row] <= self.flip_threshold)
        if np.any(flip_mask):
            cam.templates[row][flip_mask] = input_bits[flip_mask]
            # After a committed flip, restore medium confidence at that bit.
            confidence[row][flip_mask] = 1.0
            cam.match_weights[row][flip_mask] = 1.0

        changed = _count_bit_changes(before, cam.templates[row])
        return UpdateResult(
            updated=changed > 0,
            updated_row=row,
            updated_bits=changed,
            reason="confidence_weighted",
        )


@dataclass
class DualTemplateUpdate(UpdateStrategy):
    """Stable/plastic dual-template update.

    ``cam.templates`` holds the stable binary template used for matching.
    A floating plastic template tracks recent observations using EMA.
    Stable bits commit only when the plastic estimate becomes confident.

    中文：
    这个策略把模板分成：

    - stable template：真正用于匹配的模板
    - plastic template：更容易适应最近数据的可塑模板
    """

    alpha: float = 0.1
    commit_threshold: float = 0.2

    def initialize_state(self, cam: CAM) -> None:
        cam.state["plastic_templates"] = cam.templates.astype(np.float32)

    def update(self, cam: CAM, input_bits: np.ndarray, match: MatchResult, step_index: int) -> UpdateResult:
        if not match.accepted or match.best_row < 0:
            return UpdateResult(updated=False, reason="not_accepted")

        plastic = cam.state["plastic_templates"]
        row = match.best_row
        before = cam.templates[row].copy()

        plastic[row] = (1.0 - self.alpha) * plastic[row] + self.alpha * input_bits.astype(np.float32)
        target_bits = (plastic[row] >= 0.5).astype(np.uint8)
        confidence = np.abs(plastic[row] - 0.5)
        commit_mask = (target_bits != cam.templates[row]) & (confidence >= self.commit_threshold)
        if np.any(commit_mask):
            cam.templates[row][commit_mask] = target_bits[commit_mask]

        changed = _count_bit_changes(before, cam.templates[row])
        return UpdateResult(
            updated=changed > 0,
            updated_row=row,
            updated_bits=changed,
            reason="dual_template",
        )


@dataclass
class ProbabilisticUpdate(UpdateStrategy):
    """Probabilistic template update with optional confidence reuse for matching.

    中文：每个位不只保存 0/1，还保存一个“为 1 的概率”。
    """

    alpha: float = 0.05

    def initialize_state(self, cam: CAM) -> None:
        cam.state["bit_probability"] = cam.templates.astype(np.float32)

    def update(self, cam: CAM, input_bits: np.ndarray, match: MatchResult, step_index: int) -> UpdateResult:
        if not match.accepted or match.best_row < 0:
            return UpdateResult(updated=False, reason="not_accepted")

        probability = cam.state["bit_probability"]
        row = match.best_row
        before = cam.templates[row].copy()

        probability[row] = (1.0 - self.alpha) * probability[row] + self.alpha * input_bits.astype(np.float32)
        cam.templates[row] = (probability[row] >= 0.5).astype(np.uint8)
        cam.match_weights[row] = np.clip(np.abs(probability[row] - 0.5) * 2.0, 0.25, 2.0)

        changed = _count_bit_changes(before, cam.templates[row])
        return UpdateResult(
            updated=changed > 0,
            updated_row=row,
            updated_bits=changed,
            reason="probabilistic",
        )


@dataclass
class GrowingUpdate(UpdateStrategy):
    """Add new rows when one template appears insufficient.

    The new row inherits the nearest unit id and stores the current sample as
    an additional prototype. This lets one neuron occupy multiple templates.

    中文：
    当一个 unit 的单个模板不够表达其变化时，可以给它再长出一行模板。
    """

    split_threshold: float = 6.0
    allow_evict: bool = False

    def initialize_state(self, cam: CAM) -> None:
        return None

    def update(self, cam: CAM, input_bits: np.ndarray, match: MatchResult, step_index: int) -> UpdateResult:
        if match.best_row < 0 or match.best_id is None:
            return UpdateResult(updated=False, reason="no_candidate_row")
        if match.best_distance < self.split_threshold:
            return UpdateResult(updated=False, reason="distance_below_split_threshold")

        allocated_row = -1
        evicted_row = -1
        if cam.free_rows > 0:
            allocated_row = cam.allocate_row(
                neuron_id=int(match.best_id),
                bits=input_bits,
                mask=cam.masks[match.best_row],
                weights=cam.match_weights[match.best_row],
                step_index=step_index,
            )
        elif self.allow_evict:
            evicted_row = cam.select_evict_row()
            cam.replace_row(
                row_idx=evicted_row,
                neuron_id=int(match.best_id),
                bits=input_bits,
                mask=cam.masks[match.best_row],
                weights=cam.match_weights[match.best_row],
                step_index=step_index,
            )
            allocated_row = evicted_row
        else:
            return UpdateResult(updated=False, reason="cam_full")

        return UpdateResult(
            updated=True,
            updated_row=allocated_row,
            allocated_row=allocated_row,
            evicted_row=evicted_row,
            updated_bits=int(np.sum(cam.masks[allocated_row])),
            reason="growing",
        )


@dataclass
class CooldownUpdate(UpdateStrategy):
    """Simple EMA update gated by a per-row cooldown period.

    中文：避免同一行在很短时间内被连续更新得太激进。
    """

    alpha: float = 0.05
    cooldown_steps: int = 50

    def initialize_state(self, cam: CAM) -> None:
        cam.state["cooldown_ema"] = cam.templates.astype(np.float32)

    def update(self, cam: CAM, input_bits: np.ndarray, match: MatchResult, step_index: int) -> UpdateResult:
        if not match.accepted or match.best_row < 0:
            return UpdateResult(updated=False, reason="not_accepted")

        row = match.best_row
        last_step = int(cam.row_last_update_step[row])
        if last_step >= 0 and step_index - last_step < self.cooldown_steps:
            return UpdateResult(updated=False, reason="cooldown")

        ema = cam.state["cooldown_ema"]
        before = cam.templates[row].copy()
        ema[row] = (1.0 - self.alpha) * ema[row] + self.alpha * input_bits.astype(np.float32)
        cam.templates[row] = (ema[row] >= 0.5).astype(np.uint8)

        changed = _count_bit_changes(before, cam.templates[row])
        return UpdateResult(
            updated=changed > 0,
            updated_row=row,
            updated_bits=changed,
            reason="cooldown_ema",
        )


@dataclass
class Top2MarginUpdate(UpdateStrategy):
    """EMA update allowed only when the match is clearly better than runner-up.

    中文：只有当第一名明显优于第二名时，才允许做更新。
    """

    alpha: float = 0.05
    min_margin: float = 2.0

    def initialize_state(self, cam: CAM) -> None:
        cam.state["top2_ema"] = cam.templates.astype(np.float32)

    def update(self, cam: CAM, input_bits: np.ndarray, match: MatchResult, step_index: int) -> UpdateResult:
        if not match.accepted or match.best_row < 0:
            return UpdateResult(updated=False, reason="not_accepted")
        if match.top2_margin < self.min_margin:
            return UpdateResult(updated=False, reason="top2_margin_too_small")

        ema = cam.state["top2_ema"]
        row = match.best_row
        before = cam.templates[row].copy()
        ema[row] = (1.0 - self.alpha) * ema[row] + self.alpha * input_bits.astype(np.float32)
        cam.templates[row] = (ema[row] >= 0.5).astype(np.uint8)

        changed = _count_bit_changes(before, cam.templates[row])
        return UpdateResult(
            updated=changed > 0,
            updated_row=row,
            updated_bits=changed,
            reason="top2_margin_ema",
        )


__all__ = [
    "ConfidenceWeightedUpdate",
    "CooldownUpdate",
    "CounterUpdate",
    "DualTemplateUpdate",
    "GrowingUpdate",
    "MarginEmaUpdate",
    "NoUpdate",
    "ProbabilisticUpdate",
    "Top2MarginUpdate",
]
