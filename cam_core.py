"""Core CAM data structures.

This module does not hard-code one particular algorithm. Instead it provides:

- a generic :class:`CAM` storage / bookkeeping object
- typed match / update result objects
- protocol-style interfaces for matchers and update strategies

The implementation is designed for algorithm experimentation rather than
hardware-accurate simulation. The focus is:

- clear row state
- reproducible online processing
- easy extension of initialization / matching / update logic

中文说明
--------
这个文件是 CAM 端的“核心存储层”。

它尽量不直接写死任何具体算法，而是把系统拆成两部分：

- ``CAM``: 负责保存模板、mask、行状态、容量管理
- ``MatchStrategy`` / ``UpdateStrategy``: 负责算法逻辑

这样做的好处是：

- 换匹配算法时不用改 CAM 存储逻辑
- 换动态更新算法时不用改数据流
- 更适合做论文里的 ablation / comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import numpy as np


@dataclass
class MatchResult:
    """Result returned by one matching strategy.

    中文：一次匹配的完整结果，不仅有预测 id，还保留距离和 top-2 信息。
    """

    best_id: Optional[int]
    matched_id: Optional[int]
    best_row: int
    best_distance: float
    accepted: bool
    threshold: float
    second_row: int = -1
    second_distance: float = float("inf")
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def acceptance_margin(self) -> float:
        """Positive when the match is comfortably inside the threshold."""

        return float(self.threshold - self.best_distance)

    @property
    def top2_margin(self) -> float:
        """Distance gap between the best and second-best rows."""

        return float(self.second_distance - self.best_distance)


@dataclass
class UpdateResult:
    """Result returned by one update strategy.

    中文：一次更新动作的结果，用于记录是否更新、更新了哪一行、为什么更新。
    """

    updated: bool = False
    updated_row: int = -1
    allocated_row: int = -1
    evicted_row: int = -1
    updated_bits: int = 0
    reason: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """One online CAM processing step.

    中文：在线评估中的一个 time step，包含 match 和 update 两部分信息。
    """

    match: MatchResult
    update: UpdateResult
    template_count: int

    @property
    def predicted_id(self) -> int:
        """Return the accepted id or ``-1`` for reject."""

        return -1 if self.match.matched_id is None else int(self.match.matched_id)


class MatchStrategy(Protocol):
    """Protocol for match strategies.

    中文：所有匹配算法都要遵守这个接口。
    """

    def match(self, cam: "CAM", input_bits: np.ndarray, threshold: float) -> MatchResult:
        """Match one input bit vector against current CAM rows."""


class UpdateStrategy(Protocol):
    """Protocol for update strategies.

    中文：所有动态更新策略都要遵守这个接口。
    """

    def initialize_state(self, cam: "CAM") -> None:
        """Optional one-time initialization after templates are loaded."""

    def update(
        self,
        cam: "CAM",
        input_bits: np.ndarray,
        match: MatchResult,
        step_index: int,
    ) -> UpdateResult:
        """Update CAM state after one match result is produced."""


@dataclass
class TemplateRows:
    """Container used by ``CAM.load_templates``.

    中文：模板初始化模块的输出格式，专门给 CAM.load_templates 使用。
    """

    unit_ids: np.ndarray
    templates: np.ndarray
    masks: np.ndarray
    weights: np.ndarray
    support_counts: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CAM:
    """Generic content-addressable memory for binary spike templates.

    中文：
    这是实验平台里的通用 CAM 类。

    它主要负责：

    - 存模板 ``templates``
    - 存 bit mask ``masks``
    - 存匹配权重 ``match_weights``
    - 存 row 对应的 neuron id
    - 存每一行的 usage / update 次数 / 插入时间等元数据
    """

    capacity: int
    bit_width: int
    match_strategy: MatchStrategy
    update_strategy: Optional[UpdateStrategy] = None
    eviction_policy: str = "least_used"

    templates: np.ndarray = field(init=False)
    masks: np.ndarray = field(init=False)
    match_weights: np.ndarray = field(init=False)
    neuron_ids: np.ndarray = field(init=False)
    row_usage: np.ndarray = field(init=False)
    row_train_counts: np.ndarray = field(init=False)
    row_insert_step: np.ndarray = field(init=False)
    row_last_match_step: np.ndarray = field(init=False)
    row_last_update_step: np.ndarray = field(init=False)
    row_update_counts: np.ndarray = field(init=False)
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.templates = np.zeros((self.capacity, self.bit_width), dtype=np.uint8)
        self.masks = np.zeros((self.capacity, self.bit_width), dtype=np.uint8)
        self.match_weights = np.ones((self.capacity, self.bit_width), dtype=np.float32)
        self.neuron_ids = np.full(self.capacity, -1, dtype=np.int64)
        self.row_usage = np.zeros(self.capacity, dtype=np.int64)
        self.row_train_counts = np.zeros(self.capacity, dtype=np.int64)
        self.row_insert_step = np.zeros(self.capacity, dtype=np.int64)
        self.row_last_match_step = np.full(self.capacity, -1, dtype=np.int64)
        self.row_last_update_step = np.full(self.capacity, -1, dtype=np.int64)
        self.row_update_counts = np.zeros(self.capacity, dtype=np.int64)

    @property
    def used_row_indices(self) -> np.ndarray:
        """Return indices of rows currently occupied by templates.

        中文：当前哪些行是已使用的模板行。
        """

        return np.flatnonzero(self.neuron_ids != -1)

    @property
    def used_rows(self) -> int:
        """Return the number of occupied rows.

        中文：当前已占用的模板行数。
        """

        return int(self.used_row_indices.size)

    @property
    def free_rows(self) -> int:
        """Return the number of free rows.

        中文：当前还剩多少空闲行。
        """

        return int(self.capacity - self.used_rows)

    def clear(self) -> None:
        """Reset templates, metadata, and strategy state.

        中文：清空整个 CAM 和内部状态。
        """

        self.templates.fill(0)
        self.masks.fill(0)
        self.match_weights.fill(1.0)
        self.neuron_ids.fill(-1)
        self.row_usage.fill(0)
        self.row_train_counts.fill(0)
        self.row_insert_step.fill(0)
        self.row_last_match_step.fill(-1)
        self.row_last_update_step.fill(-1)
        self.row_update_counts.fill(0)
        self.state.clear()

    def load_templates(self, template_rows: TemplateRows) -> None:
        """Load a batch of initial templates into the CAM.

        中文：把 warmup 阶段得到的初始模板写进 CAM。
        """

        rows = int(template_rows.unit_ids.shape[0])
        if rows > self.capacity:
            raise ValueError(f"Template count {rows} exceeds CAM capacity {self.capacity}")
        if template_rows.templates.shape != (rows, self.bit_width):
            raise ValueError(
                f"template shape must be ({rows}, {self.bit_width}), got {template_rows.templates.shape}"
            )

        self.clear()
        self.neuron_ids[:rows] = np.asarray(template_rows.unit_ids, dtype=np.int64)
        self.templates[:rows] = np.asarray(template_rows.templates, dtype=np.uint8)
        self.masks[:rows] = np.asarray(template_rows.masks, dtype=np.uint8)
        self.match_weights[:rows] = np.asarray(template_rows.weights, dtype=np.float32)
        self.row_train_counts[:rows] = np.asarray(template_rows.support_counts, dtype=np.int64)

        if template_rows.meta:
            self.state["template_meta"] = dict(template_rows.meta)

        if hasattr(self.match_strategy, "initialize_state"):
            getattr(self.match_strategy, "initialize_state")(self)
        if self.update_strategy is not None and hasattr(self.update_strategy, "initialize_state"):
            getattr(self.update_strategy, "initialize_state")(self)

    def allocate_row(
        self,
        neuron_id: int,
        bits: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        support_count: int = 0,
        step_index: int = 0,
    ) -> int:
        """Allocate the next free row and write a template into it.

        中文：给一些支持 growth 的策略分配新模板行。
        """

        free = np.flatnonzero(self.neuron_ids == -1)
        if free.size == 0:
            raise RuntimeError("CAM has no free row to allocate")
        row_idx = int(free[0])
        self.replace_row(
            row_idx=row_idx,
            neuron_id=neuron_id,
            bits=bits,
            mask=mask,
            weights=weights,
            support_count=support_count,
            step_index=step_index,
        )
        return row_idx

    def replace_row(
        self,
        row_idx: int,
        neuron_id: int,
        bits: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        support_count: int = 0,
        step_index: int = 0,
    ) -> None:
        """Overwrite a row with a new template.

        中文：直接覆盖某一行模板，通常用于替换 / eviction 场景。
        """

        if not (0 <= row_idx < self.capacity):
            raise IndexError(f"row_idx out of range: {row_idx}")
        bit_array = np.asarray(bits, dtype=np.uint8)
        if bit_array.shape != (self.bit_width,):
            raise ValueError(f"bits must have shape ({self.bit_width},), got {bit_array.shape}")

        self.templates[row_idx] = bit_array
        self.masks[row_idx] = (
            np.ones(self.bit_width, dtype=np.uint8)
            if mask is None
            else np.asarray(mask, dtype=np.uint8)
        )
        self.match_weights[row_idx] = (
            np.ones(self.bit_width, dtype=np.float32)
            if weights is None
            else np.asarray(weights, dtype=np.float32)
        )
        self.neuron_ids[row_idx] = int(neuron_id)
        self.row_usage[row_idx] = 0
        self.row_train_counts[row_idx] = int(support_count)
        self.row_insert_step[row_idx] = int(step_index)
        self.row_last_match_step[row_idx] = -1
        self.row_last_update_step[row_idx] = -1
        self.row_update_counts[row_idx] = 0

    def evict_row(self, row_idx: int) -> None:
        """Clear one row completely.

        中文：彻底清空某一行。
        """

        if not (0 <= row_idx < self.capacity):
            raise IndexError(f"row_idx out of range: {row_idx}")
        self.templates[row_idx].fill(0)
        self.masks[row_idx].fill(0)
        self.match_weights[row_idx].fill(1.0)
        self.neuron_ids[row_idx] = -1
        self.row_usage[row_idx] = 0
        self.row_train_counts[row_idx] = 0
        self.row_insert_step[row_idx] = 0
        self.row_last_match_step[row_idx] = -1
        self.row_last_update_step[row_idx] = -1
        self.row_update_counts[row_idx] = 0

    def select_evict_row(self, policy: Optional[str] = None) -> int:
        """Select a row to evict when the CAM is full.

        中文：当 CAM 满了以后，按策略选一行替换掉。
        """

        used = self.used_row_indices
        if used.size == 0:
            raise RuntimeError("Cannot evict from an empty CAM")

        chosen_policy = self.eviction_policy if policy is None else policy
        if chosen_policy == "least_used":
            scores = self.row_usage[used]
            return int(used[np.argmin(scores)])
        if chosen_policy == "oldest":
            scores = self.row_insert_step[used]
            return int(used[np.argmin(scores)])
        raise ValueError(f"Unknown eviction policy: {chosen_policy}")

    def touch_match(self, row_idx: int, step_index: int) -> None:
        """Bookkeep that a row was the best match.

        中文：记录某一行刚刚被命中过，用于 usage 统计。
        """

        if row_idx < 0:
            return
        self.row_usage[row_idx] += 1
        self.row_last_match_step[row_idx] = int(step_index)

    def mark_update(self, row_idx: int, step_index: int) -> None:
        """Bookkeep that a row was updated.

        中文：记录某一行刚刚被更新过。
        """

        if row_idx < 0:
            return
        self.row_last_update_step[row_idx] = int(step_index)
        self.row_update_counts[row_idx] += 1

    @staticmethod
    def hamming_distance(bits1: np.ndarray, bits2: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        """Compute masked Hamming distance between two bit vectors.

        中文：最基础的 bit 级距离度量。
        """

        x1 = np.asarray(bits1, dtype=np.uint8)
        x2 = np.asarray(bits2, dtype=np.uint8)
        if x1.shape != x2.shape:
            raise ValueError(f"bits shape mismatch: {x1.shape} vs {x2.shape}")
        diff = np.bitwise_xor(x1, x2)
        if mask is not None:
            diff = np.bitwise_and(diff, np.asarray(mask, dtype=np.uint8))
        return int(np.sum(diff))

    @staticmethod
    def weighted_hamming_distance(
        bits1: np.ndarray,
        bits2: np.ndarray,
        mask: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute a weighted Hamming distance on the same scale as bit counts.

        中文：带权重的 Hamming 距离，常用于给稳定 bit 更高权重。
        """

        x1 = np.asarray(bits1, dtype=np.uint8)
        x2 = np.asarray(bits2, dtype=np.uint8)
        diff = np.bitwise_xor(x1, x2).astype(np.float32) * np.asarray(mask, dtype=np.float32)
        eff_weights = np.asarray(weights, dtype=np.float32) * np.asarray(mask, dtype=np.float32)
        denom = float(np.sum(eff_weights))
        if denom <= 1e-8:
            return 0.0
        return float(np.sum(diff * eff_weights) / denom * x1.shape[0])

    def match(self, input_bits: np.ndarray, threshold: float) -> MatchResult:
        """Run the configured matching strategy.

        中文：只做匹配，不做更新。
        """

        return self.match_strategy.match(self, np.asarray(input_bits, dtype=np.uint8), threshold)

    def process(self, input_bits: np.ndarray, threshold: float, step_index: int) -> StepResult:
        """Run one online CAM step: match first, then optional update.

        中文：在线评估的核心接口。每来一条 spike bits，就走一次这里。
        """

        bit_array = np.asarray(input_bits, dtype=np.uint8)
        if bit_array.shape != (self.bit_width,):
            raise ValueError(f"input_bits must have shape ({self.bit_width},), got {bit_array.shape}")

        match_result = self.match(bit_array, threshold)
        self.touch_match(match_result.best_row, step_index)

        # 中文：先匹配，再根据策略决定是否更新模板。
        if self.update_strategy is None:
            update_result = UpdateResult(updated=False, reason="no_update_strategy")
        else:
            update_result = self.update_strategy.update(self, bit_array, match_result, step_index)
            if update_result.updated_row >= 0:
                self.mark_update(update_result.updated_row, step_index)

        return StepResult(match=match_result, update=update_result, template_count=self.used_rows)


__all__ = [
    "CAM",
    "MatchResult",
    "MatchStrategy",
    "StepResult",
    "TemplateRows",
    "UpdateResult",
    "UpdateStrategy",
]
