from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np


class MatchStrategy(Protocol):
    """
    匹配策略接口：给定 CAM 状态和一条输入 bits，返回 (matched_id, row_idx, distance)。
    """

    def match(
        self,
        cam: "CAM",
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        ...


class UpdateStrategy(Protocol):
    """
    动态更新策略接口：内部通常会调用 MatchStrategy，然后根据结果改写 CAM 模板。
    """

    def match_and_update(
        self,
        cam: "CAM",
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        ...


@dataclass
class CAM:
    """
    通用 CAM 类。

    - capacity: 最多可存多少个模板（行数）
    - bit_width: 每个模板 / 输入指纹的 bit 数
    - match_strategy: 匹配算法（必选）
    - update_strategy: 动态更新策略（可选）

    说明：
    - 模板存放在 `templates`，掩码在 `masks`（1=有效，0=忽略），
      每行的逻辑 neuron id 存在 `neuron_ids`（-1 代表空）。
    - `state` 是一个自由的字典，供各种策略挂载自己的内部状态（计数器、EMA 等）。
    """

    capacity: int
    bit_width: int
    match_strategy: MatchStrategy
    update_strategy: Optional[UpdateStrategy] = None

    templates: np.ndarray = field(init=False)
    masks: np.ndarray = field(init=False)
    neuron_ids: np.ndarray = field(init=False)
    state: Dict[str, Any] = field(default_factory=dict)  # 策略内部状态

    def __post_init__(self) -> None:
        self.templates = np.zeros((self.capacity, self.bit_width), dtype=int)
        self.masks = np.zeros((self.capacity, self.bit_width), dtype=int)
        self.neuron_ids = np.full(self.capacity, -1, dtype=int)

    # ------------------------------------------------------------------
    # 模板管理
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """清空所有模板与状态。"""
        self.templates.fill(0)
        self.masks.fill(0)
        self.neuron_ids.fill(-1)
        self.state.clear()

    def load_templates(
        self,
        unit_ids: np.ndarray,
        template_bits: np.ndarray,
    ) -> None:
        """
        以“静态模板”的形式批量初始化 CAM。

        - unit_ids:  (N,) int
        - template_bits: (N, bit_width) 0/1
        """
        unit_ids = np.asarray(unit_ids, dtype=int)
        template_bits = np.asarray(template_bits, dtype=int)

        if template_bits.ndim != 2 or template_bits.shape[1] != self.bit_width:
            raise ValueError(f"template_bits 形状错误: {template_bits.shape}, 期望 (*, {self.bit_width})")
        if unit_ids.shape[0] > self.capacity:
            raise ValueError(f"模板数量 {unit_ids.shape[0]} 超过 CAM capacity={self.capacity}")

        self.clear()
        n = unit_ids.shape[0]
        self.templates[:n] = template_bits
        self.masks[:n] = 1
        self.neuron_ids[:n] = unit_ids

        # 通知策略做一次状态初始化（如果它们实现了 initialize_state(cam)）
        if hasattr(self.match_strategy, "initialize_state"):
            getattr(self.match_strategy, "initialize_state")(self)
        if self.update_strategy is not None and hasattr(self.update_strategy, "initialize_state"):
            getattr(self.update_strategy, "initialize_state")(self)

    def update_template(self, row_idx: int, bits: np.ndarray, neuron_id: Optional[int] = None) -> None:
        """
        单行写入 / 覆盖模板，供策略或外部手动使用。
        """
        if not (0 <= row_idx < self.capacity):
            raise IndexError(f"row_idx={row_idx} out of range [0, {self.capacity})")
        arr = np.asarray(bits, dtype=int)
        if arr.shape != (self.bit_width,):
            raise ValueError(f"bits 形状错误: {arr.shape}, 期望 ({self.bit_width},)")

        self.templates[row_idx] = arr
        self.masks[row_idx] = 1
        if neuron_id is not None:
            self.neuron_ids[row_idx] = int(neuron_id)

    # ------------------------------------------------------------------
    # 行管理：free_rows & allocate_row（供一些动态策略使用）
    # ------------------------------------------------------------------
    @property
    def free_rows(self) -> int:
        """当前还剩多少空行（neuron_ids == -1）。"""
        return int(np.sum(self.neuron_ids == -1))

    def allocate_row(self, neuron_id: Optional[int] = None) -> int:
        """
        分配一个新的空行并返回其索引。

        - 如果没有空行，会抛出 RuntimeError。
        - 不会写入模板/掩码，只会设置 neuron_id（如果提供）。
        """
        idxs = np.flatnonzero(self.neuron_ids == -1)
        if idxs.size == 0:
            raise RuntimeError("CAM 没有空行可分配（capacity 已满）")
        row_idx = int(idxs[0])
        if neuron_id is not None:
            self.neuron_ids[row_idx] = int(neuron_id)
        return row_idx

    # ------------------------------------------------------------------
    # 工具：汉明距离
    # ------------------------------------------------------------------
    @staticmethod
    def hamming_distance(
        bits1: np.ndarray,
        bits2: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> int:
        b1 = np.asarray(bits1, dtype=int)
        b2 = np.asarray(bits2, dtype=int)
        if b1.shape != b2.shape:
            raise ValueError(f"hamming_distance 形状不匹配: {b1.shape} vs {b2.shape}")

        diff = np.bitwise_xor(b1, b2)
        if mask is not None:
            m = np.asarray(mask, dtype=int)
            if m.shape != diff.shape:
                raise ValueError(f"mask 形状不匹配: {m.shape} vs {diff.shape}")
            diff = np.bitwise_and(diff, m)
        return int(np.sum(diff))

    # ------------------------------------------------------------------
    # 对外接口：匹配 & 动态更新
    # ------------------------------------------------------------------
    def match(self, input_bits: np.ndarray, threshold: int = 0) -> Tuple[Optional[int], int, int]:
        """
        使用匹配策略（不更新模板）。

        返回: (matched_neuron_id 或 None, 行号, 距离)
        """
        return self.match_strategy.match(self, np.asarray(input_bits, dtype=int), threshold=threshold)

    def match_and_update(self, input_bits: np.ndarray, threshold: int = 0) -> Tuple[Optional[int], int, int]:
        """
        使用动态更新策略（如果未配置，则退化为纯匹配）。
        """
        if self.update_strategy is None:
            return self.match(input_bits, threshold=threshold)
        return self.update_strategy.match_and_update(self, np.asarray(input_bits, dtype=int), threshold=threshold)

