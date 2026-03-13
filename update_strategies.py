from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from cam_core import CAM, UpdateStrategy, MatchStrategy


@dataclass
class NoUpdate(UpdateStrategy):
    """
    不进行任何动态更新，只是复用 match_strategy 的结果。
    主要用于对照实验。
    """

    base_match: MatchStrategy

    def match_and_update(
        self,
        cam: CAM,
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        return self.base_match.match(cam, input_bits, threshold=threshold)


@dataclass
class CounterUpdate(UpdateStrategy):
    """
    简化版“置信计数器”动态更新策略（参考你原来的 CounterDynamicCAM 思路）。

    思路：
    - 每个 bit 维护一个整数计数器 counters[row, bit]
    - 每次匹配成功：
        - 对匹配行，若输入 bit 与模板相同，则计数 +1；否则计数 -1（下限为 0）
        - 当某个 bit 的计数 > max_confidence，则把该 bit 的模板改为当前输入 bit，并计数清零
    - 可以理解为：一条轨迹上、累计“足够多次一致”，我们才允许模板做一次 bit flip。
    """

    base_match: MatchStrategy
    max_confidence: int = 10

    def _ensure_state(self, cam: CAM) -> np.ndarray:
        if "counters" not in cam.state:
            cam.state["counters"] = np.zeros((cam.capacity, cam.bit_width), dtype=int)
        return cam.state["counters"]

    def initialize_state(self, cam: CAM) -> None:
        """
        初始化计数器为 max_confidence 的一半（模仿 CounterDynamicCAM 的写法）。
        """
        counters = self._ensure_state(cam)
        initial_val = self.max_confidence // 2
        counters[:, :] = (cam.templates * 2 - 1) * initial_val
        counters[:, :] *= cam.masks

    def match_and_update(
        self,
        cam: CAM,
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        nid, row_idx, dist = self.base_match.match(cam, input_bits, threshold=threshold)
        if nid is None or row_idx < 0:
            return None, row_idx, dist

        x = np.asarray(input_bits, dtype=int)
        counters = self._ensure_state(cam)

        tpl = cam.templates[row_idx]
        c = counters[row_idx]

        same = x == tpl
        # 一致的 bit 置信度 +1， 不一致的 -1（但不低于 0）
        c[same] += 1
        c[~same] -= 1
        c[c < 0] = 0

        # 达到阈值的 bit，更新模板，并清零计数
        to_flip = c > self.max_confidence
        if np.any(to_flip):
            tpl[to_flip] = x[to_flip]
            c[to_flip] = 0

        cam.templates[row_idx] = tpl
        counters[row_idx] = c

        return nid, row_idx, dist


@dataclass
class MarginEmaUpdate(UpdateStrategy):
    """
    Margin + EMA 动态更新策略（参考 MarginEmaCAM）。

    - 仅当距离接近阈值边界 (threshold - dist <= 1) 时才更新
    - 使用浮点 EMA 模板，最后阈值 0.5 转回 0/1
    """

    base_match: MatchStrategy
    alpha: float = 0.05

    def _ensure_state(self, cam: CAM) -> np.ndarray:
        if "ema_templates" not in cam.state:
            cam.state["ema_templates"] = cam.templates.astype(float)
        return cam.state["ema_templates"]

    def initialize_state(self, cam: CAM) -> None:
        cam.state["ema_templates"] = cam.templates.astype(float)

    def match_and_update(
        self,
        cam: CAM,
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        nid, row_idx, dist = self.base_match.match(cam, input_bits, threshold=threshold)
        if nid is None or row_idx < 0:
            return None, row_idx, dist

        margin = threshold - dist
        if margin > 1:
            return nid, row_idx, dist

        ema = self._ensure_state(cam)
        x = np.asarray(input_bits, dtype=float)

        ema[row_idx] = (1.0 - self.alpha) * ema[row_idx] + self.alpha * x
        cam.templates[row_idx] = (ema[row_idx] >= 0.5).astype(int)
        cam.state["ema_templates"] = ema

        return nid, row_idx, dist


@dataclass
class ConfidenceWeightedUpdate(UpdateStrategy):
    """
    Weighted Hamming + 位置信度更新策略（参考 ConfidenceWeightedCAM）。

    注意：这里仍然用基础匹配的汉明距离做“是否 accept”判断，
    但内部的 bit_conf 只影响更新门控，不直接改距离度量（与原 notebook 行为一致）。
    """

    base_match: MatchStrategy
    lr: float = 0.1
    max_conf: float = 5.0

    def _ensure_state(self, cam: CAM) -> np.ndarray:
        if "bit_conf" not in cam.state:
            cam.state["bit_conf"] = np.ones((cam.capacity, cam.bit_width), dtype=float)
        return cam.state["bit_conf"]

    def initialize_state(self, cam: CAM) -> None:
        cam.state["bit_conf"] = np.ones((cam.capacity, cam.bit_width), dtype=float)

    def match_and_update(
        self,
        cam: CAM,
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        nid, row_idx, dist = self.base_match.match(cam, input_bits, threshold=threshold)
        if nid is None or row_idx < 0:
            return None, row_idx, dist

        margin = threshold - dist
        if margin > 1:
            return nid, row_idx, dist

        bit_conf = self._ensure_state(cam)
        x = np.asarray(input_bits, dtype=int)

        diff = x != cam.templates[row_idx]
        bit_conf[row_idx][diff] -= self.lr
        bit_conf[row_idx][~diff] += self.lr
        bit_conf[row_idx] = np.clip(bit_conf[row_idx], 0.1, self.max_conf)

        flip_mask = diff & (bit_conf[row_idx] < 1.0)
        if np.any(flip_mask):
            cam.templates[row_idx][flip_mask] = x[flip_mask]

        cam.state["bit_conf"] = bit_conf
        return nid, row_idx, dist


@dataclass
class DualTemplateUpdate(UpdateStrategy):
    """
    双模板策略（稳定 + 可塑），参考 DualTemplateCAM。

    - 维护一个浮点 `plastic` 模板（EMA）
    - 距离可基于 stable/plastic 两者中的较小者（这里只在更新门控中使用距离）
    - 当 plastic 与 stable 足够接近时，把 stable 合并为 plastic 的方向
    """

    base_match: MatchStrategy
    alpha: float = 0.1

    def _ensure_state(self, cam: CAM) -> np.ndarray:
        if "plastic" not in cam.state:
            cam.state["plastic"] = cam.templates.astype(float)
        return cam.state["plastic"]

    def initialize_state(self, cam: CAM) -> None:
        cam.state["plastic"] = cam.templates.astype(float)

    def match_and_update(
        self,
        cam: CAM,
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        nid, row_idx, dist = self.base_match.match(cam, input_bits, threshold=threshold)
        if nid is None or row_idx < 0:
            return None, row_idx, dist

        plastic = self._ensure_state(cam)
        x = np.asarray(input_bits, dtype=int)

        plastic[row_idx] = (1.0 - self.alpha) * plastic[row_idx] + self.alpha * x

        agree = np.abs(plastic[row_idx] - cam.templates[row_idx]) < 0.1
        if np.any(agree):
            cam.templates[row_idx][agree] = (plastic[row_idx][agree] >= 0.5).astype(int)

        cam.state["plastic"] = plastic
        return nid, row_idx, dist


@dataclass
class ProbabilisticUpdate(UpdateStrategy):
    """
    概率模板策略（参考 ProbabilisticCAM）。

    - 每个 bit 维护一个概率 p，代表为 1 的概率
    - 匹配时原论文用 NLL 距离，这里为了简单仍使用基础汉明距离做 accept，
      但更新逻辑严格按照 EMA 的概率更新 + 0.5 阈值二值化。
    """

    base_match: MatchStrategy
    alpha: float = 0.05
    eps: float = 1e-4

    def _ensure_state(self, cam: CAM) -> np.ndarray:
        if "prob" not in cam.state:
            cam.state["prob"] = cam.templates.astype(float)
        return cam.state["prob"]

    def initialize_state(self, cam: CAM) -> None:
        cam.state["prob"] = cam.templates.astype(float)

    def match_and_update(
        self,
        cam: CAM,
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        nid, row_idx, dist = self.base_match.match(cam, input_bits, threshold=threshold)
        if nid is None or row_idx < 0:
            return None, row_idx, dist

        prob = self._ensure_state(cam)
        x = np.asarray(input_bits, dtype=float)

        prob[row_idx] = (1.0 - self.alpha) * prob[row_idx] + self.alpha * x
        cam.templates[row_idx] = (prob[row_idx] >= 0.5).astype(int)
        cam.state["prob"] = prob

        return nid, row_idx, dist


@dataclass
class GrowingUpdate(UpdateStrategy):
    """
    GrowingCAM 策略（简单版本）：

    - 每行维护一个 usage 计数
    - 当匹配距离 > threshold 且仍有空行时，分裂出一个新行，将输入 bits 作为新模板
    """

    base_match: MatchStrategy
    split_threshold: int = 3

    def _ensure_state(self, cam: CAM) -> np.ndarray:
        if "usage" not in cam.state:
            cam.state["usage"] = np.zeros(cam.capacity, dtype=int)
        return cam.state["usage"]

    def initialize_state(self, cam: CAM) -> None:
        cam.state["usage"] = np.zeros(cam.capacity, dtype=int)

    def match_and_update(
        self,
        cam: CAM,
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        nid, row_idx, dist = self.base_match.match(cam, input_bits, threshold=threshold)
        if row_idx < 0:
            return nid, row_idx, dist

        usage = self._ensure_state(cam)
        usage[row_idx] += 1

        if dist > threshold and cam.free_rows > 0:
            new_idx = cam.allocate_row(neuron_id=nid if nid is not None else -1)
            cam.templates[new_idx] = np.asarray(input_bits, dtype=int)
            cam.masks[new_idx] = 1

        cam.state["usage"] = usage
        return nid, row_idx, dist


__all__ = [
    "NoUpdate",
    "CounterUpdate",
    "MarginEmaUpdate",
    "ConfidenceWeightedUpdate",
    "DualTemplateUpdate",
    "ProbabilisticUpdate",
    "GrowingUpdate",
]

