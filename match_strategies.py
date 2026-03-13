from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from cam_core import CAM, MatchStrategy


@dataclass
class HammingNearestMatch(MatchStrategy):
    """
    最基本的匹配策略：带 mask 的汉明距离最近邻（Winner-Take-All）。

    - 遍历所有 neuron_ids != -1 的行
    - 计算带 mask 的汉明距离
    - 取距离最小者；若 min_dist <= threshold 则返回对应 neuron_id，否则返回 None
    """

    def match(
        self,
        cam: CAM,
        input_bits: np.ndarray,
        threshold: int = 0,
    ) -> Tuple[Optional[int], int, int]:
        x = np.asarray(input_bits, dtype=int)
        if x.shape != (cam.bit_width,):
            raise ValueError(f"input_bits 形状错误: {x.shape}, 期望 ({cam.bit_width},)")

        min_dist = cam.bit_width + 1
        best_row = -1

        for i in range(cam.capacity):
            nid = cam.neuron_ids[i]
            if nid == -1:
                continue
            d = CAM.hamming_distance(x, cam.templates[i], mask=cam.masks[i])
            if d < min_dist:
                min_dist = d
                best_row = i

        if best_row == -1 or min_dist > threshold:
            return None, -1, min_dist
        return int(cam.neuron_ids[best_row]), best_row, min_dist


__all__ = [
    "HammingNearestMatch",
]

