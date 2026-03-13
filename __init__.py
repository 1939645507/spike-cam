"""
Spike CAM: 可插拔的 CAM + Encoder + 模板 + 评估 框架。

主要对外暴露的 API：

- `CAM`（见 cam_core.py）
- 匹配策略：见 match_strategies.py
- 动态更新策略：见 update_strategies.py
- 模板生成：见 templates.py
- 数据/encoder 管线：见 encoder.py
- 评估与完整 pipeline：见 evaluate.py
"""

from cam_core import CAM  # noqa: F401

