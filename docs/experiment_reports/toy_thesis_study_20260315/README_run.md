# Toy Thesis Study README

Run root: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study`

## 环境

- conda env: `spikecam_py310`
- Python: `3.10.20`
- Platform: `macOS-26.3-arm64-arm-64bit`

## Toy 数据集

- `toy_easy_stable_u8_c8_60s`: units=8, channels=8, duration=60s, noise=0.0, drift=0.0
- `toy_dense_stable_u12_c16_75s`: units=12, channels=16, duration=75s, noise=2.0, drift=0.0
- `toy_drift_u12_c16_75s`: units=12, channels=16, duration=75s, noise=4.0, drift=0.28
- `toy_open_u20_c24_90s`: units=20, channels=24, duration=90s, noise=5.5, drift=0.22

## 主要实验目录

- `threshold_easy_stable` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/threshold_easy_stable`
- `threshold_dense_stable` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/threshold_dense_stable`
- `threshold_drift_closed` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/threshold_drift_closed`
- `threshold_open_memory` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/threshold_open_memory`
- `protocol_compare_easy` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/protocol_compare_easy`
- `update_compare_dense_stable` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/update_compare_dense_stable`
- `update_compare_drift_closed` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/update_compare_drift_closed`
- `update_compare_open_memory` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/update_compare_open_memory`
- `bit_ablation_drift` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/bit_ablation_drift`
- `init_ablation_drift` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/init_ablation_drift`
- `encoder_threshold_calibration_drift` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/encoder_threshold_calibration_drift`
- `encoder_compare_drift` -> `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study/encoder_compare_drift`

## 复现入口

`conda run -n spikecam_py310 python scripts/run_toy_thesis_study.py --run-root results/experiments/2026-03-15/experiment_20260315_161004_toy_thesis_study`

## 目录说明

- `datasets/`: 本次 toy 数据集的元数据索引。
- `figures/`: 顶层汇总图。
- `tables/`: 各子实验汇总表。
- `logs/`: 命令和异常记录。
- `master_report.md`: 适合直接阅读的总报告。
