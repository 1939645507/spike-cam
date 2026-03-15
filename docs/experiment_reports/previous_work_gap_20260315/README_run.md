# Previous Work Gap Study

Run root: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiment_20260315_144914_previous_work_gap`

## 研究目的

解释为什么 previous work 里的指纹在旧实验里有较强辨别力，而当前 thesis 主线实验结果却低很多。

## 环境

- conda env: `spikecam_py310`
- Python: `3.10.20`
- Platform: `macOS-26.3-arm64-arm-64bit`

## 主要子实验

- `prevwork_csv_threshold_sweep`: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiment_20260315_144914_previous_work_gap/prevwork_csv_threshold_sweep`
- `prevwork_csv_update_compare`: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiment_20260315_144914_previous_work_gap/prevwork_csv_update_compare`
- `toy_recreated_random_threshold_sweep`: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiment_20260315_144914_previous_work_gap/toy_recreated_random_threshold_sweep`
- `toy_recreated_chronological_threshold_sweep`: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiment_20260315_144914_previous_work_gap/toy_recreated_chronological_threshold_sweep`
- `real_top8_oldlike_random_threshold_sweep`: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiment_20260315_144914_previous_work_gap/real_top8_oldlike_random_threshold_sweep`
- `real_top8_oldlike_random_update_compare`: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiment_20260315_144914_previous_work_gap/real_top8_oldlike_random_update_compare`
- `real_top8_oldlike_chronological_threshold_sweep`: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiment_20260315_144914_previous_work_gap/real_top8_oldlike_chronological_threshold_sweep`
- `real_top50_mem20_open_threshold_sweep`: `/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/results/experiment_20260315_144914_previous_work_gap/real_top50_mem20_open_threshold_sweep`

## 复现命令

`conda run -n spikecam_py310 python scripts/run_previous_work_gap_study.py --run-root results/experiment_20260315_144914_previous_work_gap`

## 目录说明

- `figures/`: 顶层汇总图。
- `tables/`: 汇总表和各子实验复制出的表格。
- `logs/`: 命令与异常记录。
- `master_report.md`: 本次研究的总分析报告。
