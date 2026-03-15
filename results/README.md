# Results Layout

中文说明：

为了避免 `results/` 根目录越跑越乱，现在建议把结果分成三类：

```text
results/
├── experiments/
│   ├── 2026-03-14/
│   └── 2026-03-15/
├── cache/
│   └── encoded_cache/
└── artifacts/
    ├── external_ae_artifacts/
    └── smoke_external_artifact/
```

含义：

- `experiments/`
  真正的实验结果目录，按日期归档。

- `cache/`
  可重复复用但不属于“实验结论”的缓存，例如 `encoded_cache`。

- `artifacts/`
  训练好的模型工件、外部 AE artifact、smoke artifact。

如果要把当前根目录已有结果自动整理进去，可以运行：

```bash
conda run -n spikecam_py310 python scripts/organize_results.py --apply
```

另外，从现在开始，如果普通配置仍使用默认的 `results_dir = "results"`，
`run_experiment.py` 会自动把结果写到：

```text
results/experiments/<YYYY-MM-DD>/<experiment_name>/
```

默认 encoded cache 也会写到：

```text
results/cache/encoded_cache/
```

如果只想先看移动计划，不实际执行：

```bash
conda run -n spikecam_py310 python scripts/organize_results.py
```
