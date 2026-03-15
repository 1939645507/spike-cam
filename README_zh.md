# Spike CAM

English quick-start: [README.md](README.md)  
详细文档： [docs/PROJECT_GUIDE_zh.md](docs/PROJECT_GUIDE_zh.md) | [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)

这是一个面向毕设 / 课程项目的 **Spike CAM 实验平台**。  
它的目标是：在固定编码条件和有限 CAM memory 下，系统比较不同的模板初始化、匹配和动态更新策略。

核心研究问题可以概括成一句话：

> 在固定 encoder 的前提下，不同 CAM 动态更新策略能否更稳定地识别 memory 内类，并拒绝 memory 外类？

## 这个项目能做什么

- 从 `.npz` 读取原始 spike 数据
- 对连续电压做 `bandpass / CMR / optional whitening`
- 提取 multi-channel waveform
- 用 `PCA` 或 `AE` 编码成 bits
- 建立 CAM 模板
- 做 **chronological / online** 评估
- 比较不同动态更新算法
- 自动保存指标、曲线、图表和报告

## 项目结构

```text
spike_cam/
├── README.md
├── README_zh.md
├── docs/
├── configs/
├── dataset/
├── external/
├── results/
├── scripts/
├── config.py
├── dataio.py
├── encoder.py
├── templates.py
├── match_strategies.py
├── update_strategies.py
├── cam_core.py
├── metrics.py
├── experiment_runner.py
└── run_experiment.py
```

## 快速开始

### 1. 环境

推荐环境：

```bash
conda activate spikecam_py310
pip install -r requirements.txt
```

如果你要用 `external/Autoencoders-in-Spike-Sorting`，请确认 `spikecam_py310` 里有 `tensorflow` 和 `scikit-learn`。

### 2. 跑一个 baseline

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python run_experiment.py --config configs/baseline_top10_ae16.json
```

这条命令会完成整条流程：

- 读数据
- 前处理和 waveform 提取
- 编码成 bits
- 用 warmup 建模板
- 跑 chronological CAM 评估
- 保存结果

### 3. 生成 toy 数据

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/generate_toy_datasets.py
```

toy 数据会保存到 `dataset/toy/`。

### 4. 跑 toy 全面实验

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/run_toy_thesis_study.py
```

这组实验会自动生成：

- 多个 toy dataset
- stable / drift / open-memory 场景
- dynamic update 对比
- bit 数消融
- 初始化方法消融
- encoder 对比
- 一份总报告

### 5. 可选：训练可复用 external AE

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/train_external_ae.py \
  --config configs/cam_compare_main.json \
  --artifact-dir results/artifacts/external_ae_artifacts/fullset_ae16_normal \
  --subset-mode all \
  --epochs 6 \
  --scale minmax
```

## 怎么改实验

这个项目是 **配置驱动** 的。大多数情况下你只需要改 `configs/*.json`。

最常改的字段：

- `dataset.npz_path`
- `dataset.subset`
- `cam.memory_subset`
- `encoder.method` / `encoder.backend` / `encoder.code_size`
- `template_init.method`
- `matcher.method`
- `update_strategy.method`
- `cam.match_threshold`
- `evaluation.warmup_ratio`

几个常见例子：

- 想改 bit 数：改 `encoder.code_size`
- 想比较 `static` 和 `counter`：改 config 里的 variants
- 想做 `top50 stream / top20 memory`：改 `dataset.subset` 和 `cam.memory_subset`

## 常用命令

查看 label 分布：

```bash
python scripts/inspect_labels.py --npz dataset/my_validation_subset_810000samples_27.00s.npz
```

只做编码：

```bash
python run_experiment.py --config configs/baseline_top10_ae16.json --encode-only
```

检查 encoded dataset：

```bash
python scripts/inspect_encoded_dataset.py --encoded results/cache/encoded_cache/<file>.npz
```

只跑部分 variants：

```bash
python run_experiment.py --config configs/cam_compare_main.json --variant static --variant counter
```

## 输出目录

普通实验默认保存到：

```text
results/experiments/<YYYY-MM-DD>/<experiment_name>/
```

常见输出包括：

- `config.json`
- `summary.json`
- `runs/<variant>/metrics.json`
- `runs/<variant>/predictions.npz`
- `runs/<variant>/curves.npz`
- `report.md`

## 推荐入口

- `configs/baseline_top10_ae16.json`：baseline / sanity check
- `configs/cam_compare_main.json`：主实验
- `configs/bits_ablation.json`：bit 数消融
- `configs/encoder_compare.json`：encoder 消融
- `configs/init_ablation.json`：初始化消融

## 详细文档

- 详细中文说明： [docs/PROJECT_GUIDE_zh.md](docs/PROJECT_GUIDE_zh.md)
- 详细英文说明： [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)
- 代码阅读顺序： [CODE_READING_ORDER_zh.md](CODE_READING_ORDER_zh.md)

## GitHub 上传建议

- 大型原始数据不要提交
- `results/`、cache、artifact 建议忽略
- toy 数据可以随时用 `scripts/generate_toy_datasets.py` 重新生成
