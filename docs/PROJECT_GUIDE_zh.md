# Spike CAM 详细说明

返回简版首页： [../README_zh.md](../README_zh.md)  
英文详细版： [PROJECT_GUIDE.md](PROJECT_GUIDE.md)  
补充文档： [../CODE_READING_ORDER_zh.md](../CODE_READING_ORDER_zh.md)

## 1. 项目简介

这个项目是一个面向毕设 / 课程项目的 **Spike CAM 实验平台**，核心目标是：

在固定 bit budget 和固定 CAM 容量下，系统比较不同的：

- template initialization
- match strategy
- dynamic update strategy

并分析它们在以下方面的权衡：

- accuracy
- macro-F1 / balanced accuracy
- reject behavior
- online stability
- memory usage
- wrong update rate

这个项目的研究主线是：

- **主角是 CAM 端**
- encoder 只是前端 waveform -> bits 的特征提取器
- 主实验固定 encoder，只比较 CAM
- bit 数、encoder、初始化方法主要是 ablation

项目的完整 pipeline 是：

1. 从 raw `.npz` 数据中读取 `raw_data / spike_times / spike_clusters`
2. 对连续电压先做可配置前处理：`bandpass / CMR / optional whitening`
3. 为每个 spike 提取 multi-channel waveform，默认保留 `top-k channels` 后 flatten
4. 用 encoder 将 waveform 压缩成固定 bit 数的二值编码
5. 用前一段时间窗口做 warmup，生成初始模板
6. 将后续 spikes 按时间顺序逐条输入 CAM
7. 每条 spike 执行：
   - match
   - accept / reject
   - optional update
8. 保存结果、曲线和中间统计

## 2. 项目结构

```text
spike_cam/
├── README.md
├── README_zh.md
├── requirements.txt
├── config.py
├── dataio.py
├── encoder.py
├── templates.py
├── match_strategies.py
├── update_strategies.py
├── cam_core.py
├── metrics.py
├── experiment_runner.py
├── run_experiment.py
├── configs/
├── scripts/
├── dataset/
├── results/
└── external/
```

### 主要文件功能

- `config.py`
  负责所有实验配置 dataclass、JSON 配置读取、variant 展开、路径管理。

- `dataio.py`
  负责原始 `.npz` 数据读取、连续信号前处理、multi-channel waveform 提取、subset 过滤、chronological split，以及 `EncodedDataset` 数据结构。

- `encoder.py`
  负责 waveform -> bits。当前支持：
  - PCA baseline
  - 轻量 numpy AE
  - 外部 `Autoencoders-in-Spike-Sorting` 适配接口
  同时也负责 encoded dataset 的统计诊断。

- `templates.py`
  负责 warmup 阶段的初始模板构建。

- `match_strategies.py`
  负责匹配算法。

- `update_strategies.py`
  负责动态模板更新算法，是本项目最核心的算法区之一。

- `cam_core.py`
  负责 CAM 的存储、行状态、容量管理、match/update 结果对象。

- `metrics.py`
  负责 accuracy、reject、online curves、memory、update 等指标的计算与保存。

- `experiment_runner.py`
  负责把整个 pipeline 串起来，执行完整实验。

- `run_experiment.py`
  命令行入口。

- `scripts/run_previous_work_gap_study.py`
  专门用于复现并解释 `previous work` 与 thesis 主线结果差异的研究脚本，会自动打包一套对照实验和总报告。

- `scripts/generate_toy_datasets.py`
  用 `spikeinterface` 生成多种可控 toy dataset，并保存成和真实数据一致的 raw `.npz` 格式。

- `scripts/run_toy_thesis_study.py`
  面向毕设的 toy-data 全面实验脚本，会自动生成 toy dataset、运行多组 CAM 对照，并输出结构化总报告。

## 3. 数据格式说明

### 原始输入数据格式

平台默认读取 `.npz` 文件，至少需要包含：

- `raw_data`
  多通道连续电压数据，形状通常是 `(samples, channels)`

- `spike_times`
  每个 spike 对应的时间点，形状 `(N,)`

- `spike_clusters`
  每个 spike 对应的 neuron id / cluster id，形状 `(N,)`

可选包含：

- `fs`
- `duration_sec`
- `n_channels`

### 当前默认前处理与 waveform 提取

新版默认前端更接近实际 extracellular spike pipeline：

- `bandpass_enable = true`
- `common_reference_enable = true`
- `whitening_enable = false`
- `channel_selection = "topk_max_abs"`
- `topk_channels = 8`
- `flatten_order = "channel_major"`

这么做的原因是：高通道数数据如果只取单个 `max_abs` channel，往往会把 spike 的 spatial footprint 丢掉太多，类内稳定性也会明显变差。

### 编码后数据格式

为了避免每次都重新跑 encoder，平台会把编码结果缓存成标准格式：

- `bits`: `(N, B)`
- `labels`: `(N,)`
- `spike_times`: `(N,)`
- `source_indices`: `(N,)`
- `meta_json`

对应的数据类是：

- `dataio.EncodedDataset`

这一步很关键，因为后面的 CAM 实验应该尽量只吃这个 encoded dataset，而不是每次从 raw data 重新开始。

## 4. 安装方式

### Python 版本建议

- Python `3.10+`

### 安装核心依赖

```bash
pip install -r requirements.txt
```

当前最小可运行依赖是：

- `numpy`
- `matplotlib`

本项目当前推荐统一使用的 conda 环境名是：

```bash
conda activate spikecam_py310
```

### 外部 AE 可选依赖

如果你希望调用 `external/Autoencoders-in-Spike-Sorting`，还需要额外安装：

- `scikit-learn`
- `tensorflow`

实践上更建议使用：

- Python `3.10`

因为原仓库要求的 `tensorflow==2.13.0` 与较新的 Python 版本可能不兼容。

如果这些依赖没有装，配置里 `backend="auto"` 会自动退回到内置的轻量 numpy AE，这样平台仍然可以直接跑通。

如果 `spikecam_py310` 里缺少 `tensorflow`，那 external AE 目前是跑不起来的；这种情况下可以先用 `backend="auto"` 或 `backend="numpy"` 做 pipeline 调试。

### 如果你已经有 conda 环境

如果你本机已经有类似下面的环境：

```bash
conda activate spikecam_py310
```

那么它非常适合拿来跑 `external/Autoencoders-in-Spike-Sorting`。

推荐两种用法：

```bash
conda activate spikecam_py310
python run_experiment.py --config configs/pilot_cam_compare_external_top20.json
```

或者不用手动切环境，直接：

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python run_experiment.py --config configs/pilot_cam_compare_external_top20.json
```

这里加 `MPLCONFIGDIR=.mplconfig` 的原因是：

- 某些环境下 `matplotlib` 默认缓存目录不可写
- 这样能避免 warning，也更稳定

### 推荐的 external AE 工作流

如果你的毕设主线要固定使用 `Autoencoders-in-Spike-Sorting`，推荐不要让每次 CAM 实验都重新训练 AE。

更合理的流程是：

1. 先单独训练 external AE artifact
2. 保存：
   - `autoencoder.weights.h5`
   - `code_thresholds.npy`
   - `scale_params.npz`
   - `metadata.json`
3. 后续所有 CAM 实验只加载这个 artifact 做 encode

现在项目已经支持这个流程，训练脚本是：

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/train_external_ae.py \
  --config configs/cam_compare_main.json \
  --artifact-dir results/artifacts/external_ae_artifacts/fullset_ae16_normal \
  --subset-mode all \
  --epochs 6 \
  --scale minmax
```

训练完成后，可以在 config 的 `encoder` 里指定：

- `backend: "external"`
- `artifact_path: "..."`
- `use_artifact: true`
- `save_artifact: true`
- `force_retrain_artifact: false`

这样主实验会直接加载现成 AE，而不是每次重新训练。

### 推荐的 toy 数据工作流

如果你想先在可控数据上研究 dynamic update 机制，再回到真实数据，推荐下面这条工作流：

1. 先生成 toy raw dataset
2. 再用同一条 Spike CAM pipeline 跑完整 toy study
3. 用 toy 结果总结 CAM 端规律
4. 最后把这些结论带回真实数据分析

生成 toy 数据：

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/generate_toy_datasets.py
```

这会在 `dataset/toy/` 下生成多种场景的 raw `.npz`：

- easy stable
- dense stable
- drift
- open memory

一键运行 toy thesis study：

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/run_toy_thesis_study.py
```

输出会自动保存到：

- `results/experiments/<YYYY-MM-DD>/experiment_<timestamp>_toy_thesis_study/`

其中会包含：

- `master_report.md / master_report.html`
- `overall_summary.csv`
- 各小实验子目录
- 统一汇总后的 `figures/ / tables/ / logs/`

## 5. 核心设计原则

### 5.1 配置驱动

实验参数不应该散落在代码里手改。

正确使用方式是：

1. 改 `configs/*.json`
2. 跑 `run_experiment.py`
3. 查看 `results/`

现在普通实验若仍使用默认 `results_dir = "results"`，实际会自动保存到：

- `results/experiments/<YYYY-MM-DD>/<experiment_name>/`

而不是直接堆在 `results/` 根目录。

常见可配置项包括：

- dataset path
- waveform extraction
- subset 规则
- encoder method
- code size
- template init
- matcher
- updater
- threshold
- capacity
- warmup ratio

### 5.2 模块解耦

系统被拆成：

- data
- encoder
- template initialization
- matcher
- updater
- CAM core
- metrics
- runner

这样后面做 comparison / ablation 时，不需要反复修改主逻辑。

### 5.3 Chronological evaluation 是默认主评估

动态更新必须按时间顺序评估。

平台默认做法：

1. 按 `spike_times` 排序
2. 前一段作为 warmup
3. 后一段逐条输入 CAM
4. 每步执行 match -> accept/reject -> optional update

`random_split` 只保留为一个可选 baseline，不是默认主实验。

### 5.4 先编码，后评估

主实验流程应该是：

1. 先得到或缓存 encoded dataset
2. 再在同一份 bits 上反复比较 CAM 方法

这样做的优点：

- encoder 和 CAM 的贡献能分开分析
- 实验更快
- 对比更公平

## 6. 支持的功能

### 6.1 现在要区分两层 subset

新版协议里，请一定把下面两个概念分开：

1. `dataset.subset`
2. `cam.memory_subset`

它们现在不是一回事。

#### `dataset.subset`

在 `dataset.subset` 中设置：

- `all`
- `topk`
- `min_count`

它控制的是：

- 哪些 spike 会被提 waveform
- 哪些 spike 会被 encoder 编码
- 哪些 spike 会进入最后的测试流

#### `cam.memory_subset`

在 `cam.memory_subset` 中设置：

- `same_as_stream`
- `all`
- `topk`
- `min_count`

它控制的是：

- warmup 阶段哪些类允许被加载进 CAM 模板
- 哪些类属于 memory 内类
- 哪些类虽然会出现在测试流里，但不在 memory 内，因此应该被 reject

例如：

```json
"dataset": {
  "subset": {
    "mode": "topk",
    "topk": 50
  }
},
"cam": {
  "memory_subset": {
    "mode": "topk",
    "topk": 20,
    "selection_source": "pre_sampling"
  }
}
```

这表示：

- 编码和测试流里一共有 top50 类
- 但 CAM 只记住其中 top20 类
- 测试时其余类仍然会出现
- 它们不能被当成 memory 内类识别，理想行为应该是 reject

这比旧协议更接近真实问题：

“当 CAM 只能记住有限数量的活跃类时，它能否识别 memory 内类，并拒绝 memory 外类？”

#### `selection_source` 是什么

`cam.memory_subset.selection_source` 用来决定“top20 按哪个阶段的类计数来选”：

- `pre_sampling`
  按 waveform 下采样前的真实类计数来选。通常最推荐。

- `encoded`
  按最终 encoded dataset 里的类计数来选。

- `warmup`
  按 warmup 时间窗口里的类计数来选。

默认更推荐：

- `selection_source = pre_sampling`

因为如果你先把每类都裁到 `max_spikes_per_unit=250`，再去选 top20，
很多类的计数会被裁平，memory top20 就不再是真正的“最活跃 top20”。

### 6.2 encoder 方法

在 `encoder.method` 中设置：

- `ae`
- `pca`

对于 `ae`，可通过 `encoder.backend` 指定：

- `auto`
- `numpy`
- `external`

另外，`pca` 现在还支持：

- `encoder.binarize_mode = "median"`
  当前主线默认做法，每一维 PCA code 用各自中位数二值化，bit 更均衡。

- `encoder.binarize_mode = "zero"`
  兼容 `previous work` 里常见的 `PCA > 0` 二值化方式，适合做旧实验复现。

### 6.3 template initialization

在 `template_init.method` 中设置：

- `majority_vote`
- `medoid`
- `stable_mask`
- `multi_template`

### 6.4 match strategy

在 `matcher.method` 中设置：

- `hamming_nearest`
- `weighted_hamming`
- `margin_reject`
- `top2_margin`

### 6.5 update strategy

在 `update.method` 中设置：

- `none`
- `counter`
- `margin_ema`
- `confidence_weighted`
- `dual_template`
- `probabilistic`
- `growing`
- `cooldown`
- `top2_margin`

## 7. 怎么运行

下面所有命令默认你已经进入项目根目录 `spike_cam/`。

### 7.1 查看原始 label 分布

```bash
python scripts/inspect_labels.py --npz dataset/my_validation_subset_810000samples_27.00s.npz
```

这个脚本会帮助你快速判断：

- 一共有多少 units
- 长尾是否严重
- 是否适合从 `top10` / `top50` / `min_count>=500` 起步

### 7.2 只做编码与 bit 统计

```bash
python run_experiment.py --config configs/baseline_top10_ae16.json --encode-only
```

这一步会：

- 生成 / 读取 encoded dataset cache
- 保存 `encoded_stats.json`

### 7.3 检查 encoded dataset 质量

```bash
python scripts/inspect_encoded_dataset.py --encoded results/cache/encoded_cache/<your_cached_file>.npz
```

这个脚本会输出：

- bit mean
- bit entropy
- unique code 数量
- intra-class Hamming
- inter-class Hamming
- Hamming gap

### 7.4 跑 baseline

```bash
python run_experiment.py --config configs/baseline_top10_ae16.json
```

这会跑一个 sanity check：

- stream: top50 units
- CAM memory: top10 units
- 16-bit AE
- majority init
- static CAM
- chronological evaluation

### 7.5 跑主实验

```bash
python run_experiment.py --config configs/cam_compare_main.json
```

这是主实验配置，固定 encoder，比较多个 CAM update strategy。

当前默认语义是：

- stream: top50 units
- CAM memory: top20 units
- memory 外类会出现在测试流里，并用于检验 reject

### 7.5.1 跑一个最直接的 memory-aware 示例

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python run_experiment.py --config configs/memory_top20_stream_top50_external_ae16.json
```

这个配置最适合拿来理解新版协议，因为它非常直接：

- stream: top50
- memory: top20
- external AE
- chronological online evaluation

### 7.6 跑 external AE 的 pilot 实验

如果你想先做一轮更贴近毕设主线的 external AE pilot，可以直接跑下面这些配置：

#### subset sweep

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python run_experiment.py --config configs/pilot_subset_sweep_external_ae16.json
```

这个实验回答的问题是：

- 如果把 stream 的问题规模本身设成 top10、top20、top50
- 哪个更适合作为后续 memory-aware 主实验的起点
- 当类数增加时，accuracy / reject_rate 会怎么变化

#### CAM strategy pilot

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python run_experiment.py --config configs/pilot_cam_compare_external_top20.json
```

这个实验回答的问题是：

- 在固定 encoder、固定 top20 stream 下
- static、counter、margin_ema、dual_template 哪个更合适

#### bit budget pilot

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python run_experiment.py --config configs/pilot_bits_external_top20.json
```

这个实验回答的问题是：

- 在 top20 stream 下
- 8 / 16 / 32 bits 的 trade-off 是什么

### 7.6.1 复现并解释 previous work 与 thesis 主线差距

如果你想专门回答：

- 为什么 `previous work` 的 PCA 指纹看起来更有辨别力？
- 为什么 thesis 主线结果会显著更低？
- 差距主要来自数据、编码还是评估协议？

可以直接运行：

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python scripts/run_previous_work_gap_study.py
```

这个脚本会自动打包一整套对照实验，通常包括：

- `previous work` 保存下来的 encoded CSV 重放
- spikeinterface toy data 的 previous-work 风格重建
- 真实数据的 closed-set random split 对照
- 真实数据的 chronological 对照
- 真实数据的 `stream > memory` open-set 对照

结果会统一保存到：

- `results/experiments/<YYYY-MM-DD>/experiment_<timestamp>_previous_work_gap/`

里面会包含：

- `master_report.md`
- `master_report.html`
- `overall_summary.csv`
- `figures/`
- `tables/`

### 7.6.2 运行 toy thesis study

如果你想先在可控 toy 数据上系统比较 dynamic update，而不是一开始就被真实数据 separability 限制住，可以直接运行：

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python scripts/run_toy_thesis_study.py
```

这组实验会自动：

- 生成 4 个难度不同的 toy dataset
- 比较 `random / chronological` 协议差异
- 比较 stable / drift / open-memory 三种场景
- 比较多种 dynamic update strategy
- 比较 bit 宽度
- 比较 template initialization
- 补一组 PCA vs external AE 的 encoder 对照

输出目录会自动打包到：

- `results/experiments/<YYYY-MM-DD>/experiment_<timestamp>_toy_thesis_study/`

如果你当前只是想把毕设主线先讲清楚，通常非常推荐先看这套 toy study，因为它更容易回答：

- 什么时候 static 就够了
- 什么时候 dynamic update 才有价值
- 什么时候 open-set / unknown pressure 会让模板污染变严重

### 7.7 跑消融实验

#### bit 数消融

```bash
python run_experiment.py --config configs/bits_ablation.json
```

#### encoder 消融

```bash
python run_experiment.py --config configs/encoder_compare.json
```

#### initialization 消融

```bash
python run_experiment.py --config configs/init_ablation.json
```

### 7.8 只跑某几个 variant

```bash
python run_experiment.py --config configs/cam_compare_main.json --variant static --variant counter
```

### 7.9 画在线曲线

```bash
python scripts/plot_results.py --result_dir results/experiments/<YYYY-MM-DD>/cam_compare_main
```

这个脚本会生成：

- `summary_plots.png`

其中包含：

- cumulative accuracy
- per-window accuracy
- per-window reject rate
- template count
- cumulative updates
- cumulative wrong updates

### 7.10 自动整理实验报告

```bash
python scripts/build_experiment_report.py --result_dir results/experiments/<YYYY-MM-DD>/cam_compare_main
```

这个脚本会在结果目录里额外生成：

- `report.md`
- `metrics_table.csv`
- `comparison_metrics.png`
- `curve_overview.png`

这一步很适合在你跑完一批实验后立刻执行，因为它会自动把：

- 实验目的
- 共享参数
- 关键指标表
- 初步结论
- 可视化

整理成一个更接近“实验记录”的结构。

## 8. 结果目录说明

每个实验会保存在：

```text
results/experiments/<YYYY-MM-DD>/<experiment_name>/
├── config.json
├── summary.json
└── runs/
    ├── <variant_name>/
    │   ├── metrics.json
    │   ├── meta.json
    │   ├── encoded_stats.json
    │   ├── predictions.npz
    │   ├── curves.npz
    │   ├── confusion.npy
    │   └── confusion_labels.npy
    └── ...
```

### 各文件用途

- `config.json`
  本次实验完整配置

- `summary.json`
  所有变体的关键指标汇总

- `metrics.json`
  单个变体的详细指标

- `encoded_stats.json`
  对应 encoded dataset 的 bit 统计与 separability 诊断

- `predictions.npz`
  逐步在线预测结果，包括距离、是否接受、是否更新等

- `curves.npz`
  在线曲线数据，用于后续画图

- `confusion.npy`
  confusion matrix

## 9. 指标说明

### 基本性能指标

- `accuracy`
  严格口径的整体正确率，reject 也算错

- `macro_f1`
  对不同类更公平

- `balanced_accuracy`
  各类 recall 的平均

- `accept_rate`
  被 CAM 接受的比例

- `accepted_accuracy`
  只在被接受样本上的正确率

### reject 相关指标

- `reject_rate`
- `false_reject_count`
- `false_reject_rate`
- `false_accept_count`
- `false_accept_rate`

这里的 `known / unknown` 是相对于 **初始模板集合** 来定义的。

### online 过程指标

- `cumulative_accuracy`
- `per_window_accuracy`
- `per_window_reject_rate`
- `cumulative_updates`
- `cumulative_wrong_updates`
- `template_count`

### 内存代价

- `initial_template_count`
- `final_template_count`
- `max_template_count`
- `template_growth`
- `used_rows_over_capacity`

### 更新质量

- `update_count`
- `wrong_update_count`
- `wrong_update_rate`

`wrong_update_rate` 很重要，因为它能帮助你分析一个动态更新算法到底是在“适应数据”，还是在“污染模板”。

## 10. 推荐实验路线

### Stage A: baseline sanity check

先确认：

- waveform 提取没问题
- bits 没退化
- static CAM 至少能跑通

推荐配置：

- `configs/baseline_top10_ae16.json`

推荐理解成：

- stream: top50
- memory: top10

### Stage B: compare dynamic update strategies

固定 encoder 和 bit width，只比较不同 updater。

推荐配置：

- `configs/cam_compare_main.json`

推荐理解成：

- stream: top50
- memory: top20

### Stage C: bit budget ablation

比较：

- 8 bits
- 16 bits
- 32 bits

推荐配置：

- `configs/bits_ablation.json`

### Stage D: encoder comparison

比较：

- PCA
- AE

推荐配置：

- `configs/encoder_compare.json`

### Stage E: initialization comparison

比较：

- majority_vote
- medoid
- stable_mask

推荐配置：

- `configs/init_ablation.json`

### Stage F: error analysis

选出表现较好的方法后，重点分析：

- 哪些 unit 常被 reject
- 哪些错误集中在某些类之间
- update 是否真的带来长期收益
- wrong update 是否随时间积累
- memory growth 是否值得

## 11. 为什么默认建议这样做

### 为什么默认 `align_mode = min`

对 extracellular spike 来说，负峰通常更关键，所以默认先按最小值对齐更合理。

### 为什么不建议上来就全量 units

因为真实数据长尾很重。直接上全量 units 通常会导致：

- warmup 覆盖不足
- 稀有类模板太弱
- reject 很高
- 指标很难解释

更合理的起点通常是：

- `top10`
- `top50`
- `min_count >= 500`

但现在要进一步区分：

- `dataset.subset` 是 stream 大小
- `cam.memory_subset` 是 memory 大小

### 为什么必须 chronological evaluation

因为动态更新本质就是时间问题。如果随机打乱：

- 会破坏时间因果
- 会让未来信息泄漏到过去
- 会让 update strategy 看起来“虚高”

### 为什么先缓存 encoded dataset

因为 encoder 不是主角。缓存后有三个好处：

- 重复 CAM 实验更快
- CAM 比较更公平
- 你可以独立分析 bits 质量

## 12. 常见问题

### 1. 为什么结果很差？

常见原因：

- subset 太难
- bit 数太小
- threshold 不合适
- encoder 输出 separability 很弱
- warmup 太短

建议先：

1. 看 `inspect_labels.py`
2. 看 `encoded_stats.json`
3. 从 baseline 配置开始

### 2. 为什么 reject 很多？

可能原因：

- 初始模板不够好
- threshold 太严格
- warmup 样本太少
- 某些 unit 在 online 阶段发生明显变化

### 3. 为什么 dynamic update 没提升，甚至更差？

很常见。因为更新算法如果不够稳，会把错误信息写进模板。

这就是为什么本平台专门保存：

- `wrong_update_rate`
- `template_count`
- `cumulative_accuracy`

### 4. 为什么 `accepted_accuracy` 和 `accuracy` 差很多？

说明模型在“接受的样本里还行”，但 reject 过多，整体效果还是差。

这通常提示你要一起看：
- accept_rate
- reject_rate
- false_reject_rate

### 5. 为什么现在要区分 `dataset.subset` 和 `cam.memory_subset`？

因为这两个问题本来就不同：

- `dataset.subset`
  决定系统面对什么测试流

- `cam.memory_subset`
  决定 CAM 真正记住哪些类

如果把它们绑死成同一个 subset，就会变成：

- 测试流里只出现 memory 内类
- memory 外类根本不参与测试
- reject 能力就测不出来

而你真正想研究的是：

- CAM memory 有限
- 测试流更复杂
- 系统能否认出 memory 内类
- 并拒绝 memory 外类

## 13. 推荐的毕设写作叙事

你可以按下面这条主线来组织论文：

1. 任务背景：compressed spike representation + CAM identification
2. 研究重点：dynamic template update in CAM
3. 系统设计：raw data -> encoder -> CAM -> online evaluation
4. 主实验：固定 encoder，比 updater
5. 消融实验：bit 数、encoder、initialization
6. 错误分析：reject、wrong update、template growth
7. 结论：不同策略在 accuracy / stability / memory 之间的权衡

## 14. 最后建议

如果你是第一次正式跑实验，推荐顺序：

1. `inspect_labels.py`
2. `baseline_top10_ae16`
3. 看 `encoded_stats.json`
4. 看 `summary.json`
5. 跑 `cam_compare_main`
6. 再做 ablation

这样会比一开始就跑最复杂的大实验更稳，也更容易定位问题。
