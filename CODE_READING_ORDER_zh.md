# 代码阅读顺序建议

这份文档是给“准备接手维护这套平台，并且后面要写毕设”的你看的。

建议不要一上来就从某个算法文件开始啃。最省力的顺序是：

## 1. 先看整体入口

先看：

- [run_experiment.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/run_experiment.py)
- [experiment_runner.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/experiment_runner.py)

为什么先看这两个：

- 它们决定一条完整实验是怎么被串起来的
- 你能最快知道“配置从哪进来，结果往哪出去”
- 你会先建立全局流程图，而不是陷进某个局部实现

读的时候重点看：

1. `load_config(...)`
2. `run_experiment_suite(...)`
3. `prepare_encoded_dataset(...)`
4. `select_memory_labels(...)`
5. `run_single_experiment(...)`

如果你只想先搞清楚“这个项目到底怎么跑”，看这两份就够了。

## 2. 再看配置系统

接着看：

- [config.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/config.py)
- [configs/baseline_top10_ae16.json](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/configs/baseline_top10_ae16.json)
- [configs/cam_compare_main.json](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/configs/cam_compare_main.json)

这一层的目标是理解：

- 一个实验到底有哪些参数
- 哪些参数控制数据
- 哪些参数控制 stream subset
- 哪些参数控制 CAM memory subset
- 哪些参数控制 encoder
- 哪些参数控制 CAM
- variant 是怎么展开的

如果你以后要加新实验，通常第一步就是改 config，不是改主代码。

## 3. 然后看数据流

再看：

- [dataio.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/dataio.py)

你需要重点理解 5 个东西：

1. `WaveformDataset`
2. `EncodedDataset`
3. `subset_by_label_rule(...)`
4. `subset_by_count_map(...)`
5. `chronological_split(...)`

这是整个项目里非常关键的一层，因为它决定了：

- 哪些类会进入编码/测试流
- 哪些类会真正进入 CAM memory
- 时间顺序有没有被保留
- warmup / online stream 怎么切

如果你关心“1200 类太多，CAM 不可能全存怎么办”，这里就是第一站。

## 4. 再看 encoder

然后看：

- [encoder.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/encoder.py)

建议按这个顺序看：

1. `encode_waveform_dataset(...)`
2. `build_encoder(...)`
3. `PCABinaryEncoder`
4. `NumpyAutoencoderBinaryEncoder`
5. `ExternalAutoencoderBinaryEncoder`
6. `compute_bit_statistics(...)`

读这一层时要记住：

- encoder 不是主研究对象
- 但它决定 bits 质量
- 后面 CAM 实验好不好，很多时候先得看 bits separability 行不行

## 5. 再看 CAM core

接下来读：

- [cam_core.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/cam_core.py)

重点看：

1. `MatchResult`
2. `UpdateResult`
3. `TemplateRows`
4. `CAM.load_templates(...)`
5. `CAM.process(...)`

你会在这里真正理解：

- CAM 里每一行存了什么
- row metadata 是什么
- match 和 update 是怎样被分开的

## 6. 再看模板初始化

继续看：

- [templates.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/templates.py)

推荐顺序：

1. `build_template_rows(...)`
2. `majority_vote_templates(...)`
3. `medoid_templates(...)`
4. `stable_mask_templates(...)`
5. `multi_template_templates(...)`

这一层和 updater 不一样，它只决定“起点模板长什么样”。

## 7. 再看 matcher

再看：

- [match_strategies.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/match_strategies.py)

推荐顺序：

1. `HammingNearestMatch`
2. `WeightedHammingMatch`
3. `MarginRejectMatch`
4. `Top2MarginMatch`

先把“怎么匹配 / 怎么拒识”弄清楚，再去看 updater 会更轻松。

## 8. 最后重点看 updater

最后重点读：

- [update_strategies.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/update_strategies.py)

建议阅读顺序：

1. `NoUpdate`
2. `CounterUpdate`
3. `MarginEmaUpdate`
4. `ConfidenceWeightedUpdate`
5. `DualTemplateUpdate`
6. `ProbabilisticUpdate`
7. `GrowingUpdate`
8. `CooldownUpdate`
9. `Top2MarginUpdate`

原因是这个顺序基本对应：

- 从最简单到更复杂
- 从静态对照到动态策略
- 从单模板小改动到更复杂的模板管理

如果你后面论文主线要写“动态模板更新策略比较”，这一部分会是你最常回来看和修改的代码。

## 9. 最后再看指标和结果

补上：

- [metrics.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/metrics.py)
- [scripts/inspect_encoded_dataset.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/scripts/inspect_encoded_dataset.py)
- [scripts/plot_results.py](/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/scripts/plot_results.py)

重点理解：

- accuracy 和 accepted_accuracy 的区别
- reject_rate / false_reject_rate / false_accept_rate 的含义
- wrong_update_rate 怎么解释
- curves.npz 里到底存了什么

## 10. 如果你要改代码，优先这样改

### 想换 subset 规则

先改 config：

- `dataset.subset.mode`
- `dataset.subset.topk`
- `dataset.subset.min_count`

### 想换 encoder

先改 config：

- `encoder.method`
- `encoder.backend`
- `encoder.code_size`

### 想加新的 matcher

先改：

- `match_strategies.py`
- `experiment_runner.py` 里的 `build_match_strategy(...)`

### 想加新的 updater

先改：

- `update_strategies.py`
- `experiment_runner.py` 里的 `build_update_strategy(...)`

### 想加新的指标

先改：

- `metrics.py`
- 如有必要，补 `scripts/plot_results.py`

## 11. 一个最实用的阅读路线

如果你只有 30 分钟，推荐这样看：

1. `README_zh.md`
2. `run_experiment.py`
3. `experiment_runner.py`
4. `configs/baseline_top10_ae16.json`
5. `dataio.py`

如果你有 2 小时，推荐这样看：

1. `README_zh.md`
2. `CODE_READING_ORDER_zh.md`
3. `experiment_runner.py`
4. `config.py`
5. `dataio.py`
6. `cam_core.py`
7. `templates.py`
8. `match_strategies.py`
9. `update_strategies.py`
10. `metrics.py`

如果你准备开始正式改算法，推荐这样看：

1. `cam_core.py`
2. `match_strategies.py`
3. `update_strategies.py`
4. `metrics.py`
5. 相关 config 文件

## 12. 最后的建议

不要一开始就试图把每一行代码都看懂。

对这个项目来说，更好的方法是：

1. 先理解 pipeline
2. 再理解数据契约
3. 再理解 CAM 的 match / update 分层
4. 最后再看具体算法

这样你后面不管是继续做实验、改配置、加方法，还是写毕设，都会轻松很多。
