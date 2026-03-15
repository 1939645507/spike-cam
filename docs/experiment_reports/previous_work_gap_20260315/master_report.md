# Previous Work Gap Study Report

## 1. 研究目标

本研究专门用于解释一个现象：`previous work` 里的指纹和模板在旧实验里看起来有明显辨别力，但当前 thesis 主线实验结果却显著偏低。
为了定位原因，我们把问题拆成四层：

1. 旧项目保存下来的 encoded CSV 本身是否仍然可分？
2. 用 spikeinterface 按旧 protocol 重新生成 synthetic toy data 后，结果是否依然较好？
3. 真实数据在旧风格的 closed-set random split 下会怎样？
4. 真实数据一旦改成 chronological + memory-limited open-set，性能会掉多少？

## 2. 核心结论

- 旧 `previous work` 编码 CSV 的 separability 明显更强：`mean_hamming_gap=2.1125`，这说明旧数据/旧编码本身就更容易做模板匹配。
- 在当前 CAM 框架里直接重放旧 CSV，最佳静态 accuracy 仍可达到 `0.4109`，最佳阈值约为 `thr18`。这说明当前 CAM 核心不是完全失效的。
- 真实数据即使切回更接近旧协议的 `top8 + PCA32 + closed-set + random split`，最佳 accuracy 也只有 `0.2111`，显著低于旧 CSV。
- 同样的真实数据一旦改成 chronological split，最佳 accuracy 进一步降到 `0.2048`。这说明时间顺序评估本身就更难。
- 再进一步改成 `stream=top50 / memory=top20` 的 open-set memory-aware 协议后，最佳 accuracy 只有 `0.0442`，同时 reject/false accept tradeoff 明显变得更敏感。
- 在较容易的 `real top8 random` 条件下，`growing` 与 `static` 的最佳 accuracy 都约为 `0.2111`；但 `growing` 实际上没有发生任何更新，所以它不能算真正优于 static。
- 真正发生了更新的 dynamic 方法里，最好的是 `dual_template`，accuracy 为 `0.2085`，但 `wrong_update_rate=1.0000`，说明 dynamic 收益很有限，而且会引入明显模板污染风险。

## 3. 结果表

| 场景 | 最佳 variant | Accuracy | Reject Rate | Accepted Accuracy |
| --- | --- | ---: | ---: | ---: |
| Previous CSV random | thr18 | 0.4109 | 0.0000 | 0.4109 |
| Real top8 random | thr16 | 0.2111 | 0.0002 | 0.2111 |
| Real top8 chronological | thr18 | 0.2048 | 0.0000 | 0.2048 |
| Real top50 memory20 open | thr16 | 0.0442 | 0.0000 | 0.0442 |

## 4. 编码表示诊断

| 数据源 | Mean Hamming Gap | Unique Code Ratio | Num Units |
| --- | ---: | ---: | ---: |
| prevwork_csv | 2.1125 | 0.9993 | 8 |
| toy_recreated | 2.2240 | 0.9994 | 8 |
| real_top8_oldlike_random | 0.1120 | 0.9929 | 8 |
| real_top8_oldlike_chrono | 0.1120 | 0.9929 | 8 |
| real_top50_mem20_open | 0.3934 | 0.9706 | 50 |

这里最关键的一点是：旧 CSV 的 `mean_hamming_gap` 明显高于真实数据。也就是说，旧问题本身就更容易，而不是只有 CAM 算法在旧 notebook 里更“神奇”。

## 5. 规律解释

### 5.1 为什么 previous work 更容易

- synthetic toy data，类更干净、漂移更小、类别数更少。
- 旧 protocol 更接近 closed-set，且 evaluation 通常更宽松。
- 旧 PCA 编码使用 `PCA > 0` 二值化，我们已经在本 study 里专门兼容这一点。

### 5.2 为什么真实数据会明显更难

- 真实 extracellular spike 的类内变化更大、类间更像。
- chronological split 比 random split 更能暴露 drift 和 warmup 覆盖不足。
- memory-limited open-set 协议要求系统既要认 memory 内类，又要拒绝 memory 外类，本质上比旧 closed-set 更难。

## 6. 对 thesis 主线的意义

这组实验说明：当前 thesis 主线结果低，不一定是代码坏了，更多是因为研究问题被升级了。

如果你要在论文里解释这一点，最自然的叙事是：

1. 先承认旧实验在 synthetic / closed-set 条件下可以得到更高结果。
2. 再说明 thesis 主线切换到了更真实、更严格的 protocol。
3. 因此主实验不应简单追求绝对 accuracy，而应更关注 reject、wrong update、memory usage 和 online stability。

## 7. 推荐后续动作

- 如果目标是先恢复一个“看起来像 previous work”的 baseline，就用 `top8 + PCA32 + closed-set + random split + zero-threshold`。
- 如果目标是 thesis 主线，就继续坚持 chronological + memory-aware open-set，但要把阈值、reject 和 wrong-update 当成主分析对象。
- 如果还想继续缩小两边差距，下一步最值得做的是：把 real-data 前端编码单独再强化，例如更稳的 channel neighborhood、对齐、或监督式 encoder。
