# Autoencoders-in-Spike-Sorting
Autoencoders, a type of neural network that allow for unsupervised learning, can be used in the feature extraction of spike sorting.

This study has been published in PLOS One:
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282810
- DOI: 10.1371/journal.pone.0282810

## Citation
We would appreciate it if you cite the paper when you use this work:

- For Plain Text:
```
E.-R. Ardelean, A. Coporîie, A.-M. Ichim, M. Dînșoreanu, and R. C. Mureșan, “A study of autoencoders as a feature extraction technique for spike sorting,” PLOS ONE, vol. 18, no. 3, p. e0282810, Mar. 2023, doi: 10.1371/journal.pone.0282810.
```

## Setup
The 'requirements.txt' file indicates the dependencies required for running the code. 

The synthetic data used in this study can be downloaded from: 
https://1drv.ms/u/s!AgNd2yQs3Ad0gSjeHumstkCYNcAk?e=QfGIJO
or
https://www.kaggle.com/datasets/ardeleanrichard/simulationsdataset.

The real data used in this study can be downloaded from:
https://www.kaggle.com/datasets/ardeleanrichard/realdata
or in the 'real_data' folder of the repository.


In the constants.py file the path to the DATA folder can be set. We recommend the following structure for the data:

DATA/
* TINS/
  * M045_009/ : insert the real data files
* SIMULATIONS/ : insert the synthetic data files


# Contact
If you have any questions, feel free to contact me. (Email: ardeleaneugenrichard@gmail.com)

---

## 使用指南（中文说明，针对当前项目）

这一部分是为了方便你在自己的 spike sorting 毕设项目里使用本工具，主要分为：

1. 环境与依赖安装  
2. 数据准备（DATA 文件夹结构）  
3. 如何运行仿真数据 / 真实数据的 AE 实验  
4. 如何只“拿 encoder 用”，把 spike 波形 encode 为低维特征

### 1. 环境与依赖

进入本目录（即 `Autoencoders-in-Spike-Sorting/`）所在的虚拟环境，安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 中比较关键的是：

- `tensorflow==2.13.0`（+ Keras，代码现在用的是 tf.keras）
- `numpy, scipy, scikit-learn, matplotlib, seaborn, pandas` 等

> 建议：在你自己的 venv 里安装，避免和系统 Python 冲突。

### 2. 数据准备（DATA 目录）

本仓库有两类数据：

- **SIMULATIONS**：仿真 spike 数据（`.mat`）
- **TINS**：真实 TINS 记录

推荐的目录结构（相对 `Autoencoders-in-Spike-Sorting/`）：

```text
Autoencoders-in-Spike-Sorting/
  utils/constants.py      # 这里设置 DATA_FOLDER_PATH
  DATA/
    SIMULATIONS/
      simulation_1.mat
      simulation_2.mat
      ...
      ground_truth.mat
    TINS/
      M045_009/
        ...  # 真实数据文件（.spktwe / .spikew 等）
```

在 `utils/constants.py` 中可以看到默认设置：

```python
DATA_FOLDER_PATH = "../../DATA/"
REAL_DATA_FOLDER_PATH = DATA_FOLDER_PATH + f'/TINS/'
SYNTHETIC_DATA_FOLDER_PATH = DATA_FOLDER_PATH + f'/SIMULATIONS/'
```

如果你把 DATA 放在别的地方，只需要改这里的路径即可。

### 3. 如何运行自带实验脚本

#### 3.1 仿真数据：`run_methods_on_synthetic_data`

入口在 `run.py`：

```bash
cd Autoencoders-in-Spike-Sorting
python run.py
```

`run.py` 默认会执行：

```python
if __name__ == '__main__':
    run_methods_on_synthetic_data()
    # run_methods_on_real_data()
```

`run_methods_on_synthetic_data()` 的流程：

1. 从 `dataset_parsing/simulations_dataset.py` 载入仿真数据：

   ```python
   data, labels = ds.get_dataset_simulation(simNr=SIM_NR, align_to_peak=2)
   ```

   - `simNr` 控制使用哪一个仿真数据集。
   - `align_to_peak=2` 表示对齐 spike 峰值到固定位置。

2. 对仿真数据跑传统降维方法：PCA / ICA / Isomap，并调用 `visualization/scatter_plot.py` 画出 2D 散点图。
3. 依次对每种 AE 变体调用 `run_autoencoder(...)`：

   ```python
   for ae_type in ["shallow", "normal", "tied", "contractive", "orthogonal",
                   "ae_pca", "ae_pt", "lstm", "fft", "wfft"]:
       run_autoencoder(
           data_type="sim", simulation_number=SIM_NR,
           data=None, labels=None, gt_labels=None, index=None,
           ae_type=ae_type, ae_layers=np.array(LAYERS), code_size=2,
           output_activation='tanh', loss_function='mse', scale="minmax",
           nr_epochs=EPOCHS, dropout=0.0,
           doPlot=True, verbose=0
       )
   ```

   - `ae_type` 控制使用哪种结构（普通 AE / tied weights / contractive / LSTM / FFT 特征等）。
   - `code_size=2`：bottleneck 维度为 2，方便可视化。
   - 如果 `doPlot=True`，会在 `figures/` 下生成重构波形/聚类散点图。

#### 3.2 真实数据：`run_methods_on_real_data`

在 `run.py` 中还有一个函数：

```python
def run_methods_on_real_data():
    EPOCHS = 100
    LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
    CHANNEL = 17

    units_in_channel, labels = get_tins_data()
    data = units_in_channel[CHANNEL - 1]
    data = np.array(data)
    labels = np.array(labels[CHANNEL-1])
    ...
```

它会：

1. 用 `dataset_parsing/read_tins_m_data.get_tins_data()` 解析 TINS 文件，得到某一通道的 spike 波形与标签。
2. 先跑 PCA / ICA / Isomap 可视化真实数据。
3. 再对同一通道数据依次训练不同 `ae_type` 的自编码器，绘制 AE latent 的聚类散点图。

要启用它，只需改 `run.py` 末尾：

```python
if __name__ == '__main__':
    # run_methods_on_synthetic_data()
    run_methods_on_real_data()
```

然后：

```bash
python run.py
```

### 4. 如何在你的项目中“只用 encoder”

你毕设的需求是：**已经有 spike 波形数据（例如从 `spike_cam.encoder.extract_waveforms_from_npz` 得到的 `(waveforms, labels)`），想直接用这里的 AE encoder 把它们压缩到低维 latent / bits，用来喂给 CAM。**

推荐两种方式：

#### 4.1 直接调用 `run_autoencoder(...)`

`ae_function.run_autoencoder` 支持传入“已经准备好的 spike 矩阵”：

```python
from spike_cam.encoder.Autoencoders-in-Spike-Sorting.ae_function import run_autoencoder
import numpy as np

# waveforms: (N, L) float32 (你的 spike 片段)
# labels:    (N,)   int （可选，用于 shuff / noNoise）

features, _, _ = run_autoencoder(
    data_type="m0",           # 直接用矩阵 data
    simulation_number=None,   # 不用 sim 数据
    data=waveforms,
    labels=None,
    gt_labels=labels,         # 若你想保持 gt 顺序
    index=None,
    ae_type="normal",         # 或 "shallow", "tied", "contractive" 等
    ae_layers=np.array([70, 60, 50, 40, 30]),
    code_size=8,              # 这里可以设成 8/16/32 等
    output_activation='tanh',
    loss_function='mse',
    scale="minmax",
    nr_epochs=50,
    dropout=0.0,
    doPlot=False,
    verbose=1,
)

# features: (N, code_size) AE 的 latent 表示
```

你可以在自己的 encoder 管线里，把 `features` 进一步二值化成 bits（例如 `features > 0`），然后配合 `spike_cam` 的 CAM 框架使用。

#### 4.2 自己构造 AutoencoderModel 并复用权重

如果你希望更细粒度地控制 encoder（比如固定网络结构、多次复用同一个训练好的模型）：

1. 使用 `ae_function.run_autoencoder` 训练并 `saveWeights=True`，保存到 `utils/ae_parameters.MODEL_PATH` 指定的目录。
2. 在你自己的代码中，导入 `AutoencoderModel`，构建相同结构，然后调用 `.load_weights(...)`，再用 `encoder = model.return_encoder()` 得到 encoder 子模型。
3. 把你的任意 spike 波形矩阵送入 `encoder.predict(...)` 得到 latent。

这部分需要你对 Keras/TensorFlow 稍微熟悉一点，但整体流程和 `ae_function.py` 中的用法是一致的。

---

## 5. 小结（如何和 `spike_cam` 联动）

- 用 `spike_cam.encoder.extract_waveforms_from_npz` 或 `spikeinterface` 等方式得到 `(waveforms, labels)`。
- 再在 Python 里导入本目录下的 `ae_function.run_autoencoder`，选择合适的 `ae_type / code_size` 把波形 encode 成低维 latent。
- 把 latent 二值化成 bits（比如 `bits = (features > 0).astype(int)`），就可以作为 `spike_cam` 框架的输入：
  - `templates.py` 生成初始模板
  - `CAM + update_strategies.py` 进行动态匹配
  - `evaluate.py` 统一评估各个动态更新算法的效果。

如果你后面希望我帮你把一个具体的 `my_encoder(waveforms, labels)` 函数写好，直接封装对 `run_autoencoder` 的调用，也可以告诉我你想用哪种 `ae_type / code_size / scale`，我可以直接在 `spike_cam/encoder` 里加一个现成可用的函数。 
