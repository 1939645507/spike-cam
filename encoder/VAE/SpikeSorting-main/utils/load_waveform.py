# T. S. Liang @ HKU EEE, Jan. 23rd 2026
# Modified for 20% subset split.

import torch
import numpy as np
import os

# 1. 加载数据
data_path = '../data/Achilles_10252013_Shank2_dataset_v2.npz'
print(f"Loading raw data from {data_path}...")
data = np.load(data_path)

arrays = [
    data['main_channel_waveform'],
    data['label'],
    data['spike_time'],
    data['channel_max'],
    data['channel_min'],
    data['main_channel_index']
]

total_samples = len(arrays[0])
print(f"Total samples in NPZ: {total_samples}")

# 2. 核心修改：确定 20% 子集的索引
seed = 42
generator = torch.Generator().manual_seed(seed)
# 先生成所有数据的随机排列
all_indices = torch.randperm(total_samples, generator=generator).tolist()

# 只取前 20% 的索引
subset_fraction = 0.2
subset_size = int(total_samples * subset_fraction)
subset_indices = all_indices[:subset_size]

print(f"Taking {subset_fraction*100}% subset: {subset_size} samples")

# 3. 在这 20% 的子集中划分训练集和验证集 (8:2)
split_idx = int(subset_size * 0.8)
train_indices = subset_indices[:split_idx]
val_indices = subset_indices[split_idx:]

# 4. 只处理和转换这部分需要的样本 (节省内存和时间)
def create_set(indices_list, name):
    print(f"Processing {name} samples...")
    dataset = []
    for i in indices_list:
        sample = {
            'waveform': torch.tensor(arrays[0][i], dtype=torch.float32),
            'label': torch.tensor(arrays[1][i], dtype=torch.float32),
            'spike_time': torch.tensor(arrays[2][i], dtype=torch.float32),
            'channel_max': torch.tensor(arrays[3][i], dtype=torch.float32),
            'channel_min': torch.tensor(arrays[4][i], dtype=torch.float32),
            'main_channel_index': torch.tensor(arrays[5][i], dtype=torch.float32)
        }
        dataset.append(sample)
    return dataset

train_set = create_set(train_indices, "Train")
val_set = create_set(val_indices, "Val")

# 5. 保存结果
# 建议在文件名中加入 0.2 标识，防止覆盖你之前的大文件
train_output_path = '../data/Achilles_Shank2_train_0.2.pt'
val_output_path = '../data/Achilles_Shank2_val_0.2.pt'

print("Saving files...")
torch.save(train_set, train_output_path)
torch.save(val_set, val_output_path)

print("-" * 30)
print(f"✅ Subset split completed with seed {seed}")
print(f"   Subset Total: {subset_size}")
print(f"   Train set size: {len(train_set)} (Saved to: {train_output_path})")
print(f"   Val set size:   {len(val_set)}   (Saved to: {val_output_path})")