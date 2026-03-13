# T. S. Liang @ HKU EEE, Jan. 29th 2026
# Email: sliang57@connect.hku.hk
# Data loader and data preprocessing of the Achilles dataset.

import torch
from torch.utils.data import Dataset, DataLoader
from math import sqrt

def compute_waveform_stats(dataset):
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for i in range(len(dataset)):
        wf = dataset[i]['waveform'].float()  # [T]

        total_sum += wf.sum().item()
        total_sq_sum += (wf ** 2).sum().item()
        total_count += wf.numel()

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean ** 2
    std = sqrt(var)

    return mean, std

class AchillesDataset(Dataset):
    def __init__(self, pt_path, mean = 1.4784424140308792, std = 104.25511002217199):
        """
        Args:
            pt_path (string): Path to the .pt file (e.g., '../data/Achilles_Shank1_train.pt')
        """
        self.mean = mean
        self.std = std

        self.data = torch.load(pt_path)
        print(f"Loaded {len(self.data)} samples from {pt_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        waveform = sample['waveform'] # [T]
        
        if self.mean is not None and self.std is not None:
            waveform = (waveform - self.mean) / (self.std + 1e-8)
            channel_max = (sample['channel_max'] - self.mean)/ ((self.std + 1e-8))
            channel_min = (sample['channel_min'] - self.mean)/ ((self.std + 1e-8))

        label = sample['label']

        return {
            'waveform': waveform,       # Shape: [T]
            'label': label,             # Shape: scalar or [1]
            'channel_max': channel_max, # Shape: [C]
            'channel_min': channel_min, # Shape: [C]
            'main_channel_index': sample['main_channel_index'] # Shape: [1]
        }

def prepare_dataset(train_path, 
                    val_path, 
                    train_batch_size=32, 
                    val_batch_size=32, 
                    num_workers=4):

    print(f"Loading Train Dataset from: {train_path}")
    train_dataset = AchillesDataset(train_path)
    
    print(f"Loading Val Dataset from: {val_path}")
    val_dataset = AchillesDataset(val_path)

    # mean, var = compute_waveform_stats(train_dataset)
    # print(f"Global waveform mean: {mean}")
    # print(f"Global waveform variance: {var}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset preparation complete.")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    return train_loader, val_loader

if __name__ == "__main__":

    prepare_dataset(train_path = "/Users/zhangxinyuanmacmini/Desktop/BMI_2026/VAE/SpikeSorting-main/data/Achilles_Shank2_train.pt", val_path = "/Users/zhangxinyuanmacmini/Desktop/BMI_2026/VAE/SpikeSorting-main/data/Achilles_Shank2_vol.pt")