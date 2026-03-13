import torch
import os
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from tqdm import tqdm

# 导入你的模型和数据类
from models.BasicBlocks import Encoder
from utils.data_loader import AchillesDataset

def extract_and_save_latents(
    encoder_weights_path, 
    data_pt_path, 
    output_path, 
    waveform_len=32, 
    num_channels=20,
    batch_size=512
):
    # 1. 设备配置 (优先使用 MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 初始化编码器并加载权重
    print(f"Loading Encoder from {encoder_weights_path}...")
    encoder = Encoder(waveform_len=waveform_len, num_channels=num_channels)
    
    # 加载 safetensors 权重
    state_dict = load_file(encoder_weights_path)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval() # 必须设为 eval 模式

    # 3. 加载数据集
    print(f"Loading Dataset from {data_pt_path}...")
    dataset = AchillesDataset(data_pt_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    latent_list = []

    # 4. 开始提取
    print("Extracting latents...")
    with torch.no_grad(): # 禁用梯度计算，节省内存和时间
        for batch in tqdm(dataloader):
            waveform = batch["waveform"].to(device)
            channel_max = batch["channel_max"].to(device)
            channel_min = batch["channel_min"].to(device)
            labels = batch["label"] # 标签不需要上 GPU
            
            # 拼接通道特征 (与训练时保持一致)
            channel_max_min = torch.concat([channel_max, channel_min], dim=1)
            
            # 前向传播得到潜变量
            latents = encoder(waveform, channel_max_min)
            
            # 将结果转回 CPU 并存储
            latents_cpu = latents.cpu()
            
            for i in range(len(latents_cpu)):
                latent_list.append({
                    "latent": latents_cpu[i],
                    "label": labels[i]
                })

    # 5. 保存为新的 pt 文件
    print(f"Saving {len(latent_list)} samples to {output_path}...")
    torch.save(latent_list, output_path)
    print("Done!")

if __name__ == "__main__":
    # --- 配置路径 ---
    # 请根据你的实际 checkpoint 步数修改路径
    CHECKPOINT_STEP = 10000 
    BASE_DIR = "/Users/zhangxinyuanmacmini/Desktop/BMI_2026/VAE/SpikeSorting-main"
    
    weights_path = os.path.join(BASE_DIR, f"output/autoencoder_test/checkpoint-{CHECKPOINT_STEP}/encoder/model.safetensors")
    input_data = os.path.join(BASE_DIR, "data/Achilles_Shank2_train_0.2.pt")
    output_data = os.path.join(BASE_DIR, f"data/Achilles_Latents_Step_{CHECKPOINT_STEP}.pt")

    extract_and_save_latents(
        encoder_weights_path=weights_path,
        data_pt_path=input_data,
        output_path=output_data
    )