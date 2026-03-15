import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

def run_pytorch_kmeans(latent_pt_path, n_clusters=15, max_iters=100):
    # 1. 设备配置
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载数据
    print(f"Loading latents from {latent_pt_path}...")
    data = torch.load(latent_pt_path)
    
    # 统一数据格式为 Tensor 并移动到 GPU
    if isinstance(data, list):
        latents = torch.stack([x['latent'] for x in data]).to(device)
        labels_gt = torch.stack([x['label'] for x in data]).cpu().numpy()
    else:
        latents = data['latents'].to(device)
        labels_gt = data['labels'].cpu().numpy()

    N, D = latents.shape
    print(f"Data shape: {N} samples, {D} dimensions")

    # 3. PyTorch K-means 核心逻辑
    print(f"Starting PyTorch K-means (k={n_clusters})...")
    start_time = time.time()

    # 随机初始化中心点 (K-means++)
    # 简单起见，先使用随机选择
    indices = torch.randperm(N)[:n_clusters]
    centroids = latents[indices]

    for i in range(max_iters):
        # 计算欧氏距离 (N, n_clusters)
        # 利用 torch.cdist 高效计算
        dist = torch.cdist(latents, centroids)
        
        # 分配最近的簇索引
        cluster_labels = torch.argmin(dist, dim=1)
        
        # 更新中心点
        new_centroids = torch.zeros_like(centroids)
        for j in range(n_clusters):
            mask = (cluster_labels == j)
            if mask.any():
                new_centroids[j] = latents[mask].mean(0)
            else:
                # 处理空簇：如果某个簇没有点，随机找一个点作为新中心
                new_centroids[j] = latents[torch.randint(0, N, (1,))]
        
        # 检查中心点位移量
        shift = torch.norm(new_centroids - centroids)
        centroids = new_centroids
        
        if i % 10 == 0:
            print(f"Iteration {i:3d} | Centroid shift: {shift:.6f}")
        
        if shift < 1e-5:
            print(f"Converged at iteration {i}")
            break
    
    end_time = time.time()
    cluster_labels_cpu = cluster_labels.cpu().numpy()
    print(f"✅ Clustering finished in {end_time - start_time:.2f} seconds.")

    # 4. 评估
    ari = adjusted_rand_score(labels_gt, cluster_labels_cpu)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    # 5. 可视化 (PCA 降维)
    print("Generating PCA visualization...")
    # PCA 在 CPU 上运行非常稳定
    latents_cpu = latents.cpu().numpy()
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents_cpu)

    plt.figure(figsize=(16, 7))

    # 子图 1: Ground Truth
    plt.subplot(1, 2, 1)
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels_gt, cmap='tab20', s=2, alpha=0.5)
    plt.title("Ground Truth Labels (Original)")
    plt.colorbar(label="Label ID")

    # 子图 2: PyTorch K-means
    plt.subplot(1, 2, 2)
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=cluster_labels_cpu, cmap='tab20', s=2, alpha=0.5)
    plt.title(f"PyTorch K-means Clustering (k={n_clusters})")
    plt.colorbar(label="Cluster ID")

    plt.tight_layout()
    
    # 保存图片
    save_path = latent_pt_path.replace(".pt", "_pytorch_kmeans.png")
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
    
    # 显示图片
    #plt.show()

if __name__ == "__main__":
    # 配置你的文件路径
    LATENT_FILE = "/Users/zhangxinyuanmacmini/Desktop/BMI_2026/VAE/SpikeSorting-main/data/Achilles_Latents_Step_10000.pt"
    
    if os.path.exists(LATENT_FILE):
        run_pytorch_kmeans(LATENT_FILE, n_clusters=6)
    else:
        print(f"Error: File {LATENT_FILE} not found. Please check the path.")