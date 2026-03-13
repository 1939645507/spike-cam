import numpy as np

npz_path = "/Users/ceker/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/文档/hku/elec4848/code/spike_cam/dataset/my_validation_subset_810000samples_27.00s.npz"

pack = np.load(npz_path)
labels = np.asarray(pack["spike_clusters"]).astype(np.int64)

unique_labels, counts = np.unique(labels, return_counts=True)
order = np.argsort(counts)[::-1]

print("=== 基本统计 ===")
print("总 spike 数:", labels.shape[0])
print("unique labels 数:", unique_labels.shape[0])
print("最小类样本数:", counts.min())
print("中位数类样本数:", int(np.median(counts)))
print("最大类样本数:", counts.max())

print("\n=== top 20 最大类 ===")
for i in order[:20]:
    print(f"label={int(unique_labels[i])}, count={int(counts[i])}")

print("\n=== 小类统计 ===")
for th in [1, 2, 5, 10, 20, 50, 100]:
    n = np.sum(counts <= th)
    print(f"样本数 <= {th} 的类有: {int(n)} 个")

print("\n=== 大类统计 ===")
for th in [100, 200, 500, 1000]:
    n = np.sum(counts >= th)
    print(f"样本数 >= {th} 的类有: {int(n)} 个")