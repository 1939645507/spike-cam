"""
从现有的 spike_pca_dataset.csv 跑一遍所有 CAM 算法的快速脚本。

假设 CSV 头为:
  bit0,...,bit(B-1),unit_id
"""

import os

import numpy as np
import pandas as pd

from evaluate import evaluate_cam_on_encoded_dataset


def main() -> None:
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "spike_pca_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    bit_cols = [c for c in df.columns if c.startswith("bit")]
    if "unit_id" not in df.columns:
        raise ValueError("CSV 中缺少 unit_id 列")

    bits = df[bit_cols].values.astype(int)
    labels = df["unit_id"].values.astype(int)

    cam_variants = [
        ("Static", "static"),
        ("Counter", "counter"),
        ("MarginEMA", "margin_ema"),
        ("ConfWeighted", "conf_weight"),
        ("DualTemplate", "dual"),
        ("ProbCAM", "prob"),
        ("Growing", "growing"),
    ]

    print(f"载入 bits={bits.shape}, labels={labels.shape}")
    results = evaluate_cam_on_encoded_dataset(
        bits=bits,
        labels=labels,
        cam_variants=cam_variants,
        train_frac=0.7,
        threshold=15,
        seed=42,
    )

    print("\n=== 结果汇总 (train_frac=0.7, threshold=15) ===")
    for name, res in results.items():
        print(f"{name:12s}  acc={res.accuracy:.4f}")


if __name__ == "__main__":
    main()

