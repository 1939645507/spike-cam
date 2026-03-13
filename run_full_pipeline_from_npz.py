"""
从 raw npz （类似 my_validation_subset_*.npz）一路跑到 CAM 评估的测试脚本。

Pipeline:
  npz -> 抽取波形 -> Autoencoder 编码成 bits -> 多种 CAM 动态策略评估

运行示例（在项目根目录）:

  python -m spike_cam.run_full_pipeline_from_npz

可选参数请用 `-h` 查看。
"""

import argparse
import os

from encoder import (
    WaveformExtractConfig,
    extract_waveforms_from_npz,
    encode_waveforms_with,
    autoencoder_bits_encoder,
)
from evaluate import evaluate_cam_on_encoded_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full AE -> bits -> CAM pipeline on a raw npz file.")
    parser.add_argument(
        "--npz",
        default="spike_cam/dataset/my_validation_subset_810000samples_27.00s.npz",
        help="原始 npz 路径（包含 raw_data/spike_times/spike_clusters）",
    )
    parser.add_argument("--waveform_len", type=int, default=79)
    parser.add_argument("--center_index", type=int, default=39)
    parser.add_argument("--align_mode", choices=["none", "max", "min"], default="max")
    parser.add_argument("--channel_select", choices=["max_abs", "fixed"], default="max_abs")
    parser.add_argument("--fixed_channel", type=int, default=0)
    parser.add_argument("--max_total_spikes", type=int, default=30000)
    parser.add_argument("--max_spikes_per_unit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ae_type", default="normal")
    parser.add_argument("--code_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--scale", default="minmax")

    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--train_frac", type=float, default=0.7)

    args = parser.parse_args()

    npz_path = args.npz
    if not os.path.isabs(npz_path):
        npz_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), npz_path)

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到 npz 文件: {npz_path}")

    print(f"[1/3] 从 npz 抽取波形: {npz_path}")
    cfg = WaveformExtractConfig(
        waveform_len=args.waveform_len,
        center_index=args.center_index,
        align_mode=args.align_mode,
        channel_select=args.channel_select,
        fixed_channel=args.fixed_channel,
    )

    waveforms, labels = extract_waveforms_from_npz(
        npz_path,
        cfg,
        max_total_spikes=args.max_total_spikes,
        max_spikes_per_unit=args.max_spikes_per_unit,
        seed=args.seed,
    )
    print(f"  waveforms={waveforms.shape}, labels={labels.shape}")


    print(f"[2/3] 使用 Autoencoder 编码为 bits (ae_type={args.ae_type}, code_size={args.code_size})")

    def _ae_encoder_fn(w, l):
        return autoencoder_bits_encoder(
            w,
            l,
            ae_type=args.ae_type,
            code_size=args.code_size,
            scale=args.scale,
            nr_epochs=args.epochs,
            verbose=1,
        )

    bits, labels_out = encode_waveforms_with(waveforms, labels, _ae_encoder_fn)
    print(f"  bits={bits.shape}, labels_out={labels_out.shape}")



    print("[3/3] 在多个 CAM 动态策略上评估")
    cam_variants = [
        ("Static", "static"),
        ("Counter", "counter"),
        ("MarginEMA", "margin_ema"),
        ("ConfWeighted", "conf_weight"),
        ("DualTemplate", "dual"),
        ("ProbCAM", "prob"),
        ("Growing", "growing"),
    ]

    results = evaluate_cam_on_encoded_dataset(
        bits=bits,
        labels=labels_out,
        cam_variants=cam_variants,
        train_frac=args.train_frac,
        threshold=args.threshold,
        seed=args.seed,
    )

    print("\n=== CAM 结果汇总 ===")
    print(f"(train_frac={args.train_frac}, threshold={args.threshold}, code_size={args.code_size})")
    for name, res in results.items():
        print(f"{name:12s}  acc={res.accuracy:.4f}")


if __name__ == "__main__":
    main()

