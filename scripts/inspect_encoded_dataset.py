"""Inspect a cached encoded dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio import EncodedDataset
from encoder import compute_bit_statistics


def main() -> None:
    """Print encoded-dataset separability diagnostics."""

    parser = argparse.ArgumentParser(description="Inspect a cached encoded dataset (.npz).")
    parser.add_argument("--encoded", required=True, help="Path to encoded dataset npz.")
    parser.add_argument("--max_pairs", type=int, default=2000, help="Sampled pair count for Hamming diagnostics.")
    parser.add_argument("--json-out", default=None, help="Optional JSON stats output path.")
    parser.add_argument("--csv-out", default=None, help="Optional one-row CSV output path.")
    parser.add_argument("--plot-out", default=None, help="Optional per-bit plot output path.")
    args = parser.parse_args()

    dataset = EncodedDataset.load_npz(Path(args.encoded))
    stats = compute_bit_statistics(dataset, max_pairs=args.max_pairs)

    lines = []
    lines.append("=== Encoded Dataset Summary ===")
    for key in [
        "num_spikes",
        "bit_width",
        "num_units",
        "unique_code_count",
        "unique_code_ratio",
        "bit_mean_mean",
        "bit_mean_std",
        "bit_entropy_mean",
        "bit_entropy_std",
        "mean_intra_hamming",
        "mean_inter_hamming",
        "mean_hamming_gap",
    ]:
        lines.append(f"{key}: {stats[key]}")

    lines.append("")
    lines.append("=== First 10 Per-Bit Means ===")
    lines.append(str(np.asarray(stats["per_bit_mean"][:10])))
    lines.append("")
    lines.append("=== First 10 Per-Bit Entropies ===")
    lines.append(str(np.asarray(stats["per_bit_entropy"][:10])))
    print("\n".join(lines))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.csv_out:
        fieldnames = [
            "num_spikes",
            "bit_width",
            "num_units",
            "unique_code_count",
            "unique_code_ratio",
            "bit_mean_mean",
            "bit_mean_std",
            "bit_entropy_mean",
            "bit_entropy_std",
            "mean_intra_hamming",
            "mean_inter_hamming",
            "mean_hamming_gap",
        ]
        with Path(args.csv_out).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({key: stats.get(key) for key in fieldnames})
    if args.plot_out:
        per_bit_mean = np.asarray(stats["per_bit_mean"], dtype=np.float32)
        per_bit_entropy = np.asarray(stats["per_bit_entropy"], dtype=np.float32)
        bit_axis = np.arange(per_bit_mean.shape[0])
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].bar(bit_axis, per_bit_mean, color="#4e79a7")
        axes[0].set_ylabel("Bit Mean")
        axes[0].set_title("Encoded Bit Statistics")
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[1].bar(bit_axis, per_bit_entropy, color="#f28e2b")
        axes[1].set_ylabel("Bit Entropy")
        axes[1].set_xlabel("Bit Index")
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
