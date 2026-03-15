"""Inspect label distribution in a raw spike ``.npz`` dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Print label distribution diagnostics for one raw dataset."""

    parser = argparse.ArgumentParser(description="Inspect label counts in a raw spike npz dataset.")
    parser.add_argument(
        "--npz",
        default="dataset/my_validation_subset_810000samples_27.00s.npz",
        help="Path to raw dataset npz.",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--text-out", default=None, help="Optional plain-text summary output path.")
    parser.add_argument("--plot-out", default=None, help="Optional plot output path.")
    args = parser.parse_args()

    pack = np.load(Path(args.npz), allow_pickle=False)
    labels = np.asarray(pack["spike_clusters"]).astype(np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]

    lines = []
    lines.append("=== Raw Dataset Summary ===")
    lines.append(f"total spikes: {int(labels.shape[0])}")
    lines.append(f"num units: {int(unique_labels.shape[0])}")
    lines.append(f"min count: {int(counts.min())}")
    lines.append(f"median count: {int(np.median(counts))}")
    lines.append(f"max count: {int(counts.max())}")
    summary = {
        "total_spikes": int(labels.shape[0]),
        "num_units": int(unique_labels.shape[0]),
        "min_count": int(counts.min()),
        "median_count": int(np.median(counts)),
        "max_count": int(counts.max()),
        "top_units": [],
        "subset_suggestions": [],
    }
    if "duration_sec" in pack:
        summary["duration_sec"] = float(np.asarray(pack["duration_sec"]).item())
        lines.append(f"duration_sec: {summary['duration_sec']}")
    if "fs" in pack:
        summary["sampling_rate_hz"] = float(np.asarray(pack["fs"]).item())
        lines.append(f"sampling_rate_hz: {summary['sampling_rate_hz']}")

    lines.append("")
    lines.append("=== Top 20 Units By Spike Count ===")
    for idx in order[:20]:
        unit_row = {"unit": int(unique_labels[idx]), "count": int(counts[idx])}
        summary["top_units"].append(unit_row)
        lines.append(f"unit={unit_row['unit']} count={unit_row['count']}")

    lines.append("")
    lines.append("=== Suggested Subset Starting Points ===")
    for topk in [10, 20, 50]:
        kept = counts[order[:topk]].sum() if unique_labels.size >= topk else counts.sum()
        row = {
            "mode": "topk",
            "value": int(topk),
            "kept_spikes": int(kept),
            "kept_units": int(min(topk, unique_labels.size)),
        }
        summary["subset_suggestions"].append(row)
        lines.append(f"topk={topk:<3d} keeps {int(kept):>6d} spikes across {min(topk, unique_labels.size):>3d} units")
    for min_count in [50, 100, 200, 500]:
        keep_units = int(np.sum(counts >= min_count))
        keep_spikes = int(np.sum(counts[counts >= min_count]))
        row = {
            "mode": "min_count",
            "value": int(min_count),
            "kept_spikes": keep_spikes,
            "kept_units": keep_units,
        }
        summary["subset_suggestions"].append(row)
        lines.append(f"min_count>={min_count:<3d} keeps {keep_spikes:>6d} spikes across {keep_units:>3d} units")

    output_text = "\n".join(lines)
    print(output_text)

    if args.text_out:
        Path(args.text_out).write_text(output_text + "\n", encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.plot_out:
        topn = min(30, unique_labels.size)
        top_counts = counts[order[:topn]]
        top_labels = unique_labels[order[:topn]]
        plt.figure(figsize=(10, 4.5))
        plt.bar([str(int(label)) for label in top_labels], top_counts, color="#4e79a7")
        plt.title("Top Unit Spike Counts")
        plt.xlabel("Unit ID")
        plt.ylabel("Spike Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(args.plot_out, dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
