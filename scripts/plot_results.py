"""Plot saved online curves from one experiment result directory.

中文说明
--------
这个脚本负责把实验目录下各个 variant 的在线曲线画出来。

相比最初的简版，这里额外补了：

- per-window accuracy
- per-window reject rate
- cumulative wrong updates

这样你后面做论文图时，能更直接观察“在线过程中性能有没有漂移、更新有没有失控”。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_variant_curves(result_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load ``curves.npz`` for every variant under ``results/<experiment>/runs``."""

    runs_dir = result_root / "runs"
    curves = {}
    if not runs_dir.exists():
        return curves
    for run_dir in sorted(runs_dir.iterdir()):
        curve_path = run_dir / "curves.npz"
        if not curve_path.exists():
            continue
        pack = np.load(curve_path)
        curves[run_dir.name] = {key: np.asarray(pack[key]) for key in pack.files}
    return curves


def main() -> None:
    """Generate a few standard plots for one experiment directory.

    中文：为一个结果目录生成统一的在线过程图。
    """

    parser = argparse.ArgumentParser(description="Plot online CAM curves from a result directory.")
    parser.add_argument("--result_dir", required=True, help="Path like results/<experiment_name>.")
    args = parser.parse_args()

    result_root = Path(args.result_dir)
    curves_by_variant = _load_variant_curves(result_root)
    if not curves_by_variant:
        raise FileNotFoundError(f"No curves.npz found under {result_root / 'runs'}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=False)
    axes = axes.ravel()

    for variant_name, curves in curves_by_variant.items():
        axes[0].plot(curves["step_index"], curves["cumulative_accuracy"], label=variant_name)
        axes[1].plot(curves["window_start_index"], curves["per_window_accuracy"], label=variant_name)
        axes[2].plot(curves["window_start_index"], curves["per_window_reject_rate"], label=variant_name)
        axes[3].plot(curves["step_index"], curves["template_count"], label=variant_name)
        axes[4].plot(curves["step_index"], curves["cumulative_updates"], label=variant_name)
        axes[5].plot(curves["step_index"], curves["cumulative_wrong_updates"], label=variant_name)

    axes[0].set_title("Cumulative Accuracy")
    axes[1].set_title("Per-Window Accuracy")
    axes[2].set_title("Per-Window Reject Rate")
    axes[3].set_title("Template Count Over Time")
    axes[4].set_title("Cumulative Update Count")
    axes[5].set_title("Cumulative Wrong Update Count")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel("Online step")

    fig.tight_layout()
    out_path = result_root / "summary_plots.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
