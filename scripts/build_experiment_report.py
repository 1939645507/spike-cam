"""Build a readable markdown report for one saved experiment directory.

中文说明
--------
这个脚本是给“已经跑完实验，想快速整理成毕设可读材料”的场景准备的。

它会读取：

- ``results/<experiment>/config.json``
- ``results/<experiment>/summary.json``
- ``results/<experiment>/runs/<variant>/*``

然后自动生成：

- ``report.md``：中文报告，包含实验目的、共享参数、指标表和初步结论
- ``metrics_table.csv``：便于后续贴进表格或继续分析
- ``comparison_metrics.png``：不同 variant 的关键指标对比图
- ``curve_overview.png``：不同 variant 的在线过程对比图

这个脚本不会替代你的论文分析，但它能把“原始结果目录”整理成更像实验记录的结构。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


KEY_METRICS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "reject_rate",
    "false_accept_rate",
    "accept_rate",
    "accepted_accuracy",
    "unknown_stream_fraction",
    "update_count",
    "wrong_update_rate",
    "final_template_count",
    "template_growth",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _variant_rows(result_root: Path) -> List[Dict[str, Any]]:
    """Collect one flat row per variant.

    中文：把每个 variant 的 metrics / meta / encoded stats 摊平成一行，便于做表。
    """

    rows: List[Dict[str, Any]] = []
    runs_dir = result_root / "runs"
    if not runs_dir.exists():
        return rows

    for run_dir in sorted(runs_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_json(metrics_path)
        meta = _load_json(run_dir / "meta.json") if (run_dir / "meta.json").exists() else {}
        encoded_stats = _load_json(run_dir / "encoded_stats.json") if (run_dir / "encoded_stats.json").exists() else {}

        row: Dict[str, Any] = {
            "variant_name": run_dir.name,
            "variant_description": meta.get("variant_description", ""),
            "encoder_backend": meta.get("encoded_dataset_meta", {}).get("encoder_backend", ""),
            "encoder_impl": meta.get("encoded_dataset_meta", {}).get("encoder_impl", ""),
            "mean_hamming_gap": encoded_stats.get("mean_hamming_gap"),
            "mean_intra_hamming": encoded_stats.get("mean_intra_hamming"),
            "mean_inter_hamming": encoded_stats.get("mean_inter_hamming"),
            "unique_code_ratio": encoded_stats.get("unique_code_ratio"),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Save a flat CSV table.

    中文：导出一个最容易后处理的表格文件。
    """

    if not rows:
        return
    fieldnames = [
        "variant_name",
        "variant_description",
        *KEY_METRICS,
        "mean_hamming_gap",
        "mean_intra_hamming",
        "mean_inter_hamming",
        "unique_code_ratio",
        "encoder_backend",
        "encoder_impl",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_metric_bars(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Create a comparison figure from scalar metrics.

    中文：把多个 variant 的关键最终指标画成对比图。
    """

    if not rows:
        return

    labels = [str(row["variant_name"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("reject_rate", "Reject Rate"),
        ("wrong_update_rate", "Wrong Update Rate"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_specs):
        values = [float(row.get(metric_key, 0.0) or 0.0) for row in rows]
        ax.bar(labels, values, color="#4e79a7")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _load_curves(result_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load ``curves.npz`` for all variants.

    中文：给报告脚本读取在线曲线。
    """

    curves: Dict[str, Dict[str, np.ndarray]] = {}
    runs_dir = result_root / "runs"
    if not runs_dir.exists():
        return curves
    for run_dir in sorted(runs_dir.iterdir()):
        curve_path = run_dir / "curves.npz"
        if not curve_path.exists():
            continue
        packed = np.load(curve_path)
        curves[run_dir.name] = {key: np.asarray(packed[key]) for key in packed.files}
    return curves


def _plot_curve_overview(result_root: Path, out_path: Path) -> None:
    """Create a compact curve overview figure.

    中文：输出一个更适合报告页插图的在线曲线总览图。
    """

    curves = _load_curves(result_root)
    if not curves:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for variant_name, bundle in curves.items():
        axes[0].plot(bundle["step_index"], bundle["cumulative_accuracy"], label=variant_name)
        axes[1].plot(bundle["window_start_index"], bundle["per_window_accuracy"], label=variant_name)
        axes[2].plot(bundle["step_index"], bundle["template_count"], label=variant_name)
        axes[3].plot(bundle["step_index"], bundle["cumulative_updates"], label=variant_name)

    axes[0].set_title("Cumulative Accuracy")
    axes[1].set_title("Per-Window Accuracy")
    axes[2].set_title("Template Count")
    axes[3].set_title("Cumulative Updates")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel("Online step")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _subset_description(config: Dict[str, Any]) -> str:
    subset = config.get("dataset", {}).get("subset", {})
    mode = subset.get("mode", "all")
    if mode == "topk":
        return f"topk={subset.get('topk')}"
    if mode == "min_count":
        return f"min_count>={subset.get('min_count')}"
    return mode


def _memory_subset_description(config: Dict[str, Any]) -> str:
    """Describe which labels are allowed into CAM memory.

    中文：
    dataset.subset 是“进入编码/测试流的数据范围”，
    cam.memory_subset 是“真正写进 CAM 模板的类集合”。
    """

    memory_subset = config.get("cam", {}).get("memory_subset", {})
    mode = memory_subset.get("mode", "same_as_stream")
    source = memory_subset.get("selection_source", "pre_sampling")
    if mode == "same_as_stream":
        return "same_as_stream"
    if mode == "topk":
        return f"topk={memory_subset.get('topk')} (source={source})"
    if mode == "min_count":
        return f"min_count>={memory_subset.get('min_count')} (source={source})"
    return f"{mode} (source={source})"


def _has_variant_specific_subset(rows: List[Dict[str, Any]]) -> bool:
    """Heuristically detect subset-sweep style experiments.

    中文：如果 variant 名字明显带 top10/top20/bits 之外的 subset 标记，
    就在报告里提醒“共享 subset 只是 base 配置，实际由 variant 覆盖”。
    """

    subset_like = 0
    for row in rows:
        name = str(row.get("variant_name", ""))
        if name.startswith("top") or "mincount" in name.lower():
            subset_like += 1
    return subset_like >= 2


def _extract_first_int(text: str) -> int | None:
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def _build_findings(rows: List[Dict[str, Any]]) -> List[str]:
    """Generate a few automatic observations.

    中文：给出“初步结论”，方便你后面继续人工补充。
    """

    findings: List[str] = []
    if not rows:
        return ["当前结果目录下还没有可读的 variant metrics。"]

    best_acc = max(rows, key=lambda row: float(row.get("accuracy", 0.0) or 0.0))
    best_f1 = max(rows, key=lambda row: float(row.get("macro_f1", 0.0) or 0.0))
    findings.append(
        f"按最终 accuracy 看，当前最好的是 `{best_acc['variant_name']}`，accuracy={float(best_acc.get('accuracy', 0.0)):.4f}。"
    )
    findings.append(
        f"按 macro-F1 看，当前最好的是 `{best_f1['variant_name']}`，macro-F1={float(best_f1.get('macro_f1', 0.0)):.4f}。"
    )

    hamming_gaps = [float(row["mean_hamming_gap"]) for row in rows if row.get("mean_hamming_gap") is not None]
    if hamming_gaps:
        mean_gap = float(np.mean(hamming_gaps))
        if mean_gap < 0.2:
            findings.append(
                f"平均 mean_hamming_gap 只有 {mean_gap:.4f}，说明当前 bits 的 intra/inter class separability 很弱，前端编码质量很可能已经成为主要瓶颈。"
            )
        elif mean_gap < 1.0:
            findings.append(
                f"平均 mean_hamming_gap 约为 {mean_gap:.4f}，说明 encoder 产生了一定区分度，但类间间隔仍然偏小，CAM 端提升空间可能有限。"
            )
        else:
            findings.append(
                f"平均 mean_hamming_gap 约为 {mean_gap:.4f}，说明 bits 已经具备一定可分性，值得继续做 CAM 端策略比较。"
            )

    subset_rows = [row for row in rows if row["variant_name"].startswith("top")]
    if len(subset_rows) >= 2:
        subset_rows.sort(key=lambda row: _extract_first_int(str(row["variant_name"])) or 0)
        description = ", ".join(
            f"{row['variant_name']} acc={float(row.get('accuracy', 0.0)):.4f}, reject={float(row.get('reject_rate', 0.0)):.4f}"
            for row in subset_rows
        )
        findings.append(f"subset 对比结果为：{description}。这能帮助你决定 CAM 里先保留多少类更合适。")

    static_row = next((row for row in rows if row["variant_name"] == "static"), None)
    if static_row is not None:
        dynamic_rows = [row for row in rows if row["variant_name"] != "static"]
        if dynamic_rows:
            best_dynamic = max(dynamic_rows, key=lambda row: float(row.get("accuracy", 0.0) or 0.0))
            delta = float(best_dynamic.get("accuracy", 0.0) or 0.0) - float(static_row.get("accuracy", 0.0) or 0.0)
            if delta > 1e-4:
                findings.append(
                    f"相对 static，对应表现最好的动态策略是 `{best_dynamic['variant_name']}`，accuracy 提升了 {delta:.4f}。"
                )
            else:
                findings.append(
                    f"当前这组结果里，动态策略并没有明显超过 static；这通常意味着 bits 质量不足，或者 update policy 还需要调 threshold / alpha。"
                )

    bit_rows = [row for row in rows if row["variant_name"].startswith("bits")]
    if bit_rows:
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for row in bit_rows:
            bit_width = _extract_first_int(str(row["variant_name"]))
            if bit_width is None:
                continue
            grouped.setdefault(bit_width, []).append(row)
        best_by_bits = []
        for bit_width in sorted(grouped):
            best_row = max(grouped[bit_width], key=lambda row: float(row.get("accuracy", 0.0) or 0.0))
            best_by_bits.append(f"{bit_width}-bit best={best_row['variant_name']} ({float(best_row.get('accuracy', 0.0)):.4f})")
        if best_by_bits:
            findings.append("bit budget 对比结果：" + "；".join(best_by_bits) + "。")

    risky_rows = [row for row in rows if float(row.get("wrong_update_rate", 0.0) or 0.0) > 0.2 and int(row.get("update_count", 0) or 0) > 0]
    if risky_rows:
        risky_names = ", ".join(str(row["variant_name"]) for row in risky_rows)
        findings.append(f"以下动态策略的 wrong_update_rate 偏高，需要重点检查是否出现模板污染：{risky_names}。")

    return findings


def _markdown_table(rows: List[Dict[str, Any]]) -> List[str]:
    headers = ["Variant", "Acc", "Macro-F1", "BalAcc", "Reject", "FalseAccept", "AcceptedAcc", "UnknownFrac", "Updates", "WrongUpdRate", "Gap"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant_name"]),
                    _format_float(row.get("accuracy")),
                    _format_float(row.get("macro_f1")),
                    _format_float(row.get("balanced_accuracy")),
                    _format_float(row.get("reject_rate")),
                    _format_float(row.get("false_accept_rate")),
                    _format_float(row.get("accepted_accuracy")),
                    _format_float(row.get("unknown_stream_fraction")),
                    _format_float(row.get("update_count"), digits=0),
                    _format_float(row.get("wrong_update_rate")),
                    _format_float(row.get("mean_hamming_gap")),
                ]
            )
            + " |"
        )
    return lines


def _write_report(result_root: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a markdown experiment report.

    中文：输出一个适合你直接阅读和后续整理的报告文件。
    """

    config = _load_json(result_root / "config.json") if (result_root / "config.json").exists() else {}
    description = config.get("description", "")
    experiment_name = config.get("experiment_name", result_root.name)

    shared_encoder = config.get("encoder", {})
    shared_dataset = config.get("dataset", {})
    shared_eval = config.get("evaluation", {})
    shared_template = config.get("template_init", {})
    shared_matcher = config.get("matcher", {})
    shared_update = config.get("update", {})
    shared_cam = config.get("cam", {})

    report_lines: List[str] = [
        f"# {experiment_name} 实验报告",
        "",
        "## 1. 实验目的",
        "",
        description or "本次实验用于比较同一 pipeline 下不同配置变体的表现。",
        "",
        "## 2. 共享实验参数",
        "",
        f"- 数据文件：`{shared_dataset.get('npz_path', '-')}`",
        f"- stream subset：`{_subset_description(config)}`" + ("（注意：该实验的不同 variant 会覆盖 subset）" if _has_variant_specific_subset(rows) else ""),
        f"- CAM memory subset：`{_memory_subset_description(config)}`",
        f"- sampling：`max_spikes_per_unit={shared_dataset.get('sampling', {}).get('max_spikes_per_unit')}`，`selection_mode={shared_dataset.get('sampling', {}).get('selection_mode')}`",
        f"- waveform：`length={shared_dataset.get('waveform', {}).get('waveform_length')}`，`center={shared_dataset.get('waveform', {}).get('center_index')}`，`align_mode={shared_dataset.get('waveform', {}).get('align_mode')}`",
        f"- encoder：`method={shared_encoder.get('method')}`，`backend={shared_encoder.get('backend')}`，`code_size={shared_encoder.get('code_size')}`，`epochs={shared_encoder.get('epochs')}`",
        f"- template init：`{shared_template.get('method')}`",
        f"- matcher：`{shared_matcher.get('method')}`，`threshold={shared_matcher.get('threshold')}`",
        f"- base update：`{shared_update.get('method')}`",
        f"- evaluation：`mode={shared_eval.get('mode')}`，`warmup_ratio={shared_eval.get('warmup_ratio')}`，`window_size={shared_eval.get('window_size')}`",
        f"- CAM：`capacity={shared_cam.get('capacity')}`，`capacity_factor={shared_cam.get('capacity_factor')}`，`extra_rows={shared_cam.get('extra_rows')}`",
        "",
        "## 3. 指标汇总",
        "",
        *_markdown_table(rows),
        "",
        "## 4. 初步结论",
        "",
    ]
    for finding in _build_findings(rows):
        report_lines.append(f"- {finding}")

    report_lines.extend(
        [
            "",
            "## 5. 结果文件",
            "",
            "- `summary.json`：各 variant 的最终核心指标汇总。",
            "- `metrics_table.csv`：平铺表格，便于继续分析。",
            "- `comparison_metrics.png`：最终指标对比图。",
            "- `curve_overview.png`：在线过程曲线总览。",
            "- `runs/<variant>/metrics.json`：单个 variant 的详细指标。",
            "- `runs/<variant>/curves.npz`：在线过程数据，可继续画图。",
            "- `runs/<variant>/encoded_stats.json`：bits 的基本统计与 separability 诊断。",
            "",
            "## 6. 如何解读这份报告",
            "",
            "- 如果 `mean_hamming_gap` 很小，优先怀疑 encoder bits 可分性不足，而不是先否定 CAM。",
            "- 如果 `accepted_accuracy` 高但 `accuracy` 低，通常说明 CAM 只在少量高置信样本上表现不错，但 reject 太多。",
            "- 如果 `wrong_update_rate` 高，说明动态更新可能在污染模板，需要调保守度或 margin 规则。",
            "- 如果 `template_growth` 明显增加，需要结合 `accuracy` 看是否换来了有效收益，而不是单纯占更多行。",
        ]
    )

    (result_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    """Entry point.

    中文：从一个结果目录自动生成报告和配套图表。
    """

    parser = argparse.ArgumentParser(description="Build a markdown report for one experiment result directory.")
    parser.add_argument("--result_dir", required=True, help="Path like results/<experiment_name>.")
    parser.add_argument(
        "--sort_by",
        default="accuracy",
        help="Metric used to sort variants in the report table. Default: accuracy.",
    )
    args = parser.parse_args()

    result_root = Path(args.result_dir)
    rows = _variant_rows(result_root)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {result_root / 'runs'}")

    sort_key = args.sort_by
    rows.sort(key=lambda row: float(row.get(sort_key, 0.0) or 0.0), reverse=True)

    _write_csv(rows, result_root / "metrics_table.csv")
    _plot_metric_bars(rows, result_root / "comparison_metrics.png")
    _plot_curve_overview(result_root, result_root / "curve_overview.png")
    _write_report(result_root, rows)

    print(f"saved report artifacts under {result_root}")


if __name__ == "__main__":
    main()
