"""Build a master study report from multiple experiment result folders.

中文说明
--------
这个脚本用于把“一整轮研究”整理成一个教授可直接阅读的总报告。

它会读取：

- study manifest
- 多个 `results/<study_root>/<experiment_name>/summary.json`
- 各实验自己的 `report.md` / `comparison_metrics.png` / `summary_plots.png`

然后在 study root 下生成：

- `README.md`
- `study_overview.csv`
- `figures/*.png`

这样最终交付给老师的，不再是分散的多个结果目录，
而是一个带总览说明、实验矩阵、关键表格、可视化和总结结论的研究包。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import resolve_path


TABLE_KEYS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "reject_rate",
    "false_accept_rate",
    "false_reject_rate",
    "accepted_accuracy",
    "unknown_stream_fraction",
    "update_count",
    "wrong_update_rate",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion for metrics pulled from JSON summaries."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _variant_rows(experiment_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary_path = experiment_root / "summary.json"
    if not summary_path.exists():
        return rows
    summary = _load_json(summary_path)
    for variant_name, metrics in summary.items():
        row = {"variant_name": variant_name, **metrics}
        encoded_stats_path = experiment_root / "runs" / variant_name / "encoded_stats.json"
        if encoded_stats_path.exists():
            encoded_stats = _load_json(encoded_stats_path)
            row["mean_hamming_gap"] = encoded_stats.get("mean_hamming_gap")
        rows.append(row)
    return rows


def _write_overview_csv(study_root: Path, experiment_entries: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for entry in experiment_entries:
        for row in entry["rows"]:
            rows.append(
                {
                    "experiment_name": entry["name"],
                    "experiment_title": entry["title"],
                    "variant_name": row["variant_name"],
                    **{key: row.get(key) for key in TABLE_KEYS},
                    "mean_hamming_gap": row.get("mean_hamming_gap"),
                }
            )

    fieldnames = ["experiment_name", "experiment_title", "variant_name", *TABLE_KEYS, "mean_hamming_gap"]
    out_path = study_root / "study_overview.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_number(name: str) -> int | None:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_memory_sweep(entry: Dict[str, Any], out_path: Path) -> None:
    rows = sorted(entry["rows"], key=lambda row: _parse_number(str(row["variant_name"])) or 0)
    memory_sizes = [_parse_number(str(row["variant_name"])) or 0 for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    specs = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("reject_rate", "Reject Rate"),
        ("false_accept_rate", "False Accept Rate"),
    ]
    for ax, (metric_key, title) in zip(axes, specs):
        values = [float(row.get(metric_key, 0.0) or 0.0) for row in rows]
        ax.plot(memory_sizes, values, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Memory Top-K")
        ax.grid(True, alpha=0.3)
    _save_figure(fig, out_path)


def _plot_update_compare(entry: Dict[str, Any], out_path: Path) -> None:
    rows = sorted(entry["rows"], key=lambda row: float(row.get("accuracy", 0.0) or 0.0), reverse=True)
    labels = [str(row["variant_name"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    specs = [
        ("accuracy", "Accuracy"),
        ("reject_rate", "Reject Rate"),
        ("false_accept_rate", "False Accept Rate"),
        ("wrong_update_rate", "Wrong Update Rate"),
    ]
    for ax, (metric_key, title) in zip(axes, specs):
        values = [float(row.get(metric_key, 0.0) or 0.0) for row in rows]
        ax.bar(labels, values, color="#4e79a7")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    _save_figure(fig, out_path)


def _plot_bits_ablation(entry: Dict[str, Any], out_path: Path) -> None:
    rows = entry["rows"]
    strategies = ["static", "counter", "margin_ema"]
    bit_sizes = [8, 16, 32]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("reject_rate", "Reject Rate"),
        ("false_accept_rate", "False Accept Rate"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_specs):
        for strategy in strategies:
            values = []
            for bit_size in bit_sizes:
                row = next(
                    (r for r in rows if str(r["variant_name"]).startswith(f"bits{bit_size}_") and str(r["variant_name"]).endswith(strategy)),
                    None,
                )
                values.append(float(row.get(metric_key, 0.0) or 0.0) if row is not None else np.nan)
            ax.plot(bit_sizes, values, marker="o", label=strategy)
        ax.set_title(title)
        ax.set_xlabel("Bit Width")
        ax.grid(True, alpha=0.3)
        ax.legend()

    _save_figure(fig, out_path)


def _plot_threshold_sweep(entry: Dict[str, Any], out_path: Path) -> None:
    rows = sorted(entry["rows"], key=lambda row: _parse_number(str(row["variant_name"])) or 0)
    thresholds = [_parse_number(str(row["variant_name"])) or 0 for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("reject_rate", "Reject Rate"),
        ("false_accept_rate", "False Accept Rate"),
    ]
    for ax, (metric_key, title) in zip(axes, metric_specs):
        values = [float(row.get(metric_key, 0.0) or 0.0) for row in rows]
        ax.plot(thresholds, values, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Hamming Threshold")
        ax.grid(True, alpha=0.3)
    _save_figure(fig, out_path)


def _plot_generic_bars(entry: Dict[str, Any], out_path: Path) -> None:
    rows = sorted(entry["rows"], key=lambda row: float(row.get("accuracy", 0.0) or 0.0), reverse=True)
    labels = [str(row["variant_name"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [
        ("accuracy", "Accuracy"),
        ("reject_rate", "Reject Rate"),
        ("false_accept_rate", "False Accept Rate"),
    ]
    for ax, (metric_key, title) in zip(axes, specs):
        values = [float(row.get(metric_key, 0.0) or 0.0) for row in rows]
        ax.bar(labels, values, color="#4e79a7")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    _save_figure(fig, out_path)


def _make_markdown_table(rows: List[Dict[str, Any]]) -> List[str]:
    headers = ["Variant", "Acc", "Macro-F1", "Reject", "FalseAccept", "AcceptedAcc", "UnknownFrac", "Updates", "WrongUpdRate", "Gap"]
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
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("macro_f1")),
                    _fmt(row.get("reject_rate")),
                    _fmt(row.get("false_accept_rate")),
                    _fmt(row.get("accepted_accuracy")),
                    _fmt(row.get("unknown_stream_fraction")),
                    _fmt(row.get("update_count"), digits=0),
                    _fmt(row.get("wrong_update_rate")),
                    _fmt(row.get("mean_hamming_gap")),
                ]
            )
            + " |"
        )
    return lines


def _study_findings(entries: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []

    memory_entry = next((entry for entry in entries if entry["name"] == "memory_sweep_external_ae16"), None)
    if memory_entry and memory_entry["rows"]:
        best = max(memory_entry["rows"], key=lambda row: _safe_float(row.get("accuracy")))
        limited_rows = [row for row in memory_entry["rows"] if _safe_float(row.get("unknown_stream_fraction")) > 0.0]
        best_limited = max(limited_rows, key=lambda row: _safe_float(row.get("accuracy"))) if limited_rows else best
        findings.append(
            "Memory sweep 显示 memory 越大通常越容易提高准确率；"
            f"在仍然保留未知类压力的 memory-limited 设定里，`{best_limited['variant_name']}` 最好，"
            f"accuracy={_safe_float(best_limited.get('accuracy')):.4f}，"
            f"false_accept={_safe_float(best_limited.get('false_accept_rate')):.4f}。"
        )
        if _safe_float(best.get("unknown_stream_fraction")) == 0.0:
            findings.append(
                f"`{best['variant_name']}` 的 accuracy 也很高 "
                f"({_safe_float(best.get('accuracy')):.4f})，但此时 unknown stream fraction 已经是 0，"
                "它更接近全量记忆上限，不再是严格的 memory-limited 场景。"
            )

    bits_entry = next((entry for entry in entries if entry["name"] == "bits_ablation_memoryaware"), None)
    if bits_entry and bits_entry["rows"]:
        best_acc = max(bits_entry["rows"], key=lambda row: _safe_float(row.get("accuracy")))
        bits16_static = next((row for row in bits_entry["rows"] if row["variant_name"] == "bits16_static"), None)
        bits8_static = next((row for row in bits_entry["rows"] if row["variant_name"] == "bits8_static"), None)
        bits32_static = next((row for row in bits_entry["rows"] if row["variant_name"] == "bits32_static"), None)
        findings.append(
            f"bit budget 对比里 raw accuracy 最好的是 `{best_acc['variant_name']}` "
            f"(accuracy={_safe_float(best_acc.get('accuracy')):.4f})。"
        )
        if bits8_static and bits16_static and bits32_static:
            findings.append(
                "`8-bit` 更容易给出更高的 raw accuracy，"
                f"但它的 false_accept 也最高 ({_safe_float(bits8_static.get('false_accept_rate')):.4f})；"
                f"`16-bit` 的 accuracy 虽然较低 ({_safe_float(bits16_static.get('accuracy')):.4f})，"
                f"但 Hamming gap 更大 ({_safe_float(bits16_static.get('mean_hamming_gap')):.4f})，"
                "更像当前阶段的平衡选择；`32-bit` 则整体退化。"
            )

    update_entry = next((entry for entry in entries if entry["name"] == "update_compare_external_ae16"), None)
    if update_entry and update_entry["rows"]:
        best_acc = max(update_entry["rows"], key=lambda row: _safe_float(row.get("accuracy")))
        static = next((row for row in update_entry["rows"] if row["variant_name"] == "static"), None)
        if static is not None:
            delta = _safe_float(best_acc.get("accuracy")) - _safe_float(static.get("accuracy"))
            findings.append(
                f"在 dynamic update 对比里，`{best_acc['variant_name']}` 相对 static 的 accuracy 变化为 {delta:+.4f}，"
                f"但它的 false_accept={_safe_float(best_acc.get('false_accept_rate')):.4f}，"
                f"wrong_update_rate={_safe_float(best_acc.get('wrong_update_rate')):.4f}。"
            )
        risky = [
            row["variant_name"]
            for row in update_entry["rows"]
            if _safe_float(row.get("wrong_update_rate")) > 0.8 and int(row.get("update_count", 0) or 0) > 0
        ]
        if risky:
            findings.append(
                "大多数动态策略的 wrong update 比例都很高，说明它们当前主要是通过减少 reject 来换取更高 raw accuracy，"
                "而不是稳定地学到了更干净的模板。高风险策略包括：" + ", ".join(map(str, risky)) + "。"
            )

    threshold_entry = next((entry for entry in entries if entry["name"] == "threshold_sweep_external_ae16"), None)
    if threshold_entry and threshold_entry["rows"]:
        low_thr = next((row for row in threshold_entry["rows"] if row["variant_name"] == "thr2"), None)
        mid_thr = next((row for row in threshold_entry["rows"] if row["variant_name"] == "thr4"), None)
        high_thr = next((row for row in threshold_entry["rows"] if row["variant_name"] == "thr6"), None)
        if low_thr and mid_thr and high_thr:
            findings.append(
                "threshold 是当前最直接的 reject/false-accept 控制杆："
                f"`thr2` 很保守 (reject={_safe_float(low_thr.get('reject_rate')):.4f}, false_accept={_safe_float(low_thr.get('false_accept_rate')):.4f})，"
                f"`thr6` 几乎不拒绝 (reject={_safe_float(high_thr.get('reject_rate')):.4f}) 但会吞掉大量 unknown "
                f"(false_accept={_safe_float(high_thr.get('false_accept_rate')):.4f})；"
                f"`thr4` 是当前比较均衡的中间点。"
            )

    encoder_entry = next((entry for entry in entries if entry["name"] == "encoder_compare_memoryaware"), None)
    if encoder_entry and encoder_entry["rows"]:
        best = max(encoder_entry["rows"], key=lambda row: _safe_float(row.get("accuracy")))
        pca = next((row for row in encoder_entry["rows"] if row["variant_name"] == "pca_static"), None)
        findings.append(
            f"encoder 比较里表现较好的是 `{best['variant_name']}`，说明前端表示质量会显著影响 memory-aware CAM 的上限。"
        )
        if pca is not None:
            findings.append(
                f"PCA baseline 更保守，false_accept={_safe_float(pca.get('false_accept_rate')):.4f} 低于 AE，"
                f"但 accuracy 也更低 ({_safe_float(pca.get('accuracy')):.4f})；"
                "当前 external AE 仍然是更好的主实验前端。"
            )

    return findings


def _recommended_setting(entries: List[Dict[str, Any]]) -> List[str]:
    """Return a concise default operating point for the current dataset/protocol."""
    lines = [
        "- stream subset: `top50`",
        "- CAM memory subset: `top20 (selection_source=pre_sampling)`",
        "- encoder: `external AE`, `code_size=16`",
        "- template init: `medoid` or `majority_vote`",
        "- matcher: `hamming_nearest`",
        "- threshold: start from `4`",
        "- update strategy: start from `static / no-update`; only enable dynamic update after adding a stronger confidence gate",
    ]

    init_entry = next((entry for entry in entries if entry["name"] == "init_ablation_external_ae16"), None)
    if init_entry and init_entry["rows"]:
        medoid = next((row for row in init_entry["rows"] if row["variant_name"] == "medoid_static"), None)
        majority = next((row for row in init_entry["rows"] if row["variant_name"] == "majority_static"), None)
        if medoid and majority:
            lines.append(
                "- Why `medoid` first: it slightly improves accuracy "
                f"({_safe_float(medoid.get('accuracy')):.4f} vs {_safe_float(majority.get('accuracy')):.4f}) "
                "without introducing dynamic-update risk."
            )

    bits_entry = next((entry for entry in entries if entry["name"] == "bits_ablation_memoryaware"), None)
    if bits_entry and bits_entry["rows"]:
        bits8 = next((row for row in bits_entry["rows"] if row["variant_name"] == "bits8_static"), None)
        bits16 = next((row for row in bits_entry["rows"] if row["variant_name"] == "bits16_static"), None)
        if bits8 and bits16:
            lines.append(
                "- `16-bit` is recommended as the first thesis setting because it is more balanced; "
                f"`8-bit` is more aggressive (false_accept={_safe_float(bits8.get('false_accept_rate')):.4f}) "
                f"even though its raw accuracy is slightly higher ({_safe_float(bits8.get('accuracy')):.4f})."
            )

    return lines


def build_study_report(manifest_path: Path) -> Path:
    manifest = _load_json(manifest_path)
    study_root = resolve_path(str(manifest["results_root"]))
    study_root.mkdir(parents=True, exist_ok=True)
    figures_dir = study_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    for exp in manifest["experiments"]:
        exp_root = study_root / exp["name"]
        rows = _variant_rows(exp_root)
        entries.append(
            {
                "name": exp["name"],
                "title": exp["title"],
                "question": exp["question"],
                "root": exp_root,
                "rows": rows,
                "config_path": manifest_path.parent / f"{exp['name']}.json",
            }
        )

    _write_overview_csv(study_root, entries)

    for entry in entries:
        if not entry["rows"]:
            continue
        if entry["name"] == "memory_sweep_external_ae16":
            _plot_memory_sweep(entry, figures_dir / "memory_sweep.png")
        elif entry["name"] == "update_compare_external_ae16":
            _plot_update_compare(entry, figures_dir / "update_compare.png")
        elif entry["name"] == "bits_ablation_memoryaware":
            _plot_bits_ablation(entry, figures_dir / "bits_ablation.png")
        elif entry["name"] == "threshold_sweep_external_ae16":
            _plot_threshold_sweep(entry, figures_dir / "threshold_sweep.png")
        else:
            _plot_generic_bars(entry, figures_dir / f"{entry['name']}.png")

    readme_lines: List[str] = [
        f"# {manifest['study_title']}",
        "",
        manifest.get("description", ""),
        "",
        "## Study Goal",
        "",
        "This study evaluates a memory-aware spike CAM protocol in which:",
        "",
        "- the encoded test stream can contain more classes than CAM can remember",
        "- CAM stores only a limited subset of active units as templates",
        "- memory-external units must ideally be rejected rather than misclassified",
        "",
        "## Protocol Summary",
        "",
        "- Dataset: `dataset/my_validation_subset_810000samples_27.00s.npz`",
        "- Default stream setting in the main study: `top50 units`, chronological stream evaluation",
        "- Default memory setting in the main study: `top20 units kept in CAM`, selected from pre-sampling label counts",
        "- Warmup/evaluation: `warmup_ratio=0.25`, then online match/reject/update in strict time order",
        "- Encoder front-end: external `Autoencoders-in-Spike-Sorting` AE unless an ablation explicitly changes it",
        "- Main research object: CAM-side memory/update behavior, not encoder training itself",
        "",
        "## Experiment Matrix",
        "",
        "| Experiment | Question | Result Dir | Config |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        readme_lines.append(
            f"| {entry['title']} | {entry['question']} | [`{entry['name']}`](./{entry['name']}) | [`{entry['config_path'].name}`](../../configs/studies/20260314/{entry['config_path'].name}) |"
        )

    readme_lines.extend(
        [
            "",
            "## Key Findings",
            "",
        ]
    )
    for finding in _study_findings(entries):
        readme_lines.append(f"- {finding}")

    readme_lines.extend(
        [
            "",
            "## Recommended Default Setting",
            "",
            "This is the operating point that currently looks most suitable for the thesis mainline:",
            "",
        ]
    )
    for line in _recommended_setting(entries):
        readme_lines.append(line)

    readme_lines.extend(
        [
            "",
            "## Figures",
            "",
            "### Cross-Experiment Figures",
            "",
        ]
    )

    figure_map = [
        ("figures/memory_sweep.png", "Memory sweep"),
        ("figures/update_compare.png", "Update comparison"),
        ("figures/bits_ablation.png", "Bit-budget ablation"),
        ("figures/threshold_sweep.png", "Threshold sweep"),
        ("figures/init_ablation_external_ae16.png", "Initialization ablation"),
        ("figures/encoder_compare_memoryaware.png", "Encoder comparison"),
    ]
    for rel_path, label in figure_map:
        if (study_root / rel_path).exists():
            readme_lines.append(f"#### {label}")
            readme_lines.append("")
            readme_lines.append(f"![{label}]({rel_path})")
            readme_lines.append("")

    readme_lines.extend(
        [
            "## Detailed Experiment Sections",
            "",
        ]
    )

    for entry in entries:
        rows = entry["rows"]
        readme_lines.extend(
            [
                f"### {entry['title']}",
                "",
                f"Question: {entry['question']}",
                "",
                f"Config: [`{entry['config_path'].name}`](../../configs/studies/20260314/{entry['config_path'].name})",
                "",
                f"Result dir: [`{entry['name']}`](./{entry['name']})",
                "",
            ]
        )
        if rows:
            rows_sorted = sorted(rows, key=lambda row: float(row.get("accuracy", 0.0) or 0.0), reverse=True)
            readme_lines.extend(_make_markdown_table(rows_sorted))
            readme_lines.extend(
                [
                    "",
                    f"Per-experiment report: [`report.md`](./{entry['name']}/report.md)",
                    "",
                    f"Summary metrics plot: ![{entry['name']} comparison](./{entry['name']}/comparison_metrics.png)",
                    "",
                    f"Online curves plot: ![{entry['name']} curves](./{entry['name']}/summary_plots.png)",
                    "",
                ]
            )
        else:
            readme_lines.extend(["No result data found for this experiment yet.", ""])

    readme_lines.extend(
        [
            "## Files",
            "",
            "- `study_overview.csv`: flat table of all variants from all experiments.",
            "- `figures/`: cross-experiment summary figures.",
            "- `<experiment_name>/`: individual experiment result folders produced by `run_experiment.py`.",
            "",
            "## How To Reproduce",
            "",
            "1. Run each config under `configs/studies/20260314/`.",
            "2. Run `python scripts/build_study_report.py --manifest configs/studies/20260314/study_manifest.json`.",
            "",
        ]
    )

    readme_path = study_root / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return readme_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a master study report from a study manifest.")
    parser.add_argument("--manifest", required=True, help="Path to a study manifest JSON file.")
    args = parser.parse_args()

    readme_path = build_study_report(resolve_path(args.manifest))
    print(f"saved study report to {readme_path}")


if __name__ == "__main__":
    main()
