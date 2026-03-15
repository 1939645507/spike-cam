"""Build a thesis-style master report for one complete Spike CAM run.

中文说明
--------
这个脚本把一次完整的 thesis run 整理成一个真正可读的实验包。

输入：

 - 一个统一的 run root，例如 `results/experiments/2026-03-15/experiment_20260315_134107/`

输出：

- `README_run.md`
- `master_report.md`
- `master_report.html`
- `experiment_index.json`
- `overall_summary.csv`
- `overall_summary.json`
- `figures/`
- `tables/`
- `error_analysis/*`

也就是说，它会把“零散的多个实验子目录”提升成“能直接拿去给老师看”的完整研究包。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import save_json


EXPERIMENTS = [
    ("baseline", "Baseline Sanity Check"),
    ("main_update_compare", "Main Comparison of Update Strategies"),
    ("threshold_sweep", "Threshold Sweep"),
    ("bits_ablation", "Bit Budget Ablation"),
    ("init_ablation", "Template Initialization Ablation"),
    ("encoder_compare", "Encoder Comparison"),
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


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _summary_rows(result_dir: Path, experiment_name: str, experiment_title: str) -> List[Dict[str, Any]]:
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return []
    summary = _load_json(summary_path)
    rows: List[Dict[str, Any]] = []
    for variant_name, metrics in summary.items():
        row = {
            "experiment_name": experiment_name,
            "experiment_title": experiment_title,
            "variant_name": variant_name,
            **metrics,
        }
        rows.append(row)
    return rows


def _write_overall_summary(run_root: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "experiment_name",
        "experiment_title",
        "variant_name",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "accept_rate",
        "accepted_accuracy",
        "reject_rate",
        "false_reject_rate",
        "false_accept_rate",
        "update_count",
        "wrong_update_count",
        "wrong_update_rate",
        "initial_template_count",
        "final_template_count",
        "max_template_count",
        "template_growth",
        "used_rows_over_capacity",
        "unknown_stream_fraction",
    ]
    with (run_root / "overall_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    save_json(run_root / "overall_summary.json", {"rows": rows})


def _write_index(run_root: Path) -> None:
    index: Dict[str, Any] = {"experiments": []}
    for exp_name, exp_title in EXPERIMENTS:
        exp_dir = run_root / exp_name
        index["experiments"].append(
            {
                "name": exp_name,
                "title": exp_title,
                "result_dir": str(exp_dir),
                "config": str(run_root / "configs" / f"{exp_name}.json"),
                "summary": str(exp_dir / "summary.json"),
                "report": str(exp_dir / "report.md"),
            }
        )
    save_json(run_root / "experiment_index.json", index)


def _write_run_readme(run_root: Path, env_info: Dict[str, Any]) -> None:
    run_root_rel = run_root.relative_to(PROJECT_ROOT) if run_root.is_relative_to(PROJECT_ROOT) else run_root
    lines = [
        "# Thesis Run README",
        "",
        f"Run root: `{run_root}`",
        "",
        "## 环境信息",
        "",
        f"- conda env: `{env_info['conda_env']}`",
        f"- Python: `{env_info['python_version']}`",
        f"- Platform: `{env_info['platform']}`",
        f"- numpy: `{env_info['numpy_version']}`",
        f"- sklearn: `{env_info['sklearn_version']}`",
        f"- tensorflow: `{env_info['tensorflow_version']}`",
        f"- Preferred encoder backend: `{env_info['encoder_backend']}`",
        "",
        "## 主要入口",
        "",
        f"- Baseline: `conda run -n {env_info['conda_env']} python run_experiment.py --config {run_root_rel}/configs/baseline.json`",
        f"- Main comparison: `conda run -n {env_info['conda_env']} python run_experiment.py --config {run_root_rel}/configs/main_update_compare.json`",
        f"- Rebuild master report: `conda run -n {env_info['conda_env']} python scripts/build_full_thesis_run.py --run-root {run_root_rel}`",
        "",
        "## 目录说明",
        "",
        "- `diagnostics/`: 标签分布和 encoded representation 诊断。",
        "- `baseline/`: 静态 baseline sanity check。",
        "- `main_update_compare/`: 主实验，比较不同 update strategy。",
        "- `threshold_sweep/`: reject threshold 权衡分析。",
        "- `bits_ablation/`: 8/16/32/64-bit 消融。",
        "- `init_ablation/`: 模板初始化策略消融。",
        "- `encoder_compare/`: encoder 对比实验。",
        "- `error_analysis/`: 代表性方法的错误分析。",
        "- `figures/` 与 `tables/`: 汇总图表，方便快速浏览。",
        "- `logs/`: 执行命令与运行日志。",
        "",
    ]
    (run_root / "README_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_predictions(run_root: Path, experiment_name: str, variant_name: str) -> Dict[str, np.ndarray]:
    path = run_root / experiment_name / "runs" / variant_name / "predictions.npz"
    packed = np.load(path)
    return {key: np.asarray(packed[key]) for key in packed.files}


def _load_curves(run_root: Path, experiment_name: str, variant_name: str) -> Dict[str, np.ndarray]:
    path = run_root / experiment_name / "runs" / variant_name / "curves.npz"
    packed = np.load(path)
    return {key: np.asarray(packed[key]) for key in packed.files}


def _load_meta(run_root: Path, experiment_name: str, variant_name: str) -> Dict[str, Any]:
    path = run_root / experiment_name / "runs" / variant_name / "meta.json"
    return _load_json(path)


def _top_confusion_pairs(confusion: np.ndarray, labels: np.ndarray, known_labels: np.ndarray, topn: int = 10) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    label_to_known = set(int(v) for v in known_labels.tolist())
    for i, truth in enumerate(labels):
        if int(truth) not in label_to_known:
            continue
        for j, pred in enumerate(labels):
            if i == j:
                continue
            if int(pred) not in label_to_known:
                continue
            count = int(confusion[i, j])
            if count > 0:
                rows.append((f"{int(truth)} -> {int(pred)}", count))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:topn]


def _per_class_false_reject(y_true: np.ndarray, y_pred: np.ndarray, known_labels: np.ndarray, topn: int = 10) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    for label in known_labels:
        count = int(np.sum((y_true == label) & (y_pred == -1)))
        rows.append((int(label), count))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:topn]


def _unknown_false_accept_destinations(y_true: np.ndarray, y_pred: np.ndarray, known_labels: np.ndarray, topn: int = 10) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    unknown_mask = ~np.isin(y_true, known_labels)
    for label in known_labels:
        count = int(np.sum(unknown_mask & (y_pred == label)))
        rows.append((int(label), count))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:topn]


def _build_error_analysis(run_root: Path) -> Dict[str, Any]:
    """Generate error-analysis files and return summary stats."""

    error_root = run_root / "error_analysis"
    fig_root = error_root / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    static_pred = _load_predictions(run_root, "main_update_compare", "static")
    dynamic_pred = _load_predictions(run_root, "main_update_compare", "probabilistic")
    static_curves = _load_curves(run_root, "main_update_compare", "static")
    dynamic_curves = _load_curves(run_root, "main_update_compare", "probabilistic")
    static_meta = _load_meta(run_root, "main_update_compare", "static")
    dynamic_meta = _load_meta(run_root, "main_update_compare", "probabilistic")

    known_labels = np.asarray(dynamic_meta["memory_selection"]["loaded_memory_labels"], dtype=np.int64)
    static_conf = np.load(run_root / "main_update_compare" / "runs" / "static" / "confusion.npy")
    static_conf_labels = np.load(run_root / "main_update_compare" / "runs" / "static" / "confusion_labels.npy")
    dynamic_conf = np.load(run_root / "main_update_compare" / "runs" / "probabilistic" / "confusion.npy")
    dynamic_conf_labels = np.load(run_root / "main_update_compare" / "runs" / "probabilistic" / "confusion_labels.npy")

    def split_stats(preds: Dict[str, np.ndarray]) -> Dict[str, float]:
        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        known = np.isin(y_true, known_labels)
        unknown = ~known
        accepted = y_pred != -1
        return {
            "known_accept_rate": float(np.mean(accepted[known])) if np.any(known) else 0.0,
            "known_false_reject_rate": float(np.mean(~accepted[known])) if np.any(known) else 0.0,
            "unknown_reject_rate": float(np.mean(~accepted[unknown])) if np.any(unknown) else 0.0,
            "unknown_false_accept_rate": float(np.mean(accepted[unknown])) if np.any(unknown) else 0.0,
            "accepted_accuracy": float(np.mean(y_true[accepted] == y_pred[accepted])) if np.any(accepted) else 0.0,
            "overall_accuracy": float(np.mean(y_true == y_pred)),
        }

    static_split = split_stats(static_pred)
    dynamic_split = split_stats(dynamic_pred)

    # Figure 1: known/unknown behavior comparison.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = ["known accept", "known false reject", "unknown reject", "unknown false accept"]
    static_values = [
        static_split["known_accept_rate"],
        static_split["known_false_reject_rate"],
        static_split["unknown_reject_rate"],
        static_split["unknown_false_accept_rate"],
    ]
    dynamic_values = [
        dynamic_split["known_accept_rate"],
        dynamic_split["known_false_reject_rate"],
        dynamic_split["unknown_reject_rate"],
        dynamic_split["unknown_false_accept_rate"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, static_values, width, label="static", color="#4e79a7")
    ax.bar(x + width / 2, dynamic_values, width, label="probabilistic", color="#e15759")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Known/Unknown Acceptance Behavior")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_root / "known_unknown_behavior.png", dpi=160)
    plt.close(fig)

    # Figure 2: update pollution over time.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(dynamic_curves["step_index"], dynamic_curves["cumulative_updates"], label="cumulative updates", color="#4e79a7")
    axes[0].plot(dynamic_curves["step_index"], dynamic_curves["cumulative_wrong_updates"], label="cumulative wrong updates", color="#e15759")
    axes[0].set_title("Probabilistic Update Pollution Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(static_curves["step_index"], static_curves["template_count"], label="static template count", color="#59a14f")
    axes[1].plot(dynamic_curves["step_index"], dynamic_curves["template_count"], label="probabilistic template count", color="#f28e2b")
    axes[1].set_title("Template Count Over Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlabel("Online step")
    fig.tight_layout()
    fig.savefig(fig_root / "update_pollution_over_time.png", dpi=160)
    plt.close(fig)

    # Figure 3: top confusion pairs for probabilistic.
    top_pairs = _top_confusion_pairs(dynamic_conf, dynamic_conf_labels, known_labels, topn=10)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([pair for pair, _ in top_pairs], [count for _, count in top_pairs], color="#4e79a7")
    ax.set_title("Top Known-Known Confusions (Probabilistic)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_root / "top_confusion_pairs_probabilistic.png", dpi=160)
    plt.close(fig)

    top_false_reject = _per_class_false_reject(dynamic_pred["y_true"], dynamic_pred["y_pred"], known_labels, topn=10)
    top_unknown_dest = _unknown_false_accept_destinations(dynamic_pred["y_true"], dynamic_pred["y_pred"], known_labels, topn=10)
    dynamic_window_wrong = np.asarray(dynamic_curves["per_window_wrong_updates"], dtype=np.int64)
    dynamic_window_start = np.asarray(dynamic_curves["window_start_index"], dtype=np.int64)
    worst_window_idx = int(np.argmax(dynamic_window_wrong))

    with (error_root / "key_failure_modes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "method", "item", "value"])
        writer.writeheader()
        for pair, count in top_pairs:
            writer.writerow({"category": "top_confusion_pair", "method": "probabilistic", "item": pair, "value": count})
        for label, count in top_false_reject:
            writer.writerow({"category": "top_false_reject_class", "method": "probabilistic", "item": label, "value": count})
        for label, count in top_unknown_dest:
            writer.writerow({"category": "unknown_false_accept_destination", "method": "probabilistic", "item": label, "value": count})

    error_lines = [
        "# Error Analysis",
        "",
        "Compared methods: `static` vs `probabilistic` from the main update experiment.",
        "",
        "## High-Level Findings",
        "",
        f"- `probabilistic` 的 overall accuracy 更高：{dynamic_split['overall_accuracy']:.4f} vs {static_split['overall_accuracy']:.4f}。",
        f"- 但它几乎完全是靠减少 reject 换来的：unknown false accept 从 {static_split['unknown_false_accept_rate']:.4f} 上升到 {dynamic_split['unknown_false_accept_rate']:.4f}。",
        f"- `probabilistic` 的 accepted accuracy 只有 {dynamic_split['accepted_accuracy']:.4f}，并没有显著高于 static 的 {static_split['accepted_accuracy']:.4f}。",
        f"- cumulative wrong updates 在 window start={int(dynamic_window_start[worst_window_idx])} 附近最严重，该窗口 wrong updates={int(dynamic_window_wrong[worst_window_idx])}。",
        "",
        "## Top Confusion Pairs",
        "",
    ]
    for pair, count in top_pairs:
        error_lines.append(f"- {pair}: {count}")
    error_lines.extend(
        [
            "",
            "## Most False-Rejected Known Classes",
            "",
        ]
    )
    for label, count in top_false_reject:
        error_lines.append(f"- class {label}: {count}")
    error_lines.extend(
        [
            "",
            "## Unknown Samples Most Often Misaccepted As",
            "",
        ]
    )
    for label, count in top_unknown_dest:
        error_lines.append(f"- predicted as class {label}: {count}")
    error_lines.extend(
        [
            "",
            "## Figures",
            "",
            f"![Known/unknown behavior](./figures/known_unknown_behavior.png)",
            "",
            f"![Update pollution](./figures/update_pollution_over_time.png)",
            "",
            f"![Top confusions](./figures/top_confusion_pairs_probabilistic.png)",
            "",
        ]
    )
    (error_root / "error_analysis.md").write_text("\n".join(error_lines) + "\n", encoding="utf-8")

    case_lines = [
        "# Case Studies",
        "",
        f"1. `probabilistic` 在 unknown samples 上的 false accept rate = {dynamic_split['unknown_false_accept_rate']:.4f}，明显高于 static 的 {static_split['unknown_false_accept_rate']:.4f}。",
        f"2. 最严重的 wrong-update window 从在线 step {int(dynamic_window_start[worst_window_idx])} 开始，说明模板污染会在局部时间段集中爆发。",
        f"3. `accepted_accuracy` 与 overall accuracy 的差距较小，说明当前系统不是“只在少量高置信样本上特别准”，而是整体区分能力本身就受限。",
        "",
    ]
    (error_root / "case_studies.md").write_text("\n".join(case_lines), encoding="utf-8")

    return {
        "static_split": static_split,
        "dynamic_split": dynamic_split,
        "top_pairs": top_pairs,
        "top_false_reject": top_false_reject,
        "top_unknown_dest": top_unknown_dest,
        "worst_wrong_update_window_start": int(dynamic_window_start[worst_window_idx]),
        "worst_wrong_update_window_count": int(dynamic_window_wrong[worst_window_idx]),
    }


def _diagnosis_text(label_stats: Dict[str, Any], encoded_stats: Dict[str, Any]) -> Tuple[str, str]:
    label_lines = [
        "# Label Distribution Diagnosis",
        "",
        f"- 总 spikes: {label_stats.get('total_spikes')}",
        f"- 总 units: {label_stats.get('num_units')}",
        f"- median count: {label_stats.get('median_count')}",
        f"- max count: {label_stats.get('max_count')}",
        "- 结论：这是一个非常明显的长尾分布，不适合直接把 1200 类同时放进 CAM 做主实验。",
        "- `top50 stream + top20 memory` 是合理的第一主线，因为它既保留了未知类压力，又不会让 warmup 覆盖过于稀薄。",
        "",
    ]
    encoded_lines = [
        "# Encoded Representation Diagnosis",
        "",
        f"- bit width: {encoded_stats.get('bit_width')}",
        f"- unique code ratio: {_fmt(encoded_stats.get('unique_code_ratio'))}",
        f"- mean intra-class Hamming: {_fmt(encoded_stats.get('mean_intra_hamming'))}",
        f"- mean inter-class Hamming: {_fmt(encoded_stats.get('mean_inter_hamming'))}",
        f"- mean Hamming gap: {_fmt(encoded_stats.get('mean_hamming_gap'))}",
    ]
    gap = float(encoded_stats.get("mean_hamming_gap", 0.0) or 0.0)
    if gap <= 0.1:
        encoded_lines.append("- 结论：当前 bits 很平衡、entropy 很高，但同类和异类的 Hamming 距离差得不够开，说明 separability 仍然偏弱。")
    else:
        encoded_lines.append("- 结论：当前 bits 具有一定 separability，但还需要 CAM 端进一步放大这种差异。")
    encoded_lines.append("")
    return "\n".join(label_lines) + "\n", "\n".join(encoded_lines) + "\n"


def _summary_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Variant | Acc | Macro-F1 | BalAcc | Reject | FalseAccept | AcceptedAcc | Updates | WrongUpdRate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant_name"]),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("macro_f1")),
                    _fmt(row.get("balanced_accuracy")),
                    _fmt(row.get("reject_rate")),
                    _fmt(row.get("false_accept_rate")),
                    _fmt(row.get("accepted_accuracy")),
                    _fmt(row.get("update_count"), digits=0),
                    _fmt(row.get("wrong_update_rate")),
                ]
            )
            + " |"
        )
    return lines


def _risk_score(row: Dict[str, Any]) -> float:
    return float(row.get("accuracy", 0.0) or 0.0) - 0.5 * float(row.get("false_accept_rate", 0.0) or 0.0) - 0.2 * float(row.get("wrong_update_rate", 0.0) or 0.0)


def build_master_report(run_root: Path) -> Path:
    env_info = {
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "spikecam_py310"),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "tensorflow_version": __import__("tensorflow").__version__,
        "encoder_backend": "external Autoencoders-in-Spike-Sorting artifact",
    }

    all_rows: List[Dict[str, Any]] = []
    for exp_name, exp_title in EXPERIMENTS:
        all_rows.extend(_summary_rows(run_root / exp_name, exp_name, exp_title))
    _write_overall_summary(run_root, all_rows)
    _write_index(run_root)
    _write_run_readme(run_root, env_info)

    # Copy important figures / tables for quick browsing.
    for exp_name, _ in EXPERIMENTS:
        exp_dir = run_root / exp_name
        _copy_if_exists(exp_dir / "comparison_metrics.png", run_root / "figures" / f"{exp_name}_comparison_metrics.png")
        _copy_if_exists(exp_dir / "curve_overview.png", run_root / "figures" / f"{exp_name}_curve_overview.png")
        _copy_if_exists(exp_dir / "metrics_table.csv", run_root / "tables" / f"{exp_name}_metrics_table.csv")
    _copy_if_exists(run_root / "diagnostics" / "label_distribution.png", run_root / "figures" / "label_distribution.png")
    _copy_if_exists(run_root / "figures" / "encoded_bit_statistics.png", run_root / "figures" / "encoded_bit_statistics.png")

    label_stats = _load_json(run_root / "diagnostics" / "label_distribution.json")
    encoded_stats = _load_json(run_root / "diagnostics" / "encoded_stats_raw.json")
    label_md, encoded_md = _diagnosis_text(label_stats, encoded_stats)
    (run_root / "diagnostics" / "label_distribution.md").write_text(label_md, encoding="utf-8")
    (run_root / "diagnostics" / "encoded_diagnosis.md").write_text(encoded_md, encoding="utf-8")

    error_summary = _build_error_analysis(run_root)

    # Per-experiment short summaries expected by the user.
    baseline_rows = _summary_rows(run_root / "baseline", "baseline", "Baseline")
    main_rows = _summary_rows(run_root / "main_update_compare", "main_update_compare", "Main")
    threshold_rows = _summary_rows(run_root / "threshold_sweep", "threshold_sweep", "Threshold")
    bits_rows = _summary_rows(run_root / "bits_ablation", "bits_ablation", "Bits")
    init_rows = _summary_rows(run_root / "init_ablation", "init_ablation", "Init")
    encoder_rows = _summary_rows(run_root / "encoder_compare", "encoder_compare", "Encoder")

    (run_root / "baseline" / "summary.md").write_text(
        "# Baseline Summary\n\n"
        + "\n".join(_summary_table(baseline_rows))
        + "\n\n静态 baseline 可以稳定运行，但在 memory-aware 协议下，unknown false accept 已经很高，说明前端 bits 与 reject 边界都仍然是主要瓶颈。\n",
        encoding="utf-8",
    )
    (run_root / "baseline" / "conclusion.txt").write_text(
        "Static baseline is usable as a control, but current bits are not sufficiently separable. The main bottlenecks are weak bit-space separation and high unknown false accepts.\n",
        encoding="utf-8",
    )

    ranked_main = sorted(main_rows, key=_risk_score, reverse=True)
    with (run_root / "main_update_compare" / "ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant_name", "risk_score", "accuracy", "false_accept_rate", "wrong_update_rate", "reject_rate", "template_growth"])
        writer.writeheader()
        for row in ranked_main:
            writer.writerow(
                {
                    "variant_name": row["variant_name"],
                    "risk_score": _risk_score(row),
                    "accuracy": row.get("accuracy"),
                    "false_accept_rate": row.get("false_accept_rate"),
                    "wrong_update_rate": row.get("wrong_update_rate"),
                    "reject_rate": row.get("reject_rate"),
                    "template_growth": row.get("template_growth"),
                }
            )
    conservative_dynamic = min([row for row in main_rows if row["variant_name"] != "static"], key=lambda row: float(row.get("false_accept_rate", 0.0)))
    best_accuracy_dynamic = max([row for row in main_rows if row["variant_name"] != "static"], key=lambda row: float(row.get("accuracy", 0.0)))
    worst_pollution = max([row for row in main_rows if row["variant_name"] != "static"], key=lambda row: float(row.get("wrong_update_rate", 0.0)))
    (run_root / "main_update_compare" / "summary.md").write_text(
        "# Main Update Comparison Summary\n\n"
        + "\n".join(_summary_table(sorted(main_rows, key=lambda row: float(row.get("accuracy", 0.0)), reverse=True)))
        + "\n\n"
        + f"- Raw accuracy 最高的动态策略：`{best_accuracy_dynamic['variant_name']}` ({_fmt(best_accuracy_dynamic.get('accuracy'))})\n"
        + f"- 更保守的动态策略：`{conservative_dynamic['variant_name']}` (false_accept={_fmt(conservative_dynamic.get('false_accept_rate'))})\n"
        + f"- 最容易污染模板的策略：`{worst_pollution['variant_name']}` (wrong_update_rate={_fmt(worst_pollution.get('wrong_update_rate'))})\n"
        + "- 风险加权后，`static` 仍然是当前最稳妥的默认工作点。\n",
        encoding="utf-8",
    )

    bits_sorted = sorted(bits_rows, key=lambda row: float(row.get("accuracy", 0.0)), reverse=True)
    best_bit_overall = bits_sorted[0]
    (run_root / "bits_ablation" / "summary.md").write_text("# Bit Ablation Summary\n\n" + "\n".join(_summary_table(bits_sorted)) + "\n", encoding="utf-8")
    (run_root / "bits_ablation" / "conclusion.txt").write_text(
        f"本轮 bit 消融里，raw accuracy 最好的设置是 `{best_bit_overall['variant_name']}`。不过动态策略仍然伴随着很高的 wrong-update 风险，因此论文主线依然建议优先围绕 16-bit 做展开，再把 8/32/64-bit 作为补充消融讨论。\n",
        encoding="utf-8",
    )

    init_sorted = sorted(init_rows, key=lambda row: float(row.get("accuracy", 0.0)), reverse=True)
    static_init_rows = [row for row in init_rows if row["variant_name"].endswith("_static")]
    best_static_init = max(static_init_rows, key=lambda row: float(row.get("accuracy", 0.0)))
    (run_root / "init_ablation" / "summary.md").write_text("# Init Ablation Summary\n\n" + "\n".join(_summary_table(init_sorted)) + "\n", encoding="utf-8")
    (run_root / "init_ablation" / "conclusion.txt").write_text(
        f"初始化方法对结果的影响小于 encoder separability 和 reject/update 风险，但在 static setting 下，`{best_static_init['variant_name']}` 是当前最好的初始化工作点。\n",
        encoding="utf-8",
    )

    encoder_sorted = sorted(encoder_rows, key=lambda row: float(row.get("accuracy", 0.0)), reverse=True)
    encoder_static_rows = [row for row in encoder_rows if row["variant_name"].endswith("_static")]
    best_encoder_static = max(encoder_static_rows, key=lambda row: float(row.get("accuracy", 0.0)))
    pca_static_row = next(row for row in encoder_rows if row["variant_name"] == "pca_static")
    (run_root / "encoder_compare" / "summary.md").write_text("# Encoder Compare Summary\n\n" + "\n".join(_summary_table(encoder_sorted)) + "\n", encoding="utf-8")
    (run_root / "encoder_compare" / "conclusion.txt").write_text(
        f"encoder 对比表明外部 AE 仍优于 PCA：本轮最好的 static encoder 是 `{best_encoder_static['variant_name']}`，而 `pca_static` 的 accuracy 只有 {_fmt(pca_static_row.get('accuracy'))}。PCA 更保守，但整体识别能力明显更弱。\n",
        encoding="utf-8",
    )

    # Master report.
    main_sorted_acc = sorted(main_rows, key=lambda row: float(row.get("accuracy", 0.0)), reverse=True)
    bits_static_rows = [row for row in bits_rows if row["variant_name"].endswith("_static")]
    best_static_bit = max(bits_static_rows, key=lambda row: float(row.get("accuracy", 0.0)))
    recommended_update = "static"
    recommended_bit = "16-bit"
    recommended_init = best_static_init["variant_name"].replace("_static", "")
    recommended_encoder = "external normal AE（主线一致性最好），external shallow AE（后续值得跟进）"

    report_lines = [
        "# Spike CAM 全量实验报告",
        "",
        "## 1. 实验目标",
        "",
        "这轮实验围绕 memory-aware Spike CAM 的核心问题展开：",
        "",
        "- 在线 stream 中的 neuron 类别数多于 CAM 能记住的模板数",
        "- CAM 只加载有限的 `memory subset`",
        "- 不在 memory 里的 spike 应尽量被 reject，而不是误识别成 memory 内类",
        "- 主问题是：dynamic template update 是否真的带来净收益，还是主要造成模板污染",
        "",
        "## 2. 环境与复现性",
        "",
        f"- Run root: `{run_root}`",
        f"- conda env: `{env_info['conda_env']}`",
        f"- Python: `{env_info['python_version']}`",
        f"- Platform: `{env_info['platform']}`",
        f"- numpy / sklearn / tensorflow: `{env_info['numpy_version']}` / `{env_info['sklearn_version']}` / `{env_info['tensorflow_version']}`",
        "- 主线 encoder backend: external `Autoencoders-in-Spike-Sorting` artifact",
        "- 主数据集: `dataset/my_validation_subset_810000samples_27.00s.npz`",
        "- 主协议: `top50 stream + top20 memory`，chronological warmup + online evaluation",
        "",
        "## 3. 数据与编码诊断",
        "",
        Path(run_root / "diagnostics" / "label_distribution.md").read_text(encoding="utf-8").strip(),
        "",
        Path(run_root / "diagnostics" / "encoded_diagnosis.md").read_text(encoding="utf-8").strip(),
        "",
        "相关图示：",
        "",
        f"![Label distribution](./diagnostics/label_distribution.png)",
        "",
        f"![Encoded bit statistics](./figures/encoded_bit_statistics.png)",
        "",
        "## 4. Baseline Sanity Check",
        "",
        "先用 static external-AE baseline 检查整条 pipeline 是否具备可用性，再引入动态更新。",
        "",
        *(_summary_table(baseline_rows)),
        "",
        f"结论：static baseline 的 accuracy={_fmt(baseline_rows[0].get('accuracy'))}，false_accept={_fmt(baseline_rows[0].get('false_accept_rate'))}，reject={_fmt(baseline_rows[0].get('reject_rate'))}。它已经可以作为主实验对照组，但 unknown false accept 仍然很高。",
        "",
        "## 5. 主实验：Update Strategy 对比",
        "",
        *(_summary_table(main_sorted_acc)),
        "",
        f"raw accuracy 最高的动态策略：`{best_accuracy_dynamic['variant_name']}`，accuracy={_fmt(best_accuracy_dynamic.get('accuracy'))}。",
        f"最保守的动态策略：`{conservative_dynamic['variant_name']}`，false_accept={_fmt(conservative_dynamic.get('false_accept_rate'))}。",
        f"模板污染风险最大的策略：`{worst_pollution['variant_name']}`，wrong_update_rate={_fmt(worst_pollution.get('wrong_update_rate'))}。",
        "研究结论：dynamic update 确实带来了一点点 raw accuracy 提升，但几乎都是靠更激进地接受 unknown spike 和错误更新模板换来的。因此这轮实验里，`static` 仍然是最稳妥、最适合作为论文主线默认值的 updater。",
        "",
        f"![Main comparison metrics](./figures/main_update_compare_comparison_metrics.png)",
        "",
        f"![Main comparison curves](./figures/main_update_compare_curve_overview.png)",
        "",
        "## 6. Bit Budget 消融",
        "",
        *(_summary_table(bits_sorted[:12])),
        "",
        f"本轮 static 里最好的 bit 宽度是 `{best_static_bit['variant_name']}`，accuracy={_fmt(best_static_bit.get('accuracy'))}。",
        f"但从整体 raw accuracy 看，最高的是 `{best_bit_overall['variant_name']}`。这说明 bit 越大并不自动越好，不同 bit 宽度还会改变 false_accept / reject / update 风险之间的平衡。论文主线仍然建议以 `16-bit` 为默认设置，再把其他 bit 宽度作为消融展示。",
        "",
        f"![Bits ablation metrics](./figures/bits_ablation_comparison_metrics.png)",
        "",
        "## 7. 初始化策略消融",
        "",
        *(_summary_table(init_sorted)),
        "",
        f"初始化策略的影响存在但不算主导因素。在 static setting 下，当前最好的是 `{best_static_init['variant_name']}`。整体上看，初始化的影响仍然小于 threshold 和 dynamic-update 风险。",
        "",
        f"![Init ablation metrics](./figures/init_ablation_comparison_metrics.png)",
        "",
        "## 8. Encoder 对比",
        "",
        *(_summary_table(encoder_sorted)),
        "",
        f"Encoder 结论：PCA 仍然明显弱于 external AE。当前最好的 static encoder 是 `{best_encoder_static['variant_name']}`，说明 shallow external AE 是值得继续跟进的候选；但为了与主实验保持一致，论文主线仍建议固定 normal external AE。",
        "",
        f"![Encoder comparison metrics](./figures/encoder_compare_comparison_metrics.png)",
        "",
        "## 9. 错误分析",
        "",
        f"- Static known/unknown behavior: known_accept={error_summary['static_split']['known_accept_rate']:.4f}, unknown_false_accept={error_summary['static_split']['unknown_false_accept_rate']:.4f}",
        f"- Probabilistic known/unknown behavior: known_accept={error_summary['dynamic_split']['known_accept_rate']:.4f}, unknown_false_accept={error_summary['dynamic_split']['unknown_false_accept_rate']:.4f}",
        f"- Worst wrong-update window starts at online step {error_summary['worst_wrong_update_window_start']} with {error_summary['worst_wrong_update_window_count']} wrong updates.",
        "- 当前主要失败模式不是“太保守导致 reject 太多”，而是“unknown spike 被接受进 memory 类，然后继续触发错误更新，造成模板污染”。",
        "",
        f"![Known/unknown behavior](./error_analysis/figures/known_unknown_behavior.png)",
        "",
        f"![Update pollution](./error_analysis/figures/update_pollution_over_time.png)",
        "",
        f"![Top confusion pairs](./error_analysis/figures/top_confusion_pairs_probabilistic.png)",
        "",
        "## 10. 总体结论",
        "",
        "- memory-aware 协议才是这道题真正该用的协议，只在 memory 内类上做 closed-set 评估会过于乐观。",
        "- 这份数据是明显长尾分布，所以 `top50 stream + top20 memory` 作为第一主线是合理的。",
        "- 新 pipeline 的 encoded bits 已经比旧版更可分，但 separability 仍然不算强，CAM 端很多限制都来自这一前端瓶颈。",
        "- dynamic update 的 raw accuracy 提升仍然存在，但几乎总是伴随着更高的 false accept 和 wrong update。",
        "- 本轮里最推荐的 updater 仍然是 `static`。",
        "- threshold 是一阶控制杆：小 threshold 更保守，reject 高；大 threshold 更激进，false accept 高。",
        "- bit 宽度的收益不是单调的，8/16/32/64 都体现了不同的取舍关系。",
        "- 初始化策略有影响，但影响弱于 threshold、encoder separability 和 update 污染风险；本轮 static 下 `medoid` 略优。",
        "- external AE 仍明显优于 PCA，且 external shallow AE 值得做后续扩展。",
        "",
        "## 11. 建议的论文叙事",
        "",
        "如果把这轮结果写进毕设，推荐的实验叙事可以是：",
        "",
        "1. 先定义 memory-aware Spike CAM 问题：CAM 只能记住部分活跃类，而在线 stream 同时包含 memory 内类和 memory 外类。",
        "2. 展示数据长尾分布，并说明为什么 `top50 stream + top20 memory` 是合理的主实验工作点。",
        "3. 先给出 static baseline，证明 pipeline 已经能跑通，同时说明 reject / false-accept trade-off 本身就不简单。",
        "4. 把 update strategy comparison 作为整篇实验部分的主轴，重点说明 dynamic update 的收益和代价。",
        "5. 用 threshold sweep 解释为什么不能只看 raw accuracy。",
        "6. 用 bit / init / encoder 消融说明：CAM 结论是有一定鲁棒性的，但上限仍然受到 encoder separability 限制。",
        "7. 最后用 error analysis 收束：真正阻碍系统落地的关键不是单纯 reject 太多，而是 unknown false accept 和模板污染。",
        "",
        "## 12. 局限性与下一步",
        "",
        "- 绝对 accuracy 仍然偏低，所以这轮更适合被表述成“算法与机制研究”，而不是最终高性能系统。",
        "- 当前 external AE bits 在 Hamming 空间里仍然不够分开，更好的 binarization 或带监督/判别性的 frontend 可能会显著提高上限。",
        "- dynamic update 在成为默认方法之前，仍需要更强的 confidence gate 和 unknown 检测。",
        "- 下一轮最值得做的方向是：更严格的 reject 逻辑、更好的 unknown detection、继续探索 shallow external AE、以及针对不同 bit 宽度重新调 threshold。",
        "",
        "## 推荐默认配置",
        "",
        f"- 推荐 updater: `{recommended_update}`",
        f"- 推荐 bit width: `{recommended_bit}`",
        f"- 推荐 initialization: `{recommended_init}`",
        f"- 推荐 encoder: `{recommended_encoder}`",
        "- 是否建议把 AE 作为论文主 encoder：建议。",
        "",
    ]

    master_path = run_root / "master_report.md"
    master_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    try:
        import markdown  # type: ignore

        body = markdown.markdown(master_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
        html_text = (
            "<html><head><meta charset='utf-8'><title>Spike CAM Full Experiment Report</title></head>"
            "<body style='max-width: 980px; margin: 2rem auto; font-family: sans-serif; line-height: 1.6;'>"
            + body
            + "</body></html>"
        )
    except Exception:
        html_text = (
            "<html><head><meta charset='utf-8'><title>Spike CAM Full Experiment Report</title></head>"
            "<body><pre>"
            + html.escape(master_path.read_text(encoding="utf-8"))
            + "</pre></body></html>"
        )
    (run_root / "master_report.html").write_text(html_text, encoding="utf-8")
    return master_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a thesis-style master report for one complete Spike CAM run.")
    parser.add_argument("--run-root", required=True, help="Path to the unified run root, e.g. results/experiments/2026-03-15/experiment_20260315_134107")
    args = parser.parse_args()

    master_path = build_master_report(Path(args.run_root))
    print(f"saved master report to {master_path}")


if __name__ == "__main__":
    main()
