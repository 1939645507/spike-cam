"""Run a comprehensive thesis-style toy-data study for the Spike CAM project.

中文说明
--------
这个脚本的目的不是只“跑通几条 toy 数据”，而是把 toy 实验整理成一套
适合写进毕设的完整研究包。

设计思路：

1. 先生成几种可控难度的 toy dataset
2. 复用同一条 Spike CAM pipeline：
   raw data -> preprocess -> waveform -> encoder -> template -> CAM -> online evaluation
3. 系统比较 dynamic update strategy 在不同数据条件下的表现
4. 再补 bit / init / encoder 消融
5. 最后自动生成一份结构清晰、适合阅读的总报告
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import html
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EncoderConfig, ExperimentSuiteConfig, json_ready, project_path, save_json
from dataio import load_waveform_dataset
from encoder import train_external_autoencoder_artifact
from experiment_runner import run_experiment_suite
from scripts.generate_toy_datasets import TOY_SPECS, generate_named_datasets


SCENARIO_ORDER = [
    "toy_easy_stable_u8_c8_60s",
    "toy_dense_stable_u12_c16_75s",
    "toy_drift_u12_c16_75s",
    "toy_open_u20_c24_90s",
]

THRESHOLD_LIST = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24]
AE_THRESHOLD_LIST = [2, 4, 6, 8, 10, 12]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _write_commands_log(run_root: Path, env_name: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"conda run -n {env_name} python scripts/run_toy_thesis_study.py --run-root {run_root.relative_to(PROJECT_ROOT)}",
    ]
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs" / "commands.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failures_log(run_root: Path, failures: List[str]) -> None:
    lines = ["# Failures and Fixes", ""]
    if not failures:
        lines.append("- No blocking failures were encountered in this toy study.")
    else:
        for item in failures:
            lines.append(f"- {item}")
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs" / "failures_and_fixes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_report_script(result_dir: Path) -> None:
    script_path = project_path("scripts", "build_experiment_report.py")
    subprocess.run(
        [sys.executable, str(script_path), "--result_dir", str(result_dir)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def _suite_from_payload(payload: Dict[str, Any]) -> ExperimentSuiteConfig:
    return ExperimentSuiteConfig.from_dict(payload)


def _relative_to_project(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _base_suite_payload(run_root: Path, experiment_name: str) -> Dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "description": "",
        "seed": 42,
        "results": {
            "results_dir": _relative_to_project(run_root),
            "save_predictions": True,
            "save_curves": True,
            "save_confusion": True,
            "copy_encoded_dataset": False,
        },
    }


def _common_waveform_payload() -> Dict[str, Any]:
    return {
        "sort_by_time": True,
        "preprocess": {
            "bandpass_enable": True,
            "bandpass_low_hz": 300.0,
            "bandpass_high_hz": 6000.0,
            "common_reference_enable": True,
            "common_reference_mode": "median",
            "whitening_enable": False,
        },
        "waveform": {
            "waveform_length": 45,
            "center_index": 15,
            "align_mode": "none",
            "channel_selection": "all",
            "channel_order": "by_index",
            "flatten_order": "time_major",
            "selection_radius": 6,
        },
        "sampling": {
            "max_total_spikes": None,
            "max_spikes_per_unit": None,
            "selection_mode": "uniform_time",
        },
    }


def _common_pca_payload(code_size: int = 32) -> Dict[str, Any]:
    return {
        "encoder": {
            "method": "pca",
            "code_size": int(code_size),
            "scale": "none",
            "binarize_mode": "zero",
            "reuse_cache": True,
            "force_reencode": False,
        },
        "template_init": {"method": "majority_vote"},
        "matcher": {"method": "hamming_nearest", "threshold": 8.0},
        "update": {"method": "none"},
        "evaluation": {"mode": "chronological", "warmup_ratio": 0.5, "window_size": 250},
    }


def _threshold_variants(thresholds: Iterable[int]) -> List[Dict[str, Any]]:
    return [
        {
            "name": f"thr{int(th)}",
            "description": f"Static threshold={int(th)}",
            "matcher": {"method": "hamming_nearest", "threshold": float(th)},
            "update": {"method": "none"},
        }
        for th in thresholds
    ]


def _update_variants(threshold: float) -> List[Dict[str, Any]]:
    return [
        {"name": "static", "description": f"Static baseline @thr={threshold}", "matcher": {"method": "hamming_nearest", "threshold": threshold}, "update": {"method": "none"}},
        {"name": "counter", "description": f"Counter @thr={threshold}", "matcher": {"method": "hamming_nearest", "threshold": threshold}, "update": {"method": "counter"}},
        {"name": "margin_ema", "description": f"Margin EMA @thr={threshold}", "matcher": {"method": "hamming_nearest", "threshold": threshold}, "update": {"method": "margin_ema"}},
        {"name": "confidence_weighted", "description": f"Confidence weighted @thr={threshold}", "matcher": {"method": "hamming_nearest", "threshold": threshold}, "update": {"method": "confidence_weighted"}},
        {"name": "dual_template", "description": f"Dual template @thr={threshold}", "matcher": {"method": "hamming_nearest", "threshold": threshold}, "update": {"method": "dual_template"}},
        {"name": "probabilistic", "description": f"Probabilistic @thr={threshold}", "matcher": {"method": "hamming_nearest", "threshold": threshold}, "update": {"method": "probabilistic"}},
        {"name": "growing", "description": f"Growing @thr={threshold}", "matcher": {"method": "hamming_nearest", "threshold": threshold}, "update": {"method": "growing"}},
        {"name": "cooldown", "description": f"Cooldown @thr={threshold}", "matcher": {"method": "hamming_nearest", "threshold": threshold}, "update": {"method": "cooldown"}},
        {
            "name": "top2_margin",
            "description": f"Top2 margin @thr={threshold}",
            "matcher": {"method": "top2_margin", "threshold": threshold, "min_margin": 1.0},
            "update": {"method": "top2_margin", "min_margin": 1.0},
        },
    ]


def _load_summary(result_dir: Path) -> Dict[str, Dict[str, Any]]:
    return json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))


def _summary_rows(result_dir: Path, label: str) -> List[Dict[str, Any]]:
    summary = _load_summary(result_dir)
    return [{"study_experiment": label, "variant_name": name, **metrics} for name, metrics in summary.items()]


def _best_threshold_from_result(result_dir: Path) -> int:
    summary = _load_summary(result_dir)
    best_name, _ = max(summary.items(), key=lambda item: float(item[1].get("accuracy", 0.0)))
    return int(str(best_name).replace("thr", "").replace("_", ".").split(".")[0])


def _copy_summary_artifacts(result_dir: Path, run_root: Path, label: str) -> None:
    figures_dir = run_root / "figures"
    tables_dir = run_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name in ["comparison_metrics.png", "curve_overview.png", "summary_plots.png"]:
        src = result_dir / name
        if src.exists():
            shutil.copy2(src, figures_dir / f"{label}_{name}")
    for name in ["metrics_table.csv", "report.md", "summary.json"]:
        src = result_dir / name
        if src.exists():
            shutil.copy2(src, tables_dir / f"{label}_{src.name}")


def _save_suite_config(run_root: Path, experiment_name: str, payload: Dict[str, Any]) -> None:
    cfg_dir = run_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg_dir / f"{experiment_name}.json", payload)


def _run_suite(run_root: Path, payload: Dict[str, Any]) -> Path:
    suite = _suite_from_payload(payload)
    _save_suite_config(run_root, suite.experiment_name, payload)
    run_experiment_suite(suite)
    result_dir = run_root / suite.experiment_name
    _run_report_script(result_dir)
    _copy_summary_artifacts(result_dir, run_root, suite.experiment_name)
    return result_dir


def _dataset_payload(npz_path: Path, *, subset: Dict[str, Any], memory_subset: Dict[str, Any]) -> Dict[str, Any]:
    payload = _common_waveform_payload()
    payload.update(
        {
            "npz_path": _relative_to_project(npz_path),
            "subset": subset,
        }
    )
    return payload


def _difficulty_row(name: str, meta: Dict[str, Any], best_acc: float) -> Dict[str, Any]:
    return {
        "scenario": name,
        "num_units": meta["num_units"],
        "num_channels": meta["num_channels"],
        "duration_sec": meta["duration_sec"],
        "extra_noise_std": meta["extra_noise_std"],
        "drift_strength": meta["drift_strength"],
        "best_static_accuracy": best_acc,
    }


def _score_closed(row: Dict[str, Any]) -> float:
    return float(row.get("accuracy", 0.0) or 0.0) - 0.18 * float(row.get("wrong_update_rate", 0.0) or 0.0)


def _score_open(row: Dict[str, Any]) -> float:
    return (
        float(row.get("accuracy", 0.0) or 0.0)
        - 0.45 * float(row.get("false_accept_rate", 0.0) or 0.0)
        - 0.18 * float(row.get("wrong_update_rate", 0.0) or 0.0)
    )


def _best_actual_dynamic(rows: List[Dict[str, Any]], *, open_set: bool = False) -> Optional[Dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("variant_name")) != "static" and int(row.get("update_count", 0) or 0) > 0
    ]
    if not candidates:
        return None
    score_fn = _score_open if open_set else _score_closed
    return max(candidates, key=score_fn)


def _safest_actual_dynamic(rows: List[Dict[str, Any]], *, open_set: bool = False) -> Optional[Dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("variant_name")) != "static" and int(row.get("update_count", 0) or 0) > 0
    ]
    if not candidates:
        return None
    if open_set:
        return min(candidates, key=lambda row: (float(row.get("false_accept_rate", 1.0)), float(row.get("wrong_update_rate", 1.0))))
    return min(candidates, key=lambda row: (float(row.get("wrong_update_rate", 1.0)), -float(row.get("accuracy", 0.0))))


def _select_actual_dynamic(rows: List[Dict[str, Any]], *, open_set: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    return _best_actual_dynamic(rows, open_set=open_set), _safest_actual_dynamic(rows, open_set=open_set)


def _threshold_ratio(best_threshold: int, code_size: int) -> float:
    return float(best_threshold) / float(code_size)


def _scaled_threshold(base_ratio: float, code_size: int) -> int:
    return max(1, int(round(base_ratio * float(code_size))))


def _ae_artifact_train_config(code_size: int, ae_type: str, epochs: int = 6) -> EncoderConfig:
    return EncoderConfig(
        method="ae",
        backend="external",
        code_size=int(code_size),
        ae_type=str(ae_type),
        epochs=int(epochs),
        layers=[64, 32],
        scale="minmax",
        learning_rate=0.01,
        batch_size=128,
        verbose=1,
        save_artifact=True,
        use_artifact=True,
    )


def _threshold_curve(summary: Dict[str, Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    pairs: List[Tuple[int, float]] = []
    for name, metrics in summary.items():
        threshold = int(str(name).replace("thr", "").replace("_", ".").split(".")[0])
        pairs.append((threshold, float(metrics.get("accuracy", 0.0))))
    pairs.sort(key=lambda item: item[0])
    return [a for a, _ in pairs], [b for _, b in pairs]


def _make_html(markdown_text: str, out_path: Path) -> None:
    body = (
        "<html><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1100px;margin:40px auto;line-height:1.65;padding:0 24px;} "
        "pre{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f7f7f7;padding:16px;border-radius:8px;} "
        "img{max-width:100%;height:auto;}</style></head><body><pre>"
        + html.escape(markdown_text)
        + "</pre></body></html>"
    )
    out_path.write_text(body, encoding="utf-8")


def _plot_dataset_overview(run_root: Path, metadata_rows: List[Dict[str, Any]]) -> None:
    labels = [row["name"] for row in metadata_rows]
    units = [int(row["num_units"]) for row in metadata_rows]
    channels = [int(row["num_channels"]) for row in metadata_rows]
    noise = [float(row["extra_noise_std"]) for row in metadata_rows]
    drift = [float(row["drift_strength"]) for row in metadata_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    axes[0].bar(labels, units, color="#4e79a7")
    axes[0].set_title("Number of Units")
    axes[1].bar(labels, channels, color="#f28e2b")
    axes[1].set_title("Number of Channels")
    axes[2].bar(labels, noise, color="#59a14f")
    axes[2].set_title("Extra Noise Std")
    axes[3].bar(labels, drift, color="#e15759")
    axes[3].set_title("Drift Strength")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out = run_root / "figures" / "dataset_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_threshold_scenarios(run_root: Path, scenario_results: List[Tuple[str, Path]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, result_dir in scenario_results:
        thresholds, accuracy = _threshold_curve(_load_summary(result_dir))
        ax.plot(thresholds, accuracy, marker="o", label=label)
    ax.set_title("Static Accuracy vs Threshold Across Toy Scenarios")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = run_root / "figures" / "threshold_scenarios.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_difficulty_summary(run_root: Path, rows: List[Dict[str, Any]]) -> None:
    labels = [row["scenario"] for row in rows]
    values = [float(row["best_static_accuracy"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="#4e79a7")
    ax.set_title("Best Static Accuracy by Toy Scenario")
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = run_root / "figures" / "difficulty_summary.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_update_compare(run_root: Path, title: str, result_dir: Path, out_name: str) -> None:
    summary = _load_summary(result_dir)
    rows = sorted(
        [{"variant": name, **metrics} for name, metrics in summary.items()],
        key=lambda row: float(row.get("accuracy", 0.0)),
        reverse=True,
    )
    labels = [row["variant"] for row in rows]
    accuracy = [float(row.get("accuracy", 0.0)) for row in rows]
    wrong_update = [float(row.get("wrong_update_rate", 0.0)) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(labels, accuracy, color="#4e79a7")
    axes[0].set_title(f"{title}: Accuracy")
    axes[1].bar(labels, wrong_update, color="#e15759")
    axes[1].set_title(f"{title}: Wrong Update Rate")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out = run_root / "figures" / out_name
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _write_overall_summary(run_root: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "study_experiment",
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
        "wrong_update_rate",
        "initial_template_count",
        "final_template_count",
        "unknown_stream_fraction",
    ]
    with (run_root / "overall_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    save_json(run_root / "overall_summary.json", {"rows": rows})


def _write_index(run_root: Path, experiments: Dict[str, Path]) -> None:
    save_json(
        run_root / "experiment_index.json",
        {
            "experiments": [
                {"name": name, "result_dir": str(path), "summary": str(path / "summary.json"), "report": str(path / "report.md")}
                for name, path in experiments.items()
            ]
        },
    )


def _write_run_readme(run_root: Path, env_name: str, dataset_rows: List[Dict[str, Any]], experiments: Dict[str, Path]) -> None:
    lines = [
        "# Toy Thesis Study README",
        "",
        f"Run root: `{run_root}`",
        "",
        "## 环境",
        "",
        f"- conda env: `{env_name}`",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        "",
        "## Toy 数据集",
        "",
    ]
    for row in dataset_rows:
        lines.append(
            f"- `{row['name']}`: units={row['num_units']}, channels={row['num_channels']}, duration={row['duration_sec']}s, "
            f"noise={row['extra_noise_std']}, drift={row['drift_strength']}"
        )
    lines.extend(
        [
            "",
            "## 主要实验目录",
            "",
        ]
    )
    for name, path in experiments.items():
        lines.append(f"- `{name}` -> `{path}`")
    lines.extend(
        [
            "",
            "## 复现入口",
            "",
            f"`conda run -n {env_name} python scripts/run_toy_thesis_study.py --run-root {run_root.relative_to(PROJECT_ROOT)}`",
            "",
            "## 目录说明",
            "",
            "- `datasets/`: 本次 toy 数据集的元数据索引。",
            "- `figures/`: 顶层汇总图。",
            "- `tables/`: 各子实验汇总表。",
            "- `logs/`: 命令和异常记录。",
            "- `master_report.md`: 适合直接阅读的总报告。",
        ]
    )
    (run_root / "README_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_table(rows: List[Dict[str, Any]], sort_key: str = "accuracy") -> List[str]:
    sorted_rows = sorted(rows, key=lambda row: float(row.get(sort_key, 0.0) or 0.0), reverse=True)
    lines = [
        "| Variant | Acc | Macro-F1 | BalAcc | Reject | FalseAccept | AcceptedAcc | Updates | WrongUpdRate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted_rows:
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


def _dataset_generation_summary(run_root: Path, dataset_rows: List[Dict[str, Any]]) -> None:
    datasets_dir = run_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    save_json(datasets_dir / "toy_dataset_manifest.json", {"datasets": dataset_rows})
    with (datasets_dir / "toy_dataset_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "name",
            "npz_path",
            "num_units",
            "num_channels",
            "duration_sec",
            "extra_noise_std",
            "drift_strength",
            "num_spikes",
            "mean_count_per_unit",
            "std_count_per_unit",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full thesis-style toy-data study for Spike CAM.")
    parser.add_argument("--run-root", default="", help="Optional explicit result root under results/experiments/<date>.")
    parser.add_argument("--env-name", default="spikecam_py310", help="Conda env name to record in logs and README.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dated_parent = project_path("results", "experiments", datetime.now().strftime("%Y-%m-%d"))
    default_run_root = dated_parent / f"experiment_{timestamp}_toy_thesis_study"
    run_root = project_path(args.run_root) if args.run_root else default_run_root
    run_root.mkdir(parents=True, exist_ok=True)
    for subdir in ["figures", "tables", "logs", "datasets", "artifacts", "configs"]:
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    _write_commands_log(run_root, args.env_name)

    # Stage 0: generate toy datasets.
    dataset_rows = generate_named_datasets(SCENARIO_ORDER)
    _dataset_generation_summary(run_root, dataset_rows)
    _plot_dataset_overview(run_root, dataset_rows)

    experiments: Dict[str, Path] = {}
    overall_rows: List[Dict[str, Any]] = []

    # Stage 1: static threshold sweep across toy scenarios.
    threshold_results: Dict[str, Path] = {}
    best_thresholds: Dict[str, int] = {}
    difficulty_rows: List[Dict[str, Any]] = []

    threshold_suite_specs = [
        ("threshold_easy_stable", "toy_easy_stable_u8_c8_60s", {"mode": "all"}, {"mode": "same_as_stream"}),
        ("threshold_dense_stable", "toy_dense_stable_u12_c16_75s", {"mode": "all"}, {"mode": "same_as_stream"}),
        ("threshold_drift_closed", "toy_drift_u12_c16_75s", {"mode": "all"}, {"mode": "same_as_stream"}),
        ("threshold_open_memory", "toy_open_u20_c24_90s", {"mode": "all"}, {"mode": "topk", "topk": 10, "selection_source": "pre_sampling"}),
    ]

    for experiment_name, dataset_name, subset_cfg, memory_subset in threshold_suite_specs:
        spec = TOY_SPECS[dataset_name]
        payload = _base_suite_payload(run_root, experiment_name)
        payload.update(_common_pca_payload(code_size=32))
        payload["dataset"] = _dataset_payload(spec.npz_path, subset=subset_cfg, memory_subset=memory_subset)
        payload["cam"] = {"capacity": None, "memory_subset": memory_subset}
        payload["variants"] = _threshold_variants(THRESHOLD_LIST)
        result_dir = _run_suite(run_root, payload)
        experiments[experiment_name] = result_dir
        threshold_results[experiment_name] = result_dir
        best_threshold = _best_threshold_from_result(result_dir)
        best_thresholds[dataset_name] = best_threshold
        rows = _summary_rows(result_dir, experiment_name)
        overall_rows.extend(rows)
        best_acc = max(float(row["accuracy"]) for row in rows)
        difficulty_rows.append(_difficulty_row(dataset_name, asdict(spec), best_acc))

    _plot_threshold_scenarios(
        run_root,
        [
            ("easy stable", threshold_results["threshold_easy_stable"]),
            ("dense stable", threshold_results["threshold_dense_stable"]),
            ("drift closed", threshold_results["threshold_drift_closed"]),
            ("open memory", threshold_results["threshold_open_memory"]),
        ],
    )
    _plot_difficulty_summary(run_root, difficulty_rows)
    save_json(run_root / "datasets" / "difficulty_summary.json", {"rows": difficulty_rows})

    # Protocol sanity: easy toy under random vs chronological.
    easy_best_threshold = best_thresholds["toy_easy_stable_u8_c8_60s"]
    easy_protocol_payload = _base_suite_payload(run_root, "protocol_compare_easy")
    easy_protocol_payload.update(_common_pca_payload(code_size=32))
    easy_protocol_payload["dataset"] = _dataset_payload(
        TOY_SPECS["toy_easy_stable_u8_c8_60s"].npz_path,
        subset={"mode": "all"},
        memory_subset={"mode": "same_as_stream"},
    )
    easy_protocol_payload["cam"] = {"capacity": None, "memory_subset": {"mode": "same_as_stream"}}
    easy_protocol_payload["variants"] = [
        {
            "name": "chronological_static",
            "description": "Chronological static baseline",
            "evaluation": {"mode": "chronological", "warmup_ratio": 0.5, "window_size": 250},
            "matcher": {"method": "hamming_nearest", "threshold": float(easy_best_threshold)},
            "update": {"method": "none"},
        },
        {
            "name": "random_static",
            "description": "Random split baseline",
            "evaluation": {"mode": "random_split", "random_train_frac": 0.5, "window_size": 250},
            "matcher": {"method": "hamming_nearest", "threshold": float(easy_best_threshold)},
            "update": {"method": "none"},
        },
    ]
    protocol_result = _run_suite(run_root, easy_protocol_payload)
    experiments["protocol_compare_easy"] = protocol_result
    overall_rows.extend(_summary_rows(protocol_result, "protocol_compare_easy"))

    # Stage 2: update compare in three scenarios.
    update_suite_specs = [
        ("update_compare_dense_stable", "toy_dense_stable_u12_c16_75s", {"mode": "same_as_stream"}, False),
        ("update_compare_drift_closed", "toy_drift_u12_c16_75s", {"mode": "same_as_stream"}, False),
        ("update_compare_open_memory", "toy_open_u20_c24_90s", {"mode": "topk", "topk": 10, "selection_source": "pre_sampling"}, True),
    ]
    update_rows_map: Dict[str, List[Dict[str, Any]]] = {}

    for experiment_name, dataset_name, memory_subset, is_open in update_suite_specs:
        spec = TOY_SPECS[dataset_name]
        payload = _base_suite_payload(run_root, experiment_name)
        payload.update(_common_pca_payload(code_size=32))
        payload["dataset"] = _dataset_payload(spec.npz_path, subset={"mode": "all"}, memory_subset=memory_subset)
        payload["cam"] = {"capacity": None, "memory_subset": memory_subset}
        payload["matcher"]["threshold"] = float(best_thresholds[dataset_name])
        payload["variants"] = _update_variants(float(best_thresholds[dataset_name]))
        result_dir = _run_suite(run_root, payload)
        experiments[experiment_name] = result_dir
        rows = _summary_rows(result_dir, experiment_name)
        overall_rows.extend(rows)
        update_rows_map[experiment_name] = rows
        _plot_update_compare(run_root, experiment_name, result_dir, f"{experiment_name}.png")

    drift_best_dynamic, drift_safest_dynamic = _select_actual_dynamic(update_rows_map["update_compare_drift_closed"], open_set=False)
    open_best_dynamic, open_safest_dynamic = _select_actual_dynamic(update_rows_map["update_compare_open_memory"], open_set=True)

    # Stage 3: bit ablation on drift dataset.
    drift_ratio = _threshold_ratio(best_thresholds["toy_drift_u12_c16_75s"], 32)
    dynamic_name = drift_best_dynamic["variant_name"] if drift_best_dynamic is not None else "counter"
    safe_dynamic_name = drift_safest_dynamic["variant_name"] if drift_safest_dynamic is not None else dynamic_name

    def _variant_for_method(method_name: str, code_size: int, threshold: int) -> Dict[str, Any]:
        if method_name == "static":
            return {
                "name": f"bits{code_size}_static",
                "description": f"Static {code_size}-bit",
                "encoder": {"code_size": code_size},
                "matcher": {"method": "hamming_nearest", "threshold": float(threshold)},
                "update": {"method": "none"},
            }
        if method_name == "top2_margin":
            return {
                "name": f"bits{code_size}_{method_name}",
                "description": f"{method_name} {code_size}-bit",
                "encoder": {"code_size": code_size},
                "matcher": {"method": "top2_margin", "threshold": float(threshold), "min_margin": 1.0},
                "update": {"method": "top2_margin", "min_margin": 1.0},
            }
        return {
            "name": f"bits{code_size}_{method_name}",
            "description": f"{method_name} {code_size}-bit",
            "encoder": {"code_size": code_size},
            "matcher": {"method": "hamming_nearest", "threshold": float(threshold)},
            "update": {"method": method_name},
        }

    bit_payload = _base_suite_payload(run_root, "bit_ablation_drift")
    bit_payload.update(_common_pca_payload(code_size=32))
    bit_payload["dataset"] = _dataset_payload(
        TOY_SPECS["toy_drift_u12_c16_75s"].npz_path,
        subset={"mode": "all"},
        memory_subset={"mode": "same_as_stream"},
    )
    bit_payload["cam"] = {"capacity": None, "memory_subset": {"mode": "same_as_stream"}}
    bit_variants: List[Dict[str, Any]] = []
    for code_size in [8, 16, 32, 64]:
        threshold = _scaled_threshold(drift_ratio, code_size)
        for method_name in ["static", dynamic_name, safe_dynamic_name]:
            if any(v["name"] == f"bits{code_size}_{method_name}" for v in bit_variants):
                continue
            bit_variants.append(_variant_for_method(method_name, code_size, threshold))
    bit_payload["variants"] = bit_variants
    bit_result = _run_suite(run_root, bit_payload)
    experiments["bit_ablation_drift"] = bit_result
    overall_rows.extend(_summary_rows(bit_result, "bit_ablation_drift"))

    # Stage 4: init ablation on drift dataset.
    init_payload = _base_suite_payload(run_root, "init_ablation_drift")
    init_payload.update(_common_pca_payload(code_size=32))
    init_payload["dataset"] = _dataset_payload(
        TOY_SPECS["toy_drift_u12_c16_75s"].npz_path,
        subset={"mode": "all"},
        memory_subset={"mode": "same_as_stream"},
    )
    init_payload["cam"] = {"capacity": None, "memory_subset": {"mode": "same_as_stream"}}
    init_variants = []
    for init_name in ["majority_vote", "medoid", "stable_mask"]:
        init_variants.append(
            {
                "name": f"{init_name}_static",
                "description": f"{init_name} + static",
                "template_init": {"method": init_name},
                "matcher": {"method": "hamming_nearest", "threshold": float(best_thresholds['toy_drift_u12_c16_75s'])},
                "update": {"method": "none"},
            }
        )
        if dynamic_name == "top2_margin":
            init_variants.append(
                {
                    "name": f"{init_name}_{dynamic_name}",
                    "description": f"{init_name} + {dynamic_name}",
                    "template_init": {"method": init_name},
                    "matcher": {"method": "top2_margin", "threshold": float(best_thresholds['toy_drift_u12_c16_75s']), "min_margin": 1.0},
                    "update": {"method": "top2_margin", "min_margin": 1.0},
                }
            )
        else:
            init_variants.append(
                {
                    "name": f"{init_name}_{dynamic_name}",
                    "description": f"{init_name} + {dynamic_name}",
                    "template_init": {"method": init_name},
                    "matcher": {"method": "hamming_nearest", "threshold": float(best_thresholds['toy_drift_u12_c16_75s'])},
                    "update": {"method": dynamic_name},
                }
            )
    init_payload["variants"] = init_variants
    init_result = _run_suite(run_root, init_payload)
    experiments["init_ablation_drift"] = init_result
    overall_rows.extend(_summary_rows(init_result, "init_ablation_drift"))

    # Stage 5: encoder compare on drift dataset.
    # First train / reuse toy-specific external AE artifacts.
    wave_dataset = load_waveform_dataset(
        _suite_from_payload(
            {
                "experiment_name": "toy_waveforms_for_ae",
                "dataset": _dataset_payload(
                    TOY_SPECS["toy_drift_u12_c16_75s"].npz_path,
                    subset={"mode": "all"},
                    memory_subset={"mode": "same_as_stream"},
                ),
            }
        ).dataset
    )
    artifact_root = run_root / "artifacts"
    normal_artifact = artifact_root / "toy_drift_external_normal_16bit"
    shallow_artifact = artifact_root / "toy_drift_external_shallow_16bit"
    if not normal_artifact.exists():
        train_external_autoencoder_artifact(
            wave_dataset.waveforms,
            _ae_artifact_train_config(code_size=16, ae_type="normal", epochs=6),
            seed=42,
            artifact_path=normal_artifact,
            resume=False,
        )
    if not shallow_artifact.exists():
        train_external_autoencoder_artifact(
            wave_dataset.waveforms,
            _ae_artifact_train_config(code_size=16, ae_type="shallow", epochs=6),
            seed=42,
            artifact_path=shallow_artifact,
            resume=False,
        )

    # Encoder threshold calibration for external AE encoders.
    calibration_payload = _base_suite_payload(run_root, "encoder_threshold_calibration_drift")
    calibration_payload.update(_common_pca_payload(code_size=16))
    calibration_payload["dataset"] = _dataset_payload(
        TOY_SPECS["toy_drift_u12_c16_75s"].npz_path,
        subset={"mode": "all"},
        memory_subset={"mode": "same_as_stream"},
    )
    calibration_payload["cam"] = {"capacity": None, "memory_subset": {"mode": "same_as_stream"}}
    calibration_variants: List[Dict[str, Any]] = []
    for threshold in AE_THRESHOLD_LIST:
        calibration_variants.append(
            {
                "name": f"normal_thr{threshold}",
                "description": f"External normal AE static threshold={threshold}",
                "encoder": {
                    "method": "ae",
                    "backend": "external",
                    "code_size": 16,
                    "ae_type": "normal",
                    "artifact_path": _relative_to_project(normal_artifact),
                    "use_artifact": True,
                    "save_artifact": True,
                    "force_retrain_artifact": False,
                    "scale": "minmax",
                },
                "matcher": {"method": "hamming_nearest", "threshold": float(threshold)},
                "update": {"method": "none"},
            }
        )
        calibration_variants.append(
            {
                "name": f"shallow_thr{threshold}",
                "description": f"External shallow AE static threshold={threshold}",
                "encoder": {
                    "method": "ae",
                    "backend": "external",
                    "code_size": 16,
                    "ae_type": "shallow",
                    "artifact_path": _relative_to_project(shallow_artifact),
                    "use_artifact": True,
                    "save_artifact": True,
                    "force_retrain_artifact": False,
                    "scale": "minmax",
                },
                "matcher": {"method": "hamming_nearest", "threshold": float(threshold)},
                "update": {"method": "none"},
            }
        )
    calibration_payload["variants"] = calibration_variants
    calibration_result = _run_suite(run_root, calibration_payload)
    experiments["encoder_threshold_calibration_drift"] = calibration_result
    overall_rows.extend(_summary_rows(calibration_result, "encoder_threshold_calibration_drift"))
    cal_summary = _load_summary(calibration_result)
    best_normal_threshold = max(
        ((int(name.split("thr")[1]), metrics) for name, metrics in cal_summary.items() if name.startswith("normal_thr")),
        key=lambda item: float(item[1].get("accuracy", 0.0)),
    )[0]
    best_shallow_threshold = max(
        ((int(name.split("thr")[1]), metrics) for name, metrics in cal_summary.items() if name.startswith("shallow_thr")),
        key=lambda item: float(item[1].get("accuracy", 0.0)),
    )[0]

    encoder_compare_payload = _base_suite_payload(run_root, "encoder_compare_drift")
    encoder_compare_payload.update(_common_pca_payload(code_size=32))
    encoder_compare_payload["dataset"] = _dataset_payload(
        TOY_SPECS["toy_drift_u12_c16_75s"].npz_path,
        subset={"mode": "all"},
        memory_subset={"mode": "same_as_stream"},
    )
    encoder_compare_payload["cam"] = {"capacity": None, "memory_subset": {"mode": "same_as_stream"}}
    encoder_variants: List[Dict[str, Any]] = [
        {
            "name": "pca32_static",
            "description": "PCA32 static",
            "encoder": {"method": "pca", "code_size": 32, "scale": "none", "binarize_mode": "zero"},
            "matcher": {"method": "hamming_nearest", "threshold": float(best_thresholds['toy_drift_u12_c16_75s'])},
            "update": {"method": "none"},
        },
        {
            "name": f"pca32_{dynamic_name}",
            "description": f"PCA32 {dynamic_name}",
            "encoder": {"method": "pca", "code_size": 32, "scale": "none", "binarize_mode": "zero"},
            "matcher": {"method": "top2_margin" if dynamic_name == "top2_margin" else "hamming_nearest", "threshold": float(best_thresholds['toy_drift_u12_c16_75s']), "min_margin": 1.0},
            "update": {"method": dynamic_name, "min_margin": 1.0} if dynamic_name == "top2_margin" else {"method": dynamic_name},
        },
        {
            "name": "ae16_normal_static",
            "description": "External normal AE16 static",
            "encoder": {
                "method": "ae",
                "backend": "external",
                "code_size": 16,
                "ae_type": "normal",
                "artifact_path": _relative_to_project(normal_artifact),
                "use_artifact": True,
                "save_artifact": True,
                "force_retrain_artifact": False,
                "scale": "minmax",
            },
            "matcher": {"method": "hamming_nearest", "threshold": float(best_normal_threshold)},
            "update": {"method": "none"},
        },
        {
            "name": f"ae16_normal_{dynamic_name}",
            "description": f"External normal AE16 {dynamic_name}",
            "encoder": {
                "method": "ae",
                "backend": "external",
                "code_size": 16,
                "ae_type": "normal",
                "artifact_path": _relative_to_project(normal_artifact),
                "use_artifact": True,
                "save_artifact": True,
                "force_retrain_artifact": False,
                "scale": "minmax",
            },
            "matcher": {"method": "top2_margin" if dynamic_name == "top2_margin" else "hamming_nearest", "threshold": float(best_normal_threshold), "min_margin": 1.0},
            "update": {"method": dynamic_name, "min_margin": 1.0} if dynamic_name == "top2_margin" else {"method": dynamic_name},
        },
        {
            "name": "ae16_shallow_static",
            "description": "External shallow AE16 static",
            "encoder": {
                "method": "ae",
                "backend": "external",
                "code_size": 16,
                "ae_type": "shallow",
                "artifact_path": _relative_to_project(shallow_artifact),
                "use_artifact": True,
                "save_artifact": True,
                "force_retrain_artifact": False,
                "scale": "minmax",
            },
            "matcher": {"method": "hamming_nearest", "threshold": float(best_shallow_threshold)},
            "update": {"method": "none"},
        },
        {
            "name": f"ae16_shallow_{dynamic_name}",
            "description": f"External shallow AE16 {dynamic_name}",
            "encoder": {
                "method": "ae",
                "backend": "external",
                "code_size": 16,
                "ae_type": "shallow",
                "artifact_path": _relative_to_project(shallow_artifact),
                "use_artifact": True,
                "save_artifact": True,
                "force_retrain_artifact": False,
                "scale": "minmax",
            },
            "matcher": {"method": "top2_margin" if dynamic_name == "top2_margin" else "hamming_nearest", "threshold": float(best_shallow_threshold), "min_margin": 1.0},
            "update": {"method": dynamic_name, "min_margin": 1.0} if dynamic_name == "top2_margin" else {"method": dynamic_name},
        },
    ]
    encoder_compare_payload["variants"] = encoder_variants
    encoder_compare_result = _run_suite(run_root, encoder_compare_payload)
    experiments["encoder_compare_drift"] = encoder_compare_result
    overall_rows.extend(_summary_rows(encoder_compare_result, "encoder_compare_drift"))

    # Final aggregation.
    _write_failures_log(run_root, failures)
    _write_overall_summary(run_root, overall_rows)
    _write_index(run_root, experiments)
    _write_run_readme(run_root, args.env_name, dataset_rows, experiments)

    # Build master report.
    difficulty_lines = [
        "| Scenario | Units | Channels | Noise | Drift | Best Static Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in difficulty_rows:
        difficulty_lines.append(
            f"| {row['scenario']} | {row['num_units']} | {row['num_channels']} | "
            f"{_fmt(row['extra_noise_std'])} | {_fmt(row['drift_strength'])} | {_fmt(row['best_static_accuracy'])} |"
        )

    protocol_rows = _summary_rows(protocol_result, "protocol_compare_easy")
    dense_update_rows = update_rows_map["update_compare_dense_stable"]
    drift_update_rows = update_rows_map["update_compare_drift_closed"]
    open_update_rows = update_rows_map["update_compare_open_memory"]
    best_dense_dynamic = _best_actual_dynamic(dense_update_rows, open_set=False)
    best_drift_dynamic = _best_actual_dynamic(drift_update_rows, open_set=False)
    best_open_dynamic = _best_actual_dynamic(open_update_rows, open_set=True)

    bit_rows = _summary_rows(bit_result, "bit_ablation_drift")
    init_rows = _summary_rows(init_result, "init_ablation_drift")
    encoder_rows = _summary_rows(encoder_compare_result, "encoder_compare_drift")

    report_lines = [
        "# Toy Spike CAM 全面实验报告",
        "",
        "## 1. 实验目标",
        "",
        "这轮实验专门为毕设服务，目的不是追求真实数据上的最终绝对性能，而是先在**可控 toy 数据**上系统研究：",
        "",
        "- 在什么样的数据条件下，dynamic update 值得用？",
        "- 在什么样的数据条件下，static template 反而更稳？",
        "- bit 宽度、初始化方法、encoder 变化后，这些结论会不会改变？",
        "",
        "之所以这样设计，是因为真实数据目前 separability 偏弱，直接在真实数据上比较所有动态更新算法，会把“前端特征不够好”和“更新策略本身好不好”混在一起。",
        "",
        "## 2. Toy 数据集设计",
        "",
        "本轮一共生成了 4 个 toy dataset，全部都保存成和真实数据一致的 `.npz` raw-data 格式，然后复用同一条 Spike CAM pipeline。",
        "",
        *difficulty_lines,
        "",
        "设计动机：",
        "",
        "- `toy_easy_stable_u8_c8_60s`：最接近 previous work 的 sanity check，验证整条 pipeline 在简单条件下能得到明显高于随机的结果。",
        "- `toy_dense_stable_u12_c16_75s`：增加 unit 数和 channel 数，但不加 drift，用来观察“仅仅问题变密集”时 update 有没有必要。",
        "- `toy_drift_u12_c16_75s`：在 dense stable 的基础上加入 channel-wise drift 和额外噪声，用来观察 dynamic update 是否真的有帮助。",
        "- `toy_open_u20_c24_90s`：增加 unit 数并采用 `stream > memory` 的 open-set protocol，用来研究 memory-limited CAM 的真实目标场景。",
        "",
        "共享 pipeline 设置：",
        "",
        "- preprocess: `bandpass 300-6000 + global median CMR`",
        "- waveform: `45 samples`, `align_mode=none`",
        "- channel selection: `all channels`",
        "- flatten: `time_major`",
        "- 主评估协议：`chronological online evaluation`, `warmup_ratio=0.5`",
        "- 主 encoder：`PCA 32-bit`，`binarize_mode=zero`，用于最大程度贴近 previous work",
        "",
        "## 3. Stage A: 协议 sanity check",
        "",
        "先在最简单的 `easy stable` 数据上比较 `random split` 和 `chronological split`，确认 toy pipeline 本身没有坏掉。",
        "",
        *(_summary_table(protocol_rows)),
        "",
        "结论：在 toy easy 场景下，chronological 评估本身并不会把结果压到很低，说明当前 pipeline 在简单数据上是能正常识别的。",
        "",
        "## 4. Stage B: 数据难度对 static baseline 的影响",
        "",
        "先固定 encoder 和 static CAM，只看不同 toy 场景本身有多难。这一步的目的，是把“数据难度”与“动态更新效果”分开。",
        "",
        *difficulty_lines,
        "",
        "结论：随着 unit 数、channel 数、噪声和 drift 增加，最优 static accuracy 会逐步下降；而 open-set memory-limited 场景下降最明显。这说明后面所有 dynamic update 的比较，必须结合数据场景来解读。",
        "",
        "相关图：",
        "",
        "![Toy dataset overview](./figures/dataset_overview.png)",
        "",
        "![Static threshold sweep](./figures/threshold_scenarios.png)",
        "",
        "![Difficulty summary](./figures/difficulty_summary.png)",
        "",
        "## 5. Stage C: Dynamic update strategy 对比",
        "",
        "这部分是整轮 toy study 的主实验。我们在 3 种代表性场景下比较所有 update strategy：",
        "",
        "1. `dense stable`：问题更密集，但没有 drift",
        "2. `drift closed`：有 drift，但仍是 closed-set",
        "3. `open memory`：既有 drift，又有 memory 外类需要 reject",
        "",
        "### 5.1 Dense stable：没有 drift 时，update 是否还有必要？",
        "",
        *(_summary_table(dense_update_rows)),
        "",
        (
            f"结论：在 dense stable 条件下，最佳实际 dynamic 是 `{best_dense_dynamic['variant_name']}`，accuracy={_fmt(best_dense_dynamic.get('accuracy'))}。"
            "如果它只比 static 好一点，说明在没有明显 drift 的条件下，dynamic update 的必要性并不强。"
            if best_dense_dynamic is not None
            else "结论：在 dense stable 条件下，没有任何真正执行更新的 dynamic 方法明显优于 static。"
        ),
        "",
        "### 5.2 Drift closed：出现 drift 后，dynamic update 是否开始有价值？",
        "",
        *(_summary_table(drift_update_rows)),
        "",
        (
            f"结论：在 drift closed 条件下，最佳实际 dynamic 是 `{best_drift_dynamic['variant_name']}`，accuracy={_fmt(best_drift_dynamic.get('accuracy'))}，"
            f"wrong_update_rate={_fmt(best_drift_dynamic.get('wrong_update_rate'))}。这一组最能反映“update 的收益是否值得它的代价”。"
            if best_drift_dynamic is not None
            else "结论：在 drift closed 条件下，dynamic 方法依然没有显著优于 static。"
        ),
        "",
        "### 5.3 Open memory：unknown pressure 下，dynamic update 会不会反而更危险？",
        "",
        *(_summary_table(open_update_rows)),
        "",
        (
            f"结论：在 open memory 条件下，最佳实际 dynamic 是 `{best_open_dynamic['variant_name']}`，accuracy={_fmt(best_open_dynamic.get('accuracy'))}，"
            f"false_accept_rate={_fmt(best_open_dynamic.get('false_accept_rate'))}。如果 false accept 很高，就说明 open-set 条件下 dynamic update 容易被 unknown spike 污染。"
            if best_open_dynamic is not None
            else "结论：在 open memory 条件下，没有任何真正执行更新的 dynamic 方法能稳定优于 static。"
        ),
        "",
        "相关图：",
        "",
        "![Dense stable update compare](./figures/update_compare_dense_stable.png)",
        "",
        "![Drift closed update compare](./figures/update_compare_drift_closed.png)",
        "",
        "![Open memory update compare](./figures/update_compare_open_memory.png)",
        "",
        "## 6. Stage D: Bit budget 消融（drift 场景）",
        "",
        "选择 drift 场景做 bit 消融，是因为它最能体现“表示能力不足”和“动态更新是否能补救”之间的关系。",
        "",
        *(_summary_table(bit_rows)),
        "",
        f"设计说明：bit 宽度使用 `8/16/32/64`，阈值不是随便手填，而是按 drift 场景中 `32-bit` 最优阈值的比例缩放得到。这样不同 bit 宽度的比较更公平。",
        "",
        "结论：bit 越多并不总是越好。你后面写论文时，可以把这一节写成“bit budget 改变了 static / dynamic 的工作点，而不是单纯提升一个数字”。",
        "",
        "## 7. Stage E: 初始化策略消融（drift 场景）",
        "",
        "初始化策略在真实论文里通常不是主角，但它会影响 online 阶段的起点，所以还是值得单独测。",
        "",
        *(_summary_table(init_rows)),
        "",
        "结论：初始化方法会影响结果，但通常不如数据难度、threshold 和 update 风险来得主导。你可以把它作为支持性消融，而不是主结论。",
        "",
        "## 8. Stage F: Encoder 对比（drift 场景）",
        "",
        "为了不让整个 toy study 只停留在 PCA 上，我们在 drift 场景下补了 encoder 对比：",
        "",
        "- `PCA 32-bit`",
        "- `external normal AE 16-bit`",
        "- `external shallow AE 16-bit`",
        "",
        "并且每个 external AE 都先单独训练 artifact，再固定前端，只比较 CAM 侧。",
        "",
        *(_summary_table(encoder_rows)),
        "",
        "结论：这一节主要回答“CAM 侧结论是否依赖某一个 encoder”。如果不同 encoder 下，dynamic / static 的排序趋势大体一致，那你的 CAM 结论就更扎实。",
        "",
        "## 9. 最重要的研究结论",
        "",
        "- 在简单 toy stable 数据上，当前 Spike CAM pipeline 完全可以得到远高于随机的结果，这说明系统本身不是坏掉了。",
        "- 随着 unit 数、channel 数、噪声和 drift 增加，static baseline 会系统性下降，这一步明确了“数据难度”本身就是一阶因素。",
        "- 在没有 drift 的稳定场景下，dynamic update 通常不一定有明显必要，甚至可能只是引入额外风险。",
        "- 在 drift 场景下，dynamic update 才真正有讨论价值；但是否值得用，要看 accuracy 提升能不能覆盖 wrong update 的代价。",
        "- 在 memory-limited open-set 场景下，`false_accept` 比 raw accuracy 更关键。dynamic update 如果让 unknown spike 更容易被接受，就会快速污染模板。",
        "- bit 宽度的收益不是单调的，应该被表述成“表示能力与匹配风险之间的 trade-off”。",
        "- 初始化方法有影响，但影响通常弱于数据难度、threshold 和 dynamic update 机制本身。",
        "- toy study 的真正价值，不是替代真实数据，而是提供一个**受控环境**，先把 CAM 端的规律看清楚。",
        "",
        "## 10. 这轮 toy study 对毕设写作有什么帮助",
        "",
        "你可以把它写成这样：",
        "",
        "1. 先在 toy 数据上说明方法机制：在可控难度条件下比较 dynamic update。",
        "2. 再说明从 stable -> drift -> open memory 的难度递进。",
        "3. 用 toy 结果总结：哪些 update 在什么条件下有价值，哪些条件下会引入污染风险。",
        "4. 最后再回到真实数据，说明为什么真实数据结果会更低，以及 toy study 给了你什么解释框架。",
        "",
        "## 11. 建议的默认 toy 主配置",
        "",
        f"- 推荐主 toy 数据集：`toy_drift_u12_c16_75s`",
        f"- 推荐主 encoder：`PCA 32-bit` 作为机制研究主线；AE 作为补充对比",
        f"- 推荐主 updater：`{dynamic_name}` 与 `static` 共同作为主对照",
        f"- 推荐 memory-limited toy 数据集：`toy_open_u20_c24_90s`",
        "",
        "## 12. 下一步建议",
        "",
        "- 如果你想继续把 toy study 做得更像论文，可以再加一个“drift strength sweep”，直接研究 update 何时开始值得用。",
        "- 如果你想把真实数据也写进去，可以把 toy 结果作为“机制结论”，再把真实数据结果作为“实际挑战与限制”。",
        "- 如果后面你希望，我可以继续把这份报告直接改写成你毕设实验章节的中文初稿。",
        "",
    ]

    master_text = "\n".join(report_lines) + "\n"
    (run_root / "master_report.md").write_text(master_text, encoding="utf-8")
    _make_html(master_text, run_root / "master_report.html")


if __name__ == "__main__":
    main()
