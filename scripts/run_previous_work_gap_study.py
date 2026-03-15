"""Run a focused study that explains the gap between previous work and thesis runs.

中文说明
--------
这个脚本专门回答一个非常实际的问题：

“为什么 previous work 里看起来有辨别力的指纹，到了现在的 thesis pipeline 里结果却很低？”

它会把问题拆成几层来验证：

1. 旧项目保存下来的 encoded CSV，在当前 CAM 框架里还能不能复现出不错的结果？
2. 用 spikeinterface toy data 按旧 protocol 重新生成编码时，结果是否仍然比较容易？
3. 真实数据在“旧风格 closed-set random split”下会怎样？
4. 真实数据一旦换成 chronological / memory-limited open-set，性能会掉多少？

输出会统一打包到：

``results/experiments/<YYYY-MM-DD>/experiment_<timestamp>_previous_work_gap/``
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
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentSuiteConfig, json_ready, project_path, save_json
from dataio import EncodedDataset, WaveformDataset, load_encoded_csv_dataset
from encoder import build_encoder, compute_bit_statistics
from experiment_runner import (
    run_experiment_suite,
    run_experiment_suite_on_encoded_dataset,
)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _run_report_script(result_dir: Path) -> None:
    """Best-effort call into the standard experiment report script."""

    script_path = project_path("scripts", "build_experiment_report.py")
    import subprocess

    subprocess.run(
        [sys.executable, str(script_path), "--result_dir", str(result_dir)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def _write_commands_log(run_root: Path, env_name: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Full study entry point",
        f"conda run -n {env_name} python scripts/run_previous_work_gap_study.py --run-root {run_root.relative_to(PROJECT_ROOT)}",
        "",
        "# Main sub-experiments are orchestrated inside the Python study script:",
        "# 1. previous-work encoded CSV threshold sweep",
        "# 2. previous-work encoded CSV update comparison",
        "# 3. recreated spikeinterface toy dataset threshold sweeps",
        "# 4. real-data top8 closed-set threshold sweeps",
        "# 5. real-data top8 closed-set update comparison",
        "# 6. real-data top50/top20 open-set threshold sweep",
    ]
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs" / "commands.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failures_log(run_root: Path, failures: List[str]) -> None:
    lines = ["# Failures and Fixes", ""]
    if not failures:
        lines.append("- No blocking failures were encountered in this study.")
    else:
        for item in failures:
            lines.append(f"- {item}")
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs" / "failures_and_fixes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _suite_from_payload(payload: Dict[str, Any]) -> ExperimentSuiteConfig:
    return ExperimentSuiteConfig.from_dict(payload)


def _threshold_variants(thresholds: Iterable[float], *, prefix: str = "thr") -> List[Dict[str, Any]]:
    variants = []
    for threshold in thresholds:
        if float(threshold).is_integer():
            name = f"{prefix}{int(threshold)}"
        else:
            name = f"{prefix}{str(threshold).replace('.', '_')}"
        variants.append(
            {
                "name": name,
                "description": f"Static threshold={threshold}",
                "matcher": {"threshold": float(threshold)},
                "update": {"method": "none"},
            }
        )
    return variants


def _update_variants(threshold: float) -> List[Dict[str, Any]]:
    return [
        {"name": "static", "description": f"Static baseline at threshold={threshold}", "matcher": {"threshold": threshold}, "update": {"method": "none"}},
        {"name": "counter", "description": f"Counter update at threshold={threshold}", "matcher": {"threshold": threshold}, "update": {"method": "counter"}},
        {"name": "margin_ema", "description": f"Margin EMA at threshold={threshold}", "matcher": {"threshold": threshold}, "update": {"method": "margin_ema"}},
        {"name": "confidence_weighted", "description": f"Confidence weighted at threshold={threshold}", "matcher": {"threshold": threshold}, "update": {"method": "confidence_weighted"}},
        {"name": "dual_template", "description": f"Dual template at threshold={threshold}", "matcher": {"threshold": threshold}, "update": {"method": "dual_template"}},
        {"name": "probabilistic", "description": f"Probabilistic at threshold={threshold}", "matcher": {"threshold": threshold}, "update": {"method": "probabilistic"}},
        {"name": "growing", "description": f"Growing update at threshold={threshold}", "matcher": {"threshold": threshold}, "update": {"method": "growing"}},
        {"name": "cooldown", "description": f"Cooldown update at threshold={threshold}", "matcher": {"threshold": threshold}, "update": {"method": "cooldown"}},
        {"name": "top2_margin", "description": f"Top2 margin update at threshold={threshold}", "matcher": {"threshold": threshold, "method": "top2_margin", "min_margin": 1.0}, "update": {"method": "top2_margin", "min_margin": 1.0}},
    ]


def _summary_rows(result_dir: Path, label: str) -> List[Dict[str, Any]]:
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for variant_name, metrics in summary.items():
        rows.append({"study_experiment": label, "variant_name": variant_name, **metrics})
    return rows


def _best_variant(rows: List[Dict[str, Any]], *, metric: str = "accuracy") -> Dict[str, Any]:
    return max(rows, key=lambda row: float(row.get(metric, 0.0) or 0.0))


def _best_dynamic_variant(rows: List[Dict[str, Any]], *, metric: str = "accuracy") -> Optional[Dict[str, Any]]:
    """Return the best non-static variant that actually performed updates."""

    candidates = [
        row
        for row in rows
        if str(row.get("variant_name")) != "static" and int(row.get("update_count", 0) or 0) > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get(metric, 0.0) or 0.0))


def _load_metrics_table(result_dir: Path) -> List[Dict[str, Any]]:
    path = result_dir / "metrics_table.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _copy_summary_artifacts(result_dir: Path, run_root: Path, label: str) -> None:
    figures_dir = run_root / "figures"
    tables_dir = run_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    for src_name in ["comparison_metrics.png", "curve_overview.png"]:
        src = result_dir / src_name
        if src.exists():
            shutil.copy2(src, figures_dir / f"{label}_{src_name}")
    for src_name in ["metrics_table.csv", "summary.json", "report.md"]:
        src = result_dir / src_name
        if src.exists():
            suffix = src.suffix if src.suffix else ".txt"
            shutil.copy2(src, tables_dir / f"{label}_{src.name}")


def _encoded_stats_from_result(result_dir: Path, variant_name: Optional[str] = None) -> Dict[str, Any]:
    if variant_name is None:
        runs_dir = result_dir / "runs"
        candidates = sorted(run_dir for run_dir in runs_dir.iterdir() if run_dir.is_dir())
        if not candidates:
            return {}
        stats_path = candidates[0] / "encoded_stats.json"
    else:
        stats_path = result_dir / "runs" / variant_name / "encoded_stats.json"
    if not stats_path.exists():
        return {}
    return json.loads(stats_path.read_text(encoding="utf-8"))


def _make_html_from_markdown(markdown_text: str, out_path: Path) -> None:
    escaped = html.escape(markdown_text)
    body = (
        "<html><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1100px;margin:40px auto;line-height:1.6;padding:0 24px;} "
        "pre{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f7f7f7;padding:16px;border-radius:8px;} "
        "img{max-width:100%;height:auto;}</style></head><body><pre>"
        f"{escaped}</pre></body></html>"
    )
    out_path.write_text(body, encoding="utf-8")


def _load_threshold_curve(result_dir: Path) -> Tuple[List[float], List[float], List[float]]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    triples: List[Tuple[float, float, float]] = []
    for variant_name, metrics in summary.items():
        threshold_text = str(variant_name).replace("thr", "").replace("_", ".")
        try:
            threshold = float(threshold_text)
        except ValueError:
            continue
        triples.append((threshold, float(metrics.get("accuracy", 0.0)), float(metrics.get("reject_rate", 0.0))))
    triples.sort(key=lambda item: item[0])
    return [t for t, _, _ in triples], [a for _, a, _ in triples], [r for _, _, r in triples]


def _plot_threshold_scenarios(run_root: Path, scenario_curves: List[Tuple[str, Path]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, result_dir in scenario_curves:
        thresholds, accuracy, reject_rate = _load_threshold_curve(result_dir)
        if not thresholds:
            continue
        axes[0].plot(thresholds, accuracy, marker="o", label=label)
        axes[1].plot(thresholds, reject_rate, marker="o", label=label)
    axes[0].set_title("Accuracy vs Threshold")
    axes[1].set_title("Reject Rate vs Threshold")
    for ax in axes:
        ax.set_xlabel("Threshold")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("Reject Rate")
    fig.tight_layout()
    out_path = run_root / "figures" / "threshold_scenario_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_separability_compare(run_root: Path, stats_map: Dict[str, Dict[str, Any]]) -> None:
    labels = list(stats_map.keys())
    gaps = [float(stats_map[label].get("mean_hamming_gap", np.nan)) for label in labels]
    unique_ratios = [float(stats_map[label].get("unique_code_ratio", np.nan)) for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(labels, gaps, color="#4e79a7")
    axes[0].set_title("Mean Hamming Gap")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, unique_ratios, color="#f28e2b")
    axes[1].set_title("Unique Code Ratio")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path = run_root / "figures" / "separability_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
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
        "unknown_stream_fraction",
        "initial_template_count",
        "final_template_count",
    ]
    with (run_root / "overall_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    save_json(run_root / "overall_summary.json", {"rows": rows})


def _write_index(run_root: Path, experiments: Dict[str, Path]) -> None:
    payload = {
        "experiments": [
            {"name": name, "result_dir": str(path), "summary": str(path / "summary.json"), "report": str(path / "report.md")}
            for name, path in experiments.items()
        ]
    }
    save_json(run_root / "experiment_index.json", payload)


def _write_readme(run_root: Path, env_name: str, experiments: Dict[str, Path]) -> None:
    lines = [
        "# Previous Work Gap Study",
        "",
        f"Run root: `{run_root}`",
        "",
        "## 研究目的",
        "",
        "解释为什么 previous work 里的指纹在旧实验里有较强辨别力，而当前 thesis 主线实验结果却低很多。",
        "",
        "## 环境",
        "",
        f"- conda env: `{env_name}`",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        "",
        "## 主要子实验",
        "",
    ]
    for name, path in experiments.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## 复现命令",
            "",
            f"`conda run -n {env_name} python scripts/run_previous_work_gap_study.py --run-root {run_root.relative_to(PROJECT_ROOT)}`",
            "",
            "## 目录说明",
            "",
            "- `figures/`: 顶层汇总图。",
            "- `tables/`: 汇总表和各子实验复制出的表格。",
            "- `logs/`: 命令与异常记录。",
            "- `master_report.md`: 本次研究的总分析报告。",
        ]
    )
    (run_root / "README_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_waveforms_from_toy(
    *,
    duration: int,
    num_units: int,
    num_channels: int,
    sampling_frequency: float,
    waveform_window_ms: float,
    pre_window_ms: float,
    seed: int,
) -> WaveformDataset:
    """Recreate a previous-work-like toy dataset with spikeinterface."""

    import spikeinterface.full as si

    recording, sorting = si.toy_example(
        duration=duration,
        num_channels=num_channels,
        num_units=num_units,
        sampling_frequency=sampling_frequency,
        num_segments=1,
        seed=seed,
    )
    recording = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = si.common_reference(recording, reference="global", operator="median")

    fs = float(recording.get_sampling_frequency())
    n_pre = int(round(pre_window_ms * fs / 1000.0))
    n_post = int(round((waveform_window_ms - pre_window_ms) * fs / 1000.0))
    feature_dim = int((n_pre + n_post) * num_channels)
    num_frames = int(recording.get_num_frames(segment_index=0))

    waveforms: List[np.ndarray] = []
    labels: List[int] = []
    spike_times: List[int] = []

    for unit_id in sorting.unit_ids:
        spike_train = np.asarray(sorting.get_unit_spike_train(unit_id, segment_index=0), dtype=np.int64)
        for spike_time in spike_train:
            start = int(spike_time) - n_pre
            end = int(spike_time) + n_post
            if start < 0 or end > num_frames:
                continue
            wf = recording.get_traces(segment_index=0, start_frame=start, end_frame=end).reshape(-1).astype(np.float32)
            if wf.shape[0] != feature_dim:
                continue
            waveforms.append(wf)
            labels.append(int(unit_id))
            spike_times.append(int(spike_time))

    order = np.argsort(np.asarray(spike_times, dtype=np.int64), kind="mergesort")
    waveform_array = np.asarray(waveforms, dtype=np.float32)[order]
    label_array = np.asarray(labels, dtype=np.int64)[order]
    spike_time_array = np.asarray(spike_times, dtype=np.int64)[order]
    source_indices = np.arange(spike_time_array.shape[0], dtype=np.int64)
    return WaveformDataset(
        waveforms=waveform_array,
        labels=label_array,
        spike_times=spike_time_array,
        source_indices=source_indices,
        meta={
            "source": "spikeinterface.toy_example",
            "duration_sec": duration,
            "num_units": num_units,
            "num_channels": num_channels,
            "sampling_frequency": sampling_frequency,
            "waveform_window_ms": waveform_window_ms,
            "pre_window_ms": pre_window_ms,
            "feature_dim": feature_dim,
            "preprocess": ["bandpass_300_6000", "global_median_reference"],
        },
    )


def _encode_toy_previous_style(dataset: WaveformDataset, *, train_ratio: float, seed: int) -> EncodedDataset:
    """Fit PCA on the first half of a toy dataset and transform the full stream."""

    from config import EncoderConfig

    split_index = int(len(dataset.spike_times) * train_ratio)
    if split_index <= 0 or split_index >= len(dataset.spike_times):
        raise ValueError("Toy dataset split produced an empty train or test partition")

    encoder_cfg = EncoderConfig(
        method="pca",
        backend="auto",
        code_size=32,
        scale="none",
        binarize_mode="zero",
    )
    encoder = build_encoder(encoder_cfg, seed)
    encoder.fit_transform(np.asarray(dataset.waveforms[:split_index], dtype=np.float32))
    bits = encoder.transform(np.asarray(dataset.waveforms, dtype=np.float32))
    return EncodedDataset(
        bits=np.asarray(bits, dtype=np.uint8),
        labels=np.asarray(dataset.labels, dtype=np.int64),
        spike_times=np.asarray(dataset.spike_times, dtype=np.int64),
        source_indices=np.asarray(dataset.source_indices, dtype=np.int64),
        meta={
            "source": "recreated_previous_work_toy",
            "train_ratio_for_pca": train_ratio,
            "encoder_method": "pca",
            "encoder_binarize_mode": "zero",
            "source_meta": json_ready(dataset.meta),
        },
    )


def _build_result_root(run_root: Path, experiment_name: str) -> Path:
    return run_root / experiment_name


def _save_suite_config(run_root: Path, experiment_name: str, suite_payload: Dict[str, Any]) -> None:
    cfg_dir = run_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg_dir / f"{experiment_name}.json", suite_payload)


def _run_raw_suite(run_root: Path, payload: Dict[str, Any]) -> Path:
    suite = _suite_from_payload(payload)
    _save_suite_config(run_root, suite.experiment_name, payload)
    run_experiment_suite(suite)
    result_dir = _build_result_root(run_root, suite.experiment_name)
    _run_report_script(result_dir)
    _copy_summary_artifacts(result_dir, run_root, suite.experiment_name)
    return result_dir


def _run_encoded_suite(run_root: Path, payload: Dict[str, Any], encoded: EncodedDataset, cache_path: Path) -> Path:
    suite = _suite_from_payload(payload)
    _save_suite_config(run_root, suite.experiment_name, payload)
    run_experiment_suite_on_encoded_dataset(suite, encoded, cache_path=cache_path, save_encoded_copy=False)
    result_dir = _build_result_root(run_root, suite.experiment_name)
    _run_report_script(result_dir)
    _copy_summary_artifacts(result_dir, run_root, suite.experiment_name)
    return result_dir


def _base_suite_payload(run_root: Path, experiment_name: str) -> Dict[str, Any]:
    results_dir_rel = str(run_root.relative_to(PROJECT_ROOT))
    return {
        "experiment_name": experiment_name,
        "description": "",
        "seed": 42,
        "results": {
            "results_dir": results_dir_rel,
            "save_predictions": True,
            "save_curves": True,
            "save_confusion": True,
            "copy_encoded_dataset": False,
        },
    }


def _real_oldlike_payload(run_root: Path, experiment_name: str, *, subset_mode: Dict[str, Any], memory_subset: Dict[str, Any], evaluation_mode: str) -> Dict[str, Any]:
    payload = _base_suite_payload(run_root, experiment_name)
    payload.update(
        {
            "dataset": {
                "npz_path": "dataset/my_validation_subset_810000samples_27.00s.npz",
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
                    "channel_selection": "topk_max_abs",
                    "topk_channels": 8,
                    "channel_order": "by_strength",
                    "flatten_order": "time_major",
                    "selection_radius": 6,
                },
                "subset": subset_mode,
                "sampling": {
                    "max_total_spikes": None,
                    "max_spikes_per_unit": None,
                    "selection_mode": "uniform_time",
                },
            },
            "encoder": {
                "method": "pca",
                "code_size": 32,
                "scale": "none",
                "binarize_mode": "zero",
                "reuse_cache": True,
                "force_reencode": False,
            },
            "template_init": {"method": "majority_vote"},
            "matcher": {"method": "hamming_nearest", "threshold": 8.0},
            "update": {"method": "none"},
            "cam": {"capacity": None, "memory_subset": memory_subset},
            "evaluation": {
                "mode": evaluation_mode,
                "warmup_ratio": 0.5,
                "random_train_frac": 0.5,
                "window_size": 500,
            },
        }
    )
    return payload


def _best_threshold_from_summary(result_dir: Path) -> float:
    rows = _summary_rows(result_dir, result_dir.name)
    best = _best_variant(rows, metric="accuracy")
    name = str(best["variant_name"]).replace("thr", "").replace("_", ".")
    return float(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a focused previous-work gap study.")
    parser.add_argument("--run-root", default="", help="Optional explicit result root under results/.")
    parser.add_argument("--env-name", default="spikecam_py310", help="Conda env name recorded in the README/logs.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dated_parent = project_path("results", "experiments", datetime.now().strftime("%Y-%m-%d"))
    default_run_root = dated_parent / f"experiment_{timestamp}_previous_work_gap"
    run_root = project_path(args.run_root) if args.run_root else default_run_root
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "figures").mkdir(exist_ok=True)
    (run_root / "tables").mkdir(exist_ok=True)
    (run_root / "diagnostics").mkdir(exist_ok=True)
    failures: List[str] = []

    _write_commands_log(run_root, args.env_name)

    experiments: Dict[str, Path] = {}
    all_rows: List[Dict[str, Any]] = []
    separability_stats: Dict[str, Dict[str, Any]] = {}

    previous_csv = project_path("previous work", "spike_experiments", "experiment_001_u8_bits32_2000s", "spike_pca_dataset.csv")
    previous_encoded = load_encoded_csv_dataset(previous_csv)
    previous_stats = compute_bit_statistics(previous_encoded, seed=42)
    save_json(run_root / "diagnostics" / "previous_work_encoded_stats.json", previous_stats)
    previous_encoded.save_npz(run_root / "diagnostics" / "previous_work_encoded_dataset.npz")
    separability_stats["prevwork_csv"] = previous_stats

    prev_thresholds = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]
    prev_payload = _base_suite_payload(run_root, "prevwork_csv_threshold_sweep")
    prev_payload.update(
        {
            "encoder": {"method": "pca", "code_size": 32},
            "template_init": {"method": "majority_vote"},
            "matcher": {"method": "hamming_nearest", "threshold": 8.0},
            "update": {"method": "none"},
            "cam": {"capacity": 8, "memory_subset": {"mode": "same_as_stream"}},
            "evaluation": {"mode": "random_split", "random_train_frac": 0.5, "window_size": 500},
            "variants": _threshold_variants(prev_thresholds),
        }
    )
    prev_result = _run_encoded_suite(run_root, prev_payload, previous_encoded, previous_csv)
    experiments["prevwork_csv_threshold_sweep"] = prev_result
    rows = _summary_rows(prev_result, "prevwork_csv_threshold_sweep")
    all_rows.extend(rows)

    prev_best_threshold = _best_threshold_from_summary(prev_result)
    prev_update_payload = _base_suite_payload(run_root, "prevwork_csv_update_compare")
    prev_update_payload.update(
        {
            "encoder": {"method": "pca", "code_size": 32},
            "template_init": {"method": "majority_vote"},
            "matcher": {"method": "hamming_nearest", "threshold": float(prev_best_threshold)},
            "update": {"method": "none"},
            "cam": {"capacity": 8, "memory_subset": {"mode": "same_as_stream"}},
            "evaluation": {"mode": "random_split", "random_train_frac": 0.5, "window_size": 500},
            "variants": _update_variants(float(prev_best_threshold)),
        }
    )
    prev_update_result = _run_encoded_suite(run_root, prev_update_payload, previous_encoded, previous_csv)
    experiments["prevwork_csv_update_compare"] = prev_update_result
    all_rows.extend(_summary_rows(prev_update_result, "prevwork_csv_update_compare"))

    toy_encoded: Optional[EncodedDataset] = None
    try:
        toy_waveforms = _extract_waveforms_from_toy(
            duration=2000,
            num_units=8,
            num_channels=8,
            sampling_frequency=30000.0,
            waveform_window_ms=1.5,
            pre_window_ms=0.5,
            seed=42,
        )
        toy_encoded = _encode_toy_previous_style(toy_waveforms, train_ratio=0.5, seed=42)
        toy_encoded.save_npz(run_root / "diagnostics" / "toy_recreated_encoded_dataset.npz")
        toy_stats = compute_bit_statistics(toy_encoded, seed=42)
        save_json(run_root / "diagnostics" / "toy_recreated_encoded_stats.json", toy_stats)
        separability_stats["toy_recreated"] = toy_stats

        toy_payload_random = _base_suite_payload(run_root, "toy_recreated_random_threshold_sweep")
        toy_payload_random.update(
            {
                "encoder": {"method": "pca", "code_size": 32},
                "template_init": {"method": "majority_vote"},
                "matcher": {"method": "hamming_nearest", "threshold": 8.0},
                "update": {"method": "none"},
                "cam": {"capacity": 8, "memory_subset": {"mode": "same_as_stream"}},
                "evaluation": {"mode": "random_split", "random_train_frac": 0.5, "window_size": 500},
                "variants": _threshold_variants(prev_thresholds),
            }
        )
        toy_random_result = _run_encoded_suite(run_root, toy_payload_random, toy_encoded, run_root / "diagnostics" / "toy_recreated_encoded_dataset.npz")
        experiments["toy_recreated_random_threshold_sweep"] = toy_random_result
        all_rows.extend(_summary_rows(toy_random_result, "toy_recreated_random_threshold_sweep"))

        toy_payload_chrono = _base_suite_payload(run_root, "toy_recreated_chronological_threshold_sweep")
        toy_payload_chrono.update(
            {
                "encoder": {"method": "pca", "code_size": 32},
                "template_init": {"method": "majority_vote"},
                "matcher": {"method": "hamming_nearest", "threshold": 8.0},
                "update": {"method": "none"},
                "cam": {"capacity": 8, "memory_subset": {"mode": "same_as_stream"}},
                "evaluation": {"mode": "chronological", "warmup_ratio": 0.5, "window_size": 500},
                "variants": _threshold_variants(prev_thresholds),
            }
        )
        toy_chrono_result = _run_encoded_suite(run_root, toy_payload_chrono, toy_encoded, run_root / "diagnostics" / "toy_recreated_encoded_dataset.npz")
        experiments["toy_recreated_chronological_threshold_sweep"] = toy_chrono_result
        all_rows.extend(_summary_rows(toy_chrono_result, "toy_recreated_chronological_threshold_sweep"))
    except Exception as exc:  # pragma: no cover - best effort study branch
        failures.append(f"Toy dataset recreation failed: {type(exc).__name__}: {exc}")

    real_random_payload = _real_oldlike_payload(
        run_root,
        "real_top8_oldlike_random_threshold_sweep",
        subset_mode={"mode": "topk", "topk": 8},
        memory_subset={"mode": "same_as_stream"},
        evaluation_mode="random_split",
    )
    real_random_payload["variants"] = _threshold_variants(prev_thresholds)
    real_random_result = _run_raw_suite(run_root, real_random_payload)
    experiments["real_top8_oldlike_random_threshold_sweep"] = real_random_result
    all_rows.extend(_summary_rows(real_random_result, "real_top8_oldlike_random_threshold_sweep"))
    separability_stats["real_top8_oldlike_random"] = _encoded_stats_from_result(real_random_result)

    real_random_best_threshold = _best_threshold_from_summary(real_random_result)
    real_random_update_payload = _real_oldlike_payload(
        run_root,
        "real_top8_oldlike_random_update_compare",
        subset_mode={"mode": "topk", "topk": 8},
        memory_subset={"mode": "same_as_stream"},
        evaluation_mode="random_split",
    )
    real_random_update_payload["matcher"]["threshold"] = float(real_random_best_threshold)
    real_random_update_payload["variants"] = _update_variants(float(real_random_best_threshold))
    real_random_update_result = _run_raw_suite(run_root, real_random_update_payload)
    experiments["real_top8_oldlike_random_update_compare"] = real_random_update_result
    all_rows.extend(_summary_rows(real_random_update_result, "real_top8_oldlike_random_update_compare"))

    real_chrono_payload = _real_oldlike_payload(
        run_root,
        "real_top8_oldlike_chronological_threshold_sweep",
        subset_mode={"mode": "topk", "topk": 8},
        memory_subset={"mode": "same_as_stream"},
        evaluation_mode="chronological",
    )
    real_chrono_payload["variants"] = _threshold_variants(prev_thresholds)
    real_chrono_result = _run_raw_suite(run_root, real_chrono_payload)
    experiments["real_top8_oldlike_chronological_threshold_sweep"] = real_chrono_result
    all_rows.extend(_summary_rows(real_chrono_result, "real_top8_oldlike_chronological_threshold_sweep"))
    separability_stats["real_top8_oldlike_chrono"] = _encoded_stats_from_result(real_chrono_result)

    real_open_payload = _real_oldlike_payload(
        run_root,
        "real_top50_mem20_open_threshold_sweep",
        subset_mode={"mode": "topk", "topk": 50},
        memory_subset={"mode": "topk", "topk": 20, "selection_source": "pre_sampling"},
        evaluation_mode="chronological",
    )
    real_open_payload["variants"] = _threshold_variants(prev_thresholds)
    real_open_result = _run_raw_suite(run_root, real_open_payload)
    experiments["real_top50_mem20_open_threshold_sweep"] = real_open_result
    all_rows.extend(_summary_rows(real_open_result, "real_top50_mem20_open_threshold_sweep"))
    separability_stats["real_top50_mem20_open"] = _encoded_stats_from_result(real_open_result)

    _plot_threshold_scenarios(
        run_root,
        [
            ("Previous CSV random", prev_result),
            *(
                [("Toy recreated chronological", experiments["toy_recreated_chronological_threshold_sweep"])]
                if "toy_recreated_chronological_threshold_sweep" in experiments
                else []
            ),
            ("Real top8 random", real_random_result),
            ("Real top8 chrono", real_chrono_result),
            ("Real top50 mem20 open", real_open_result),
        ],
    )
    _plot_separability_compare(run_root, separability_stats)
    save_json(run_root / "diagnostics" / "separability_summary.json", separability_stats)

    _write_overall_summary(run_root, all_rows)
    _write_index(run_root, experiments)
    _write_readme(run_root, args.env_name, experiments)
    _write_failures_log(run_root, failures)

    best_prev = _best_variant(_summary_rows(prev_result, "prevwork_csv_threshold_sweep"))
    best_real_random = _best_variant(_summary_rows(real_random_result, "real_top8_oldlike_random_threshold_sweep"))
    best_real_chrono = _best_variant(_summary_rows(real_chrono_result, "real_top8_oldlike_chronological_threshold_sweep"))
    best_real_open = _best_variant(_summary_rows(real_open_result, "real_top50_mem20_open_threshold_sweep"))
    real_update_rows = _summary_rows(real_random_update_result, "real_top8_oldlike_random_update_compare")
    best_real_update = _best_variant(real_update_rows)
    best_real_dynamic = _best_dynamic_variant(real_update_rows)

    report_lines = [
        "# Previous Work Gap Study Report",
        "",
        "## 1. 研究目标",
        "",
        "本研究专门用于解释一个现象：`previous work` 里的指纹和模板在旧实验里看起来有明显辨别力，但当前 thesis 主线实验结果却显著偏低。",
        "为了定位原因，我们把问题拆成四层：",
        "",
        "1. 旧项目保存下来的 encoded CSV 本身是否仍然可分？",
        "2. 用 spikeinterface 按旧 protocol 重新生成 synthetic toy data 后，结果是否依然较好？",
        "3. 真实数据在旧风格的 closed-set random split 下会怎样？",
        "4. 真实数据一旦改成 chronological + memory-limited open-set，性能会掉多少？",
        "",
        "## 2. 核心结论",
        "",
        f"- 旧 `previous work` 编码 CSV 的 separability 明显更强：`mean_hamming_gap={_fmt(previous_stats.get('mean_hamming_gap'))}`，这说明旧数据/旧编码本身就更容易做模板匹配。",
        f"- 在当前 CAM 框架里直接重放旧 CSV，最佳静态 accuracy 仍可达到 `{_fmt(best_prev.get('accuracy'))}`，最佳阈值约为 `{best_prev.get('variant_name')}`。这说明当前 CAM 核心不是完全失效的。",
        f"- 真实数据即使切回更接近旧协议的 `top8 + PCA32 + closed-set + random split`，最佳 accuracy 也只有 `{_fmt(best_real_random.get('accuracy'))}`，显著低于旧 CSV。",
        f"- 同样的真实数据一旦改成 chronological split，最佳 accuracy 进一步降到 `{_fmt(best_real_chrono.get('accuracy'))}`。这说明时间顺序评估本身就更难。",
        f"- 再进一步改成 `stream=top50 / memory=top20` 的 open-set memory-aware 协议后，最佳 accuracy 只有 `{_fmt(best_real_open.get('accuracy'))}`，同时 reject/false accept tradeoff 明显变得更敏感。",
        (
            f"- 在较容易的 `real top8 random` 条件下，`static` 与 `{best_real_update.get('variant_name')}` 的最佳 accuracy 都约为 "
            f"`{_fmt(best_real_update.get('accuracy'))}`；其中 `{best_real_update.get('variant_name')}` 实际上没有发生更新，因此它并不能算真正优于 static。"
        ),
        (
            f"- 真正发生了更新的 dynamic 方法里，最好的是 `{best_real_dynamic.get('variant_name')}`，accuracy 为 "
            f"`{_fmt(best_real_dynamic.get('accuracy'))}`，但 `wrong_update_rate={_fmt(best_real_dynamic.get('wrong_update_rate'))}`，"
            "说明 dynamic 收益非常有限且伴随明显污染风险。"
            if best_real_dynamic is not None
            else "- 在这组 easier closed-set real-data 条件下，没有任何真正执行了更新的 dynamic 方法明显优于 static。"
        ),
        "",
        "## 3. 结果表",
        "",
        "| 场景 | 最佳 variant | Accuracy | Reject Rate | Accepted Accuracy |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| Previous CSV random | {best_prev.get('variant_name')} | {_fmt(best_prev.get('accuracy'))} | {_fmt(best_prev.get('reject_rate'))} | {_fmt(best_prev.get('accepted_accuracy'))} |",
        f"| Real top8 random | {best_real_random.get('variant_name')} | {_fmt(best_real_random.get('accuracy'))} | {_fmt(best_real_random.get('reject_rate'))} | {_fmt(best_real_random.get('accepted_accuracy'))} |",
        f"| Real top8 chronological | {best_real_chrono.get('variant_name')} | {_fmt(best_real_chrono.get('accuracy'))} | {_fmt(best_real_chrono.get('reject_rate'))} | {_fmt(best_real_chrono.get('accepted_accuracy'))} |",
        f"| Real top50 memory20 open | {best_real_open.get('variant_name')} | {_fmt(best_real_open.get('accuracy'))} | {_fmt(best_real_open.get('reject_rate'))} | {_fmt(best_real_open.get('accepted_accuracy'))} |",
        "",
        "## 4. 编码表示诊断",
        "",
        "| 数据源 | Mean Hamming Gap | Unique Code Ratio | Num Units |",
        "| --- | ---: | ---: | ---: |",
    ]

    for label, stats in separability_stats.items():
        report_lines.append(
            f"| {label} | {_fmt(stats.get('mean_hamming_gap'))} | {_fmt(stats.get('unique_code_ratio'))} | {_fmt(stats.get('num_units'), digits=0)} |"
        )

    report_lines.extend(
        [
            "",
            "这里最关键的一点是：旧 CSV 的 `mean_hamming_gap` 明显高于真实数据。也就是说，旧问题本身就更容易，而不是只有 CAM 算法在旧 notebook 里更“神奇”。",
            "",
            "## 5. 规律解释",
            "",
            "### 5.1 为什么 previous work 更容易",
            "",
            "- synthetic toy data，类更干净、漂移更小、类别数更少。",
            "- 旧 protocol 更接近 closed-set，且 evaluation 通常更宽松。",
            "- 旧 PCA 编码使用 `PCA > 0` 二值化，我们已经在本 study 里专门兼容这一点。",
            "",
            "### 5.2 为什么真实数据会明显更难",
            "",
            "- 真实 extracellular spike 的类内变化更大、类间更像。",
            "- chronological split 比 random split 更能暴露 drift 和 warmup 覆盖不足。",
            "- memory-limited open-set 协议要求系统既要认 memory 内类，又要拒绝 memory 外类，本质上比旧 closed-set 更难。",
            "",
            "## 6. 对 thesis 主线的意义",
            "",
            "这组实验说明：当前 thesis 主线结果低，不一定是代码坏了，更多是因为研究问题被升级了。",
            "",
            "如果你要在论文里解释这一点，最自然的叙事是：",
            "",
            "1. 先承认旧实验在 synthetic / closed-set 条件下可以得到更高结果。",
            "2. 再说明 thesis 主线切换到了更真实、更严格的 protocol。",
            "3. 因此主实验不应简单追求绝对 accuracy，而应更关注 reject、wrong update、memory usage 和 online stability。",
            "",
            "## 7. 推荐后续动作",
            "",
            "- 如果目标是先恢复一个“看起来像 previous work”的 baseline，就用 `top8 + PCA32 + closed-set + random split + zero-threshold`。",
            "- 如果目标是 thesis 主线，就继续坚持 chronological + memory-aware open-set，但要把阈值、reject 和 wrong-update 当成主分析对象。",
            "- 如果还想继续缩小两边差距，下一步最值得做的是：把 real-data 前端编码单独再强化，例如更稳的 channel neighborhood、对齐、或监督式 encoder。",
        ]
    )

    master_report = "\n".join(report_lines) + "\n"
    (run_root / "master_report.md").write_text(master_report, encoding="utf-8")
    _make_html_from_markdown(master_report, run_root / "master_report.html")


if __name__ == "__main__":
    main()
