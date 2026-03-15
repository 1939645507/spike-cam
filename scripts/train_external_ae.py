"""Train or resume an external Autoencoders-in-Spike-Sorting artifact.

中文说明
--------
这个脚本的目标是把 external AE 的使用流程正式化：

1. 先从 raw dataset 提取 waveform
2. 用 `Autoencoders-in-Spike-Sorting` 训练 AE
3. 把权重、threshold、scale 参数保存成 artifact
4. 后续 CAM 实验只加载 artifact 做 encode，不再反复训练

推荐用法：

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/train_external_ae.py \
  --config configs/cam_compare_main.json \
  --artifact-dir results/artifacts/external_ae_artifacts/fullset_ae16_normal \
  --subset-mode all \
  --epochs 8
```
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SubsetConfig, load_config, resolve_path, save_json
from dataio import load_waveform_dataset
from encoder import train_external_autoencoder_artifact


def _pick_base_config(config_path: str, variant_name: Optional[str]):
    """Load one config suite and pick a concrete base config for training."""

    from config import expand_variants

    suite = load_config(config_path)
    configs = expand_variants(suite)
    if variant_name is None:
        return configs[0]
    for cfg in configs:
        if cfg.variant_name == variant_name:
            return cfg
    raise ValueError(f"Variant not found: {variant_name}")


def main() -> None:
    """Train one reusable external AE artifact from waveform data."""

    parser = argparse.ArgumentParser(description="Train or resume a reusable external AE artifact.")
    parser.add_argument("--config", default="configs/cam_compare_main.json", help="Base experiment config used for waveform extraction and AE hyperparameters.")
    parser.add_argument("--variant", default=None, help="Optional variant name to inherit settings from.")
    parser.add_argument("--artifact-dir", required=True, help="Directory where the trained artifact will be saved.")
    parser.add_argument("--subset-mode", default="all", choices=["all", "topk", "min_count"], help="Label subset rule used only for AE training data.")
    parser.add_argument("--topk", type=int, default=None, help="Used when --subset-mode=topk.")
    parser.add_argument("--min-count", type=int, default=None, help="Used when --subset-mode=min_count.")
    parser.add_argument("--max-spikes-per-unit", type=int, default=None, help="Optional per-unit cap for training speed control.")
    parser.add_argument("--epochs", type=int, default=None, help="Override AE epoch count.")
    parser.add_argument("--code-size", type=int, default=None, help="Override AE code size.")
    parser.add_argument("--ae-type", default=None, help="Override ae_type, e.g. normal/shallow/contractive.")
    parser.add_argument("--scale", default=None, help="Override scale; for external AE we recommend minmax.")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing artifact if present.")
    args = parser.parse_args()

    config = _pick_base_config(args.config, args.variant)
    config.encoder.backend = "external"
    config.encoder.artifact_path = args.artifact_dir
    config.encoder.use_artifact = True
    config.encoder.save_artifact = True
    config.encoder.force_retrain_artifact = True

    if args.epochs is not None:
        config.encoder.epochs = int(args.epochs)
    if args.code_size is not None:
        config.encoder.code_size = int(args.code_size)
    if args.ae_type is not None:
        config.encoder.ae_type = str(args.ae_type)
    if args.scale is not None:
        config.encoder.scale = str(args.scale)

    config.dataset.subset = SubsetConfig(mode=str(args.subset_mode), topk=args.topk, min_count=args.min_count)
    config.dataset.sampling.max_spikes_per_unit = args.max_spikes_per_unit
    config.dataset.sort_by_time = True

    waveform_dataset = load_waveform_dataset(config.dataset)
    summary = train_external_autoencoder_artifact(
        waveform_dataset.waveforms,
        config.encoder,
        seed=config.seed,
        artifact_path=resolve_path(args.artifact_dir),
        resume=bool(args.resume),
    )

    artifact_root = resolve_path(args.artifact_dir)
    save_json(
        artifact_root / "training_summary.json",
        {
            **summary,
            "config_path": str(resolve_path(args.config)),
            "variant": config.variant_name,
            "dataset_meta": waveform_dataset.meta,
            "subset_mode": args.subset_mode,
            "topk": args.topk,
            "min_count": args.min_count,
            "max_spikes_per_unit": args.max_spikes_per_unit,
        },
    )
    print(f"saved external AE artifact to {artifact_root}")


if __name__ == "__main__":
    main()
