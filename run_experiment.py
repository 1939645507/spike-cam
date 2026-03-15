"""CLI entry point for the spike CAM experiment platform.

中文说明
--------
这是命令行入口文件。

你平时真正运行实验时，最常用的命令就是：

``python run_experiment.py --config configs/xxx.json``
"""

from __future__ import annotations

import argparse

from config import load_config
from experiment_runner import run_experiment_suite


def main() -> None:
    """Parse CLI arguments and run one config suite.

    中文：读取命令行参数，然后调 experiment runner 执行实验。
    """

    parser = argparse.ArgumentParser(description="Run spike CAM experiments from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to a config JSON file under configs/ or an absolute path.")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Optional variant name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Only prepare/load encoded datasets and save encoded statistics.",
    )
    args = parser.parse_args()

    suite = load_config(args.config)
    run_experiment_suite(
        suite,
        selected_variants=args.variant or None,
        encode_only=bool(args.encode_only),
    )


if __name__ == "__main__":
    main()
