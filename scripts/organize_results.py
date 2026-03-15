"""Organize the top-level results folder into experiments / cache / artifacts.

中文说明
--------
随着实验次数变多，`results/` 根目录很容易堆满：

- 单次实验目录
- thesis run 总包
- encoded cache
- external AE artifact
- 临时 smoke / pilot 结果

这个脚本把它们整理成更清晰的结构：

results/
├── experiments/
│   ├── 2026-03-14/
│   └── 2026-03-15/
├── cache/
└── artifacts/

整理规则：

- `encoded_cache` -> `results/cache/encoded_cache`
- 名字里包含 `artifact` 或明显是 smoke artifact 的目录 -> `results/artifacts/`
- 其他实验目录 / 报告文件 -> `results/experiments/<date>/`

日期优先从名字中提取；提取不到时回退到文件修改时间。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATE_PATTERNS = [
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})"),
    re.compile(r"(?P<date>\d{8})"),
]

EXCLUDE_NAMES = {
    "README.md",
    "experiments",
    "cache",
    "artifacts",
}


@dataclass
class MovePlan:
    source: Path
    destination: Path
    category: str


def _normalize_date_token(token: str) -> str:
    if "-" in token:
        return token
    if len(token) == 8 and token.isdigit():
        return f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    raise ValueError(f"Unsupported date token: {token}")


def _date_from_name(name: str) -> Optional[str]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(name)
        if match:
            return _normalize_date_token(match.group("date"))
    return None


def _date_from_mtime(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime)
    return stamp.strftime("%Y-%m-%d")


def _is_cache_entry(path: Path) -> bool:
    return path.name == "encoded_cache"


def _is_artifact_entry(path: Path) -> bool:
    name = path.name.lower()
    return "artifact" in name or name.startswith("_smoke")


def _category_for(path: Path) -> str:
    if _is_cache_entry(path):
        return "cache"
    if _is_artifact_entry(path):
        return "artifacts"
    return "experiments"


def _destination_for(path: Path, results_root: Path) -> Path:
    category = _category_for(path)
    if category == "cache":
        return results_root / "cache" / path.name
    if category == "artifacts":
        return results_root / "artifacts" / path.name.lstrip("_")

    date_token = _date_from_name(path.name) or _date_from_mtime(path)
    return results_root / "experiments" / date_token / path.name


def build_plan(results_root: Path) -> List[MovePlan]:
    plans: List[MovePlan] = []
    for item in sorted(results_root.iterdir()):
        if item.name in EXCLUDE_NAMES:
            continue
        if item.name.startswith("."):
            continue
        destination = _destination_for(item, results_root)
        if destination.resolve() == item.resolve():
            continue
        plans.append(MovePlan(source=item, destination=destination, category=_category_for(item)))
    return plans


def execute_plan(plans: Iterable[MovePlan]) -> None:
    for plan in plans:
        plan.destination.parent.mkdir(parents=True, exist_ok=True)
        if plan.destination.exists():
            raise FileExistsError(f"Destination already exists: {plan.destination}")
        shutil.move(str(plan.source), str(plan.destination))


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize results/ into experiments, cache, and artifacts.")
    parser.add_argument("--results-root", default="results", help="Results root relative to the project root.")
    parser.add_argument("--apply", action="store_true", help="Actually move files. Without this flag, only print the plan.")
    args = parser.parse_args()

    results_root = (PROJECT_ROOT / args.results_root).resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    plans = build_plan(results_root)
    if not plans:
        print("No top-level entries need organizing.")
        return

    print(f"Results root: {results_root}")
    print(f"Planned moves: {len(plans)}")
    for plan in plans:
        rel_src = plan.source.relative_to(results_root)
        rel_dst = plan.destination.relative_to(results_root)
        print(f"[{plan.category}] {rel_src} -> {rel_dst}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to perform the moves.")
        return

    execute_plan(plans)
    print("\nDone.")


if __name__ == "__main__":
    main()
