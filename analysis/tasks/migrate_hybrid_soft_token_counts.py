#!/usr/bin/env python3
"""Align stored hybrid-soft token counts with run_all.sh semantics."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregate_results import build_average


REPEAT_PATTERN = re.compile(r"repeat_(\d+)\.json$")


def _average(total: int, samples: int) -> float:
    return round(total / samples, 4) if samples else 0.0


def migrate_repeat(document: dict[str, Any]) -> bool:
    """Count each hybrid-soft latent step as one vocabulary decode token."""
    run = document.get("run", {})
    if run.get("method") != "latent_mas_hybrid" or run.get("align_method") != "soft":
        raise ValueError("Expected a latent_mas_hybrid soft result")

    results = document["results"]
    role_metrics = results["role_metrics"]
    changed = False

    for role_name, role in role_metrics.items():
        if role_name.lower() == "judger":
            continue
        latent_total = int(role["tokens"]["latent_output"]["total"])
        samples = int(role["samples"])
        expected = {
            "total": latent_total,
            "average_per_problem": _average(latent_total, samples),
        }
        if role["tokens"].get("text_output") != expected:
            role["tokens"]["text_output"] = expected
            changed = True

        text_total = int(role["tokens"]["text_output"]["total"])
        expected_type = (
            "mixed" if text_total and latent_total else
            "text" if text_total else
            "latent" if latent_total else
            "none"
        )
        if role.get("output_type") != expected_type:
            role["output_type"] = expected_type
            changed = True

    text_total = sum(
        int(role["tokens"]["text_output"]["total"])
        for role in role_metrics.values()
    )
    expected_summary = {
        "total": text_total,
        "average_per_problem": _average(text_total, int(results["processed"])),
    }
    if results["tokens"].get("text_output") != expected_summary:
        results["tokens"]["text_output"] = expected_summary
        changed = True
    if results.get("output_tokens") != text_total:
        results["output_tokens"] = text_total
        changed = True
    return changed


def repeat_sort_key(path: Path) -> int:
    match = REPEAT_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unexpected repeat filename: {path}")
    return int(match.group(1))


def migrate_directory(directory: Path) -> tuple[int, bool]:
    repeat_paths = sorted(directory.glob("repeat_*.json"), key=repeat_sort_key)
    if not repeat_paths:
        return 0, False

    documents = []
    changed_count = 0
    for path in repeat_paths:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if migrate_repeat(document):
            path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            changed_count += 1
        documents.append(document)

    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return changed_count, False

    summary = build_average(documents, repeat_paths)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    summary_changed = summary_path.read_text(encoding="utf-8-sig") != rendered
    if summary_changed:
        summary_path.write_text(rendered, encoding="utf-8")
    return changed_count, summary_changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    args = parser.parse_args()

    directories = sorted(args.result_root.glob("*_latent_mas_hybrid_soft_*"))
    repeats_changed = 0
    summaries_changed = 0
    directories_with_repeats = 0
    for directory in directories:
        changed, summary_changed = migrate_directory(directory)
        repeat_count = len(list(directory.glob("repeat_*.json")))
        if repeat_count:
            directories_with_repeats += 1
        repeats_changed += changed
        summaries_changed += int(summary_changed)

    print(
        f"Processed {directories_with_repeats} directories; "
        f"updated {repeats_changed} repeats and {summaries_changed} summaries."
    )


if __name__ == "__main__":
    main()
