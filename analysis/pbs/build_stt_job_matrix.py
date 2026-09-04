#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.config import PRIMARY_DATASETS, load_stt_config
from analysis.core.schemas import stable_hash


MATRIX_TASKS = {
    "stt_planner.jsonl": "collect_stt_planner_contexts",
    "stt_evaluation.jsonl": "evaluate_bidirectional_stt",
    "stt_analysis.jsonl": "analyze_bidirectional_stt",
    "stt_report.jsonl": "build_bidirectional_stt_report",
}
FORMAL_COUNTS = {"stt_planner.jsonl": 6, "stt_evaluation.jsonl": 12,
                 "stt_analysis.jsonl": 3, "stt_report.jsonl": 1}


@lru_cache(maxsize=1)
def _code_revision() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def _cache_id(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}-{stable_hash(payload)[:24]}"


def _row(task: str, cache_id: str, **fields: Any) -> dict[str, Any]:
    return {"task": task, "effective_cache_id": cache_id, **fields}


def planner_cache_id(config: dict[str, Any], dataset: str, model_key: str,
                     *, smoke: bool) -> str:
    payload = {
        "protocol": config["protocol_version"], "kind": "planner-context",
        "dataset": dataset, "split": config["datasets"][dataset]["split"],
        "selection": "first-1" if smoke else "all", "model": config["models"][model_key],
        "sender_budget": config["generation"]["sender_budget"], "do_sample": False,
        "code_revision": _code_revision(),
    }
    return _cache_id("stt-planner", payload)


def evaluation_cache_id(config: dict[str, Any], dataset: str, system: str,
                        *, smoke: bool) -> str:
    receiver_key = "qwen" if system in {"qwen_only", "mistral_to_qwen"} else "mistral"
    payload: dict[str, Any] = {
        "protocol": config["protocol_version"], "kind": "receiver-evaluation",
        "dataset": dataset, "split": config["datasets"][dataset]["split"],
        "selection": "first-1" if smoke else "all", "system": system,
        "receiver": config["models"][receiver_key],
        "max_new_tokens": config["datasets"][dataset]["max_new_tokens"],
        "do_sample": False,
        "code_revision": _code_revision(),
    }
    if "_to_" in system:
        sender_key = system.split("_to_", 1)[0]
        payload.update(sender=config["models"][sender_key],
                       planner_cache_id=planner_cache_id(config, dataset, sender_key, smoke=smoke),
                       artifact=config["transport"]["artifacts"][system],
                       tau=config["transport"]["tau"], causal_shift=False)
    else:
        payload.update(sender="receiver-only", planner_cache_id=None, artifact=None)
    return _cache_id("stt-receiver", payload)


def build_stt_matrices(config_path: str | Path, *, smoke: bool = False,
                       dataset_filter: str | None = None) -> dict[str, list[dict[str, Any]]]:
    config = load_stt_config(config_path).raw
    datasets = [name for name in PRIMARY_DATASETS if dataset_filter in (None, name)]
    result = {name: [] for name in MATRIX_TASKS}
    for dataset in datasets:
        task = config["datasets"][dataset]
        for model_key in ("qwen", "mistral"):
            result["stt_planner.jsonl"].append(_row(
                "collect_stt_planner_contexts",
                planner_cache_id(config, dataset, model_key, smoke=smoke),
                dataset=dataset, split=task["split"], sender_key=model_key,
                sender_model=config["models"][model_key],
                sender_budget=config["generation"]["sender_budget"],
                max_samples=1 if smoke else None,
            ))
        for system in config["systems"]:
            receiver_key = "qwen" if system in {"qwen_only", "mistral_to_qwen"} else "mistral"
            sender_key = system.split("_to_", 1)[0] if "_to_" in system else None
            fields: dict[str, Any] = {
                "dataset": dataset, "split": task["split"], "system": system,
                "receiver_key": receiver_key, "receiver_model": config["models"][receiver_key],
                "sender_key": sender_key,
                "sender_model": config["models"][sender_key] if sender_key else None,
                "planner_cache_id": planner_cache_id(config, dataset, sender_key, smoke=smoke)
                if sender_key else None,
                "artifact": config["transport"]["artifacts"].get(system),
                "tau": config["transport"]["tau"], "causal_shift": False,
                "max_new_tokens": task["max_new_tokens"],
                "generation_batch_size": task["generation_batch_size"],
                "max_samples": 1 if smoke else None,
            }
            result["stt_evaluation.jsonl"].append(_row(
                "evaluate_bidirectional_stt",
                evaluation_cache_id(config, dataset, system, smoke=smoke), **fields,
            ))
        result["stt_analysis.jsonl"].append(_row(
            "analyze_bidirectional_stt",
            _cache_id("stt-analysis", {"protocol": config["protocol_version"],
                                       "dataset": dataset, "smoke": smoke,
                                       "code_revision": _code_revision()}),
            dataset=dataset, split=task["split"], cache_only=True,
            selection_policy="first-1" if smoke else "all", smoke=smoke,
        ))
    result["stt_report.jsonl"].append(_row(
        "build_bidirectional_stt_report",
        _cache_id("stt-report", {"protocol": config["protocol_version"],
                                 "datasets": datasets, "smoke": smoke,
                                 "code_revision": _code_revision()}),
        datasets=datasets, cache_only=True, smoke=smoke,
    ))
    return result


def validate_stt_matrices(matrices: dict[str, list[dict[str, Any]]], *, formal: bool) -> None:
    if set(matrices) != set(MATRIX_TASKS):
        raise ValueError("unexpected STT matrix file set")
    all_ids: list[str] = []
    for filename, rows in matrices.items():
        if any(row.get("task") != MATRIX_TASKS[filename] for row in rows):
            raise ValueError(f"invalid task in {filename}")
        ids = [row.get("effective_cache_id") for row in rows]
        if any(not value for value in ids):
            raise ValueError(f"missing effective cache ID in {filename}")
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate effective cache IDs in {filename}: {duplicates}")
        all_ids.extend(ids)
        if formal and len(rows) != FORMAL_COUNTS[filename]:
            raise ValueError(f"{filename}: expected {FORMAL_COUNTS[filename]} rows, got {len(rows)}")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("STT cache IDs must be unique across matrices")


def write_stt_matrices(matrices: dict[str, list[dict[str, Any]]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in matrices.items():
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        fd, temporary = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=output)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, output / filename)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="analysis/configs/bidirectional_stt.yaml")
    parser.add_argument("--output", default="analysis/jobs")
    parser.add_argument("--dataset", choices=PRIMARY_DATASETS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    matrices = build_stt_matrices(args.config, smoke=args.smoke, dataset_filter=args.dataset)
    validate_stt_matrices(matrices, formal=not args.smoke and args.dataset is None)
    print(json.dumps({"counts": {name: len(rows) for name, rows in matrices.items()},
                      "by_dataset": {name: dict(Counter(row.get("dataset", "all") for row in rows))
                                     for name, rows in matrices.items()}}, indent=2, sort_keys=True))
    if not args.dry_run:
        write_stt_matrices(matrices, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
