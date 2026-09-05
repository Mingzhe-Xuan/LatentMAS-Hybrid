#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.config import ALL_DATASETS, PRIMARY_DATASETS, load_config
from analysis.pbs.build_job_matrix import build_matrices, write_matrices
from analysis.pbs.build_stt_job_matrix import build_stt_matrices, write_stt_matrices


KERNEL_CONFIG = "analysis/configs/kernel_analysis.yaml"
STT_CONFIG = "analysis/configs/bidirectional_stt.yaml"
KERNEL_EVALUATION = (
    ("kernel_scaling.jsonl", "evaluate_kernel_scaling"),
    ("perturbation.jsonl", "evaluate_perturbation_stability"),
    ("model_pairs.jsonl", "evaluate_sender_receiver_performance"),
)
KERNEL_ANALYSIS = (
    ("entropy_analysis.jsonl", "analyze_logit_entropy"),
    ("scaling_analysis.jsonl", "analyze_kernel_scaling"),
    ("variance_analysis.jsonl", "analyze_aligned_state_variance"),
    ("stability_analysis.jsonl", "analyze_perturbation_stability"),
    ("model_pair_analysis.jsonl", "analyze_sender_receiver_performance"),
)


def _task(config: str, matrix_root: str, filename: str, task: str, index: int, *,
          cache_only: bool = False, exclusive_key: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "config": config, "matrix": f"{matrix_root}/{filename}", "task": task,
        "job_index": index, "cache_mode": "cache-only" if cache_only else "reuse",
    }
    if exclusive_key:
        value["exclusive_key"] = exclusive_key
    return value


def _selected(rows: list[dict[str, Any]], **conditions: Any) -> list[tuple[int, dict[str, Any]]]:
    return [(index, row) for index, row in enumerate(rows, 1)
            if all(row.get(key) == value for key, value in conditions.items())]


def build_dataset_run_manifests(*, target: str = "all", stage: str = "all",
                                dataset_filter: str | None = None, smoke: bool = False,
                                smoke_samples: int = 1,
                                matrix_root: str = "analysis/jobs") -> tuple[
                                    dict[str, list[dict[str, Any]]],
                                    dict[str, list[dict[str, Any]]],
                                    list[dict[str, Any]], list[dict[str, Any]]]:
    if target not in {"all", "kernel", "stt"}:
        raise ValueError(f"invalid target: {target}")
    if stage not in {"all", "collect", "evaluate", "analyze", "report"}:
        raise ValueError(f"invalid stage: {stage}")
    include_kernel = target in {"all", "kernel"} and dataset_filter in (None, *ALL_DATASETS)
    include_stt = target in {"all", "stt"} and dataset_filter in (None, *PRIMARY_DATASETS)
    if target == "kernel" and not include_kernel:
        raise ValueError(f"dataset {dataset_filter!r} is not supported by kernel analysis")
    if target == "stt" and not include_stt:
        raise ValueError(f"dataset {dataset_filter!r} is not supported by STT analysis")

    kernel = (build_matrices(KERNEL_CONFIG, smoke=smoke, dataset_filter=dataset_filter)
              if include_kernel else {})
    stt = (build_stt_matrices(STT_CONFIG, smoke=smoke, dataset_filter=dataset_filter,
                              smoke_samples=smoke_samples) if include_stt else {})
    compute: list[dict[str, Any]] = []
    finalize: list[dict[str, Any]] = []

    if include_kernel and stage in {"all", "collect", "evaluate"}:
        config = load_config(KERNEL_CONFIG).raw
        datasets = [name for name in ALL_DATASETS if dataset_filter in (None, name)]
        seeds = config["generation"]["seeds"]
        if stage == "collect":
            for dataset in datasets:
                tasks = [_task(KERNEL_CONFIG, matrix_root, "sender.jsonl", "collect_sender_trajectories",
                               index, exclusive_key=row["effective_cache_id"])
                         for index, row in _selected(kernel["sender.jsonl"], dataset=dataset)]
                compute.append({"protocol": "kernel", "dataset": dataset,
                                "run": "shared-collect", "seed": None, "tasks": tasks})
        else:
            for seed in seeds:
                for dataset in datasets:
                    tasks: list[dict[str, Any]] = []
                    if stage == "all":
                        tasks.extend(_task(KERNEL_CONFIG, matrix_root, "sender.jsonl",
                                           "collect_sender_trajectories", index,
                                           exclusive_key=row["effective_cache_id"])
                                     for index, row in _selected(
                                         kernel["sender.jsonl"], dataset=dataset))
                    for filename, task in KERNEL_EVALUATION:
                        tasks.extend(_task(KERNEL_CONFIG, matrix_root, filename, task, index)
                                     for index, _ in _selected(
                                         kernel[filename], dataset=dataset,
                                         generation_seed=seed))
                    compute.append({"protocol": "kernel", "dataset": dataset,
                                    "run": seeds.index(seed) + 1, "seed": seed,
                                    "tasks": tasks})

    if include_stt and stage in {"all", "collect", "evaluate"}:
        datasets = [name for name in PRIMARY_DATASETS if dataset_filter in (None, name)]
        for dataset in datasets:
            tasks = []
            if stage in {"all", "collect"}:
                tasks.extend(_task(STT_CONFIG, matrix_root, "stt_planner.jsonl",
                                   "collect_stt_planner_contexts", index,
                                   exclusive_key=row["effective_cache_id"])
                             for index, row in _selected(stt["stt_planner.jsonl"],
                                                         dataset=dataset))
            if stage in {"all", "evaluate"}:
                tasks.extend(_task(STT_CONFIG, matrix_root, "stt_evaluation.jsonl",
                                   "evaluate_bidirectional_stt", index)
                             for index, _ in _selected(stt["stt_evaluation.jsonl"],
                                                       dataset=dataset))
            compute.append({"protocol": "stt", "dataset": dataset, "run": 1,
                            "seed": None, "deterministic": True, "tasks": tasks})

    if include_kernel and stage in {"all", "analyze", "report"}:
        tasks = []
        if stage in {"all", "analyze"}:
            for filename, task in KERNEL_ANALYSIS:
                tasks.extend(_task(KERNEL_CONFIG, matrix_root, filename, task, index, cache_only=True)
                             for index, _ in enumerate(kernel[filename], 1))
        if stage in {"all", "report"}:
            tasks.extend(_task(KERNEL_CONFIG, matrix_root, "report.jsonl", "build_kernel_analysis_report",
                               index, cache_only=True)
                         for index, _ in enumerate(kernel["report.jsonl"], 1))
        finalize.append({"protocol": "kernel", "tasks": tasks})

    if include_stt and stage in {"all", "analyze", "report"}:
        tasks = []
        if stage in {"all", "analyze"}:
            tasks.extend(_task(STT_CONFIG, matrix_root, "stt_analysis.jsonl", "analyze_bidirectional_stt",
                               index, cache_only=True)
                         for index, _ in enumerate(stt["stt_analysis.jsonl"], 1))
        if stage in {"all", "report"}:
            tasks.extend(_task(STT_CONFIG, matrix_root, "stt_report.jsonl",
                               "build_bidirectional_stt_report", index, cache_only=True)
                         for index, _ in enumerate(stt["stt_report.jsonl"], 1))
        finalize.append({"protocol": "stt", "tasks": tasks})
    if finalize:
        finalize = [{
            "protocol": "finalize",
            "targets": [row["protocol"] for row in finalize],
            "tasks": [task for row in finalize for task in row["tasks"]],
        }]
    return kernel, stt, compute, finalize


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("all", "kernel", "stt"), default="all")
    parser.add_argument("--stage", choices=("all", "collect", "evaluate", "analyze", "report"),
                        default="all")
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", default="analysis/jobs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and not args.smoke:
        parser.error("--max-samples requires --smoke")
    output = Path(args.output)
    try:
        matrix_root = output.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as exc:
        parser.error(f"--output must be inside the repository: {exc}")
    if not matrix_root.startswith("analysis/jobs/") and matrix_root != "analysis/jobs":
        parser.error("--output must be analysis/jobs or one of its descendants")
    kernel, stt, compute, finalize = build_dataset_run_manifests(
        target=args.target, stage=args.stage, dataset_filter=args.dataset, smoke=args.smoke,
        smoke_samples=args.max_samples or 1, matrix_root=matrix_root,
    )
    summary = {
        "compute_array_rows": len(compute), "finalize_rows": len(finalize),
        "compute_by_protocol": {
            name: sum(row["protocol"] == name for row in compute) for name in ("kernel", "stt")
        },
        "compute_task_count": sum(len(row["tasks"]) for row in compute),
        "finalize_task_count": sum(len(row["tasks"]) for row in finalize),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.dry_run:
        if kernel:
            write_matrices(kernel, output)
        if stt:
            write_stt_matrices(stt, output)
        _atomic_jsonl(output / "dataset_runs.jsonl", compute)
        _atomic_jsonl(output / "analysis_finalize.jsonl", finalize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
