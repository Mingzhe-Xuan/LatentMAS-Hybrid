#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.config import ALL_DATASETS, PRIMARY_DATASETS, SPLITS, load_config
from analysis.core.schemas import stable_hash


MODEL_8B = "Qwen/Qwen3-8B"
MODEL_14B = "Qwen/Qwen3-14B"
MATRIX_TASKS = {
    "sender.jsonl": "collect_sender_trajectories",
    "kernel_scaling.jsonl": "evaluate_kernel_scaling",
    "perturbation.jsonl": "evaluate_perturbation_stability",
    "model_pairs.jsonl": "evaluate_sender_receiver_performance",
    "entropy_analysis.jsonl": "analyze_logit_entropy",
    "scaling_analysis.jsonl": "analyze_kernel_scaling",
    "variance_analysis.jsonl": "analyze_aligned_state_variance",
    "stability_analysis.jsonl": "analyze_perturbation_stability",
    "model_pair_analysis.jsonl": "analyze_sender_receiver_performance",
    "report.jsonl": "build_kernel_analysis_report",
}
EXPECTED_COUNTS = dict(zip(MATRIX_TASKS, (18, 54, 126, 144, 3, 3, 3, 3, 9, 1)))


def sender_source_id(dataset: str, model: str, *, smoke: bool = False) -> str:
    kmax = 4 if smoke else (160 if dataset in PRIMARY_DATASETS and model == MODEL_8B else 40)
    selection = "first-1" if smoke else "all"
    return f"sender-{stable_hash({'dataset': dataset, 'split': SPLITS[dataset], 'model': model, 'kmax': kmax, 'selection': selection, 'protocol': 'kernel-analysis-v1'})[:24]}"


def receiver_cache_id(*, dataset: str, seed: int, receiver: str, k: int,
                      sender: str | None = None, alignment: str = "kernel",
                      alpha: float = 0.0, generation: dict[str, Any] | None = None,
                      smoke: bool = False) -> str:
    payload: dict[str, Any] = {
        "dataset": dataset, "split": SPLITS[dataset], "seed": seed,
        "receiver": receiver, "k": k, "temperature": .6, "top_p": .95,
        "evaluator": "task-evaluator-v1", "protocol": "kernel-analysis-v1",
        "selection": "first-1" if smoke else "all",
    }
    payload.update(generation or {})
    if k == 0:
        payload.update(sender="receiver-only", alignment="kernel", perturbation="clean")
    else:
        payload.update(sender=sender, alignment=alignment,
                       perturbation="clean" if alpha == 0 else {"gaussian": alpha, "scheme": "sha256-normal-v1"})
    return f"receiver-{stable_hash(payload)[:24]}"


def _row(task: str, cache_id: str, **fields: Any) -> dict[str, Any]:
    return {"task": task, "effective_cache_id": cache_id, **fields}


def _generation_fields(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    task = config["datasets"][dataset]
    generation = config["generation"]
    return {"max_new_tokens": task["max_new_tokens"],
            "generation_batch_size": task["generation_batch_size"],
            "temperature": generation["temperature"], "top_p": generation["top_p"]}


def build_matrices(config_path: str | Path, *, smoke: bool = False,
                   dataset_filter: str | None = None) -> dict[str, list[dict[str, Any]]]:
    config = load_config(config_path).raw
    seeds = config["generation"]["seeds"]
    datasets = [d for d in ALL_DATASETS if dataset_filter in (None, d)]
    primary = [d for d in PRIMARY_DATASETS if d in datasets]
    result = {name: [] for name in MATRIX_TASKS}

    for dataset in datasets:
        for model in (MODEL_8B, MODEL_14B):
            if model == MODEL_8B or model == MODEL_14B:
                kmax = 160 if dataset in PRIMARY_DATASETS and model == MODEL_8B else 40
                result["sender.jsonl"].append(_row(
                    "collect_sender_trajectories", sender_source_id(dataset, model, smoke=smoke),
                    dataset=dataset, split=SPLITS[dataset], sender_model=model,
                    kmax=4 if smoke else kmax, max_samples=1 if smoke else None,
                ))

    for dataset in primary:
        sender_id = sender_source_id(dataset, MODEL_8B, smoke=smoke)
        for seed in seeds:
            scaling_grid = (0, 2, 4) if smoke else config["experiments"]["scaling_k"]
            for k in scaling_grid:
                effective_k = k
                cid = receiver_cache_id(dataset=dataset, seed=seed, receiver=MODEL_8B,
                                        sender=sender_id, k=effective_k,
                                        generation=_generation_fields(config, dataset), smoke=smoke)
                result["kernel_scaling.jsonl"].append(_row(
                    "evaluate_kernel_scaling", cid, dataset=dataset, split=SPLITS[dataset],
                    sender_model=MODEL_8B, receiver_model=MODEL_8B,
                    sender_cache_id=sender_id, k=effective_k, alignment="kernel", alpha=0,
                    generation_seed=seed, max_samples=1 if smoke else None,
                    **_generation_fields(config, dataset),
                ))
            for alignment in ("soft", "linear"):
                cid = receiver_cache_id(dataset=dataset, seed=seed, receiver=MODEL_8B,
                                        sender=sender_id, k=40 if not smoke else 4,
                                        alignment=alignment,
                                        generation=_generation_fields(config, dataset), smoke=smoke)
                result["perturbation.jsonl"].append(_row(
                    "evaluate_perturbation_stability", cid, dataset=dataset,
                    split=SPLITS[dataset], sender_model=MODEL_8B, receiver_model=MODEL_8B,
                    sender_cache_id=sender_id, k=40 if not smoke else 4,
                    alignment=alignment, alpha=0, generation_seed=seed,
                    max_samples=1 if smoke else None,
                    **_generation_fields(config, dataset),
                ))
            for alpha in config["experiments"]["perturbation_alpha"][1:]:
                for alignment in ("kernel", "soft", "linear"):
                    cid = receiver_cache_id(dataset=dataset, seed=seed, receiver=MODEL_8B,
                                            sender=sender_id, k=40 if not smoke else 4,
                                            alignment=alignment, alpha=alpha,
                                            generation=_generation_fields(config, dataset), smoke=smoke)
                    result["perturbation.jsonl"].append(_row(
                        "evaluate_perturbation_stability", cid, dataset=dataset,
                        split=SPLITS[dataset], sender_model=MODEL_8B, receiver_model=MODEL_8B,
                        sender_cache_id=sender_id, k=40 if not smoke else 4,
                        alignment=alignment, alpha=alpha, generation_seed=seed,
                        max_samples=1 if smoke else None,
                        **_generation_fields(config, dataset),
                    ))

    for dataset in datasets:
        for seed in seeds:
            for receiver in (MODEL_8B, MODEL_14B):
                # The 8B baseline on primary datasets is the scaling K=0 cell.
                if not (dataset in PRIMARY_DATASETS and receiver == MODEL_8B):
                    cid = receiver_cache_id(dataset=dataset, seed=seed, receiver=receiver, k=0,
                                            generation=_generation_fields(config, dataset), smoke=smoke)
                    result["model_pairs.jsonl"].append(_row(
                        "evaluate_sender_receiver_performance", cid, dataset=dataset,
                        split=SPLITS[dataset], sender_model=None, receiver_model=receiver,
                        sender_cache_id=None, k=0, alignment="kernel", alpha=0,
                        generation_seed=seed, max_samples=1 if smoke else None,
                        **_generation_fields(config, dataset),
                    ))
            for sender_model in (MODEL_8B, MODEL_14B):
                for receiver in (MODEL_8B, MODEL_14B):
                    if dataset in PRIMARY_DATASETS and sender_model == receiver == MODEL_8B:
                        continue
                    sender_id = sender_source_id(dataset, sender_model, smoke=smoke)
                    cid = receiver_cache_id(dataset=dataset, seed=seed, receiver=receiver,
                                            sender=sender_id, k=40 if not smoke else 4,
                                            generation=_generation_fields(config, dataset), smoke=smoke)
                    result["model_pairs.jsonl"].append(_row(
                        "evaluate_sender_receiver_performance", cid, dataset=dataset,
                        split=SPLITS[dataset], sender_model=sender_model,
                        receiver_model=receiver, sender_cache_id=sender_id,
                        k=40 if not smoke else 4, alignment="kernel", alpha=0,
                        generation_seed=seed, max_samples=1 if smoke else None,
                        **_generation_fields(config, dataset),
                    ))

    analysis_specs = {
        "entropy_analysis.jsonl": ("analyze_logit_entropy", primary),
        "scaling_analysis.jsonl": ("analyze_kernel_scaling", primary),
        "variance_analysis.jsonl": ("analyze_aligned_state_variance", primary),
        "stability_analysis.jsonl": ("analyze_perturbation_stability", primary),
        "model_pair_analysis.jsonl": ("analyze_sender_receiver_performance", datasets),
    }
    for filename, (task, selected) in analysis_specs.items():
        for dataset in selected:
            cid = f"analysis-{stable_hash({'task': task, 'dataset': dataset, 'smoke': smoke})[:24]}"
            result[filename].append(_row(task, cid, dataset=dataset, cache_only=True,
                                         max_samples=1 if smoke else None, smoke=smoke,
                                         selection_policy="first-1" if smoke else "all"))
    result["report.jsonl"].append(_row("build_kernel_analysis_report",
                                               f"report-{stable_hash({'protocol': 'kernel-analysis-v1', 'smoke': smoke})[:24]}",
                                               cache_only=True, smoke=smoke))
    return result


def validate_matrices(matrices: dict[str, list[dict[str, Any]]], *, formal: bool) -> None:
    for filename, rows in matrices.items():
        ids = [row["effective_cache_id"] for row in rows]
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate effective cache IDs in {filename}: {duplicates}")
        if formal and len(rows) != EXPECTED_COUNTS[filename]:
            raise ValueError(f"{filename}: expected {EXPECTED_COUNTS[filename]} rows, got {len(rows)}")
    receiver_ids = [row["effective_cache_id"] for name in
                    ("kernel_scaling.jsonl", "perturbation.jsonl", "model_pairs.jsonl")
                    for row in matrices[name]]
    duplicates = [key for key, count in Counter(receiver_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"receiver matrices are not deduplicated: {duplicates}")


def write_matrices(matrices: dict[str, list[dict[str, Any]]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in matrices.items():
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        fd, temporary = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=output)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, output / filename)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="analysis/configs/kernel_analysis.yaml")
    parser.add_argument("--output", default="analysis/jobs")
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    matrices = build_matrices(args.config, smoke=args.smoke, dataset_filter=args.dataset)
    validate_matrices(matrices, formal=not args.smoke and args.dataset is None)
    counts = {name: len(rows) for name, rows in matrices.items()}
    by_dataset = {name: dict(Counter(row.get("dataset", "all") for row in rows))
                  for name, rows in matrices.items()}
    print(json.dumps({"counts": counts, "by_dataset": by_dataset}, indent=2, sort_keys=True))
    if not args.dry_run:
        write_matrices(matrices, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
