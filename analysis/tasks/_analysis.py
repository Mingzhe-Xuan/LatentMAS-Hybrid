from __future__ import annotations

import json
import hashlib
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from analysis.core.cache import CacheError, atomic_write_json, file_sha256
from analysis.core.config import load_config
from analysis.core.statistics import nested_question_seed_bootstrap, robust_slope
from analysis.core.analysis import model_pair_summary, scaling_summary, stability_summary
from analysis.tasks._common import load_job
from analysis.core.schemas import stable_hash


def _validated_manifests(root: Path, kind: str, dataset: str) -> list[tuple[Path, dict]]:
    manifests = []
    directory = root / kind
    for path in directory.glob("*/manifest.json") if directory.exists() else ():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        identity = manifest.get("identity", {})
        if identity.get("dataset") != dataset or not manifest.get("complete"):
            continue
        if kind == "receiver_evaluations":
            artifact = path.parent / "answers.parquet"
            expected = manifest.get("answers_sha256")
        else:
            artifact = path.parent / "questions.parquet"
            expected = manifest.get("questions_sha256")
        if not artifact.exists() or file_sha256(artifact) != expected:
            raise CacheError(f"corrupt linked cache: {path.parent}")
        if manifest.get("identity_hash") != stable_hash(identity):
            raise CacheError(f"incompatible identity hash in linked cache: {path.parent}")
        if kind == "receiver_evaluations":
            import pyarrow.parquet as pq
            if pq.read_metadata(artifact).num_rows != manifest.get("question_count"):
                raise CacheError(f"Receiver row-count mismatch: {path.parent}")
            prompts = manifest.get("prompt_records", [])
            if (len(prompts) != manifest.get("question_count")
                    or any(record.get("role") != "judger"
                           or hashlib.sha256(record.get("rendered", "").encode()).hexdigest() != record.get("sha256")
                           for record in prompts)):
                raise CacheError(f"Receiver prompt provenance is corrupt: {path.parent}")
        manifests.append((path, manifest))
    return manifests


def _result_dir(args, job: dict) -> Path:
    path = Path(args.result_root) / job["task"] / job["effective_cache_id"]
    for child in ("metrics", "summaries", "figures", "provenance"):
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def _validated_result_hit(args, job: dict) -> bool:
    destination = Path(args.result_root) / job["task"] / job["effective_cache_id"]
    manifest_path = destination / "run_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.force:
        return False
    if (manifest.get("status") != "complete" or manifest.get("task") != job["task"]
            or manifest.get("result_id") != job["effective_cache_id"]):
        raise CacheError(f"incomplete or incompatible analysis result: {destination}")
    for relative, digest in manifest.get("artifacts", {}).items():
        path = destination / relative
        if not path.exists() or file_sha256(path) != digest:
            raise CacheError(f"corrupt analysis artifact: {path}")
    for source in manifest.get("sources", []):
        path = Path(source["path"])
        if not path.exists() or file_sha256(path) != source["sha256"]:
            raise CacheError(f"linked source changed after analysis: {path}")
    return True


def _finalize_result(destination: Path, job: dict, sources: list[dict[str, Any]]) -> None:
    artifacts = {}
    for directory in ("metrics", "summaries", "figures", "provenance"):
        for path in sorted((destination / directory).glob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(destination)).replace("\\", "/")] = file_sha256(path)
    atomic_write_json(destination / "run_manifest.json", {
        "status": "complete", "task": job["task"], "result_id": job["effective_cache_id"],
        "cache_only": True, "artifacts": artifacts,
        "sources": [{"path": source["path"], "sha256": source["manifest_hash"]}
                    for source in sources if source.get("path")],
    })


def _write_rows(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path)


def _plot(path: Path, groups: dict[str, list[tuple[float, float]]], xlabel: str,
          ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(6, 4))
    for label, values in groups.items():
        ordered = sorted(values)
        axis.plot([x for x, _ in ordered], [y for _, y in ordered], marker="o", label=label)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    if len(groups) > 1:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _receiver_rows(root: Path, dataset: str) -> tuple[list[dict], list[dict]]:
    import pyarrow.parquet as pq
    rows, provenance = [], []
    for path, manifest in _validated_manifests(root, "receiver_evaluations", dataset):
        table_rows = pq.read_table(path.parent / "answers.parquet").to_pylist()
        identity = manifest["identity"]
        for row in table_rows:
            row.update(condition=identity, cache_id=manifest["cache_id"])
            rows.append(row)
        provenance.append({"cache_id": manifest["cache_id"],
                           "manifest_hash": file_sha256(path), "path": str(path)})
    return rows, provenance


def _linked_kernel_performance(root: Path, dataset: str, k: int = 160) -> list[dict]:
    linked = []
    for path, manifest in _validated_manifests(root, "receiver_evaluations", dataset):
        identity = manifest.get("identity", {})
        if (identity.get("alignment") == "kernel" and int(identity.get("k", -1)) == k
                and float(identity.get("alpha", 0)) == 0):
            linked.append({"cache_id": manifest["cache_id"],
                           "manifest_hash": file_sha256(path),
                           "path": str(path),
                           "generation_seed": identity.get("generation_seed"),
                           "performance": manifest["summary"]})
    if not linked:
        raise CacheError(f"canonical Kernel K={k} Receiver cache is unavailable")
    if sorted(x["generation_seed"] for x in linked) != [42, 43, 44]:
        raise CacheError(f"canonical Kernel K={k} Receiver seed coverage is incomplete")
    return linked


def _assert_receiver_coverage(task: str, rows: list[dict], *, smoke: bool) -> None:
    seeds = (42, 43, 44)
    if task == "analyze_kernel_scaling":
        grid = (0, 2, 4) if smoke else (0, 10, 20, 40, 80, 160)
        expected = {(seed, k, "kernel", 0.0) for seed in seeds for k in grid}
        actual = {(int(r["condition"]["generation_seed"]), int(r["condition"]["k"]),
                   r["condition"]["alignment"], float(r["condition"].get("alpha", 0)))
                  for r in rows}
    elif task == "analyze_perturbation_stability":
        doses = (0.0, .01, .025, .05, .1)
        expected = {(seed, 4 if smoke else 40, method, alpha)
                    for seed in seeds for method in ("kernel", "soft", "linear")
                    for alpha in doses}
        actual = {(int(r["condition"]["generation_seed"]), int(r["condition"]["k"]),
                   r["condition"]["alignment"], float(r["condition"].get("alpha", 0)))
                  for r in rows}
    elif task == "analyze_sender_receiver_performance":
        model8, model14 = "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"
        expected = set()
        for seed in seeds:
            expected.update((seed, "receiver-only", receiver, 0)
                            for receiver in (model8, model14))
            expected.update((seed, sender, receiver, 4 if smoke else 40)
                            for sender in (model8, model14) for receiver in (model8, model14))
        actual = {(int(r["condition"]["generation_seed"]),
                   "receiver-only" if int(r["condition"]["k"]) == 0
                   else r["condition"]["sender_model_id"],
                   r["condition"]["receiver_model_id"], int(r["condition"]["k"]))
                  for r in rows}
    else:
        return
    if actual != expected:
        raise CacheError(f"{task} cache coverage mismatch: missing={sorted(expected - actual, key=str)}, extra={sorted(actual - expected, key=str)}")
    # Every logical cell must cover the same ordered question IDs exactly once.
    item_sets: dict[tuple, list[int]] = defaultdict(list)
    for row in rows:
        c = row["condition"]
        if task == "analyze_sender_receiver_performance":
            cell = (c["generation_seed"], "receiver-only" if int(c["k"]) == 0 else c["sender_model_id"],
                    c["receiver_model_id"], c["k"])
        else:
            cell = (c["generation_seed"], c["k"], c["alignment"], float(c.get("alpha", 0)))
        item_sets[cell].append(int(row["item_id"]))
    reference = None
    for cell, item_ids in item_sets.items():
        if len(item_ids) != len(set(item_ids)):
            raise CacheError(f"duplicate question rows in Receiver cell {cell}")
        ordered = tuple(sorted(item_ids))
        reference = ordered if reference is None else reference
        if ordered != reference:
            raise CacheError(f"Receiver question coverage differs in cell {cell}")


def analyze_receiver_table(args, expected_task: str,
                           selector: Callable[[dict], bool], x_key: str,
                           group_key: str) -> int:
    load_config(args.config)
    if not args.cache_only:
        raise ValueError("analysis tasks require --cache-only")
    job = load_job(args)
    if job.get("task") != expected_task:
        raise ValueError("job task does not match entry point")
    if _validated_result_hit(args, job):
        return 10
    rows, provenance = _receiver_rows(Path(args.cache_root), job["dataset"])
    rows = [row for row in rows if
            row["condition"].get("selection_policy", "all") == job.get("selection_policy", "all")
            and selector(row["condition"])]
    if not rows:
        raise CacheError(f"no validated Receiver caches for {expected_task}/{job['dataset']}")
    _assert_receiver_coverage(expected_task, rows, smoke=bool(job.get("smoke")))
    grouped: dict[tuple[Any, Any], list[dict]] = defaultdict(list)
    for row in rows:
        condition = row["condition"]
        grouped[(condition.get(group_key), condition.get(x_key))].append({
            "item_id": row["item_id"], "seed": condition["generation_seed"],
            "value": float(row["correct"]),
        })
    summaries, plot_groups = [], defaultdict(list)
    for (group, x), values in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        estimate = nested_question_seed_bootstrap(values, samples=2000, seed=101)
        summary = {group_key: group, x_key: x, **estimate}
        summaries.append(summary)
        plot_groups[str(group)].append((float(x or 0), estimate["estimate"]))
    destination = _result_dir(args, job)
    _write_rows(destination / "metrics" / "question_metrics.parquet", rows)
    comparative = ({
        "analyze_kernel_scaling": scaling_summary,
        "analyze_perturbation_stability": stability_summary,
        "analyze_sender_receiver_performance": model_pair_summary,
    }.get(expected_task, lambda _: {}))(rows)
    atomic_write_json(destination / "summaries" / "summary.json", {
        "dataset": job["dataset"], "metric_source": "linked_receiver_cache",
        "selection_policy": job.get("selection_policy", "all"),
        "cells": summaries, "comparative": comparative,
    })
    atomic_write_json(destination / "provenance" / "cache_manifests.json", provenance)
    _plot(destination / "figures" / "accuracy.png", plot_groups, x_key, "accuracy")
    _finalize_result(destination, job, provenance)
    return 0


def analyze_entropy(args) -> int:
    import torch
    import pyarrow.parquet as pq
    from analysis.core.cache import CacheHandle, SenderTrajectoryStore
    from analysis.tasks._common import load_wrapper, model_args
    from alignment import compute_logits_entropy

    config = load_config(args.config).raw
    if not args.cache_only:
        raise ValueError("entropy analysis requires cache-only mode")
    job = load_job(args)
    if _validated_result_hit(args, job):
        return 10
    manifests = _validated_manifests(Path(args.cache_root), "sender_trajectories", job["dataset"])
    required_k = 4 if job.get("smoke") else 160
    selected = [(p, m) for p, m in manifests if m["identity"].get("model_id") == "Qwen/Qwen3-8B"
                and int(m["identity"].get("kmax", 0)) >= required_k
                and m["identity"].get("selection_policy", "all") == job.get("selection_policy", "all")]
    if len(selected) != 1:
        raise CacheError(f"expected one canonical Sender cache, found {len(selected)}")
    path, manifest = selected[0]
    wrapper = load_wrapper("Qwen/Qwen3-8B", args.device,
                           model_args({"dataset": job["dataset"], "sender_model": "Qwen/Qwen3-8B"}, alignment="kernel"))
    base = getattr(wrapper, "HF_model", getattr(wrapper, "model", None))
    head = base.get_output_embeddings()
    store = SenderTrajectoryStore(args.cache_root)
    handle = CacheHandle(manifest["cache_id"], path.parent, manifest["identity_hash"])
    rows = []
    for item_id in manifest["expected_item_ids"]:
        trajectory = store.read_item(handle, item_id)
        for step, hidden in enumerate(trajectory.hidden, start=1):
            entropy = compute_logits_entropy(hidden.to(head.weight.device), head.weight,
                                             getattr(head, "bias", None), temperature=1.0,
                                             query_chunk_size=32).squeeze()
            rows.append({"item_id": item_id, "step": step, "entropy": float(entropy),
                         "finite": bool(torch.isfinite(entropy)),
                         "probe_unembedding_evaluations": 1})
    finite = [row for row in rows if row["finite"]]
    by_step = defaultdict(list)
    for row in finite:
        by_step[row["step"]].append(row["entropy"])
    curve = []
    for step, values in sorted(by_step.items()):
        step_rows = [{"item_id": row["item_id"], "seed": 0, "value": row["entropy"]}
                     for row in finite if row["step"] == step]
        interval = nested_question_seed_bootstrap(step_rows, samples=2000, seed=101)
        curve.append({"step": step, "mean_entropy": interval["estimate"],
                      "ci_low": interval["ci_low"], "ci_high": interval["ci_high"]})
    x, y = [x["step"] for x in curve], [x["mean_entropy"] for x in curve]
    linked_performance = _linked_kernel_performance(Path(args.cache_root), job["dataset"], required_k)
    summary = {"entropy_auc": float(np.trapezoid(y, x)), "entropy_curve": curve,
               "early_to_late_change": float(np.mean(y[len(y)//2:]) - np.mean(y[:len(y)//2])),
               "linear_slope": float(np.polyfit(x, y, 1)[0]), "robust_slope": robust_slope(x, y),
               "non_finite": len(rows) - len(finite), "missing_states": 0,
               "probe_unembedding_evaluations": len(rows),
               "metric_source": "linked_receiver_cache",
               "selection_policy": job.get("selection_policy", "all"),
               "linked_receiver_performance": linked_performance}
    destination = _result_dir(args, job)
    _write_rows(destination / "metrics" / "entropy.parquet", rows)
    atomic_write_json(destination / "summaries" / "summary.json", summary)
    provenance = ([{"cache_id": manifest["cache_id"], "manifest_hash": file_sha256(path),
                    "path": str(path)}]
                  + [{"cache_id": x["cache_id"], "manifest_hash": x["manifest_hash"]}
                     for x in linked_performance])
    atomic_write_json(destination / "provenance" / "cache_manifests.json", provenance)
    _plot(destination / "figures" / "entropy.png", {job["dataset"]: list(zip(x, y))}, "latent step", "entropy")
    _finalize_result(destination, job, provenance)
    return 0


def analyze_variance(args) -> int:
    import gc
    import torch
    from alignment import apply_alignment, build_kernel_state, build_soft_state
    from analysis.core.cache import CacheHandle, SenderTrajectoryStore
    from analysis.tasks._common import load_wrapper, model_args

    config = load_config(args.config).raw
    if not args.cache_only:
        raise ValueError("variance analysis requires cache-only mode")
    job = load_job(args)
    if _validated_result_hit(args, job):
        return 10
    manifests = _validated_manifests(Path(args.cache_root), "sender_trajectories", job["dataset"])
    required_k = 4 if job.get("smoke") else 160
    selected = [(p, m) for p, m in manifests if m["identity"].get("model_id") == "Qwen/Qwen3-8B"
                and int(m["identity"].get("kmax", 0)) >= required_k
                and m["identity"].get("selection_policy", "all") == job.get("selection_policy", "all")]
    if len(selected) != 1:
        raise CacheError("canonical Sender cache is unavailable or ambiguous")
    path, manifest = selected[0]
    wrapper = load_wrapper("Qwen/Qwen3-8B", args.device,
                           model_args({"dataset": job["dataset"], "sender_model": "Qwen/Qwen3-8B"}, alignment="kernel"))
    base = getattr(wrapper, "HF_model", getattr(wrapper, "model", None))
    output_weight, input_weight, bias = wrapper._embedding_weights(base, base)
    # Variance needs only the tied vocabulary projections.  Keeping the full
    # causal LM resident leaves virtually no workspace on a 32 GiB GPU (the
    # 8B checkpoint is loaded in fp32 by the production wrapper).  Stage the
    # two projections through host memory, release the model, and put only the
    # required tensors back on the requested analysis device.
    output_weight_host = output_weight.detach().cpu().clone()
    input_weight_host = input_weight.detach().cpu().clone()
    bias_host = None if bias is None else bias.detach().cpu().clone()
    del output_weight, input_weight, bias, base, wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    analysis_device = torch.device(args.device)
    output_weight = output_weight_host.to(analysis_device)
    input_weight = input_weight_host.to(analysis_device)
    bias = None if bias_host is None else bias_host.to(analysis_device)
    del output_weight_host, input_weight_host, bias_host
    soft = build_soft_state(output_weight, input_weight, bias, temperature=.6, query_chunk_size=32)
    store, handle = SenderTrajectoryStore(args.cache_root), CacheHandle(manifest["cache_id"], path.parent, manifest["identity_hash"])
    hidden = torch.cat([store.read_item(handle, i).hidden.float() for i in manifest["expected_item_ids"]])
    chunk_size = 32
    oracle_parts = []
    for start in range(0, hidden.shape[0], chunk_size):
        with torch.inference_mode():
            oracle_parts.append(apply_alignment(hidden[start:start + chunk_size].to(output_weight.device), soft).float().cpu())
    oracle = torch.cat(oracle_parts)
    rows = []
    for features in config["experiments"]["variance_features"]:
        aligned_sum = torch.zeros_like(oracle)
        squared_norm_sum = 0.0
        squared_error_sum = 0.0
        relative_error_sum = 0.0
        cosine_sum = 0.0
        tail_count = 0
        observation_count = 0
        for seed in range(config["experiments"]["variance_seeds"]):
            with torch.inference_mode():
                state = build_kernel_state(output_weight, input_weight, bias, feature_count=features,
                                           temperature=.6, seed=seed, chunk_size=4096)
            for start in range(0, hidden.shape[0], chunk_size):
                stop = min(start + chunk_size, hidden.shape[0])
                with torch.inference_mode():
                    estimate = apply_alignment(hidden[start:stop].to(output_weight.device), state).float().cpu()
                target = oracle[start:stop]
                difference = estimate - target
                relative = difference.norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-12)
                aligned_sum[start:stop] += estimate
                squared_norm_sum += float((estimate * estimate).sum())
                squared_error_sum += float((difference * difference).sum(dim=-1).sum())
                relative_error_sum += float(relative.sum())
                cosine_sum += float(torch.nn.functional.cosine_similarity(estimate, target, dim=-1).sum())
                tail_count += int((relative > .1).sum())
                observation_count += estimate.shape[0]
            del state
        seeds = config["experiments"]["variance_seeds"]
        centered_sum_squares = squared_norm_sum - float((aligned_sum * aligned_sum).sum()) / seeds
        variance = centered_sum_squares / ((seeds - 1) * aligned_sum.numel())
        rows.append({"features": features, "mean_coordinate_variance": variance,
                     "mean_squared_l2_error": squared_error_sum / observation_count,
                     "mean_relative_l2_error": relative_error_sum / observation_count,
                     "mean_cosine": cosine_sum / observation_count,
                     "tail_probability_relative_l2_gt_0_1": tail_count / observation_count,
                     "probe_unembedding_evaluations": int(hidden.shape[0])})
    slope = float(np.polyfit(np.log([r["features"] for r in rows]),
                             np.log([r["mean_coordinate_variance"] for r in rows]), 1)[0])
    destination = _result_dir(args, job)
    _write_rows(destination / "metrics" / "variance.parquet", rows)
    linked_performance = _linked_kernel_performance(Path(args.cache_root), job["dataset"], required_k)
    atomic_write_json(destination / "summaries" / "summary.json",
                      {"log_log_variance_slope": slope, "cells": rows,
                       "metric_source": "linked_receiver_cache",
                       "selection_policy": job.get("selection_policy", "all"),
                       "linked_receiver_performance": linked_performance})
    provenance = ([{"cache_id": manifest["cache_id"], "manifest_hash": file_sha256(path),
                    "path": str(path)}]
                  + [{"cache_id": x["cache_id"], "manifest_hash": x["manifest_hash"]}
                     for x in linked_performance])
    atomic_write_json(destination / "provenance" / "cache_manifests.json", provenance)
    _plot(destination / "figures" / "variance.png", {"kernel": [(r["features"], r["mean_coordinate_variance"]) for r in rows]}, "features", "variance")
    _finalize_result(destination, job, provenance)
    return 0
