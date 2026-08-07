#!/usr/bin/env python
"""Latent-CoT C0 and sequential LatentMAS C1/C2/C3 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_aime2025, load_arc_challenge, load_gsm8k, load_mbppplus
from trajectory import (
    ALIGNMENTS,
    collect,
    load_model,
    prompt_template_sha256,
    prompt_template_version,
)
from utils import auto_device, set_seed


SCHEMA_VERSION = 4
OUTPUT_ROOT = ROOT / "exp_result" / "latent_cot"
RUNS_DIR = OUTPUT_ROOT / "runs"
TRAJECTORY_DIR = ROOT / "exp" / "cache" / "trajectories"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["c0", "c1", "c2", "c3"], default="c0")
    parser.add_argument("--model_name", default=None)
    parser.add_argument(
        "--dataset",
        choices=["all", "gsm8k", "mbppplus", "arc_challenge", "aime2025"],
        default=None,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--latent_steps", type=int, default=150)
    parser.add_argument(
        "--latent_step_values",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100, 120, 140, 160, 180],
    )
    parser.add_argument(
        "--alignments",
        nargs="+",
        choices=["identical", "linear", "soft", "kernel", "text"],
        default=["identical", "linear", "soft", "kernel", "text"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--generation_seed", type=int, default=77)
    parser.add_argument("--probe_seed", type=int, default=42)
    parser.add_argument("--bootstrap_replicates", type=int, default=1000)
    parser.add_argument("--entropy_chunk_size", type=int, default=8)
    parser.add_argument("--kernel_features", type=int, default=2048)
    parser.add_argument("--kernel_temperature", type=float, default=1.0)
    parser.add_argument("--kernel_seed", type=int, default=101)
    parser.add_argument("--kernel_chunk_size", type=int, default=4096)
    parser.add_argument("--soft_chunk_size", type=int, default=32)
    parser.add_argument("--align_ridge", type=float, default=1e-5)
    parser.add_argument(
        "--reuse_trajectory",
        action="store_true",
        help="Require an existing compatible C0 trajectory or C1/C2/C3 metrics cache.",
    )
    parser.add_argument(
        "--force_recollect",
        action="store_true",
        help="Ignore a compatible cache and collect rollouts again.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args(argv)
    if args.model_name is None:
        args.model_name = (
            "Qwen/Qwen3-4B" if args.study == "c0" else "Qwen/Qwen3-8B"
        )
    if args.dataset is None:
        args.dataset = "all" if args.study == "c0" else "mbppplus"
    if args.max_questions is None:
        args.max_questions = 512 if args.study == "c0" else 30
    if len(set(args.latent_step_values)) != len(args.latent_step_values):
        parser.error("--latent_step_values must not contain duplicates")
    if any(value < 1 for value in args.latent_step_values):
        parser.error("--latent_step_values must contain positive integers")
    if args.study in {"c1", "c2", "c3"} and args.dataset != "mbppplus":
        parser.error("C1/C2/C3 currently require --dataset mbppplus")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be positive")
    if args.reuse_trajectory and args.force_recollect:
        parser.error("--reuse_trajectory and --force_recollect are mutually exclusive")
    if args.max_questions < 1 or args.latent_steps < 1:
        parser.error("--max_questions and --latent_steps must be positive")
    if args.bootstrap_replicates < 1 or args.entropy_chunk_size < 1:
        parser.error("bootstrap/chunk sizes must be positive")
    if args.kernel_features < 1 or args.kernel_chunk_size < 1:
        parser.error("kernel feature/chunk sizes must be positive")
    if args.soft_chunk_size < 1:
        parser.error("soft chunk size must be positive")
    if args.kernel_temperature <= 0 or args.align_ridge < 0:
        parser.error("kernel temperature must be positive and ridge non-negative")
    args.device = auto_device(args.device)
    args.task = args.dataset
    args.seed = args.probe_seed
    return args


def configure_logger():
    logger = logging.getLogger("latent_cot.c0")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(
        Path.cwd() / "exp_state.txt", mode="a", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def files_sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()



def run_implementation_sha256():
    return files_sha256(
        (
            ROOT / "exp" / "latent_cot" / "run.py",
            ROOT / "exp" / "latent_cot" / "trajectory.py",
        )
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression="zstd")
    temporary.replace(path)


def cache_component(value):
    encoded = []
    for byte in str(value).encode("utf-8"):
        character = chr(byte)
        if 48 <= byte <= 57 or 97 <= byte <= 122 or character in "._-":
            encoded.append(character)
        elif 65 <= byte <= 90:
            encoded.append(f"~U{character}")
        else:
            encoded.append(f"~{byte:02X}")
    return "v-" + "".join(encoded)


def trajectory_paths(args):
    stem = "__".join(
        (
            "c0",
            f"ds={cache_component(args.dataset)}",
            f"sp={cache_component(args.split)}",
            f"m={cache_component(args.model_name)}",
            f"q={args.max_questions}",
            f"seed={args.probe_seed}",
            f"k={args.latent_steps}",
            f"a={cache_component('-'.join(ALIGNMENTS))}",
            f"km={args.kernel_features}",
            f"kt={cache_component(repr(args.kernel_temperature))}",
            f"ks={args.kernel_seed}",
            f"kc={args.kernel_chunk_size}",
            f"lr={cache_component(repr(args.align_ridge))}",
            f"prompt={cache_component(prompt_template_version(args.dataset))}",
            f"rc={int(args.trust_remote_code)}",
        )
    )
    if len(stem) + len(".manifest.json") > 240:
        raise ValueError("C0 trajectory cache filename exceeds 240 characters.")
    return (
        TRAJECTORY_DIR / f"{stem}.pt",
        TRAJECTORY_DIR / f"{stem}.manifest.json",
    )


def selected_datasets(args):
    if args.dataset == "all":
        return ("gsm8k", "mbppplus", "arc_challenge", "aime2025")
    return (args.dataset,)


def resolved_dataset_split(dataset, requested_split):
    # The Hugging Face AIME 2025 dataset exposes its evaluation set as `train`.
    return "train" if dataset == "aime2025" else requested_split


def sampled_items(args):
    loaders = {
        "gsm8k": load_gsm8k,
        "mbppplus": load_mbppplus,
        "arc_challenge": load_arc_challenge,
        "aime2025": load_aime2025,
    }
    indexed = list(enumerate(loaders[args.dataset](split=args.split)))
    random.Random(args.probe_seed).shuffle(indexed)
    return indexed[: args.max_questions]


def tokenizer_fingerprint(tokenizer):
    payload = {
        "vocab": sorted(tokenizer.get_vocab().items()),
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "padding_side": tokenizer.padding_side,
        "resolved_commit": tokenizer.init_kwargs.get("_commit_hash"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def model_fingerprint(wrapper):
    config = wrapper.model.config
    payload = config.to_dict()
    for field in ("transformers_version", "_commit_hash", "_name_or_path"):
        payload.pop(field, None)
    return {
        "resolved_commit": getattr(config, "_commit_hash", None),
        "config_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
        ).hexdigest(),
        "parameter_dtype": str(next(wrapper.model.parameters()).dtype),
    }


def expected_manifest(args, indexed_items, wrapper):
    prompted_questions = [
        {"item_id": item_id, "question": item["question"]}
        for item_id, item in indexed_items
    ]
    content_hash = hashlib.sha256(
        json.dumps(
            prompted_questions, sort_keys=True, ensure_ascii=False, default=str
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "cache_identity": {
            "experiment": "c0",
            "dataset": args.dataset,
            "split": args.split,
            "question_ids": [item_id for item_id, _ in indexed_items],
            "question_content_sha256": content_hash,
            "model_name": args.model_name,
            "model_fingerprint": model_fingerprint(wrapper),
            "tokenizer_fingerprint": tokenizer_fingerprint(wrapper.tokenizer),
            "prompt_template_version": prompt_template_version(args.dataset),
            "prompt_template_sha256": prompt_template_sha256(args.dataset),
            "latent_steps": args.latent_steps,
            "question_selection_seed": args.probe_seed,
            "alignments": list(ALIGNMENTS),
            "recurrence": "soft_kernel_and_greedy_text_feedback_comparison_v4",
            "alignment_config": {
                "linear_ridge": args.align_ridge,
                "kernel_features": args.kernel_features,
                "kernel_temperature": args.kernel_temperature,
                "kernel_seed": args.kernel_seed,
                "kernel_chunk_size": args.kernel_chunk_size,
                "soft_chunk_size": args.soft_chunk_size,
            },
            "recurrence_target_embedding_mean_norm": float(
                wrapper.target_embedding_mean_norm.detach().cpu()
            ),
            "generation": {
                "do_sample": False,
                "decoding": "greedy",
                "text_generation_performed": True,
                "text_decoding": "greedy_fixed_step_count",
            },
            "trust_remote_code": bool(args.trust_remote_code),

        },
        "provenance": {
            "git_commit": git_commit(),
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "output_embedding_has_bias": bool(
                wrapper.model.get_output_embeddings() is not None
                and getattr(wrapper.model.get_output_embeddings(), "bias", None)
                is not None
            ),
        },
    }


def identity_differences(expected, actual, prefix=""):
    differences = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            field = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                differences.append(
                    {"field": field, "expected": expected.get(key), "actual": actual.get(key)}
                )
            else:
                differences.extend(identity_differences(expected[key], actual[key], field))
    elif expected != actual:
        differences.append({"field": prefix, "expected": expected, "actual": actual})
    return differences


def trajectory_cache_differences(expected, actual):
    differences = identity_differences(
        expected.get("schema_version"),
        actual.get("schema_version"),
        "schema_version",
    )
    expected_identity = expected["cache_identity"]
    actual_identity = actual.get("cache_identity", {})
    for field, expected_value in expected_identity.items():
        differences.extend(
            identity_differences(
                expected_value,
                actual_identity.get(field),
                f"cache_identity.{field}",
            )
        )
    return differences


def validate_trajectory(trajectory, manifest, args):
    if trajectory.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("C0 trajectory schema mismatch.")
    records = trajectory.get("records")
    if not isinstance(records, list) or len(records) != manifest.get("record_count"):
        raise RuntimeError("C0 trajectory record count mismatch.")
    expected_pairs = {
        (item_id, alignment)
        for item_id in manifest.get("cache_identity", {}).get("question_ids", [])
        for alignment in ALIGNMENTS
    }
    actual_pairs = {
        (record.get("item_id"), record.get("alignment")) for record in records
    }
    if actual_pairs != expected_pairs or len(records) != len(expected_pairs):
        raise RuntimeError("C0 trajectory alignment coverage mismatch.")
    for record in records:
        hidden = record.get("hidden_states")
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.device.type != "cpu"
            or hidden.dtype != torch.float32
            or hidden.ndim != 2
            or hidden.shape[0] > args.latent_steps
            or hidden.shape[0] != record.get("valid_step_count")
        ):
            raise RuntimeError(
                f"Invalid C0 hidden trajectory for item {record.get('item_id')} "
                f"alignment {record.get('alignment')}."
            )
        if record.get("requested_step_count") != args.latent_steps:
            raise RuntimeError("C0 requested step count mismatch.")
        if record.get("rollout_complete") and hidden.shape[0] != args.latent_steps:
            raise RuntimeError("Complete C0 rollout lacks required hidden states.")

def load_or_collect(args, indexed_items, wrapper, logger):
    expected = expected_manifest(args, indexed_items, wrapper)
    trajectory_path, manifest_path = trajectory_paths(args)
    have_pt, have_manifest = trajectory_path.exists(), manifest_path.exists()
    if have_pt != have_manifest and not args.force_recollect:
        raise RuntimeError(
            f"Incomplete C0 trajectory cache; use --force_recollect: {trajectory_path}"
        )
    cache_hit = have_pt and have_manifest and not args.force_recollect
    if cache_hit:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        differences = trajectory_cache_differences(expected, manifest)
        if differences:
            raise RuntimeError(
                "Refusing incompatible C0 trajectory cache:\n"
                + json.dumps(differences, indent=2, ensure_ascii=False, default=str)
            )
        actual_sha = file_sha256(trajectory_path)
        if actual_sha != manifest.get("trajectory_sha256"):
            raise RuntimeError("C0 trajectory SHA256 integrity check failed.")
        logger.info("C0 Phase A skipped: reusing %s", trajectory_path)
    else:
        if args.reuse_trajectory:
            raise FileNotFoundError(
                f"--reuse_trajectory requested but cache is absent: {trajectory_path}"
            )
        records = collect(wrapper, indexed_items, args, logger)
        trajectory = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "c0",
            "records": records,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "split": args.split,
            "latent_steps": args.latent_steps,
            "alignments": list(ALIGNMENTS),
            "prompt_template_version": prompt_template_version(args.dataset),
            "trajectory_is_complete": all(r["rollout_complete"] for r in records),
        }
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = trajectory_path.with_name(trajectory_path.name + ".tmp")
        torch.save(trajectory, temporary)
        temporary.replace(trajectory_path)
        failed_records_by_reason = {}
        for record in records:
            if record["failure_reason"]:
                reason = record["failure_reason"]
                failed_records_by_reason[reason] = (
                    failed_records_by_reason.get(reason, 0) + 1
                )
        manifest = {
            **expected,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "trajectory_sha256": file_sha256(trajectory_path),
            "question_count": len(indexed_items),
            "record_count": len(records),
            "complete_record_count": sum(r["rollout_complete"] for r in records),
            "failed_record_count": sum(not r["rollout_complete"] for r in records),
            "failed_records_by_reason": failed_records_by_reason,
        }
        write_json(manifest_path, manifest)
        logger.info("C0 Phase A saved %d questions to %s", len(records), trajectory_path)
    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_trajectory(trajectory, manifest, args)
    return trajectory_path, manifest_path, trajectory, manifest, cache_hit


@torch.inference_mode()
def entropy_rows(trajectory, wrapper, args, logger):
    output_head = wrapper.model.get_output_embeddings()
    if output_head is None or not hasattr(output_head, "weight"):
        raise RuntimeError("C0 requires an accessible model output embedding.")
    output_weight = output_head.weight.detach().float()
    output_bias = getattr(output_head, "bias", None)
    output_bias = None if output_bias is None else output_bias.detach().float()
    rows = []
    for number, record in enumerate(trajectory["records"], start=1):
        hidden = record["hidden_states"]
        entropies = []
        entropy_failure = None
        for start in range(0, len(hidden), args.entropy_chunk_size):
            if entropy_failure:
                break
            stop = min(start + args.entropy_chunk_size, len(hidden))
            try:
                logits = torch.nn.functional.linear(
                    hidden[start:stop].to(wrapper.device, dtype=torch.float32),
                    output_weight,
                    output_bias,
                )
                log_probabilities = torch.log_softmax(logits, dim=-1)
                values = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
                entropies.extend(values.detach().cpu().tolist())
            except Exception as error:
                entropy_failure = f"{type(error).__name__}: {error}"
        stopped_reason = None
        for step in range(args.latent_steps):
            entropy = None
            finite = False
            failure_reason = None
            if stopped_reason is not None:
                failure_reason = stopped_reason
            elif step < len(entropies):
                value = float(entropies[step])
                if np.isfinite(value):
                    entropy = value
                    finite = True
                else:
                    stopped_reason = "non_finite_entropy"
                    failure_reason = stopped_reason
            else:
                failure_reason = (
                    entropy_failure
                    or record.get("failure_reason")
                    or "missing_hidden_state"
                )
            rows.append(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "alignment": record["alignment"],
                    "item_id": int(record["item_id"]),
                    "step": int(step),
                    "entropy_nats": entropy,
                    "finite": finite,
                    "failure_reason": failure_reason,
                }
            )
        logger.info(
            "C0 Phase B: question %d/%d (item_id=%d) processed.",
            number,
            len(trajectory["records"]),
            record["item_id"],
        )
    return rows


def bootstrap_interval(values, replicates, seed):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(rows, args):
    steps = []
    for step in range(args.latent_steps):
        step_rows = [row for row in rows if row["step"] == step]
        values = [row["entropy_nats"] for row in step_rows if row["finite"]]
        low, high = bootstrap_interval(
            values, args.bootstrap_replicates, args.probe_seed + step
        )
        steps.append(
            {
                "step": step,
                "n_questions": len(step_rows),
                "n_valid_questions": len(values),
                "n_failed_questions": len(step_rows) - len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    reasons = {}
    for row in rows:
        reason = row.get("failure_reason")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "study": "c0",
        "metric": "pre_unembedding_output_entropy_nats",
        "entropy_compute_dtype": "float32",
        "interpretation_boundary": (
            "Entropy only measures token-distribution sharpness; it does not "
            "establish reasoning correctness, information content, or a unique thought."
        ),
        "total_questions": len({row["item_id"] for row in rows}),
        "latent_steps": args.latent_steps,
        "steps": steps,
        "failure_rows_by_reason": reasons,
    }


def plot_summary(summaries, path, context):
    datasets = list(summaries)
    columns = min(2, len(datasets))
    rows = (len(datasets) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7 * columns, 5 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = [axis for row in axes for axis in row]
    display_names = {
        "gsm8k": "GSM8K",
        "mbppplus": "MBPP+",
        "arc_challenge": "ARC-Challenge",
        "aime2025": "AIME 2025",
    }
    colors = {
        "identical": "#4c78a8",
        "linear": "#f58518",
        "soft": "#b279a2",
        "kernel": "#54a24b",
        "text": "#e45756",
    }
    for axis, dataset in zip(flat_axes, datasets):
        for alignment in ALIGNMENTS:
            steps = summaries[dataset]["alignments"][alignment]["steps"]
            x = np.array([row["step"] for row in steps])
            mean = np.array(
                [np.nan if row["mean"] is None else row["mean"] for row in steps]
            )
            low = np.array(
                [
                    np.nan if row["ci95_low"] is None else row["ci95_low"]
                    for row in steps
                ]
            )
            high = np.array(
                [
                    np.nan if row["ci95_high"] is None else row["ci95_high"]
                    for row in steps
                ]
            )
            axis.plot(
                x,
                mean,
                label=alignment,
                color=colors[alignment],
                linewidth=2,
            )
            axis.fill_between(
                x,
                low,
                high,
                color=colors[alignment],
                alpha=0.12,
            )
        axis.set_xlabel("Latent step")
        axis.set_title(display_names.get(dataset, dataset))
        axis.grid(alpha=0.25)
        axis.legend(title="Recurrence")
    for axis in flat_axes[len(datasets):]:
        axis.set_visible(False)
    flat_axes[0].set_ylabel("Output entropy (nats)")
    figure.suptitle(
        "C0: entropy by latent and text recurrence\n"
        "Solid lines: mean across questions; shaded bands: 95% bootstrap CI"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    write_json(path.with_suffix(".json"), context)

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("._-") or "unknown"


def create_run_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {**vars(args), "implementation_sha256": run_implementation_sha256()}
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    model_leaf = args.model_name.rsplit("/", 1)[-1].rsplit(chr(92), 1)[-1]
    model = safe_name(model_leaf)
    name = (
        f"{safe_name(args.dataset)}_{safe_name(args.split)}_c0_{model}_"
        f"k{args.latent_steps}_"
        f"q{args.max_questions}_seed{args.probe_seed}_{timestamp}_{digest}"
    )
    path = RUNS_DIR / name
    suffix = 1
    while path.exists():
        path = RUNS_DIR / f"{name}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def main(argv=None):
    args = parse_args(argv)
    logger = configure_logger()
    set_seed(args.probe_seed)
    if args.study in {"c1", "c2", "c3"}:
        from mas_analysis import run_mas_study

        return run_mas_study(args, logger)
    run_dir = create_run_dir(args)
    run_manifest_path = run_dir / "run_manifest.json"
    started = time.time()
    datasets = selected_datasets(args)
    run_manifest = {
        "status": "running",
        "study": "c0",
        "datasets": list(datasets),
        "run_directory": str(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "git_commit": git_commit(),
        "implementation_sha256": run_implementation_sha256(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    write_json(run_manifest_path, run_manifest)
    logger.info("C0 run directory: %s", run_dir)
    try:
        wrapper = load_model(args)
        all_rows = []
        summaries = {}
        dataset_contexts = {}
        for dataset in datasets:
            dataset_args = argparse.Namespace(
                **{
                    **vars(args),
                    "dataset": dataset,
                    "split": resolved_dataset_split(dataset, args.split),
                }
            )
            logger.info(
                "C0 dataset %s split %s started.", dataset, dataset_args.split
            )
            indexed_items = sampled_items(dataset_args)
            trajectory_path, manifest_path, trajectory, cache_manifest, cache_hit = (
                load_or_collect(dataset_args, indexed_items, wrapper, logger)
            )
            rows = entropy_rows(trajectory, wrapper, dataset_args, logger)
            expected_row_count = len(trajectory["records"]) * args.latent_steps
            if len(rows) != expected_row_count:
                raise RuntimeError(
                    f"C0 row-count invariant failed for {dataset}: "
                    f"{len(rows)} != {expected_row_count}."
                )
            alignment_summaries = {
                alignment: summarize(
                    [row for row in rows if row["alignment"] == alignment],
                    dataset_args,
                )
                for alignment in ALIGNMENTS
            }
            summary = {
                "dataset": dataset,
                "split": dataset_args.split,
                "alignments": alignment_summaries,
            }
            summary.update(
                {
                    "trajectory": str(trajectory_path),
                    "trajectory_cache_hit": cache_hit,
                    "complete_rollout_record_count": cache_manifest[
                        "complete_record_count"
                    ],
                    "failed_rollout_record_count": cache_manifest[
                        "failed_record_count"
                    ],
                    "failed_rollout_records_by_reason": cache_manifest[
                        "failed_records_by_reason"
                    ],
                    "model_name": args.model_name,
                    "model_revision": cache_manifest["cache_identity"][
                        "model_fingerprint"
                    ]["resolved_commit"],
                    "prompt_template_version": prompt_template_version(dataset),
                    "prompt_template_sha256": prompt_template_sha256(dataset),
                    "tokenizer_fingerprint": cache_manifest["cache_identity"][
                        "tokenizer_fingerprint"
                    ],
                    "output_embedding_has_bias": cache_manifest["provenance"][
                        "output_embedding_has_bias"
                    ],
                }
            )
            summaries[dataset] = summary
            all_rows.extend(rows)
            dataset_contexts[dataset] = {
                "trajectory": str(trajectory_path),
                "trajectory_manifest": str(manifest_path),
                "trajectory_manifest_sha256": file_sha256(manifest_path),
                "trajectory_cache_hit": cache_hit,
                "split": dataset_args.split,
                "prompt_template_version": prompt_template_version(dataset),
                "prompt_template_sha256": prompt_template_sha256(dataset),
            }
            logger.info("C0 dataset %s completed.", dataset)

        metrics_path = run_dir / "metrics" / "c0_entropy_by_step.parquet"
        summary_path = run_dir / "summaries" / "c0_summary.json"
        figure_path = run_dir / "figures" / "c0_entropy_vs_step.pdf"
        write_parquet(metrics_path, all_rows)
        summary_payload = {
            "study": "c0",
            "datasets": summaries,
            "dataset_order": list(datasets),
        }
        write_json(summary_path, summary_payload)
        context = {
            "study": "c0",
            "model_name": args.model_name,
            "datasets": dataset_contexts,
            "dataset_order": list(datasets),
            "split": args.split,
            "question_selection_seed": args.probe_seed,
            "latent_steps": args.latent_steps,
            "alignments": list(ALIGNMENTS),
            "metrics": str(metrics_path),
            "summary": str(summary_path),
        }
        plot_summary(summaries, figure_path, context)

        valid_rows_by_series = {
            f"{dataset}.{alignment}": sum(
                row["finite"]
                for row in all_rows
                if row["dataset"] == dataset and row["alignment"] == alignment
            )
            for dataset in datasets
            for alignment in ALIGNMENTS
        }
        if any(count == 0 for count in valid_rows_by_series.values()):
            failed = [name for name, count in valid_rows_by_series.items() if not count]
            raise RuntimeError(
                f"C0 produced no finite entropy observations for: {failed}"
            )
        first_summary = summaries[datasets[0]]
        run_manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": time.time() - started,
                "dataset_runs": dataset_contexts,
                "model_revision": first_summary["model_revision"],
                "tokenizer_fingerprint": first_summary["tokenizer_fingerprint"],
                "generation": {
                    "do_sample": False,
                    "decoding": "greedy",
                    "text_generation_performed": True,
                    "text_decoding": "greedy_fixed_step_count",
                },
                "output_embedding_has_bias": first_summary[
                    "output_embedding_has_bias"
                ],
                "row_count": len(all_rows),
                "finite_row_count": sum(valid_rows_by_series.values()),
                "finite_rows_by_series": valid_rows_by_series,
                "artifacts": {
                    "metrics": str(metrics_path),
                    "summary": str(summary_path),
                    "figure": str(figure_path),
                },
            }
        )
        write_json(run_manifest_path, run_manifest)
        logger.info("C0 completed in %.1fs.", time.time() - started)
    except Exception as error:
        run_manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": time.time() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(run_manifest_path, run_manifest)
        logger.exception("C0 failed.")
        raise

if __name__ == "__main__":
    main()
