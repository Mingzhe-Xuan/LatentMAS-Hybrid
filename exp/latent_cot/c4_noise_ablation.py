"""C4: hierarchical AIME2025/GPQA pre-/post-alignment replacement ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import load_aime2025, load_gpqa_diamond
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from utils import set_seed

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "exp_result" / "latent_cot" / "runs"
CACHE_DIR = ROOT / "exp" / "cache" / "latent_cot_c4"
SCHEMA_VERSION = 2
ALIGNMENTS = ("kernel",)
CONDITIONS = ("clean", "pre_replace", "post_replace")
ROLES = ("planner", "critic", "refiner")
SAMPLED_STEPS = (0, 20, 40, 60, 80, 100, 120)
COLORS = {"clean": "#4c78a8", "pre_replace": "#e45756", "post_replace": "#54a24b"}
DATASETS = {
    "aime2025": {"split": "train", "task": "aime2025", "loader": load_aime2025},
    "gpqa_diamond": {"split": "test", "task": "gpqa", "loader": load_gpqa_diamond},
}


def _sampled_steps(latent_steps, noise_site):
    upper_bound = latent_steps if noise_site != "post_alignment_target_embedding" else latent_steps - 1
    return tuple(step for step in SAMPLED_STEPS if step <= upper_bound)


NOISE_SITES = {
    "clean": "none",
    "pre_replace": "pre_alignment_source_hidden",
    "post_replace": "post_alignment_target_embedding",
}


def _safe(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("._-") or "unknown"


def _json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parquet(path: Path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(list(rows)), tmp, compression="zstd")
    tmp.replace(path)


def _read_parquet(path: Path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha():
    digest = hashlib.sha256()
    for path in (
        ROOT / "exp" / "latent_cot" / "c4_noise_ablation.py",
        ROOT / "methods" / "latent_mas.py",
        ROOT / "models.py",
        ROOT / "prompts.py",
    ):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _items(dataset, max_questions):
    config = DATASETS[dataset]
    selected = []
    for item_id, item in enumerate(config["loader"](split=config["split"])):
        if len(selected) >= max_questions:
            break
        selected.append((item_id, item))
    if len(selected) != max_questions:
        raise RuntimeError(
            f"Requested {max_questions} {dataset} questions, loaded {len(selected)}"
        )
    return selected
def _dataset_identity(indexed_items):
    payload = [
        {"item_id": item_id, "question": item["question"], "gold": item.get("gold", "")}
        for item_id, item in indexed_items
    ]
    return {
        "selection": "dataset_order_prefix",
        "item_ids": [item_id for item_id, _ in indexed_items],
        "content_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }


def _identity(args, dataset, indexed_items, alignment, condition, seed):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown C4 condition: {condition}")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "c4_hierarchical_pre_post_replace",
        "dataset": dataset,
        "split": DATASETS[dataset]["split"],
        "dataset_identity": _dataset_identity(indexed_items),
        "model_name": args.model_name,
        "prompt": "hierarchical",
        "latent_steps_per_agent": int(args.latent_steps),
        "latent_roles": list(ROLES),
        "alignment": alignment,
        "condition": condition,
        "generation_seed": int(seed),
        "noise": {
            "site": NOISE_SITES[condition],
            "applied_at_steps": (
                f"prefill_output_step_0_and_recurrent_output_steps_1_to_{args.latent_steps}"
                if condition == "pre_replace"
                else f"aligned_latent_input_steps_0_to_{args.latent_steps - 1}"
                if condition == "post_replace"
                else "none"
            ),
            "distribution": "independent_standard_gaussian_then_per_vector_l2_norm_matched",
            "noise_seed": int(args.noise_seed_offset + seed),
            "generator": "dedicated_device_torch_generator",
        },
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.generate_bs,
            "manual_think": bool(args.think),
        },
        "alignment_config": {
            "linear_ridge": args.align_ridge,
            "kernel_features": args.kernel_features,
            "kernel_temperature": args.kernel_temperature,
            "kernel_seed": int(seed),
            "kernel_chunk_size": args.kernel_chunk_size,
        },
        "implementation_sha256": _implementation_sha(),
    }


def _cache_paths(identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    stem = f"c4_{identity['dataset']}_{identity['alignment']}_{identity['condition']}_seed{identity['generation_seed']}_{digest}"
    return (
        CACHE_DIR / f"{stem}.answers.parquet",
        CACHE_DIR / f"{stem}.hidden.parquet",
        CACHE_DIR / f"{stem}.manifest.json",
    )


def _validated_cache_rows(answers_path, hidden_path, manifest_path, identity):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha(answers_path) != manifest["sha256"]["answers"] or _sha(hidden_path) != manifest["sha256"]["hidden"]:
        raise RuntimeError(f"C4 cache integrity check failed: {manifest_path}")
    return _read_parquet(answers_path), _read_parquet(hidden_path), manifest


def _legacy_identity_matches(old, identity):
    """Accept only v1 cells whose experimental settings match this C4 cell."""
    legacy_condition = {"clean": "clean", "pre_replace": "random_hidden"}.get(identity["condition"])
    if legacy_condition is None:
        return False
    if old.get("schema_version") != 1 or old.get("experiment") != "c4_hierarchical_random_output_hidden":
        return False
    fields = (
        "dataset", "split", "dataset_identity", "model_name", "prompt",
        "latent_steps_per_agent", "latent_roles", "alignment",
        "generation_seed", "generation", "alignment_config",
    )
    if any(old.get(field) != identity.get(field) for field in fields):
        return False
    expected_noise = {
        "applied_at_steps": "prefill_output_step_0_and_every_recurrent_output_step_1_to_120",
        "distribution": "independent_standard_gaussian_then_per_vector_l2_norm_matched",
        "noise_seed": identity["noise"]["noise_seed"],
        "generator": "dedicated_device_torch_generator",
    }
    return old.get("condition") == legacy_condition and old.get("noise") == expected_noise


def _legacy_rows(rows, dataset, condition):
    site = NOISE_SITES[condition]
    return [{**row, "dataset": dataset, "condition": condition, "noise_site": site} for row in rows]


def _previous_identity_matches(old, identity):
    if old.get("schema_version") != SCHEMA_VERSION:
        return False
    old_without_code = {
        key: value for key, value in old.items() if key != "implementation_sha256"
    }
    new_without_code = {
        key: value for key, value in identity.items() if key != "implementation_sha256"
    }
    return old_without_code == new_without_code


def _load_previous_cell(args, identity):
    if args.force_recollect:
        return None
    patterns = (
        f"c4_{identity['alignment']}_{identity['condition']}_seed{identity['generation_seed']}_*.manifest.json",
        f"c4_{identity['dataset']}_{identity['alignment']}_{identity['condition']}_seed{identity['generation_seed']}_*.manifest.json",
    )
    for pattern in patterns:
        for manifest_path in sorted(CACHE_DIR.glob(pattern)):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not _previous_identity_matches(manifest.get("cache_identity", {}), identity):
                continue
            stem = manifest_path.name.removesuffix(".manifest.json")
            answers_path = CACHE_DIR / f"{stem}.answers.parquet"
            hidden_path = CACHE_DIR / f"{stem}.hidden.parquet"
            if not answers_path.exists() or not hidden_path.exists():
                continue
            answers, hidden, validated_manifest = _validated_cache_rows(
                answers_path, hidden_path, manifest_path, identity
            )
            return (
                _legacy_rows(answers, identity["dataset"], identity["condition"]),
                _legacy_rows(hidden, identity["dataset"], identity["condition"]),
                validated_manifest,
                (answers_path, hidden_path, manifest_path),
            )
    return None

def _load_legacy_cell(args, identity):
    if args.force_recollect or identity["alignment"] != "kernel":
        return None
    legacy_condition = {"clean": "clean", "pre_replace": "random_hidden"}.get(identity["condition"])
    if legacy_condition is None:
        return None
    pattern = f"c4_kernel_{legacy_condition}_seed{identity['generation_seed']}_*.manifest.json"
    for manifest_path in sorted(CACHE_DIR.glob(pattern)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_identity = manifest.get("cache_identity", {})
        if not _legacy_identity_matches(old_identity, identity):
            continue
        stem = manifest_path.name.removesuffix(".manifest.json")
        answers_path = CACHE_DIR / f"{stem}.answers.parquet"
        hidden_path = CACHE_DIR / f"{stem}.hidden.parquet"
        if not answers_path.exists() or not hidden_path.exists():
            continue
        answers, hidden, validated_manifest = _validated_cache_rows(
            answers_path, hidden_path, manifest_path, identity
        )
        return _legacy_rows(answers, identity["dataset"], identity["condition"]), _legacy_rows(hidden, identity["dataset"], identity["condition"]), validated_manifest, (answers_path, hidden_path, manifest_path)
    return None


def _load_cell(args, identity):
    answers_path, hidden_path, manifest_path = _cache_paths(identity)
    present = [answers_path.exists(), hidden_path.exists(), manifest_path.exists()]
    if any(present) and not all(present) and not args.force_recollect:
        raise RuntimeError(f"Incomplete C4 cache cell; use --force_recollect: {manifest_path}")
    if all(present) and not args.force_recollect:
        answers, hidden, manifest = _validated_cache_rows(answers_path, hidden_path, manifest_path, identity)
        if manifest.get("cache_identity") != identity:
            raise RuntimeError(f"C4 cache identity mismatch: {manifest_path}")
        return answers, hidden, manifest, (answers_path, hidden_path, manifest_path)
    previous = _load_previous_cell(args, identity)
    if previous is not None:
        return previous
    legacy = _load_legacy_cell(args, identity)
    if legacy is not None:
        return legacy
    if args.reuse_trajectory:
        raise FileNotFoundError(f"--reuse_trajectory requested but C4 cache is absent: {manifest_path}")
    return None, None, None, (answers_path, hidden_path, manifest_path)
def _save_cell(paths, identity, answer_rows, hidden_rows, model_info):
    answers_path, hidden_path, manifest_path = paths
    _parquet(answers_path, answer_rows)
    _parquet(hidden_path, hidden_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cache_identity": identity,
        "row_count": {"answers": len(answer_rows), "hidden": len(hidden_rows)},
        "sha256": {"answers": _sha(answers_path), "hidden": _sha(hidden_path)},
        "model": model_info,
    }
    _json(manifest_path, manifest)
    return manifest


class VectorReplaceTransform:
    """Norm-matched independent replacement at one explicitly named C4 site."""

    def __init__(self, dataset, condition, noise_seed, alignment, generation_seed, noise_site, latent_steps):
        self.dataset = dataset
        self.condition = condition
        self.noise_seed = int(noise_seed)
        self.alignment = alignment
        self.generation_seed = int(generation_seed)
        self.noise_site = noise_site
        self.sampled_steps = set(_sampled_steps(latent_steps, noise_site))
        self.generator = None
        self.item_ids = []
        self.rows = []
        self.previous = {}

    def begin_batch(self, item_ids):
        self.item_ids = [int(value) for value in item_ids]
        self.previous = {}

    def _replace(self, vector):
        original = vector.detach().float()
        if self.condition in {"pre_replace", "post_replace"}:
            if self.generator is None:
                self.generator = torch.Generator(device=vector.device)
                self.generator.manual_seed(self.noise_seed)
            noise = torch.randn(
                original.shape, dtype=torch.float32, device=vector.device,
                generator=self.generator,
            )
            original_norm = original.norm(dim=-1, keepdim=True)
            noise_norm = noise.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
            effective = (noise * (original_norm / noise_norm)).to(dtype=vector.dtype)
        else:
            effective = vector
        return original, effective

    @torch.inference_mode()
    def __call__(self, role, step, vector):
        original, effective = self._replace(vector)
        effective_float = effective.detach().float()
        for index, item_id in enumerate(self.item_ids):
            key = (role, item_id)
            previous = self.previous.get(key)
            if int(step) in self.sampled_steps:
                self.rows.append({
                    "dataset": self.dataset,
                    "item_id": item_id,
                    "alignment": self.alignment,
                    "condition": self.condition,
                    "noise_site": self.noise_site,
                    "generation_seed": self.generation_seed,
                    "agent": role,
                    "local_step": int(step),
                    "original_norm": float(original[index].norm().item()),
                    "effective_norm": float(effective_float[index].norm().item()),
                    "original_effective_cosine": float(F.cosine_similarity(original[index:index+1], effective_float[index:index+1], dim=-1)[0].item()),
                    "adjacent_effective_cosine": None if previous is None else float(F.cosine_similarity(previous, effective_float[index:index+1], dim=-1)[0].item()),
                    "finite": bool(torch.isfinite(effective_float[index]).all().item()),
                })
            self.previous[key] = effective_float[index:index+1].detach().clone()
        return effective
def _method_args(args, dataset, alignment, seed):
    return argparse.Namespace(**{
        **vars(args),
        "task": DATASETS[dataset]["task"],
        "method": "latent_mas",
        "prompt": "hierarchical",
        "align_method": alignment,
        "seed": int(seed),
        "kernel_seed": int(seed),
        "device2": args.device,
        "sequential_info_only": False,
        "latent_only": False,
        "use_vllm": False,
        "use_second_HF_model": False,
        "enable_prefix_caching": False,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "soft_temperature": args.kernel_temperature,
    })


def _sum_agent_metric(result, section, key):
    return sum(float(agent.get("metrics", {}).get(section, {}).get(key, 0) or 0) for agent in result.get("agents", []))


def _answer_row(args, dataset, item_id, item, result, alignment, condition, seed, seconds, failure=None):
    prediction = result.get("prediction")
    return {
        "dataset": dataset,
        "split": DATASETS[dataset]["split"],
        "item_id": int(item_id),
        "alignment": alignment,
        "condition": condition,
        "generation_seed": int(seed),
        "latent_steps_per_agent": int(args.latent_steps),
        "total_latent_steps": int(args.latent_steps) * len(ROLES),
        "question": item["question"],
        "gold": item.get("gold", ""),
        "prediction": prediction,
        "raw_prediction": result.get("raw_prediction", ""),
        "correct": bool(result.get("correct", False)),
        "parse_success": prediction is not None,
        "failure_reason": failure or result.get("error"),
        "wall_seconds_per_question": float(seconds),
        "text_input_tokens": int(_sum_agent_metric(result, "tokens", "text_input")),
        "latent_input_tokens": int(_sum_agent_metric(result, "tokens", "latent_input")),
        "text_output_tokens": int(_sum_agent_metric(result, "tokens", "text_output")),
        "latent_output_tokens": int(_sum_agent_metric(result, "tokens", "latent_output")),
        "prefill_seconds": _sum_agent_metric(result, "timing", "prefill_seconds"),
        "latent_decode_seconds": _sum_agent_metric(result, "timing", "latent_decode_seconds"),
        "alignment_seconds": _sum_agent_metric(result, "timing", "alignment_seconds"),
        "text_decode_seconds": _sum_agent_metric(result, "timing", "text_decode_seconds"),
    }


def _collect_cell(args, dataset, indexed_items, wrapper, alignment, condition, seed, logger):
    set_seed(seed)
    method_args = _method_args(args, dataset, alignment, seed)
    wrapper.args = method_args
    wrapper.align_method = alignment
    if alignment == "kernel" and getattr(wrapper, "_c4_kernel_seed", None) != seed:
        # run.sh starts a fresh process per repeat, so its ORF map follows that
        # repeat's seed. Rebuild once here, then share it with the paired noise cell.
        wrapper._alignment_states = {
            key: value for key, value in wrapper._alignment_states.items()
            if key[2] != "kernel"
        }
        wrapper._c4_kernel_seed = seed
    source_transform = None
    embedding_transform = None
    if condition in {"clean", "pre_replace"}:
        source_transform = VectorReplaceTransform(
            dataset, condition, args.noise_seed_offset + seed, alignment, seed,
            NOISE_SITES[condition], args.latent_steps,
        )
    elif condition == "post_replace":
        embedding_transform = VectorReplaceTransform(
            dataset, condition, args.noise_seed_offset + seed, alignment, seed,
            NOISE_SITES[condition], args.latent_steps,
        )
    else:
        raise ValueError(f"Unknown C4 condition: {condition}")
    transform = source_transform or embedding_transform
    method = LatentMASMethod(
        wrapper,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        args=method_args,
        latent_hidden_transform=source_transform,
        latent_embedding_transform=embedding_transform,
    )
    answer_rows = []
    for start in range(0, len(indexed_items), args.generate_bs):
        batch = indexed_items[start:start + args.generate_bs]
        if transform is not None:
            transform.begin_batch([item_id for item_id, _ in batch])
        started = time.perf_counter()
        failure = None
        try:
            results = method.run_batch([item for _, item in batch])
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            results = [{"prediction": None, "raw_prediction": "", "correct": False, "error": failure, "agents": []} for _ in batch]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        per_question = (time.perf_counter() - started) / len(batch)
        for (item_id, item), result in zip(batch, results):
            answer_rows.append(_answer_row(args, dataset, item_id, item, result, alignment, condition, seed, per_question, failure))
        logger.info("C4 %s %s/%s seed=%d completed %d/%d", dataset, alignment, condition, seed, min(start + len(batch), len(indexed_items)), len(indexed_items))
    return answer_rows, transform.rows


def _two_level_paired_ci(clean, noise, replicates, seed):
    by_seed = {}
    for key in clean:
        generation_seed, item_id = key
        if key in noise:
            by_seed.setdefault(generation_seed, []).append(noise[key] - clean[key])
    seeds = sorted(by_seed)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for selected in sampled_seeds:
            array = np.asarray(by_seed[int(selected)], dtype=np.float64)
            seed_means.append(float(rng.choice(array, size=len(array), replace=True).mean()))
        values.append(np.mean(seed_means))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _summarize(answer_rows, hidden_rows, args):
    series = {}
    mean_metrics = (
        "wall_seconds_per_question", "text_input_tokens", "latent_input_tokens",
        "text_output_tokens", "latent_output_tokens", "prefill_seconds",
        "latent_decode_seconds", "alignment_seconds", "text_decode_seconds",
    )
    for dataset in DATASETS:
        for alignment in ALIGNMENTS:
            for condition in CONDITIONS:
                selected = [
                    row for row in answer_rows
                    if row["dataset"] == dataset
                    and row["alignment"] == alignment
                    and row["condition"] == condition
                ]
                seeds = []
                for seed in args.repeat_seeds:
                    rows = [row for row in selected if row["generation_seed"] == seed]
                    seed_summary = {
                        "seed": seed,
                        "correct": sum(row["correct"] for row in rows),
                        "total": len(rows),
                        "accuracy": float(np.mean([row["correct"] for row in rows])),
                        "parse_success_rate": float(np.mean([row["parse_success"] for row in rows])),
                    }
                    seed_summary.update({
                        f"mean_{metric}": float(np.mean([row[metric] for row in rows]))
                        for metric in mean_metrics
                    })
                    seeds.append(seed_summary)
                aggregate = {
                    "seeds": seeds,
                    "accuracy_mean": float(np.mean([row["accuracy"] for row in seeds])),
                    "accuracy_std": float(np.std([row["accuracy"] for row in seeds], ddof=1)),
                    "parse_success_rate_mean": float(np.mean([row["parse_success_rate"] for row in seeds])),
                }
                aggregate.update({
                    f"{metric}_mean": float(np.mean([row[f"mean_{metric}"] for row in seeds]))
                    for metric in mean_metrics
                })
                aggregate["wall_seconds_mean"] = aggregate["wall_seconds_per_question_mean"]
                series[f"{dataset}.{alignment}.{condition}"] = aggregate

    paired = {}
    for dataset in DATASETS:
        paired[dataset] = {}
        for alignment in ALIGNMENTS:
            clean = {
                (row["generation_seed"], row["item_id"]): float(row["correct"])
                for row in answer_rows
                if row["dataset"] == dataset
                and row["alignment"] == alignment
                and row["condition"] == "clean"
            }
            comparison = {}
            for condition in ("pre_replace", "post_replace"):
                replaced = {
                    (row["generation_seed"], row["item_id"]): float(row["correct"])
                    for row in answer_rows
                    if row["dataset"] == dataset
                    and row["alignment"] == alignment
                    and row["condition"] == condition
                }
                deltas = [replaced[key] - clean[key] for key in clean]
                flips = {
                    "clean_only_correct": 0, "replace_only_correct": 0,
                    "both_correct": 0, "both_wrong": 0,
                }
                for key in clean:
                    state = (bool(clean[key]), bool(replaced[key]))
                    label = {
                        (True, False): "clean_only_correct",
                        (False, True): "replace_only_correct",
                        (True, True): "both_correct",
                        (False, False): "both_wrong",
                    }[state]
                    flips[label] += 1
                comparison[f"{condition}_minus_clean"] = {
                    "accuracy_delta": float(np.mean(deltas)),
                    "bootstrap_95_ci": _two_level_paired_ci(
                        clean, replaced, args.bootstrap_replicates, args.probe_seed
                    ),
                    "answer_transitions": flips,
                }
            paired[dataset][alignment] = comparison

    diagnostics = {}
    for dataset in DATASETS:
        diagnostics[dataset] = {}
        for condition in CONDITIONS:
            selected = [
                row for row in hidden_rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            diagnostics[dataset][condition] = {
                "noise_site": NOISE_SITES[condition],
                "mean_original_effective_cosine": float(np.mean([
                    row["original_effective_cosine"] for row in selected
                ])),
                "mean_effective_to_original_norm_ratio": float(np.mean([
                    row["effective_norm"] / max(row["original_norm"], 1e-12)
                    for row in selected
                ])),
                "finite_fraction": float(np.mean([row["finite"] for row in selected])),
            }
    return {
        "study": "c4",
        "design": {
            "model": args.model_name,
            "datasets": {
                dataset: {"split": config["split"], "question_count": args.max_questions}
                for dataset, config in DATASETS.items()
            },
            "prompt": "hierarchical",
            "K": int(args.latent_steps),
            "alignments": list(ALIGNMENTS),
            "conditions": list(CONDITIONS),
            "repeat_seeds": list(args.repeat_seeds),
        },
        "series": series,
        "paired_comparison": paired,
        "hidden_diagnostics": diagnostics,
    }
def _plot(summary, path):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    width = 0.22
    datasets = list(DATASETS)
    x = np.arange(len(datasets))
    for offset, condition in enumerate(CONDITIONS):
        entries = [
            summary["series"][f"{dataset}.kernel.{condition}"]
            for dataset in datasets
        ]
        positions = x + (offset - 1) * width
        axes[0].bar(
            positions, [entry["accuracy_mean"] for entry in entries], width,
            color=COLORS[condition], label=condition, alpha=0.85,
        )
        for position, entry in zip(positions, entries):
            axes[0].scatter(
                [position] * len(entry["seeds"]),
                [value["accuracy"] for value in entry["seeds"]],
                color="black", s=15, zorder=3,
            )
        axes[1].bar(
            positions, [entry["wall_seconds_mean"] for entry in entries], width,
            color=COLORS[condition], alpha=0.85,
        )
        axes[2].bar(
            positions, [entry["text_output_tokens_mean"] for entry in entries], width,
            color=COLORS[condition], alpha=0.85,
        )
    for axis, ylabel in zip(
        axes,
        ("Accuracy", "Mean wall seconds / question", "Mean text output tokens / question"),
    ):
        axis.set_xticks(x, ["AIME2025", "GPQA-Diamond"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    figure.suptitle("C4: Qwen3-8B hierarchical pre/post alignment replacement")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
def _run_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(
        json.dumps(vars(args), sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    path = RUNS_DIR / (
        f"aime2025_gpqa_diamond_c4_Qwen3-8B_k{args.latent_steps}_"
        f"q{args.max_questions}_r{len(args.repeat_seeds)}_{timestamp}_{digest}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_c4(args, logger: logging.Logger | None = None):
    logger = logger or logging.getLogger("latent_cot.c4")
    indexed_by_dataset = {
        dataset: _items(dataset, args.max_questions) for dataset in DATASETS
    }
    run_dir = _run_dir(args)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "status": "running",
        "study": "c4",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dataset_identity": {
            dataset: _dataset_identity(items)
            for dataset, items in indexed_by_dataset.items()
        },
        "cache_cells": [],
    }
    _json(manifest_path, manifest)
    wrapper = None
    model_info = None
    answer_rows, hidden_rows = [], []
    try:
        for dataset, indexed_items in indexed_by_dataset.items():
            for seed in args.repeat_seeds:
                for alignment in ALIGNMENTS:
                    for condition in CONDITIONS:
                        identity = _identity(
                            args, dataset, indexed_items, alignment, condition, seed
                        )
                        cached_answers, cached_hidden, cell_manifest, paths = _load_cell(
                            args, identity
                        )
                        cache_hit = cached_answers is not None
                        if not cache_hit:
                            if wrapper is None:
                                model_args = _method_args(args, dataset, alignment, seed)
                                wrapper = ModelWrapper(
                                    args.model_name, args.device,
                                    use_vllm=False, args=model_args,
                                )
                                config = wrapper.model.config.to_dict()
                                model_info = {
                                    "name": args.model_name,
                                    "resolved_commit": getattr(
                                        wrapper.model.config, "_commit_hash", None
                                    ),
                                    "config_sha256": hashlib.sha256(
                                        json.dumps(
                                            config, sort_keys=True, default=str
                                        ).encode()
                                    ).hexdigest(),
                                    "parameter_dtype": str(
                                        next(wrapper.model.parameters()).dtype
                                    ),
                                }
                            cached_answers, cached_hidden = _collect_cell(
                                args, dataset, indexed_items, wrapper,
                                alignment, condition, seed, logger,
                            )
                            if len(cached_answers) != args.max_questions:
                                raise RuntimeError("C4 answer-row invariant failed")
                            expected_hidden = (
                                args.max_questions
                                * len(ROLES)
                                * len(_sampled_steps(
                                    args.latent_steps, NOISE_SITES[condition]
                                ))
                            )
                            if len(cached_hidden) != expected_hidden:
                                raise RuntimeError(
                                    "C4 hidden-row invariant failed: "
                                    f"{len(cached_hidden)} != {expected_hidden}"
                                )
                            cell_manifest = _save_cell(
                                paths, identity, cached_answers, cached_hidden, model_info
                            )
                        else:
                            model_info = model_info or cell_manifest.get("model")
                            logger.info("C4 cache hit: %s", paths[2])
                        answer_rows.extend(cached_answers)
                        hidden_rows.extend(cached_hidden)
                        manifest["cache_cells"].append({
                            "dataset": dataset,
                            "alignment": alignment,
                            "condition": condition,
                            "seed": seed,
                            "cache_hit": cache_hit,
                            "manifest": str(paths[2]),
                        })
                        _json(manifest_path, manifest)

        expected_answers = (
            len(DATASETS) * len(ALIGNMENTS) * len(CONDITIONS)
            * len(args.repeat_seeds) * args.max_questions
        )
        if len(answer_rows) != expected_answers:
            raise RuntimeError(
                f"C4 total answer-row invariant failed: "
                f"{len(answer_rows)} != {expected_answers}"
            )
        metrics_path = run_dir / "metrics" / "c4_accuracy_cost_by_question.parquet"
        hidden_path = run_dir / "metrics" / "c4_hidden_diagnostics.parquet"
        summary_path = run_dir / "summaries" / "c4_summary.json"
        figure_path = run_dir / "figures" / "c4_clean_pre_post_replace.pdf"
        _parquet(metrics_path, answer_rows)
        _parquet(hidden_path, hidden_rows)
        summary = _summarize(answer_rows, hidden_rows, args)
        _json(summary_path, summary)
        _plot(summary, figure_path)
        manifest.update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_info,
            "row_count": {
                "answers": len(answer_rows), "hidden": len(hidden_rows)
            },
            "all_cache_hits": all(
                cell["cache_hit"] for cell in manifest["cache_cells"]
            ),
            "artifacts": {
                "metrics": str(metrics_path),
                "hidden_diagnostics": str(hidden_path),
                "summary": str(summary_path),
                "figure": str(figure_path),
            },
        })
        _json(manifest_path, manifest)
        return run_dir
    except Exception as error:
        manifest.update({
            "status": "failed",
            "failed_at": datetime.now().isoformat(timespec="seconds"),
            "error": f"{type(error).__name__}: {error}",
        })
        _json(manifest_path, manifest)
        raise