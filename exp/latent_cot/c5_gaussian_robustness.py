"""C5: Planner-only additive-Gaussian robustness of linear and kernel alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F

from data import load_aime2024, load_gsm8k, load_mbppplus
from methods import Agent
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from utils import set_seed

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "exp_result" / "latent_cot" / "runs"
CACHE_DIR = ROOT / "exp" / "cache" / "latent_cot_c5"
PARAMS_PATH = ROOT / "params_dict.json"
SCHEMA_VERSION = 1
ALIGNMENTS = ("linear", "kernel")
CONDITIONS = ("clean", "gaussian_005")
COLORS = {"linear": "#f58518", "kernel": "#54a24b"}
HATCHES = {"clean": "", "gaussian_005": "//"}
C5_AGENTS = (Agent("Planner", "planner"), Agent("Judger", "judger"))
DATASETS = {
    "gsm8k": {"split": "test", "task": "gsm8k", "loader": load_gsm8k, "label": "GSM8K"},
    "mbppplus": {"split": "test", "task": "mbppplus", "loader": load_mbppplus, "label": "MBPP+"},
    "aime2024": {"split": "train", "task": "aime2024", "loader": load_aime2024, "label": "AIME2024"},
}


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(list(rows)), tmp, compression="zstd")
    tmp.replace(path)


def _read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha():
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "methods/latent_mas.py", ROOT / "models.py", ROOT / "prompts.py", PARAMS_PATH):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _runtime(args, dataset):
    config = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))[DATASETS[dataset]["task"]]
    if args.c5_dev_allow_override:
        return {"latent_steps": args.latent_steps, "max_new_tokens": args.max_new_tokens,
                "generate_bs": args.generate_bs, "source": "c5_dev_override"}
    return {"latent_steps": int(config["latent_steps"]["hierarchical"]),
            "max_new_tokens": int(config["max_token"]),
            "generate_bs": int(config["generation_bs"]), "source": "params_dict.json"}


def _items(dataset, count, sample_seed):
    config = DATASETS[dataset]
    available = list(enumerate(config["loader"](split=config["split"])))
    if len(available) < count:
        raise RuntimeError(f"Requested {count} {dataset} questions, loaded {len(available)}")
    if len(available) == count:
        return available
    offset = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:4], "little")
    ids = sorted(np.random.default_rng(sample_seed + offset).choice(len(available), count, replace=False).tolist())
    return [available[index] for index in ids]


def _dataset_identity(items, sample_seed):
    payload = [{"item_id": i, "question": x["question"], "gold": x.get("gold", "")} for i, x in items]
    return {"selection": "all_if_exact_else_fixed_random_without_replacement",
            "sample_seed": sample_seed, "item_ids": [i for i, _ in items],
            "content_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}


def _kernel_seed(args, repeat_index):
    return int(args.kernel_seed) + repeat_index


def _alpha(args, condition):
    return 0.0 if condition == "clean" else float(args.noise_alpha)


def _identity(args, dataset, items, runtime, alignment, condition, repeat_index, seed):
    return {
        "schema_version": SCHEMA_VERSION, "experiment": "c5_planner_gaussian_alignment_robustness",
        "dataset": dataset, "split": DATASETS[dataset]["split"],
        "dataset_identity": _dataset_identity(items, args.sample_seed), "model_name": args.model_name,
        "prompt": "hierarchical", "roles": ["planner", "judger"], "latent_roles": ["planner"],
        "latent_steps": runtime["latent_steps"], "step_source": runtime["source"],
        "alignment": alignment, "condition": condition, "repeat_index": repeat_index,
        "generation_seed": seed,
        "noise": {"site": "pre_alignment_source_hidden",
                  "distribution": "isotropic_gaussian_scaled_by_source_hidden_l2_norm",
                  "alpha": _alpha(args, condition),
                  "coordinate_std": "alpha_times_hidden_l2_norm_div_sqrt_hidden_dim",
                  "base_seed": args.noise_seed_offset + seed, "keying": "dataset_item_role_step_repeat"},
        "generation": {"temperature": args.temperature, "top_p": args.top_p,
                       "max_new_tokens": runtime["max_new_tokens"], "batch_size": runtime["generate_bs"],
                       "manual_think": bool(args.think)},
        "alignment_config": {"linear_ridge": args.align_ridge, "kernel_features": args.kernel_features,
                             "kernel_temperature": args.kernel_temperature,
                             "kernel_seed": _kernel_seed(args, repeat_index),
                             "kernel_chunk_size": args.kernel_chunk_size},
        "implementation_sha256": _implementation_sha(),
    }


def _cache_paths(identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    stem = f"c5_{identity['dataset']}_{identity['alignment']}_{identity['condition']}_seed{identity['generation_seed']}_{digest}"
    return (CACHE_DIR / f"{stem}.answers.parquet", CACHE_DIR / f"{stem}.diagnostics.parquet",
            CACHE_DIR / f"{stem}.manifest.json")


def _load_cell(args, identity):
    answers, diagnostics, manifest_path = _cache_paths(identity)
    present = [path.exists() for path in (answers, diagnostics, manifest_path)]
    if any(present) and not all(present) and not args.force_recollect:
        raise RuntimeError(f"Incomplete C5 cache cell; use --force_recollect: {manifest_path}")
    if all(present) and not args.force_recollect:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_identity") != identity:
            raise RuntimeError(f"C5 cache identity mismatch: {manifest_path}")
        if _sha(answers) != manifest["sha256"]["answers"] or _sha(diagnostics) != manifest["sha256"]["diagnostics"]:
            raise RuntimeError(f"C5 cache integrity check failed: {manifest_path}")
        return _read_parquet(answers), _read_parquet(diagnostics), manifest, (answers, diagnostics, manifest_path)
    if args.reuse_trajectory:
        raise FileNotFoundError(f"C5 cache is absent: {manifest_path}")
    return None, None, None, (answers, diagnostics, manifest_path)


def _save_cell(paths, identity, answers, diagnostics, model_info, setup_seconds):
    answer_path, diagnostic_path, manifest_path = paths
    _parquet(answer_path, answers); _parquet(diagnostic_path, diagnostics)
    manifest = {"schema_version": SCHEMA_VERSION, "created_at": datetime.now().isoformat(timespec="seconds"),
                "cache_identity": identity, "row_count": {"answers": len(answers), "diagnostics": len(diagnostics)},
                "sha256": {"answers": _sha(answer_path), "diagnostics": _sha(diagnostic_path)},
                "alignment_setup_seconds_excluded_from_online_wall_time": setup_seconds, "model": model_info}
    _json(manifest_path, manifest)
    return manifest

def _sampled_steps(steps):
    last = max(0, steps - 1)
    return tuple(sorted({0, last // 4, last // 2, 3 * last // 4, last}))


class AdditiveGaussianTransform:
    def __init__(self, dataset, condition, alpha, base_seed, alignment, repeat_index, generation_seed, steps):
        self.dataset, self.condition, self.alpha = dataset, condition, float(alpha)
        self.base_seed, self.alignment = int(base_seed), alignment
        self.repeat_index, self.generation_seed, self.steps = repeat_index, generation_seed, int(steps)
        self.sampled_steps, self.item_ids, self.rows = set(_sampled_steps(steps)), [], []
        self.pending = []

    def begin_batch(self, item_ids):
        if self.pending:
            raise RuntimeError("C5 diagnostics must be flushed before the next batch")
        self.item_ids = [int(value) for value in item_ids]

    def _seed(self, item_id, role, step):
        key = f"{self.base_seed}|{self.dataset}|{item_id}|{role}|{step}|{self.repeat_index}"
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)

    @torch.inference_mode()
    def __call__(self, role, step, vector):
        if role != "planner":
            raise RuntimeError(f"C5 transform unexpectedly called for role={role}")
        if int(step) >= self.steps:
            return vector
        original = vector.detach().float()
        effective, noises = [], []
        for index, item_id in enumerate(self.item_ids):
            if self.alpha == 0:
                noise = torch.zeros_like(original[index])
            else:
                generator = torch.Generator(device=vector.device)
                generator.manual_seed(self._seed(item_id, role, step))
                standard = torch.randn(original[index].shape, dtype=torch.float32,
                                       device=vector.device, generator=generator)
                noise = standard * (self.alpha * original[index].norm() / math.sqrt(original.shape[-1]))
            noises.append(noise); effective.append(original[index] + noise)
        effective = torch.stack(effective)
        if int(step) in self.sampled_steps:
            for index, item_id in enumerate(self.item_ids):
                metadata = {
                    "dataset": self.dataset, "item_id": item_id, "alignment": self.alignment,
                    "condition": self.condition, "noise_site": "pre_alignment_source_hidden",
                    "noise_alpha": self.alpha, "repeat_index": self.repeat_index,
                    "generation_seed": self.generation_seed, "agent": role, "local_step": int(step),
                    "direction_seed": self._seed(item_id, role, step),
                }
                self.pending.append((
                    metadata, original[index].detach(), noises[index].detach(), effective[index].detach()
                ))
        return effective.to(device=vector.device, dtype=vector.dtype)

    @torch.inference_mode()
    def flush_diagnostics(self):
        for metadata, source, noise, perturbed in self.pending:
            source, noise, perturbed = source.float().cpu(), noise.float().cpu(), perturbed.float().cpu()
            source_norm, noise_norm = source.norm(), noise.norm()
            self.rows.append({
                **metadata,
                "original_norm": float(source_norm.item()),
                "noise_norm": float(noise_norm.item()),
                "perturbed_norm": float(perturbed.norm().item()),
                "relative_noise_norm": float((noise_norm / source_norm.clamp_min(1e-12)).item()),
                "original_perturbed_cosine": float(F.cosine_similarity(
                    source.unsqueeze(0), perturbed.unsqueeze(0), dim=-1
                )[0].item()),
                "noise_fingerprint": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                "finite": bool(torch.isfinite(perturbed).all().item()),
            })
        self.pending.clear()


def _method_args(args, dataset, runtime, alignment, repeat_index, seed):
    return argparse.Namespace(**{
        **vars(args), "task": DATASETS[dataset]["task"], "method": "latent_mas",
        "prompt": "hierarchical", "align_method": alignment,
        "latent_steps": runtime["latent_steps"], "max_new_tokens": runtime["max_new_tokens"],
        "generate_bs": runtime["generate_bs"], "seed": seed,
        "kernel_seed": _kernel_seed(args, repeat_index), "device2": args.device,
        "sequential_info_only": False, "latent_only": False, "use_vllm": False,
        "use_second_HF_model": False, "enable_prefix_caching": False,
        "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9,
        "soft_temperature": args.kernel_temperature,
    })


def _sum_metric(result, section, key, role=None):
    return sum(float(agent.get("metrics", {}).get(section, {}).get(key, 0) or 0)
               for agent in result.get("agents", []) if role is None or agent.get("role") == role)


def _answer_row(runtime, dataset, item_id, item, result, alignment, condition,
                alpha, repeat_index, seed, seconds, failure):
    prediction = result.get("prediction")
    return {
        "dataset": dataset, "split": DATASETS[dataset]["split"], "item_id": item_id,
        "alignment": alignment, "condition": condition, "noise_alpha": alpha,
        "repeat_index": repeat_index, "generation_seed": seed,
        "latent_steps": runtime["latent_steps"], "roles": "planner,judger",
        "question": item["question"], "gold": item.get("gold", ""), "prediction": prediction,
        "raw_prediction": result.get("raw_prediction", ""), "correct": bool(result.get("correct", False)),
        "parse_success": prediction is not None, "failure_reason": failure or result.get("error"),
        "wall_seconds_per_question": seconds,
        "text_input_tokens": int(_sum_metric(result, "tokens", "text_input")),
        "text_output_tokens": int(_sum_metric(result, "tokens", "text_output")),
        "judger_text_output_tokens": int(_sum_metric(result, "tokens", "text_output", "judger")),
        "latent_input_tokens": int(_sum_metric(result, "tokens", "latent_input")),
        "latent_output_tokens": int(_sum_metric(result, "tokens", "latent_output")),
        "prefill_seconds": _sum_metric(result, "timing", "prefill_seconds"),
        "latent_decode_seconds": _sum_metric(result, "timing", "latent_decode_seconds"),
        "alignment_seconds": _sum_metric(result, "timing", "alignment_seconds"),
        "text_decode_seconds": _sum_metric(result, "timing", "text_decode_seconds"),
    }


def _collect_cell(args, dataset, runtime, items, wrapper, alignment, condition,
                  repeat_index, seed, logger):
    set_seed(seed)
    method_args = _method_args(args, dataset, runtime, alignment, repeat_index, seed)
    wrapper.args, wrapper.align_method = method_args, alignment
    if alignment == "kernel" and getattr(wrapper, "_c5_kernel_seed", None) != method_args.kernel_seed:
        wrapper._alignment_states = {key: value for key, value in wrapper._alignment_states.items()
                                     if key[2] != "kernel"}
        wrapper._c5_kernel_seed = method_args.kernel_seed
    setup_started = time.perf_counter()
    wrapper._ensure_alignment_state(wrapper.model, wrapper.model)
    setup_seconds = time.perf_counter() - setup_started
    alpha = _alpha(args, condition)
    transform = AdditiveGaussianTransform(dataset, condition, alpha, args.noise_seed_offset + seed,
                                          alignment, repeat_index, seed, runtime["latent_steps"])
    method = LatentMASMethod(
        wrapper, latent_steps=runtime["latent_steps"],
        judger_max_new_tokens=runtime["max_new_tokens"], temperature=args.temperature,
        top_p=args.top_p, generate_bs=runtime["generate_bs"], args=method_args,
        latent_hidden_transform=transform,
    )
    method.agents = list(C5_AGENTS)
    answers = []
    for start in range(0, len(items), runtime["generate_bs"]):
        batch = items[start:start + runtime["generate_bs"]]
        transform.begin_batch([item_id for item_id, _ in batch])
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started, failure = time.perf_counter(), None
        try:
            results = method.run_batch([item for _, item in batch])
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            results = [{"prediction": None, "raw_prediction": "", "correct": False,
                        "error": failure, "agents": []} for _ in batch]
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        seconds = (time.perf_counter() - started) / len(batch)
        transform.flush_diagnostics()
        answers.extend(_answer_row(runtime, dataset, item_id, item, result, alignment, condition,
                                   alpha, repeat_index, seed, seconds, failure)
                       for (item_id, item), result in zip(batch, results))
        logger.info("C5 %s %s/%s repeat=%d seed=%d completed %d/%d", dataset, alignment,
                    condition, repeat_index, seed, min(start + len(batch), len(items)), len(items))
    return answers, transform.rows, setup_seconds


def _paired_ci(values, replicates, seed):
    by_repeat = {}
    for (repeat_index, _), value in values.items():
        by_repeat.setdefault(repeat_index, []).append(float(value))
    repeats, rng, draws = sorted(by_repeat), np.random.default_rng(seed), []
    for _ in range(replicates):
        selected = rng.choice(repeats, len(repeats), replace=True)
        means = []
        for repeat_index in selected:
            array = np.asarray(by_repeat[int(repeat_index)])
            means.append(float(rng.choice(array, len(array), replace=True).mean()))
        draws.append(np.mean(means))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _summarize(answers, diagnostics, args, runtimes):
    noisy_directions = {}
    for row in diagnostics:
        if row["condition"] != "gaussian_005":
            continue
        key = (row["dataset"], row["repeat_index"], row["item_id"], row["agent"], row["local_step"])
        noisy_directions.setdefault(key, {})[row["alignment"]] = row["direction_seed"]
    if any(set(value) != set(ALIGNMENTS) or len(set(value.values())) != 1
           for value in noisy_directions.values()):
        raise RuntimeError("C5 linear/kernel Gaussian-direction pairing invariant failed")
    mean_metrics = ("wall_seconds_per_question", "judger_text_output_tokens", "text_input_tokens",
                    "text_output_tokens", "latent_input_tokens", "latent_output_tokens",
                    "prefill_seconds", "latent_decode_seconds", "alignment_seconds", "text_decode_seconds")
    series = {}
    for dataset in DATASETS:
        for alignment in ALIGNMENTS:
            for condition in CONDITIONS:
                selected = [row for row in answers if row["dataset"] == dataset
                            and row["alignment"] == alignment and row["condition"] == condition]
                repeats = []
                for repeat_index, seed in enumerate(args.repeat_seeds):
                    rows = [row for row in selected if row["repeat_index"] == repeat_index]
                    entry = {"repeat_index": repeat_index, "generation_seed": seed,
                             "correct": sum(row["correct"] for row in rows), "total": len(rows),
                             "accuracy": float(np.mean([row["correct"] for row in rows])),
                             "parse_success_rate": float(np.mean([row["parse_success"] for row in rows]))}
                    entry.update({f"mean_{metric}": float(np.mean([row[metric] for row in rows]))
                                  for metric in mean_metrics})
                    repeats.append(entry)
                aggregate = {"repeats": repeats,
                             "accuracy_mean": float(np.mean([row["accuracy"] for row in repeats])),
                             "accuracy_std": float(np.std([row["accuracy"] for row in repeats], ddof=1)) if len(repeats) > 1 else 0.0,
                             "parse_success_rate_mean": float(np.mean([row["parse_success_rate"] for row in repeats]))}
                aggregate.update({f"{metric}_mean": float(np.mean([row[f'mean_{metric}'] for row in repeats]))
                                  for metric in mean_metrics})
                series[f"{dataset}.{alignment}.{condition}"] = aggregate

    paired = {}
    for dataset_index, dataset in enumerate(DATASETS):
        paired[dataset], alignment_deltas = {}, {}
        for alignment_index, alignment in enumerate(ALIGNMENTS):
            clean = {(row["repeat_index"], row["item_id"]): row for row in answers
                     if row["dataset"] == dataset and row["alignment"] == alignment
                     and row["condition"] == "clean"}
            noisy = {(row["repeat_index"], row["item_id"]): row for row in answers
                     if row["dataset"] == dataset and row["alignment"] == alignment
                     and row["condition"] == "gaussian_005"}
            if clean.keys() != noisy.keys():
                raise RuntimeError(f"C5 paired-key mismatch for {dataset}/{alignment}")
            delta = {key: float(noisy[key]["correct"]) - float(clean[key]["correct"]) for key in clean}
            alignment_deltas[alignment] = delta
            transitions = {"clean_only_correct": 0, "noisy_only_correct": 0,
                           "both_correct": 0, "both_wrong": 0}
            labels = {(True, False): "clean_only_correct", (False, True): "noisy_only_correct",
                      (True, True): "both_correct", (False, False): "both_wrong"}
            for key in clean:
                transitions[labels[(bool(clean[key]["correct"]), bool(noisy[key]["correct"]))]] += 1
            paired[dataset][alignment] = {
                "noisy_minus_clean_accuracy": float(np.mean(list(delta.values()))),
                "bootstrap_95_ci": _paired_ci(delta, args.bootstrap_replicates,
                                               args.probe_seed + dataset_index * 10 + alignment_index),
                "answer_transitions": transitions,
                "wall_seconds_per_question_delta": float(np.mean([
                    noisy[key]["wall_seconds_per_question"] - clean[key]["wall_seconds_per_question"] for key in clean])),
                "judger_text_output_tokens_delta": float(np.mean([
                    noisy[key]["judger_text_output_tokens"] - clean[key]["judger_text_output_tokens"] for key in clean])),
            }
        difference = {key: alignment_deltas["kernel"][key] - alignment_deltas["linear"][key]
                      for key in alignment_deltas["linear"]}
        paired[dataset]["kernel_minus_linear_robustness"] = {
            "difference_in_differences": float(np.mean(list(difference.values()))),
            "bootstrap_95_ci": _paired_ci(difference, args.bootstrap_replicates,
                                           args.probe_seed + 100 + dataset_index)}

    diagnostic_summary = {}
    for dataset in DATASETS:
        diagnostic_summary[dataset] = {}
        for condition in CONDITIONS:
            rows = [row for row in diagnostics if row["dataset"] == dataset and row["condition"] == condition]
            diagnostic_summary[dataset][condition] = {
                "noise_alpha": _alpha(args, condition),
                "mean_relative_noise_norm": float(np.mean([row["relative_noise_norm"] for row in rows])),
                "mean_original_perturbed_cosine": float(np.mean([row["original_perturbed_cosine"] for row in rows])),
                "finite_fraction": float(np.mean([row["finite"] for row in rows])),
            }
    return {"study": "c5", "design": {"model": args.model_name,
            "datasets": {dataset: {"split": DATASETS[dataset]["split"],
                                    "question_count": args.max_questions, **runtimes[dataset]}
                         for dataset in DATASETS},
            "prompt": "hierarchical", "roles": ["planner", "judger"], "latent_roles": ["planner"],
            "alignments": list(ALIGNMENTS), "conditions": list(CONDITIONS),
            "noise_alpha": args.noise_alpha, "repeat_seeds": list(args.repeat_seeds)},
            "series": series, "paired_comparison": paired,
            "perturbation_diagnostics": diagnostic_summary}


def _plot_main(summary, path):
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    datasets, x, width = list(DATASETS), np.arange(len(DATASETS)), 0.19
    metrics = (("accuracy_mean", "accuracy", "Accuracy"),
               ("wall_seconds_per_question_mean", "mean_wall_seconds_per_question", "Mean wall seconds / question"),
               ("judger_text_output_tokens_mean", "mean_judger_text_output_tokens", "Mean Judger text output tokens / question"))
    combinations = [(alignment, condition) for alignment in ALIGNMENTS for condition in CONDITIONS]
    for combo_index, (alignment, condition) in enumerate(combinations):
        entries = [summary["series"][f"{dataset}.{alignment}.{condition}"] for dataset in datasets]
        positions = x + (combo_index - 1.5) * width
        for axis, (metric, repeat_metric, _) in zip(axes, metrics):
            axis.bar(positions, [entry[metric] for entry in entries], width, color=COLORS[alignment],
                     edgecolor="black", linewidth=0.6, hatch=HATCHES[condition], alpha=0.88)
            for position, entry in zip(positions, entries):
                axis.scatter([position] * len(entry["repeats"]),
                             [repeat[repeat_metric] for repeat in entry["repeats"]],
                             color="black", s=9, alpha=0.6, zorder=3)
    for axis, (_, _, ylabel) in zip(axes, metrics):
        axis.set_xticks(x, [DATASETS[dataset]["label"] for dataset in datasets])
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 1)
    axes[0].legend(handles=[Patch(facecolor=COLORS[a], edgecolor="black", label=a) for a in ALIGNMENTS]
                   + [Patch(facecolor="white", edgecolor="black", hatch=HATCHES[c], label=c) for c in CONDITIONS],
                   fontsize=9)
    figure.suptitle("C5: Planner hidden-state perturbation robustness (Planner to Judger)")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path); plt.close(figure)


def _plot_degradation(summary, path):
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    datasets, x, width = list(DATASETS), np.arange(len(DATASETS)), 0.34
    for index, alignment in enumerate(ALIGNMENTS):
        entries = [summary["paired_comparison"][dataset][alignment] for dataset in datasets]
        values = [entry["noisy_minus_clean_accuracy"] for entry in entries]
        errors = [[value - entry["bootstrap_95_ci"][0] for value, entry in zip(values, entries)],
                  [entry["bootstrap_95_ci"][1] - value for value, entry in zip(values, entries)]]
        positions = x + (index - 0.5) * width
        axis.bar(positions, values, width, color=COLORS[alignment], label=alignment, alpha=0.88)
        axis.errorbar(positions, values, yerr=errors, fmt="none", color="black", capsize=3)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, [DATASETS[d]["label"] for d in datasets])
    axis.set_ylabel("Accuracy(noisy) - Accuracy(clean)"); axis.set_title("C5 paired accuracy degradation")
    axis.grid(axis="y", alpha=0.25); axis.legend(); figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path); plt.close(figure)


def _run_dir(args, runtimes):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {**vars(args), "runtimes": runtimes, "implementation_sha256": _implementation_sha()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]
    steps = "-".join(f"{dataset}{runtimes[dataset]['latent_steps']}" for dataset in DATASETS)
    name = (f"gsm8k_mbppplus_aime2024_c5_Qwen3-8B_{steps}_q{args.max_questions}_"
            f"r{len(args.repeat_seeds)}_{timestamp}_{digest}")
    path, suffix = RUNS_DIR / name, 1
    while path.exists():
        path, suffix = RUNS_DIR / f"{name}_{suffix:02d}", suffix + 1
    path.mkdir(parents=True)
    return path


def run_c5(args, logger=None):
    logger = logger or logging.getLogger("latent_cot.c5")
    runtimes = {dataset: _runtime(args, dataset) for dataset in DATASETS}
    run_dir = _run_dir(args, runtimes)
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics/c5_accuracy_cost_by_question.parquet"
    diagnostics_path = run_dir / "metrics/c5_perturbation_diagnostics.parquet"
    summary_path = run_dir / "summaries/c5_summary.json"
    figure_path = run_dir / "figures/c5_accuracy_time_tokens.pdf"
    degradation_path = run_dir / "figures/c5_paired_accuracy_degradation.pdf"
    manifest = {"status": "loading_datasets", "study": "c5",
                "started_at": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
                "dataset_runtime": runtimes, "dataset_identity": {}, "cache_cells": [],
                "artifacts": {"metrics": str(metrics_path), "perturbation_diagnostics": str(diagnostics_path),
                              "summary": str(summary_path), "main_figure": str(figure_path),
                              "degradation_figure": str(degradation_path)}}
    _json(manifest_path, manifest)
    wrapper = model_info = None
    all_answers, all_diagnostics = [], []
    try:
        indexed = {dataset: _items(dataset, args.max_questions, args.sample_seed) for dataset in DATASETS}
        manifest["dataset_identity"] = {dataset: _dataset_identity(items, args.sample_seed)
                                        for dataset, items in indexed.items()}
        manifest["status"] = "running"; _json(manifest_path, manifest)
        for dataset, items in indexed.items():
            runtime = runtimes[dataset]
            for repeat_index, seed in enumerate(args.repeat_seeds):
                for alignment in ALIGNMENTS:
                    for condition in CONDITIONS:
                        identity = _identity(args, dataset, items, runtime, alignment, condition,
                                             repeat_index, seed)
                        answers, diagnostics, cell_manifest, paths = _load_cell(args, identity)
                        cache_hit = answers is not None
                        if not cache_hit:
                            if wrapper is None:
                                model_args = _method_args(args, dataset, runtime, alignment, repeat_index, seed)
                                wrapper = ModelWrapper(args.model_name, args.device, use_vllm=False, args=model_args)
                                config = wrapper.model.config.to_dict()
                                model_info = {"name": args.model_name,
                                              "resolved_commit": getattr(wrapper.model.config, "_commit_hash", None),
                                              "config_sha256": hashlib.sha256(json.dumps(
                                                  config, sort_keys=True, default=str).encode()).hexdigest(),
                                              "parameter_dtype": str(next(wrapper.model.parameters()).dtype)}
                            answers, diagnostics, setup_seconds = _collect_cell(
                                args, dataset, runtime, items, wrapper, alignment, condition,
                                repeat_index, seed, logger)
                            expected_diagnostics = args.max_questions * len(_sampled_steps(runtime["latent_steps"]))
                            if len(answers) != args.max_questions:
                                raise RuntimeError("C5 answer-row invariant failed")
                            if len(diagnostics) != expected_diagnostics:
                                raise RuntimeError(f"C5 diagnostic-row invariant failed: "
                                                   f"{len(diagnostics)} != {expected_diagnostics}")
                            cell_manifest = _save_cell(paths, identity, answers, diagnostics,
                                                       model_info, setup_seconds)
                        else:
                            model_info = model_info or cell_manifest.get("model")
                            logger.info("C5 cache hit: %s", paths[2])
                        all_answers.extend(answers); all_diagnostics.extend(diagnostics)
                        _parquet(metrics_path, all_answers); _parquet(diagnostics_path, all_diagnostics)
                        completed = {"dataset": dataset, "alignment": alignment, "condition": condition,
                                     "repeat_index": repeat_index, "seed": seed}
                        manifest["row_count"] = {"answers": len(all_answers),
                                                 "diagnostics": len(all_diagnostics)}
                        manifest["last_completed_cell"] = completed
                        manifest["cache_cells"].append({**completed, "cache_hit": cache_hit,
                                                        "manifest": str(paths[2])})
                        _json(manifest_path, manifest)
        expected = len(DATASETS) * len(ALIGNMENTS) * len(CONDITIONS) * len(args.repeat_seeds) * args.max_questions
        if len(all_answers) != expected:
            raise RuntimeError(f"C5 total answer-row invariant failed: {len(all_answers)} != {expected}")
        summary = _summarize(all_answers, all_diagnostics, args, runtimes)
        _json(summary_path, summary); _plot_main(summary, figure_path)
        _plot_degradation(summary, degradation_path)
        manifest.update({"status": "completed", "completed_at": datetime.now().isoformat(timespec="seconds"),
                         "model": model_info,
                         "row_count": {"answers": len(all_answers), "diagnostics": len(all_diagnostics)},
                         "all_cache_hits": all(cell["cache_hit"] for cell in manifest["cache_cells"])})
        _json(manifest_path, manifest)
        return run_dir
    except Exception as error:
        manifest.update({"status": "failed", "failed_at": datetime.now().isoformat(timespec="seconds"),
                         "error": f"{type(error).__name__}: {error}"})
        _json(manifest_path, manifest)
        raise
