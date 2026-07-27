#!/usr/bin/env python
"""Two-stage Refiner-to-Judger approximator experiment."""

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment import build_kernel_state
from data import (
    load_arc_challenge,
    load_arc_easy,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_medqa,
)
from models import _ensure_pad_token
from stages import common, s0, s1, s2, s3, s4
from stages.mapping import mapping_rows
from trajectory import collect, load_hybrid
from utils import auto_device

OUTPUT_ROOT = ROOT / "exp_result" / "approximator"
EXP_CACHE_ROOT = ROOT / "exp" / "cache"
TRAJECTORY_DIR = EXP_CACHE_ROOT / "trajectories"
MAPPING_CACHE_DIR = EXP_CACHE_ROOT / "mappings"
RUNS_DIR = OUTPUT_ROOT / "runs"
RESULT = OUTPUT_ROOT
ROLES = ("planner", "critic", "refiner", "judger")
MAPPING_CACHE_SCHEMA = 1


@dataclass
class State:
    vector: torch.Tensor
    item_id: int
    role: str
    agent_id: int
    turn_id: int
    state_kind: str
    position: int
    model_name: str

    @property
    def source(self):
        return f"{self.role}_{self.state_kind}"


def configure_logger():
    logger = logging.getLogger("approximator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(
        Path.cwd() / "exp_state.txt", mode="a", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("._-") or "unknown"


def short_float(value):
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def cache_component(value):
    """Encode a value reversibly as a Windows-safe, case-stable path component."""
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


def model_tag(agent_models):
    short = [safe_name(name.replace("\\", "/").rsplit("/", 1)[-1]) for name in agent_models]
    if len(set(short)) == 1:
        tag = short[0]
    else:
        tag = "_".join(
            f"{role}-{name}" for role, name in zip(("P", "C", "R", "J"), short)
        )
    if len(tag) > 80:
        digest = hashlib.sha256(
            json.dumps(agent_models, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:8]
        tag = f"{tag[:71]}-{digest}"
    return tag


def run_implementation_fingerprint():
    return files_fingerprint(
        (
            ROOT / "alignment.py",
            ROOT / "models.py",
            ROOT / "prompts.py",
            ROOT / "methods" / "latent_mas_hybrid.py",
            ROOT / "exp" / "approximator" / "trajectory.py",
            ROOT / "exp" / "approximator" / "run.py",
            ROOT / "exp" / "approximator" / "stages" / "common.py",
            ROOT / "exp" / "approximator" / "stages" / "mapping.py",
            ROOT / "exp" / "approximator" / "stages" / "s0.py",
            ROOT / "exp" / "approximator" / "stages" / "s1.py",
            ROOT / "exp" / "approximator" / "stages" / "s2.py",
            ROOT / "exp" / "approximator" / "stages" / "s3.py",
            ROOT / "exp" / "approximator" / "stages" / "s4.py",
        )
    )


def create_run_dir(args):
    config_payload = {
        "args": vars(args),
        "git_commit": git_commit(),
        "implementation_sha256": run_implementation_fingerprint(),
    }
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "_".join(
        (
            safe_name(args.dataset),
            safe_name(args.split),
            safe_name(args.study),
            model_tag(args.agent_models),
            f"lat{args.latent_steps}",
            f"genT{short_float(args.temperature)}",
            f"topP{short_float(args.top_p)}",
            f"tok{args.max_new_tokens}",
            f"km{args.kernel_features}",
            f"kt{short_float(args.kernel_temperature)}",
            f"ks{args.kernel_seed}",
            f"probe{args.probe_seed}",
            f"q{args.max_questions}",
            f"states{args.max_states_per_question}",
            timestamp,
            config_hash,
        )
    )
    run_dir = RUNS_DIR / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{run_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir, config_hash, timestamp


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_models", nargs="+", required=True)
    parser.add_argument("--prompt", choices=["sequential"], default="sequential")
    parser.add_argument("--latent_steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--kernel_features", type=int, default=2048)
    parser.add_argument("--kernel_temperature", type=float, default=1.0)
    parser.add_argument("--kernel_seed", type=int, default=101)
    parser.add_argument("--kernel_chunk_size", type=int, default=4096)
    parser.add_argument(
        "--dataset",
        choices=["arc_easy", "arc_challenge", "gsm8k", "medqa", "mbppplus", "gpqa"],
        default="arc_easy",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_questions", type=int, default=10)
    parser.add_argument("--max_states_per_question", type=int, default=20)
    parser.add_argument("--probe_seed", type=int, default=42)
    parser.add_argument(
        "--study", choices=["s0", "s1", "s2", "s3", "s4", "all"], default="all"
    )
    parser.add_argument(
        "--reuse_trajectory",
        action="store_true",
        help="Require a matching cache and skip Phase A.",
    )
    parser.add_argument("--force_recollect", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_float64_audit", action="store_true")
    parser.add_argument("--s3_replicates", type=int, default=32)
    parser.add_argument("--s3_max_questions", type=int, default=50)
    parser.add_argument("--bootstrap_replicates", type=int, default=1000)
    parser.add_argument("--run_s2_calibration", action="store_true")
    parser.add_argument("--run_s1_performance", action="store_true")
    parser.add_argument("--s4_tsne", action="store_true")
    args = parser.parse_args(argv)
    if len(args.agent_models) == 1:
        args.agent_models *= 4
    elif len(args.agent_models) != 4:
        parser.error("--agent_models must contain exactly one or four model names")
    if args.reuse_trajectory and args.force_recollect:
        parser.error("--reuse_trajectory and --force_recollect are mutually exclusive")
    if args.max_questions < 1 or args.max_states_per_question < 1:
        parser.error("question/state limits must be positive")
    if args.latent_steps < 0 or args.max_new_tokens < 1:
        parser.error("generation lengths are invalid")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("--temperature must be > 0 and --top_p must be in (0, 1]")
    if args.s3_replicates < 2:
        parser.error("--s3_replicates must be at least 2 for sample variance")

    args.agent_models = tuple(args.agent_models)
    args.task = args.dataset
    args.method = "latent_mas_hybrid"
    args.model_name = args.agent_models[0]
    args.device = auto_device(args.device)
    args.device2 = args.device
    args.align_method = "kernel"
    args.align_ridge = 1e-5
    args.seed = args.probe_seed
    args.think = True
    args.latent_only = False
    args.sequential_info_only = False
    args.use_vllm = False
    return args


def load_data(name, split):
    return {
        "arc_easy": load_arc_easy,
        "arc_challenge": load_arc_challenge,
        "gsm8k": load_gsm8k,
        "medqa": load_medqa,
        "mbppplus": load_mbppplus,
        "gpqa": load_gpqa_diamond,
    }[name](split=split)


def sampled_items(args):
    items = list(enumerate(load_data(args.dataset, args.split)))
    random.Random(args.probe_seed).shuffle(items)
    return items[: args.max_questions]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def tokenizer_fingerprint(model_name, trust_remote_code):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=False,
        trust_remote_code=trust_remote_code,
    )
    _ensure_pad_token(tokenizer)
    payload = {
        "vocab": sorted(tokenizer.get_vocab().items()),
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "padding_side": tokenizer.padding_side,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def files_fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def trajectory_cache_stem(args):
    if len(set(args.agent_models)) == 1:
        model_fields = (f"m={cache_component(args.agent_models[0])}",)
    else:
        model_fields = tuple(
            f"{role[0].upper()}={cache_component(model_name)}"
            for role, model_name in zip(ROLES, args.agent_models)
        )
    return "__".join(
        (
            "traj",
            f"ds={cache_component(args.dataset)}",
            f"sp={cache_component(args.split)}",
            *model_fields,
            f"p={cache_component(args.prompt)}",
            f"q={args.max_questions}",
            f"ps={args.probe_seed}",
            f"cap={args.max_states_per_question}",
            f"lat={args.latent_steps}",
            f"t={repr(args.temperature)}",
            f"tp={repr(args.top_p)}",
            f"tok={args.max_new_tokens}",
            "a=kernel",
            f"km={args.kernel_features}",
            f"kt={repr(args.kernel_temperature)}",
            f"ks={args.kernel_seed}",
            f"c={args.kernel_chunk_size}",
            f"rc={int(args.trust_remote_code)}",
        )
    )


def cache_file(directory, stem, suffix):
    name = f"{stem}{suffix}"
    if len(name) > 240:
        raise ValueError(
            "Cache filename is too long for portable filesystems. "
            "Use shorter model identifiers."
        )
    return directory / name


def trajectory_paths(args):
    stem = trajectory_cache_stem(args)
    return (
        cache_file(TRAJECTORY_DIR, stem, ".pt"),
        cache_file(TRAJECTORY_DIR, stem, ".manifest.json"),
    )


def generation_config(args):
    return {
        "prompt": "sequential",
        "latent_steps": args.latent_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "think": True,
        "latent_only": False,
        "sequential_info_only": False,
    }


def alignment_config(args):
    return {
        "align_method": "kernel",
        "kernel_features": args.kernel_features,
        "kernel_temperature": args.kernel_temperature,
        "kernel_seed": args.kernel_seed,
        "kernel_chunk_size": args.kernel_chunk_size,
    }


def expected_manifest(args, indexed_items):
    fingerprints = {
        name: tokenizer_fingerprint(name, args.trust_remote_code)
        for name in dict.fromkeys(args.agent_models)
    }
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "split": args.split,
        "question_ids": [item_id for item_id, _ in indexed_items],
        "question_content_sha256": hashlib.sha256(
            json.dumps(
                indexed_items, sort_keys=True, ensure_ascii=False, default=str
            ).encode("utf-8")
        ).hexdigest(),
        "max_questions": args.max_questions,
        "max_states_per_question": args.max_states_per_question,
        "agent_models": list(args.agent_models),
        "role_mapping": dict(zip(ROLES, args.agent_models)),
        "tokenizer_fingerprint": fingerprints,
        "generation_config": generation_config(args),
        "alignment_config": alignment_config(args),
        "probe_seed": args.probe_seed,
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "git_commit": git_commit(),
        "trajectory_implementation_sha256": files_fingerprint(
            (
                ROOT / "alignment.py",
                ROOT / "models.py",
                ROOT / "prompts.py",
                ROOT / "methods" / "latent_mas_hybrid.py",
                ROOT / "exp" / "approximator" / "trajectory.py",
            )
        ),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def manifest_differences(expected, actual, prefix=""):
    output = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key == "generated_at":
                continue
            field = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                output.append(
                    {
                        "field": field,
                        "expected": expected.get(key),
                        "actual": actual.get(key),
                    }
                )
            else:
                output.extend(manifest_differences(expected[key], actual[key], field))
    elif expected != actual:
        output.append({"field": prefix, "expected": expected, "actual": actual})
    return output


def load_or_collect_trajectory(args, indexed_items, logger):
    expected = expected_manifest(args, indexed_items)
    trajectory_path, manifest_path = trajectory_paths(args)
    have_trajectory = trajectory_path.exists()
    have_manifest = manifest_path.exists()
    if have_trajectory != have_manifest and not args.force_recollect:
        raise RuntimeError(
            "Incomplete trajectory cache; pass --force_recollect to replace it: "
            f"{trajectory_path}"
        )
    have_cache = have_trajectory and have_manifest
    method = None
    cache_hit = False
    if have_cache and not args.force_recollect:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        differences = manifest_differences(expected, actual)
        if differences:
            raise RuntimeError(
                "Refusing to reuse incompatible trajectory manifest. Differences:\n"
                + json.dumps(differences, indent=2, ensure_ascii=False, default=str)
            )
        logger.info("Phase A skipped: reusing %s", trajectory_path)
        cache_hit = True
    else:
        if args.reuse_trajectory:
            raise FileNotFoundError(
                f"--reuse_trajectory requested but cache is absent: {trajectory_path}"
            )
        logger.info("Phase A: real latent-MAS-hybrid collection started.")
        method = load_hybrid(args)
        states, questions = collect(method, indexed_items, args, logger)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "states": states,
                "questions": questions,
                "agent_models": list(args.agent_models),
                "generation_config": generation_config(args),
                "alignment_config": alignment_config(args),
            },
            trajectory_path,
        )
        write_json(
            manifest_path,
            {**expected, "generated_at": datetime.now().isoformat(timespec="seconds")},
        )
        logger.info("Phase A: saved %d states to %s", len(states), trajectory_path)
    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    return trajectory_path, trajectory, method, cache_hit


def get_weights(refiner, judger):
    if refiner.tokenizer.get_vocab() != judger.tokenizer.get_vocab():
        raise ValueError("Refiner and Judger token-to-ID vocabularies must match.")
    output = refiner.model.get_output_embeddings()
    inputs = judger.model.get_input_embeddings()
    if output is None or inputs is None:
        raise ValueError("Refiner/Judger embedding weights are unavailable.")
    if output.weight.shape[0] != inputs.weight.shape[0]:
        raise ValueError("Refiner W_out and Judger W_in vocabulary rows differ.")
    bias = getattr(output, "bias", None)
    return (
        output.weight.detach().float(),
        inputs.weight.detach().float(),
        None if bias is None else bias.detach().float(),
    )


def mapping_metadata(path, args):
    return {
        "trajectory": str(path),
        "mapping": "refiner_to_judger",
        "mapping_source_role": "refiner",
        "mapping_target_role": "judger",
        "mapping_source_model": args.agent_models[2],
        "mapping_target_model": args.agent_models[3],
    }


def mapping_cache_identity(path, args, state_count):
    manifest_path = path.with_suffix(".manifest.json")
    trajectory_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    implementation_hash = hashlib.sha256()
    for implementation_path in (
        ROOT / "alignment.py",
        ROOT / "exp" / "approximator" / "stages" / "common.py",
        ROOT / "exp" / "approximator" / "stages" / "mapping.py",
    ):
        implementation_hash.update(implementation_path.read_bytes())
    payload = {
        "schema_version": MAPPING_CACHE_SCHEMA,
        "trajectory": str(path.resolve()),
        "trajectory_manifest_sha256": trajectory_manifest_sha256,
        "mapping_implementation_sha256": implementation_hash.hexdigest(),
        "state_count": state_count,
        "mapping_source_role": "refiner",
        "mapping_target_role": "judger",
        "mapping_source_model": args.agent_models[2],
        "mapping_target_model": args.agent_models[3],
        "kernel_features": args.kernel_features,
        "kernel_temperature": args.kernel_temperature,
        "kernel_seed": args.kernel_seed,
        "kernel_chunk_size": args.kernel_chunk_size,
        "probe_seed": args.probe_seed,
    }
    cache_id = "__".join(
        (
            "mapping",
            "source=refiner",
            "target=judger",
            f"km={args.kernel_features}",
            f"temperature={repr(args.kernel_temperature)}",
            f"seed={args.kernel_seed}",
            f"c={args.kernel_chunk_size}",
            f"rc={int(args.trust_remote_code)}",
            f"ps={args.probe_seed}",
        )
    )
    return cache_id, payload


def mapping_cache_paths(trajectory_path, cache_id):
    del cache_id
    stem = trajectory_path.stem.replace("traj__", "map__", 1) + "__src=refiner__dst=judger"
    return (
        cache_file(MAPPING_CACHE_DIR, stem, ".full.parquet"),
        cache_file(MAPPING_CACHE_DIR, stem, ".single-kernel.parquet"),
        cache_file(MAPPING_CACHE_DIR, stem, ".mapping.manifest.json"),
    )


def load_or_compute_mapping_cache(
    states,
    wo,
    wi,
    bias,
    kernel,
    args,
    logger,
    cache_id,
    cache_payload,
    trajectory_path,
):
    mapping_path, single_path, manifest_path = mapping_cache_paths(
        trajectory_path, cache_id
    )
    if mapping_path.exists() and single_path.exists() and manifest_path.exists():
        try:
            actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual_identity = {
                key: actual_manifest.get(key) for key in cache_payload
            }
            differences = manifest_differences(cache_payload, actual_identity)
            if differences:
                raise ValueError(
                    "mapping manifest differs:\n"
                    + json.dumps(
                        differences, indent=2, ensure_ascii=False, default=str
                    )
                )
            rows = common.read_rows(mapping_path)
            single = common.read_rows(single_path)
            if len(rows) != len(states):
                raise ValueError(
                    f"cached mapping has {len(rows)} rows; expected {len(states)}"
                )
            logger.info("Full mapping cache hit: %s", mapping_path)
            return rows, single, True, mapping_path, single_path
        except Exception as error:
            logger.warning("Ignoring invalid full mapping cache: %s", error)

    logger.info("Full mapping cache miss; computing %d states once.", len(states))
    rows, single = mapping_rows(
        states, wo, wi, bias, kernel, args, include_s2=True, logger=logger
    )
    common.write_rows_path(rows, mapping_path)
    common.write_rows_path(single, single_path)
    logger.info(
        "Full mapping cache saved: %d mapping rows, %d single-kernel rows.",
        len(rows),
        len(single),
    )
    return rows, single, False, mapping_path, single_path


def summary_header(stage, metadata):
    return {
        "metadata": {
            "stage": stage,
            **metadata,
        },
        "aggregation": {
            "independent_unit": "question",
            "within_question": "mean",
            "variance_ddof": 1,
            "bootstrap_unit": "question",
        },
    }


def metric_summaries(rows, metrics, args):
    return {
        output_name: common.clustered_metric(rows, source_name, args)
        for output_name, source_name in metrics.items()
    }


def rank_metric_summaries(rows, metrics, args):
    output = {}
    for rank_band in sorted({row["rank_band"] for row in rows}):
        subset = [row for row in rows if row["rank_band"] == rank_band]
        output[rank_band] = metric_summaries(subset, metrics, args)
    return output


def s0_summary(rows, wo, wi, args, metadata):
    hidden_norm = {}
    for role, state_kind in sorted(
        {(row["role"], row["state_kind"]) for row in rows}
    ):
        subset = [
            row
            for row in rows
            if row["role"] == role and row["state_kind"] == state_kind
        ]
        hidden_norm[f"{role}.{state_kind}"] = common.clustered_metric(
            subset, "hidden_norm", args
        )
    return {
        **summary_header("s0", metadata),
        "metrics": {
            "hidden_norm": hidden_norm,
            "weight_norm": {
                "refiner_output_embedding": common.describe_values(
                    wo.norm(dim=1).float().cpu().numpy()
                ),
                "judger_input_embedding": common.describe_values(
                    wi.norm(dim=1).float().cpu().numpy()
                ),
            },
        },
    }


def s1_summary(rows, single_rows, performance_rows, args, metadata):
    payload = {
        **summary_header("s1", metadata),
        "metrics": {
            "mapping": metric_summaries(
                rows,
                {
                    "f_relative_l2_error": "f_rel_l2",
                    "f_cosine": "f_cosine",
                    "confidence": "confidence",
                    "entropy": "entropy",
                },
                args,
            ),
            "quality_rates": {
                "denominator_valid": common.rate_summary(rows, "denom_valid"),
                "nan_inf": common.rate_summary(rows, "nan_inf"),
            },
            "single_kernel_by_rank": rank_metric_summaries(
                single_rows,
                {
                    "absolute_error": "kernel_abs_error",
                    "relative_error": "kernel_relative_error",
                    "log_error": "kernel_log_error",
                    "kernel_ratio": "kernel_ratio",
                },
                args,
            ),
        },
    }
    if performance_rows:
        payload["metrics"]["performance_latency_us"] = {
            method: common.clustered_metric(
                [row for row in performance_rows if row["method"] == method],
                "latency_us",
                args,
            )
            for method in ("exact", "kernel")
        }
    return payload


def s2_summary(rows, args, metadata):
    return {
        **summary_header("s2", metadata),
        "metrics": {
            "distribution_error": metric_summaries(
                rows,
                {
                    "kl_p_to_phat": "kl_p_phat",
                    "js_divergence": "js",
                    "total_variation": "tv",
                    "l1_error": "l1",
                    "top10_overlap": "top10_overlap",
                    "top100_overlap": "top100_overlap",
                    "exact_top10_mass": "exact_top10_mass",
                },
                args,
            ),
            "top1_agreement": common.rate_summary(rows, "top1_agree"),
        },
    }


def s2_calibration_summary(rows, args, metadata):
    groups = {}
    for family, feature_count, temperature in sorted(
        {
            (row["feature_family"], row["m"], row["tau"])
            for row in rows
        }
    ):
        subset = [
            row
            for row in rows
            if row["feature_family"] == family
            and row["m"] == feature_count
            and row["tau"] == temperature
        ]
        key = f"{family}.m{feature_count}.tau{short_float(temperature)}"
        groups[key] = {
            "relative_l2_error": common.clustered_metric(
                subset, "rel_l2", args
            ),
            "denominator_valid": common.rate_summary(subset, "denom_valid"),
        }
    return {
        **summary_header("s2_calibration", metadata),
        "configurations": groups,
    }


def s3_metric_block(rows, args):
    f_rows = [row for row in rows if row["kind"] == "F"]
    kernel_rows = [row for row in rows if row["kind"] == "kernel"]
    return {
        "F": metric_summaries(
            f_rows,
            {
                "variance": "variance",
                "std": "std",
                "relative_std": "relative_std",
                "bias_squared": "bias2",
                "mse": "mse",
            },
            args,
        ),
        "single_kernel_by_rank": rank_metric_summaries(
            kernel_rows,
            {
                "variance": "variance",
                "std": "std",
                "relative_std": "relative_std",
                "bias_squared": "bias2",
                "mse": "mse",
            },
            args,
        ),
    }


def s3_summaries(rows, args, metadata):
    headline_rows = [
        row
        for row in rows
        if row["m"] == 2048 and abs(row["tau"] - 1.0) < 1e-8
    ]
    headline = {
        **summary_header("s3", metadata),
        "configuration": {"m": 2048, "tau": 1.0},
        "metrics": s3_metric_block(headline_rows, args),
    }
    grid = {}
    for feature_count, temperature in sorted(
        {(row["m"], row["tau"]) for row in rows}
    ):
        subset = [
            row
            for row in rows
            if row["m"] == feature_count and row["tau"] == temperature
        ]
        grid[f"m{feature_count}.tau{short_float(temperature)}"] = s3_metric_block(
            subset, args
        )
    return headline, {
        **summary_header("s3_grid", metadata),
        "configurations": grid,
    }


def run_phase_b(path, trajectory, method, args, logger):
    all_states = [State(**record) for record in trajectory["states"]]
    analysis_states = [
        state
        for state in all_states
        if state.role == "refiner" and state.state_kind == "latent_reply_hidden"
    ]
    if not analysis_states:
        raise RuntimeError("No Refiner latent_reply_hidden states in trajectory.")
    logger.info("Phase B: %d saved Refiner latent reply states.", len(analysis_states))
    if method is None:
        method = load_hybrid(args)
    refiner = method.models[args.agent_models[2]]
    judger = method.models[args.agent_models[3]]
    wo, wi, bias = get_weights(refiner, judger)
    wo, wi = wo.to(args.device), wi.to(args.device)
    bias = None if bias is None else bias.to(args.device)
    cache_id, cache_payload = mapping_cache_identity(
        path, args, len(analysis_states)
    )
    metadata = {
        **mapping_metadata(path, args),
        "mapping_cache_id": cache_id,
        **{
            key: cache_payload[key]
            for key in (
                "kernel_features",
                "kernel_temperature",
                "kernel_seed",
                "kernel_chunk_size",
                "probe_seed",
            )
        },
    }
    common.set_artifact_context(metadata)
    write_json(
        RESULT / "manifests" / "analysis.json",
        {
            **metadata,
            "study": args.study,
            "analysis_state_count": len(analysis_states),
            "complete_trajectory_state_count": len(all_states),
        },
    )
    logger.info(
        "Float64 audit p99=%s",
        common.audit(analysis_states, wo, wi, bias, args),
    )
    kernel = build_kernel_state(
        wo,
        wi,
        bias,
        feature_count=args.kernel_features,
        temperature=args.kernel_temperature,
        seed=args.kernel_seed,
        chunk_size=args.kernel_chunk_size,
    )
    studies = [args.study] if args.study != "all" else ["s0", "s1", "s2", "s3", "s4"]
    cached_rows = cached_single = None
    mapping_cache_info = None
    if any(study in ("s1", "s2") for study in studies):
        (
            cached_rows,
            cached_single,
            mapping_cache_hit,
            mapping_path,
            single_path,
        ) = load_or_compute_mapping_cache(
            analysis_states,
            wo,
            wi,
            bias,
            kernel,
            args,
            logger,
            cache_id,
            cache_payload,
            path,
        )
        mapping_cache_info = {
            "cache_id": cache_id,
            "cache_hit": mapping_cache_hit,
            "mapping_file": str(mapping_path),
            "single_kernel_file": str(single_path),
        }
        cache_manifest = {
            **cache_payload,
            "cache_id": cache_id,
            "mapping_file": str(mapping_path),
            "single_kernel_file": str(single_path),
            "mapping_rows": len(cached_rows),
            "single_kernel_rows": len(cached_single),
        }
        _, _, mapping_manifest_path = mapping_cache_paths(path, cache_id)
        write_json(mapping_manifest_path, cache_manifest)
        write_json(
            RESULT / "manifests" / "mapping.json",
            {**cache_manifest, "cache_hit": mapping_cache_hit},
        )
    for study in studies:
        logger.info("Phase B %s started.", study.upper())
        if study == "s0":
            rows = s0.run(all_states, wo, wi, args, logger)
            common.write_rows(rows, "s0_hidden_states")
            common.write_summary(
                s0_summary(rows, wo, wi, args, metadata),
                "s0_summary",
            )
        elif study == "s1":
            common.write_rows(cached_rows, "s1_mapping")
            common.write_rows(cached_single, "s1_single_kernel")
            common.histogram_ecdf(cached_rows, "f_rel_l2", "s1")
            s1.plot(cached_rows)
            performance_rows = None
            if args.run_s1_performance:
                performance_rows = s1.performance(
                    analysis_states, wo, wi, bias, kernel, args, logger
                )
                common.write_rows(performance_rows, "s1_performance")
            common.write_summary(
                s1_summary(
                    cached_rows,
                    cached_single,
                    performance_rows,
                    args,
                    metadata,
                ),
                "s1_summary",
            )
        elif study == "s2":
            common.write_rows(cached_rows, "s2_mapping")
            common.write_rows(cached_single, "s2_single_kernel")
            common.histogram_ecdf(cached_rows, "f_rel_l2", "s2")
            s2.plot(cached_rows)
            common.write_summary(
                s2_summary(cached_rows, args, metadata),
                "s2_summary",
            )
            if args.run_s2_calibration:
                calibration_rows = s2.calibration(
                    analysis_states, wo, wi, bias, args, logger
                )
                common.write_rows(calibration_rows, "s2_calibration")
                common.write_summary(
                    s2_calibration_summary(
                        calibration_rows, args, metadata
                    ),
                    "s2_calibration_summary",
                )
        elif study == "s3":
            rows = s3.run(analysis_states, wo, wi, bias, args, logger)
            common.write_rows(rows, "s3_variance")
            headline, grid = s3_summaries(rows, args, metadata)
            common.write_summary(headline, "s3_summary")
            common.write_summary(grid, "s3_grid_summary")
            s3.plot_s3(rows)
            s3.plot_kernel_variance(rows)
            s3.plot_forest(rows)
        else:
            rows, summary = s4.run(
                analysis_states, wo, wi, bias, kernel, args, logger
            )
            common.write_rows(rows, "s4_embeddings")
            common.write_summary(
                {
                    **summary_header("s4", metadata),
                    "metrics": summary,
                },
                "s4_summary",
            )
        logger.info("Phase B %s completed.", study.upper())
    return {
        "mapping_cache": mapping_cache_info,
        "analysis_state_count": len(analysis_states),
        "complete_trajectory_state_count": len(all_states),
    }


def main(argv=None):
    global RESULT
    args = parse_args(argv)
    run_dir, config_hash, timestamp = create_run_dir(args)
    RESULT = run_dir
    common.set_result_root(run_dir)
    logger = configure_logger()
    started = time.time()
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = {
        "status": "running",
        "run_name": run_dir.name,
        "run_directory": str(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "directory_timestamp": timestamp,
        "config_hash": config_hash,
        "implementation_sha256": run_implementation_fingerprint(),
        "args": vars(args),
        "git_commit": git_commit(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    write_json(run_manifest_path, run_manifest)
    logger.info("Run directory: %s", run_dir)
    random.seed(args.probe_seed)
    np.random.seed(args.probe_seed)
    torch.manual_seed(args.probe_seed)
    try:
        items = sampled_items(args)
        path, trajectory, method, trajectory_cache_hit = load_or_collect_trajectory(
            args, items, logger
        )
        trajectory_manifest = path.with_suffix(".manifest.json")
        write_json(
            run_dir / "manifests" / "trajectory.json",
            {
                "cache_hit": trajectory_cache_hit,
                "trajectory_file": str(path),
                "manifest_file": str(trajectory_manifest),
                "manifest_sha256": hashlib.sha256(
                    trajectory_manifest.read_bytes()
                ).hexdigest(),
            },
        )
        phase_b = run_phase_b(path, trajectory, method, args, logger)
        elapsed = time.time() - started
        run_manifest.update(
            status="completed",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=elapsed,
            trajectory_cache={
                "cache_hit": trajectory_cache_hit,
                "trajectory_file": str(path),
                "manifest_file": str(trajectory_manifest),
            },
            **phase_b,
        )
        write_json(run_manifest_path, run_manifest)
        logger.info("Two-stage experiment completed in %.1fs.", elapsed)
    except Exception as error:
        run_manifest.update(
            status="failed",
            failed_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=time.time() - started,
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json(run_manifest_path, run_manifest)
        logger.exception("Two-stage experiment failed.")
        raise


if __name__ == "__main__":
    main()

