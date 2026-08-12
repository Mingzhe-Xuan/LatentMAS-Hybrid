#!/usr/bin/env python
"""M0 prefill transfer with question-blind, question-visible, and text controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_arc_easy
from models import ModelWrapper
from utils import extract_gsm8k_answer, normalize_answer, set_seed

MODELS = ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B")
MODEL_PAIRS = tuple((source, target) for source in MODELS for target in MODELS)
ALIGNMENTS = ("linear", "kernel", "soft", "text")
CACHE_SCHEMA_VERSION = 2
RESULT_CACHE_SCHEMA_VERSION = 3
VISIBLE_RESULT_CACHE_SCHEMA_VERSION = 4
CACHE_DIR = ROOT / "exp" / "cache" / "latent_comm_m0"
RUNS_DIR = ROOT / "exp_result" / "latent_comm" / "runs"
RECEIVER_PROMPT_VERSION = "m0_question_blind_prefill_receiver_v2"
DIRECT_TEXT_PROMPT_VERSION = "m0_direct_text_receiver_v1"
VISIBLE_RECEIVER_PROMPT_VERSION = "m0_question_visible_prefill_receiver_v1"
SENDER_PROMPT_VERSION = "m0_question_only_single_prefill_v2"
ALIGNMENT_COLORS = {
    "linear": "#4c78a8",
    "kernel": "#54a24b",
    "soft": "#f58518",
    "text": "#7f7f7f",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["m0"], default="m0")
    parser.add_argument("--dataset", choices=["arc_easy"], default="arc_easy")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_questions", type=int, default=100)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--generation_seed", type=int, default=77)
    parser.add_argument("--prompt", choices=["sequential"], default="sequential")
    parser.add_argument("--model_pair", choices=["all"], default="all")
    parser.add_argument("--method", choices=["all"], default="all")
    parser.add_argument("--alignments", nargs="+", choices=ALIGNMENTS, default=list(ALIGNMENTS))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--bootstrap_replicates", type=int, default=1000)
    parser.add_argument("--probe_seed", type=int, default=42)
    parser.add_argument("--kernel_features", "--m", dest="kernel_features", type=int, default=2048)
    parser.add_argument("--kernel_temperature", "--tau", dest="kernel_temperature", type=float, default=0.6)
    parser.add_argument("--kernel_seed", "--orf_seed", dest="kernel_seed", type=int, default=101)
    parser.add_argument("--kernel_chunk_size", type=int, default=4096)
    parser.add_argument("--soft_chunk_size", type=int, default=32)
    parser.add_argument("--align_ridge", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument("--force_recollect", action="store_true")
    args = parser.parse_args(argv)
    if args.max_questions != 100:
        parser.error("M0 requires exactly --max_questions 100")
    if len(args.alignments) != len(ALIGNMENTS) or set(args.alignments) != set(ALIGNMENTS):
        parser.error("M0 requires --alignments linear kernel soft text")
    if args.reuse_cache and args.force_recollect:
        parser.error("--reuse_cache and --force_recollect are mutually exclusive")
    if args.max_new_tokens < 1 or args.bootstrap_replicates < 1:
        parser.error("generation/bootstrap sizes must be positive")
    if args.kernel_features < 1 or args.kernel_chunk_size < 1 or args.soft_chunk_size < 1:
        parser.error("alignment sizes must be positive")
    if args.kernel_temperature <= 0 or args.align_ridge < 0:
        parser.error("temperature must be positive and ridge non-negative")
    return args


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("._-") or "unknown"


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression="zstd")
    temporary.replace(path)


def _read_parquet(path):
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _configure_logger():
    logger = logging.getLogger("latent_comm.m0")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def sampled_items(args):
    indexed = list(enumerate(load_arc_easy(split=args.split)))
    random.Random(args.sample_seed).shuffle(indexed)
    selected = indexed[: args.max_questions]
    if len(selected) != args.max_questions:
        raise RuntimeError(
            f"Requested {args.max_questions} ARC-Easy questions, loaded {len(selected)}"
        )
    return selected


def _sample_identity(items, args):
    payload = [
        {"item_id": item_id, "question": item["question"], "gold": item.get("gold", "")}
        for item_id, item in items
    ]
    return {
        "dataset": "arc_easy",
        "split": args.split,
        "selection": "seeded_without_replacement",
        "sample_seed": args.sample_seed,
        "question_ids": [item_id for item_id, _ in items],
        "content_sha256": _sha256_bytes(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ),
    }


def _model_args(args, alignment):
    return argparse.Namespace(
        **vars(args),
        task="arc_easy",
        align_method=alignment,
        soft_temperature=args.kernel_temperature,
        seed=args.generation_seed,
        device2=args.device,
        use_vllm=False,
    )


def _sender_messages(question, model_name):
    del model_name
    # Agent A is only an encoder: no Planner instruction and no generated thought.
    return [{"role": "user", "content": question}]


def _receiver_messages(model_name):
    system = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        if "qwen" in model_name.lower()
        else "You are a helpful assistant."
    )
    user = (
        "You are Agent B in a two-agent system. Agent A encoded a private "
        "four-choice science question that is not shown to you. Agent A used one "
        "prefill pass and did not reason or "
        "generate an answer. The aligned hidden-state sequence prepended to this "
        "prompt is your only information about the question and its choices. "
        "Reason from that representation and respond with exactly one final choice "
        "inside \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}. Do not request "
        "or assume access to the original question."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _direct_text_receiver_messages(model_name, question):
    system = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        if "qwen" in model_name.lower()
        else "You are a helpful assistant."
    )
    user = (
        "Agent A transmitted the following four-choice science question directly "
        "as text. Solve it and respond with exactly one final choice inside "
        "\\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.\n\n"
        f"Question:\n{question}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _visible_receiver_messages(model_name, question):
    system = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        if "qwen" in model_name.lower()
        else "You are a helpful assistant."
    )
    user = (
        "You are Agent B in a two-agent system. Agent A encoded the four-choice "
        "science question into the aligned hidden-state sequence prepended to this "
        "prompt. You are also given the original question below. Use both sources "
        "to solve it and respond with exactly one final choice inside \\boxed{A}, "
        "\\boxed{B}, \\boxed{C}, or \\boxed{D}.\n\n"
        f"Question:\n{question}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _direct_text_receiver_prompt(wrapper, question):
    messages = _direct_text_receiver_messages(wrapper.model_name, question)
    rendered = wrapper.render_chat(messages, add_generation_prompt=True)
    normalized_question = " ".join(question.split())
    normalized_prompt = " ".join(rendered.split())
    if normalized_question not in normalized_prompt:
        raise RuntimeError("M0 direct-text receiver prompt omitted the original question")
    return messages, rendered


def _visible_receiver_prompt(wrapper, question):
    messages = _visible_receiver_messages(wrapper.model_name, question)
    rendered = wrapper.render_chat(messages, add_generation_prompt=True)
    normalized_question = " ".join(question.split())
    normalized_prompt = " ".join(rendered.split())
    if normalized_question not in normalized_prompt:
        raise RuntimeError(
            "M0 question-visible receiver prompt omitted the original question"
        )
    return messages, rendered


def _receiver_prompt(wrapper, question):
    messages = _receiver_messages(wrapper.model_name)
    rendered = wrapper.render_chat(messages, add_generation_prompt=True)
    normalized_question = " ".join(question.split())
    normalized_prompt = " ".join(rendered.split())
    leaked = bool(normalized_question) and normalized_question in normalized_prompt
    if leaked:
        raise RuntimeError("M0 receiver prompt leaked the original question")
    return messages, rendered


def _trajectory_identity(args, source_model, sample_identity):
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "experiment": "m0_sender_trajectory",
        "source_model": source_model,
        "sample": sample_identity,
        "prompt": SENDER_PROMPT_VERSION,
        "sender_operation": "single_prefill_full_last_layer_sequence",
        "trust_remote_code": bool(args.trust_remote_code),
    }


def _trajectory_paths(args, source_model, identity):
    digest = _sha256_bytes(json.dumps(identity, sort_keys=True).encode("utf-8"))[:16]
    stem = f"sender_{_safe_name(source_model.rsplit('/', 1)[-1])}_q{args.max_questions}_{digest}"
    return CACHE_DIR / f"{stem}.pt", CACHE_DIR / f"{stem}.manifest.json"


def _load_trajectory_cache(args, source_model, identity):
    trajectory_path, manifest_path = _trajectory_paths(args, source_model, identity)
    presence = (trajectory_path.exists(), manifest_path.exists())
    if any(presence) and not all(presence) and not args.force_recollect:
        raise RuntimeError(f"Incomplete M0 sender cache: {trajectory_path}")
    if not all(presence) or args.force_recollect:
        return None, None, trajectory_path, manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_identity") != identity:
        raise RuntimeError(f"Incompatible M0 sender cache: {trajectory_path}")
    if _sha256_file(trajectory_path) != manifest.get("trajectory_sha256"):
        raise RuntimeError("M0 sender trajectory SHA256 check failed")
    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    records = trajectory.get("records", [])
    if len(records) != args.max_questions:
        raise RuntimeError("M0 sender trajectory record count mismatch")
    expected_ids = identity["sample"]["question_ids"]
    if [record.get("item_id") for record in records] != expected_ids:
        raise RuntimeError("M0 sender trajectory item coverage mismatch")
    for record in records:
        states = record.get("prefill_hidden_states")
        if (
            not isinstance(states, torch.Tensor)
            or states.device.type != "cpu"
            or states.dtype != torch.float32
            or states.ndim != 2
            or states.shape[0] < 1
            or states.shape[0] != record.get("sender_prompt_token_count")
        ):
            raise RuntimeError(
                f"Invalid M0 sender states for item_id={record.get('item_id')}"
            )
    return trajectory, manifest, trajectory_path, manifest_path


@torch.inference_mode()
def _collect_sender_trajectory(wrapper, items, args, logger):
    records = []
    for number, (item_id, item) in enumerate(items, start=1):
        messages = _sender_messages(item["question"], wrapper.model_name)
        rendered = wrapper.render_chat(messages, add_generation_prompt=True)
        encoded = wrapper.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(wrapper.device)
        attention_mask = encoded["attention_mask"].to(wrapper.device)
        outputs = wrapper.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        prefill_hidden_states = outputs.hidden_states[-1][0].detach().to(
            "cpu", dtype=torch.float32
        )
        records.append(
            {
                "item_id": int(item_id),
                "question": item["question"],
                "gold": item.get("gold", ""),
                "source_model": wrapper.model_name,
                "sender_messages": json.dumps(messages, ensure_ascii=False),
                "sender_prompt_sha256": _sha256_bytes(rendered.encode("utf-8")),
                "sender_prompt_token_count": int(attention_mask.sum().item()),
                "prefill_hidden_states": prefill_hidden_states,
            }
        )
        logger.info(
            "M0 sender=%s question=%d/%d item_id=%d prefill_tokens=%d collected",
            wrapper.model_name,
            number,
            len(items),
            item_id,
            prefill_hidden_states.shape[0],
        )
    return records


def _save_trajectory_cache(path, manifest_path, records, identity):
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "experiment": "m0_sender_trajectory",
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_identity": identity,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "record_count": len(records),
        "trajectory": str(path),
        "trajectory_sha256": _sha256_file(path),
    }
    _write_json(manifest_path, manifest)
    return payload, manifest


def _result_identity(
    args,
    source_model,
    target_model,
    alignment,
    sample_identity,
    trajectory_sha,
    receiver_visibility,
):
    direct_text = alignment == "text"
    if direct_text and receiver_visibility != "direct_text":
        raise ValueError("text requires receiver_visibility=direct_text")
    if not direct_text and receiver_visibility not in {"blind", "visible"}:
        raise ValueError("latent alignments require blind or visible receiver mode")

    schema_version = CACHE_SCHEMA_VERSION
    receiver_prompt = RECEIVER_PROMPT_VERSION
    if direct_text:
        schema_version = RESULT_CACHE_SCHEMA_VERSION
        receiver_prompt = DIRECT_TEXT_PROMPT_VERSION
    elif receiver_visibility == "visible":
        schema_version = VISIBLE_RESULT_CACHE_SCHEMA_VERSION
        receiver_prompt = VISIBLE_RECEIVER_PROMPT_VERSION

    identity = {
        "schema_version": schema_version,
        "experiment": "m0_receiver_answers",
        "source_model": source_model,
        "target_model": target_model,
        "alignment": alignment,
        "sample": sample_identity,
        "source_trajectory_sha256": None if direct_text else trajectory_sha,
        "receiver_prompt": receiver_prompt,
        "sender_operation": (
            "direct_original_text_no_agent_a"
            if direct_text
            else "single_prefill_full_last_layer_sequence"
        ),
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "alignment_config": None if direct_text else {
            "linear_ridge": args.align_ridge,
            "kernel_features": args.kernel_features,
            "kernel_temperature": args.kernel_temperature,
            "kernel_seed": args.kernel_seed,
            "kernel_chunk_size": args.kernel_chunk_size,
            "soft_chunk_size": args.soft_chunk_size,
        },
        "generation_seed": args.generation_seed,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    # Keep the v2 blind and v3 direct-text identities unchanged so their caches
    # remain reusable. Only the new visible condition adds this identity field.
    if receiver_visibility == "visible":
        identity["receiver_visibility"] = "visible"
    return identity


def _result_paths(args, source_model, target_model, alignment, identity):
    digest = _sha256_bytes(json.dumps(identity, sort_keys=True).encode("utf-8"))[:16]
    source = _safe_name(source_model.rsplit("/", 1)[-1])
    target = _safe_name(target_model.rsplit("/", 1)[-1])
    stem = f"answers_{source}_to_{target}_{alignment}_q{args.max_questions}_{digest}"
    return CACHE_DIR / f"{stem}.parquet", CACHE_DIR / f"{stem}.manifest.json"


def _identity_receiver_visibility(identity):
    if identity["alignment"] == "text":
        return "direct_text"
    return identity.get("receiver_visibility", "blind")


def _load_result_cache(args, identity, paths):
    metrics_path, manifest_path = paths
    presence = (metrics_path.exists(), manifest_path.exists())
    if any(presence) and not all(presence) and not args.force_recollect:
        raise RuntimeError(f"Incomplete M0 answer cache: {metrics_path}")
    if not all(presence) or args.force_recollect:
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_identity") != identity:
        raise RuntimeError(f"Incompatible M0 answer cache: {metrics_path}")
    if _sha256_file(metrics_path) != manifest.get("metrics_sha256"):
        raise RuntimeError("M0 answer cache SHA256 check failed")
    rows = _read_parquet(metrics_path)
    visibility_mode = _identity_receiver_visibility(identity)
    for row in rows:
        row.setdefault("transfer_mode", "aligned_hidden_states")
        row.setdefault("direct_text_token_count", 0)
        row.setdefault("receiver_visibility", visibility_mode)
        row.setdefault(
            "receiver_original_question_token_count",
            row.get("direct_text_token_count", 0) if visibility_mode == "direct_text" else 0,
        )
    if len(rows) != args.max_questions:
        raise RuntimeError("M0 answer cache row count mismatch")
    expected_ids = set(identity["sample"]["question_ids"])
    if {row.get("item_id") for row in rows} != expected_ids:
        raise RuntimeError("M0 answer cache item coverage mismatch")
    expected_visible = visibility_mode != "blind"
    if any(bool(row.get("receiver_question_visible")) != expected_visible for row in rows):
        raise RuntimeError("M0 cached receiver prompt failed visibility audit")
    return rows, manifest


def _save_result_cache(paths, rows, identity):
    metrics_path, manifest_path = paths
    _write_parquet(metrics_path, rows)
    expected_visible = _identity_receiver_visibility(identity) != "blind"
    visibility_passed = all(
        bool(row.get("receiver_question_visible")) == expected_visible for row in rows
    )
    manifest = {
        "schema_version": identity["schema_version"],
        "cache_identity": identity,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": len(rows),
        "metrics": str(metrics_path),
        "metrics_sha256": _sha256_file(metrics_path),
        "receiver_visibility_audit_passed": visibility_passed,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _validate_vocab(source, target):
    if source.tokenizer.get_vocab() != target.tokenizer.get_vocab():
        raise ValueError(
            "M0 requires identical token-to-ID vocabularies: "
            f"{source.model_name} -> {target.model_name}"
        )


@torch.inference_mode()
def _aligned_message(record, source, target, alignment):
    if alignment == "text":
        raise ValueError("Direct text is handled without Agent A hidden states")
    hidden = record["prefill_hidden_states"].to(source.device)
    source.align_method = alignment
    source.args.align_method = alignment
    aligned = source.align_hidden_to(hidden, target)
    return aligned.unsqueeze(0)


def _parse_arc_answer(text):
    prediction = normalize_answer(extract_gsm8k_answer(text))
    if prediction in {"a", "b", "c", "d"}:
        return prediction
    matches = re.findall(r"(?i)(?:\\boxed\{)?\b([ABCD])\b", text)
    return matches[-1].lower() if matches else None


@torch.inference_mode()
def _run_receiver_cell(
    records,
    source,
    target,
    alignment,
    receiver_visibility,
    args,
    logger,
):
    direct_text = alignment == "text"
    source_model_name = source if isinstance(source, str) else source.model_name
    if direct_text and receiver_visibility != "direct_text":
        raise ValueError("text requires direct_text receiver visibility")
    if not direct_text:
        _validate_vocab(source, target)
        if receiver_visibility not in {"blind", "visible"}:
            raise ValueError("latent receiver visibility must be blind or visible")

    rows = []
    for number, record in enumerate(records, start=1):
        set_seed(args.generation_seed)
        if direct_text:
            messages, receiver_prompt = _direct_text_receiver_prompt(
                target, record["question"]
            )
        elif receiver_visibility == "visible":
            messages, receiver_prompt = _visible_receiver_prompt(
                target, record["question"]
            )
        else:
            messages, receiver_prompt = _receiver_prompt(
                target, record["question"]
            )

        encoded = target.tokenizer(
            receiver_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = encoded["input_ids"].to(target.device)
        prompt_mask = encoded["attention_mask"].to(target.device)
        question_token_count = 0
        if receiver_visibility != "blind":
            question_token_count = len(
                target.tokenizer(
                    record["question"], add_special_tokens=False
                )["input_ids"]
            )

        started = time.perf_counter()
        if direct_text:
            generated, _ = target.generate_text_batch(
                prompt_ids,
                prompt_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=0.6,
                top_p=1.0,
            )
            source_prefill_token_count = None
            communication_hidden_state_count = 0
            direct_text_token_count = question_token_count
        else:
            prompt_embeds = target.model.get_input_embeddings()(prompt_ids)
            message_embeds = _aligned_message(record, source, target, alignment)
            message_embeds = message_embeds.to(
                device=target.device,
                dtype=prompt_embeds.dtype,
            )
            combined = torch.cat([message_embeds, prompt_embeds], dim=1)
            combined_mask = torch.cat(
                [
                    torch.ones(
                        message_embeds.shape[:2],
                        dtype=prompt_mask.dtype,
                        device=target.device,
                    ),
                    prompt_mask,
                ],
                dim=1,
            )
            generated, _ = target.generate_text_from_embeds_batch(
                combined,
                combined_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=0.6,
                top_p=1.0,
            )
            source_prefill_token_count = int(record["sender_prompt_token_count"])
            communication_hidden_state_count = int(message_embeds.shape[1])
            direct_text_token_count = 0

        wall_seconds = time.perf_counter() - started
        raw_prediction = generated[0]
        prediction = _parse_arc_answer(raw_prediction)
        gold = normalize_answer(record["gold"])
        rows.append(
            {
                "dataset": "arc_easy",
                "split": args.split,
                "item_id": int(record["item_id"]),
                "source_model": source_model_name,
                "target_model": target.model_name,
                "model_pair": f"{source_model_name}->{target.model_name}",
                "alignment": alignment,
                "receiver_visibility": receiver_visibility,
                "transfer_mode": (
                    "direct_original_text" if direct_text else "aligned_hidden_states"
                ),
                "source_prefill_token_count": source_prefill_token_count,
                "communication_hidden_state_count": communication_hidden_state_count,
                "direct_text_token_count": direct_text_token_count,
                "receiver_original_question_token_count": question_token_count,
                "question": record["question"],
                "gold": gold,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "correct": prediction == gold,
                "parse_success": prediction is not None,
                "receiver_messages": json.dumps(messages, ensure_ascii=False),
                "receiver_prompt": receiver_prompt,
                "receiver_prompt_sha256": _sha256_bytes(receiver_prompt.encode("utf-8")),
                "receiver_prompt_token_count": int(prompt_mask.sum().item()),
                "receiver_question_visible": receiver_visibility != "blind",
                "message_token_ids": None,
                "wall_seconds": wall_seconds,
                "receiver_output_tokens": target.last_generation_metrics[
                    "output_token_counts"
                ][0],
            }
        )
        logger.info(
            "M0 %s -> %s alignment=%s visibility=%s question=%d/%d correct=%s",
            source_model_name,
            target.model_name,
            alignment,
            receiver_visibility,
            number,
            len(records),
            prediction == gold,
        )
    return rows


def _bootstrap(values, replicates, seed):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize(rows, args):
    series = {}
    for pair_index, (source, target) in enumerate(MODEL_PAIRS):
        pair = f"{source}->{target}"
        series[pair] = []
        condition_index = 0
        for alignment in args.alignments:
            visibility_modes = (
                ("direct_text",) if alignment == "text" else ("blind", "visible")
            )
            for receiver_visibility in visibility_modes:
                selected = [
                    row for row in rows
                    if row["model_pair"] == pair
                    and row["alignment"] == alignment
                    and row["receiver_visibility"] == receiver_visibility
                ]
                values = [float(row["correct"]) for row in selected]
                low, high = _bootstrap(
                    values,
                    args.bootstrap_replicates,
                    args.probe_seed + pair_index * 100 + condition_index,
                )
                expected_visible = receiver_visibility != "blind"
                series[pair].append(
                    {
                        "alignment": alignment,
                        "receiver_visibility": receiver_visibility,
                        "processed": len(selected),
                        "correct": int(sum(values)),
                        "accuracy": float(np.mean(values)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "parse_success": sum(row["parse_success"] for row in selected),
                        "question_visible_count": sum(
                            row["receiver_question_visible"] for row in selected
                        ),
                        "unexpected_question_visibility_count": sum(
                            bool(row["receiver_question_visible"]) != expected_visible
                            for row in selected
                        ),
                        "mean_communication_hidden_states": float(
                            np.mean([
                                row["communication_hidden_state_count"]
                                for row in selected
                            ])
                        ),
                        "mean_original_question_tokens": float(
                            np.mean([
                                row["receiver_original_question_token_count"]
                                for row in selected
                            ])
                        ),
                    }
                )
                condition_index += 1

    latent_lengths = [
        row["communication_hidden_state_count"]
        for row in rows
        if row["alignment"] != "text"
    ]
    visible_question_lengths = [
        row["receiver_original_question_token_count"]
        for row in rows
        if row["receiver_visibility"] != "blind"
    ]
    return {
        "study": "m0",
        "dataset": "arc_easy",
        "questions": args.max_questions,
        "conditions_per_model_pair": 7,
        "transfer_protocols": {
            "latent": "single_prefill_full_last_layer_sequence",
            "text": "direct_original_text_no_agent_a",
        },
        "receiver_visibility": {
            "blind": "aligned hidden states plus a question-blind B prompt",
            "visible": "aligned hidden states plus the original question in B prompt",
            "direct_text": "original question only; Agent A is bypassed",
        },
        "communication_length": {
            "aligned_hidden_states": {
                "minimum": min(latent_lengths),
                "maximum": max(latent_lengths),
                "mean": float(np.mean(latent_lengths)),
            },
            "visible_original_question_tokens": {
                "minimum": min(visible_question_lengths),
                "maximum": max(visible_question_lengths),
                "mean": float(np.mean(visible_question_lengths)),
            },
        },
        "series": series,
    }


def _plot(summary, path, args):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    latent_alignments = [item for item in args.alignments if item != "text"]
    width = 0.34

    def point_for(points, alignment, visibility):
        return next(
            point for point in points
            if point["alignment"] == alignment
            and point["receiver_visibility"] == visibility
        )

    def add_errorbar(axis, x, point):
        mean = point["accuracy"]
        axis.errorbar(
            [x],
            [mean],
            yerr=np.asarray([[mean - point["ci95_low"]], [point["ci95_high"] - mean]]),
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1,
        )

    for axis, (source, target) in zip(
        [item for row in axes for item in row],
        MODEL_PAIRS,
    ):
        pair = f"{source}->{target}"
        points = summary["series"][pair]
        for index, alignment in enumerate(latent_alignments):
            color = ALIGNMENT_COLORS[alignment]
            blind = point_for(points, alignment, "blind")
            visible = point_for(points, alignment, "visible")
            blind_x = index - width / 2
            visible_x = index + width / 2
            axis.bar(blind_x, blind["accuracy"], width=width, color=color, alpha=0.9)
            axis.bar(
                visible_x,
                visible["accuracy"],
                width=width,
                color=color,
                alpha=0.45,
                edgecolor="black",
                hatch="///",
            )
            add_errorbar(axis, blind_x, blind)
            add_errorbar(axis, visible_x, visible)

        text_index = len(latent_alignments)
        text_point = point_for(points, "text", "direct_text")
        axis.bar(
            text_index,
            text_point["accuracy"],
            width=0.58,
            color=ALIGNMENT_COLORS["text"],
            alpha=0.85,
        )
        add_errorbar(axis, text_index, text_point)
        axis.set_xticks(
            np.arange(len(latent_alignments) + 1),
            [*latent_alignments, "text only"],
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("ARC-Easy accuracy")
        axis.set_title(
            f"{source.rsplit('/', 1)[-1]} -> {target.rsplit('/', 1)[-1]}"
        )
        axis.grid(axis="y", alpha=0.2)

    legend_handles = [
        Patch(facecolor="#666666", alpha=0.9, label="Latent; B question-blind"),
        Patch(
            facecolor="#999999",
            alpha=0.45,
            edgecolor="black",
            hatch="///",
            label="Latent + original question",
        ),
        Patch(facecolor=ALIGNMENT_COLORS["text"], alpha=0.85, label="Direct text; no A"),
    ]
    figure.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(
        "M0: aligned Agent-A prefill states with and without B seeing the question\n"
        "Bars=accuracy over 100 questions; error bars=question bootstrap 95% CI"
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _run_directory(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = _sha256_bytes(json.dumps(vars(args), sort_keys=True, default=str).encode("utf-8"))[:8]
    path = RUNS_DIR / f"m0_arc_easy_q{args.max_questions}_prefill_{timestamp}_{digest}"
    suffix = 1
    while path.exists():
        path = RUNS_DIR / f"{path.name}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def main(argv=None):
    args = parse_args(argv)
    logger = _configure_logger()
    set_seed(args.generation_seed)
    run_dir = _run_directory(args)
    manifest_path = run_dir / "run_manifest.json"
    started = time.time()
    manifest = {
        "status": "running",
        "study": "m0",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "run_directory": str(run_dir),
    }
    _write_json(manifest_path, manifest)
    try:
        items = sampled_items(args)
        sample_identity = _sample_identity(items, args)
        direct_text_records = [
            {
                "item_id": int(item_id),
                "question": item["question"],
                "gold": item.get("gold", ""),
            }
            for item_id, item in items
        ]
        wrappers = {}

        def wrapper_for(model_name, alignment="identical"):
            wrapper = wrappers.get(model_name)
            model_args = _model_args(args, alignment)
            if wrapper is None:
                wrapper = ModelWrapper(
                    model_name,
                    args.device,
                    use_vllm=False,
                    args=model_args,
                )
                wrappers[model_name] = wrapper
            else:
                wrapper.args = model_args
                wrapper.align_method = alignment
            return wrapper

        trajectories = {}
        trajectory_manifests = {}
        trajectory_cache_hits = {}
        for source_model in MODELS:
            identity = _trajectory_identity(args, source_model, sample_identity)
            trajectory, cache_manifest, path, cache_manifest_path = _load_trajectory_cache(
                args, source_model, identity
            )
            cache_hit = trajectory is not None
            if not cache_hit:
                if args.reuse_cache:
                    raise FileNotFoundError(
                        f"--reuse_cache requested but sender cache is absent: {path}"
                    )
                source = wrapper_for(source_model, "identical")
                records = _collect_sender_trajectory(source, items, args, logger)
                trajectory, cache_manifest = _save_trajectory_cache(
                    path, cache_manifest_path, records, identity
                )
            trajectories[source_model] = trajectory
            trajectory_manifests[source_model] = cache_manifest
            trajectory_cache_hits[source_model] = cache_hit

        all_rows = []
        answer_cache = {}
        for source_model, target_model in MODEL_PAIRS:
            trajectory = trajectories[source_model]
            trajectory_manifest = trajectory_manifests[source_model]
            for alignment in args.alignments:
                visibility_modes = (
                    ("direct_text",)
                    if alignment == "text"
                    else ("blind", "visible")
                )
                for receiver_visibility in visibility_modes:
                    identity = _result_identity(
                        args,
                        source_model,
                        target_model,
                        alignment,
                        sample_identity,
                        trajectory_manifest["trajectory_sha256"],
                        receiver_visibility,
                    )
                    paths = _result_paths(
                        args, source_model, target_model, alignment, identity
                    )
                    rows, result_manifest = _load_result_cache(args, identity, paths)
                    cache_hit = rows is not None
                    if not cache_hit:
                        if args.reuse_cache:
                            raise FileNotFoundError(
                                "--reuse_cache requested but answer cache is absent: "
                                f"{paths[0]}"
                            )
                        if alignment == "text":
                            source = source_model
                            target = wrapper_for(target_model, "identical")
                            cell_records = direct_text_records
                        else:
                            source = wrapper_for(source_model, alignment)
                            target = wrapper_for(target_model, alignment)
                            cell_records = trajectory["records"]
                        rows = _run_receiver_cell(
                            cell_records,
                            source,
                            target,
                            alignment,
                            receiver_visibility,
                            args,
                            logger,
                        )
                        result_manifest = _save_result_cache(paths, rows, identity)
                    all_rows.extend(rows)
                    answer_cache[
                        f"{source_model}->{target_model}.{alignment}.{receiver_visibility}"
                    ] = {
                        "cache_hit": cache_hit,
                        "metrics": str(paths[0]),
                        "manifest": str(paths[1]),
                        "metrics_sha256": result_manifest["metrics_sha256"],
                    }

        conditions_per_pair = sum(
            1 if alignment == "text" else 2 for alignment in args.alignments
        )
        expected = len(MODEL_PAIRS) * conditions_per_pair * args.max_questions
        if len(all_rows) != expected:
            raise RuntimeError(f"M0 row count mismatch: {len(all_rows)} != {expected}")
        if any(
            bool(row["receiver_question_visible"])
            != (row["receiver_visibility"] != "blind")
            for row in all_rows
        ):
            raise RuntimeError("M0 receiver visibility invariant failed")

        metrics_path = run_dir / "metrics" / "m0_answers.parquet"
        summary_path = run_dir / "summaries" / "m0_summary.json"
        figure_path = run_dir / "figures" / "m0_accuracy.pdf"
        _write_parquet(metrics_path, all_rows)
        summary = _summarize(all_rows, args)
        _write_json(summary_path, summary)
        _plot(summary, figure_path, args)
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": time.time() - started,
                "sample_identity": sample_identity,
                "row_count": len(all_rows),
                "unexpected_receiver_question_visibility_count": 0,
                "question_blind_row_count": sum(
                    row["receiver_visibility"] == "blind" for row in all_rows
                ),
                "latent_question_visible_row_count": sum(
                    row["receiver_visibility"] == "visible" for row in all_rows
                ),
                "direct_text_visible_row_count": sum(
                    row["receiver_visibility"] == "direct_text" for row in all_rows
                ),
                "sender_trajectory_cache_hits": trajectory_cache_hits,
                "answer_cache": answer_cache,
                "artifacts": {
                    "metrics": str(metrics_path),
                    "summary": str(summary_path),
                    "figure": str(figure_path),
                },
            }
        )
        _write_json(manifest_path, manifest)
        logger.info("M0 completed: %s", run_dir)
        return run_dir
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": time.time() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
