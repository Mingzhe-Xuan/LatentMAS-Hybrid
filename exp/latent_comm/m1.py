"""M1: two-role kernel-latent budget and entropy-probe study on AIME2024."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_aime2024
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, set_seed


MODEL_NAME = "Qwen/Qwen3-8B"
K_VALUES = (0, 40, 80, 120, 160)
CACHE_DIR = ROOT / "exp" / "cache" / "latent_comm_m1"
RUNS_DIR = ROOT / "exp_result" / "latent_comm" / "runs"
CACHE_SCHEMA_VERSION = 1
PROMPT_VERSION = "m1_sender_planner_kv_receiver_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["m1"], default="m1")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--dataset", choices=["aime2024"], default="aime2024")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Defaults to the complete AIME2024 split.")
    parser.add_argument("--latent_step_values", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--probe_seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--bootstrap_replicates", type=int, default=1000)
    parser.add_argument("--kernel_features", "--m", dest="kernel_features", type=int, default=1024)
    parser.add_argument("--kernel_temperature", "--tau", dest="kernel_temperature", type=float, default=0.6)
    parser.add_argument("--kernel_seed", "--orf_seed", dest="kernel_seed", type=int, default=101)
    parser.add_argument("--kernel_chunk_size", type=int, default=4096)
    parser.add_argument("--align_ridge", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument("--force_recollect", action="store_true")
    args = parser.parse_args(argv)
    if args.model_name != MODEL_NAME:
        parser.error(f"M1 requires --model_name {MODEL_NAME}")
    if tuple(args.latent_step_values) != K_VALUES:
        parser.error(f"M1 requires --latent_step_values {' '.join(map(str, K_VALUES))}")
    if args.max_questions is not None and args.max_questions < 1:
        parser.error("--max_questions must be positive")
    if args.max_new_tokens < 1 or args.bootstrap_replicates < 1:
        parser.error("generation/bootstrap sizes must be positive")
    if args.kernel_features < 1 or args.kernel_chunk_size < 1 or args.kernel_temperature <= 0:
        parser.error("kernel settings must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("--temperature must be non-negative and --top_p must be in (0, 1]")
    if args.reuse_cache and args.force_recollect:
        parser.error("--reuse_cache and --force_recollect are mutually exclusive")
    args.task = "aime2024"
    args.align_method = "kernel"
    args.soft_temperature = args.kernel_temperature
    args.seed = args.generation_seed
    args.device2 = args.device
    args.use_vllm = False
    return args


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
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


def _logger():
    logger = logging.getLogger("latent_comm.m1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(Path.cwd() / "exp_state.txt", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _items(args):
    items = list(enumerate(load_aime2024(split=args.split)))
    if args.max_questions is not None:
        items = items[:args.max_questions]
    if not items:
        raise RuntimeError("AIME2024 selection is empty")
    return items


def _sample_identity(items, args):
    payload = [{"item_id": item_id, "question": item["question"], "gold": item["gold"]} for item_id, item in items]
    return {
        "dataset": args.dataset,
        "split": args.split,
        "selection": "dataset_order_full_prefix",
        "item_ids": [item_id for item_id, _ in items],
        "content_sha256": _sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()),
    }


def _cache_identity(args, sample):
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "experiment": "m1_two_role_kernel_kv_entropy_probe",
        "model_name": args.model_name,
        "dataset": args.dataset,
        "split": args.split,
        "sample": sample,
        "latent_step_values": list(args.latent_step_values),
        "alignment": "kernel",
        "prompt_version": PROMPT_VERSION,
        "generation": {"seed": args.generation_seed, "max_new_tokens": args.max_new_tokens,
                       "temperature": args.temperature, "top_p": args.top_p, "think": args.think},
        "kernel": {"features": args.kernel_features, "temperature": args.kernel_temperature,
                   "seed": args.kernel_seed, "chunk_size": args.kernel_chunk_size},
    }


def _cache_paths(identity):
    digest = _sha256(json.dumps(identity, sort_keys=True).encode())[:20]
    stem = CACHE_DIR / f"m1_aime2024_q{len(identity['sample']['item_ids'])}_{digest}"
    return stem.with_suffix(".answers.parquet"), stem.with_suffix(".entropy.parquet"), stem.with_suffix(".manifest.json")


def _load_cache(args, identity):
    answers_path, entropy_path, manifest_path = _cache_paths(identity)
    exists = [path.exists() for path in (answers_path, entropy_path, manifest_path)]
    if any(exists) and not all(exists) and not args.force_recollect:
        raise RuntimeError(f"Incomplete M1 cache: {manifest_path}")
    if not all(exists) or args.force_recollect:
        if args.reuse_cache:
            raise FileNotFoundError(f"--reuse_cache requested but M1 cache is absent: {manifest_path}")
        return None, None, None, (answers_path, entropy_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_identity") != identity:
        raise RuntimeError(f"Incompatible M1 cache: {manifest_path}")
    if _sha256_file(answers_path) != manifest.get("sha256", {}).get("answers"):
        raise RuntimeError("M1 answer cache SHA256 integrity check failed")
    if _sha256_file(entropy_path) != manifest.get("sha256", {}).get("entropy"):
        raise RuntimeError("M1 entropy cache SHA256 integrity check failed")
    return _read_parquet(answers_path), _read_parquet(entropy_path), manifest, (answers_path, entropy_path, manifest_path)


def _save_cache(paths, identity, answers, entropy, model_info):
    answers_path, entropy_path, manifest_path = paths
    _write_parquet(answers_path, answers)
    _write_parquet(entropy_path, entropy)
    manifest = {"cache_identity": identity, "created_at": datetime.now().isoformat(timespec="seconds"),
                "row_count": {"answers": len(answers), "entropy": len(entropy)},
                "sha256": {"answers": _sha256_file(answers_path), "entropy": _sha256_file(entropy_path)},
                "model": model_info}
    _write_json(manifest_path, manifest)
    return manifest


def _sender_messages(question, args):
    return build_agent_message_sequential_latent_mas(
        role="planner", question=question, context="", method="latent_mas", args=args
    )


def _receiver_continuation():
    # Sender's prompt ends in an assistant generation slot. Close that slot and
    # start a separate receiver turn while retaining every sender KV entry.
    return (
        "<|im_end|>\n<|im_start|>user\n"
        "You are the Receiver/Judger. The complete cached context from a Planner "
        "contains the target AIME problem and latent reasoning. Solve that target "
        "problem step by step. End with the numeric answer in \\boxed{YOUR_FINAL_ANSWER}."
        "\n<|im_end|>\n<|im_start|>assistant\n"
    )


def _sync(device):
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _entropy(hidden, output_head):
    logits = output_head(hidden).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return float((-(log_probs.exp() * log_probs).sum(dim=-1))[0].item())


@torch.inference_mode()
def _sender_rollout(wrapper, prompt_ids, prompt_mask, latent_steps):
    """Run one Sender prefix and retain its real KV cache for the Receiver."""
    model = wrapper.model
    _sync(wrapper.device)
    started = time.perf_counter()
    outputs = model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True,
                    output_hidden_states=True, return_dict=True)
    past = outputs.past_key_values
    hidden = outputs.hidden_states[-1][:, -1, :]
    output_head = model.get_output_embeddings()
    if output_head is None:
        raise RuntimeError("M1 requires an accessible Sender output embedding")
    # Entropy is a probe: exclude its full-vocabulary readout cost from the
    # Sender/total timings used for the fixed-budget comparison.
    probe_seconds = 0.0
    probe_started = time.perf_counter(); entropy_rows = [(0, _entropy(hidden, output_head))]; probe_seconds += time.perf_counter() - probe_started
    for step in range(1, latent_steps + 1):
        latent_embed = wrapper._apply_latent_realignment(hidden, model).unsqueeze(1)
        attention_mask = torch.ones((1, _past_length(past) + 1), dtype=torch.long, device=wrapper.device)
        outputs = model(inputs_embeds=latent_embed, attention_mask=attention_mask,
                        past_key_values=past, use_cache=True, output_hidden_states=True,
                        return_dict=True)
        past = outputs.past_key_values
        hidden = outputs.hidden_states[-1][:, -1, :]
        probe_started = time.perf_counter()
        entropy_rows.append((step, _entropy(hidden, output_head)))
        probe_seconds += time.perf_counter() - probe_started
    _sync(wrapper.device)
    return past, entropy_rows, time.perf_counter() - started - probe_seconds, probe_seconds


def _parse_answer(text):
    return normalize_answer(extract_gsm8k_answer(text))


@torch.inference_mode()
def _collect(args, items, logger):
    wrapper = ModelWrapper(args.model_name, args.device, use_vllm=False, args=args)
    wrapper.align_method = "kernel"
    answers, entropy = [], []
    receiver_text = _receiver_continuation()
    receiver_encoded = wrapper.tokenizer(receiver_text, return_tensors="pt", add_special_tokens=False)
    receiver_ids = receiver_encoded["input_ids"].to(wrapper.device)
    receiver_mask = receiver_encoded["attention_mask"].to(wrapper.device)
    for ordinal, (item_id, item) in enumerate(items, start=1):
        sender_messages = _sender_messages(item["question"], args)
        sender_prompt = wrapper.render_chat(sender_messages, add_generation_prompt=True)
        if args.think:
            sender_prompt += "<think>"
        encoded = wrapper.tokenizer(sender_prompt, return_tensors="pt", add_special_tokens=False)
        sender_ids = encoded["input_ids"].to(wrapper.device)
        sender_mask = encoded["attention_mask"].to(wrapper.device)
        if item["question"] not in sender_prompt:
            raise RuntimeError("M1 Sender prompt omitted the target question")
        for budget in args.latent_step_values:
            set_seed(args.generation_seed)
            past, observed, sender_seconds, probe_seconds = _sender_rollout(wrapper, sender_ids, sender_mask, budget)
            receiver_started = time.perf_counter()
            generated, _ = wrapper.generate_text_batch(
                receiver_ids, receiver_mask, past_key_values=past,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
            )
            receiver_seconds = time.perf_counter() - receiver_started
            raw_prediction = generated[0]
            prediction = _parse_answer(raw_prediction)
            gold = normalize_answer(item["gold"])
            for step, value in observed:
                entropy.append({"dataset": args.dataset, "split": args.split, "item_id": int(item_id),
                                "latent_steps_per_agent": int(budget), "local_step": int(step),
                                "entropy_nats": value})
            answers.append({
                "dataset": args.dataset, "split": args.split, "item_id": int(item_id),
                "latent_steps_per_agent": int(budget), "total_latent_steps": int(budget),
                "alignment": "kernel", "transfer_mode": "direct_sender_past_key_values",
                "sender_model": args.model_name, "receiver_model": args.model_name,
                "sender_prompt_token_count": int(sender_mask.sum().item()),
                "receiver_prompt_token_count": int(receiver_mask.sum().item()),
                "sender_kv_token_count": int(_past_length(past)),
                "question": item["question"], "gold": gold, "prediction": prediction,
                "raw_prediction": raw_prediction, "correct": bool(prediction == gold),
                "parse_success": prediction is not None,
                "receiver_text_output_tokens": int(wrapper.last_generation_metrics["output_token_counts"][0]),
                "sender_wall_seconds": sender_seconds, "receiver_wall_seconds": receiver_seconds,
                "entropy_probe_seconds": probe_seconds,
                "total_wall_seconds": sender_seconds + receiver_seconds,
            })
            logger.info("M1 question=%d/%d item_id=%d K=%d correct=%s", ordinal, len(items), item_id, budget, prediction == gold)
    model_info = {"name": args.model_name, "parameter_dtype": str(next(wrapper.model.parameters()).dtype),
                  "resolved_commit": getattr(wrapper.model.config, "_commit_hash", None)}
    return answers, entropy, model_info


def _bootstrap(values, replicates, seed):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(replicates, len(array)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _summary(answers, entropy, args):
    series = []
    for index, budget in enumerate(args.latent_step_values):
        rows = [row for row in answers if row["latent_steps_per_agent"] == budget]
        last = [row["entropy_nats"] for row in entropy if row["latent_steps_per_agent"] == budget and row["local_step"] == budget]
        entry = {"latent_steps_per_agent": budget, "questions": len(rows),
                 "accuracy": float(np.mean([row["correct"] for row in rows])),
                 "correct": int(sum(row["correct"] for row in rows)),
                 "entropy_last_mean": float(np.mean(last)),
                 "entropy_last_ci95": _bootstrap(last, args.bootstrap_replicates, args.probe_seed + 1000 + index)}
        entry["accuracy_ci95"] = _bootstrap([float(row["correct"]) for row in rows], args.bootstrap_replicates, args.probe_seed + index)
        for metric_offset, metric in enumerate(("receiver_text_output_tokens", "total_wall_seconds", "sender_wall_seconds", "receiver_wall_seconds"), start=1):
            values = [float(row[metric]) for row in rows]
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_ci95"] = _bootstrap(values, args.bootstrap_replicates, args.probe_seed + index + metric_offset * 100)
        series.append(entry)
    return {"study": "m1", "design": {"sender": args.model_name, "receiver": args.model_name,
            "dataset": args.dataset, "split": args.split, "alignment": "kernel",
            "transfer": "receiver directly reuses Sender past_key_values including prompt KV cache",
            "latent_step_values": list(args.latent_step_values), "entropy": "Sender output-logit entropy; probe only"},
            "series": series}


def _plot_metric(summary, metric, label, path):
    points = summary["series"]
    x = [point["latent_steps_per_agent"] for point in points]
    y = [point[f"{metric}_mean"] if metric != "accuracy" else point["accuracy"] for point in points]
    ci_key = f"{metric}_ci95" if metric != "accuracy" else "accuracy_ci95"
    lower = [point[ci_key][0] for point in points]
    upper = [point[ci_key][1] for point in points]
    entropy = [point["entropy_last_mean"] for point in points]
    entropy_low = [point["entropy_last_ci95"][0] for point in points]
    entropy_high = [point["entropy_last_ci95"][1] for point in points]
    figure, left = plt.subplots(figsize=(7.2, 4.6))
    left.plot(x, y, marker="o", color="#4c78a8", label=label)
    left.fill_between(x, lower, upper, color="#4c78a8", alpha=0.2)
    left.set_xlabel("Fixed Sender latent steps (K)")
    left.set_ylabel(label, color="#4c78a8")
    left.tick_params(axis="y", labelcolor="#4c78a8")
    left.grid(axis="x", alpha=0.25)
    if metric == "accuracy":
        left.set_ylim(0, 1)
    right = left.twinx()
    right.plot(x, entropy, marker="s", linestyle="--", color="#e45756", label="Sender entropy (last step)")
    right.fill_between(x, entropy_low, entropy_high, color="#e45756", alpha=0.15)
    right.set_ylabel("Sender output-logit entropy (nats)", color="#e45756")
    right.tick_params(axis="y", labelcolor="#e45756")
    handles, labels = [], []
    for axis in (left, right):
        h, l = axis.get_legend_handles_labels(); handles += h; labels += l
    left.legend(handles, labels, loc="best")
    figure.suptitle(f"M1 AIME2024: {label} and Sender entropy vs latent budget")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _run_dir(args):
    digest = _sha256(json.dumps(vars(args), sort_keys=True, default=str).encode())[:8]
    path = RUNS_DIR / f"m1_aime2024_Qwen3-8B_k{'-'.join(map(str, args.latent_step_values))}_q{args.max_questions or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def main(argv=None):
    args = parse_args(argv)
    logger = _logger()
    items = _items(args)
    sample = _sample_identity(items, args)
    identity = _cache_identity(args, sample)
    answers, entropy, cache_manifest, cache_paths = _load_cache(args, identity)
    run_dir = _run_dir(args)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {"status": "running", "study": "m1", "started_at": datetime.now().isoformat(timespec="seconds"),
                "args": vars(args), "cache_identity": identity}
    _write_json(manifest_path, manifest)
    started = time.time()
    try:
        cache_hit = answers is not None
        if not cache_hit:
            answers, entropy, model_info = _collect(args, items, logger)
            expected_answers = len(items) * len(args.latent_step_values)
            expected_entropy = len(items) * sum(budget + 1 for budget in args.latent_step_values)
            if len(answers) != expected_answers or len(entropy) != expected_entropy:
                raise RuntimeError(f"M1 row-count invariant failed: answers={len(answers)}/{expected_answers}, entropy={len(entropy)}/{expected_entropy}")
            cache_manifest = _save_cache(cache_paths, identity, answers, entropy, model_info)
        summary = _summary(answers, entropy, args)
        metrics = run_dir / "metrics" / "m1_accuracy_cost_by_question.parquet"
        entropy_metrics = run_dir / "metrics" / "m1_sender_entropy_by_step.parquet"
        summary_path = run_dir / "summaries" / "m1_summary.json"
        _write_parquet(metrics, answers); _write_parquet(entropy_metrics, entropy); _write_json(summary_path, summary)
        figures = {"accuracy": "Accuracy", "receiver_text_output_tokens": "Receiver text-output tokens",
                   "total_wall_seconds": "Total wall time (seconds)", "sender_wall_seconds": "Sender wall time (seconds)",
                   "receiver_wall_seconds": "Receiver wall time (seconds)"}
        figure_paths = {}
        for metric, label in figures.items():
            path = run_dir / "figures" / f"m1_{metric}_vs_steps_entropy.pdf"
            _plot_metric(summary, metric, label, path)
            figure_paths[metric] = str(path)
        manifest.update({"status": "completed", "completed_at": datetime.now().isoformat(timespec="seconds"),
                         "elapsed_seconds": time.time() - started, "cache_hit": cache_hit,
                         "cache_manifest": str(cache_paths[2]), "cache_created_at": cache_manifest.get("created_at"),
                         "row_count": {"answers": len(answers), "entropy": len(entropy)},
                         "artifacts": {"answers": str(metrics), "entropy": str(entropy_metrics), "summary": str(summary_path), "figures": figure_paths}})
        _write_json(manifest_path, manifest)
        logger.info("M1 completed: %s", run_dir)
        return run_dir
    except Exception as error:
        manifest.update({"status": "failed", "failed_at": datetime.now().isoformat(timespec="seconds"),
                         "elapsed_seconds": time.time() - started, "error": f"{type(error).__name__}: {error}"})
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
