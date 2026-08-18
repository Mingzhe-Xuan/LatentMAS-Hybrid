import argparse
import json
import os
import re

# Must be set before importing transformers/huggingface_hub.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.latent_mas_hybrid import LatentMASMethod as LatentMASHybridMethod
from methods.text_mas import TextMASMethod
from reasoning_models import resolve_manual_think
from models import ModelWrapper
from utils import auto_device, set_seed
import time

DEFAULT_MAX_NEW_TOKENS = 20000
PARAMS_CONFIG_PATH = Path(__file__).with_name("params_dict.json")


def default_max_new_tokens(task: str) -> int:
    """Return the task-specific generation limit, or the safe fallback."""
    try:
        task_params = json.loads(PARAMS_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_MAX_NEW_TOKENS

    params = task_params.get(task, {})
    value = params.get("max_token", DEFAULT_MAX_NEW_TOKENS) if isinstance(params, dict) else DEFAULT_MAX_NEW_TOKENS
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return DEFAULT_MAX_NEW_TOKENS
    return value

def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def count_output_tokens(preds: List[Dict], tokenizer: Any) -> int:
    """Count visible agent-output tokens, excluding prompts and latent states."""
    outputs = [
        output
        for pred in preds
        for agent in pred.get("agents", [])
        if isinstance((output := agent.get("output")), str) and output
    ]
    if not outputs:
        return 0

    encoded = tokenizer(outputs, add_special_tokens=False)
    return sum(len(token_ids) for token_ids in encoded["input_ids"])


def summarize_role_metrics(preds: List[Dict]) -> Tuple[Dict, Dict, Dict]:
    """Aggregate token and model-stage timing metrics independently by role."""
    token_fields = ("text_input", "latent_input", "text_output", "latent_output")
    timing_fields = (
        "prefill_seconds",
        "latent_decode_seconds",
        "alignment_seconds",
        "text_decode_seconds",
    )
    buckets: Dict[str, Dict] = {}
    for pred in preds:
        for agent in pred.get("agents", []):
            metrics = agent.get("metrics")
            if not isinstance(metrics, dict):
                continue
            role = str(agent.get("role") or agent.get("name") or "unknown").lower()
            bucket = buckets.setdefault(
                role,
                {
                    "samples": 0,
                    "tokens": {field: 0 for field in token_fields},
                    "timing": {field: 0.0 for field in timing_fields},
                    "timing_sources": set(),
                },
            )
            bucket["samples"] += 1
            token_values = metrics.get("tokens", {})
            timing_values = metrics.get("timing", {})
            for field in token_fields:
                bucket["tokens"][field] += int(token_values.get(field, 0))
            for field in timing_fields:
                bucket["timing"][field] += float(timing_values.get(field, 0.0))
            if timing_values.get("source"):
                bucket["timing_sources"].add(str(timing_values["source"]))

    roles: Dict[str, Dict] = {}
    overall_tokens = {field: 0 for field in token_fields}
    overall_timing = {field: 0.0 for field in timing_fields}
    for role, bucket in buckets.items():
        samples = bucket["samples"]
        text_out = bucket["tokens"]["text_output"]
        latent_out = bucket["tokens"]["latent_output"]
        output_type = (
            "mixed" if text_out and latent_out else "text" if text_out else "latent" if latent_out else "none"
        )
        role_tokens = {}
        for field in token_fields:
            total = bucket["tokens"][field]
            overall_tokens[field] += total
            role_tokens[field] = {
                "total": total,
                "average_per_problem": round(total / samples, 4) if samples else 0.0,
            }
        role_timing = {}
        for field in timing_fields:
            total = bucket["timing"][field]
            overall_timing[field] += total
            if total or field == "prefill_seconds":
                role_timing[field] = {
                    "total": round(total, 6),
                    "average_per_problem": round(total / samples, 6) if samples else 0.0,
                }
        roles[role] = {
            "samples": samples,
            "output_type": output_type,
            "tokens": role_tokens,
            "timing": role_timing,
            "timing_sources": sorted(bucket["timing_sources"]),
        }

    problem_count = len(preds)
    token_summary = {
        field: {
            "total": total,
            "average_per_problem": round(total / problem_count, 4) if problem_count else 0.0,
        }
        for field, total in overall_tokens.items()
    }
    timing_summary = {
        field: {
            "total": round(total, 6),
            "average_per_problem": round(total / problem_count, 6) if problem_count else 0.0,
        }
        for field, total in overall_timing.items()
    }
    return roles, token_summary, timing_summary

def configure_run_files(args: argparse.Namespace) -> Tuple[logging.Logger, Optional[Path]]:
    """Configure the detail log and, unless disabled, the summary output path."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Model identifiers commonly contain path separators (for example,
    # ``Qwen/Qwen3-8B``), which cannot be used directly in a filename.
    model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_name).strip("._-")
    run_name = (
        f"{args.task}_{args.method}_prompt_{args.prompt}_model_{model_name}_"
        f"align_{args.align_method}_{run_id}"
    )

    logger = logging.getLogger("run_details")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path("logging")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{run_name}.log"

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.info("Run configuration:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    if args.no_write_result:
        result_path = None
    elif args.result_path:
        result_path = Path(args.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = Path("result")
        result_dir.mkdir(exist_ok=True)
        result_path = result_dir / f"{run_name}.json"

    return logger, result_path


def log_problem_detail(logger: logging.Logger, problem_idx: int, result: Dict) -> None:
    """Write the full result for one problem to the per-run detail log."""
    lines = [
        "=" * 20 + f" Problem #{problem_idx} " + "=" * 20,
        "Question:",
        result.get("question", "").strip(),
    ]
    for agent in result.get("agents", []):
        name = agent.get("name", "Agent")
        role = agent.get("role", "")
        lines.extend([
            f"----- Agent: {name} ({role}) -----",
            "[To Tokenize]",
            agent.get("input", "").rstrip(),
        ])
        latent_steps = agent.get("latent_steps")
        if latent_steps is not None:
            lines.extend(["[Latent Steps]", str(latent_steps)])
        metrics = agent.get("metrics")
        if metrics:
            lines.extend(["[Metrics]", json.dumps(metrics, ensure_ascii=False)])
        lines.extend([
            "[Output]",
            agent.get("output", "").rstrip(),
            "-" * 46,
        ])
    lines.append(
        f"Result: Pred={result.get('prediction')} | Gold={result.get('gold')} | "
        f"OK={result.get('correct')}"
    )
    logger.info("\n".join(lines))

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        log_problem_detail(logger, problem_idx, res)

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas", "latent_mas_hybrid"], required=True,
                        help="Which multi-agent method to run: 'baseline', 'text_mas', 'latent_mas', or 'latent_mas_hybrid'.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name to use (e.g. 'Qwen/Qwen3-8B', 'Qwen/Qwen2.5-1.5B-Instruct', etc.)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="humanevalplus",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Maximum new tokens per agent; defaults to params_dict.json[task].max_token, otherwise 20000.")
    parser.add_argument("--latent_steps", type=int, default=45, help="Number of latent steps for LatentMAS method")
    parser.add_argument(
        "--sequential_info_only",
        action="store_true",
        help=(
            "For LatentMAS methods, retain only the current agent's prompt and "
            "latent KV cache before passing context to the next agent."
        ),
    )
    parser.add_argument(
        "--latent_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For LatentMAS methods, retain only latent-step KV cache before "
            "passing context to the next agent; implies --sequential_info_only."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether LatentMAS manually adds <think>. By default this is disabled "
            "for registered reasoning models whose chat template adds it, and "
            "enabled for all other models."
        ),
    )
    parser.add_argument("--align_method", dest="align_method", choices=["identical", "linear", "kernel", "kernel_early_stopping", "soft"], default="identical",
                        help="Latent-to-input alignment: identity, linear, kernel, entropy-stopped kernel, or exact soft-token expectation.")
    parser.add_argument("--align_ridge", dest="align_ridge", type=float, default=1e-5,
                        help="Ridge regularization for --align_method linear.")
    parser.add_argument("--kernel_features", dest="kernel_features", type=int, default=1024,
                        help="Number m of orthogonal random features for --align_method kernel.")
    parser.add_argument("--kernel_temperature", dest="kernel_temperature", type=float, default=0.6,
                        help="Kernel softmax temperature tau; distinct from generation temperature.")
    parser.add_argument("--kernel_seed", dest="kernel_seed", type=int, default=None,
                        help="ORF seed; defaults to --seed when omitted.")
    parser.add_argument("--kernel_chunk_size", dest="kernel_chunk_size", type=int, default=4096,
                        help="Vocabulary chunk size used to precompute kernel statistics.")
    parser.add_argument("--soft_temperature", dest="soft_temperature", type=float, default=0.6,
                        help="Exact soft-token temperature; distinct from generation and kernel temperatures.")
    parser.add_argument("--soft_chunk_size", dest="soft_chunk_size", type=int, default=32,
                        help="Number of hidden queries per exact softmax chunk.")
    parser.add_argument("--early_stopping_length_threshold", type=int, default=None,
                        help="Low-entropy latent-step span; kernel_early_stopping samples every 10 steps, so its default 20 requires two consecutive low-entropy samples.")
    parser.add_argument("--early_stopping_entropy_threshold", type=float, default=None,
                        help="Low-entropy cutoff; defaults to 0.25 for kernel_early_stopping and 0.01 otherwise.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        help="Optional exact path for the standalone JSON summary.",
    )
    parser.add_argument(
        "--no_write_result",
        action="store_true",
        help="Do not create or write a JSON result file; the final summary is still logged.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Optional exact path for the detail log.",
    )


    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Allow a model repository to provide custom Python code.")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default=None, help="Second device for HF model (defaults to same as --device)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")
    
    # Hybrid method arguments
    parser.add_argument("--agent_models", type=str, nargs="+", default=None,
                        help="List of models for each agent in hybrid mode (e.g., 'Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-8B Qwen/Qwen2.5-0.5B-Instruct')")

    args = parser.parse_args()

    args.think_requested = args.think
    args.think = resolve_manual_think(args.model_name, args.think)

    if args.early_stopping_length_threshold is None:
        args.early_stopping_length_threshold = (
            20 if args.align_method == "kernel_early_stopping" else 256
        )
    if args.early_stopping_entropy_threshold is None:
        args.early_stopping_entropy_threshold = (
            0.25 if args.align_method == "kernel_early_stopping" else 0.01
        )

    if args.soft_temperature <= 0:
        parser.error("--soft_temperature must be positive")
    if args.soft_chunk_size <= 0:
        parser.error("--soft_chunk_size must be positive")
    if args.early_stopping_length_threshold <= 0:
        parser.error("--early_stopping_length_threshold must be positive")
    if args.early_stopping_entropy_threshold < 0:
        parser.error("--early_stopping_entropy_threshold must be non-negative")

    # An explicit --max_new_tokens value takes precedence over the task default.
    if args.max_new_tokens is None:
        args.max_new_tokens = default_max_new_tokens(args.task)
    
    # Default device2 to device if not specified
    if args.device2 is None:
        args.device2 = args.device
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    if args.kernel_seed is None:
        args.kernel_seed = args.seed

    set_seed(args.seed)
    logger, result_path = configure_run_files(args)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )
    elif args.method == 'latent_mas_hybrid':
        method = LatentMASHybridMethod(
            model,
            agent_models=args.agent_models,  # Can be None (same model) or list of models
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples, desc="Evaluating", unit="problem")

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
                logger,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
            logger=logger,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)
    role_metrics, token_metrics, phase_timing = summarize_role_metrics(preds)
    output_tokens = token_metrics["text_output"]["total"]
    if not role_metrics:
        output_tokens = count_output_tokens(preds, model.tokenizer)
    
    summary = {
        "run": {
            "method": args.method,
            "prompt": args.prompt,
            "align_method": args.align_method,
            "soft_temperature": args.soft_temperature,
            "soft_chunk_size": args.soft_chunk_size,
            "soft_latent_max_steps": 10000,
            "kernel_early_stopping_max_steps": 200,
            "kernel_entropy_check_interval": 10,
            "kernel_stable_change_threshold": 0.1,
            "kernel_stable_change_count": 4,
            "early_stopping_length_threshold": args.early_stopping_length_threshold,
            "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
            "model": args.model_name,
            "task": args.task,
            "split": args.split,
            "seed": args.seed,
        },
        "results": {
            "processed": processed,
            "correct": correct,
            "accuracy": round(acc, 6),
            "output_tokens": output_tokens,
            "tokens": token_metrics,
            "role_metrics": role_metrics,
        },
        "timing": {
            "total_seconds": round(total_time, 4),
            "seconds_per_sample": round(total_time / processed, 4) if processed else 0.0,
            "model_phases": phase_timing,
        },
    }
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    if result_path is None:
        logger.info("Final summary (result writing disabled):\n%s", summary_json)
    else:
        result_path.write_text(summary_json + "\n", encoding="utf-8")
        logger.info("Final summary written to %s:\n%s", result_path, summary_json)


if __name__ == "__main__":
    main()
