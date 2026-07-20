import argparse
import json
import os

# Must be set before importing transformers/huggingface_hub.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
from models import ModelWrapper
from utils import auto_device, set_seed
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct

def configure_run_files(args: argparse.Namespace) -> Tuple[logging.Logger, Path]:
    """Create per-run detail and summary output files."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.task}_{args.method}_{run_id}"

    log_dir = Path("logging")
    result_dir = Path("result")
    log_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("run_details")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_dir / f"{run_name}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.info("Run configuration:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    return logger, result_dir / f"{run_name}.json"


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
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--align_method", dest="align_method", choices=["identical", "linear", "kernel"], default="identical",
                        help="Latent-to-input alignment: identity with norm scaling, linear least-squares, or ORF kernel approximation.")
    parser.add_argument("--align_ridge", dest="align_ridge", type=float, default=1e-5,
                        help="Ridge regularization for --align_method linear.")
    parser.add_argument("--kernel_features", dest="kernel_features", type=int, default=1024,
                        help="Number m of orthogonal random features for --align_method kernel.")
    parser.add_argument("--kernel_temperature", dest="kernel_temperature", type=float, default=1.0,
                        help="Kernel softmax temperature tau; distinct from generation temperature.")
    parser.add_argument("--kernel_seed", dest="kernel_seed", type=int, default=None,
                        help="ORF seed; defaults to --seed when omitted.")
    parser.add_argument("--kernel_chunk_size", dest="kernel_chunk_size", type=int, default=4096,
                        help="Vocabulary chunk size used to precompute kernel statistics.")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default=None, help="Second device for HF model (defaults to same as --device)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")
    
    # Hybrid method arguments
    parser.add_argument("--agent_models", type=str, nargs="+", default=None,
                        help="List of models for each agent in hybrid mode (e.g., 'Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-8B Qwen/Qwen2.5-0.5B-Instruct')")

    args = parser.parse_args()
    
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
    
    summary = {
        "run": {
            "method": args.method,
            "model": args.model_name,
            "task": args.task,
            "split": args.split,
            "seed": args.seed,
        },
        "results": {
            "processed": processed,
            "correct": correct,
            "accuracy": round(acc, 6),
        },
        "timing": {
            "total_seconds": round(total_time, 4),
            "seconds_per_sample": round(total_time / processed, 4) if processed else 0.0,
        },
    }
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Final summary written to %s:\n%s", result_path, json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
