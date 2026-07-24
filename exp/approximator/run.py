#!/usr/bin/env python
"""Operator-layer S0--S4 experiments from ``docs/plan_v2.md``.

Exact softmax always scans the complete vocabulary.  See
``exp/approximator/IMPLEMENTATION_STATUS.md`` for explicit implemented/pending
stage requirements; this entry point must not be treated as experimental evidence
until artifacts are produced by an actual HPC run.  Every raw table, summary,
figure and manifest is written under ``exp_result/approximator``.
"""

from __future__ import annotations

import argparse, json, logging, random, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from alignment import apply_alignment, build_kernel_state
from data import (
    load_arc_challenge,
    load_arc_easy,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_medqa,
)
from methods import default_agents
from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)
from stages import common
from stages import s0, s1, s2, s3, s4

RESULT = ROOT / "exp_result" / "approximator"
SOURCES = ("prompt", "reply")


@dataclass
class State:
    vector: torch.Tensor
    item_id: int
    source: str
    position: int
    prompt_length: int
    reply_length: int
    turn_id: int = 0
    agent_id: int = 0


def configure_logger():
    RESULT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("approximator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        RESULT / "exp_state.txt", mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--study", choices=["s0", "s1", "s2", "s3", "s4", "all"], default="all"
    )
    p.add_argument("--model_pair", choices=["x1", "x2"], default="x1", help=argparse.SUPPRESS)
    p.add_argument("--source_model")
    p.add_argument("--target_model")
    p.add_argument(
        "--dataset",
        choices=["arc_easy", "arc_challenge", "gsm8k", "medqa", "mbppplus", "gpqa"],
        default="arc_easy",
    )
    p.add_argument("--split", default="test")
    p.add_argument("--max_questions", type=int, default=10)
    p.add_argument("--max_states_per_question", type=int, default=20)
    p.add_argument("--max_reply_tokens", type=int, default=512)
    p.add_argument("--prompt_limit", type=int, default=512)
    p.add_argument(
        "--prompt",
        choices=["sequential"],
        default="sequential",
        help="Only sequential latent-MAS is supported by this experiment.",
    )
    p.add_argument(
        "--role", choices=["planner", "critic", "refiner", "judger"], default="planner",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--agent_models",
        nargs="+",
        default=["Qwen/Qwen3-4B"],
        help="One shared model, or four models in Planner/Critic/Refiner/Judger order.",
    )
    p.add_argument("--latent_steps", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--reuse_trajectory", action="store_true")
    p.add_argument("--force_recollect", action="store_true")
    p.add_argument("--kernel_features", type=int, default=2048)
    p.add_argument("--kernel_temperature", type=float, default=1.0)
    p.add_argument("--kernel_seed", type=int, default=101)
    p.add_argument("--kernel_chunk_size", type=int, default=4096)
    p.add_argument("--probe_seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--skip_float64_audit", action="store_true")
    p.add_argument("--s3_replicates", type=int, default=32)
    p.add_argument("--s3_max_questions", type=int, default=50)
    p.add_argument("--bootstrap_replicates", type=int, default=1000)
    p.add_argument("--run_s2_calibration", action="store_true")
    p.add_argument("--run_s1_performance", action="store_true")
    p.add_argument(
        "--s4_tsne",
        action="store_true",
        help="Optional shared t-SNE; requires scikit-learn.",
    )
    return p.parse_args()


def resolve_model_names(args):
    default_model_names = {
        "x1": ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"),
        "x2": ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B"),
    }[args.model_pair]
    return (
        args.source_model or default_model_names[0],
        args.target_model or default_model_names[1],
    )


def resolve_agent_models(args):
    if len(args.agent_models) == 1:
        return tuple(args.agent_models * len(default_agents()))
    if len(args.agent_models) != len(default_agents()):
        raise ValueError("--agent_models requires one model or four role-ordered models")
    return tuple(args.agent_models)


def write_json(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            x,
            indent=2,
            ensure_ascii=False,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        )
        + "\n",
        encoding="utf8",
    )


def write_rows(rows, stem):
    import pyarrow as pa, pyarrow.parquet as pq

    path = RESULT / "metrics" / f"{stem}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return path


def commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def check_compatibility(
    source_tokenizer, target_tokenizer, source_model, target_model, args
):
    failures = []
    if len(source_tokenizer) != len(target_tokenizer):
        failures.append("vocab_size")
    if source_tokenizer.get_vocab() != target_tokenizer.get_vocab():
        failures.append("token_to_id")
    source_special_token_ids = {
        key: getattr(source_tokenizer, key, None)
        for k in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    }
    target_special_token_ids = {
        key: getattr(target_tokenizer, key, None) for key in source_special_token_ids
    }
    if source_special_token_ids != target_special_token_ids:
        failures.append("special_token_ids")
    payload = {
        "source_model": resolve_model_names(args)[0],
        "target_model": resolve_model_names(args)[1],
        "source_vocab_size": len(source_tokenizer),
        "target_vocab_size": len(target_tokenizer),
        "source_special_token_ids": source_special_token_ids,
        "target_special_token_ids": target_special_token_ids,
        "source_hidden_size": source_model.config.hidden_size,
        "target_hidden_size": target_model.config.hidden_size,
        "source_tied_word_embeddings": bool(
            getattr(source_model.config, "tie_word_embeddings", False)
        ),
        "target_tied_word_embeddings": bool(
            getattr(target_model.config, "tie_word_embeddings", False)
        ),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "failures": failures,
    }
    write_json(RESULT / "manifests" / "compatibility.json", payload)
    if failures:
        raise RuntimeError("Stopped: incompatible tokenizer(s): " + ", ".join(failures))


def load_data(name, split):
    return {
        "arc_easy": load_arc_easy,
        "arc_challenge": load_arc_challenge,
        "gsm8k": load_gsm8k,
        "medqa": load_medqa,
        "mbppplus": load_mbppplus,
        "gpqa": load_gpqa_diamond,
    }[name](split=split)


def positions(n, limit):
    return (
        list(range(n))
        if n <= limit
        else sorted(set(np.linspace(0, n - 1, limit, dtype=int).tolist() + [n - 1]))
    )


def trim(ids, limit):
    return ids if len(ids) <= limit else torch.cat((ids[: limit - 32], ids[-32:]))


def sample_token(logits, temperature, top_p):
    """Sample one token using the same temperature/top-p policy as latent-MAS."""
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("generation_temperature must be > 0 and generation_top_p in (0, 1]")
    sorted_logits, sorted_ids = torch.sort(logits / temperature, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0)
    probs = torch.zeros_like(sorted_probs).scatter(-1, sorted_ids, sorted_probs)
    return torch.multinomial(probs, 1).squeeze(-1)


@torch.inference_mode()
def collect(model, tok, items, args, logger):
    """First-turn source-model states from the sequential latent-MAS prompt.

    Gold answers are never injected. Text decoding uses the repository's normal
    temperature/top-p sampling policy rather than greedy decoding.
    """
    states = []
    eos = tok.eos_token_id
    for item_id, item in enumerate(items):
        started = time.time()
        logger.info(
            "State collection: question %d/%d started.", item_id + 1, len(items)
        )
        shim = argparse.Namespace(
            model_name=resolve_model_names(args)[0], task=args.dataset
        )
        messages = build_agent_message_sequential_latent_mas(
            args.role, item["question"], method="latent_mas", args=shim
        )
        text = (
            tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if tok.chat_template
            else item["question"]
        )
        ids = trim(
            tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0],
            args.prompt_limit,
        ).to(model.device)
        out = model(
            input_ids=ids[None],
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        for p in positions(
            out.hidden_states[-1].shape[1], args.max_states_per_question
        ):
            states.append(
                State(
                    out.hidden_states[-1][0, p].float().cpu(),
                    item_id,
                    "prompt",
                    p,
                    len(ids),
                    0,
                )
            )
        past, logits = out.past_key_values, out.logits[:, -1]
        reply = []
        for _ in range(args.max_reply_tokens):
            token = sample_token(
                logits, args.temperature, args.top_p
            )
            if eos is not None and int(token.item()) == eos:
                break
            step = model(
                input_ids=token[:, None],
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            reply.append(step.hidden_states[-1][0, -1].float().cpu())
            past, logits = step.past_key_values, step.logits[:, -1]
        for p in positions(len(reply), args.max_states_per_question):
            states.append(State(reply[p], item_id, "reply", p, len(ids), len(reply)))
        logger.info(
            "State collection: question %d/%d completed (prompt=%d, reply=%d, %.1fs).",
            item_id + 1,
            len(items),
            len(ids),
            len(reply),
            time.time() - started,
        )
    return states


def get_weights(source_model, target_model):
    """Return source output weights and target input weights for the operator."""
    source_output_embeddings = source_model.get_output_embeddings()
    if source_output_embeddings is None:
        raise ValueError("Source model does not expose output embeddings.")
    output_bias = getattr(source_output_embeddings, "bias", None)
    return (
        source_output_embeddings.weight.detach().float(),
        target_model.get_input_embeddings().weight.detach().float(),
        None if output_bias is None else output_bias.detach().float(),
    )


def main():
    args = parse_args()
    logger = configure_logger()
    started = time.time()
    logger.info("=" * 72)
    logger.info(
        "Operator experiment started: study=%s, pair=%s, dataset=%s/%s",
        args.study,
        args.model_pair,
        args.dataset,
        args.split,
    )
    logger.info(
        "Configuration: %s", json.dumps(vars(args), ensure_ascii=False, sort_keys=True)
    )

    # Set random seeds
    random.seed(args.probe_seed)
    np.random.seed(args.probe_seed)
    torch.manual_seed(args.probe_seed)
    source_model_name, target_model_name = resolve_model_names(args)
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    logger.info(
        "Loading tokenizers: source=%s; target=%s",
        source_model_name,
        target_model_name,
    )
    # Load tokenizers with trust_remote_code if specified
    source_tokenizer = AutoTokenizer.from_pretrained(
        source_model_name, token=False, trust_remote_code=args.trust_remote_code
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        target_model_name, token=False, trust_remote_code=args.trust_remote_code
    )
    logger.info("Loading source model on %s (%s).", args.device, dtype)
    source_model = (
        AutoModelForCausalLM.from_pretrained(
            source_model_name,
            dtype=dtype,
            token=False,
            trust_remote_code=args.trust_remote_code,
        )
        .to(args.device)
        .eval()
    )
    logger.info("Loading target model on %s (%s).", args.device, dtype)
    target_model = (
        AutoModelForCausalLM.from_pretrained(
            target_model_name,
            dtype=dtype,
            token=False,
            trust_remote_code=args.trust_remote_code,
        )
        .to(args.device)
        .eval()
    )

    # Check tokenizer and model compatibility (the tokenizer vocab and special tokens must match, and the hidden sizes must match)
    check_compatibility(
        source_tokenizer, target_tokenizer, source_model, target_model, args
    )
    logger.info("Tokenizer compatibility check passed.")
    split = args.split
    logger.info(
        "Loading %s split %s; sampling at most %d questions with seed %d.",
        args.dataset,
        split,
        args.max_questions,
        args.probe_seed,
    )
    items = list(load_data(args.dataset, split))
    random.Random(args.probe_seed).shuffle(items)
    items = items[: args.max_questions]
    states = collect(source_model, source_tokenizer, items, args, logger)
    logger.info(
        "State collection complete: %d states from %d questions.",
        len(states),
        len(items),
    )
    wo, wi, bias = get_weights(source_model, target_model)
    wo, wi = wo.to(args.device), wi.to(args.device)
    bias = None if bias is None else bias.float().to(args.device)
    audit_p99 = common.audit(states, wo, wi, bias, args)
    logger.info("Float64 exact-F audit passed: relative-L2 p99=%.3e.", audit_p99)
    logger.info(
        "Building main ORF kernel (m=%d, tau=%s, seed=%d).",
        args.kernel_features,
        args.kernel_temperature,
        args.kernel_seed,
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
    write_json(
        RESULT / "manifests" / f"{args.study}_{args.model_pair}_{args.dataset}.json",
        {
            "args": vars(args),
            "source": source_model_name,
            "target": target_model_name,
            "git_commit": commit(),
            "questions": len(items),
            "states": len(states),
            "timestamp": time.time(),
        },
    )
    studies = [args.study] if args.study != "all" else ["s0", "s1", "s2", "s3", "s4"]
    for index, study in enumerate(studies, start=1):
        logger.info("Stage %d/%d (%s) started.", index, len(studies), study.upper())
        stage_started = time.time()
        if study == "s0":
            rows = s0.run(states, wo, wi, args, logger)
            common.write_rows(rows, f"s0_states_{args.model_pair}_{args.dataset}")
            common.write_rows(
                s0.hidden_norm_summary(rows),
                f"s0_hidden_norm_summary_{args.model_pair}_{args.dataset}",
            )
            common.write_rows(
                s0.weight_norm_summary(wo, wi),
                f"s0_embedding_norm_summary_{args.model_pair}_{args.dataset}",
            )
        elif study == "s1":
            rows, single = s1.run(states, wo, wi, bias, kernel, args, logger)
            common.write_rows(rows, f"s1_mapping_{args.model_pair}_{args.dataset}")
            common.write_rows(
                single, f"s1_single_kernel_{args.model_pair}_{args.dataset}"
            )
            common.write_rows(
                common.stat_rows(rows, "f_rel_l2", args),
                f"s1_summary_{args.model_pair}_{args.dataset}",
            )
            common.histogram_ecdf(rows, "f_rel_l2", "s1")
            s1.plot(rows)
            if args.run_s1_performance:
                common.write_rows(
                    s1.performance(states, wo, wi, bias, kernel, args, logger),
                    "s1_performance",
                )
        elif study == "s2":
            rows, single = s2.run(states, wo, wi, bias, kernel, args, logger)
            common.write_rows(rows, f"s2_mapping_{args.model_pair}_{args.dataset}")
            common.write_rows(
                single, f"s2_single_kernel_{args.model_pair}_{args.dataset}"
            )
            common.write_rows(
                common.stat_rows(rows, "f_rel_l2", args),
                f"s2_summary_{args.model_pair}_{args.dataset}",
            )
            common.histogram_ecdf(rows, "f_rel_l2", "s2")
            s2.plot(rows)
            if args.run_s2_calibration:
                if not (args.dataset == "arc_easy" and args.split == "train"):
                    raise ValueError(
                        "S2 calibration is prescribed only for ARC-Easy train"
                    )
                logger.info(
                    "S2 calibration grid started (ORF/iid, 5 m values, 3 tau values, 5 seeds)."
                )
                calibration = s2.calibration(states, wo, wi, bias, args, logger)
                common.write_rows(calibration, "s2_calibration")
                logger.info("S2 calibration completed: %d rows.", len(calibration))
        elif study == "s3":
            rows = s3.run(states, wo, wi, bias, args, logger)
            common.write_rows(rows, f"s3_variance_{args.model_pair}_{args.dataset}")
            common.write_rows(
                common.stat_rows(
                    [x for x in rows if x["kind"] == "F"],
                    "variance",
                    args,
                    extra=("m", "tau", "kind"),
                ),
                f"s3_summary_{args.model_pair}_{args.dataset}",
            )
            s3.plot_s3(rows)
            s3.plot_forest(rows)
        else:
            rows = s4.run(states, wo, wi, bias, kernel, args, logger)
            common.write_rows(rows, f"s4_embeddings_{args.model_pair}_{args.dataset}")
        logger.info(
            "Stage %s completed in %.1fs.", study.upper(), time.time() - stage_started
        )
    logger.info(
        "Operator experiment completed successfully in %.1fs. State log: %s",
        time.time() - started,
        RESULT / "exp_state.txt",
    )


if __name__ == "__main__":
    main()

