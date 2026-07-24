#!/usr/bin/env python
"""Reproducible implementation of S0--S4 from ``docs/plan_v2.md``.

The program intentionally keeps exact softmax evaluation separate from the
deployable ORF approximation.  It never truncates the vocabulary for exact
metrics.  Results are written only below this experiment directory.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from alignment import build_kernel_state, build_linear_state, positive_features
from data import (load_arc_challenge, load_arc_easy, load_gsm8k, load_gpqa_diamond,
                  load_mbppplus, load_medqa)
from prompts import build_agent_message_sequential_latent_mas

HERE = Path(__file__).resolve().parent
RESULT = HERE / "result"


@dataclass
class State:
    vector: torch.Tensor
    item_id: int
    source: str
    position: int
    prompt_length: int
    reply_length: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=["s0", "s1", "s2", "s3", "s4", "all"], default="s1")
    p.add_argument("--model_pair", choices=["x1", "x2"], default="x1")
    p.add_argument("--source_model", default=None)
    p.add_argument("--target_model", default=None)
    p.add_argument("--dataset", default="arc_easy", choices=["arc_easy", "arc_challenge", "gsm8k", "medqa", "mbppplus", "gpqa"])
    p.add_argument("--split", default="test")
    p.add_argument("--max_questions", type=int, default=50)
    p.add_argument("--max_states_per_question", type=int, default=50)
    p.add_argument("--max_reply_tokens", type=int, default=512)
    p.add_argument("--prompt_limit", type=int, default=512)
    p.add_argument("--role", default="planner", choices=["planner", "critic", "refiner", "judger"])
    p.add_argument("--m", type=int, default=2048)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--orf_seed", type=int, default=101)
    p.add_argument("--kernel_chunk_size", type=int, default=4096)
    p.add_argument("--probe_seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--skip_float64_audit", action="store_true")
    p.add_argument("--s3_replicates", type=int, default=32)
    p.add_argument("--s3_max_questions", type=int, default=50)
    p.add_argument("--run_s2_calibration", action="store_true", help="Run the prescribed ORF/iid m,tau,seed grid (ARC-Easy train only).")
    return p.parse_args()


def pair(args: argparse.Namespace) -> tuple[str, str]:
    defaults = {
        "x1": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
        "x2": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    }
    a, b = defaults[args.model_pair]
    return args.source_model or a, args.target_model or b


def json_default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_metrics(rows: list[dict[str, Any]], stem: str) -> Path:
    """Write parquet without making pandas part of the experiment API."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = RESULT / "metrics" / f"{stem}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return path


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def compat(a_tok: Any, b_tok: Any, a_model: Any, b_model: Any, args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    if len(a_tok) != len(b_tok): failures.append("vocab_size")
    if a_tok.get_vocab() != b_tok.get_vocab(): failures.append("token_to_id")
    special_a = {k: getattr(a_tok, k, None) for k in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")}
    special_b = {k: getattr(b_tok, k, None) for k in special_a}
    if special_a != special_b: failures.append("special_token_ids")
    info = {"model_pair": args.model_pair, "source_model": pair(args)[0], "target_model": pair(args)[1],
            "vocab_size_a": len(a_tok), "vocab_size_b": len(b_tok), "special_a": special_a, "special_b": special_b,
            "hidden_a": a_model.config.hidden_size, "hidden_b": b_model.config.hidden_size,
            "tie_a": bool(getattr(a_model.config, "tie_word_embeddings", False)),
            "tie_b": bool(getattr(b_model.config, "tie_word_embeddings", False)), "failures": failures,
            "torch": torch.__version__}
    write_json(RESULT / "manifests" / "compatibility.json", info)
    if failures:
        raise RuntimeError("Stopped: incompatible tokenizer(s): " + ", ".join(failures))
    return info


def load_dataset(name: str, split: str) -> Iterable[dict[str, Any]]:
    loaders = {"arc_easy": load_arc_easy, "arc_challenge": load_arc_challenge, "gsm8k": load_gsm8k,
               "medqa": load_medqa, "mbppplus": load_mbppplus, "gpqa": load_gpqa_diamond}
    return loaders[name](split=split)


def choose_positions(n: int, limit: int) -> list[int]:
    if n <= limit: return list(range(n))
    return sorted(set(np.linspace(0, n - 1, limit, dtype=int).tolist() + [n - 1]))


def trim_prompt(ids: torch.Tensor, limit: int) -> torch.Tensor:
    if ids.numel() <= limit: return ids
    return torch.cat((ids[:480], ids[-32:])) if limit == 512 else torch.cat((ids[: limit - 32], ids[-32:]))


@torch.inference_mode()
def collect_states(model: Any, tokenizer: Any, items: list[dict[str, Any]], args: argparse.Namespace) -> list[State]:
    states: list[State] = []
    eos = tokenizer.eos_token_id
    for item_id, item in enumerate(items):
        shim = argparse.Namespace(model_name=pair(args)[0], task=args.dataset)
        messages = build_agent_message_sequential_latent_mas(args.role, item["question"], method="latent_mas", args=shim)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if tokenizer.chat_template else item["question"]
        ids = trim_prompt(tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0], args.prompt_limit).to(model.device)
        out = model(input_ids=ids.unsqueeze(0), output_hidden_states=True, use_cache=True, return_dict=True)
        hidden = out.hidden_states[-1][0].float()
        for pos in choose_positions(hidden.shape[0], args.max_states_per_question):
            states.append(State(hidden[pos].cpu(), item_id, "prompt", pos, int(ids.numel()), 0))
        past, next_logits = out.past_key_values, out.logits[:, -1, :]
        reply: list[torch.Tensor] = []
        for _ in range(args.max_reply_tokens):
            token = next_logits.argmax(dim=-1)
            if eos is not None and int(token.item()) == eos: break
            step = model(input_ids=token[:, None], past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
            reply.append(step.hidden_states[-1][0, -1].float().cpu())
            past, next_logits = step.past_key_values, step.logits[:, -1, :]
        for pos in choose_positions(len(reply), args.max_states_per_question):
            states.append(State(reply[pos], item_id, "reply", pos, int(ids.numel()), len(reply)))
    return states


def weights(source: Any, target: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    out = source.get_output_embeddings(); inp = target.get_input_embeddings()
    return out.weight.detach().float(), inp.weight.detach().float(), getattr(out, "bias", None)


def exact(q: torch.Tensor, w_out: torch.Tensor, w_in: torch.Tensor, bias: torch.Tensor | None, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.to(device=w_out.device, dtype=w_out.dtype)
    logits = w_out @ (q / tau)
    if bias is not None: logits = logits + bias.to(device=w_out.device, dtype=w_out.dtype)
    p = torch.softmax(logits, dim=0)
    return p @ w_in, p


def mapping(q: torch.Tensor, kernel: Any) -> tuple[torch.Tensor, bool]:
    u = positive_features(q.to(kernel.omega.device).float()[None] / kernel.temperature, kernel.omega, stabilize=True)[0]
    denom = u @ kernel.denominator
    valid = bool(torch.isfinite(denom) and denom > torch.finfo(denom.dtype).eps)
    return (u @ kernel.numerator.T) / denom if valid else torch.full((kernel.numerator.shape[0],), float("nan"), device=u.device), valid


def audit(states: list[State], w_out: torch.Tensor, w_in: torch.Tensor, bias: torch.Tensor | None, args: argparse.Namespace) -> None:
    if args.skip_float64_audit: return
    sampled = states[: min(256, len(states))]
    errors = []
    for st in sampled:
        f32, _ = exact(st.vector, w_out, w_in, bias, args.tau)
        f64, _ = exact(st.vector.double(), w_out.double(), w_in.double(), None if bias is None else bias.double(), args.tau)
        errors.append(float((f32.double() - f64).norm() / f64.norm().clamp_min(1e-12)))
    if errors and np.quantile(errors, .99) > 1e-4:
        raise RuntimeError(f"Stopped: float64 audit p99={np.quantile(errors, .99):.3e} > 1e-4")


def rows_s0(states: list[State], w_out: torch.Tensor, w_in: torch.Tensor, args: argparse.Namespace) -> list[dict[str, Any]]:
    key_norm = w_out.norm(dim=1); value_norm = w_in.norm(dim=1)
    rows = []
    for st in states:
        q = st.vector.float().to(w_out.device) / args.tau
        # Computing all pair norms is intentionally exact but summary-sized.
        combined = (w_out + q).norm(dim=1)
        rows.append({**asdict(st) | {"vector": None}, "hidden_norm": float(st.vector.norm()), "q_norm": float(q.norm()),
                     "key_norm_p50": float(key_norm.median()), "value_norm_p50": float(value_norm.median()),
                     "w_plus_q_p50": float(combined.median()), "w_plus_q_p99": float(torch.quantile(combined, .99))})
    return rows


def rank_indices(p: torch.Tensor, item_id: int, seed: int) -> list[tuple[str, int]]:
    order = torch.argsort(p, descending=True).cpu().tolist(); rng = random.Random(seed + item_id)
    answer = [("rank_1", order[0])]
    for name, lo, hi in (("rank_2_10", 1, 10), ("rank_11_100", 10, 100), ("rank_101_1000", 100, 1000), ("rank_gt_1000", 1000, len(order))):
        pool = order[lo:min(hi, len(order))]; answer += [(name, x) for x in rng.sample(pool, min(3, len(pool)))]
    return answer


def rows_s1_s2(states: list[State], w_out: torch.Tensor, w_in: torch.Tensor, bias: torch.Tensor | None, kernel: Any, args: argparse.Namespace, include_s2: bool) -> list[dict[str, Any]]:
    rows = []
    for st in states:
        f, p = exact(st.vector, w_out, w_in, bias, args.tau); approx, valid = mapping(st.vector, kernel)
        rel = (approx - f).norm() / f.norm().clamp_min(1e-8)
        row = {"item_id": st.item_id, "source": st.source, "position": st.position, "prompt_length": st.prompt_length, "reply_length": st.reply_length,
               "f_rel_l2": float(rel), "f_cosine": float(torch.nn.functional.cosine_similarity(approx[None], f[None])), "denom_valid": valid,
               "nan_inf": bool(not torch.isfinite(approx).all())}
        if include_s2:
            # p_hat is recovered from the same unnormalized RF kernel values.
            phi_q = positive_features((st.vector.to(w_out.device) / args.tau)[None], kernel.omega, stabilize=False)[0]
            phi_w = positive_features(w_out, kernel.omega)
            khat = phi_w @ phi_q
            alpha = torch.ones_like(khat) if bias is None else torch.exp((bias.float() - bias.float().max()).to(khat.device))
            phat = (alpha * khat).clamp_min(0); phat = phat / phat.sum().clamp_min(1e-30)
            tv = .5 * (p - phat).abs().sum(); js = .5 * ((p * (p.clamp_min(1e-30).log() - ((p + phat).mul(.5)).clamp_min(1e-30).log())).sum() + (phat * (phat.clamp_min(1e-30).log() - ((p + phat).mul(.5)).clamp_min(1e-30).log())).sum())
            row.update({"tv": float(tv), "js": float(js), "kl_p_phat": float((p * (p.clamp_min(1e-30).log() - phat.clamp_min(1e-30).log())).sum()),
                        "top1_agree": bool(p.argmax() == phat.argmax()), "top10_overlap": len(set(torch.topk(p, 10).indices.tolist()) & set(torch.topk(phat, 10).indices.tolist())) / 10})
        rows.append(row)
        for band, idx in rank_indices(p, st.item_id * 1000 + st.position, args.probe_seed):
            k = torch.exp(torch.dot(w_out[idx], st.vector.to(w_out.device) / args.tau)); khat = torch.dot(positive_features(w_out[idx:idx+1], kernel.omega)[0], positive_features((st.vector.to(w_out.device) / args.tau)[None], kernel.omega)[0])
            rows.append({"item_id": st.item_id, "source": st.source, "position": st.position, "metric": "single_kernel", "rank_band": band,
                         "kernel_abs_error": float((khat-k).abs()), "kernel_relative_error": float((khat-k).abs()/(k+1e-8)), "kernel_log_error": float((khat+1e-8).log().sub((k+1e-8).log()).abs())})
    return rows


def run_s3(states: list[State], w_out: torch.Tensor, w_in: torch.Tensor, bias: torch.Tensor | None, args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [s for s in states if s.source == "prompt" and s.position >= 0][:args.s3_max_questions]
    selected += [s for s in states if s.source == "reply"][:args.s3_max_questions * 16]
    rows = []
    for m in (512, 1024, 2048):
        for tau in np.arange(.5, 2.01, .1):
            vals: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(selected))}
            for seed in range(1001, 1001 + args.s3_replicates):
                state = build_kernel_state(w_out, w_in, bias, feature_count=m, temperature=float(tau), seed=seed, chunk_size=args.kernel_chunk_size)
                for i, s in enumerate(selected): vals[i].append(mapping(s.vector, state)[0].cpu())
            for i, s in enumerate(selected):
                stack = torch.stack(vals[i]); f, _ = exact(s.vector, w_out, w_in, bias, float(tau)); mean = stack.mean(0)
                rows.append({"item_id": s.item_id, "source": s.source, "m": m, "tau": float(tau), "f_variance": float(stack.var(0, unbiased=True).mean()),
                             "f_std": float(stack.var(0, unbiased=True).mean().sqrt()), "bias2": float((mean-f.cpu()).square().mean()), "mse": float((mean-f.cpu()).square().mean() + stack.var(0, unbiased=True).mean())})
    return rows


def plot(rows: list[dict[str, Any]], study: str) -> None:
    numeric = [r for r in rows if "f_rel_l2" in r]
    if study in ("s1", "s2") and numeric:
        plt.figure(figsize=(6, 4));
        for source in ("prompt", "reply"):
            x = sorted(r["f_rel_l2"] for r in numeric if r["source"] == source)
            if x: plt.plot(x, np.arange(1, len(x)+1)/len(x), label=source)
        plt.xlabel("relative L2 error"); plt.ylabel("ECDF"); plt.legend(); plt.tight_layout()
        (RESULT / "figures").mkdir(parents=True, exist_ok=True); plt.savefig(RESULT / "figures" / f"{study}_error_ecdf.pdf"); plt.close()


def main() -> None:
    args = parse_args(); random.seed(args.probe_seed); np.random.seed(args.probe_seed); torch.manual_seed(args.probe_seed)
    RESULT.mkdir(exist_ok=True); (RESULT / "manifests").mkdir(exist_ok=True)
    source_name, target_name = pair(args); dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    a_tok = AutoTokenizer.from_pretrained(source_name, token=False, trust_remote_code=args.trust_remote_code)
    b_tok = AutoTokenizer.from_pretrained(target_name, token=False, trust_remote_code=args.trust_remote_code)
    source = AutoModelForCausalLM.from_pretrained(source_name, dtype=dtype, token=False, trust_remote_code=args.trust_remote_code).to(args.device).eval()
    target = AutoModelForCausalLM.from_pretrained(target_name, dtype=dtype, token=False, trust_remote_code=args.trust_remote_code).to(args.device).eval()
    compat(a_tok, b_tok, source, target, args)
    questions = list(load_dataset(args.dataset, "train" if args.dataset == "arc_easy" and args.split == "train" else args.split))
    random.Random(args.probe_seed).shuffle(questions); questions = questions[:args.max_questions]
    states = collect_states(source, a_tok, questions, args)
    w_out, w_in, bias = weights(source, target); w_out, w_in = w_out.to(args.device), w_in.to(args.device)
    bias = None if bias is None else bias.detach().float().to(args.device)
    audit(states, w_out, w_in, bias, args)
    kernel = build_kernel_state(w_out, w_in, bias, feature_count=args.m, temperature=args.tau, seed=args.orf_seed, chunk_size=args.kernel_chunk_size)
    manifest = {"args": vars(args), "git_commit": git_commit(), "source": source_name, "target": target_name, "states": len(states), "questions": len(questions), "timestamp": time.time()}
    write_json(RESULT / "manifests" / f"{args.study}_{args.dataset}.json", manifest)
    studies = [args.study] if args.study != "all" else ["s0", "s1", "s2", "s3", "s4"]
    for study in studies:
        if study == "s0": rows = rows_s0(states, w_out, w_in, args)
        elif study == "s1": rows = rows_s1_s2(states, w_out, w_in, bias, kernel, args, False)
        elif study == "s2": rows = rows_s1_s2(states, w_out, w_in, bias, kernel, args, True)
        elif study == "s3": rows = run_s3(states, w_out, w_in, bias, args)
        else:
            # S4 stores all four mappings for a shared-PCA downstream fit.
            linear = build_linear_state(w_out, w_in, ridge=1e-5)
            rows = []
            for s in states:
                f, _ = exact(s.vector, w_out, w_in, bias, args.tau); k, _ = mapping(s.vector, kernel)
                ident = s.vector.to(args.device); ident = ident * (w_in.norm(dim=1).mean()/ident.norm().clamp_min(1e-6)); lin = (s.vector.to(args.device) @ linear.matrix); lin = lin * (linear.target_norm/lin.norm().clamp_min(1e-6))
                for method, vec in (("exact", f), ("identical", ident), ("linear", lin), ("kernel", k)):
                    rows.append({"item_id": s.item_id, "source": s.source, "position": s.position, "method": method, "embedding": vec.cpu().tolist()})
        write_metrics(rows, f"{study}_{args.model_pair}_{args.dataset}")
        plot(rows, study)


if __name__ == "__main__": main()
