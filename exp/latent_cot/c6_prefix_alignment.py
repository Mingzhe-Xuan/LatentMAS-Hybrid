#!/usr/bin/env python
"""C6 text-prefix one-step alignment agreement and cached-K entropy analysis.

The experiment is deliberately cache-led.  It never recollects C0 trajectories:
it discovers validated C0 ``.pt``/manifest pairs, replays the cached greedy text
prefix to recover its KV state, branches one transformer step through linear,
kernel, soft, and hard-text feedback, and compares the resulting token
distributions.  It also redraws entropy curves for every compatible cached K.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory import build_alignment_states, load_model
from alignment import apply_alignment
from utils import auto_device


DATASETS = ("aime2025", "mbppplus")
METHODS = ("linear", "kernel", "soft", "text")
PAIRS = (
    ("linear", "text"),
    ("kernel", "text"),
    ("linear", "soft"),
    ("kernel", "soft"),
)
SCHEMA_VERSION = 1
DEFAULT_TRAJECTORY_DIR = ROOT / "exp" / "cache" / "trajectories"
DEFAULT_OUTPUT_ROOT = ROOT / "exp_result" / "latent_cot" / "runs"

COLORS = {
    "linear": "#0072B2",
    "kernel": "#009E73",
    "soft": "#CC79A7",
    "text": "#D55E00",
    "linear|text": "#0072B2",
    "kernel|text": "#009E73",
    "linear|soft": "#56B4E9",
    "kernel|soft": "#E69F00",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--comparison_steps",
        type=int,
        default=None,
        help="K used for prefix branching; default is the largest compatible cached K.",
    )
    parser.add_argument("--positions_per_trajectory", type=int, default=100)
    parser.add_argument("--position_seed", type=int, default=42)
    parser.add_argument("--token_sample_seed", type=int, default=42)
    parser.add_argument("--entropy_chunk_size", type=int, default=8)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--prefix_cosine_tolerance", type=float, default=0.999)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_sha256", action="store_true")
    parser.add_argument("--skip_prefix_comparison", action="store_true")
    parser.add_argument("--skip_entropy", action="store_true")
    args = parser.parse_args(argv)
    if args.positions_per_trajectory < 1:
        parser.error("--positions_per_trajectory must be positive")
    if args.comparison_steps is not None and args.comparison_steps < 1:
        parser.error("--comparison_steps must be positive")
    if args.entropy_chunk_size < 1 or args.bootstrap_replicates < 1:
        parser.error("entropy chunk size and bootstrap replicates must be positive")
    if not -1.0 <= args.prefix_cosine_tolerance <= 1.0:
        parser.error("--prefix_cosine_tolerance must lie in [-1, 1]")
    if args.skip_prefix_comparison and args.skip_entropy:
        parser.error("both analyses cannot be skipped")
    args.device = auto_device(args.device)
    return args


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
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


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def parse_trajectory_name(path):
    name = Path(path).name
    dataset_match = re.search(r"__ds=v-(.*?)__sp=v-", name)
    split_match = re.search(r"__sp=v-(.*?)__m=v-", name)
    model_match = re.search(r"__m=v-(.*?)__q=", name)
    question_match = re.search(r"__q=(\d+)__", name)
    steps_match = re.search(r"__k=(\d+)(?:__|\.)", name)
    alignments_match = re.search(r"__a=v-(.*?)__km=", name)
    if not all((dataset_match, split_match, model_match, question_match, steps_match)):
        return None
    encoded_model = model_match.group(1)
    model = encoded_model.replace("~2F", "/").replace("~U", "")
    alignments = (
        tuple(alignments_match.group(1).split("-")) if alignments_match else tuple()
    )
    return {
        "path": Path(path),
        "dataset": dataset_match.group(1),
        "split": split_match.group(1),
        "model_name": model,
        "questions": int(question_match.group(1)),
        "latent_steps": int(steps_match.group(1)),
        "alignments": alignments,
    }


def discover_trajectories(directory, datasets):
    required = set(METHODS)
    discovered = defaultdict(dict)
    for path in sorted(Path(directory).glob("c0__*.pt")):
        info = parse_trajectory_name(path)
        if info is None or info["dataset"] not in datasets:
            continue
        if not required.issubset(set(info["alignments"])):
            continue
        key = info["latent_steps"]
        if key in discovered[info["dataset"]]:
            other = discovered[info["dataset"]][key]["path"]
            raise RuntimeError(
                f"Ambiguous compatible C0 trajectories for {info['dataset']} K={key}: "
                f"{other} and {path}"
            )
        manifest = path.with_name(path.name[:-3] + ".manifest.json")
        if not manifest.exists():
            raise FileNotFoundError(f"Trajectory manifest is missing: {manifest}")
        info["manifest_path"] = manifest
        discovered[info["dataset"]][key] = info
    missing = [dataset for dataset in datasets if not discovered.get(dataset)]
    if missing:
        raise FileNotFoundError(
            "No C0 trajectory containing linear/kernel/soft/text for: "
            + ", ".join(missing)
        )
    return {dataset: dict(sorted(entries.items())) for dataset, entries in discovered.items()}


def load_validated_trajectory(info, validate_sha256=True):
    manifest = json.loads(info["manifest_path"].read_text(encoding="utf-8"))
    if validate_sha256:
        expected = manifest.get("trajectory_sha256")
        if not expected:
            raise RuntimeError(f"Manifest lacks trajectory_sha256: {info['manifest_path']}")
        actual = file_sha256(info["path"])
        if actual != expected:
            raise RuntimeError(
                f"Trajectory SHA256 mismatch for {info['path']}: {actual} != {expected}"
            )
    trajectory = torch.load(info["path"], map_location="cpu", weights_only=True)
    if trajectory.get("trajectory_is_complete") is not True:
        raise RuntimeError(f"Trajectory is not complete: {info['path']}")
    records = trajectory.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"Trajectory has no record list: {info['path']}")
    expected_pairs = {
        (record.get("item_id"), record.get("alignment")) for record in records
    }
    if len(expected_pairs) != len(records):
        raise RuntimeError(f"Trajectory has duplicate item/alignment records: {info['path']}")
    for record in records:
        hidden = record.get("hidden_states")
        if (
            not torch.is_tensor(hidden)
            or hidden.ndim != 2
            or int(hidden.shape[0]) != info["latent_steps"]
            or record.get("rollout_complete") is not True
        ):
            raise RuntimeError(
                f"Incomplete hidden trajectory for item={record.get('item_id')} "
                f"alignment={record.get('alignment')} in {info['path']}"
            )
    return trajectory, manifest


def alignment_config(manifest):
    identity = manifest.get("cache_identity", {})
    config = identity.get("alignment_config", {})
    required = {
        "linear_ridge",
        "kernel_features",
        "kernel_temperature",
        "kernel_seed",
        "kernel_chunk_size",
        "soft_chunk_size",
    }
    absent = sorted(required - set(config))
    if absent:
        raise RuntimeError(f"Trajectory manifest lacks alignment settings: {absent}")
    return config


def stable_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_prefix_positions(length, count, seed, dataset, item_id):
    if length < 1:
        return []
    population = list(range(1, length + 1))
    rng = random.Random(stable_seed(seed, dataset, item_id, length))
    return sorted(rng.sample(population, min(count, length)))


def deterministic_sample(probabilities, seed, dataset, item_id, prefix_length, method):
    integer = stable_seed(seed, dataset, item_id, prefix_length, method)
    uniform = (integer + 0.5) / float(2**64)
    cumulative = probabilities.cumsum(dim=-1)
    value = torch.tensor(uniform, device=probabilities.device, dtype=probabilities.dtype)
    token = torch.searchsorted(cumulative, value).clamp_max(probabilities.shape[-1] - 1)
    return int(token.item())


def _past_length(past_key_values):
    if not past_key_values:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    return int(past_key_values[0][0].shape[-2])


def expand_legacy_cache(past_key_values, batch_size):
    """Share an immutable legacy tuple cache across one-step branches."""
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "C6 currently requires the legacy tuple KV cache used by C0; "
            f"received {type(past_key_values).__name__}."
        )
    expanded = []
    for layer in past_key_values:
        expanded.append(
            tuple(
                tensor.expand(batch_size, *tensor.shape[1:])
                if torch.is_tensor(tensor)
                else tensor
                for tensor in layer
            )
        )
    return tuple(expanded)


def pair_metrics(logits_by_method, sampled_tokens, top5, top10):
    log_probabilities = {
        method: torch.log_softmax(logits.float(), dim=-1)
        for method, logits in logits_by_method.items()
    }
    rows = []
    for left, right in PAIRS:
        left5, right5 = set(top5[left]), set(top5[right])
        left10, right10 = set(top10[left]), set(top10[right])
        lp, lq = log_probabilities[left], log_probabilities[right]
        p, q = lp.exp(), lq.exp()
        kl_lr = float((p * (lp - lq)).sum().clamp_min(0).item())
        kl_rl = float((q * (lq - lp)).sum().clamp_min(0).item())
        rows.append(
            {
                "pair": f"{left}|{right}",
                "left_method": left,
                "right_method": right,
                "sampled_token_equal": sampled_tokens[left] == sampled_tokens[right],
                "top5_overlap_count": len(left5 & right5),
                "top5_overlap_fraction": len(left5 & right5) / 5.0,
                "top5_any_overlap": bool(left5 & right5),
                "top10_overlap_count": len(left10 & right10),
                "top10_overlap_fraction": len(left10 & right10) / 10.0,
                "kl_left_right_nats": kl_lr,
                "kl_right_left_nats": kl_rl,
                "symmetric_kl_nats": 0.5 * (kl_lr + kl_rl),
            }
        )
    return rows


def _embedding_dtype(wrapper):
    return wrapper.model.get_input_embeddings().weight.dtype


@torch.inference_mode()
def collect_prefix_comparisons(
    trajectory,
    dataset,
    latent_steps,
    wrapper,
    states,
    args,
    checkpoint=None,
):
    records = [
        record for record in trajectory["records"] if record.get("alignment") == "text"
    ]
    if not records:
        raise RuntimeError(f"No text records in {dataset} K={latent_steps}")
    input_embedding = wrapper.model.get_input_embeddings()
    output_head = wrapper.model.get_output_embeddings()
    rows = []
    dtype = _embedding_dtype(wrapper)
    for record_index, record in enumerate(records, start=1):
        item_id = int(record["item_id"])
        generated = [int(token) for token in record.get("generated_token_ids", [])]
        if len(generated) != latent_steps:
            raise RuntimeError(
                f"Text record item={item_id} has {len(generated)} generated tokens, "
                f"expected {latent_steps}."
            )
        positions = sample_prefix_positions(
            latent_steps,
            args.positions_per_trajectory,
            args.position_seed,
            dataset,
            item_id,
        )
        selected = set(positions)
        cached_sources = torch.stack(
            [
                record["hidden_states"][position]
                if position < latent_steps
                else record["final_hidden"]
                for position in positions
            ],
            dim=0,
        ).to(wrapper.device, dtype=torch.float32)
        aligned_by_method = {
            method: apply_alignment(cached_sources, states[method])
            for method in ("linear", "kernel", "soft")
        }
        text_tokens = output_head(cached_sources.to(dtype=dtype)).argmax(dim=-1)
        aligned_by_method["text"] = input_embedding(text_tokens).float()
        position_index = {position: index for index, position in enumerate(positions)}
        input_ids = torch.tensor(
            [record["prompt_token_ids"]], dtype=torch.long, device=wrapper.device
        )
        attention_mask = torch.tensor(
            [record["prompt_attention_mask"]], dtype=torch.long, device=wrapper.device
        )
        output = wrapper.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values
        last_hidden = output.hidden_states[-1][:, -1, :]
        for prefix_length in range(1, max(positions) + 1):
            next_token = torch.tensor(
                [[generated[prefix_length - 1]]], dtype=torch.long, device=wrapper.device
            )
            output = wrapper.model(
                input_ids=next_token,
                attention_mask=torch.ones(
                    (1, _past_length(past) + 1),
                    dtype=torch.long,
                    device=wrapper.device,
                ),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values
            last_hidden = output.hidden_states[-1][:, -1, :]
            if prefix_length not in selected:
                continue

            cached_hidden = cached_sources[position_index[prefix_length]].cpu()
            replay = last_hidden[0].detach().float().cpu()
            cosine = float(
                torch.nn.functional.cosine_similarity(replay, cached_hidden, dim=0)
            )
            max_abs = float((replay - cached_hidden).abs().max())
            if cosine < args.prefix_cosine_tolerance:
                raise RuntimeError(
                    f"Text-prefix replay mismatch: dataset={dataset}, item={item_id}, "
                    f"t={prefix_length}, cosine={cosine:.8f}."
                )

            branch_inputs = torch.stack(
                [
                    aligned_by_method[method][position_index[prefix_length]]
                    for method in METHODS
                ],
                dim=0,
            )
            branch_past = expand_legacy_cache(past, len(METHODS))
            branch_output = wrapper.model(
                inputs_embeds=branch_inputs.to(dtype=dtype).unsqueeze(1),
                attention_mask=torch.ones(
                    (len(METHODS), _past_length(past) + 1),
                    dtype=torch.long,
                    device=wrapper.device,
                ),
                past_key_values=branch_past,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            branch_hidden = branch_output.hidden_states[-1][:, -1, :]
            branch_logits = output_head(branch_hidden).float()
            logits_by_method = {
                method: branch_logits[index] for index, method in enumerate(METHODS)
            }
            probabilities = {
                method: torch.softmax(logits, dim=-1)
                for method, logits in logits_by_method.items()
            }
            sampled_tokens = {
                method: deterministic_sample(
                    probabilities[method],
                    args.token_sample_seed,
                    dataset,
                    item_id,
                    prefix_length,
                    method,
                )
                for method in METHODS
            }
            top5 = {
                method: torch.topk(logits, 5).indices.detach().cpu().tolist()
                for method, logits in logits_by_method.items()
            }
            top10 = {
                method: torch.topk(logits, 10).indices.detach().cpu().tolist()
                for method, logits in logits_by_method.items()
            }
            for pair_row in pair_metrics(logits_by_method, sampled_tokens, top5, top10):
                left, right = pair_row["left_method"], pair_row["right_method"]
                pair_row.update(
                    {
                        "dataset": dataset,
                        "split": "train" if dataset == "aime2025" else "test",
                        "latent_steps": latent_steps,
                        "item_id": item_id,
                        "prefix_length": prefix_length,
                        "prefix_fraction": prefix_length / latent_steps,
                        "prefix_replay_cosine": cosine,
                        "prefix_replay_max_abs": max_abs,
                        "left_sampled_token_id": sampled_tokens[left],
                        "right_sampled_token_id": sampled_tokens[right],
                        "left_sampled_token": wrapper.tokenizer.convert_ids_to_tokens(
                            sampled_tokens[left]
                        ),
                        "right_sampled_token": wrapper.tokenizer.convert_ids_to_tokens(
                            sampled_tokens[right]
                        ),
                    }
                )
                rows.append(pair_row)
        print(
            f"C6 prefix comparison: {dataset} item {record_index}/{len(records)} "
            f"(item_id={item_id}, positions={len(positions)})",
            flush=True,
        )
        if checkpoint is not None:
            checkpoint(rows)
    return rows


@torch.inference_mode()
def entropy_rows(trajectory, dataset, latent_steps, wrapper, chunk_size):
    output_head = wrapper.model.get_output_embeddings()
    output_weight = output_head.weight.detach().float()
    output_bias = getattr(output_head, "bias", None)
    output_bias = None if output_bias is None else output_bias.detach().float()
    rows = []
    records = [record for record in trajectory["records"] if record["alignment"] in METHODS]
    for index, record in enumerate(records, start=1):
        hidden = record["hidden_states"]
        values = []
        for start in range(0, len(hidden), chunk_size):
            stop = min(start + chunk_size, len(hidden))
            logits = torch.nn.functional.linear(
                hidden[start:stop].to(wrapper.device, dtype=torch.float32),
                output_weight,
                output_bias,
            )
            log_probabilities = torch.log_softmax(logits, dim=-1)
            entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
            values.extend(entropy.detach().cpu().tolist())
        rows.extend(
            {
                "dataset": dataset,
                "split": "train" if dataset == "aime2025" else "test",
                "latent_steps": latent_steps,
                "alignment": record["alignment"],
                "item_id": int(record["item_id"]),
                "step": step,
                "entropy_nats": float(value),
            }
            for step, value in enumerate(values)
        )
        if index % 25 == 0 or index == len(records):
            print(
                f"C6 entropy: {dataset} K={latent_steps} record {index}/{len(records)}",
                flush=True,
            )
    return rows


def cluster_bootstrap_interval(rows, value_key, replicates, seed):
    by_item = defaultdict(list)
    for row in rows:
        by_item[int(row["item_id"])].append(float(row[value_key]))
    item_means = np.asarray(
        [np.mean(values) for _, values in sorted(by_item.items())], dtype=np.float64
    )
    if not len(item_means):
        return None, None
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(item_means), size=(replicates, len(item_means)))
    means = item_means[sampled].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_pairs(rows, args):
    result = {
        "study": "c6",
        "sampling_unit": "one sampled prefix position within each question",
        "uncertainty_unit": "question-cluster bootstrap",
        "datasets": {},
    }
    for dataset in args.datasets:
        result["datasets"][dataset] = {}
        for left, right in PAIRS:
            pair = f"{left}|{right}"
            selected = [row for row in rows if row["dataset"] == dataset and row["pair"] == pair]
            entry = {"rows": len(selected), "questions": len({r["item_id"] for r in selected})}
            for key in (
                "sampled_token_equal",
                "top5_overlap_fraction",
                "top5_any_overlap",
                "top10_overlap_fraction",
                "kl_left_right_nats",
                "kl_right_left_nats",
                "symmetric_kl_nats",
            ):
                values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
                low, high = cluster_bootstrap_interval(
                    selected,
                    key,
                    args.bootstrap_replicates,
                    stable_seed(args.position_seed, dataset, pair, key) % (2**32),
                )
                entry[key] = {
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            result["datasets"][dataset][pair] = entry
    return result


def summarize_entropy(rows, args):
    result = {"study": "c6", "metric": "pre_unembedding_output_entropy_nats", "datasets": {}}
    for dataset in args.datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        result["datasets"][dataset] = {"latent_steps": sorted({r["latent_steps"] for r in dataset_rows})}
    return result


def plot_top10_distribution(rows, path, datasets):
    figure, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.7), sharey=True, squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        for left, right in PAIRS:
            pair = f"{left}|{right}"
            values = [
                int(row["top10_overlap_count"])
                for row in rows
                if row["dataset"] == dataset and row["pair"] == pair
            ]
            counts = np.bincount(values, minlength=11).astype(np.float64)
            probabilities = counts / counts.sum()
            axis.plot(
                np.arange(11),
                probabilities,
                marker="o",
                markersize=2.8,
                label=pair.replace("|", "–"),
                color=COLORS[pair],
            )
        axis.set_title("AIME 2025" if dataset == "aime2025" else "MBPP+")
        axis.set_xlabel("Top-10 intersection size")
        axis.set_xticks(range(0, 11, 2))
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axes[0][0].set_ylabel("Empirical probability")
    axes[0][-1].legend(frameon=False, title="Pair", ncol=2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_kl_distribution(rows, path, datasets):
    figure, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.7), sharey=True, squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        for left, right in PAIRS:
            pair = f"{left}|{right}"
            values = np.sort(
                np.asarray(
                    [
                        max(float(row["kl_left_right_nats"]), 1e-12)
                        for row in rows
                        if row["dataset"] == dataset and row["pair"] == pair
                    ],
                    dtype=np.float64,
                )
            )
            cdf = np.arange(1, len(values) + 1) / len(values)
            axis.plot(values, cdf, label=pair.replace("|", r"$\rightarrow$"), color=COLORS[pair])
        axis.set_xscale("log")
        axis.set_xlim(left=1e-8)
        axis.set_ylim(0, 1.01)
        axis.set_title("AIME 2025" if dataset == "aime2025" else "MBPP+")
        axis.set_xlabel("Directional KL divergence (nats)")
        axis.grid(alpha=0.2, linewidth=0.6)
    axes[0][0].set_ylabel("Empirical CDF")
    axes[0][-1].legend(frameon=False, title="Direction", ncol=2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _bootstrap_step(rows, replicates, seed):
    values = np.asarray([float(row["entropy_nats"]) for row in rows], dtype=np.float64)
    if not len(values):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def plot_entropy_by_k(rows, output_dir, datasets, replicates, seed):
    paths = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        step_values = sorted({int(row["latent_steps"]) for row in dataset_rows})
        figure, axes = plt.subplots(
            1,
            len(step_values),
            figsize=(3.45 * len(step_values), 2.8),
            sharey=True,
            squeeze=False,
        )
        for axis, latent_steps in zip(axes[0], step_values):
            for method in METHODS:
                means, lows, highs = [], [], []
                for step in range(latent_steps):
                    selected = [
                        row
                        for row in dataset_rows
                        if row["latent_steps"] == latent_steps
                        and row["alignment"] == method
                        and row["step"] == step
                    ]
                    mean, low, high = _bootstrap_step(
                        selected,
                        replicates,
                        stable_seed(seed, dataset, latent_steps, method, step) % (2**32),
                    )
                    means.append(mean)
                    lows.append(low)
                    highs.append(high)
                x = np.arange(latent_steps)
                axis.plot(x, means, color=COLORS[method], label=method)
                axis.fill_between(x, lows, highs, color=COLORS[method], alpha=0.12, linewidth=0)
            axis.set_title(f"K = {latent_steps}")
            axis.set_xlabel("Recurrence step")
            axis.grid(alpha=0.2, linewidth=0.6)
        axes[0][0].set_ylabel("Output entropy (nats)")
        axes[0][-1].legend(frameon=False, title="Recurrence", ncol=2)
        figure.tight_layout()
        path = output_dir / "figures" / f"c6_entropy_by_cached_k_{dataset}.pdf"
        figure.savefig(path)
        plt.close(figure)
        paths.append(path)
    return paths


def create_run_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(
        json.dumps(vars(args), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    path = Path(args.output_root) / f"c6_prefix_alignment_{timestamp}_{digest}"
    path.mkdir(parents=True, exist_ok=False)
    for child in ("metrics", "summaries", "figures"):
        (path / child).mkdir()
    return path


def main(argv=None):
    args = parse_args(argv)
    configure_plot_style()
    discovered = discover_trajectories(args.trajectory_dir, args.datasets)
    model_names = {
        info["model_name"]
        for dataset_entries in discovered.values()
        for info in dataset_entries.values()
    }
    if len(model_names) != 1:
        raise RuntimeError(f"C6 requires one shared model, found: {sorted(model_names)}")
    args.model_name = next(iter(model_names))
    wrapper = load_model(args)
    wrapper.model.eval()
    run_dir = create_run_dir(args)
    provenance = []
    pair_rows = []
    entropy_metric_rows = []
    states_by_config = {}

    for dataset in args.datasets:
        entries = discovered[dataset]
        comparison_k = args.comparison_steps or max(entries)
        if comparison_k not in entries and not args.skip_prefix_comparison:
            raise FileNotFoundError(
                f"No compatible {dataset} trajectory with K={comparison_k}; "
                f"available K values: {sorted(entries)}"
            )
        for latent_steps, info in entries.items():
            trajectory, manifest = load_validated_trajectory(
                info, validate_sha256=not args.skip_sha256
            )
            provenance.append(
                {
                    "dataset": dataset,
                    "latent_steps": latent_steps,
                    "trajectory": str(info["path"]),
                    "manifest": str(info["manifest_path"]),
                    "trajectory_sha256": manifest.get("trajectory_sha256"),
                }
            )
            if not args.skip_entropy:
                entropy_metric_rows.extend(
                    entropy_rows(
                        trajectory,
                        dataset,
                        latent_steps,
                        wrapper,
                        args.entropy_chunk_size,
                    )
                )
                write_parquet(
                    run_dir / "metrics" / "c6_entropy_by_step.parquet",
                    entropy_metric_rows,
                )
            if not args.skip_prefix_comparison and latent_steps == comparison_k:
                config = alignment_config(manifest)
                config_key = json.dumps(config, sort_keys=True)
                if config_key not in states_by_config:
                    state_args = SimpleNamespace(
                        kernel_features=int(config["kernel_features"]),
                        kernel_temperature=float(config["kernel_temperature"]),
                        kernel_seed=int(config["kernel_seed"]),
                        kernel_chunk_size=int(config["kernel_chunk_size"]),
                        soft_chunk_size=int(config["soft_chunk_size"]),
                        align_ridge=float(config["linear_ridge"]),
                    )
                    states_by_config[config_key] = build_alignment_states(wrapper, state_args)
                pair_rows.extend(
                    collect_prefix_comparisons(
                        trajectory,
                        dataset,
                        latent_steps,
                        wrapper,
                        states_by_config[config_key],
                        args,
                        checkpoint=lambda current_rows: write_parquet(
                            run_dir / "metrics" / "c6_pair_metrics.parquet",
                            [*pair_rows, *current_rows],
                        ),
                    )
                )
                write_parquet(run_dir / "metrics" / "c6_pair_metrics.parquet", pair_rows)
            del trajectory

    artifacts = {}
    if pair_rows:
        pair_path = run_dir / "metrics" / "c6_pair_metrics.parquet"
        write_parquet(pair_path, pair_rows)
        pair_summary = summarize_pairs(pair_rows, args)
        summary_path = run_dir / "summaries" / "c6_pair_summary.json"
        write_json(summary_path, pair_summary)
        top10_path = run_dir / "figures" / "c6_top10_overlap_distribution.pdf"
        kl_path = run_dir / "figures" / "c6_kl_distribution.pdf"
        plot_top10_distribution(pair_rows, top10_path, args.datasets)
        plot_kl_distribution(pair_rows, kl_path, args.datasets)
        artifacts.update(
            pair_metrics=str(pair_path),
            pair_summary=str(summary_path),
            top10_figure=str(top10_path),
            kl_figure=str(kl_path),
        )
    if entropy_metric_rows:
        entropy_path = run_dir / "metrics" / "c6_entropy_by_step.parquet"
        write_parquet(entropy_path, entropy_metric_rows)
        entropy_summary_path = run_dir / "summaries" / "c6_entropy_summary.json"
        write_json(entropy_summary_path, summarize_entropy(entropy_metric_rows, args))
        entropy_figures = plot_entropy_by_k(
            entropy_metric_rows,
            run_dir,
            args.datasets,
            args.bootstrap_replicates,
            args.position_seed,
        )
        artifacts.update(
            entropy_metrics=str(entropy_path),
            entropy_summary=str(entropy_summary_path),
            entropy_figures=[str(path) for path in entropy_figures],
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study": "c6",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "configuration": vars(args),
        "semantics": {
            "position_unit": "greedy text-recurrence token, not Unicode character",
            "prefix_sampling": "without replacement independently within each question",
            "branch": "one transformer step from shared replayed text-prefix KV cache",
            "distribution": "softmax of output-head logits after the branched step",
            "kl_primary": "directional KL(left || right) in the pair order",
            "top5_overlap": "intersection size divided by five; any-overlap also reported",
            "token_sampling": "inverse-CDF categorical sample with SHA256-keyed uniform variate",
        },
        "trajectory_provenance": provenance,
        "artifacts": artifacts,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    print(f"C6 completed: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
