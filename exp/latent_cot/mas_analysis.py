"""C1/C2 analysis for sequential LatentMAS on MBPP+."""

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

from data import load_mbppplus
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from utils import set_seed


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "exp_result" / "latent_cot" / "runs"
MAS_ALIGNMENTS = ("identical", "linear", "soft", "kernel")
LATENT_ROLES = ("planner", "critic", "refiner")
ROLE_INDEX = {role: index for index, role in enumerate(LATENT_ROLES)}
COLORS = {
    "identical": "#4c78a8",
    "linear": "#f58518",
    "soft": "#b279a2",
    "kernel": "#54a24b",
}
LINESTYLES = {"planner": "-", "critic": "--", "refiner": ":"}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(path: Path, rows) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression="zstd")
    temporary.replace(path)


def first_mbppplus_items(split: str, max_questions: int):
    """Return the dataset-order prefix, preserving stable original item IDs."""
    items = []
    for item_id, item in enumerate(load_mbppplus(split=split)):
        if len(items) >= max_questions:
            break
        items.append((item_id, item))
    return items


def _bootstrap_mean(values, replicates: int, seed: int):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _safe_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("._-") or "unknown"


def _run_directory(args) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = _safe_name(args.model_name.rsplit("/", 1)[-1])
    steps = "-".join(str(value) for value in args.latent_step_values)
    identity = hashlib.sha256(
        json.dumps(vars(args), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    name = (
        f"mbppplus_{args.split}_{args.study}_{model}_"
        f"k{_safe_name(steps)}_q{args.max_questions}_{timestamp}_{identity}"
    )
    path = RUNS_DIR / name
    suffix = 1
    while path.exists():
        path = RUNS_DIR / f"{name}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def _implementation_sha256():
    digest = hashlib.sha256()
    for path in (
        ROOT / "exp" / "latent_cot" / "run.py",
        ROOT / "exp" / "latent_cot" / "mas_analysis.py",
        ROOT / "methods" / "latent_mas.py",
        ROOT / "models.py",
    ):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataset_identity(indexed_items):
    payload = [
        {"item_id": item_id, "question": item["question"], "gold": item.get("gold", "")}
        for item_id, item in indexed_items
    ]
    return {
        "selection": "dataset_order_prefix",
        "item_ids": [item_id for item_id, _ in indexed_items],
        "content_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }


class EntropyObserver:
    def __init__(
        self, wrapper, *, item_id: int, alignment: str, latent_steps: int, split: str
    ):
        self.wrapper = wrapper
        self.item_id = int(item_id)
        self.alignment = alignment
        self.latent_steps = int(latent_steps)
        self.split = split
        self.rows = []
        self.previous = {}

    @torch.inference_mode()
    def __call__(self, role: str, local_step: int, hidden: torch.Tensor) -> None:
        output_head = self.wrapper.model.get_output_embeddings()
        if output_head is None:
            raise RuntimeError("C1 requires an accessible output embedding.")
        hidden_float = hidden.detach().float()
        finite_hidden = bool(torch.isfinite(hidden_float).all().item())
        entropy = None
        hidden_norm = None
        adjacent_cosine = None
        failure_reason = None
        if finite_hidden:
            hidden_norm = float(hidden_float.norm(dim=-1)[0].item())
            previous = self.previous.get(role)
            if previous is not None:
                adjacent_cosine = float(
                    torch.nn.functional.cosine_similarity(
                        previous, hidden_float, dim=-1
                    )[0].item()
                )
            logits = output_head(hidden).float()
            log_probabilities = torch.log_softmax(logits, dim=-1)
            entropy_tensor = -(
                log_probabilities.exp() * log_probabilities
            ).sum(dim=-1)
            if torch.isfinite(entropy_tensor).all():
                entropy = float(entropy_tensor[0].item())
            else:
                failure_reason = "non_finite_entropy"
            self.previous[role] = hidden_float.detach().clone()
        else:
            failure_reason = "non_finite_hidden"
        self.rows.append(
            {
                "dataset": "mbppplus",
                "split": self.split,
                "item_id": self.item_id,
                "alignment": self.alignment,
                "latent_steps_per_agent": self.latent_steps,
                "total_latent_steps": 3 * self.latent_steps,
                "agent": role,
                "local_step": int(local_step),
                "cumulative_step": ROLE_INDEX[role] * self.latent_steps
                + int(local_step),
                "entropy_nats": entropy,
                "hidden_norm": hidden_norm,
                "adjacent_cosine": adjacent_cosine,
                "finite": entropy is not None,
                "failure_reason": failure_reason,
            }
        )


def _method_args(args, alignment: str):
    values = {
        **vars(args),
        "task": "mbppplus",
        "method": "latent_mas",
        "prompt": "sequential",
        "align_method": alignment,
        "seed": args.generation_seed,
        "device2": args.device,
        "think": False,
        "sequential_info_only": False,
        "latent_only": False,
        "use_vllm": False,
        "use_second_HF_model": False,
        "enable_prefix_caching": False,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "soft_temperature": args.kernel_temperature,
    }
    return argparse.Namespace(**values)


def _summarize_c1(rows, args):
    series = {}
    for latent_steps in args.latent_step_values:
        for alignment in args.alignments:
            for role in LATENT_ROLES:
                key = f"k={latent_steps}.{alignment}.{role}"
                points = []
                for local_step in range(1, latent_steps + 1):
                    selected = [
                        row["entropy_nats"]
                        for row in rows
                        if row["latent_steps_per_agent"] == latent_steps
                        and row["alignment"] == alignment
                        and row["agent"] == role
                        and row["local_step"] == local_step
                        and row["finite"]
                    ]
                    low, high = _bootstrap_mean(
                        selected,
                        args.bootstrap_replicates,
                        args.probe_seed
                        + latent_steps * 10000
                        + ROLE_INDEX[role] * 1000
                        + local_step,
                    )
                    points.append(
                        {
                            "local_step": local_step,
                            "cumulative_step": ROLE_INDEX[role] * latent_steps
                            + local_step,
                            "n_valid_questions": len(selected),
                            "mean": float(np.mean(selected)) if selected else None,
                            "ci95_low": low,
                            "ci95_high": high,
                        }
                    )
                series[key] = {
                    "latent_steps_per_agent": latent_steps,
                    "alignment": alignment,
                    "agent": role,
                    "points": points,
                }
    failure_rows_by_reason = {}
    for row in rows:
        reason = row.get("failure_reason")
        if reason:
            failure_rows_by_reason[reason] = (
                failure_rows_by_reason.get(reason, 0) + 1
            )
    return {
        "study": "c1",
        "metric": "pre_unembedding_output_entropy_nats",
        "total_questions": len({row["item_id"] for row in rows}),
        "failure_rows_by_reason": failure_rows_by_reason,
        "step_definition": "post_feedback_hidden_state",
        "cumulative_step_definition": (
            "planner=t, critic=K+t, refiner=2K+t for local t in [1,K]"
        ),
        "series": series,
    }


def _summarize_c2(rows, args):
    series = {}
    for alignment in args.alignments:
        points = []
        for latent_steps in args.latent_step_values:
            selected = [
                float(row["correct"])
                for row in rows
                if row["alignment"] == alignment
                and row["latent_steps_per_agent"] == latent_steps
            ]
            low, high = _bootstrap_mean(
                selected,
                args.bootstrap_replicates,
                args.probe_seed + latent_steps * 100 + MAS_ALIGNMENTS.index(alignment),
            )
            correct = int(sum(selected))
            points.append(
                {
                    "latent_steps_per_agent": latent_steps,
                    "total_latent_steps": 3 * latent_steps,
                    "processed": len(selected),
                    "correct": correct,
                    "accuracy": correct / len(selected) if selected else None,
                    "ci95_low": low,
                    "ci95_high": high,
                    "parse_success": sum(
                        1
                        for row in rows
                        if row["alignment"] == alignment
                        and row["latent_steps_per_agent"] == latent_steps
                        and row["parse_success"]
                    ),
                    "failures": len(selected) - correct,
                }
            )
        series[alignment] = points
    return {
        "study": "c2",
        "metric": "mbppplus_pass_rate",
        "latent_budget": "three non-judger agents each use K latent steps",
        "series": series,
    }


def _plot_c1(summary, path: Path, args):
    figure, axes = plt.subplots(3, 3, figsize=(18, 14), squeeze=False)
    flat = [axis for row in axes for axis in row]
    for axis, latent_steps in zip(flat, args.latent_step_values):
        for alignment in args.alignments:
            for role in LATENT_ROLES:
                entry = summary["series"][f"k={latent_steps}.{alignment}.{role}"]
                points = entry["points"]
                x = np.asarray([point["cumulative_step"] for point in points])
                mean = np.asarray([
                    np.nan if point["mean"] is None else point["mean"]
                    for point in points
                ])
                low = np.asarray([
                    np.nan if point["ci95_low"] is None else point["ci95_low"]
                    for point in points
                ])
                high = np.asarray([
                    np.nan if point["ci95_high"] is None else point["ci95_high"]
                    for point in points
                ])
                label = f"{alignment}/{role}"
                axis.plot(
                    x,
                    mean,
                    color=COLORS[alignment],
                    linestyle=LINESTYLES[role],
                    linewidth=1.6,
                    label=label,
                )
                axis.fill_between(x, low, high, color=COLORS[alignment], alpha=0.06)
        axis.axvline(latent_steps, color="black", alpha=0.25, linewidth=0.8)
        axis.axvline(2 * latent_steps, color="black", alpha=0.25, linewidth=0.8)
        axis.set_title(f"K={latent_steps} per agent")
        axis.set_xlabel("Cumulative latent step")
        axis.grid(alpha=0.2)
    flat[0].set_ylabel("Output entropy (nats)")
    handles, labels = flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.suptitle(
        "C1: sequential LatentMAS entropy trajectories\n"
        "Color=alignment; linestyle=agent; bands=question bootstrap 95% CI"
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _plot_c2(summary, path: Path, args):
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for alignment in args.alignments:
        points = summary["series"][alignment]
        x = np.asarray([point["latent_steps_per_agent"] for point in points])
        accuracy = np.asarray([
            np.nan if point["accuracy"] is None else point["accuracy"]
            for point in points
        ])
        low = np.asarray([
            np.nan if point["ci95_low"] is None else point["ci95_low"]
            for point in points
        ])
        high = np.asarray([
            np.nan if point["ci95_high"] is None else point["ci95_high"]
            for point in points
        ])
        axis.plot(x, accuracy, marker="o", color=COLORS[alignment], label=alignment)
        axis.fill_between(x, low, high, color=COLORS[alignment], alpha=0.12)
        for point in points:
            if point["accuracy"] is not None:
                axis.annotate(
                    f'{point["correct"]}/{point["processed"]}',
                    (point["latent_steps_per_agent"], point["accuracy"]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
    axis.set_xlabel("Latent steps per Planner/Critic/Refiner agent (K)")
    axis.set_ylabel("MBPP+ accuracy")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(title="Alignment")
    axis.set_title("C2: sequential LatentMAS accuracy vs. per-agent latent steps")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def run_mas_study(args, logger: logging.Logger | None = None):
    if args.dataset != "mbppplus":
        raise ValueError("C1/C2 currently require --dataset mbppplus.")
    logger = logger or logging.getLogger(f"latent_cot.{args.study}")
    run_dir = _run_directory(args)
    manifest_path = run_dir / "run_manifest.json"
    indexed_items = first_mbppplus_items(args.split, args.max_questions)
    if len(indexed_items) != args.max_questions:
        raise RuntimeError(
            f"Requested {args.max_questions} MBPP+ items but loaded {len(indexed_items)}."
        )
    manifest = {
        "status": "running",
        "study": args.study,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_directory": str(run_dir),
        "args": vars(args),
        "dataset_identity": _dataset_identity(indexed_items),
        "implementation_sha256": _implementation_sha256(),
        "generation": {
            "decoding": "greedy",
            "temperature": 0.0,
            "top_p_ignored": True,
        },
        "sequential_context": {
            "roles": ["planner", "critic", "refiner", "judger"],
            "sequential_info_only": False,
            "latent_only": False,
        },
    }
    _write_json(manifest_path, manifest)
    started = time.time()
    try:
        model_args = _method_args(args, args.alignments[0])
        wrapper = ModelWrapper(args.model_name, args.device, use_vllm=False, args=model_args)
        config_payload = wrapper.model.config.to_dict()
        manifest["model"] = {
            "name": args.model_name,
            "resolved_commit": getattr(wrapper.model.config, "_commit_hash", None),
            "config_sha256": hashlib.sha256(
                json.dumps(
                    config_payload, sort_keys=True, default=str
                ).encode("utf-8")
            ).hexdigest(),
            "parameter_dtype": str(next(wrapper.model.parameters()).dtype),
        }
        _write_json(manifest_path, manifest)
        entropy_rows = []
        accuracy_rows = []
        output_head = wrapper.model.get_output_embeddings()
        if output_head is None:
            raise RuntimeError("Model does not expose an output embedding.")
        for alignment in args.alignments:
            wrapper.align_method = alignment
            model_args.align_method = alignment
            for latent_steps in args.latent_step_values:
                logger.info(
                    "%s alignment=%s K=%d started",
                    args.study.upper(),
                    alignment,
                    latent_steps,
                )
                for number, (item_id, item) in enumerate(indexed_items, start=1):
                    set_seed(args.generation_seed)
                    observer = (
                        EntropyObserver(
                            wrapper,
                            item_id=item_id,
                            alignment=alignment,
                            latent_steps=latent_steps,
                            split=args.split,
                        )
                        if args.study == "c1"
                        else None
                    )
                    method = LatentMASMethod(
                        wrapper,
                        latent_steps=latent_steps,
                        judger_max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        generate_bs=1,
                        args=model_args,
                        latent_step_observer=observer,
                        latent_roles_only=args.study == "c1",
                    )
                    item_started = time.time()
                    failure_reason = None
                    try:
                        result = method.run_batch([item])[0]
                    except Exception as error:
                        result = {
                            "prediction": None,
                            "raw_prediction": "",
                            "correct": False,
                            "error": f"{type(error).__name__}: {error}",
                            "agents": [],
                        }
                        failure_reason = result["error"]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    elapsed = time.time() - item_started
                    if observer is not None:
                        observed = {
                            (row["agent"], row["local_step"])
                            for row in observer.rows
                        }
                        entropy_rows.extend(observer.rows)
                        for role in LATENT_ROLES:
                            for local_step in range(1, latent_steps + 1):
                                if (role, local_step) in observed:
                                    continue
                                entropy_rows.append(
                                    {
                                        "dataset": "mbppplus",
                                        "split": args.split,
                                        "item_id": int(item_id),
                                        "alignment": alignment,
                                        "latent_steps_per_agent": int(latent_steps),
                                        "total_latent_steps": int(3 * latent_steps),
                                        "agent": role,
                                        "local_step": local_step,
                                        "cumulative_step": (
                                            ROLE_INDEX[role] * latent_steps
                                            + local_step
                                        ),
                                        "entropy_nats": None,
                                        "hidden_norm": None,
                                        "adjacent_cosine": None,
                                        "finite": False,
                                        "failure_reason": (
                                            failure_reason
                                            or "missing_hidden_state"
                                        ),
                                    }
                                )
                    else:
                        prediction = result.get("prediction")
                        accuracy_rows.append(
                            {
                                "dataset": "mbppplus",
                                "split": args.split,
                                "item_id": int(item_id),
                                "alignment": alignment,
                                "latent_steps_per_agent": int(latent_steps),
                                "total_latent_steps": int(3 * latent_steps),
                                "question": item["question"],
                                "gold": item.get("gold", ""),
                                "prediction": prediction,
                                "raw_prediction": result.get("raw_prediction", ""),
                                "correct": bool(result.get("correct", False)),
                                "parse_success": prediction is not None,
                                "failure_reason": failure_reason or result.get("error"),
                                "wall_seconds": elapsed,
                                "judger_output_tokens": next(
                                    (
                                        agent.get("metrics", {})
                                        .get("tokens", {})
                                        .get("text_output", 0)
                                        for agent in result.get("agents", [])
                                        if agent.get("role") == "judger"
                                    ),
                                    0,
                                ),
                            }
                        )
                    logger.info(
                        "%s alignment=%s K=%d item=%d (%d/%d) completed",
                        args.study.upper(),
                        alignment,
                        latent_steps,
                        item_id,
                        number,
                        len(indexed_items),
                    )
                source_rows = entropy_rows if args.study == "c1" else accuracy_rows
                cell_rows = [
                    row
                    for row in source_rows
                    if row["alignment"] == alignment
                    and row["latent_steps_per_agent"] == latent_steps
                ]
                checkpoint_path = (
                    run_dir
                    / "checkpoints"
                    / f"{args.study}_{alignment}_k{latent_steps}.parquet"
                )
                _write_parquet(checkpoint_path, cell_rows)
                logger.info(
                    "%s checkpoint saved: %s", args.study.upper(), checkpoint_path
                )
        if args.study == "c1":
            expected_rows = (
                len(indexed_items)
                * len(args.alignments)
                * 3
                * sum(args.latent_step_values)
            )
            if len(entropy_rows) != expected_rows:
                raise RuntimeError(
                    f"C1 row-count invariant failed: "
                    f"{len(entropy_rows)} != {expected_rows}"
                )
            finite_row_count = sum(row["finite"] for row in entropy_rows)
            finite_rows_by_series = {
                f"k={latent_steps}.{alignment}.{role}": sum(
                    row["finite"]
                    for row in entropy_rows
                    if row["latent_steps_per_agent"] == latent_steps
                    and row["alignment"] == alignment
                    and row["agent"] == role
                )
                for latent_steps in args.latent_step_values
                for alignment in args.alignments
                for role in LATENT_ROLES
            }
            empty_series = [
                name
                for name, count in finite_rows_by_series.items()
                if count == 0
            ]
            if empty_series:
                raise RuntimeError(
                    "C1 produced no finite entropy observations for: "
                    + ", ".join(empty_series)
                )
            metrics_path = run_dir / "metrics" / "c1_entropy_by_agent_step.parquet"
            summary_path = run_dir / "summaries" / "c1_summary.json"
            figure_path = run_dir / "figures" / "c1_entropy_vs_cumulative_step.pdf"
            _write_parquet(metrics_path, entropy_rows)
            summary = _summarize_c1(entropy_rows, args)
            _write_json(summary_path, summary)
            _plot_c1(summary, figure_path, args)
            row_count = len(entropy_rows)
        else:
            expected_rows = (
                len(indexed_items)
                * len(args.alignments)
                * len(args.latent_step_values)
            )
            if len(accuracy_rows) != expected_rows:
                raise RuntimeError(
                    f"C2 row-count invariant failed: "
                    f"{len(accuracy_rows)} != {expected_rows}"
                )
            metrics_path = run_dir / "metrics" / "c2_accuracy_by_question.parquet"
            summary_path = run_dir / "summaries" / "c2_summary.json"
            figure_path = run_dir / "figures" / "c2_accuracy_vs_steps.pdf"
            _write_parquet(metrics_path, accuracy_rows)
            summary = _summarize_c2(accuracy_rows, args)
            _write_json(summary_path, summary)
            _plot_c2(summary, figure_path, args)
            row_count = len(accuracy_rows)
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": time.time() - started,
                "row_count": row_count,
                "finite_row_count": (
                    finite_row_count if args.study == "c1" else None
                ),
                "finite_rows_by_series": (
                    finite_rows_by_series if args.study == "c1" else None
                ),
                "artifacts": {
                    "metrics": str(metrics_path),
                    "summary": str(summary_path),
                    "figure": str(figure_path),
                },
            }
        )
        _write_json(manifest_path, manifest)
        logger.info("%s completed: %s", args.study.upper(), run_dir)
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
