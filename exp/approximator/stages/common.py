"""Shared primitives for the fixed Refiner-to-Judger studies."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from alignment import build_kernel_state, build_linear_state, positive_features

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "exp_result" / "approximator"
SOURCES = ("refiner_latent_reply_hidden",)
ARTIFACT_CONTEXT = {}


def set_result_root(path):
    global RESULT
    RESULT = Path(path)


def set_artifact_context(context):
    global ARTIFACT_CONTEXT
    ARTIFACT_CONTEXT = dict(context)


def contextual_stem(stem):
    return stem


def save_figure(figure, stem):
    figure_dir = RESULT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"{contextual_stem(stem)}.pdf"
    figure.savefig(path)
    path.with_suffix(".json").write_text(
        json.dumps(ARTIFACT_CONTEXT, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def write_rows(rows, stem):
    return write_rows_path(rows, RESULT / "metrics" / f"{stem}.parquet")


def write_rows_path(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows)), path, compression="zstd")
    return path


def read_rows(path):
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def write_csv(rows, stem):
    rows = list(rows)
    path = RESULT / "metrics" / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return path


def write_summary(payload, stem):
    path = RESULT / "summaries" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    return path


def describe_values(
    values,
    *,
    state_count=None,
    question_count=None,
    bootstrap_replicates=0,
    seed=42,
):
    values = np.asarray(list(values), dtype=np.float64)
    finite = values[np.isfinite(values)]
    output = {
        "count": int(finite.size),
        "state_count": int(len(values) if state_count is None else state_count),
        "finite_count": int(finite.size),
        "invalid_count": int(len(values) - finite.size),
    }
    if question_count is not None:
        output["question_count"] = int(question_count)
    if not finite.size:
        return output
    variance = float(finite.var(ddof=1)) if finite.size > 1 else 0.0
    output.update(
        mean=float(finite.mean()),
        median=float(np.median(finite)),
        variance=variance,
        std=float(np.sqrt(variance)),
        min=float(finite.min()),
        q01=float(np.quantile(finite, 0.01)),
        q05=float(np.quantile(finite, 0.05)),
        q25=float(np.quantile(finite, 0.25)),
        q50=float(np.quantile(finite, 0.50)),
        q75=float(np.quantile(finite, 0.75)),
        q90=float(np.quantile(finite, 0.90)),
        q95=float(np.quantile(finite, 0.95)),
        q99=float(np.quantile(finite, 0.99)),
        max=float(finite.max()),
    )
    if bootstrap_replicates and finite.size:
        rng = np.random.default_rng(seed)
        bootstrap_indices = rng.integers(
            0, finite.size, size=(bootstrap_replicates, finite.size)
        )
        bootstrap = finite[bootstrap_indices].mean(axis=1)
        output["ci95_low"] = float(np.quantile(bootstrap, 0.025))
        output["ci95_high"] = float(np.quantile(bootstrap, 0.975))
    return output


def clustered_metric(rows, metric, args):
    rows = list(rows)
    finite_by_question = {}
    finite_state_count = 0
    invalid_count = 0
    for row in rows:
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            invalid_count += 1
            continue
        finite_state_count += 1
        finite_by_question.setdefault(row["item_id"], []).append(float(value))
    question_values = [
        float(np.mean(values)) for values in finite_by_question.values()
    ]
    output = describe_values(
        question_values,
        state_count=len(rows),
        question_count=len(question_values),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.probe_seed,
    )
    output["finite_state_count"] = finite_state_count
    output["invalid_state_count"] = invalid_count
    return output


def rate_summary(rows, metric):
    rows = list(rows)
    values = [
        bool(row[metric])
        for row in rows
        if row.get(metric) is not None
    ]
    true_count = sum(values)
    return {
        "count": len(values),
        "state_count": len(rows),
        "valid_count": len(values),
        "true_count": int(true_count),
        "false_count": int(len(values) - true_count),
        "rate": float(true_count / len(values)) if values else None,
    }


def exact(q, wo, wi, bias, tau):
    logits = wo @ q.to(wo.device, dtype=wo.dtype)
    if bias is not None:
        logits = logits + bias
    probabilities = torch.softmax(logits / tau, 0)
    return probabilities @ wi, probabilities


def kernel_map(q, kernel):
    query = positive_features(
        q.to(kernel.omega.device).float()[None] / kernel.temperature,
        kernel.omega,
        stabilize=True,
    )[0]
    denominator = query @ kernel.denominator
    valid = bool(
        torch.isfinite(denominator)
        and denominator > torch.finfo(denominator.dtype).eps
    )
    if not valid:
        return (
            torch.full(
                (kernel.numerator.shape[0],),
                float("nan"),
                device=query.device,
            ),
            False,
        )
    return (query @ kernel.numerator.T) / denominator, True


def iid_kernel(wo, wi, bias, m, tau, seed, chunk):
    generator = torch.Generator(device=wo.device)
    generator.manual_seed(seed)
    omega = torch.randn((m, wo.shape[1]), generator=generator, device=wo.device)
    numerator = torch.zeros((wi.shape[1], m), device=wo.device)
    denominator = torch.zeros(m, device=wo.device)
    b = torch.zeros(len(wo), device=wo.device) if bias is None else bias.float()
    shift = b.max()
    for start in range(0, len(wo), chunk):
        stop = min(start + chunk, len(wo))
        features = positive_features(wo[start:stop], omega)
        alpha = torch.exp((b[start:stop] - shift) / tau)[:, None]
        numerator += wi[start:stop].T @ (alpha * features)
        denominator += (alpha * features).sum(0)
    return type(
        "Kernel",
        (),
        {
            "omega": omega,
            "numerator": numerator,
            "denominator": denominator,
            "temperature": tau,
        },
    )()


def audit(states, wo, wi, bias, args):
    if args.skip_float64_audit:
        return None
    errors = []
    for state in states[:256]:
        f32, _ = exact(state.vector, wo, wi, bias, args.kernel_temperature)
        f64, _ = exact(
            state.vector.double(),
            wo.double(),
            wi.double(),
            None if bias is None else bias.double(),
            args.kernel_temperature,
        )
        errors.append(float((f32.double() - f64).norm() / f64.norm().clamp_min(1e-12)))
    p99 = float(np.quantile(errors, 0.99)) if errors else 0.0
    if p99 > 1e-4:
        raise RuntimeError(f"Stopped: float64 audit p99={p99:.3e} > 1e-4")
    return p99


def base(state):
    return {
        **ARTIFACT_CONTEXT,
        "item_id": state.item_id,
        "source": state.source,
        "role": state.role,
        "state_kind": state.state_kind,
        "model_name": state.model_name,
        "position": state.position,
        "turn_id": state.turn_id,
        "agent_id": state.agent_id,
    }


def stat_rows(rows, metric, args, extra=()):
    output = []
    rng = np.random.default_rng(args.probe_seed)
    groups = {}
    for row in rows:
        if metric in row and np.isfinite(row[metric]):
            key = tuple(row[field] for field in extra) + (row["source"],)
            groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        by_question = {}
        for row in group:
            by_question.setdefault(row["item_id"], []).append(row[metric])
        values = np.array([np.mean(items) for items in by_question.values()])
        if not len(values):
            continue
        bootstrap = np.array(
            [
                np.mean(rng.choice(values, len(values), replace=True))
                for _ in range(args.bootstrap_replicates)
            ]
        )
        output.append(
            {
                **ARTIFACT_CONTEXT,
                "metric": metric,
                **dict(zip(extra + ("source",), key)),
                "n_questions": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    return output


def histogram_ecdf(rows, metric, study):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for source in sorted({row.get("source") for row in rows}):
        values = np.array(
            [
                row[metric]
                for row in rows
                if row.get("source") == source
                and np.isfinite(row.get(metric, np.nan))
            ]
        )
        if len(values):
            axes[0].hist(values, bins=40, density=True, histtype="step", label=source)
            axes[1].plot(
                np.sort(values),
                np.arange(1, len(values) + 1) / len(values),
                label=source,
            )
    axes[0].set_title(metric + " histogram")
    axes[1].set_title(metric + " ECDF")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    save_figure(figure, f"{study}_{metric}")
    plt.close(figure)


def rank_ids(probabilities, state, args):
    order = torch.argsort(probabilities, descending=True).cpu().tolist()
    rng = random.Random(args.probe_seed + state.item_id * 1009 + state.position)
    answer = [("rank_1", order[0])]
    for name, low, high in (
        ("rank_2_10", 1, 10),
        ("rank_11_100", 10, 100),
        ("rank_101_1000", 100, 1000),
        ("rank_gt_1000", 1000, len(order)),
    ):
        pool = order[low : min(high, len(order))]
        answer.extend((name, index) for index in rng.sample(pool, min(3, len(pool))))
    return answer


def overlap(first, second, k):
    k = min(k, len(first), len(second))
    return len(
        set(torch.topk(first, k).indices.tolist())
        & set(torch.topk(second, k).indices.tolist())
    ) / max(k, 1)
