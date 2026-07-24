"""Shared primitives for S0--S4 stages."""

from __future__ import annotations
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from alignment import build_kernel_state, build_linear_state, positive_features

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "exp_result" / "approximator"
SOURCES = ("prompt", "reply")


def write_rows(rows, stem):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = RESULT / "metrics" / f"{stem}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return path


def exact(q, wo, wi, bias, tau):
    logits = wo @ (q.to(wo.device, dtype=wo.dtype) / tau)
    logits = logits if bias is None else logits + bias
    p = torch.softmax(logits, 0)
    return p @ wi, p


def kernel_map(q, k):
    u = positive_features(
        q.to(k.omega.device).float()[None] / k.temperature, k.omega, stabilize=True
    )[0]
    d = u @ k.denominator
    ok = bool(torch.isfinite(d) and d > torch.finfo(d.dtype).eps)
    return (
        (u @ k.numerator.T) / d
        if ok
        else torch.full((k.numerator.shape[0],), float("nan"), device=u.device)
    ), ok


def iid_kernel(wo, wi, bias, m, tau, seed, chunk):
    g = torch.Generator(device=wo.device)
    g.manual_seed(seed)
    omega = torch.randn((m, wo.shape[1]), generator=g, device=wo.device)
    # Reuse aggregation with a tiny state-like object.
    num = torch.zeros((wi.shape[1], m), device=wo.device)
    den = torch.zeros(m, device=wo.device)
    b = torch.zeros(len(wo), device=wo.device) if bias is None else bias.float()
    shift = b.max()
    for start in range(0, len(wo), chunk):
        stop = min(start + chunk, len(wo))
        ph = positive_features(wo[start:stop], omega)
        alpha = torch.exp(b[start:stop] - shift)[:, None]
        num += wi[start:stop].T @ (alpha * ph)
        den += (alpha * ph).sum(0)
    return type(
        "K",
        (),
        {"omega": omega, "numerator": num, "denominator": den, "temperature": tau},
    )()


def audit(states, wo, wi, bias, args):
    if args.skip_float64_audit:
        return
    errors = []
    for state in states[:256]:
        f32, _ = exact(state.vector, wo, wi, bias, args.tau)
        f64, _ = exact(
            state.vector.double(),
            wo.double(),
            wi.double(),
            None if bias is None else bias.double(),
            args.tau,
        )
        errors.append(float((f32.double() - f64).norm() / f64.norm().clamp_min(1e-12)))
    p99 = float(np.quantile(errors, 0.99)) if errors else 0.0
    if p99 > 1e-4:
        raise RuntimeError(f"Stopped: float64 audit p99={p99:.3e} > 1e-4")
    return p99


def base(s):
    return {
        "item_id": s.item_id,
        "source": s.source,
        "position": s.position,
        "prompt_length": s.prompt_length,
        "reply_length": s.reply_length,
        "turn_id": s.turn_id,
        "agent_id": s.agent_id,
    }


def stat_rows(rows, metric, args, extra=()):
    out = []
    rng = np.random.default_rng(args.probe_seed)
    groups = {}
    for r in rows:
        if metric in r and np.isfinite(r[metric]):
            groups.setdefault(tuple(r[x] for x in extra) + (r["source"],), []).append(r)
    for key, rs in groups.items():
        by = {}
        for r in rs:
            by.setdefault(r["item_id"], []).append(r[metric])
        v = np.array([np.mean(x) for x in by.values()])
        boots = (
            np.array(
                [
                    np.mean(rng.choice(v, len(v), replace=True))
                    for _ in range(args.bootstrap_replicates)
                ]
            )
            if len(v)
            else np.array([])
        )
        d = {
            "metric": metric,
            **dict(zip(extra + ("source",), key)),
            "n_questions": len(v),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p90": float(np.quantile(v, 0.9)),
            "p95": float(np.quantile(v, 0.95)),
            "p99": float(np.quantile(v, 0.99)),
            "ci95_low": float(np.quantile(boots, 0.025)),
            "ci95_high": float(np.quantile(boots, 0.975)),
        }
        out.append(d)
    return out


def histogram_ecdf(rows, metric, study):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for source in SOURCES:
        x = np.array(
            [
                r[metric]
                for r in rows
                if r.get("source") == source and np.isfinite(r.get(metric, np.nan))
            ]
        )
        if len(x):
            ax[0].hist(x, bins=40, density=True, histtype="step", label=source)
            ax[1].plot(np.sort(x), np.arange(1, len(x) + 1) / len(x), label=source)
    ax[0].set_title(metric + " histogram")
    ax[1].set_title(metric + " ECDF")
    [a.legend() for a in ax]
    (RESULT / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(RESULT / "figures" / f"{study}_{metric}.pdf")
    plt.close(fig)


def rank_ids(p, s, args):
    order = torch.argsort(p, descending=True).cpu().tolist()
    rng = random.Random(args.probe_seed + s.item_id * 1009 + s.position)
    ans = [("rank_1", order[0])]
    for name, lo, hi in (
        ("rank_2_10", 1, 10),
        ("rank_11_100", 10, 100),
        ("rank_101_1000", 100, 1000),
        ("rank_gt_1000", 1000, len(order)),
    ):
        pool = order[lo : min(hi, len(order))]
        ans += [(name, x) for x in rng.sample(pool, min(3, len(pool)))]
    return ans


def overlap(p, q, k):
    return (
        len(
            set(torch.topk(p, k).indices.tolist())
            & set(torch.topk(q, k).indices.tolist())
        )
        / k
    )
