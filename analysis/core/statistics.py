from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def paired_question_bootstrap(left: Sequence[float], right: Sequence[float], *,
                              samples: int = 2000, seed: int = 0,
                              confidence: float = .95) -> dict[str, float]:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired inputs must be non-empty equal-length vectors")
    delta = a - b
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
    low, high = _interval(draws, confidence)
    return {"estimate": float(delta.mean()), "ci_low": low, "ci_high": high,
            "questions": int(len(delta)), "bootstrap_samples": samples}


def nested_question_seed_bootstrap(rows: Iterable[dict], *, value_key: str = "value",
                                   question_key: str = "item_id", seed_key: str = "seed",
                                   samples: int = 2000, seed: int = 0,
                                   confidence: float = .95) -> dict[str, float]:
    grouped: dict[object, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[question_key]].append(float(row[value_key]))
    if not grouped:
        raise ValueError("rows must not be empty")
    groups = list(grouped.values())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for i in range(samples):
        selected = rng.integers(0, len(groups), size=len(groups))
        means = [np.mean(np.asarray(groups[j])[rng.integers(0, len(groups[j]), size=len(groups[j]))]) for j in selected]
        draws[i] = np.mean(means)
    estimate = float(np.mean([np.mean(x) for x in groups]))
    low, high = _interval(draws, confidence)
    return {"estimate": estimate, "ci_low": low, "ci_high": high,
            "questions": len(groups), "bootstrap_samples": samples}


def robust_slope(x: Sequence[float], y: Sequence[float]) -> float:
    xv, yv = np.asarray(x, float), np.asarray(y, float)
    if len(xv) != len(yv) or len(xv) < 2:
        raise ValueError("x and y require at least two paired values")
    slopes = [(yv[j] - yv[i]) / (xv[j] - xv[i]) for i in range(len(xv))
              for j in range(i + 1, len(xv)) if xv[j] != xv[i]]
    return float(np.median(slopes))
