from __future__ import annotations

from typing import Callable, Iterable

from analysis.core.schemas import AnalysisItem, DatasetSnapshot, stable_hash


_LOADERS = {
    "gsm8k": "load_gsm8k", "aime2025": "load_aime2025",
    "aime2024": "load_aime2024", "gpqa": "load_gpqa_diamond",
    "arc_easy": "load_arc_easy", "arc_challenge": "load_arc_challenge",
    "mbppplus": "load_mbppplus", "humanevalplus": "load_humanevalplus",
    "medqa": "load_medqa",
}


def snapshot_from_rows(dataset: str, split: str, rows: Iterable[dict], *,
                       max_samples: int | None = None) -> DatasetSnapshot:
    items = []
    for index, row in enumerate(rows):
        if max_samples is not None and index >= max_samples:
            break
        items.append(AnalysisItem(index, str(row["question"]),
                                  str(row.get("solution", "")), str(row.get("gold", ""))))
    policy = "all" if max_samples is None else f"first-{max_samples}"
    fingerprint = stable_hash({
        "dataset": dataset, "split": split, "selection_policy": policy,
        "items": [{"item_id": x.item_id, "question_hash": x.question_hash,
                   "solution": x.solution, "gold": x.gold} for x in items],
    })
    return DatasetSnapshot(dataset, split, policy, tuple(items), fingerprint)


def load_analysis_items(dataset: str, split: str, *, cache_dir: str | None = None,
                        max_samples: int | None = None,
                        loader: Callable[..., Iterable[dict]] | None = None) -> DatasetSnapshot:
    if dataset not in _LOADERS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if loader is None:
        import data
        loader = getattr(data, _LOADERS[dataset])
    rows = loader(split=split, cache_dir=cache_dir)
    return snapshot_from_rows(dataset, split, rows, max_samples=max_samples)
