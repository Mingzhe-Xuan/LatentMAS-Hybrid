#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.cache import CacheError, CacheHandle, ReceiverEvaluationStore, atomic_write_json, file_sha256
from analysis.core.config import load_stt_config
from analysis.core.schemas import stable_hash
from analysis.core.statistics import exact_mcnemar, paired_question_bootstrap
from analysis.tasks._analysis import (_finalize_result, _plot, _result_dir,
                                      _validated_result_hit, _write_rows)
from analysis.tasks._common import load_job, parser


SYSTEMS = ("qwen_only", "mistral_only", "qwen_to_mistral", "mistral_to_qwen")


def _load_rows(cache_root: Path, dataset: str, selection_policy: str) -> tuple[list[dict], list[dict]]:
    import pyarrow.parquet as pq

    store = ReceiverEvaluationStore(cache_root, namespace="stt_receiver_evaluations")
    rows, provenance = [], []
    directory = cache_root / "stt_receiver_evaluations"
    for path in directory.glob("*/manifest.json") if directory.exists() else ():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        identity = manifest.get("identity", {})
        if identity.get("dataset") != dataset or identity.get("selection_policy") != selection_policy:
            continue
        if manifest.get("identity_hash") != stable_hash(identity):
            raise CacheError(f"incompatible STT Receiver identity: {path.parent}")
        handle = CacheHandle(manifest["cache_id"], path.parent, manifest["identity_hash"])
        store.validate(handle)
        for row in pq.read_table(path.parent / "answers.parquet").to_pylist():
            row.update(system=identity["system"], condition=identity,
                       cache_id=manifest["cache_id"])
            rows.append(row)
        provenance.append({"cache_id": manifest["cache_id"], "path": str(path),
                           "manifest_hash": file_sha256(path)})
    return rows, provenance


def _coverage(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["system"]].append(row)
    if set(grouped) != set(SYSTEMS):
        raise CacheError(f"STT system coverage mismatch: found={sorted(grouped)}")
    reference = None
    for system in SYSTEMS:
        ordered = sorted(grouped[system], key=lambda row: int(row["item_id"]))
        keys = [(int(row["item_id"]), row["question_hash"]) for row in ordered]
        if len(keys) != len(set(keys)):
            raise CacheError(f"duplicate STT question rows for {system}")
        reference = keys if reference is None else reference
        if keys != reference:
            raise CacheError(f"STT question coverage differs for {system}")
        grouped[system] = ordered
    return grouped


def run(args) -> int:
    load_stt_config(args.config)
    if not args.cache_only:
        raise ValueError("STT analysis requires --cache-only")
    job = load_job(args)
    if job.get("task") != "analyze_bidirectional_stt":
        raise ValueError("job task does not match entry point")
    if _validated_result_hit(args, job):
        return 10
    rows, provenance = _load_rows(Path(args.cache_root), job["dataset"], job["selection_policy"])
    grouped = _coverage(rows)
    cells = []
    for system in SYSTEMS:
        selected = grouped[system]
        cells.append({
            "system": system, "questions": len(selected),
            "accuracy": sum(bool(row["correct"]) for row in selected) / len(selected),
            "unparseable_rate": sum(row.get("prediction") is None for row in selected) / len(selected),
            "error_rate": sum(row.get("error") is not None for row in selected) / len(selected),
        })
    effects = []
    for name, left, right in (
        ("qwen_to_mistral_minus_mistral_only", "qwen_to_mistral", "mistral_only"),
        ("mistral_to_qwen_minus_qwen_only", "mistral_to_qwen", "qwen_only"),
    ):
        left_values = [float(row["correct"]) for row in grouped[left]]
        right_values = [float(row["correct"]) for row in grouped[right]]
        effects.append({
            "effect": name, "left": left, "right": right,
            "paired_bootstrap": paired_question_bootstrap(
                left_values, right_values, samples=10_000, seed=101),
            "exact_mcnemar": exact_mcnemar(left_values, right_values),
        })
    destination = _result_dir(args, job)
    _write_rows(destination / "metrics" / "question_metrics.parquet", rows)
    summary = {
        "dataset": job["dataset"], "selection_policy": job["selection_policy"],
        "systems": cells, "effects": effects,
        "pairing_key": ["item_id", "question_hash"],
        "bootstrap_unit": "question", "bootstrap_samples": 10_000,
    }
    atomic_write_json(destination / "summaries" / "summary.json", summary)
    atomic_write_json(destination / "provenance" / "cache_manifests.json", provenance)
    _plot(destination / "figures" / "accuracy.png",
          {cell["system"]: [(0.0, cell["accuracy"])] for cell in cells},
          "system", "accuracy/pass@1")
    _finalize_result(destination, job, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parser(analysis=True).parse_args()))
