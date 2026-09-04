#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.cache import CacheError, atomic_write_bytes, atomic_write_json, file_sha256
from analysis.core.config import load_stt_config
from analysis.tasks._analysis import _finalize_result, _result_dir, _validated_result_hit, _write_rows
from analysis.tasks._common import load_job, parser


def run(args) -> int:
    config = load_stt_config(args.config).raw
    if not args.cache_only:
        raise ValueError("STT report construction requires --cache-only")
    job = load_job(args)
    if job.get("task") != "build_bidirectional_stt_report":
        raise ValueError("job task does not match entry point")
    if _validated_result_hit(args, job):
        return 10
    expected_selection = "first-1" if job.get("smoke") else "all"
    expected_datasets = tuple(job.get("datasets") or config["datasets"])
    expected_analysis_ids = job.get("analysis_cache_ids", {})
    if set(expected_analysis_ids) != set(expected_datasets):
        raise CacheError("STT report job must name exactly one analysis result per dataset")
    root = Path(args.result_root) / "analyze_bidirectional_stt"
    summaries: dict[str, dict] = {}
    provenance = []
    for expected_dataset in expected_datasets:
        analysis_id = expected_analysis_ids[expected_dataset]
        path = root / analysis_id / "summaries" / "summary.json"
        if not path.exists():
            raise CacheError(f"required STT analysis result does not exist: {analysis_id}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        dataset = summary.get("dataset")
        if dataset != expected_dataset or summary.get("selection_policy") != expected_selection:
            raise CacheError(f"STT analysis dependency identity mismatch: {analysis_id}")
        manifest_path = path.parents[1] / "run_manifest.json"
        if not manifest_path.exists():
            raise CacheError(f"STT analysis result has no run manifest: {path.parents[1]}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative = str(path.relative_to(path.parents[1])).replace("\\", "/")
        if (manifest.get("status") != "complete"
                or manifest.get("result_id") != analysis_id
                or manifest.get("task") != "analyze_bidirectional_stt"
                or manifest.get("artifacts", {}).get(relative) != file_sha256(path)):
            raise CacheError(f"STT analysis summary is not validated: {path}")
        summaries[dataset] = summary
        provenance.append({"path": str(manifest_path), "manifest_hash": file_sha256(manifest_path),
                           "dataset": dataset})
    if set(summaries) != set(expected_datasets):
        raise CacheError(f"STT report dataset coverage mismatch: found={sorted(summaries)}")

    system_scores: dict[str, list[float]] = defaultdict(list)
    effect_scores: dict[str, list[float]] = defaultdict(list)
    dataset_rows = []
    for dataset in expected_datasets:
        summary = summaries[dataset]
        for cell in summary["systems"]:
            system_scores[cell["system"]].append(float(cell["accuracy"]))
            dataset_rows.append({"dataset": dataset, "kind": "system",
                                 "name": cell["system"], "estimate": float(cell["accuracy"])})
        for effect in summary["effects"]:
            estimate = float(effect["paired_bootstrap"]["estimate"])
            effect_scores[effect["effect"]].append(estimate)
            dataset_rows.append({"dataset": dataset, "kind": "effect",
                                 "name": effect["effect"], "estimate": estimate})
    macro_systems = [{"system": name, "macro_average": sum(values) / len(values),
                      "dataset_count": len(values)} for name, values in sorted(system_scores.items())]
    macro_effects = [{"effect": name, "macro_average": sum(values) / len(values),
                      "dataset_count": len(values)} for name, values in sorted(effect_scores.items())]
    combined = {
        "protocol_version": config["protocol_version"],
        "selection_policy": expected_selection, "macro_weighting": "equal_weight_per_dataset",
        "datasets": [summaries[name] for name in expected_datasets],
        "system_macro_averages": macro_systems, "effect_macro_averages": macro_effects,
    }
    destination = _result_dir(args, job)
    _write_rows(destination / "metrics" / "dataset_table.parquet", dataset_rows)
    atomic_write_json(destination / "summaries" / "combined.json", combined)
    lines = ["# Bidirectional Exact STT analysis", "", "## System macro averages", "",
             "| System | Macro score |", "|---|---:|"]
    lines.extend(f"| `{row['system']}` | {row['macro_average']:.6f} |" for row in macro_systems)
    lines.extend(["", "## Paired effect macro averages", "",
                  "| Effect | Macro difference |", "|---|---:|"])
    lines.extend(f"| `{row['effect']}` | {row['macro_average']:+.6f} |" for row in macro_effects)
    lines.extend(["", "Per-dataset 95% paired-bootstrap confidence intervals and exact McNemar tests are stored in `summaries/combined.json`.", ""])
    atomic_write_bytes(destination / "summaries" / "report.md", "\n".join(lines).encode("utf-8"))
    atomic_write_json(destination / "provenance" / "analysis_manifests.json", provenance)
    _finalize_result(destination, job, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parser(analysis=True).parse_args()))
