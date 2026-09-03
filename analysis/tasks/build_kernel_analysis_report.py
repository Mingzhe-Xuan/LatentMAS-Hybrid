#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.core.cache import CacheError, atomic_write_json, file_sha256
from analysis.core.config import load_config
from analysis.tasks._common import load_job, parser

def run(args) -> int:
    config = load_config(args.config).raw
    if not args.cache_only:
        raise ValueError("report construction requires cache-only mode")
    job = load_job(args)
    if job.get("task") != "build_kernel_analysis_report":
        raise ValueError("job task does not match entry point")
    root = Path(args.result_root)
    sources = []
    for task in ("analyze_logit_entropy", "analyze_kernel_scaling",
                 "analyze_aligned_state_variance", "analyze_perturbation_stability",
                 "analyze_sender_receiver_performance"):
        for path in (root / task).glob("*/summaries/summary.json") if (root / task).exists() else ():
            summary = json.loads(path.read_text(encoding="utf-8"))
            expected_selection = "first-1" if job.get("smoke") else "all"
            if summary.get("selection_policy", "all") == expected_selection:
                sources.append({"task": task, "path": str(path), "sha256": file_sha256(path),
                                "summary": summary})
    if not sources:
        raise CacheError("no validated analysis summaries are available")
    destination = root / job["task"] / job["effective_cache_id"]
    for child in ("metrics", "summaries", "figures", "provenance"):
        (destination / child).mkdir(parents=True, exist_ok=True)
    pair_rows = []
    families = {dataset: spec["family"] for dataset, spec in config["datasets"].items()}
    for source in sources:
        if source["task"] != "analyze_sender_receiver_performance":
            continue
        summary = source["summary"]
        dataset = summary["dataset"]
        for cell in summary.get("comparative", {}).get("cells", []):
            pair_rows.append({"dataset": dataset, "family": families[dataset],
                              "sender": cell["sender"], "receiver": cell["receiver"],
                              "accuracy": cell["accuracy"]["estimate"]})
    macro_groups, family_groups = defaultdict(list), defaultdict(list)
    for row in pair_rows:
        macro_groups[(row["sender"], row["receiver"])].append(row["accuracy"])
        family_groups[(row["family"], row["sender"], row["receiver"])].append(row["accuracy"])
    macro = [{"sender": key[0], "receiver": key[1], "accuracy": sum(values) / len(values),
              "dataset_count": len(values)} for key, values in sorted(macro_groups.items())]
    by_family = [{"family": key[0], "sender": key[1], "receiver": key[2],
                  "accuracy": sum(values) / len(values), "dataset_count": len(values)}
                 for key, values in sorted(family_groups.items())]
    combined = {"analyses": sources, "model_pair_macro_average": macro,
                "model_pair_task_family": by_family,
                "macro_weighting": "equal_weight_per_dataset"}
    atomic_write_json(destination / "summaries" / "combined.json", combined)
    if pair_rows:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pq.write_table(pa.Table.from_pylist(pair_rows), destination / "metrics" / "model_pair_dataset_table.parquet")
        labels = [f"{row['sender'].split('-')[-1]}→{row['receiver'].split('-')[-1]}" for row in macro]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar(labels, [row["accuracy"] for row in macro])
        axis.set(ylabel="equal-weight dataset macro accuracy", ylim=(0, 1))
        axis.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(destination / "figures" / "model_pair_macro_accuracy.png", dpi=160)
        plt.close(fig)
    atomic_write_json(destination / "provenance" / "analysis_summaries.json",
                      [{k: x[k] for k in ("task", "path", "sha256")} for x in sources])
    atomic_write_json(destination / "run_manifest.json", {"task": job["task"],
        "cache_only": True, "source_summary_hashes": [x["sha256"] for x in sources]})
    return 0

if __name__ == "__main__":
    raise SystemExit(run(parser(analysis=True).parse_args()))
