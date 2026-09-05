#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


ALLOWED_TASKS = {
    "collect_sender_trajectories", "evaluate_kernel_scaling",
    "evaluate_perturbation_stability", "evaluate_sender_receiver_performance",
    "analyze_logit_entropy", "analyze_kernel_scaling",
    "analyze_aligned_state_variance", "analyze_perturbation_stability",
    "analyze_sender_receiver_performance", "build_kernel_analysis_report",
    "collect_stt_planner_contexts", "evaluate_bidirectional_stt",
    "analyze_bidirectional_stt", "build_bidirectional_stt_report",
}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_bundle(manifest: Path, index: int, root: Path) -> dict:
    manifest = manifest.resolve(strict=True)
    jobs_root = (root / "analysis/jobs").resolve(strict=True)
    if not _inside(manifest, jobs_root):
        raise ValueError("bundle manifest must be under analysis/jobs")
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("bundle manifest is empty or contains blank lines")
    if not 1 <= index <= len(lines):
        raise ValueError(f"bundle index {index} is outside 1..{len(lines)}")
    bundle = json.loads(lines[index - 1])
    if not isinstance(bundle.get("tasks"), list) or not bundle["tasks"]:
        raise ValueError("bundle must contain at least one task")
    for task in bundle["tasks"]:
        if task.get("task") not in ALLOWED_TASKS:
            raise ValueError(f"unsupported task: {task.get('task')}")
        if task.get("cache_mode") not in {"reuse", "cache-only"}:
            raise ValueError("invalid bundle cache mode")
        matrix = (root / task["matrix"]).resolve(strict=True)
        config = (root / task["config"]).resolve(strict=True)
        if not _inside(matrix, jobs_root):
            raise ValueError("task matrix must be under analysis/jobs")
        if not _inside(config, (root / "analysis/configs").resolve(strict=True)):
            raise ValueError("task config must be under analysis/configs")
        if not isinstance(task.get("job_index"), int) or task["job_index"] <= 0:
            raise ValueError("task job_index must be positive")
        matrix_lines = matrix.read_text(encoding="utf-8").splitlines()
        if any(not line for line in matrix_lines) or task["job_index"] > len(matrix_lines):
            raise ValueError("task job_index exceeds a valid nonblank matrix")
        matrix_row = json.loads(matrix_lines[task["job_index"] - 1])
        if matrix_row.get("task") != task["task"]:
            raise ValueError("bundle task does not match its referenced matrix row")
    return bundle


@contextmanager
def exclusive_lock(root: Path, key: str | None) -> Iterator[None]:
    if not key:
        yield
        return
    if not key.replace("-", "").isalnum():
        raise ValueError("invalid exclusive lock key")
    directory = root / "bundle_locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{key}.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_bundle(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve(strict=True)
    bundle = load_bundle(Path(args.manifest), args.index, root)
    extra = shlex.split(os.environ.get("ANALYSIS_EXTRA_ARGS", ""))
    print(json.dumps({key: value for key, value in bundle.items() if key != "tasks"},
                     sort_keys=True), flush=True)
    for ordinal, task in enumerate(bundle["tasks"], 1):
        command = [
            sys.executable, str(root / "analysis/tasks" / f"{task['task']}.py"),
            "--config", str(root / task["config"]),
            "--job-matrix", str(root / task["matrix"]),
            "--job-index", str(task["job_index"]),
            "--cache-root", args.cache_root,
            "--result-root", args.result_root,
            "--device", args.device,
        ]
        if task["cache_mode"] == "cache-only":
            command.append("--cache-only")
        command.extend(extra)
        print(f"START {ordinal}/{len(bundle['tasks'])}: {task['task']} "
              f"row={task['job_index']}", flush=True)
        with exclusive_lock(Path(args.state_root), task.get("exclusive_key")):
            status = subprocess.run(command, cwd=root).returncode
        if status == 10:
            print(f"SKIP  {ordinal}/{len(bundle['tasks'])}: validated cache hit", flush=True)
        elif status == 0:
            print(f"DONE  {ordinal}/{len(bundle['tasks'])}: {task['task']}", flush=True)
        else:
            print(f"FAIL  {ordinal}/{len(bundle['tasks'])}: {task['task']} exit={status}",
                  file=sys.stderr, flush=True)
            return status
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--cache-root", default="analysis_cache")
    parser.add_argument("--result-root", default="analysis_result")
    parser.add_argument("--state-root", default="state/analysis")
    parser.add_argument("--device", default="cuda")
    return run_bundle(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
