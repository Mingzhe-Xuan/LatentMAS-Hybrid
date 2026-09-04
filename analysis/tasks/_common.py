from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def repository_bootstrap() -> Path:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def parser(*, analysis: bool = False) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", required=True)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--job-spec")
    source.add_argument("--job-matrix")
    result.add_argument("--job-index", type=int)
    result.add_argument("--cache-root", default="analysis_cache")
    result.add_argument("--result-root", default="analysis_result")
    result.add_argument("--device", default="cuda")
    result.add_argument("--cache-only", action="store_true", default=analysis)
    result.add_argument("--force", action="store_true")
    return result


def load_job(args: argparse.Namespace) -> dict[str, Any]:
    if args.job_spec:
        path = Path(args.job_spec)
        if path.exists():
            job = json.loads(path.read_text(encoding="utf-8"))
        else:
            job = json.loads(args.job_spec)
    else:
        if args.job_index is None or args.job_index < 1:
            raise ValueError("--job-index must be a positive one-based index")
        lines = Path(args.job_matrix).read_text(encoding="utf-8").splitlines()
        if any(not line.strip() for line in lines):
            raise ValueError("blank lines are forbidden in job matrices")
        if args.job_index > len(lines):
            raise ValueError("job index exceeds matrix row count")
        job = json.loads(lines[args.job_index - 1])
    if not isinstance(job, dict):
        raise ValueError("job spec must be a JSON object")
    return job


def model_args(job: dict[str, Any], *, alignment: str) -> SimpleNamespace:
    return SimpleNamespace(
        task=job["dataset"], model_name=job.get("sender_model") or job.get("receiver_model"),
        method="latent_mas", align_method=alignment, latent_space_realign=False,
        kernel_features=2048, kernel_temperature=.6, kernel_seed=101,
        kernel_chunk_size=4096, soft_temperature=.6, soft_chunk_size=32,
        align_ridge=1e-5, trust_remote_code=False, use_second_HF_model=False,
    )


def load_wrapper(model_id: str, device: str, args: SimpleNamespace):
    import torch
    from models import ModelWrapper
    args.model_name = model_id
    return ModelWrapper(model_id, torch.device(device), use_vllm=False, args=args)


def model_revision(wrapper: Any) -> str:
    base = getattr(wrapper, "HF_model", getattr(wrapper, "model", None))
    config = getattr(base, "config", None)
    revision = getattr(config, "_commit_hash", None)
    if not revision:
        revision = getattr(config, "_name_or_path", None)
    if not revision:
        raise ValueError("loaded model does not expose a resolved revision or source path")
    return str(revision)


def repository_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    )
    revision = completed.stdout.strip()
    if len(revision) != 40:
        raise ValueError("repository does not expose a full Git revision")
    return revision
