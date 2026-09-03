from __future__ import annotations

import json
import time
from pathlib import Path

from analysis.core.artifacts import summarize_condition_metrics
from analysis.core.cache import (CacheError, CacheHandle, CacheLock, DatasetSnapshotStore,
                                 ReceiverEvaluationStore, SenderTrajectoryStore,
                                 file_sha256)
from analysis.core.config import load_config
from analysis.core.datasets import load_analysis_items
from analysis.core.receiver import evaluate_receiver_batch, tokenizer_mapping_hash
from analysis.core.schemas import (KernelConfig, ReceiverCondition, build_role_messages,
                                   render_role_prompt, stable_hash)
from analysis.tasks._common import load_job, load_wrapper, model_args, model_revision


def _sender_handle(cache_root: str, cache_id: str) -> tuple[CacheHandle, dict]:
    path = Path(cache_root) / "sender_trajectories" / cache_id
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise CacheError(f"required Sender cache is missing: {cache_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return CacheHandle(cache_id, path, manifest["identity_hash"]), manifest


def run_evaluation(args, expected_task: str) -> int:
    config_file = load_config(args.config).raw
    job = load_job(args)
    if job.get("task") != expected_task:
        raise ValueError("job task does not match entry point")
    snapshot = load_analysis_items(job["dataset"], job["split"], max_samples=job.get("max_samples"))
    DatasetSnapshotStore(args.cache_root).write(snapshot)
    receiver = load_wrapper(job["receiver_model"], args.device,
                            model_args(job, alignment=job["alignment"]))
    sender_wrapper = None
    sender_store = SenderTrajectoryStore(args.cache_root)
    sender_handle = None
    sender_manifest_hash = "receiver-only"
    if job["k"]:
        sender_handle, sender_manifest = _sender_handle(args.cache_root, job["sender_cache_id"])
        sender_store.validate(sender_handle)
        sender_manifest_hash = file_sha256(sender_handle.path / "manifest.json")
        if job["sender_model"] == job["receiver_model"]:
            sender_wrapper = receiver
        else:
            sender_wrapper = load_wrapper(job["sender_model"], args.device,
                                          model_args(job, alignment=job["alignment"]))
    prompt_hash = stable_hash([
        render_role_prompt(receiver, build_role_messages(role="judger", question=item.question,
                                                         task=job["dataset"], model_name=job["receiver_model"]),
                           job["receiver_model"])
        for item in snapshot.items
    ])
    max_tokens = int(job["max_new_tokens"])
    generation_batch_size = int(job["generation_batch_size"])
    condition = ReceiverCondition(
        job["dataset"], job["split"], sender_manifest_hash,
        job.get("sender_model") or "receiver-only", job["receiver_model"], model_revision(receiver),
        tokenizer_mapping_hash(receiver.tokenizer), prompt_hash, int(job["k"]),
        job["alignment"], KernelConfig(), float(job.get("alpha", 0)),
        generation_seed=int(job["generation_seed"]), max_new_tokens=max_tokens,
        generation_batch_size=generation_batch_size,
        temperature=float(config_file["generation"]["temperature"]),
        top_p=float(config_file["generation"]["top_p"]),
        dataset_fingerprint=snapshot.fingerprint,
        selection_policy=snapshot.selection_policy,
    )
    store = ReceiverEvaluationStore(args.cache_root)
    handle = store.resolve(job["effective_cache_id"], condition.identity_payload)
    try:
        store.validate(handle)
        if not args.force:
            return 10
        raise CacheError("--force refuses to overwrite an immutable complete cache")
    except CacheError as exc:
        if (handle.path / "manifest.json").exists() and "incomplete" not in str(exc):
            raise
    wall_started = time.perf_counter()
    rows = []
    with CacheLock(handle.path / ".lock"):
        for start in range(0, len(snapshot.items), generation_batch_size):
            batch = list(snapshot.items[start:start + generation_batch_size])
            trajectories = [sender_store.read_item(sender_handle, item.item_id)
                            if sender_handle else None for item in batch]
            rows.extend(evaluate_receiver_batch(batch, trajectories, condition, receiver,
                                                sender_model=sender_wrapper))
        summary = summarize_condition_metrics(rows,
                                              execution_wall_seconds=time.perf_counter() - wall_started)
        store.write(handle, condition.identity_payload, rows, summary)
    return 0
