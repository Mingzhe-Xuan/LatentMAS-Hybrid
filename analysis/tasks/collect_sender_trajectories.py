#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.cache import CacheError, CacheLock, DatasetSnapshotStore, SenderTrajectoryStore
from analysis.core.config import load_config
from analysis.core.datasets import load_analysis_items
from analysis.core.receiver import tokenizer_mapping_hash
from analysis.core.schemas import (KernelConfig, SenderCacheIdentity, SenderConfig,
                                   build_role_messages, render_role_prompt, stable_hash)
from analysis.core.sender import collect_sender_item
from analysis.tasks._common import load_job, load_wrapper, model_args, model_revision, parser


def run(args) -> int:
    load_config(args.config)
    job = load_job(args)
    if job.get("task") != "collect_sender_trajectories":
        raise ValueError("job task does not match entry point")
    snapshot = load_analysis_items(job["dataset"], job["split"], max_samples=job.get("max_samples"))
    DatasetSnapshotStore(args.cache_root).write(snapshot)
    wrapper = load_wrapper(job["sender_model"], args.device,
                           model_args(job, alignment="kernel"))
    tokenizer_hash = tokenizer_mapping_hash(wrapper.tokenizer)
    prompt_texts = [
        render_role_prompt(wrapper, build_role_messages(role="planner", question=item.question,
                                                        task=job["dataset"], model_name=job["sender_model"]),
                           job["sender_model"])
        for item in snapshot.items
    ]
    prompt_hash = stable_hash(prompt_texts)
    revision = model_revision(wrapper)
    kernel = KernelConfig()
    identity = SenderCacheIdentity(
        snapshot.dataset, snapshot.split, snapshot.fingerprint, snapshot.selection_policy,
        job["sender_model"], revision, tokenizer_hash, prompt_hash, "kernel", kernel,
        int(job["kmax"]), "bfloat16",
    )
    store = SenderTrajectoryStore(args.cache_root)
    handle = store.resolve(identity, cache_id=job["effective_cache_id"])
    try:
        store.validate(handle)
        if not args.force:
            return 10
        raise CacheError("--force refuses to overwrite an immutable complete cache")
    except CacheError as exc:
        if "incomplete" not in str(exc) and "does not exist" not in str(exc):
            if (handle.path / "manifest.json").exists():
                manifest = __import__("json").loads((handle.path / "manifest.json").read_text())
                if manifest.get("complete"):
                    raise
    config = SenderConfig(snapshot.dataset, snapshot.split, job["sender_model"], revision,
                          tokenizer_hash, snapshot.fingerprint, prompt_hash, int(job["kmax"]),
                          kernel=kernel)
    with CacheLock(handle.path / ".lock"):
        store.initialize(handle, identity, [x.item_id for x in snapshot.items])
        store.write_questions(handle, [dataclasses.asdict(x) for x in snapshot.items])
        store.write_prompt_records(handle, [
            {"item_id": item.item_id, "role": "planner", "rendered": rendered,
             "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest()}
            for item, rendered in zip(snapshot.items, prompt_texts)
        ])
        by_id = {x.item_id: x for x in snapshot.items}
        for item_id in store.missing_item_ids(handle):
            store.write_item(handle, collect_sender_item(by_id[item_id], wrapper, config))
        store.finalize(handle)
    return 0


def main() -> int:
    args = parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
