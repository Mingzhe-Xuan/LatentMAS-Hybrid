#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.cache import CacheError, CacheLock, DatasetSnapshotStore, STTPlannerContextStore
from analysis.core.config import load_stt_config
from analysis.core.datasets import load_analysis_items
from analysis.core.schemas import (STTPlannerCacheIdentity, build_role_messages,
                                   render_role_prompt, stable_hash)
from analysis.core.stt import collect_stt_planner_item, transport_tokenizer_fingerprint
from analysis.tasks._common import (load_job, load_wrapper, model_args, model_revision,
                                    parser, repository_revision)


def run(args) -> int:
    config = load_stt_config(args.config).raw
    job = load_job(args)
    if job.get("task") != "collect_stt_planner_contexts":
        raise ValueError("job task does not match entry point")
    if (job.get("sender_key") not in config["models"]
            or job.get("sender_model") != config["models"][job["sender_key"]]
            or job.get("split") != config["datasets"].get(job.get("dataset"), {}).get("split")
            or int(job.get("sender_budget", 0)) != int(config["generation"]["sender_budget"])):
        raise ValueError("STT planner job does not match the formal configuration")
    snapshot = load_analysis_items(job["dataset"], job["split"], max_samples=job.get("max_samples"))
    DatasetSnapshotStore(args.cache_root).write(snapshot)
    wrapper = load_wrapper(job["sender_model"], args.device, model_args(job, alignment="identical"))
    messages = [build_role_messages(role="planner", question=item.question,
                                    task=job["dataset"], model_name=job["sender_model"])
                for item in snapshot.items]
    rendered = [render_role_prompt(wrapper, value, job["sender_model"]) for value in messages]
    prompt_set_hash = stable_hash(rendered)
    code_revision = repository_revision()
    identity = STTPlannerCacheIdentity(
        snapshot.dataset, snapshot.split, snapshot.fingerprint, snapshot.selection_policy,
        job["sender_model"], model_revision(wrapper),
        transport_tokenizer_fingerprint(wrapper.tokenizer), prompt_set_hash,
        int(config["generation"]["sender_budget"]), code_revision, False,
    )
    store = STTPlannerContextStore(args.cache_root)
    handle = store.resolve(identity, cache_id=job["effective_cache_id"])
    try:
        store.validate(handle)
        if not args.force:
            return 10
        raise CacheError("--force refuses to overwrite an immutable complete cache")
    except CacheError as exc:
        if (handle.path / "manifest.json").exists() and "incomplete" not in str(exc) \
                and "does not exist" not in str(exc):
            raise
    prompt_records = [
        {"item_id": item.item_id, "role": "planner", "messages": value,
         "messages_hash": stable_hash(value), "rendered": text,
         "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        for item, value, text in zip(snapshot.items, messages, rendered)
    ]
    with CacheLock(handle.path / ".lock"):
        store.initialize(handle, identity, [item.item_id for item in snapshot.items],
                         [dataclasses.asdict(item) for item in snapshot.items], prompt_records)
        by_id = {item.item_id: item for item in snapshot.items}
        for item_id in store.missing_item_ids(handle):
            store.write_item(handle, collect_stt_planner_item(
                by_id[item_id], wrapper, model_id=job["sender_model"],
                sender_budget=int(job["sender_budget"]),
            ))
        store.finalize(handle)
    return 0


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
