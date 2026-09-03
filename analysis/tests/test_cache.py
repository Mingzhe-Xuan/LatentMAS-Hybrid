from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch

from analysis.core.cache import CacheError, DatasetSnapshotStore, SenderTrajectoryStore
from analysis.core.datasets import snapshot_from_rows
from analysis.core.schemas import KernelConfig, SenderCacheIdentity, SenderItemTrajectory


def identity() -> SenderCacheIdentity:
    return SenderCacheIdentity("aime2024", "train", "data", "all", "model", "rev",
                               "tok", "prompt", "kernel", KernelConfig(), 2, "float32")


def trajectory(item_id: int) -> SenderItemTrajectory:
    return SenderItemTrajectory(item_id, f"q{item_id}", torch.ones(2, 3) * item_id,
                                [.1, .2], "prompt", "hash")


def test_atomic_shards_resume_and_hash_validation(tmp_path) -> None:
    store = SenderTrajectoryStore(tmp_path)
    handle = store.resolve(identity())
    store.initialize(handle, identity(), [0, 1])
    store.write_questions(handle, [{"item_id": 0}, {"item_id": 1}])
    store.write_prompt_records(handle, [
        {"item_id": item_id, "role": "planner", "rendered": "prompt",
         "sha256": hashlib.sha256(b"prompt").hexdigest()} for item_id in (0, 1)
    ])
    store.write_item(handle, trajectory(0))
    assert store.missing_item_ids(handle) == [1]
    store.write_item(handle, trajectory(1))
    store.finalize(handle)
    assert store.validate(handle)["question_count"] == 2
    with (handle.path / "states/item_0001.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(CacheError, match="corrupt"):
        store.validate(handle)


def test_dataset_snapshot_is_content_addressed_and_validated(tmp_path) -> None:
    snapshot = snapshot_from_rows("aime2024", "train", [
        {"question": "q", "solution": "s", "gold": "1"}
    ])
    store = DatasetSnapshotStore(tmp_path)
    path = store.write(snapshot)
    assert store.validate(path, snapshot.fingerprint)["question_count"] == 1
    changed = snapshot_from_rows("aime2024", "train", [
        {"question": "changed", "solution": "s", "gold": "1"}
    ])
    assert changed.fingerprint != snapshot.fingerprint
