from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from analysis.core.schemas import (DatasetSnapshot, ReceiverItemResult, SenderCacheIdentity,
                                   SenderItemTrajectory, canonical_json,
                                   stable_hash)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True,
                                         ensure_ascii=False) + "\n").encode("utf-8"))


class CacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheHandle:
    cache_id: str
    path: Path
    identity_hash: str


class CacheLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "CacheLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, str(os.getpid()).encode())
        except FileExistsError as exc:
            raise CacheError(f"cache is already locked: {self.path}") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


class DatasetSnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root) / "datasets"

    def write(self, snapshot: DatasetSnapshot) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.root / f"{snapshot.dataset}-{snapshot.fingerprint[:24]}"
        path.mkdir(parents=True, exist_ok=True)
        table_path = path / "questions.parquet"
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            self.validate(path, snapshot.fingerprint)
            return path
        rows = [dataclasses.asdict(item) for item in snapshot.items]
        fd, temporary = tempfile.mkstemp(prefix=".questions.", suffix=".tmp", dir=path)
        os.close(fd)
        try:
            pq.write_table(pa.Table.from_pylist(rows), temporary)
            os.replace(temporary, table_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        atomic_write_json(manifest_path, {
            "dataset": snapshot.dataset, "split": snapshot.split,
            "selection_policy": snapshot.selection_policy,
            "fingerprint": snapshot.fingerprint, "question_count": len(snapshot.items),
            "questions_sha256": file_sha256(table_path), "complete": True,
        })
        return path

    @staticmethod
    def validate(path: Path, expected_fingerprint: str) -> dict[str, Any]:
        manifest_path, table_path = path / "manifest.json", path / "questions.parquet"
        if not manifest_path.exists() or not table_path.exists():
            raise CacheError("dataset snapshot is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not manifest.get("complete") or manifest.get("fingerprint") != expected_fingerprint
                or file_sha256(table_path) != manifest.get("questions_sha256")):
            raise CacheError("dataset snapshot is corrupt or incompatible")
        return manifest


class SenderTrajectoryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root) / "sender_trajectories"

    def resolve(self, identity: SenderCacheIdentity, *, cache_id: str | None = None) -> CacheHandle:
        selected_id = cache_id or identity.cache_id
        path = self.root / selected_id
        expected_hash = stable_hash(identity)
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("identity_hash") != expected_hash:
                raise CacheError("existing Sender cache has an incompatible identity")
        path.mkdir(parents=True, exist_ok=True)
        (path / "states").mkdir(exist_ok=True)
        return CacheHandle(selected_id, path, expected_hash)

    def initialize(self, handle: CacheHandle, identity: SenderCacheIdentity,
                   expected_item_ids: Iterable[int]) -> None:
        manifest_path = handle.path / "manifest.json"
        if manifest_path.exists():
            return
        atomic_write_json(manifest_path, {
            "cache_id": handle.cache_id, "identity": dataclasses.asdict(identity),
            "identity_hash": handle.identity_hash, "complete": False,
            "expected_item_ids": list(expected_item_ids), "files": {},
            "hidden_semantics": "final_layer_last_position_before_readout_after_feedback",
            "prompt_role": "planner", "prompt_builder": "build_agent_message_sequential_latent_mas",
        })

    def write_questions(self, handle: CacheHandle, rows: list[dict[str, Any]]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = handle.path / "questions.parquet"
        if path.exists():
            return
        fd, temporary = tempfile.mkstemp(prefix=".questions.", suffix=".tmp", dir=handle.path)
        os.close(fd)
        try:
            pq.write_table(pa.Table.from_pylist(rows), temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        manifest_path = handle.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["questions_sha256"] = file_sha256(path)
        atomic_write_json(manifest_path, manifest)

    def write_prompt_records(self, handle: CacheHandle, records: list[dict[str, Any]]) -> None:
        manifest_path = handle.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("prompt_records") not in (None, records):
            raise CacheError("Sender prompt records are incompatible")
        manifest["prompt_records"] = records
        atomic_write_json(manifest_path, manifest)

    @staticmethod
    def _item_path(handle: CacheHandle, item_id: int) -> Path:
        return handle.path / "states" / f"item_{item_id:04d}.safetensors"

    def missing_item_ids(self, handle: CacheHandle) -> list[int]:
        manifest = json.loads((handle.path / "manifest.json").read_text(encoding="utf-8"))
        missing = []
        for item_id in manifest["expected_item_ids"]:
            path = self._item_path(handle, item_id)
            entry = manifest.get("files", {}).get(path.name)
            if not path.exists() or not entry or file_sha256(path) != entry.get("sha256"):
                missing.append(item_id)
        return missing

    def write_item(self, handle: CacheHandle, item: SenderItemTrajectory) -> None:
        from safetensors.torch import save_file

        path = self._item_path(handle, item.item_id)
        tensors = {
            "hidden": item.hidden.detach().cpu().contiguous(),
            "cumulative_sender_seconds": torch.tensor(item.cumulative_sender_seconds, dtype=torch.float64),
        }
        if item.h0 is not None:
            tensors["h0"] = item.h0.detach().cpu().contiguous()
        metadata = {"item_id": str(item.item_id), "question_hash": item.question_hash,
                    "prompt_text": item.prompt_text, "prompt_hash": item.prompt_hash,
                    "hidden_semantics": item.hidden_semantics}
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            save_file(tensors, temporary, metadata=metadata)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        manifest_path = handle.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][path.name] = {
            "sha256": file_sha256(path), "item_id": item.item_id,
            "state_count": int(item.hidden.shape[0]),
            "tensor_schema": {"hidden": list(item.hidden.shape), "dtype": str(item.hidden.dtype)},
            "question_hash": item.question_hash,
        }
        atomic_write_json(manifest_path, manifest)

    def read_item(self, handle: CacheHandle, item_id: int) -> SenderItemTrajectory:
        from safetensors import safe_open
        from safetensors.torch import load_file

        self._validate_item(handle, item_id)
        path = self._item_path(handle, item_id)
        tensors = load_file(path, device="cpu")
        with safe_open(path, framework="pt", device="cpu") as stream:
            metadata = stream.metadata()
        return SenderItemTrajectory(
            int(metadata["item_id"]), metadata["question_hash"], tensors["hidden"],
            tensors["cumulative_sender_seconds"].tolist(), metadata["prompt_text"],
            metadata["prompt_hash"], metadata["hidden_semantics"], tensors.get("h0"),
        )

    def _validate_item(self, handle: CacheHandle, item_id: int) -> None:
        manifest = json.loads((handle.path / "manifest.json").read_text(encoding="utf-8"))
        path = self._item_path(handle, item_id)
        entry = manifest.get("files", {}).get(path.name)
        if not path.exists() or not entry or file_sha256(path) != entry["sha256"]:
            raise CacheError(f"missing or corrupt Sender shard: {item_id}")

    def finalize(self, handle: CacheHandle) -> dict[str, Any]:
        missing = self.missing_item_ids(handle)
        if missing:
            raise CacheError(f"cannot finalize; missing Sender items: {missing}")
        manifest_path = handle.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["complete"] = True
        manifest["question_count"] = len(manifest["expected_item_ids"])
        manifest["state_count"] = sum(x["state_count"] for x in manifest["files"].values())
        atomic_write_json(manifest_path, manifest)
        manifest["manifest_hash"] = file_sha256(manifest_path)
        return manifest

    def validate(self, handle: CacheHandle) -> dict[str, Any]:
        manifest_path = handle.path / "manifest.json"
        if not manifest_path.exists():
            raise CacheError("Sender manifest does not exist")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity_hash") != handle.identity_hash or not manifest.get("complete"):
            raise CacheError("Sender cache is incomplete or incompatible")
        questions = handle.path / "questions.parquet"
        if (not questions.exists() or file_sha256(questions) != manifest.get("questions_sha256")):
            raise CacheError("Sender questions table is missing or corrupt")
        prompts = manifest.get("prompt_records", [])
        if (len(prompts) != len(manifest["expected_item_ids"])
                or any(record.get("role") != "planner"
                       or hashlib.sha256(record.get("rendered", "").encode("utf-8")).hexdigest() != record.get("sha256")
                       for record in prompts)):
            raise CacheError("Sender prompt provenance is missing or corrupt")
        missing = self.missing_item_ids(handle)
        if missing:
            raise CacheError(f"Sender cache has missing/corrupt items: {missing}")
        manifest["manifest_hash"] = file_sha256(manifest_path)
        return manifest


class ReceiverEvaluationStore:
    """One immutable answers table per Receiver condition."""

    def __init__(self, root: str | Path):
        self.root = Path(root) / "receiver_evaluations"

    def resolve(self, cache_id: str, identity_payload: dict[str, Any]) -> CacheHandle:
        identity_hash = stable_hash(identity_payload)
        path = self.root / cache_id
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("identity_hash") != identity_hash:
                raise CacheError("existing Receiver cache has an incompatible identity")
        return CacheHandle(cache_id, path, identity_hash)

    def write(self, handle: CacheHandle, identity_payload: dict[str, Any],
              rows: list[ReceiverItemResult], summary: dict[str, Any]) -> dict[str, Any]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        answer_path = handle.path / "answers.parquet"
        fd, temporary = tempfile.mkstemp(prefix=".answers.", suffix=".tmp", dir=handle.path)
        os.close(fd)
        try:
            pq.write_table(pa.Table.from_pylist([dataclasses.asdict(x) for x in rows]), temporary)
            os.replace(temporary, answer_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        manifest = {
            "cache_id": handle.cache_id, "identity": identity_payload,
            "identity_hash": handle.identity_hash, "complete": True,
            "question_count": len(rows), "answers_sha256": file_sha256(answer_path),
            "summary": summary, "prompt_role": "judger",
            "prompt_builder": "build_agent_message_sequential_latent_mas",
            "prompt_records": [{"item_id": row.item_id, "role": "judger",
                                "rendered": row.diagnostics.get("prompt_text", ""),
                                "sha256": row.diagnostics.get("prompt_hash", "")}
                               for row in rows],
        }
        atomic_write_json(handle.path / "manifest.json", manifest)
        manifest["manifest_hash"] = file_sha256(handle.path / "manifest.json")
        return manifest

    def validate(self, handle: CacheHandle) -> dict[str, Any]:
        manifest_path, answers = handle.path / "manifest.json", handle.path / "answers.parquet"
        if not manifest_path.exists() or not answers.exists():
            raise CacheError("Receiver cache is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not manifest.get("complete") or manifest.get("identity_hash") != handle.identity_hash
                or file_sha256(answers) != manifest.get("answers_sha256")):
            raise CacheError("Receiver cache is corrupt or incompatible")
        prompts = manifest.get("prompt_records", [])
        if (len(prompts) != manifest.get("question_count")
                or any(record.get("role") != "judger"
                       or hashlib.sha256(record.get("rendered", "").encode("utf-8")).hexdigest() != record.get("sha256")
                       for record in prompts)):
            raise CacheError("Receiver prompt provenance is missing or corrupt")
        manifest["manifest_hash"] = file_sha256(manifest_path)
        return manifest
