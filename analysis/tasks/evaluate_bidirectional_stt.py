#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.artifacts import summarize_condition_metrics
from analysis.core.cache import (CacheError, CacheHandle, CacheLock, DatasetSnapshotStore,
                                 ReceiverEvaluationStore, STTPlannerContextStore, file_sha256)
from analysis.core.config import load_stt_config
from analysis.core.datasets import load_analysis_items
from analysis.core.schemas import (STTReceiverCondition, build_role_messages,
                                   render_role_prompt, stable_hash)
from analysis.core.stt import (STTArtifactSpec, evaluate_stt_item, load_stt_artifact,
                               transport_tokenizer_fingerprint)
from analysis.tasks._common import (load_job, load_wrapper, model_args, model_revision,
                                    parser, repository_revision)


def _planner_handle(cache_root: str, cache_id: str) -> tuple[CacheHandle, dict]:
    path = Path(cache_root) / "stt_planner_contexts" / cache_id
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise CacheError(f"required STT planner cache is missing: {cache_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return CacheHandle(cache_id, path, manifest["identity_hash"]), manifest


def run(args) -> int:
    config = load_stt_config(args.config).raw
    job = load_job(args)
    if job.get("task") != "evaluate_bidirectional_stt":
        raise ValueError("job task does not match entry point")
    if (job.get("system") not in config["systems"]
            or job.get("split") != config["datasets"].get(job.get("dataset"), {}).get("split")
            or int(job.get("max_new_tokens", 0))
            != int(config["datasets"].get(job.get("dataset"), {}).get("max_new_tokens", -1))
            or float(job.get("tau", -1)) != float(config["transport"]["tau"])
            or bool(job.get("causal_shift"))):
        raise ValueError("STT evaluation job does not match the formal configuration")
    expected_receiver_key = "qwen" if job["system"] in {"qwen_only", "mistral_to_qwen"} else "mistral"
    expected_sender_key = job["system"].split("_to_", 1)[0] if "_to_" in job["system"] else None
    if (job.get("receiver_key") != expected_receiver_key
            or job.get("receiver_model") != config["models"][expected_receiver_key]
            or job.get("receiver_revision") != config["model_revisions"][expected_receiver_key]
            or job.get("sender_key") != expected_sender_key
            or job.get("sender_model") != (config["models"][expected_sender_key]
                                            if expected_sender_key else None)
            or job.get("sender_revision") != (config["model_revisions"][expected_sender_key]
                                               if expected_sender_key else None)):
        raise ValueError("STT system/model direction mismatch")
    if ((expected_sender_key is None and (job.get("planner_cache_id") is not None
                                          or job.get("artifact") is not None))
            or (expected_sender_key is not None and (not job.get("planner_cache_id")
                                                     or not isinstance(job.get("artifact"), dict)))):
        raise ValueError("STT planner/artifact dependencies do not match the selected system")
    snapshot = load_analysis_items(job["dataset"], job["split"], max_samples=job.get("max_samples"))
    DatasetSnapshotStore(args.cache_root).write(snapshot)
    receiver = load_wrapper(
        job["receiver_model"], args.device,
        model_args(job, alignment="identical", revision=job["receiver_revision"]),
    )
    receiver_revision = model_revision(receiver)
    if receiver_revision != job["receiver_revision"]:
        raise ValueError("loaded receiver model revision does not match the formal configuration")
    receiver_fingerprint = transport_tokenizer_fingerprint(receiver.tokenizer)
    prompt_texts = [
        render_role_prompt(receiver, build_role_messages(
            role="judger", question=item.question, task=job["dataset"],
            model_name=job["receiver_model"]), job["receiver_model"])
        for item in snapshot.items
    ]
    receiver_prompt_hash = stable_hash(prompt_texts)

    sender = None
    planner_store = STTPlannerContextStore(args.cache_root)
    planner_handle = None
    sender_manifest_hash = "receiver-only"
    sender_revision = "receiver-only"
    sender_fingerprint = "receiver-only"
    artifact = None
    artifact_sha = "none"
    artifact_source_fingerprint = "none"
    artifact_target_fingerprint = "none"
    artifact_source_name = "none"
    artifact_target_name = "none"
    code_revision = repository_revision()
    if job["sender_model"]:
        planner_handle, planner_manifest = _planner_handle(args.cache_root, job["planner_cache_id"])
        planner_store.validate(planner_handle)
        sender_manifest_hash = file_sha256(planner_handle.path / "manifest.json")
        sender = load_wrapper(
            job["sender_model"], args.device,
            model_args(job, alignment="identical", revision=job["sender_revision"]),
        )
        sender_revision = model_revision(sender)
        sender_fingerprint = transport_tokenizer_fingerprint(sender.tokenizer)
        declaration = job["artifact"]
        if declaration != config["transport"]["artifacts"].get(job["system"]):
            raise ValueError("STT job artifact does not match the formal directed declaration")
        if sender_revision != declaration["source_revision"] or receiver_revision != declaration["target_revision"]:
            raise ValueError("loaded model revisions do not match the directed transport declaration")
        planner_identity = planner_manifest.get("identity", {})
        expected_planner_identity = {
            "dataset": snapshot.dataset, "split": snapshot.split,
            "dataset_fingerprint": snapshot.fingerprint,
            "selection_policy": snapshot.selection_policy,
            "model_id": job["sender_model"], "model_revision": sender_revision,
            "tokenizer_fingerprint": sender_fingerprint,
            "sender_budget": int(config["generation"]["sender_budget"]),
            "do_sample": False, "code_revision": code_revision,
        }
        mismatch = {key: (planner_identity.get(key), value)
                    for key, value in expected_planner_identity.items()
                    if planner_identity.get(key) != value}
        if mismatch:
            raise CacheError(f"STT planner cache does not match evaluation job: {mismatch}")
        spec = STTArtifactSpec(
            Path(declaration["path"]), declaration["sha256"], declaration["source"],
            declaration["target"], declaration["source_revision"], declaration["target_revision"],
        )
        artifact = load_stt_artifact(
            spec, source_vocab_size=len(sender.tokenizer), target_vocab_size=len(receiver.tokenizer),
            source_fingerprint=sender_fingerprint, target_fingerprint=receiver_fingerprint,
        )
        artifact_sha = declaration["sha256"]
        artifact_source_fingerprint = artifact.source_fingerprint
        artifact_target_fingerprint = artifact.target_fingerprint
        artifact_source_name = declaration["source"]
        artifact_target_name = declaration["target"]

    condition = STTReceiverCondition(
        dataset=snapshot.dataset, split=snapshot.split,
        dataset_fingerprint=snapshot.fingerprint, selection_policy=snapshot.selection_policy,
        system=job["system"], receiver_model_id=job["receiver_model"],
        receiver_revision=receiver_revision,
        receiver_tokenizer_fingerprint=receiver_fingerprint,
        receiver_prompt_hash=receiver_prompt_hash, max_new_tokens=int(job["max_new_tokens"]),
        sender_manifest_hash=sender_manifest_hash,
        sender_model_id=job.get("sender_model") or "receiver-only",
        sender_revision=sender_revision, sender_tokenizer_fingerprint=sender_fingerprint,
        artifact_sha256=artifact_sha, artifact_source_fingerprint=artifact_source_fingerprint,
        artifact_target_fingerprint=artifact_target_fingerprint,
        artifact_source_name=artifact_source_name, artifact_target_name=artifact_target_name,
        tau=float(job["tau"]), sender_budget=int(config["generation"]["sender_budget"]),
        position_chunk_size=int(config["transport"]["position_chunk_size"]),
        target_chunk_size=int(config["transport"]["target_chunk_size"]),
        code_revision=code_revision,
    )
    store = ReceiverEvaluationStore(args.cache_root, namespace="stt_receiver_evaluations")
    handle = store.resolve(job["effective_cache_id"], condition.identity_payload)
    try:
        store.validate(handle)
        if not args.force:
            return 10
        raise CacheError("--force refuses to overwrite an immutable complete cache")
    except CacheError as exc:
        if (handle.path / "manifest.json").exists() and "incomplete" not in str(exc):
            raise
    rows = []
    wall_started = time.perf_counter()
    with CacheLock(handle.path / ".lock"):
        for item in snapshot.items:
            planner = planner_store.read_item(planner_handle, item.item_id) if planner_handle else None
            if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            row = evaluate_stt_item(
                item, receiver, receiver_model_id=job["receiver_model"],
                max_new_tokens=int(job["max_new_tokens"]), planner=planner,
                sender=sender, artifact=artifact, tau=float(job["tau"]),
                position_chunk_size=int(config["transport"]["position_chunk_size"]),
                target_chunk_size=int(config["transport"]["target_chunk_size"]),
            )
            row.diagnostics["peak_gpu_memory_bytes"] = (
                int(torch.cuda.max_memory_allocated(torch.device(args.device)))
                if torch.cuda.is_available() and torch.device(args.device).type == "cuda" else 0
            )
            row.diagnostics["planner_cache_id"] = job.get("planner_cache_id")
            row.diagnostics["receiver_cache_id"] = job["effective_cache_id"]
            rows.append(row)
        summary = summarize_condition_metrics(rows,
                                              execution_wall_seconds=time.perf_counter() - wall_started)
        store.write(handle, condition.identity_payload, rows, summary)
    return 0


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
