#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.cache import file_sha256
from analysis.core.stt import transport_tokenizer_fingerprint


def normalize_runtime_tokenizer(tokenizer: Any) -> Any:
    """Mirror models._ensure_pad_token without importing model weights/runtime."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"
    return tokenizer


def _literal_text(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    if not text:
        token = tokenizer.convert_ids_to_tokens(token_id)
        text = "" if token is None else str(token)
    if not text:
        raise ValueError(f"cannot obtain a literal representation for source token {token_id}")
    return text


def complete_csc_source_support(*, indptr: np.ndarray, indices: np.ndarray,
                                data: np.ndarray, source_token_ids: np.ndarray,
                                target_token_ids: np.ndarray, source_tokenizer: Any,
                                target_tokenizer: Any) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    source_vocab_size = len(source_tokenizer)
    target_vocab_size = len(target_tokenizer)
    if len(indptr) != len(source_token_ids) + 1 or indptr[0] != 0 or indptr[-1] != len(data):
        raise ValueError("invalid parent CSC structure")
    if not np.array_equal(source_token_ids, np.sort(source_token_ids)):
        raise ValueError("parent source token IDs must be ordered")
    if len(np.unique(target_token_ids)) != len(target_token_ids):
        raise ValueError("parent target token IDs must be unique")
    target_rows = {int(token_id): row for row, token_id in enumerate(target_token_ids)}
    existing_columns = {int(token_id): column for column, token_id in enumerate(source_token_ids)}
    new_indptr = [0]
    new_indices: list[np.ndarray] = []
    new_data: list[np.ndarray] = []
    completion: list[dict[str, Any]] = []
    for source_id in range(source_vocab_size):
        old_column = existing_columns.get(source_id)
        if old_column is not None:
            start, stop = int(indptr[old_column]), int(indptr[old_column + 1])
            column_indices = indices[start:stop]
            column_data = data[start:stop]
        else:
            literal = _literal_text(source_tokenizer, source_id)
            encoded = target_tokenizer(literal, add_special_tokens=False)["input_ids"]
            if isinstance(encoded, np.ndarray):
                encoded = encoded.tolist()
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            if not encoded:
                raise ValueError(f"target tokenizer produced no fallback tokens for source {source_id}")
            counts: dict[int, int] = {}
            for target_id in encoded:
                target_id = int(target_id)
                if not 0 <= target_id < target_vocab_size or target_id not in target_rows:
                    raise ValueError(f"fallback target token {target_id} is outside active support")
                counts[target_id] = counts.get(target_id, 0) + 1
            ordered = sorted(counts)
            column_indices = np.asarray([target_rows[token_id] for token_id in ordered], dtype=np.int64)
            column_data = np.asarray([counts[token_id] / len(encoded) for token_id in ordered], dtype=np.float64)
            completion.append({"source_token_id": source_id, "literal": literal,
                               "target_token_ids": ordered,
                               "weights": column_data.tolist()})
        new_indices.append(np.asarray(column_indices, dtype=np.int64))
        new_data.append(np.asarray(column_data, dtype=np.float64))
        new_indptr.append(new_indptr[-1] + len(column_data))
    return {
        "indptr": np.asarray(new_indptr, dtype=np.int64),
        "indices": np.concatenate(new_indices),
        "data": np.concatenate(new_data),
        "shape": np.asarray([len(target_token_ids), source_vocab_size], dtype=np.int64),
        "source_token_ids": np.arange(source_vocab_size, dtype=np.int64),
        "target_token_ids": np.asarray(target_token_ids, dtype=np.int64),
    }, completion


def complete_artifact(input_path: Path, output_path: Path, *, source_tokenizer: Any,
                      target_tokenizer: Any, source_revision: str,
                      target_revision: str) -> str:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_path}")
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    source_fingerprint = transport_tokenizer_fingerprint(source_tokenizer)
    target_fingerprint = transport_tokenizer_fingerprint(target_tokenizer)
    parent_source_fingerprint = metadata.get("source_fingerprint")
    parent_target_fingerprint = metadata.get("target_fingerprint")
    provenance_text = json.dumps(metadata, sort_keys=True)
    if source_revision not in provenance_text or target_revision not in provenance_text:
        raise ValueError("parent artifact does not contain both requested model revisions")
    completed, rows = complete_csc_source_support(
        indptr=arrays["indptr"], indices=arrays["indices"], data=arrays["data"],
        source_token_ids=arrays["source_token_ids"], target_token_ids=arrays["target_token_ids"],
        source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer,
    )
    old_columns = {int(token_id): index for index, token_id in enumerate(arrays["source_token_ids"])}
    if "source_marginal" in arrays:
        marginal = np.zeros(len(source_tokenizer), dtype=arrays["source_marginal"].dtype)
        for token_id, column in old_columns.items():
            marginal[token_id] = arrays["source_marginal"][column]
        arrays["source_marginal"] = marginal
    if "candidate_columns" in arrays:
        arrays["candidate_columns"] = arrays["source_token_ids"][arrays["candidate_columns"]]
    added_rows, added_columns, added_evidence, added_sources = [], [], [], []
    target_rows = {int(token_id): row for row, token_id in enumerate(arrays["target_token_ids"])}
    for row in rows:
        for target_id, weight in zip(row["target_token_ids"], row["weights"]):
            added_rows.append(target_rows[target_id])
            added_columns.append(row["source_token_id"])
            added_evidence.append(weight)
            added_sources.append("literal-special-fallback")
    if added_rows and "candidate_rows" in arrays:
        arrays["candidate_rows"] = np.concatenate((arrays["candidate_rows"], np.asarray(added_rows, np.int64)))
        arrays["candidate_columns"] = np.concatenate((arrays["candidate_columns"], np.asarray(added_columns, np.int64)))
        arrays["candidate_evidence"] = np.concatenate((arrays["candidate_evidence"], np.asarray(added_evidence, np.float64)))
        arrays["candidate_sources"] = np.concatenate((arrays["candidate_sources"].astype(str), np.asarray(added_sources)))
    if rows:
        metadata["build_config"]["support_policy"]["source"] = "full-vocabulary-with-literal-special-fallback"
    derivation = metadata.get("derivation")
    if isinstance(derivation, dict):
        derivation["fingerprint_validation"] = "runtime-normalized"
    metadata["source_fingerprint"] = source_fingerprint
    metadata["target_fingerprint"] = target_fingerprint
    metadata["runtime_tokenizer_validation"] = {
        "scheme": "analysis-tokenizer-mapping-plus-special-ids-sha256-v1",
        "source_revision": source_revision, "target_revision": target_revision,
        "requested_revisions": {"source": source_revision, "target": target_revision},
        "source_fingerprint": source_fingerprint, "target_fingerprint": target_fingerprint,
        "parent_source_fingerprint": parent_source_fingerprint,
        "parent_target_fingerprint": parent_target_fingerprint,
    }
    metadata["source_support_completion"] = {
        "method": "target-tokenized-literal-special-fallback-v1",
        "parent_sha256": file_sha256(input_path), "added_columns": rows,
        "code_revision": subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                        capture_output=True, text=True).stdout.strip(),
    }
    arrays.update(completed)
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return file_sha256(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-revision", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    source = AutoTokenizer.from_pretrained(args.source_model, revision=args.source_revision,
                                           use_fast=True, local_files_only=False,
                                           fix_mistral_regex=False)
    target = AutoTokenizer.from_pretrained(args.target_model, revision=args.target_revision,
                                           use_fast=True, local_files_only=False,
                                           fix_mistral_regex=False)
    normalize_runtime_tokenizer(source)
    normalize_runtime_tokenizer(target)
    digest = complete_artifact(Path(args.input), Path(args.output),
                               source_tokenizer=source, target_tokenizer=target,
                               source_revision=args.source_revision,
                               target_revision=args.target_revision)
    print(json.dumps({"output": args.output, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
