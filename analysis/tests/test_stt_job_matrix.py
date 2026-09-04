from __future__ import annotations

import json

from analysis.pbs.build_stt_job_matrix import (FORMAL_COUNTS, build_stt_matrices,
                                                validate_stt_matrices)


CONFIG = "analysis/configs/bidirectional_stt.yaml"


def test_formal_stt_matrix_counts_and_systems() -> None:
    matrices = build_stt_matrices(CONFIG)
    validate_stt_matrices(matrices, formal=True)
    assert {name: len(rows) for name, rows in matrices.items()} == FORMAL_COUNTS
    evaluation = matrices["stt_evaluation.jsonl"]
    assert {row["system"] for row in evaluation} == {
        "qwen_only", "mistral_only", "qwen_to_mistral", "mistral_to_qwen"
    }
    assert len({row["effective_cache_id"] for rows in matrices.values() for row in rows}) == 22
    assert all(len(row["sender_revision"]) == 40 for row in matrices["stt_planner.jsonl"])
    assert all(len(row["receiver_revision"]) == 40 for row in evaluation)
    assert all(row["sender_revision"] is None or len(row["sender_revision"]) == 40
               for row in evaluation)
    evaluations = {(row["dataset"], row["system"]): row["effective_cache_id"]
                   for row in evaluation}
    for row in matrices["stt_analysis.jsonl"]:
        assert row["receiver_cache_ids"] == {
            system: evaluations[(row["dataset"], system)]
            for system in ("qwen_only", "mistral_only", "qwen_to_mistral", "mistral_to_qwen")
        }
    assert matrices["stt_report.jsonl"][0]["analysis_cache_ids"] == {
        row["dataset"]: row["effective_cache_id"] for row in matrices["stt_analysis.jsonl"]
    }


def test_stt_smoke_matrix_uses_first_one_and_directional_artifacts() -> None:
    matrices = build_stt_matrices(CONFIG, smoke=True, dataset_filter="aime2024")
    validate_stt_matrices(matrices, formal=False)
    assert {name: len(rows) for name, rows in matrices.items()} == {
        "stt_planner.jsonl": 2, "stt_evaluation.jsonl": 4,
        "stt_analysis.jsonl": 1, "stt_report.jsonl": 1,
    }
    cross = [row for row in matrices["stt_evaluation.jsonl"] if row["sender_model"]]
    assert {row["system"] for row in cross} == {"qwen_to_mistral", "mistral_to_qwen"}
    assert all(row["artifact"]["source"] == row["sender_key"] for row in cross)
    assert all(row["max_samples"] == 1 for row in cross)
    assert matrices["stt_report.jsonl"][0]["selection_policy"] == "first-1"


def test_stt_integration_smoke_uses_first_four_for_every_dataset() -> None:
    matrices = build_stt_matrices(CONFIG, smoke=True, smoke_samples=4)
    validate_stt_matrices(matrices, formal=False)
    assert {name: len(rows) for name, rows in matrices.items()} == FORMAL_COUNTS
    assert all(row["max_samples"] == 4 for row in matrices["stt_planner.jsonl"])
    assert all(row["max_samples"] == 4 for row in matrices["stt_evaluation.jsonl"])
    assert all(row["selection_policy"] == "first-4" for row in matrices["stt_analysis.jsonl"])
    assert matrices["stt_report.jsonl"][0]["selection_policy"] == "first-4"


def test_stt_matrix_rows_are_json_serializable_without_blank_lines() -> None:
    matrices = build_stt_matrices(CONFIG)
    for rows in matrices.values():
        rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        assert all(line for line in rendered.splitlines())
