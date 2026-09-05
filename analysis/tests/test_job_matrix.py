from __future__ import annotations

from collections import Counter
from pathlib import Path

from analysis.pbs.build_job_matrix import EXPECTED_COUNTS, build_matrices, validate_matrices


CONFIG = Path(__file__).resolve().parents[1] / "configs/kernel_analysis.yaml"


def test_formal_matrix_counts_and_uniqueness() -> None:
    matrices = build_matrices(CONFIG)
    validate_matrices(matrices, formal=True)
    assert {name: len(rows) for name, rows in matrices.items()} == EXPECTED_COUNTS
    assert {row["alignment"] for row in matrices["kernel_scaling.jsonl"] if row["k"]} == {"kernel"}
    assert {row["alignment"] for row in matrices["perturbation.jsonl"]} == {"kernel", "soft", "linear"}
    assert len({row["effective_cache_id"] for name in
                ("kernel_scaling.jsonl", "perturbation.jsonl", "model_pairs.jsonl")
                for row in matrices[name]}) == 297


def test_model_pair_matrix_reuses_primary_cells() -> None:
    matrices = build_matrices(CONFIG)
    primary = {"aime2024", "humanevalplus", "arc_challenge"}
    rows = matrices["model_pairs.jsonl"]
    assert not any(row["dataset"] in primary and row["sender_model"] == row["receiver_model"] == "Qwen/Qwen3-8B" for row in rows)
    assert not any(row["dataset"] in primary and row["k"] == 0 and row["receiver_model"] == "Qwen/Qwen3-8B" for row in rows)
    for dataset in primary:
        assert sum(row["dataset"] == dataset for row in rows) == 12


def test_perturbation_has_three_nonzero_doses_and_clean_controls() -> None:
    rows = build_matrices(CONFIG)["perturbation.jsonl"]
    cell = [row for row in rows if row["dataset"] == "aime2024" and row["generation_seed"] == 42]
    assert len(cell) == 11
    assert Counter(row["alignment"] for row in cell if row["alpha"] > 0) == {"kernel": 3, "soft": 3, "linear": 3}
    assert {row["alpha"] for row in cell} == {0, 0.01, 0.05, 0.10}
    assert {(row["alignment"], row["alpha"]) for row in cell if row["alpha"] == 0} == {("soft", 0), ("linear", 0)}
