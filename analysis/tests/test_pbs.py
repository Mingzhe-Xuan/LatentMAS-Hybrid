from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from analysis.pbs.build_dataset_run_matrix import build_dataset_run_manifests
from analysis.pbs.build_job_matrix import build_matrices, validate_matrices


ROOT = Path(__file__).resolve().parents[2]


def test_smoke_matrix_is_small_and_unique() -> None:
    matrices = build_matrices(ROOT / "analysis/configs/kernel_analysis.yaml",
                              smoke=True, dataset_filter="aime2024")
    validate_matrices(matrices, formal=False)
    assert len(matrices["sender.jsonl"]) == 2
    assert len(matrices["kernel_scaling.jsonl"]) == 9
    assert {row["k"] for row in matrices["kernel_scaling.jsonl"]} == {0, 2, 4}


def test_pbs_scripts_have_valid_shell_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    for script in ("analysis_job.pbs", "submit_analysis.sh", "analysis_dataset_run.pbs",
                   "analysis_finalize.pbs"):
        subprocess.run([bash, "-n", str(ROOT / "analysis/pbs" / script)], check=True)
    for script in ("analysis_job.slurm", "submit_analysis.sh"):
        subprocess.run([bash, "-n", str(ROOT / "analysis/slurm" / script)], check=True)
    subprocess.run([bash, "-n", str(ROOT / "analysis.sh")], check=True)


def test_repository_analysis_entrypoint_submits_two_gpu_array_and_finalizer() -> None:
    text = (ROOT / "analysis.sh").read_text(encoding="utf-8")
    assert 'ANALYSIS_TARGET="${ANALYSIS_TARGET:-all}"' in text
    assert 'ANALYSIS_MAX_GPUS="${ANALYSIS_MAX_GPUS:-2}"' in text
    assert 'case "${ANALYSIS_MAX_GPUS}" in 1|2)' in text
    assert '-J "1-${COMPUTE_ROWS}%${ANALYSIS_MAX_GPUS}"' in text
    assert 'depend=${DEPENDENCY_OPERATOR}:${COMPUTE_JOB}' in text
    assert 'PBS_DEPENDENCY_OPERATOR:-afterokarray' in text
    for worker in ("analysis_dataset_run.pbs", "analysis_finalize.pbs"):
        worker_text = (ROOT / "analysis/pbs" / worker).read_text(encoding="utf-8")
        assert "#PBS -l select=1:ncpus=12:ngpus=1" in worker_text


def test_dataset_run_manifest_has_one_cell_per_dataset_run() -> None:
    kernel_matrices, stt_matrices, compute, finalize = build_dataset_run_manifests()
    kernel = [row for row in compute if row["protocol"] == "kernel"]
    stt = [row for row in compute if row["protocol"] == "stt"]
    assert len(kernel) == 27
    assert len({(row["dataset"], row["seed"]) for row in kernel}) == 27
    assert {row["run"] for row in kernel} == {1, 2, 3}
    assert len(stt) == 3
    assert len({row["dataset"] for row in stt}) == 3
    assert all(row["run"] == 1 and row["deterministic"] for row in stt)
    assert len(finalize) == 1
    assert finalize[0]["targets"] == ["kernel", "stt"]
    assert all(task["cache_mode"] == "cache-only" for task in finalize[0]["tasks"])
    for bundle in kernel:
        for task in bundle["tasks"]:
            row = kernel_matrices[Path(task["matrix"]).name][task["job_index"] - 1]
            assert row["dataset"] == bundle["dataset"]
            if row["task"].startswith("evaluate_"):
                assert row["generation_seed"] == bundle["seed"]
    for bundle in stt:
        for task in bundle["tasks"]:
            row = stt_matrices[Path(task["matrix"]).name][task["job_index"] - 1]
            assert row["dataset"] == bundle["dataset"]


def test_combined_aime_smoke_has_three_kernel_runs_and_one_stt_run() -> None:
    _, _, compute, finalize = build_dataset_run_manifests(
        dataset_filter="aime2024", smoke=True, smoke_samples=4,
    )
    assert [(row["protocol"], row["run"]) for row in compute] == [
        ("kernel", 1), ("kernel", 2), ("kernel", 3), ("stt", 1),
    ]
    assert len(finalize) == 1


def test_worker_contains_required_validation_and_status_contract() -> None:
    text = (ROOT / "analysis/pbs/analysis_job.pbs").read_text(encoding="utf-8")
    assert "analysis/jobs/*" in text and "analysis/configs/*" in text
    assert "STATUS == 10" in text and "SKIPPED" in text and "FAILED" in text
    assert "#PBS -l select=1:ncpus=12:ngpus=1" in text


def test_slurm_worker_preserves_validation_and_status_contract() -> None:
    worker = (ROOT / "analysis/slurm/analysis_job.slurm").read_text(encoding="utf-8")
    submitter = (ROOT / "analysis/slurm/submit_analysis.sh").read_text(encoding="utf-8")
    assert "SLURM_ARRAY_TASK_ID" in worker and "analysis/jobs/*" in worker
    assert "STATUS == 10" in worker and "SKIPPED" in worker and "FAILED" in worker
    assert "--gres" in submitter and "afterok:" in submitter and "SLURM_MAX_RUNNING:-1" in submitter
    assert ".venv/bin/python" in submitter and "ANALYSIS_PYTHON" in submitter
