from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

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
    for script in ("analysis_job.pbs", "submit_analysis.sh"):
        subprocess.run([bash, "-n", str(ROOT / "analysis/pbs" / script)], check=True)
    for script in ("analysis_job.slurm", "submit_analysis.sh"):
        subprocess.run([bash, "-n", str(ROOT / "analysis/slurm" / script)], check=True)
    subprocess.run([bash, "-n", str(ROOT / "analysis.sh")], check=True)


def test_repository_analysis_entrypoint_is_serial_and_pbs_guarded() -> None:
    text = (ROOT / "analysis.sh").read_text(encoding="utf-8")
    assert "#PBS -l select=1:ncpus=12:ngpus=1" in text
    assert 'ANALYSIS_TARGET="${ANALYSIS_TARGET:-all}"' in text
    assert '[[ "${ANALYSIS_TARGET}" == all || "${ANALYSIS_TARGET}" == kernel ]] && run_kernel' in text
    assert '[[ "${ANALYSIS_TARGET}" == all || "${ANALYSIS_TARGET}" == stt ]] && run_stt' in text
    assert text.index('&& run_kernel\n') < text.index('&& run_stt\n')
    assert 'if [[ -z "${PBS_JOBID:-}" ]]' in text
    assert "status == 10" in text and "validated cache hit" in text


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
