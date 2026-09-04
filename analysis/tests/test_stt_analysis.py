from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from analysis.core.cache import ReceiverEvaluationStore
from analysis.core.schemas import ReceiverItemResult
from analysis.tasks.analyze_bidirectional_stt import run as run_analysis
from analysis.tasks.build_bidirectional_stt_report import run as run_report


CONFIG = "analysis/configs/bidirectional_stt.yaml"


def _row(item_id: int, correct: bool) -> ReceiverItemResult:
    prompt = f"judger-{item_id}"
    return ReceiverItemResult(
        item_id, hashlib.sha256(f"q{item_id}".encode()).hexdigest(), "7" if correct else "8",
        "7" if correct else "8", "7", correct, None, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        {"prompt_text": prompt, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()},
    )


def _write_condition(root: Path, system: str, values: list[bool], *, cache_id: str | None = None) -> None:
    identity = {
        "schema_version": "stt-receiver-v2", "dataset": "aime2024", "split": "train",
        "dataset_fingerprint": "dataset", "selection_policy": "first-1",
        "system": system,
    }
    store = ReceiverEvaluationStore(root, namespace="stt_receiver_evaluations")
    handle = store.resolve(cache_id or f"cache-{system}", identity)
    rows = [_row(index, value) for index, value in enumerate(values)]
    store.write(handle, identity, rows, {"accuracy": sum(values) / len(values)})


def test_stt_cache_only_analysis_and_report(tmp_path: Path) -> None:
    values = {
        "qwen_only": [True, False], "mistral_only": [False, False],
        "qwen_to_mistral": [True, False], "mistral_to_qwen": [True, True],
    }
    for system, correctness in values.items():
        _write_condition(tmp_path / "cache", system, correctness)
    _write_condition(tmp_path / "cache", "qwen_only", [False, False], cache_id="stale-cache")
    analysis_job = {
        "task": "analyze_bidirectional_stt", "effective_cache_id": "analysis-aime",
        "dataset": "aime2024", "split": "train", "cache_only": True,
        "selection_policy": "first-1", "smoke": True,
        "receiver_cache_ids": {system: f"cache-{system}" for system in values},
    }
    args = SimpleNamespace(
        config=CONFIG, job_spec=json.dumps(analysis_job), job_matrix=None, job_index=None,
        cache_root=str(tmp_path / "cache"), result_root=str(tmp_path / "result"),
        device="cpu", cache_only=True, force=False,
    )
    assert run_analysis(args) == 0
    summary_path = tmp_path / "result/analyze_bidirectional_stt/analysis-aime/summaries/summary.json"
    summary = json.loads(summary_path.read_text())
    effects = {row["effect"]: row for row in summary["effects"]}
    assert effects["mistral_to_qwen_minus_qwen_only"]["paired_bootstrap"]["estimate"] == 0.5

    report_job = {
        "task": "build_bidirectional_stt_report", "effective_cache_id": "report-smoke",
        "datasets": ["aime2024"], "cache_only": True, "smoke": True,
        "analysis_cache_ids": {"aime2024": "analysis-aime"},
    }
    args.job_spec = json.dumps(report_job)
    assert run_report(args) == 0
    report = tmp_path / "result/build_bidirectional_stt_report/report-smoke/summaries/report.md"
    rendered = report.read_text(encoding="utf-8")
    assert "mistral_to_qwen_minus_qwen_only" in rendered
    assert "95% paired-bootstrap CI" in rendered
    assert "McNemar p" in rendered
