from __future__ import annotations

import time
from dataclasses import dataclass

from utils import extract_gsm8k_answer, extract_markdown_python_block, normalize_answer, run_with_timeout


@dataclass(frozen=True)
class EvaluationResult:
    prediction: str | None
    correct: bool
    error: str | None
    seconds: float


def evaluate_answer(task: str, text: str, gold: str, *, timeout: int = 10) -> EvaluationResult:
    started = time.perf_counter()
    error = None
    if task in {"humanevalplus", "mbppplus"}:
        prediction = extract_markdown_python_block(text)
        if prediction is None:
            correct, error = False, "python error: No python code block found"
        else:
            try:
                correct, error = run_with_timeout(prediction + "\n" + gold, timeout=timeout)
            except Exception as exc:
                # Infrastructure/execution failures are question-level failures,
                # never grounds for silently dropping a benchmark row.
                correct, error = False, f"python executor error: {type(exc).__name__}: {exc}"
    else:
        prediction = normalize_answer(extract_gsm8k_answer(text))
        normalized_gold = normalize_answer(str(gold))
        if task in {"aime2024", "aime2025"}:
            try:
                correct = int(prediction) == int(normalized_gold)
            except (TypeError, ValueError):
                correct = False
                error = f"Value error in parsing answer. Pred: {prediction}, Gold: {normalized_gold}"
        else:
            correct = bool(prediction and normalized_gold and prediction == normalized_gold)
    return EvaluationResult(prediction, bool(correct), error, time.perf_counter() - started)
