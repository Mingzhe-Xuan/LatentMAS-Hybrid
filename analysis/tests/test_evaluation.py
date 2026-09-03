from __future__ import annotations

from analysis.core.evaluation import evaluate_answer


def test_humaneval_failures_are_retained_as_incorrect_rows() -> None:
    missing = evaluate_answer("humanevalplus", "no code here", "assert candidate(1) == 1")
    assert not missing.correct
    assert missing.error == "python error: No python code block found"

    runtime = evaluate_answer("humanevalplus", "```python\ndef candidate(x):\n    raise RuntimeError('boom')\n```",
                              "assert candidate(1) == 1", timeout=1)
    assert not runtime.correct
    assert runtime.error
