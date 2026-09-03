from __future__ import annotations

from collections.abc import Sequence

from analysis.core.schemas import ReceiverItemResult


def summarize_condition_metrics(rows: Sequence[ReceiverItemResult], *,
                                execution_wall_seconds: float) -> dict:
    processed = len(rows)
    correct = sum(bool(row.correct) for row in rows)
    components = {
        "sender_recurrence": sum(row.sender_recurrence_output_tokens for row in rows),
        "transfer_alignment": sum(row.transfer_alignment_output_tokens for row in rows),
        "receiver_decode": sum(row.receiver_decode_output_tokens for row in rows),
    }
    output_total = sum(components.values())
    timing_components = {
        "sender_prefix_seconds": sum(row.sender_prefix_seconds for row in rows),
        "alignment_seconds": sum(row.alignment_seconds for row in rows),
        "receiver_prefill_seconds": sum(row.receiver_prefill_seconds for row in rows),
        "receiver_text_decode_seconds": sum(row.receiver_text_decode_seconds for row in rows),
        "evaluation_seconds": sum(row.evaluation_seconds for row in rows),
    }
    total_seconds = sum(timing_components.values())
    return {
        "results": {"processed": processed, "correct": correct,
                    "accuracy": correct / processed if processed else 0.0,
                    "output_tokens": {"total": output_total,
                                      "average_per_problem": output_total / processed if processed else 0.0,
                                      "components": components}},
        "timing": {"total_seconds": total_seconds,
                   "seconds_per_problem": total_seconds / processed if processed else 0.0,
                   "components": timing_components},
        "execution_wall_seconds": execution_wall_seconds,
    }
