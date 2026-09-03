from __future__ import annotations

from analysis.core.analysis import model_pair_summary, scaling_summary, stability_summary


M8 = "Qwen/Qwen3-8B"
M14 = "Qwen/Qwen3-14B"


def row(item, seed, correct, *, k, alignment="kernel", alpha=0, sender=M8,
        receiver=M8, prediction=None):
    return {"item_id": item, "correct": correct,
            "raw_prediction": prediction or str(correct),
            "sender_prefix_seconds": k / 100, "alignment_seconds": .01,
            "receiver_prefill_seconds": .02, "receiver_text_decode_seconds": .03,
            "evaluation_seconds": .001, "sender_recurrence_output_tokens": 0,
            "transfer_alignment_output_tokens": k if alignment == "soft" else 0,
            "receiver_decode_output_tokens": 2, "diagnostics": {},
            "condition": {"k": k, "alignment": alignment, "alpha": alpha,
                          "sender_model_id": sender, "receiver_model_id": receiver,
                          "generation_seed": seed}}


def test_scaling_reports_adjacent_changes_and_plateau() -> None:
    rows = [row(item, 42, k >= 10, k=k) for item in range(3) for k in (0, 10, 20)]
    summary = scaling_summary(rows)
    assert len(summary["adjacent_paired_changes"]) == 2
    assert summary["smallest_k_in_best_confidence_region"] == 10


def test_stability_reports_flip_and_difference_in_differences() -> None:
    rows = []
    for method in ("kernel", "soft", "linear"):
        rows += [row(i, 42, True, k=40, alignment=method, alpha=0, prediction="clean") for i in range(2)]
        rows += [row(i, 42, method != "kernel", k=40, alignment=method, alpha=.1,
                     prediction="noisy") for i in range(2)]
    summary = stability_summary(rows)
    assert len(summary["difference_in_differences"]) == 2
    assert all(cell["answer_flip_rate"] == 1 for cell in summary["cells"] if cell["alpha"])


def test_model_pairs_report_capacity_and_interaction_contrasts() -> None:
    rows = []
    for receiver in (M8, M14):
        rows.append(row(0, 42, False, k=0, sender="receiver-only", receiver=receiver))
        for sender in (M8, M14):
            rows.append(row(0, 42, sender == M14 or receiver == M14, k=40,
                            sender=sender, receiver=receiver))
    summary = model_pair_summary(rows)
    names = {value["contrast"] for value in summary["capacity_contrasts"]}
    assert "sender_by_receiver_interaction" in names
    assert "heterogeneous_14B_to_8B_vs_same_size_8B" in names
    assert len(summary["paired_gain_over_receiver_only"]) == 4
