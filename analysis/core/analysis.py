from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from analysis.core.statistics import nested_question_seed_bootstrap, paired_question_bootstrap


def _key(row: dict) -> tuple[int, int]:
    return int(row["item_id"]), int(row["condition"]["generation_seed"])


def _logical_seconds(row: dict) -> float:
    return sum(float(row.get(name, 0.0)) for name in (
        "sender_prefix_seconds", "alignment_seconds", "receiver_prefill_seconds",
        "receiver_text_decode_seconds", "evaluation_seconds"))


def _output_tokens(row: dict) -> int:
    return sum(int(row.get(name, 0)) for name in (
        "sender_recurrence_output_tokens", "transfer_alignment_output_tokens",
        "receiver_decode_output_tokens"))


def scaling_summary(rows: Iterable[dict]) -> dict[str, Any]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["condition"]["k"])].append(row)
    cells = []
    for k, values in sorted(groups.items()):
        ci = nested_question_seed_bootstrap([
            {"item_id": r["item_id"], "seed": r["condition"]["generation_seed"],
             "value": float(r["correct"])} for r in values], seed=101)
        mean_seconds = float(np.mean([_logical_seconds(r) for r in values]))
        mean_tokens = float(np.mean([_output_tokens(r) for r in values]))
        cells.append({"k": k, "accuracy": ci, "seconds_per_problem": mean_seconds,
                      "output_tokens_per_problem": mean_tokens,
                      "accuracy_per_second": ci["estimate"] / mean_seconds if mean_seconds else None,
                      "accuracy_per_latent_state": ci["estimate"] / k if k else None})
    adjacent = []
    for left, right in zip(sorted(groups), sorted(groups)[1:]):
        a, b = {_key(r): float(r["correct"]) for r in groups[right]}, {_key(r): float(r["correct"]) for r in groups[left]}
        keys = sorted(set(a) & set(b))
        adjacent.append({"from_k": left, "to_k": right,
                         **paired_question_bootstrap([a[x] for x in keys], [b[x] for x in keys], seed=101)})
    best = max(cells, key=lambda x: x["accuracy"]["estimate"])
    eligible = [x["k"] for x in cells if x["accuracy"]["ci_high"] >= best["accuracy"]["ci_low"]]
    return {"cells": cells, "adjacent_paired_changes": adjacent,
            "best_observed_k": best["k"], "smallest_k_in_best_confidence_region": min(eligible)}


def stability_summary(rows: Iterable[dict]) -> dict[str, Any]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        c = row["condition"]
        groups[(c["alignment"], float(c.get("alpha", 0)))].append(row)
    degradation: dict[tuple[str, float], dict[tuple[int, int], float]] = {}
    cells = []
    for (method, alpha), values in sorted(groups.items()):
        baseline = {_key(r): r for r in groups.get((method, 0.0), [])}
        current = {_key(r): r for r in values}
        keys = sorted(set(baseline) & set(current))
        if not keys:
            continue
        clean = [float(baseline[k]["correct"]) for k in keys]
        noisy = [float(current[k]["correct"]) for k in keys]
        degradation[(method, alpha)] = {k: noisy[i] - clean[i] for i, k in enumerate(keys)}
        paired = paired_question_bootstrap(noisy, clean, seed=101)
        flips = np.mean([current[k]["raw_prediction"] != baseline[k]["raw_prediction"] for k in keys])
        perturbations = [p for r in values for p in r.get("diagnostics", {}).get("perturbations", [])]
        cells.append({"alignment": method, "alpha": alpha,
                      "accuracy_degradation": paired, "answer_flip_rate": float(flips),
                      "achieved_relative_noise_norm": float(np.mean([p["relative_noise_norm"] for p in perturbations])) if perturbations else 0.0,
                      "source_perturbed_cosine": float(np.mean([p["original_perturbed_cosine"] for p in perturbations])) if perturbations else 1.0})
        if values:
            cells[-1].update(
                aligned_relative_l2_change=float(np.mean([r.get("diagnostics", {}).get("aligned_relative_l2_change", 0.0) for r in values])),
                aligned_cosine=float(np.mean([r.get("diagnostics", {}).get("aligned_cosine", 1.0) for r in values])),
                amplification_ratio=float(np.mean([r.get("diagnostics", {}).get("amplification_ratio", 0.0) for r in values])),
                probe_unembedding_evaluations=sum(int(r.get("diagnostics", {}).get("probe_unembedding_evaluations", 0)) for r in values),
            )
    did = []
    for alpha in sorted({alpha for _, alpha in degradation if alpha > 0}):
        for control in ("soft", "linear"):
            kernel, other = degradation.get(("kernel", alpha), {}), degradation.get((control, alpha), {})
            keys = sorted(set(kernel) & set(other))
            if keys:
                did.append({"alpha": alpha, "contrast": f"kernel-minus-{control}",
                            **paired_question_bootstrap([kernel[k] for k in keys], [other[k] for k in keys], seed=101)})
    return {"cells": cells, "difference_in_differences": did}


def model_pair_summary(rows: Iterable[dict]) -> dict[str, Any]:
    rows = list(rows)
    values: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    for row in rows:
        c = row["condition"]
        sender = c.get("sender_model_id", "receiver-only") if int(c["k"]) else "receiver-only"
        values[(sender, c["receiver_model_id"])][_key(row)] = float(row["correct"])
    baselines = {receiver: mapping for (sender, receiver), mapping in values.items() if sender == "receiver-only"}
    cells, gains = [], []
    for (sender, receiver), mapping in sorted(values.items()):
        source_rows = [row for row in rows if
                       ((row["condition"].get("sender_model_id", "receiver-only")
                         if int(row["condition"]["k"]) else "receiver-only") == sender
                        and row["condition"]["receiver_model_id"] == receiver)]
        observations = [{"item_id": k[0], "seed": k[1], "value": v} for k, v in mapping.items()]
        cells.append({"sender": sender, "receiver": receiver,
                      "accuracy": nested_question_seed_bootstrap(observations, seed=101),
                      "seconds_per_problem": float(np.mean([_logical_seconds(r) for r in source_rows])),
                      "output_tokens_per_problem": float(np.mean([_output_tokens(r) for r in source_rows])),
                      "sender_seconds_per_problem": float(np.mean([r.get("sender_prefix_seconds", 0.0) for r in source_rows])),
                      "receiver_seconds_per_problem": float(np.mean([
                          r.get("receiver_prefill_seconds", 0.0) + r.get("receiver_text_decode_seconds", 0.0)
                          for r in source_rows])),
                      "transmitted_latent_states": int(source_rows[0]["condition"]["k"])})
        if sender != "receiver-only" and receiver in baselines:
            keys = sorted(set(mapping) & set(baselines[receiver]))
            gains.append({"sender": sender, "receiver": receiver,
                          **paired_question_bootstrap([mapping[k] for k in keys],
                                                      [baselines[receiver][k] for k in keys], seed=101)})

    model8, model14 = "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"
    contrasts = []
    def contrast(name: str, left: tuple[str, str], right: tuple[str, str]) -> None:
        a, b = values.get(left, {}), values.get(right, {})
        keys = sorted(set(a) & set(b))
        if keys:
            contrasts.append({"contrast": name,
                              **paired_question_bootstrap([a[k] for k in keys], [b[k] for k in keys], seed=101)})
    for receiver in (model8, model14):
        contrast(f"sender_size_at_{receiver}", (model14, receiver), (model8, receiver))
    for sender in (model8, model14):
        contrast(f"receiver_size_at_{sender}", (sender, model14), (sender, model8))
    contrast("heterogeneous_14B_to_8B_vs_same_size_8B", (model14, model8), (model8, model8))
    contrast("heterogeneous_8B_to_14B_vs_same_size_14B", (model8, model14), (model14, model14))
    # Interaction is computed per paired question/seed before bootstrap.
    four = [values.get(pair, {}) for pair in ((model14, model14), (model8, model14),
                                               (model14, model8), (model8, model8))]
    keys = sorted(set.intersection(*(set(x) for x in four))) if all(four) else []
    if keys:
        interaction = [(four[0][k] - four[1][k]) - (four[2][k] - four[3][k]) for k in keys]
        zero = [0.0] * len(keys)
        contrasts.append({"contrast": "sender_by_receiver_interaction",
                          **paired_question_bootstrap(interaction, zero, seed=101)})
    return {"cells": cells, "paired_gain_over_receiver_only": gains, "capacity_contrasts": contrasts}
