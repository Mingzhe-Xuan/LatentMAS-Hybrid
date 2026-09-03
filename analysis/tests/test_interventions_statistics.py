from __future__ import annotations

import torch

from analysis.core.interventions import perturb_source_states
from analysis.core.schemas import NoiseKey
from analysis.core.statistics import nested_question_seed_bootstrap, paired_question_bootstrap


def test_noise_is_paired_across_alignment_methods() -> None:
    hidden = torch.arange(1, 9, dtype=torch.float32)
    key = NoiseKey("aime2024", 3, "planner", 7, 42)
    first, diag_a = perturb_source_states(hidden, alpha=.05, noise_key=key)
    second, diag_b = perturb_source_states(hidden, alpha=.05, noise_key=key)
    assert torch.equal(first, second)
    assert diag_a.noise_seed == diag_b.noise_seed
    assert 0 < diag_a.relative_noise_norm < .2


def test_bootstraps_preserve_pairing_and_question_clusters() -> None:
    paired = paired_question_bootstrap([1, 1, 1], [0, 0, 0], samples=100, seed=1)
    assert paired["estimate"] == paired["ci_low"] == paired["ci_high"] == 1
    rows = [{"item_id": q, "seed": s, "value": float(q)} for q in (0, 1) for s in (42, 43, 44)]
    nested = nested_question_seed_bootstrap(rows, samples=100, seed=1)
    assert nested["estimate"] == .5
    assert nested["questions"] == 2
