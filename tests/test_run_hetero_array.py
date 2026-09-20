import re
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from methods.latent_mas_hybrid import LatentMASMethod
from methods.text_mas import TextMASMethod


ROOT = Path(__file__).resolve().parents[1]
HETERO = (ROOT / "run_hetero.sh").read_text(encoding="utf-8")
RUN = (ROOT / "run.sh").read_text(encoding="utf-8")
HYBRID = (ROOT / "methods" / "latent_mas_hybrid.py").read_text(encoding="utf-8")


def _bash_array(name: str) -> list[str]:
    match = re.search(rf"{name}=\((.*?)\)", HETERO, re.DOTALL)
    assert match is not None
    return re.findall(r'"([^"]+)"|([A-Za-z0-9_]+)', match.group(1))


def _values(name: str) -> list[str]:
    return [quoted or bare for quoted, bare in _bash_array(name)]


def test_default_cross_model_matrix_matches_table_two_scope() -> None:
    assert _values("DATASETS") == [
        "aime2024", "aime2025", "gpqa", "humanevalplus", "mbppplus", "medqa",
    ]
    assert _values("SENDERS") == ["Qwen/Qwen3-14B", "Qwen/Qwen3-8B"]
    assert _values("RECEIVERS") == ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
    assert _values("METHODS") == [
        "text_mas", "latent_mas_hybrid", "latent_mas_hybrid", "latent_mas_hybrid",
    ]
    assert _values("ALIGNMENTS") == ["identical", "linear", "soft", "kernel"]
    assert "TOTAL_COUNT=$((DATASET_COUNT * DIRECTION_COUNT * EXPERIMENT_COUNT))" in HETERO
    assert "#PBS -J 1-48%3" in HETERO
    assert 'qsub -J "1-${TOTAL_COUNT}%${MAX_CONCURRENT_GPUS}"' in HETERO


def test_hetero_reruns_completed_configs_by_default() -> None:
    assert 'FORCE_ALL="${FORCE_ALL:-true}"' in HETERO
    assert '[[ "${FORCE_ALL}" != true ]] && state_file_completed' in HETERO


def test_hetero_enables_repetition_penalty_without_changing_global_default() -> None:
    assert "REPETITION_PENALTY=1.05" in HETERO
    assert "export REPETITION_PENALTY" in HETERO
    assert "REPETITION_PENALTY=${REPETITION_PENALTY}" in HETERO
    assert 'REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"' in RUN
    assert '--repetition_penalty "${REPETITION_PENALTY}"' in RUN


def test_hybrid_role_mapping_and_run_sh_forwarding() -> None:
    assert 'AGENT_MODELS="${SENDER_MODEL} ${RECEIVER_MODEL}"' in HETERO
    assert 'CONFIG_METHOD="${METHODS[${EXPERIMENT_INDEX}]}"' in HETERO
    assert "text_mas" in HETERO
    assert "latent_mas_hybrid" in HETERO
    assert "latent_mas|latent_mas_hybrid)" in RUN
    assert 'command+=(--agent_models "${HETERO_AGENT_MODELS[@]}")' in RUN
    assert 'elif len(agent_models) == 2:' in HYBRID
    assert 'Agent(name="Planner", role="planner")' in HYBRID
    assert 'Agent(name="Judger", role="judger")' in HYBRID


def test_hybrid_soft_counts_vocab_decodes_as_text_output() -> None:
    assert "latent_vocab_decode_steps" in HYBRID
    assert '"text_output_counts": [' in HYBRID
    assert 'text_output_tokens=phase_metrics["text_output_counts"][idx]' in HYBRID


def test_hybrid_soft_uses_run_all_entropy_stopping_and_actual_steps() -> None:
    class FakeModel:
        def __call__(self, *, inputs_embeds, **_kwargs):
            return SimpleNamespace(
                past_key_values=None,
                hidden_states=[inputs_embeds],
            )

    class FakeWrapper:
        device = "cpu"
        align_method = "soft"

        def __init__(self):
            self.model = FakeModel()
            self.calls = 0

        def _apply_latent_realignment(self, hidden, _model, *, return_entropy=False):
            self.calls += 1
            assert return_entropy
            return hidden, torch.zeros(hidden.shape[0])

    class FakeTimer:
        def measure(self, callback):
            return callback()

        def seconds(self):
            return 0.0

    method = object.__new__(LatentMASMethod)
    method.args = Namespace(
        early_stopping_length_threshold=2,
        early_stopping_entropy_threshold=0.01,
    )
    method.latent_steps = 0
    method.latent_only = True
    method.sequential_info_only = True
    wrapper = FakeWrapper()

    context, mask, _ = method._prefill_and_latent(
        wrapper,
        torch.ones(2, 3, 4),
        torch.ones(2, 3, dtype=torch.long),
        prompt_width=3,
        alignment_timer=FakeTimer(),
    )

    assert wrapper.calls == 2
    assert context.shape == (2, 2, 4)
    assert mask.shape == (2, 2)
    assert wrapper.last_latent_metrics["latent_output_counts"] == [2, 2]
    assert wrapper.last_latent_metrics["text_output_counts"] == [2, 2]


def test_hetero_forwards_run_all_early_stopping_controls() -> None:
    assert 'EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-auto}"' in HETERO
    assert 'EARLY_STOPPING_ENTROPY_THRESHOLD="${EARLY_STOPPING_ENTROPY_THRESHOLD:-auto}"' in HETERO
    assert "EARLY_STOPPING_LENGTH_THRESHOLD=${EARLY_STOPPING_LENGTH_THRESHOLD}" in HETERO
    assert "EARLY_STOPPING_ENTROPY_THRESHOLD=${EARLY_STOPPING_ENTROPY_THRESHOLD}" in HETERO
    assert "export EARLY_STOPPING_LENGTH_THRESHOLD EARLY_STOPPING_ENTROPY_THRESHOLD" in HETERO


def test_hetero_explicitly_preserves_complete_context_in_order() -> None:
    assert 'SEQUENTIAL_INFO_ONLY="${SEQUENTIAL_INFO_ONLY:-false}"' in RUN
    assert 'LATENT_ONLY="${LATENT_ONLY:-false}"' in RUN
    assert "SEQUENTIAL_INFO_ONLY=false" in HETERO
    assert "LATENT_ONLY=false" in HETERO
    assert "export SEQUENTIAL_INFO_ONLY LATENT_ONLY" in HETERO
    assert "sender prompt states || sender latent-output states || receiver prompt" in HETERO
    assert "torch.cat([prefill_hidden, latent_hidden_states], dim=1)" in HYBRID
    assert "torch.cat([aligned_context, prompt_embeds], dim=1)" in HYBRID


def test_two_model_mode_constructs_only_planner_and_judger() -> None:
    model = type("FakeModel", (), {"model_name": "sender", "use_vllm": False})()
    args = Namespace(
        device="cpu",
        device2="cpu",
        max_new_tokens=16,
        task="aime2024",
        latent_only=False,
        sequential_info_only=False,
    )
    with (
        patch.object(LatentMASMethod, "_load_additional_models"),
        patch.object(LatentMASMethod, "_validate_alignment_chain"),
    ):
        method = LatentMASMethod(
            model,
            agent_models=["sender", "receiver"],
            args=args,
        )

    assert [(agent.name, agent.role) for agent in method.agents] == [
        ("Planner", "planner"),
        ("Judger", "judger"),
    ]
    assert method.agent_models == ["sender", "receiver"]


def test_heterogeneous_text_mas_constructs_only_planner_and_judger() -> None:
    model = type(
        "FakeModel",
        (),
        {"model_name": "sender", "use_vllm": False},
    )()
    args = Namespace(task="aime2024", device="cpu")
    with patch("methods.text_mas.ModelWrapper"):
        method = TextMASMethod(
            model,
            agent_models=["sender", "receiver"],
            args=args,
        )

    assert [(agent.name, agent.role) for agent in method.agents] == [
        ("Planner", "planner"),
        ("Judger", "judger"),
    ]
    assert method.agent_models == ["sender", "receiver"]
    assert set(method.models) == {"sender", "receiver"}


def test_default_text_mas_keeps_original_four_agent_single_model_mode() -> None:
    model = type(
        "FakeModel",
        (),
        {"model_name": "single", "use_vllm": False},
    )()
    method = TextMASMethod(
        model,
        args=Namespace(task="aime2024", device="cpu"),
    )

    assert [agent.role for agent in method.agents] == [
        "planner", "critic", "refiner", "judger",
    ]
    assert method.agent_models == ["single"] * 4


def test_task_parameters_remain_owned_by_params_dict() -> None:
    assert 'RESOLVED_MAX_NEW_TOKENS="$(resolve_max_new_tokens "${TASK}")"' in RUN
    assert 'RESOLVED_GENERATE_BS="$(resolve_generate_bs "${TASK}")"' in RUN
    assert 'RESOLVED_TIMES="$(resolve_times "${TASK}")"' in RUN
    assert 'RESOLVED_LATENT_STEPS="$(resolve_latent_steps "${TASK}" "${prompt}")"' in RUN
    assert "MAX_NEW_TOKENS=" not in HETERO
    assert "LATENT_STEPS=" not in HETERO
    assert "TIMES=" not in HETERO
