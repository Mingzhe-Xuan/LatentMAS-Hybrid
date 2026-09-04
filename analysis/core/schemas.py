from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Sequence

import torch


def canonical_json(value: Any) -> str:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnalysisItem:
    item_id: int
    question: str
    solution: str
    gold: str
    question_hash: str = ""

    def __post_init__(self) -> None:
        expected = hashlib.sha256(self.question.encode("utf-8")).hexdigest()
        if self.question_hash and self.question_hash != expected:
            raise ValueError("question_hash does not match question")
        object.__setattr__(self, "question_hash", expected)


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset: str
    split: str
    selection_policy: str
    items: tuple[AnalysisItem, ...]
    fingerprint: str


@dataclass(frozen=True)
class KernelConfig:
    features: int = 2048
    temperature: float = 0.6
    seed: int = 101
    chunk_size: int = 4096


@dataclass(frozen=True)
class SenderConfig:
    dataset: str
    split: str
    model_id: str
    model_revision: str
    tokenizer_fingerprint: str
    dataset_fingerprint: str
    prompt_hash: str
    kmax: int
    dtype: str = "bfloat16"
    kernel: KernelConfig = field(default_factory=KernelConfig)
    schema_version: str = "sender-trajectory-v1"


@dataclass(frozen=True)
class SenderCacheIdentity:
    dataset: str
    split: str
    dataset_fingerprint: str
    selection_policy: str
    model_id: str
    model_revision: str
    tokenizer_fingerprint: str
    prompt_hash: str
    recurrence_alignment: str
    kernel: KernelConfig
    kmax: int
    dtype: str
    schema_version: str = "sender-trajectory-v1"

    @property
    def cache_id(self) -> str:
        return f"sender-{stable_hash(self)[:24]}"


@dataclass(frozen=True)
class ReceiverCondition:
    dataset: str
    split: str
    sender_manifest_hash: str
    sender_model_id: str
    receiver_model_id: str
    receiver_revision: str
    tokenizer_fingerprint: str
    prompt_hash: str
    k: int
    alignment: Literal["kernel", "soft", "linear"]
    kernel: KernelConfig = field(default_factory=KernelConfig)
    alpha: float = 0.0
    noise_scheme: str = "sha256-normal-v1"
    generation_seed: int = 42
    max_new_tokens: int = 256
    generation_batch_size: int = 1
    temperature: float = 0.6
    top_p: float = 0.95
    evaluator_version: str = "task-evaluator-v1"
    dataset_fingerprint: str = ""
    selection_policy: str = "all"

    def __post_init__(self) -> None:
        if self.k < 0 or self.alpha < 0:
            raise ValueError("k and alpha must be non-negative")
        if self.alignment not in {"kernel", "soft", "linear"}:
            raise ValueError("unsupported transfer alignment")
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "generation_seed", int(self.generation_seed))
        if self.generation_batch_size <= 0:
            raise ValueError("generation_batch_size must be positive")

    @property
    def perturbation(self) -> str:
        return "clean" if self.alpha == 0.0 else f"gaussian-{self.alpha:g}"

    @property
    def identity_payload(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["alpha"] = self.alpha
        payload["perturbation"] = "clean" if self.alpha == 0.0 else f"gaussian-{self.alpha:g}"
        # Sender is irrelevant for a receiver-only baseline.
        if self.k == 0:
            payload["sender_manifest_hash"] = "receiver-only"
            payload["sender_model_id"] = "receiver-only"
            payload["alignment"] = "kernel"
            payload["kernel"] = dataclasses.asdict(KernelConfig())
            payload["alpha"] = 0.0
            payload["perturbation"] = "clean"
            payload["noise_scheme"] = "none"
        return payload

    @property
    def cache_id(self) -> str:
        return f"receiver-{stable_hash(self.identity_payload)[:24]}"


@dataclass(frozen=True)
class STTReceiverCondition:
    dataset: str
    split: str
    dataset_fingerprint: str
    selection_policy: str
    system: Literal["qwen_only", "mistral_only", "qwen_to_mistral", "mistral_to_qwen"]
    receiver_model_id: str
    receiver_revision: str
    receiver_tokenizer_fingerprint: str
    receiver_prompt_hash: str
    max_new_tokens: int
    sender_manifest_hash: str = "receiver-only"
    sender_model_id: str = "receiver-only"
    sender_revision: str = "receiver-only"
    sender_tokenizer_fingerprint: str = "receiver-only"
    artifact_sha256: str = "none"
    artifact_source_fingerprint: str = "none"
    artifact_target_fingerprint: str = "none"
    artifact_source_name: str = "none"
    artifact_target_name: str = "none"
    tau: float = 0.6
    sender_budget: int = 1024
    causal_shift: bool = False
    do_sample: bool = False
    accumulation_dtype: str = "float32"
    position_chunk_size: int = 32
    target_chunk_size: int = 8192
    context_scope: str = "full-prompt-plus-plan"
    prefix_order: str = "aligned-sender-then-native-judger"
    evaluator_version: str = "task-evaluator-v1"
    code_revision: str = "unknown"
    schema_version: str = "stt-receiver-v2"

    def __post_init__(self) -> None:
        cross = "_to_" in self.system
        if self.max_new_tokens <= 0 or self.sender_budget != 1024:
            raise ValueError("invalid STT generation budgets")
        if self.tau != 0.6 or self.causal_shift or self.do_sample:
            raise ValueError("STT receiver protocol must use tau=0.6, no shift and greedy decoding")
        if self.accumulation_dtype != "float32":
            raise ValueError("STT transport accumulation must use float32")
        if self.position_chunk_size <= 0 or self.target_chunk_size <= 0:
            raise ValueError("STT chunk sizes must be positive")
        if (self.context_scope, self.prefix_order) != (
                "full-prompt-plus-plan", "aligned-sender-then-native-judger"):
            raise ValueError("STT context scope or prefix order is invalid")
        if cross and any(value in {"receiver-only", "none", ""} for value in (
                self.sender_manifest_hash, self.sender_model_id, self.sender_revision,
                self.sender_tokenizer_fingerprint, self.artifact_sha256,
                self.artifact_source_fingerprint, self.artifact_target_fingerprint)):
            raise ValueError("cross-model STT condition has incomplete sender/artifact identity")

    @property
    def identity_payload(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        if "_to_" not in self.system:
            payload.update(
                sender_manifest_hash="receiver-only", sender_model_id="receiver-only",
                sender_revision="receiver-only", sender_tokenizer_fingerprint="receiver-only",
                artifact_sha256="none", artifact_source_fingerprint="none",
                artifact_target_fingerprint="none", artifact_source_name="none",
                artifact_target_name="none",
            )
        return payload

    @property
    def cache_id(self) -> str:
        return f"stt-receiver-{stable_hash(self.identity_payload)[:24]}"


@dataclass
class SenderItemTrajectory:
    item_id: int
    question_hash: str
    hidden: torch.Tensor
    cumulative_sender_seconds: list[float]
    prompt_text: str
    prompt_hash: str
    hidden_semantics: str = "final_layer_last_position_before_readout_after_feedback"
    h0: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.hidden.ndim != 2:
            raise ValueError("hidden must have shape [steps, hidden_dim]")
        if len(self.cumulative_sender_seconds) != self.hidden.shape[0]:
            raise ValueError("one cumulative time is required per stored state")
        if any(b < a for a, b in zip(self.cumulative_sender_seconds, self.cumulative_sender_seconds[1:])):
            raise ValueError("cumulative sender times must be monotonic")


@dataclass(frozen=True)
class STTPlannerCacheIdentity:
    dataset: str
    split: str
    dataset_fingerprint: str
    selection_policy: str
    model_id: str
    model_revision: str
    tokenizer_fingerprint: str
    prompt_hash: str
    sender_budget: int
    code_revision: str = "unknown"
    do_sample: bool = False
    dtype: str = "bfloat16"
    hidden_semantics: str = "final_layer_all_valid_positions_full_prompt_plus_plan_no_shift"
    schema_version: str = "stt-planner-context-v1"

    def __post_init__(self) -> None:
        if self.sender_budget <= 0:
            raise ValueError("sender_budget must be positive")
        if self.do_sample:
            raise ValueError("formal STT planner decoding must be greedy")

    @property
    def cache_id(self) -> str:
        return f"stt-planner-{stable_hash(self)[:24]}"


@dataclass
class STTPlannerItemContext:
    item_id: int
    question_hash: str
    hidden: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_text: str
    prompt_hash: str
    messages_hash: str
    plan_text: str
    prompt_token_count: int
    plan_token_count: int
    generation_seconds: float
    full_forward_seconds: float
    hidden_semantics: str = "final_layer_all_valid_positions_full_prompt_plus_plan_no_shift"

    def __post_init__(self) -> None:
        if self.hidden.ndim != 2:
            raise ValueError("STT hidden must have shape [sequence, hidden_dim]")
        if self.input_ids.ndim != 1 or self.attention_mask.ndim != 1:
            raise ValueError("STT token IDs and mask must be one-dimensional")
        if self.hidden.shape[0] != len(self.input_ids) or len(self.input_ids) != len(self.attention_mask):
            raise ValueError("STT hidden, token IDs and mask lengths must match")
        if int(self.attention_mask.sum()) != len(self.attention_mask):
            raise ValueError("stored STT planner contexts must not contain padding")
        if self.prompt_token_count + self.plan_token_count != len(self.input_ids):
            raise ValueError("STT full context must equal prompt plus plan")
        if self.plan_token_count <= 0:
            raise ValueError("STT planner output must be non-empty")
        if self.generation_seconds < 0 or self.full_forward_seconds < 0:
            raise ValueError("STT timings must be non-negative")


@dataclass
class ReceiverItemResult:
    item_id: int
    question_hash: str
    prediction: str | None
    raw_prediction: str
    gold: str
    correct: bool
    error: str | None
    sender_prefix_seconds: float
    alignment_seconds: float
    receiver_prefill_seconds: float
    receiver_text_decode_seconds: float
    evaluation_seconds: float
    sender_recurrence_output_tokens: int
    transfer_alignment_output_tokens: int
    receiver_decode_output_tokens: int
    aligned_prefix_length: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return sum((self.sender_prefix_seconds, self.alignment_seconds,
                    self.receiver_prefill_seconds, self.receiver_text_decode_seconds,
                    self.evaluation_seconds))


@dataclass(frozen=True)
class NoiseKey:
    dataset: str
    item_id: int
    role: str
    step: int
    repeat: int


@dataclass(frozen=True)
class TokenizerCompatibilityReport:
    compatible: bool
    sender_hash: str
    receiver_hash: str
    vocabulary_size: int


def build_role_messages(*, role: Literal["planner", "judger"], question: str,
                        task: str, model_name: str) -> list[dict[str, str]]:
    if role not in ("planner", "judger"):
        raise ValueError("analysis only supports planner and judger prompts")
    from prompts import build_agent_message_sequential_latent_mas

    args = SimpleNamespace(task=task, model_name=model_name)
    messages = build_agent_message_sequential_latent_mas(
        role=role, question=question, context="", method="latent_mas", args=args
    )
    if not isinstance(messages, list):
        raise TypeError("prompt builder must return a list of messages")
    return messages


def prompt_fingerprint(messages: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash(list(messages))


def render_role_prompt(model: Any, messages: Sequence[Mapping[str, Any]],
                       model_name: str) -> str:
    """Render exactly as the production LatentMAS path, including its cue."""
    from reasoning_models import append_manual_reasoning_cue, resolve_manual_think

    rendered = model.render_chat(list(messages), add_generation_prompt=True)
    return append_manual_reasoning_cue(
        rendered, model_name, resolve_manual_think(model_name, None)
    )
