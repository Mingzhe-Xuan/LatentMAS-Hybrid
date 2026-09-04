from __future__ import annotations

import json
import hashlib
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from analysis.core.cache import file_sha256
from analysis.core.evaluation import evaluate_answer
from analysis.core.schemas import (AnalysisItem, ReceiverItemResult, STTPlannerItemContext,
                                   build_role_messages, canonical_json, render_role_prompt,
                                   stable_hash)


class STTArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class STTArtifactSpec:
    path: Path
    sha256: str
    source_name: str
    target_name: str
    source_revision: str
    target_revision: str


@dataclass(frozen=True)
class ValidatedSTTArtifact:
    spec: STTArtifactSpec
    indptr: torch.Tensor
    indices: torch.Tensor
    data: torch.Tensor
    shape: tuple[int, int]
    source_token_ids: torch.Tensor
    target_token_ids: torch.Tensor
    source_fingerprint: str
    target_fingerprint: str
    metadata: dict[str, Any]
    max_column_mass_error: float

    def sparse_coo(self, device: torch.device | str) -> torch.Tensor:
        columns = torch.repeat_interleave(
            torch.arange(self.shape[1], dtype=torch.long),
            self.indptr[1:] - self.indptr[:-1],
        )
        coordinates = torch.stack((self.indices, columns)).to(device)
        values = self.data.to(device=device, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            coordinates, values, self.shape, device=device, check_invariants=False
        ).coalesce()


@dataclass(frozen=True)
class STTChunkDiagnostics:
    source_mass_max_error: float
    target_mass_max_error: float
    source_entropy_mean: float
    target_entropy_mean: float
    source_effective_support_mean: float
    target_effective_support_mean: float
    aligned_nonfinite_count: int
    aligned_norm_mean: float


def transport_tokenizer_fingerprint(tokenizer: Any) -> str:
    """Canonical token-ID mapping fingerprint used at the runtime artifact gate."""
    rows = sorted((int(token_id), str(token)) for token, token_id in tokenizer.get_vocab().items())
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


def _base_model(wrapper: Any) -> Any:
    model = getattr(wrapper, "HF_model", getattr(wrapper, "model", None))
    if model is None:
        raise TypeError("model wrapper must expose model or HF_model")
    return model


def _sync(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _lm_head(model: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    head = getattr(model, "lm_head", None)
    if callable(head):
        return head
    embeddings = model.get_output_embeddings()
    if not callable(embeddings):
        raise TypeError("sender model does not expose a callable LM head")
    return embeddings


@torch.no_grad()
def collect_stt_planner_item(item: AnalysisItem, wrapper: Any, *, model_id: str,
                             sender_budget: int) -> STTPlannerItemContext:
    if sender_budget <= 0:
        raise ValueError("sender_budget must be positive")
    messages = build_role_messages(role="planner", question=item.question,
                                   task=getattr(wrapper.args, "task", ""), model_name=model_id)
    prompt_text = render_role_prompt(wrapper, messages, model_id)
    encoded = wrapper.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    device = getattr(wrapper, "device", "cpu")
    prompt_ids = encoded["input_ids"].to(device)
    prompt_mask = encoded["attention_mask"].to(device)
    if prompt_ids.shape[0] != 1 or not bool(prompt_mask.bool().all()):
        raise ValueError("STT planner collection requires one unpadded prompt")
    model = _base_model(wrapper)
    _sync(device)
    started = time.perf_counter()
    generated = model.generate(
        input_ids=prompt_ids, attention_mask=prompt_mask, max_new_tokens=sender_budget,
        do_sample=False, pad_token_id=wrapper.tokenizer.pad_token_id,
        return_dict_in_generate=True, output_scores=False, use_cache=True,
    )
    _sync(device)
    generation_seconds = time.perf_counter() - started
    full_ids = generated.sequences
    if full_ids.shape[0] != 1 or full_ids.shape[1] <= prompt_ids.shape[1]:
        raise ValueError("STT planner generation returned an empty plan")
    if not torch.equal(full_ids[:, :prompt_ids.shape[1]], prompt_ids):
        raise ValueError("STT planner generation does not start with the rendered prompt")
    plan_ids = full_ids[:, prompt_ids.shape[1]:]
    full_mask = torch.ones_like(full_ids, device=device)
    _sync(device)
    started = time.perf_counter()
    output = model(input_ids=full_ids, attention_mask=full_mask, output_hidden_states=True,
                   use_cache=False, return_dict=True)
    _sync(device)
    full_forward_seconds = time.perf_counter() - started
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise ValueError("sender full forward did not return hidden states")
    hidden = hidden_states[-1][0]
    if hidden.shape[0] != full_ids.shape[1]:
        raise ValueError("sender hidden sequence does not match full context")
    return STTPlannerItemContext(
        item_id=item.item_id, question_hash=item.question_hash, hidden=hidden.detach().cpu(),
        input_ids=full_ids[0].detach().cpu(), attention_mask=full_mask[0].detach().cpu(),
        prompt_text=prompt_text, prompt_hash=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        messages_hash=stable_hash(messages),
        plan_text=wrapper.tokenizer.decode(plan_ids[0], skip_special_tokens=True).strip(),
        prompt_token_count=int(prompt_ids.shape[1]), plan_token_count=int(plan_ids.shape[1]),
        generation_seconds=generation_seconds, full_forward_seconds=full_forward_seconds,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise STTArtifactError(message)


def _revision_pairs(value: Any) -> list[tuple[str, str, str, str]]:
    """Find fingerprint-bound revision pairs anywhere in nested provenance."""
    found: list[tuple[str, str, str, str]] = []
    if isinstance(value, dict):
        revisions = value.get("requested_revisions")
        source_fp, target_fp = value.get("source_fingerprint"), value.get("target_fingerprint")
        if (isinstance(revisions, dict) and isinstance(source_fp, str)
                and isinstance(target_fp, str) and isinstance(revisions.get("source"), str)
                and isinstance(revisions.get("target"), str)):
            found.append((source_fp, target_fp, revisions["source"], revisions["target"]))
        for child in value.values():
            found.extend(_revision_pairs(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_revision_pairs(child))
    return found


def _validate_revisions(metadata: dict[str, Any], *, source_fingerprint: str,
                        target_fingerprint: str, source_revision: str,
                        target_revision: str) -> None:
    for source_fp, target_fp, source_rev, target_rev in _revision_pairs(metadata):
        if (source_fp, target_fp, source_rev, target_rev) == (
                source_fingerprint, target_fingerprint, source_revision, target_revision):
            return
        if (source_fp, target_fp, source_rev, target_rev) == (
                target_fingerprint, source_fingerprint, target_revision, source_revision):
            return
    raise STTArtifactError("artifact provenance does not bind the requested model revisions to its fingerprints")


def load_stt_artifact(spec: STTArtifactSpec, *, source_vocab_size: int,
                      target_vocab_size: int, source_fingerprint: str,
                      target_fingerprint: str, mass_tolerance: float = 1e-6) -> ValidatedSTTArtifact:
    if file_sha256(spec.path) != spec.sha256:
        raise STTArtifactError("transport artifact SHA-256 mismatch")
    required = {"indptr", "indices", "data", "shape", "metadata",
                "source_token_ids", "target_token_ids"}
    try:
        with np.load(spec.path, allow_pickle=False) as archive:
            _require(required <= set(archive.files), "transport artifact schema is incomplete")
            indptr = np.asarray(archive["indptr"], dtype=np.int64)
            indices = np.asarray(archive["indices"], dtype=np.int64)
            data = np.asarray(archive["data"], dtype=np.float64)
            shape_array = np.asarray(archive["shape"], dtype=np.int64)
            source_ids = np.asarray(archive["source_token_ids"], dtype=np.int64)
            target_ids = np.asarray(archive["target_token_ids"], dtype=np.int64)
            metadata_value = archive["metadata"]
            _require(metadata_value.ndim == 0, "transport metadata must be a scalar JSON string")
            metadata = json.loads(str(metadata_value.item()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, STTArtifactError):
            raise
        raise STTArtifactError(f"cannot load transport artifact: {exc}") from exc

    _require(shape_array.shape == (2,), "transport shape must contain two dimensions")
    shape = (int(shape_array[0]), int(shape_array[1]))
    _require(shape == (len(target_ids), len(source_ids)),
             "transport shape does not match active token supports")
    _require(indptr.ndim == indices.ndim == data.ndim == 1, "invalid CSC tensor ranks")
    _require(len(indptr) == shape[1] + 1 and indptr[0] == 0 and indptr[-1] == len(data),
             "invalid CSC indptr")
    _require(len(indices) == len(data), "CSC indices/data length mismatch")
    _require(bool(np.all(indptr[1:] >= indptr[:-1])), "CSC indptr must be monotonic")
    _require(bool(np.all((indices >= 0) & (indices < shape[0]))), "CSC row index out of range")
    _require(bool(np.all(np.isfinite(data))) and bool(np.all(data >= 0)),
             "transport weights must be finite and non-negative")
    _require(len(np.unique(source_ids)) == len(source_ids), "source token IDs must be unique")
    _require(len(np.unique(target_ids)) == len(target_ids), "target token IDs must be unique")
    _require(np.array_equal(np.sort(source_ids), np.arange(source_vocab_size, dtype=np.int64)),
             "source token IDs do not cover the complete sender tokenizer vocabulary")
    _require(bool(np.all((target_ids >= 0) & (target_ids < target_vocab_size))),
             "target token ID is outside the receiver tokenizer vocabulary")
    _require(isinstance(metadata, dict), "transport metadata root must be an object")
    _require(metadata.get("schema_version") == 1, "unsupported transport schema version")
    _require(metadata.get("coordinate_system") == "active-support-target-by-source",
             "unexpected transport coordinate system")
    _require(metadata.get("source_fingerprint") == source_fingerprint,
             "sender tokenizer fingerprint mismatch")
    _require(metadata.get("target_fingerprint") == target_fingerprint,
             "receiver tokenizer fingerprint mismatch")
    _validate_revisions(metadata, source_fingerprint=source_fingerprint,
                        target_fingerprint=target_fingerprint,
                        source_revision=spec.source_revision,
                        target_revision=spec.target_revision)

    column_mass = np.fromiter(
        (data[indptr[index]:indptr[index + 1]].sum(dtype=np.float64)
         for index in range(shape[1])), dtype=np.float64, count=shape[1],
    )
    max_mass_error = float(np.max(np.abs(column_mass - 1.0))) if len(column_mass) else float("inf")
    _require(max_mass_error <= mass_tolerance, "transport source columns are not normalized")

    return ValidatedSTTArtifact(
        spec=spec,
        indptr=torch.from_numpy(indptr.copy()),
        indices=torch.from_numpy(indices.copy()),
        data=torch.from_numpy(data.copy()),
        shape=shape,
        source_token_ids=torch.from_numpy(source_ids.copy()),
        target_token_ids=torch.from_numpy(target_ids.copy()),
        source_fingerprint=source_fingerprint,
        target_fingerprint=target_fingerprint,
        metadata=metadata,
        max_column_mass_error=max_mass_error,
    )


def _embedding_weight(receiver_embeddings: torch.Tensor | torch.nn.Module) -> torch.Tensor:
    if isinstance(receiver_embeddings, torch.Tensor):
        return receiver_embeddings
    weight = getattr(receiver_embeddings, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("receiver embeddings must be a tensor or expose a tensor weight")
    return weight


@torch.no_grad()
def exact_stt(hidden: torch.Tensor, lm_head: Callable[[torch.Tensor], torch.Tensor],
              receiver_embeddings: torch.Tensor | torch.nn.Module,
              artifact: ValidatedSTTArtifact, *, tau: float = 0.6,
              position_chunk_size: int | None = None) -> tuple[torch.Tensor, STTChunkDiagnostics]:
    if hidden.ndim < 2:
        raise ValueError("hidden must have shape [..., sequence, hidden_dim]")
    if tau <= 0:
        raise ValueError("tau must be positive")
    positions = int(np.prod(hidden.shape[:-1]))
    chunk_size = position_chunk_size or positions
    if chunk_size <= 0:
        raise ValueError("position_chunk_size must be positive")
    flat_hidden = hidden.reshape(positions, hidden.shape[-1])
    sparse = artifact.sparse_coo(hidden.device)
    source_ids = artifact.source_token_ids.to(hidden.device)
    target_ids = artifact.target_token_ids.to(hidden.device)
    embedding_weight = _embedding_weight(receiver_embeddings)
    active_embeddings = embedding_weight.index_select(0, target_ids.to(embedding_weight.device)).float()
    outputs: list[torch.Tensor] = []
    source_errors: list[torch.Tensor] = []
    target_errors: list[torch.Tensor] = []
    source_entropies: list[torch.Tensor] = []
    target_entropies: list[torch.Tensor] = []
    source_effective_support: list[torch.Tensor] = []
    target_effective_support: list[torch.Tensor] = []
    for start in range(0, positions, chunk_size):
        logits = lm_head(flat_hidden[start:start + chunk_size]).float()
        if logits.ndim != 2 or logits.shape[1] < len(source_ids):
            raise ValueError("sender LM head output is smaller than the tokenizer vocabulary")
        logits = logits[:, :len(source_ids)]
        probabilities_full = torch.softmax(logits / tau, dim=-1)
        probabilities_source = probabilities_full.index_select(1, source_ids)
        probabilities_target = torch.sparse.mm(sparse, probabilities_source.transpose(0, 1)).transpose(0, 1)
        aligned = probabilities_target @ active_embeddings.to(probabilities_target.device)
        outputs.append(aligned.to(dtype=embedding_weight.dtype, device=hidden.device))
        source_errors.append((probabilities_source.sum(-1) - 1).abs())
        target_errors.append((probabilities_target.sum(-1) - 1).abs())
        source_entropy = -(probabilities_source * probabilities_source.clamp_min(1e-30).log()).sum(-1)
        target_entropy = -(probabilities_target * probabilities_target.clamp_min(1e-30).log()).sum(-1)
        source_entropies.append(source_entropy)
        target_entropies.append(target_entropy)
        source_effective_support.append(source_entropy.exp())
        target_effective_support.append(target_entropy.exp())
    result = torch.cat(outputs, dim=0).reshape(*hidden.shape[:-1], active_embeddings.shape[-1])
    diagnostics = STTChunkDiagnostics(
        source_mass_max_error=float(torch.cat(source_errors).max()),
        target_mass_max_error=float(torch.cat(target_errors).max()),
        source_entropy_mean=float(torch.cat(source_entropies).mean()),
        target_entropy_mean=float(torch.cat(target_entropies).mean()),
        source_effective_support_mean=float(torch.cat(source_effective_support).mean()),
        target_effective_support_mean=float(torch.cat(target_effective_support).mean()),
        aligned_nonfinite_count=int((~torch.isfinite(result)).sum()),
        aligned_norm_mean=float(result.float().norm(dim=-1).mean()),
    )
    return result, diagnostics


def pack_stt_prefix(aligned_sender: torch.Tensor, sender_mask: torch.Tensor,
                    receiver_embeddings: torch.Tensor,
                    receiver_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if aligned_sender.ndim != 3 or receiver_embeddings.ndim != 3:
        raise ValueError("sender and receiver embeddings must be rank three")
    if aligned_sender.shape[0] != receiver_embeddings.shape[0]:
        raise ValueError("sender and receiver batch sizes must match")
    if aligned_sender.shape[-1] != receiver_embeddings.shape[-1]:
        raise ValueError("sender and receiver embedding widths must match")
    if sender_mask.shape != aligned_sender.shape[:2] or receiver_mask.shape != receiver_embeddings.shape[:2]:
        raise ValueError("attention masks must match their embedding sequences")
    packed = [torch.cat((aligned_sender[index][sender_mask[index].bool()],
                         receiver_embeddings[index][receiver_mask[index].bool()]), dim=0)
              for index in range(aligned_sender.shape[0])]
    maximum = max(value.shape[0] for value in packed)
    output = aligned_sender.new_zeros((len(packed), maximum, aligned_sender.shape[-1]))
    mask = sender_mask.new_zeros((len(packed), maximum))
    for index, value in enumerate(packed):
        output[index, :value.shape[0]] = value
        mask[index, :value.shape[0]] = 1
    position_ids = mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return output, mask, position_ids


@torch.no_grad()
def greedy_decode_from_embeddings(wrapper: Any, inputs_embeds: torch.Tensor,
                                  attention_mask: torch.Tensor, position_ids: torch.Tensor,
                                  *, max_new_tokens: int) -> tuple[str, list[int], float, float]:
    if inputs_embeds.shape[0] != 1:
        raise ValueError("strict STT greedy decoding currently requires batch size one")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model = _base_model(wrapper)
    device = getattr(wrapper, "device", inputs_embeds.device)
    inputs_embeds = inputs_embeds.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)
    _sync(device)
    started = time.perf_counter()
    output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                   position_ids=position_ids, use_cache=True, return_dict=True)
    _sync(device)
    prefill_seconds = time.perf_counter() - started
    next_logits = output.logits[:, -1, :]
    past = output.past_key_values
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(wrapper.tokenizer, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (list, tuple, set)) else [eos]) - {None}
    answer_ids: list[int] = []
    _sync(device)
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        next_id = next_logits.argmax(dim=-1).reshape(1, 1)
        token_id = int(next_id.item())
        answer_ids.append(token_id)
        if token_id in eos_ids:
            break
        previous_valid = attention_mask.long().sum(dim=-1, keepdim=True)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_id)), dim=-1)
        output = model(input_ids=next_id, attention_mask=attention_mask,
                       position_ids=previous_valid, past_key_values=past,
                       use_cache=True, return_dict=True)
        next_logits = output.logits[:, -1, :]
        past = output.past_key_values
    _sync(device)
    decode_seconds = time.perf_counter() - started
    text = wrapper.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return text, answer_ids, prefill_seconds, decode_seconds


@torch.no_grad()
def evaluate_stt_item(item: AnalysisItem, receiver: Any, *, receiver_model_id: str,
                      max_new_tokens: int, planner: STTPlannerItemContext | None = None,
                      sender: Any | None = None, artifact: ValidatedSTTArtifact | None = None,
                      tau: float = 0.6, position_chunk_size: int | None = None) -> ReceiverItemResult:
    cross = planner is not None or sender is not None or artifact is not None
    if cross and (planner is None or sender is None or artifact is None):
        raise ValueError("cross-vocabulary STT requires planner context, sender model and artifact")
    if planner is not None and planner.question_hash != item.question_hash:
        raise ValueError("planner context question does not match receiver item")
    messages = build_role_messages(role="judger", question=item.question, task=getattr(receiver.args, "task", ""),
                                   model_name=receiver_model_id)
    prompt_text = render_role_prompt(receiver, messages, receiver_model_id)
    encoded = receiver.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    receiver_ids = encoded["input_ids"].to(receiver.device)
    receiver_mask = encoded["attention_mask"].to(receiver.device)
    receiver_embedding_layer = _base_model(receiver).get_input_embeddings()
    receiver_embeddings = receiver_embedding_layer(receiver_ids)
    aligned = None
    alignment_seconds = 0.0
    transport_diagnostics: dict[str, Any] = {}
    if planner is not None and sender is not None and artifact is not None:
        _sync(receiver.device)
        started = time.perf_counter()
        aligned, diagnostics = exact_stt(
            planner.hidden.to(sender.device), _lm_head(_base_model(sender)), receiver_embedding_layer,
            artifact, tau=tau, position_chunk_size=position_chunk_size,
        )
        _sync(receiver.device)
        alignment_seconds = time.perf_counter() - started
        aligned = aligned.unsqueeze(0).to(device=receiver.device, dtype=receiver_embeddings.dtype)
        sender_mask = planner.attention_mask.unsqueeze(0).to(receiver.device)
        inputs_embeds, generation_mask, position_ids = pack_stt_prefix(
            aligned, sender_mask, receiver_embeddings, receiver_mask,
        )
        transport_diagnostics = dataclasses.asdict(diagnostics)
    else:
        inputs_embeds = receiver_embeddings
        generation_mask = receiver_mask
        position_ids = generation_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(generation_mask == 0, 0)
    raw, answer_ids, prefill_seconds, decode_seconds = greedy_decode_from_embeddings(
        receiver, inputs_embeds, generation_mask, position_ids,
        max_new_tokens=max_new_tokens,
    )
    started = time.perf_counter()
    evaluation = evaluate_answer(getattr(receiver.args, "task", ""), raw, item.gold)
    evaluation_seconds = time.perf_counter() - started
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    prefix_length = int(planner.attention_mask.sum()) if planner is not None else 0
    return ReceiverItemResult(
        item_id=item.item_id, question_hash=item.question_hash,
        prediction=evaluation.prediction, raw_prediction=raw, gold=item.gold,
        correct=evaluation.correct, error=evaluation.error,
        sender_prefix_seconds=(planner.generation_seconds + planner.full_forward_seconds)
        if planner is not None else 0.0,
        alignment_seconds=alignment_seconds, receiver_prefill_seconds=prefill_seconds,
        receiver_text_decode_seconds=decode_seconds, evaluation_seconds=evaluation_seconds,
        sender_recurrence_output_tokens=planner.plan_token_count if planner is not None else 0,
        transfer_alignment_output_tokens=prefix_length,
        receiver_decode_output_tokens=len(answer_ids), aligned_prefix_length=prefix_length,
        diagnostics={
            "prompt_text": prompt_text, "prompt_hash": prompt_hash,
            "messages": messages, "messages_hash": stable_hash(messages),
            "receiver_prompt_token_ids": receiver_ids[0].detach().cpu().tolist(),
            "receiver_attention_mask": receiver_mask[0].detach().cpu().tolist(),
            "prefill_attention_mask": generation_mask[0].detach().cpu().tolist(),
            "prefill_position_ids": position_ids[0].detach().cpu().tolist(),
            "prefix_order": ["aligned_sender_prompt", "aligned_sender_plan", "receiver_native_prompt"]
            if planner is not None else ["receiver_native_prompt"],
            "causal_shift": False, "do_sample": False,
            "sender_prompt_token_count": planner.prompt_token_count if planner is not None else 0,
            "sender_plan_token_count": planner.plan_token_count if planner is not None else 0,
            "sender_full_context_token_count": prefix_length,
            "receiver_prompt_token_count": int(receiver_mask.sum()),
            "transferred_length_ratio": prefix_length / int(generation_mask.sum()),
            "planner_generation_seconds": planner.generation_seconds if planner is not None else 0.0,
            "planner_full_forward_seconds": planner.full_forward_seconds if planner is not None else 0.0,
            "transport": transport_diagnostics,
        },
    )
