from __future__ import annotations

import hashlib
import time
from typing import Any

import torch

from analysis.core.evaluation import evaluate_answer
from analysis.core.interventions import perturb_source_states
from analysis.core.schemas import (AnalysisItem, NoiseKey, ReceiverCondition,
                                   ReceiverItemResult, SenderItemTrajectory,
                                   TokenizerCompatibilityReport,
                                   build_role_messages, canonical_json,
                                   render_role_prompt)


def tokenizer_mapping_hash(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    rows = sorted(((int(row), str(token)) for token, row in vocab.items()))
    special = {
        name: getattr(tokenizer, name, None) for name in
        ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    }
    return hashlib.sha256(canonical_json({"rows": rows, "special": special}).encode("utf-8")).hexdigest()


def validate_cross_model_alignment(sender: Any, receiver: Any) -> TokenizerCompatibilityReport:
    sender_hash = tokenizer_mapping_hash(sender.tokenizer)
    receiver_hash = tokenizer_mapping_hash(receiver.tokenizer)
    sender_vocab, receiver_vocab = sender.tokenizer.get_vocab(), receiver.tokenizer.get_vocab()
    report = TokenizerCompatibilityReport(sender_hash == receiver_hash,
                                          sender_hash, receiver_hash, len(sender_vocab))
    if not report.compatible or len(sender_vocab) != len(receiver_vocab):
        raise ValueError("Sender and Receiver tokenizer row mappings are not semantically identical")
    return report


def _input_embedding(model: Any) -> Any:
    base = getattr(model, "HF_model", getattr(model, "model", None))
    if base is None:
        raise TypeError("Receiver wrapper must expose model or HF_model")
    return base.get_input_embeddings()


def _sync(device: Any) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


@torch.no_grad()
def evaluate_receiver_batch(items: list[AnalysisItem],
                            senders: list[SenderItemTrajectory | None],
                            config: ReceiverCondition, model: Any, *,
                            sender_model: Any | None = None) -> list[ReceiverItemResult]:
    if not items or len(items) != len(senders):
        raise ValueError("items and senders must be non-empty equal-length lists")
    if config.k > 0 and any(sender is None for sender in senders):
        raise ValueError("positive-K Receiver condition requires Sender trajectories")
    if any(sender is not None and config.k > sender.hidden.shape[0] for sender in senders):
        raise ValueError("a Sender trajectory is shorter than requested prefix")
    source = sender_model or model
    compatibility = validate_cross_model_alignment(source, model)
    messages = [build_role_messages(role="judger", question=item.question,
                                    task=config.dataset, model_name=config.receiver_model_id)
                for item in items]
    prompt_texts = [render_role_prompt(model, value, config.receiver_model_id)
                    for value in messages]
    encoded = model.tokenizer(prompt_texts, return_tensors="pt", padding=True,
                              add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    rendered_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in prompt_texts]

    aligned = None
    perturbation_rows: list[list[dict[str, float | int]]] = [[] for _ in items]
    aligned_diagnostics: list[dict[str, float | int]] = [{
        "aligned_relative_l2_change": 0.0, "aligned_cosine": 1.0,
        "amplification_ratio": 0.0, "probe_unembedding_evaluations": 0,
    } for _ in items]
    alignment_seconds = 0.0
    if config.k:
        hidden = torch.stack([sender.hidden[:config.k] for sender in senders if sender is not None]).to(
            getattr(source, "device", "cpu"))
        clean_hidden = hidden
        if config.alpha:
            changed_batch = []
            for batch_index, (item, states) in enumerate(zip(items, hidden)):
                changed = []
                for index, state in enumerate(states, start=1):
                    value, diagnostics = perturb_source_states(
                        state, alpha=config.alpha,
                        noise_key=NoiseKey(config.dataset, item.item_id, "planner", index,
                                           config.generation_seed),
                    )
                    changed.append(value)
                    perturbation_rows[batch_index].append({
                        "step": index, "relative_noise_norm": diagnostics.relative_noise_norm,
                        "original_perturbed_cosine": diagnostics.original_perturbed_cosine,
                        "noise_seed": diagnostics.noise_seed,
                    })
                changed_batch.append(torch.stack(changed))
            hidden = torch.stack(changed_batch)
        old_method = getattr(source, "align_method", None)
        try:
            source.align_method = config.alignment
            args = getattr(source, "args", None)
            if args is not None:
                args.kernel_features = config.kernel.features
                args.kernel_temperature = config.kernel.temperature
                args.kernel_seed = config.kernel.seed
                args.kernel_chunk_size = config.kernel.chunk_size
                args.soft_temperature = config.kernel.temperature
            if old_method != config.alignment:
                source._alignment_states = {}
            _sync(getattr(source, "device", "cpu"))
            started = time.perf_counter()
            aligned = source.align_hidden_to(hidden, model)
            _sync(getattr(source, "device", "cpu"))
            alignment_seconds = (time.perf_counter() - started) / len(items)
            if config.alpha:
                clean_aligned = source.align_hidden_to(clean_hidden, model)
                relative = (aligned.float() - clean_aligned.float()).norm(dim=-1) / clean_aligned.float().norm(dim=-1).clamp_min(1e-12)
                cosine = torch.nn.functional.cosine_similarity(aligned.float(), clean_aligned.float(), dim=-1)
                for index in range(len(items)):
                    source_relative = sum(float(row["relative_noise_norm"]) for row in perturbation_rows[index]) / len(perturbation_rows[index])
                    aligned_relative = float(relative[index].mean())
                    aligned_diagnostics[index] = {
                        "aligned_relative_l2_change": aligned_relative,
                        "aligned_cosine": float(cosine[index].mean()),
                        "amplification_ratio": aligned_relative / source_relative if source_relative else 0.0,
                        "probe_unembedding_evaluations": config.k if config.alignment == "soft" else 0,
                    }
        finally:
            source.align_method = old_method

    torch.manual_seed(config.generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.generation_seed)
    _sync(getattr(model, "device", "cpu"))
    embedding_started = time.perf_counter()
    prompt_embeddings = _input_embedding(model)(input_ids)
    _sync(getattr(model, "device", "cpu"))
    embedding_seconds = (time.perf_counter() - embedding_started) / len(items)
    if aligned is not None:
        aligned = aligned.to(device=prompt_embeddings.device, dtype=prompt_embeddings.dtype)
        inputs_embeds = torch.cat((prompt_embeddings, aligned), dim=1)
        prefix_mask = torch.ones((attention_mask.shape[0], config.k),
                                 dtype=attention_mask.dtype, device=attention_mask.device)
        generation_mask = torch.cat((attention_mask, prefix_mask), dim=1)
    else:
        inputs_embeds, generation_mask = prompt_embeddings, attention_mask
    generations, _ = model.generate_text_from_embeds_batch(
        inputs_embeds, generation_mask, max_new_tokens=config.max_new_tokens,
        temperature=config.temperature, top_p=config.top_p,
    )
    metrics = getattr(model, "last_generation_metrics", {})
    counts = metrics.get("output_token_counts")
    if counts is None:
        counts = [int(model.tokenize_text(raw).shape[-1]) for raw in generations]
    results = []
    for index, (item, sender, raw) in enumerate(zip(items, senders, generations)):
        evaluation = evaluate_answer(config.dataset, raw, item.gold)
        sender_seconds = sender.cumulative_sender_seconds[config.k - 1] if config.k and sender else 0.0
        results.append(ReceiverItemResult(
            item.item_id, item.question_hash, evaluation.prediction, raw, item.gold,
            evaluation.correct, evaluation.error, sender_seconds, alignment_seconds,
            float(metrics.get("prefill_seconds", 0.0)) / len(items) + embedding_seconds,
            float(metrics.get("text_decode_seconds", 0.0)) / len(items), evaluation.seconds,
            0, config.k if config.alignment == "soft" else 0, int(counts[index]), config.k,
            diagnostics={"prompt_text": prompt_texts[index], "prompt_hash": rendered_hashes[index],
                         "tokenizer_compatibility": compatibility.__dict__,
                         "perturbations": perturbation_rows[index], **aligned_diagnostics[index]},
        ))
    return results


def evaluate_receiver_item(item: AnalysisItem, sender: SenderItemTrajectory | None,
                           config: ReceiverCondition, model: Any, *,
                           sender_model: Any | None = None) -> ReceiverItemResult:
    return evaluate_receiver_batch([item], [sender], config, model,
                                   sender_model=sender_model)[0]
