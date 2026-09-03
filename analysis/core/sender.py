from __future__ import annotations

import hashlib
import time
from typing import Any

import torch

from analysis.core.schemas import (AnalysisItem, SenderConfig,
                                   SenderItemTrajectory, build_role_messages,
                                   render_role_prompt)


def _past_length(past: Any) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    first = past[0][0] if isinstance(past[0], (tuple, list)) else past[0]
    return int(first.shape[-2])


def _sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def collect_sender_item(item: AnalysisItem, model: Any,
                        config: SenderConfig) -> SenderItemTrajectory:
    """Collect h_1..h_K immediately before readout after each feedback step."""
    if config.kmax <= 0:
        raise ValueError("Sender kmax must be positive")
    messages = build_role_messages(role="planner", question=item.question,
                                   task=config.dataset, model_name=config.model_id)
    prompt_text = render_role_prompt(model, messages, config.model_id)
    encoded = model.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    source_model = getattr(model, "HF_model", getattr(model, "model", None))
    if source_model is None:
        raise TypeError("model wrapper must expose model or HF_model")
    device = getattr(model, "HF_device", getattr(model, "device", input_ids.device))
    _sync(device)
    started = time.perf_counter()
    outputs = source_model(input_ids=input_ids, attention_mask=attention_mask,
                           use_cache=True, output_hidden_states=True, return_dict=True)
    past = outputs.past_key_values
    # h_0 is diagnostic only and never transmitted.
    h0 = outputs.hidden_states[-1][:, -1, :]
    last_hidden = h0
    states: list[torch.Tensor] = []
    cumulative: list[float] = []
    for _step in range(1, config.kmax + 1):
        aligned = model._apply_latent_realignment(last_hidden, source_model)
        past_len = _past_length(past)
        mask = torch.ones((aligned.shape[0], past_len + 1), dtype=torch.long,
                          device=aligned.device)
        outputs = source_model(inputs_embeds=aligned.unsqueeze(1), attention_mask=mask,
                               past_key_values=past, use_cache=True,
                               output_hidden_states=True, return_dict=True)
        past = outputs.past_key_values
        # This line is the protocol-defining capture point.
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        states.append(last_hidden.detach().to("cpu"))
        _sync(device)
        cumulative.append(time.perf_counter() - started)
    hidden = torch.cat(states, dim=0)
    return SenderItemTrajectory(item.item_id, item.question_hash, hidden, cumulative,
                                prompt_text, prompt_hash, h0=h0.detach().to("cpu"))
