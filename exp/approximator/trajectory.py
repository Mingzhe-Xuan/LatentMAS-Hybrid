"""Real latent-MAS-hybrid trajectory collection for the approximator study."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import torch

from methods import default_agents
from methods.latent_mas_hybrid import LatentMASMethod, transfer_via_realignment
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas



def _cpu_vector(vector: torch.Tensor) -> torch.Tensor:
    return vector.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _state(
    *,
    vector: torch.Tensor,
    item_id: int,
    role: str,
    agent_id: int,
    state_kind: str,
    position: int,
    model_name: str,
) -> Dict[str, Any]:
    return {
        "item_id": int(item_id),
        "role": role,
        "agent_id": int(agent_id),
        "turn_id": 0,
        "state_kind": state_kind,
        "position": int(position),
        "vector": _cpu_vector(vector),
        "model_name": model_name,
    }


def _json_safe(value: Any) -> Any:
    """Make dataset/prompt provenance safe for torch weights-only loading."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _attention_with_past(mask: torch.Tensor, past: Optional[Tuple]) -> torch.Tensor:
    past_len = _past_length(past)
    if not past_len:
        return mask
    return torch.cat(
        [
            torch.ones(
                (mask.shape[0], past_len), dtype=mask.dtype, device=mask.device
            ),
            mask,
        ],
        dim=-1,
    )


@torch.inference_mode()
def _latent_rollout(
    wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past: Optional[Tuple],
    latent_steps: int,
) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
    """The same kernel-aligned latent recurrence used by latent_mas_hybrid."""
    outputs = wrapper.model(
        input_ids=input_ids,
        attention_mask=_attention_with_past(attention_mask, past),
        past_key_values=past,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    prompt_hidden = outputs.hidden_states[-1][0, -int(attention_mask.sum()) :, :]
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]
    latent_hidden: List[torch.Tensor] = []
    for _ in range(latent_steps):
        # This is the real hidden used by the main hybrid recurrence to form the
        # next latent input embedding.
        latent_hidden.append(last_hidden[0])
        latent_embed = wrapper._apply_latent_realignment(
            last_hidden, wrapper.model
        ).unsqueeze(1)
        outputs = wrapper.model(
            inputs_embeds=latent_embed,
            attention_mask=torch.ones(
                (1, _past_length(past) + 1),
                dtype=torch.long,
                device=wrapper.device,
            ),
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
    hidden_size = last_hidden.shape[-1]
    latent_tensor = (
        torch.stack(latent_hidden)
        if latent_hidden
        else torch.empty((0, hidden_size), device=last_hidden.device)
    )
    return past, prompt_hidden, latent_tensor


def _generation_hidden_steps(generation) -> List[torch.Tensor]:
    """Extract hidden states computed during sampled autoregressive decoding."""
    hidden_steps = getattr(generation, "hidden_states", None) or ()
    # Step zero is the new Judger prompt. Later steps are sampled reply tokens.
    return [step[-1][0, -1, :] for step in hidden_steps[1:]]


@torch.inference_mode()
def _sample_judger(
    wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past: Optional[Tuple],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    full_mask = _attention_with_past(attention_mask, past)
    past_len = _past_length(past)
    cache_position = (
        torch.arange(
            past_len,
            past_len + input_ids.shape[-1],
            dtype=torch.long,
            device=wrapper.device,
        )
        if past is not None
        else None
    )
    generation = wrapper.model.generate(
        input_ids=input_ids,
        attention_mask=full_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=wrapper.tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=False,
        past_key_values=past,
        cache_position=cache_position,
    )
    hidden_steps = getattr(generation, "hidden_states", None) or ()
    if not hidden_steps:
        raise RuntimeError("Judger generation did not return hidden states.")
    prompt_hidden = hidden_steps[0][-1][0, -int(attention_mask.sum()) :, :]
    generated_ids = generation.sequences[0, input_ids.shape[-1] :].tolist()
    reply_hidden = _generation_hidden_steps(generation)

    # Transformers' final sampled token has not yet been forwarded when
    # generation stops. Forward exactly that actually sampled token against the
    # returned real cache so its hidden state is represented too.
    if generated_ids and len(reply_hidden) < len(generated_ids):
        final_id = torch.tensor(
            [[generated_ids[-1]]], dtype=input_ids.dtype, device=wrapper.device
        )
        final = wrapper.model(
            input_ids=final_id,
            attention_mask=torch.ones(
                (1, _past_length(generation.past_key_values) + 1),
                dtype=attention_mask.dtype,
                device=wrapper.device,
            ),
            past_key_values=generation.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        reply_hidden.append(final.hidden_states[-1][0, -1, :])
    if len(reply_hidden) != len(generated_ids):
        raise RuntimeError(
            "Judger audit mismatch: "
            f"{len(generated_ids)} sampled tokens but {len(reply_hidden)} hidden states."
        )
    return prompt_hidden, reply_hidden, generated_ids


@torch.inference_mode()
def _rebuild_switched_cache(
    *,
    target: ModelWrapper,
    source: ModelWrapper,
    cumulative_prompts: str,
    cumulative_latent_hiddens: Optional[torch.Tensor],
) -> Tuple[Optional[Tuple], torch.Tensor]:
    encoded = target.tokenizer(
        cumulative_prompts,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_ids = encoded["input_ids"].to(target.device)
    prompt_mask = encoded["attention_mask"].to(target.device)
    prompt_embeds = target.model.get_input_embeddings()(prompt_ids)
    parts = [prompt_embeds]
    masks = [prompt_mask]
    if cumulative_latent_hiddens is not None and cumulative_latent_hiddens.numel():
        transferred = transfer_via_realignment(
            cumulative_latent_hiddens.unsqueeze(0), source, target
        )
        parts.append(transferred)
        masks.append(
            torch.ones(
                (1, transferred.shape[1]),
                dtype=prompt_mask.dtype,
                device=target.device,
            )
        )
    outputs = target.model(
        inputs_embeds=torch.cat(parts, dim=1),
        attention_mask=torch.cat(masks, dim=1),
        past_key_values=None,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    prompt_length = prompt_ids.shape[1]
    previous_prompt_hidden = outputs.hidden_states[-1][0, :prompt_length, :]
    return outputs.past_key_values, previous_prompt_hidden


def load_hybrid(args: argparse.Namespace) -> LatentMASMethod:
    """Load the four role wrappers through the repository hybrid method."""
    initial = ModelWrapper(args.agent_models[0], args.device, use_vllm=False, args=args)
    return LatentMASMethod(
        initial,
        agent_models=list(args.agent_models),
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=1,
        args=args,
    )


@torch.inference_mode()
def collect(
    method: LatentMASMethod,
    indexed_items: List[Tuple[int, Dict[str, Any]]],
    args: argparse.Namespace,
    logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the real four-role sequential hybrid flow once per question."""
    saved_states: List[Dict[str, Any]] = []
    question_records: List[Dict[str, Any]] = []
    agents = default_agents()

    for question_number, (item_id, item) in enumerate(indexed_items, start=1):
        logger.info(
            "Phase A: question %d/%d (item_id=%d) started.",
            question_number,
            len(indexed_items),
            item_id,
        )
        question = item["question"]
        source_record = _json_safe(item)
        question_record = {
            "item_id": int(item_id),
            "question": question,
            "source_record": source_record,
            "source_record_sha256": hashlib.sha256(
                json.dumps(
                    source_record, sort_keys=True, ensure_ascii=False
                ).encode("utf-8")
            ).hexdigest(),
            "roles": [],
        }
        question_records.append(question_record)
        past = None
        current_model_name: Optional[str] = None
        cumulative_prompts = ""
        cumulative_latent_hiddens: Optional[torch.Tensor] = None
        previous_prompt_hidden = torch.empty((0, 0))

        for agent_id, agent in enumerate(agents):
            model_name = args.agent_models[agent_id]
            wrapper = method.models[model_name]
            switched = (
                current_model_name is not None
                and model_name != current_model_name
            )
            if switched:
                source = method.models[current_model_name]
                past, previous_prompt_hidden = _rebuild_switched_cache(
                    target=wrapper,
                    source=source,
                    cumulative_prompts=cumulative_prompts,
                    cumulative_latent_hiddens=cumulative_latent_hiddens,
                )
            cache_length_before_prompt = _past_length(past)

            candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            if current_model_name is not None:
                for position, vector in enumerate(previous_prompt_hidden):
                    candidates["previous_kv_prompt_hidden"].append(
                        _state(
                            vector=vector,
                            item_id=item_id,
                            role=agent.role,
                            agent_id=agent_id,
                            state_kind="previous_kv_prompt_hidden",
                            position=position,
                            model_name=model_name,
                        )
                    )

            prompt_args = argparse.Namespace(
                model_name=model_name,
                task=args.dataset,
            )
            messages = build_agent_message_sequential_latent_mas(
                role=agent.role,
                question=question,
                context="",
                method="latent_mas_hybrid",
                args=prompt_args,
            )
            prompt = wrapper.render_chat(messages, add_generation_prompt=True)
            if args.think:
                prompt += "<think>"
            encoded = wrapper.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(wrapper.device)
            attention_mask = encoded["attention_mask"].to(wrapper.device)
            role_record = {
                "agent_id": int(agent_id),
                "role": agent.role,
                "model_name": model_name,
                "model_switched": bool(switched),
                "previous_model_name": current_model_name,
                "kv_length_before_prompt": int(cache_length_before_prompt),
                "messages": _json_safe(messages),
                "rendered_prompt": prompt,
                "prompt_token_ids": input_ids[0].detach().cpu().tolist(),
                "prompt_attention_mask": attention_mask[0].detach().cpu().tolist(),
                "prompt_token_count": int(attention_mask.sum().item()),
            }

            if agent.role != "judger":
                past, new_prompt_hidden, latent_hidden = _latent_rollout(
                    wrapper,
                    input_ids,
                    attention_mask,
                    past,
                    args.latent_steps,
                )
                for position, vector in enumerate(latent_hidden):
                    candidates["latent_reply_hidden"].append(
                        _state(
                            vector=vector,
                            item_id=item_id,
                            role=agent.role,
                            agent_id=agent_id,
                            state_kind="latent_reply_hidden",
                            position=position,
                            model_name=model_name,
                        )
                    )
                if switched or cumulative_latent_hiddens is None:
                    cumulative_latent_hiddens = latent_hidden
                else:
                    cumulative_latent_hiddens = torch.cat(
                        [cumulative_latent_hiddens, latent_hidden], dim=0
                    )
                role_record.update(
                    {
                        "reply_kind": "latent",
                        "latent_step_count": int(latent_hidden.shape[0]),
                        "kv_length_after_reply": int(_past_length(past)),
                    }
                )
            else:
                new_prompt_hidden, text_hidden, generated_ids = _sample_judger(
                    wrapper, input_ids, attention_mask, past, args
                )
                for position, vector in enumerate(text_hidden):
                    candidates["text_reply_hidden"].append(
                        _state(
                            vector=vector,
                            item_id=item_id,
                            role=agent.role,
                            agent_id=agent_id,
                            state_kind="text_reply_hidden",
                            position=position,
                            model_name=model_name,
                        )
                    )
                logger.info(
                    "Phase A: item_id=%d Judger sampled %d tokens.",
                    item_id,
                    len(generated_ids),
                )
                role_record.update(
                    {
                        "reply_kind": "text",
                        "generated_token_ids": [int(token) for token in generated_ids],
                        "generated_tokens": wrapper.tokenizer.convert_ids_to_tokens(
                            generated_ids
                        ),
                        "generated_text": wrapper.tokenizer.decode(
                            generated_ids, skip_special_tokens=False
                        ),
                        "generated_text_skip_special_tokens": wrapper.tokenizer.decode(
                            generated_ids, skip_special_tokens=True
                        ),
                        "generated_token_count": len(generated_ids),
                    }
                )

            for position, vector in enumerate(new_prompt_hidden):
                candidates["new_prompt_hidden"].append(
                    _state(
                        vector=vector,
                        item_id=item_id,
                        role=agent.role,
                        agent_id=agent_id,
                        state_kind="new_prompt_hidden",
                        position=position,
                        model_name=model_name,
                    )
                )

            for state_candidates in candidates.values():
                saved_states.extend(state_candidates)
            role_record["saved_state_counts"] = {
                state_kind: len(state_candidates)
                for state_kind, state_candidates in candidates.items()
            }
            question_record["roles"].append(role_record)

            # These are the real prompt hiddens represented by the current cache.
            previous_prompt_hidden = (
                new_prompt_hidden
                if not previous_prompt_hidden.numel()
                else torch.cat(
                    [previous_prompt_hidden.to(new_prompt_hidden.device), new_prompt_hidden],
                    dim=0,
                )
            )
            cumulative_prompts += prompt
            current_model_name = model_name

        logger.info(
            "Phase A: question %d/%d completed; %d complete states total.",
            question_number,
            len(indexed_items),
            len(saved_states),
        )
    return saved_states, question_records

