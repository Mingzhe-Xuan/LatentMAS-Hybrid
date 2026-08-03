"""TextMAS Refiner-to-Judger token-embedding collection for S4."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict, List

import torch

from methods import default_agents
from prompts import build_agent_messages_sequential_text_mas


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _context_limit(context: str, configured_limit: int) -> int:
    return len(context) if configured_limit < 0 else configured_limit


def _messages(role, question, context, model_name, args):
    prompt_args = argparse.Namespace(
        model_name=model_name,
        task=args.dataset,
        text_mas_context_length=_context_limit(
            context, args.text_mas_context_length
        ),
    )
    return build_agent_messages_sequential_text_mas(
        role=role,
        question=question,
        context=context,
        method="text_mas",
        args=prompt_args,
    )


def _refiner_token_rows(
    *,
    wrapper,
    prompt: str,
    visible_context: str,
    refiner_start: int,
    refiner_end: int,
    item_id: int,
    model_name: str,
) -> List[Dict[str, Any]]:
    if refiner_end <= refiner_start:
        return []
    context_start = prompt.rfind(visible_context)
    if context_start < 0:
        raise RuntimeError(
            f"TextMAS item {item_id}: visible context is absent from Judger prompt."
        )
    start = context_start + refiner_start
    end = context_start + refiner_end
    try:
        encoded = wrapper.tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as error:
        raise RuntimeError(
            "S4 TextMAS requires a fast Judger tokenizer with offset mappings."
        ) from error
    token_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError(
            "S4 TextMAS requires Judger tokenizer offset mappings."
        )
    selected = [
        (position, int(token_id), int(left), int(right))
        for position, (token_id, (left, right)) in enumerate(zip(token_ids, offsets))
        if right > left and right > start and left < end
    ]
    if not selected:
        raise RuntimeError(
            f"TextMAS item {item_id}: no Judger tokens overlap the Refiner text."
        )
    input_weight = wrapper.model.get_input_embeddings().weight.detach()
    rows = []
    for text_position, (prompt_position, token_id, left, right) in enumerate(selected):
        if token_id >= input_weight.shape[0]:
            raise RuntimeError(
                f"TextMAS item {item_id}: token ID {token_id} exceeds Judger vocabulary."
            )
        rows.append(
            {
                "item_id": int(item_id),
                "role": "refiner",
                "agent_id": 2,
                "turn_id": 0,
                "state_kind": "text_mas_token_embedding",
                "position": int(text_position),
                "prompt_position": int(prompt_position),
                "token_id": token_id,
                "token": wrapper.tokenizer.convert_ids_to_tokens(token_id),
                "character_start": left,
                "character_end": right,
                "model_name": model_name,
                "vector": input_weight[token_id]
                .to(device="cpu", dtype=torch.float32)
                .contiguous(),
            }
        )
    return rows


@torch.inference_mode()
def collect_text_mas(method, indexed_items, args, logger):
    """Run TextMAS through Refiner and collect its actual Judger input tokens."""
    torch.manual_seed(args.probe_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.probe_seed)
    agents = default_agents()
    token_rows: List[Dict[str, Any]] = []
    question_records: List[Dict[str, Any]] = []
    for question_number, (item_id, item) in enumerate(indexed_items, start=1):
        logger.info(
            "S4 TextMAS: question %d/%d (item_id=%d) started.",
            question_number,
            len(indexed_items),
            item_id,
        )
        question = item["question"]
        context = ""
        roles = []
        refiner_text = ""
        refiner_text_start = 0
        for agent_id, agent in enumerate(agents[:3]):
            model_name = args.agent_models[agent_id]
            wrapper = method.models[model_name]
            messages = _messages(
                agent.role, question, context, model_name, args
            )
            prompt, input_ids, attention_mask, _ = wrapper.prepare_chat_input(
                messages, add_generation_prompt=True
            )
            generated, _ = wrapper.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=args.s4_text_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output = generated[0].strip()
            formatted = f"[{agent.name}]:\n{output}\n\n"
            if agent.role == "refiner":
                refiner_text_start = len(context) + len(f"[{agent.name}]:\n")
            context += formatted
            if agent.role == "refiner":
                refiner_text = output
            roles.append(
                {
                    "agent_id": int(agent_id),
                    "role": agent.role,
                    "model_name": model_name,
                    "messages": _json_safe(messages),
                    "rendered_prompt": prompt,
                    "generated_text": output,
                    "generated_token_count": int(
                        wrapper.last_generation_metrics["output_token_counts"][0]
                    ),
                }
            )

        judger_name = args.agent_models[3]
        judger = method.models[judger_name]
        judger_messages = _messages(
            "judger", question, context, judger_name, args
        )
        judger_prompt = judger.render_chat(
            judger_messages, add_generation_prompt=True
        )
        visible_context_length = _context_limit(
            context, args.text_mas_context_length
        )
        visible_context = context[:visible_context_length]
        visible_refiner_end = min(
            refiner_text_start + len(refiner_text), visible_context_length
        )
        visible_refiner_text = context[
            refiner_text_start:visible_refiner_end
        ]
        rows = _refiner_token_rows(
            wrapper=judger,
            prompt=judger_prompt,
            visible_context=visible_context,
            refiner_start=refiner_text_start,
            refiner_end=visible_refiner_end,
            item_id=item_id,
            model_name=judger_name,
        )
        token_rows.extend(rows)
        source_record = _json_safe(item)
        question_records.append(
            {
                "item_id": int(item_id),
                "question": question,
                "source_record": source_record,
                "source_record_sha256": hashlib.sha256(
                    json.dumps(
                        source_record, sort_keys=True, ensure_ascii=False
                    ).encode("utf-8")
                ).hexdigest(),
                "roles": roles,
                "judger_model_name": judger_name,
                "judger_messages": _json_safe(judger_messages),
                "judger_rendered_prompt": judger_prompt,
                "refiner_text": refiner_text,
                "visible_refiner_text": visible_refiner_text,
                "refiner_judger_token_count": len(rows),
            }
        )
        logger.info(
            "S4 TextMAS: item_id=%d yielded %d Refiner tokens in Judger space.",
            item_id,
            len(rows),
        )
    return token_rows, question_records
