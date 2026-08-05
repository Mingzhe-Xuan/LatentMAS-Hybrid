"""C0 single-model latent-recurrence trajectory collection."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment import (
    AlignmentState,
    apply_alignment,
    build_exact_state,
    build_kernel_state,
    build_linear_state,
)

ALIGNMENTS = ("identical", "linear", "exact", "kernel", "text")

SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
PROMPT_TEMPLATES = {
    "gsm8k": {
        "version": "c0_gsm8k_question_v1",
        "user_template": """Solve the following math problem. Reason step by step.

Question: {question}

Work out the solution carefully.""",
    },
    "mbppplus": {
        "version": "c0_mbppplus_question_v1",
        "user_template": """Solve the following Python programming problem. Reason step by step.

Task: {question}

Work out a correct and self-contained solution carefully.""",
    },
}


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"


def _past_length(past_key_values) -> int:
    if not past_key_values:
        return 0
    return int(past_key_values[0][0].shape[-2])


def prompt_template_version(dataset: str) -> str:
    return PROMPT_TEMPLATES[dataset]["version"]


def prompt_messages(question: str, dataset: str) -> List[Dict[str, str]]:
    template = PROMPT_TEMPLATES[dataset]["user_template"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": template.format(question=question)},
    ]


def prompt_template_sha256(dataset: str) -> str:
    payload = {
        "version": prompt_template_version(dataset),
        "system": SYSTEM_PROMPT,
        "user_template": PROMPT_TEMPLATES[dataset]["user_template"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class C0Model:
    """Minimal loader for same-model aligned latent recurrence."""

    def __init__(self, args):
        self.model_name = args.model_name
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=True,
            token=False,
            trust_remote_code=args.trust_remote_code,
        )
        _ensure_pad_token(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            token=False,
            trust_remote_code=args.trust_remote_code,
        ).to(self.device)
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.target_embedding_mean_norm = (
            self.model.get_input_embeddings().weight.detach().float().norm(dim=1).mean()
        )
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        self._alignment_cache = {}

    def render_chat(self, messages, add_generation_prompt=True):
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        parts = [
            f"<|{message.get('role', 'user')}|>\n{message.get('content', '')}\n"
            f"</|{message.get('role', 'user')}|>"
            for message in messages
        ]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)


def load_model(args) -> C0Model:
    return C0Model(args)


def _empty_hidden(hidden_size: int) -> torch.Tensor:
    return torch.empty((0, hidden_size), dtype=torch.float32, device="cpu")


def build_alignment_states(wrapper: C0Model, args) -> Dict[str, AlignmentState]:
    key = (
        args.kernel_features,
        args.kernel_temperature,
        args.kernel_seed,
        args.kernel_chunk_size,
        args.align_ridge,
    )
    if key in wrapper._alignment_cache:
        return wrapper._alignment_cache[key]
    output_head = wrapper.model.get_output_embeddings()
    input_head = wrapper.model.get_input_embeddings()
    if output_head is None or input_head is None:
        raise RuntimeError("C0 alignment requires input and output embedding weights.")
    output_weight = output_head.weight.detach().float()
    input_weight = input_head.weight.detach().float()
    output_bias = getattr(output_head, "bias", None)
    output_bias = None if output_bias is None else output_bias.detach().float()
    states = {
        "identical": AlignmentState(
            method="identical",
            target_norm=wrapper.target_embedding_mean_norm,
        ),
        "linear": build_linear_state(
            output_weight,
            input_weight,
            ridge=args.align_ridge,
        ),
        "exact": build_exact_state(
            output_weight,
            input_weight,
            output_bias,
            temperature=args.kernel_temperature,
            chunk_size=args.kernel_chunk_size,
        ),
        "kernel": build_kernel_state(
            output_weight,
            input_weight,
            output_bias,
            feature_count=args.kernel_features,
            temperature=args.kernel_temperature,
            seed=args.kernel_seed,
            chunk_size=args.kernel_chunk_size,
        ),
    }
    wrapper._alignment_cache[key] = states
    return states


@torch.inference_mode()
def collect_item(
    wrapper: C0Model,
    item_id: int,
    item: Dict[str, Any],
    latent_steps: int,
    dataset: str,
    alignment: str,
    alignment_state: AlignmentState | None,
) -> Dict[str, Any]:
    messages = prompt_messages(item["question"], dataset)
    rendered_prompt = wrapper.render_chat(messages, add_generation_prompt=True)
    encoded = wrapper.tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(wrapper.device)
    attention_mask = encoded["attention_mask"].to(wrapper.device)
    hidden_size = int(wrapper.model.config.hidden_size)
    hidden_states: List[torch.Tensor] = []
    generated_token_ids: List[int] = []
    final_hidden = _empty_hidden(hidden_size)
    failure_step = None
    failure_reason = None

    try:
        outputs = wrapper.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        for step in range(latent_steps):
            if not torch.isfinite(last_hidden).all():
                failure_step = step
                failure_reason = "non_finite_pre_unembedding_hidden"
                break
            hidden_states.append(
                last_hidden[0].detach().to(device="cpu", dtype=torch.float32)
            )

            model_inputs = {}
            if alignment == "text":
                output_head = wrapper.model.get_output_embeddings()
                if output_head is None:
                    raise RuntimeError("C0 text recurrence requires an output head.")
                next_token = output_head(last_hidden).argmax(dim=-1)
                generated_token_ids.append(int(next_token.item()))
                model_inputs["input_ids"] = next_token.unsqueeze(1)
            else:
                if alignment_state is None:
                    raise RuntimeError(
                        f"Missing C0 alignment state for {alignment}."
                    )
                latent_vec = apply_alignment(last_hidden, alignment_state)
                model_inputs["inputs_embeds"] = latent_vec.to(
                    last_hidden.dtype
                ).unsqueeze(1)
            outputs = wrapper.model(
                **model_inputs,
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
        else:
            if torch.isfinite(last_hidden).all():
                final_hidden = last_hidden[0].detach().to(
                    device="cpu", dtype=torch.float32
                )
            else:
                failure_step = latent_steps
                failure_reason = "non_finite_final_hidden"
    except Exception as error:
        failure_step = len(hidden_states)
        failure_reason = f"{type(error).__name__}: {error}"

    stacked = (
        torch.stack(hidden_states)
        if hidden_states
        else _empty_hidden(hidden_size)
    )
    return {
        "item_id": int(item_id),
        "question": item["question"],
        "messages": messages,
        "rendered_prompt": rendered_prompt,
        "prompt_token_ids": input_ids[0].detach().cpu().tolist(),
        "prompt_attention_mask": attention_mask[0].detach().cpu().tolist(),
        "prompt_token_count": int(attention_mask.sum().item()),
        "generated_token_ids": generated_token_ids,
        "model_name": wrapper.model_name,
        "dataset": dataset,
        "alignment": alignment,
        "prompt_template_version": prompt_template_version(dataset),
        "hidden_states": stacked,
        "final_hidden": final_hidden,
        "valid_step_count": int(stacked.shape[0]),
        "requested_step_count": int(latent_steps),
        "rollout_complete": failure_reason is None,
        "failure_step": failure_step,
        "failure_reason": failure_reason,
    }


def collect(
    wrapper: C0Model,
    indexed_items: List[Tuple[int, Dict[str, Any]]],
    args,
    logger,
) -> List[Dict[str, Any]]:
    records = []
    alignment_states = build_alignment_states(wrapper, args)
    for number, (item_id, item) in enumerate(indexed_items, start=1):
        logger.info(
            "C0 Phase A: question %d/%d (item_id=%d) started.",
            number,
            len(indexed_items),
            item_id,
        )
        for alignment in ALIGNMENTS:
            record = collect_item(
                wrapper,
                item_id,
                item,
                args.latent_steps,
                args.dataset,
                alignment,
                alignment_states.get(alignment),
            )
            records.append(record)
            logger.info(
                "C0 Phase A: item_id=%d alignment=%s saved %d/%d hidden states; "
                "failure=%s.",
                item_id,
                alignment,
                record["valid_step_count"],
                args.latent_steps,
                record["failure_reason"],
            )
            if record["failure_reason"] and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records
