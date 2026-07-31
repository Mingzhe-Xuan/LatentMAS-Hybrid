"""C0 single-model latent-recurrence trajectory collection."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Minimal loader for the repository's same-model identical recurrence."""

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


@torch.inference_mode()
def collect_item(
    wrapper: C0Model,
    item_id: int,
    item: Dict[str, Any],
    latent_steps: int,
    dataset: str,
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

            # Match the repository's same-model `identical` recurrence: identity
            # feedback with input-embedding mean-norm scaling. C0's readout below
            # remains W_out-only and does not compare cross-space mappings.
            latent_vec = last_hidden.float()
            latent_vec = latent_vec * (
                wrapper.target_embedding_mean_norm
                / latent_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            )
            latent_embed = latent_vec.to(last_hidden.dtype).unsqueeze(1)
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
        "model_name": wrapper.model_name,
        "dataset": dataset,
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
    for number, (item_id, item) in enumerate(indexed_items, start=1):
        logger.info(
            "C0 Phase A: question %d/%d (item_id=%d) started.",
            number,
            len(indexed_items),
            item_id,
        )
        record = collect_item(
            wrapper, item_id, item, args.latent_steps, args.dataset
        )
        records.append(record)
        logger.info(
            "C0 Phase A: item_id=%d saved %d/%d hidden states; failure=%s.",
            item_id,
            record["valid_step_count"],
            args.latent_steps,
            record["failure_reason"],
        )
        if record["failure_reason"] and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records
