from __future__ import annotations

from types import SimpleNamespace

import torch

from analysis.core.artifacts import summarize_condition_metrics
from analysis.core.receiver import evaluate_receiver_batch, evaluate_receiver_item
from analysis.core.schemas import (AnalysisItem, KernelConfig, ReceiverCondition,
                                   SenderConfig)
from analysis.core.sender import collect_sender_item


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3
    def get_vocab(self): return {"<pad>": 0, "x": 1, "<eos>": 2, "?": 3}
    def __call__(self, prompts, return_tensors="pt", padding=False, add_special_tokens=False):
        batch = len(prompts) if isinstance(prompts, list) else 1
        return {"input_ids": torch.tensor([[0, 1]] * batch),
                "attention_mask": torch.ones(batch, 2, dtype=torch.long)}


class FakeEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    def forward(self, ids): return torch.nn.functional.embedding(ids, self.weight)


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = FakeEmbedding()
        self.calls = 0
    def get_input_embeddings(self): return self.embedding
    def get_output_embeddings(self): return self.embedding
    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.calls += 1
        source = self.embedding(input_ids) if input_ids is not None else inputs_embeds
        batch, length, dim = source.shape
        # Other layers and positions are sentinels; only the final layer/position equals calls.
        wrong = torch.full((batch, length, dim), -1000.0 - self.calls)
        final = torch.full((batch, length, dim), -100.0 - self.calls)
        final[:, -1, :] = float(self.calls)
        past = ((torch.zeros(batch, 1, self.calls, dim), torch.zeros(batch, 1, self.calls, dim)),)
        return SimpleNamespace(hidden_states=(wrong, final), past_key_values=past)


class FakeWrapper:
    def __init__(self):
        self.model = FakeBase()
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizer()
        self.align_method = "kernel"
        self.args = SimpleNamespace()
        self._alignment_states = {}
        self.last_generation_metrics = {}
        self.generated_shape = None
        self.full_vocab_evaluations = 0
    def render_chat(self, messages, add_generation_prompt=True):
        return "rendered:" + messages[-1]["content"]
    def prepare_chat_input(self, messages, add_generation_prompt=True):
        return "rendered:" + messages[-1]["content"], torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.long), []
    def prepare_chat_batch(self, batch_messages, add_generation_prompt=True):
        batch = len(batch_messages)
        return (["rendered:" + messages[-1]["content"] for messages in batch_messages],
                torch.tensor([[0, 1]] * batch), torch.ones(batch, 2, dtype=torch.long), [[]] * batch)
    def _apply_latent_realignment(self, hidden, source): return hidden + 10
    def align_hidden_to(self, hidden, target):
        if self.align_method == "soft":
            self.full_vocab_evaluations += hidden.numel() // hidden.shape[-1]
        return hidden + 20
    def generate_text_from_embeds_batch(self, inputs_embeds, attention_mask, **kwargs):
        self.generated_shape = tuple(inputs_embeds.shape)
        batch = inputs_embeds.shape[0]
        self.last_generation_metrics = {"prefill_seconds": .2, "text_decode_seconds": .3,
                                        "output_token_counts": [3] * batch}
        return ["The answer is \\boxed{7}."] * batch, None
    def tokenize_text(self, text): return torch.tensor([[1, 2, 3]])


def test_sender_captures_only_final_layer_last_position_h1_to_hk() -> None:
    wrapper = FakeWrapper()
    item = AnalysisItem(0, "question", "7", "7")
    config = SenderConfig("aime2024", "train", "Qwen/Qwen3-8B", "r", "t", "d", "", 4,
                          dtype="float32", kernel=KernelConfig())
    trajectory = collect_sender_item(item, wrapper, config)
    assert trajectory.hidden.shape == (4, 4)
    assert torch.equal(trajectory.h0, torch.ones(1, 4))
    assert torch.equal(trajectory.hidden[:, 0], torch.tensor([2., 3., 4., 5.]))
    assert torch.all(trajectory.hidden > 0)  # rejects wrong layer/position sentinels
    assert wrapper.model.calls == 5


def test_receiver_injects_exact_prefix_and_counts_readouts() -> None:
    receiver = FakeWrapper()
    item = AnalysisItem(0, "question", "7", "7")
    sender = collect_sender_item(item, FakeWrapper(), SenderConfig(
        "aime2024", "train", "Qwen/Qwen3-8B", "r", "t", "d", "", 4,
        dtype="float32", kernel=KernelConfig()))
    condition = ReceiverCondition("aime2024", "train", "manifest", "Qwen/Qwen3-8B",
        "Qwen/Qwen3-8B", "r", "t", "", 2, "soft", generation_seed=42)
    row = evaluate_receiver_item(item, sender, condition, receiver)
    assert receiver.generated_shape == (1, 4, 4)  # two prompt embeddings + h1,h2
    assert row.correct
    assert row.transfer_alignment_output_tokens == 2
    assert receiver.full_vocab_evaluations == row.transfer_alignment_output_tokens
    assert row.receiver_decode_output_tokens == 3
    summary = summarize_condition_metrics([row], execution_wall_seconds=1.0)
    assert summary["results"]["output_tokens"]["total"] == 5
    assert summary["timing"]["total_seconds"] >= sender.cumulative_sender_seconds[1] + .5


def test_receiver_batch_counts_per_sample_not_per_launch() -> None:
    receiver = FakeWrapper()
    items = [AnalysisItem(i, f"question {i}", "7", "7") for i in range(2)]
    trajectories = [collect_sender_item(item, FakeWrapper(), SenderConfig(
        "aime2024", "train", "Qwen/Qwen3-8B", "r", "t", "d", "", 2,
        dtype="float32", kernel=KernelConfig())) for item in items]
    condition = ReceiverCondition("aime2024", "train", "manifest", "Qwen/Qwen3-8B",
        "Qwen/Qwen3-8B", "r", "t", "", 2, "soft", generation_seed=42)
    rows = evaluate_receiver_batch(items, trajectories, condition, receiver)
    assert receiver.generated_shape == (2, 4, 4)
    assert [row.transfer_alignment_output_tokens for row in rows] == [2, 2]
    assert receiver.full_vocab_evaluations == sum(row.transfer_alignment_output_tokens for row in rows)
    assert [row.receiver_decode_output_tokens for row in rows] == [3, 3]
    assert .2 <= sum(row.receiver_prefill_seconds for row in rows) < .21
