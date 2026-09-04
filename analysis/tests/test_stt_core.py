from __future__ import annotations

import hashlib
import json
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis.core.cache import CacheError, STTPlannerContextStore
from analysis.core.config import load_config, load_stt_config
from analysis.core.schemas import AnalysisItem, STTPlannerCacheIdentity, STTPlannerItemContext
from analysis.core.stt import (STTArtifactError, STTArtifactSpec, exact_stt,
                               collect_stt_planner_item, evaluate_stt_item,
                               load_stt_artifact, pack_stt_prefix)
from analysis.core.statistics import exact_mcnemar
from analysis.transport.complete_reverse_support import (complete_csc_source_support,
                                                         normalize_runtime_tokenizer)


def _artifact(path: Path, *, source_ids: np.ndarray | None = None) -> STTArtifactSpec:
    source_ids = np.array([2, 0, 1], dtype=np.int64) if source_ids is None else source_ids
    metadata = {
        "schema_version": 1,
        "coordinate_system": "active-support-target-by-source",
        "source_fingerprint": "source-fp",
        "target_fingerprint": "target-fp",
        "provenance": {
            "source_fingerprint": "source-fp",
            "target_fingerprint": "target-fp",
            "requested_revisions": {"source": "source-rev", "target": "target-rev"},
        },
    }
    np.savez(
        path,
        indptr=np.array([0, 1, 3, 4], dtype=np.int64),
        indices=np.array([0, 0, 1, 1], dtype=np.int64),
        data=np.array([1.0, 0.25, 0.75, 1.0], dtype=np.float64),
        shape=np.array([2, 3], dtype=np.int64),
        source_token_ids=source_ids,
        target_token_ids=np.array([1, 3], dtype=np.int64),
        metadata=json.dumps(metadata),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return STTArtifactSpec(path, digest, "source", "target", "source-rev", "target-rev")


def _load(path: Path):
    return load_stt_artifact(
        _artifact(path), source_vocab_size=3, target_vocab_size=4,
        source_fingerprint="source-fp", target_fingerprint="target-fp",
    )


def test_stt_config_is_separate_from_kernel_config() -> None:
    stt = load_stt_config("analysis/configs/bidirectional_stt.yaml")
    assert stt.raw["systems"] == [
        "qwen_only", "mistral_only", "qwen_to_mistral", "mistral_to_qwen"
    ]
    assert load_config("analysis/configs/kernel_analysis.yaml").raw["protocol_version"] == "kernel-analysis-v1"


def test_load_stt_artifact_validates_csc_and_revisions(tmp_path: Path) -> None:
    artifact = _load(tmp_path / "transport.npz")
    assert artifact.shape == (2, 3)
    assert artifact.max_column_mass_error < 1e-12
    assert artifact.sparse_coo("cpu").layout == torch.sparse_coo


def test_load_stt_artifact_fails_closed_on_hash_and_full_support(tmp_path: Path) -> None:
    path = tmp_path / "transport.npz"
    spec = _artifact(path)
    bad_hash = STTArtifactSpec(path, "0" * 64, "source", "target", "source-rev", "target-rev")
    with pytest.raises(STTArtifactError, match="SHA-256"):
        load_stt_artifact(bad_hash, source_vocab_size=3, target_vocab_size=4,
                          source_fingerprint="source-fp", target_fingerprint="target-fp")
    with pytest.raises(STTArtifactError, match="complete sender"):
        load_stt_artifact(spec, source_vocab_size=4, target_vocab_size=4,
                          source_fingerprint="source-fp", target_fingerprint="target-fp")


def test_exact_stt_matches_dense_oracle_and_chunks(tmp_path: Path) -> None:
    artifact = _load(tmp_path / "transport.npz")
    hidden = torch.tensor([[[1.0, -1.0], [0.5, 0.25]]])
    lm_head = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]]))
    embeddings = torch.tensor([
        [9.0, 9.0], [1.0, 2.0], [8.0, 8.0], [3.0, -1.0],
    ])
    actual, diagnostics = exact_stt(hidden, lm_head, embeddings, artifact, tau=0.6)
    chunked, _ = exact_stt(hidden, lm_head, embeddings, artifact, tau=0.6,
                           position_chunk_size=1)

    logits = lm_head(hidden).float()
    full = torch.softmax(logits / 0.6, dim=-1)
    gathered = full[..., [2, 0, 1]]
    dense_transport = torch.tensor([[1.0, 0.25, 0.0], [0.0, 0.75, 1.0]])
    expected = (gathered @ dense_transport.T) @ embeddings[[1, 3]]
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(chunked, expected, atol=1e-6)
    assert diagnostics.source_mass_max_error < 1e-6
    assert diagnostics.target_mass_max_error < 1e-6
    assert diagnostics.aligned_nonfinite_count == 0


def test_pack_stt_prefix_removes_internal_padding_and_sets_positions() -> None:
    sender = torch.tensor([[[1.0], [99.0], [2.0]], [[3.0], [4.0], [99.0]]])
    sender_mask = torch.tensor([[1, 0, 1], [1, 1, 0]])
    receiver = torch.tensor([[[5.0], [6.0]], [[7.0], [99.0]]])
    receiver_mask = torch.tensor([[1, 1], [1, 0]])
    packed, mask, positions = pack_stt_prefix(sender, sender_mask, receiver, receiver_mask)
    assert packed[:, :, 0].tolist() == [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 0.0]]
    assert mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert positions.tolist() == [[0, 1, 2, 3], [0, 1, 2, 0]]


def test_stt_planner_cache_round_trip_and_identity(tmp_path: Path) -> None:
    identity = STTPlannerCacheIdentity(
        "aime2024", "train", "dataset-fp", "first-1", "model", "revision",
        "tokenizer-fp", "prompt-set-fp", 1024,
    )
    store = STTPlannerContextStore(tmp_path)
    handle = store.resolve(identity)
    rendered = "planner prompt"
    store.initialize(
        handle, identity, [0], [{"item_id": 0, "question": "q"}],
        [{"item_id": 0, "role": "planner", "rendered": rendered,
          "sha256": hashlib.sha256(rendered.encode()).hexdigest(), "messages_hash": "messages"}],
    )
    item = STTPlannerItemContext(
        0, hashlib.sha256(b"q").hexdigest(), torch.ones(3, 2), torch.tensor([1, 2, 3]),
        torch.ones(3, dtype=torch.long), rendered, hashlib.sha256(rendered.encode()).hexdigest(),
        "messages", "plan", 2, 1, 0.1, 0.2,
    )
    store.write_item(handle, item)
    store.finalize(handle)
    assert store.validate(handle)["state_count"] == 3
    loaded = store.read_item(handle, 0)
    assert torch.equal(loaded.hidden, item.hidden)
    assert torch.equal(loaded.input_ids, item.input_ids)
    assert loaded.plan_text == "plan"

    incompatible = dataclasses.replace(identity, sender_budget=2048)
    with pytest.raises(CacheError, match="incompatible identity"):
        store.resolve(incompatible, cache_id=handle.cache_id)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __len__(self): return 4
    def get_vocab(self): return {"p": 0, "q": 1, "e": 2, "x": 3}
    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        return {"input_ids": torch.tensor([[0, 1]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long)}
    def decode(self, ids, skip_special_tokens=True): return r"The answer is \boxed{7}."


class _Causal(torch.nn.Module):
    def __init__(self, vocab: int = 4):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, 2)
        self.lm_head = torch.nn.Linear(2, vocab, bias=False)
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.config = SimpleNamespace(_commit_hash="revision")
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[2].fill_(1.0)
    def get_input_embeddings(self): return self.embedding
    def get_output_embeddings(self): return self.lm_head
    def generate(self, input_ids, **kwargs):
        return SimpleNamespace(sequences=torch.cat((input_ids, torch.tensor([[2]])), dim=1))
    def forward(self, input_ids=None, inputs_embeds=None, output_hidden_states=False, **kwargs):
        source = self.embedding(input_ids) if input_ids is not None else inputs_embeds
        logits = self.lm_head(source)
        logits[..., 2] = 100.0
        return SimpleNamespace(logits=logits, hidden_states=(source, source + 1.0),
                               past_key_values=((torch.zeros(1), torch.zeros(1)),))


class _Wrapper:
    def __init__(self, task="aime2024"):
        self.model = _Causal()
        self.tokenizer = _Tokenizer()
        self.device = torch.device("cpu")
        self.args = SimpleNamespace(task=task)
    def render_chat(self, messages, add_generation_prompt=True): return "rendered prompt"


def test_planner_collection_and_stt_receiver_runtime(tmp_path: Path) -> None:
    item = AnalysisItem(0, "q", "7", "7")
    sender, receiver = _Wrapper(), _Wrapper()
    planner = collect_stt_planner_item(item, sender, model_id="sender", sender_budget=1024)
    assert planner.prompt_token_count == 2
    assert planner.plan_token_count == 1
    assert planner.hidden.shape == (3, 2)
    artifact = _load(tmp_path / "transport.npz")
    row = evaluate_stt_item(
        item, receiver, receiver_model_id="receiver", max_new_tokens=4,
        planner=planner, sender=sender, artifact=artifact, position_chunk_size=1,
    )
    assert row.correct
    assert row.aligned_prefix_length == 3
    assert row.receiver_decode_output_tokens == 1
    assert row.diagnostics["prefix_order"] == [
        "aligned_sender_prompt", "aligned_sender_plan", "receiver_native_prompt"
    ]
    assert row.diagnostics["causal_shift"] is False


def test_exact_mcnemar_counts_discordant_pairs() -> None:
    result = exact_mcnemar([1, 1, 0, 0], [1, 0, 1, 0])
    assert result == {"left_only": 1, "right_only": 1, "discordant": 2, "p_value": 1.0}


def test_reverse_support_completion_adds_literal_special_columns() -> None:
    class Source:
        def __len__(self): return 4
        def decode(self, ids, **kwargs): return "special-zero"
        def convert_ids_to_tokens(self, token_id): return f"source-{token_id}"

    class Target:
        def __len__(self): return 4
        def __call__(self, text, add_special_tokens=False): return {"input_ids": [2, 3, 2]}

    completed, records = complete_csc_source_support(
        indptr=np.array([0, 1, 2, 3]), indices=np.array([0, 1, 0]),
        data=np.ones(3), source_token_ids=np.array([1, 2, 3]),
        target_token_ids=np.arange(4), source_tokenizer=Source(), target_tokenizer=Target(),
    )
    assert completed["source_token_ids"].tolist() == [0, 1, 2, 3]
    assert completed["indptr"].tolist() == [0, 2, 3, 4, 5]
    assert completed["indices"][:2].tolist() == [2, 3]
    assert completed["data"][:2].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert records == [{"source_token_id": 0, "literal": "special-zero",
                        "target_token_ids": [2, 3], "weights": pytest.approx([2 / 3, 1 / 3])}]


def test_artifact_tokenizer_normalization_matches_model_wrapper_pad_policy() -> None:
    class Tokenizer:
        pad_token_id = None
        eos_token_id = 2
        eos_token = "</s>"
        pad_token = None
        padding_side = "right"

    tokenizer = normalize_runtime_tokenizer(Tokenizer())
    assert tokenizer.pad_token == "</s>"
    assert tokenizer.padding_side == "left"
