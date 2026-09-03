from __future__ import annotations

import ast
from pathlib import Path

from analysis.core.config import load_config
from analysis.core.schemas import KernelConfig, ReceiverCondition


ROOT = Path(__file__).resolve().parents[2]


def test_formal_config_contract() -> None:
    config = load_config(ROOT / "analysis/configs/kernel_analysis.yaml")
    assert config.models == ("Qwen/Qwen3-8B", "Qwen/Qwen3-14B")


def test_clean_and_receiver_only_cache_identity_is_canonical() -> None:
    base = dict(dataset="aime2024", split="train", sender_manifest_hash="x",
                sender_model_id="Qwen/Qwen3-8B", receiver_model_id="Qwen/Qwen3-8B",
                receiver_revision="r", tokenizer_fingerprint="t", prompt_hash="p",
                k=40, alignment="kernel", generation_seed=42)
    assert ReceiverCondition(**base, alpha=0).cache_id == ReceiverCondition(**base, alpha=0.0).cache_id
    baseline_a = ReceiverCondition(**{**base, "k": 0, "sender_manifest_hash": "a"})
    baseline_b = ReceiverCondition(**{**base, "k": 0, "sender_manifest_hash": "b",
                                      "sender_model_id": "Qwen/Qwen3-14B", "alignment": "soft"})
    assert baseline_a.cache_id == baseline_b.cache_id
    baseline_c = ReceiverCondition(**{**base, "k": 0, "sender_manifest_hash": "c",
                                      "alpha": .1, "noise_scheme": "irrelevant"})
    assert baseline_a.cache_id == baseline_c.cache_id


def test_analysis_does_not_import_exp() -> None:
    violations = []
    for path in (ROOT / "analysis").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend((path, alias.name) for alias in node.names if alias.name == "exp" or alias.name.startswith("exp."))
            elif isinstance(node, ast.ImportFrom) and node.module and (node.module == "exp" or node.module.startswith("exp.")):
                violations.append((path, node.module))
    assert not violations
