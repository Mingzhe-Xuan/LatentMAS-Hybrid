from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRIMARY_DATASETS = ("aime2024", "humanevalplus", "arc_challenge")
ALL_DATASETS = (
    "aime2024", "aime2025", "arc_challenge", "arc_easy", "gpqa",
    "gsm8k", "humanevalplus", "mbppplus", "medqa",
)
SPLITS = {
    "aime2024": "train", "aime2025": "train", "arc_challenge": "test",
    "arc_easy": "test", "gpqa": "test", "gsm8k": "test",
    "humanevalplus": "test", "mbppplus": "test", "medqa": "train",
}


@dataclass(frozen=True)
class AnalysisConfig:
    raw: dict[str, Any]

    @property
    def models(self) -> tuple[str, str]:
        return tuple(self.raw["models"])  # type: ignore[return-value]

    def validate(self) -> "AnalysisConfig":
        required = {"protocol_version", "models", "datasets", "generation", "kernel", "experiments"}
        unknown = set(self.raw) - required
        missing = required - set(self.raw)
        if missing or unknown:
            raise ValueError(f"configuration keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
        if tuple(self.raw["models"]) != ("Qwen/Qwen3-8B", "Qwen/Qwen3-14B"):
            raise ValueError("formal model list must be Qwen3-8B, Qwen3-14B")
        if tuple(self.raw["generation"]["seeds"]) != (42, 43, 44):
            raise ValueError("generation seeds must be exactly 42--44")
        kernel = self.raw["kernel"]
        if (kernel["features"], kernel["temperature"], kernel["seed"], kernel["chunk_size"]) != (2048, .6, 101, 4096):
            raise ValueError("canonical Kernel parameters do not match the protocol")
        if tuple(self.raw["experiments"]["scaling_k"]) != (0, 10, 20, 40, 80, 160):
            raise ValueError("invalid scaling grid")
        if tuple(self.raw["experiments"]["perturbation_alpha"]) != (0, .01, .05, .1):
            raise ValueError("invalid perturbation grid")
        return self


@dataclass(frozen=True)
class STTAnalysisConfig:
    raw: dict[str, Any]

    def validate(self) -> "STTAnalysisConfig":
        required = {"protocol_version", "models", "model_revisions", "datasets",
                    "generation", "transport", "systems"}
        unknown = set(self.raw) - required
        missing = required - set(self.raw)
        if missing or unknown:
            raise ValueError(f"STT configuration keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
        if self.raw["protocol_version"] != "bidirectional-stt-v1":
            raise ValueError("unsupported STT protocol version")
        models = self.raw["models"]
        if set(models) != {"qwen", "mistral"} or any(not isinstance(value, str) or not value for value in models.values()):
            raise ValueError("STT models must define non-empty qwen and mistral IDs")
        revisions = self.raw["model_revisions"]
        if (set(revisions) != set(models)
                or any(not isinstance(value, str) or len(value) != 40 for value in revisions.values())):
            raise ValueError("STT model revisions must be full qwen and mistral commit hashes")
        datasets = self.raw["datasets"]
        if tuple(datasets) != PRIMARY_DATASETS:
            raise ValueError(f"STT datasets must be exactly {PRIMARY_DATASETS}")
        for dataset, task in datasets.items():
            if task.get("split") != SPLITS[dataset]:
                raise ValueError(f"invalid STT split for {dataset}")
            if int(task.get("max_new_tokens", 0)) <= 0 or int(task.get("generation_batch_size", 0)) <= 0:
                raise ValueError(f"invalid STT generation settings for {dataset}")
        generation = self.raw["generation"]
        if (generation.get("sender_budget"), generation.get("do_sample")) != (1024, False):
            raise ValueError("STT requires sender_budget=1024 and greedy decoding")
        transport = self.raw["transport"]
        if (transport.get("tau"), transport.get("causal_shift"),
                transport.get("accumulation_dtype")) != (0.6, False, "float32"):
            raise ValueError("STT transport parameters do not match the formal protocol")
        if (int(transport.get("position_chunk_size", 0)) <= 0
                or int(transport.get("target_chunk_size", 0)) <= 0):
            raise ValueError("STT position and target chunk sizes must be positive")
        artifacts = transport.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {"qwen_to_mistral", "mistral_to_qwen"}:
            raise ValueError("STT requires both directed transport artifacts")
        required_artifact = {"path", "sha256", "source", "target", "source_revision", "target_revision"}
        directions = {"qwen_to_mistral": ("qwen", "mistral"),
                      "mistral_to_qwen": ("mistral", "qwen")}
        for name, direction in directions.items():
            value = artifacts[name]
            if set(value) != required_artifact or (value["source"], value["target"]) != direction:
                raise ValueError(f"invalid STT artifact declaration for {name}")
            if len(value["sha256"]) != 64 or any(not value[key] for key in required_artifact):
                raise ValueError(f"incomplete STT artifact declaration for {name}")
            if (value["source_revision"], value["target_revision"]) != (
                    revisions[direction[0]], revisions[direction[1]]):
                raise ValueError(f"STT artifact revisions do not match locked models for {name}")
        expected_systems = ("qwen_only", "mistral_only", "qwen_to_mistral", "mistral_to_qwen")
        if tuple(self.raw["systems"]) != expected_systems:
            raise ValueError(f"STT systems must be exactly {expected_systems}")
        return self


def load_config(path: str | Path) -> AnalysisConfig:
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("configuration is not JSON and PyYAML is unavailable") from exc
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return AnalysisConfig(raw).validate()


def load_stt_config(path: str | Path) -> STTAnalysisConfig:
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("configuration is not JSON and PyYAML is unavailable") from exc
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return STTAnalysisConfig(raw).validate()
