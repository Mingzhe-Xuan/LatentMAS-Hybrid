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
        if tuple(self.raw["experiments"]["perturbation_alpha"]) != (0, .01, .025, .05, .1):
            raise ValueError("invalid perturbation grid")
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
