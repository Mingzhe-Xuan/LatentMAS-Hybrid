"""Model registry for checkpoints whose chat template adds a think tag."""

from typing import Optional


# Store canonical Hugging Face model IDs in lowercase for case-insensitive lookup.
# Add a model here only when its chat template already emits the reasoning prefix.
REASONING_MODEL_NAMES = frozenset(
    {
        "deepseek-ai/deepseek-r1",
        "deepseek-ai/deepseek-r1-distill-llama-8b",
    }
)


def model_adds_think_tag(model_name: str) -> bool:
    """Return whether the model's own chat template supplies ``<think>``."""
    normalized = str(model_name).replace("\\", "/").rstrip("/").lower()
    if normalized in REASONING_MODEL_NAMES:
        return True

    # Also recognize a locally downloaded checkpoint by its canonical basename.
    basename = normalized.rsplit("/", 1)[-1]
    return any(name.rsplit("/", 1)[-1] == basename for name in REASONING_MODEL_NAMES)


def resolve_manual_think(model_name: str, requested: Optional[bool]) -> bool:
    """Resolve the framework-added tag while preserving explicit CLI overrides."""
    if requested is not None:
        return requested
    return not model_adds_think_tag(model_name)
