from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from analysis.core.schemas import NoiseKey, canonical_json


@dataclass(frozen=True)
class PerturbationDiagnostics:
    relative_noise_norm: float
    original_perturbed_cosine: float
    noise_seed: int


def noise_seed(key: NoiseKey) -> int:
    digest = hashlib.sha256(canonical_json(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def standard_gaussian_like(hidden: torch.Tensor, key: NoiseKey) -> tuple[torch.Tensor, int]:
    seed = noise_seed(key)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(hidden.shape, generator=generator, dtype=torch.float32)
    return noise.to(device=hidden.device, dtype=hidden.dtype), seed


def perturb_source_states(hidden: torch.Tensor, *, alpha: float,
                          noise_key: NoiseKey) -> tuple[torch.Tensor, PerturbationDiagnostics]:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    noise, seed = standard_gaussian_like(hidden, noise_key)
    scale = hidden.float().norm(dim=-1, keepdim=True) / (hidden.shape[-1] ** .5)
    perturbed = hidden + alpha * scale.to(hidden.dtype) * noise
    source_norm = hidden.float().norm(dim=-1)
    delta_norm = (perturbed.float() - hidden.float()).norm(dim=-1)
    relative = torch.where(source_norm > 0, delta_norm / source_norm, torch.zeros_like(source_norm))
    cosine = torch.nn.functional.cosine_similarity(hidden.float(), perturbed.float(), dim=-1)
    return perturbed, PerturbationDiagnostics(float(relative.mean()), float(cosine.mean()), seed)
