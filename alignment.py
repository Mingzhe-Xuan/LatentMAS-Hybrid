"""Latent-to-embedding alignment primitives.

The paper notation in ``docs/algo_detail.md`` uses column vectors.  This
module deliberately uses PyTorch's batched *row-vector* convention:
``hidden`` is ``[..., d_a]``, ``W_out`` is ``[vocab, d_a]`` and ``W_in`` is
``[vocab, d_b]``.  In particular, the paper's ``S u`` is implemented as
``u @ S.T``.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


AlignMethod = Literal["identical", "linear", "kernel"]


@dataclass
class AlignmentState:
    method: AlignMethod
    target_norm: torch.Tensor
    matrix: Optional[torch.Tensor] = None
    omega: Optional[torch.Tensor] = None  # [m, d_a]
    numerator: Optional[torch.Tensor] = None  # S, [d_b, m]
    denominator: Optional[torch.Tensor] = None  # z, [m]
    temperature: float = 1.0


def build_orf(feature_count: int, dimension: int, *, seed: int, device: torch.device) -> torch.Tensor:
    """Build the block orthogonal Gaussian directions Omega from section 3."""
    if feature_count <= 0 or dimension <= 0:
        raise ValueError("feature_count and dimension must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    blocks = []
    while sum(block.shape[0] for block in blocks) < feature_count:
        gaussian = torch.randn((dimension, dimension), generator=generator, device=device, dtype=torch.float32)
        q, r = torch.linalg.qr(gaussian)
        # Fix QR signs so Q has the Haar distribution induced by the Gaussian.
        signs = torch.sign(torch.diagonal(r))
        signs[signs == 0] = 1
        q = q * signs.unsqueeze(0)
        radii = gaussian.norm(dim=0)
        blocks.append(radii.unsqueeze(1) * q.T)  # rows are omega_r^T
    return torch.cat(blocks, dim=0)[:feature_count]


def positive_features(x: torch.Tensor, omega: torch.Tensor, *, stabilize: bool = False) -> torch.Tensor:
    """Return phi_orth(x), with x represented by row vectors.

    When ``stabilize`` is true, one common log-scale is removed per query.
    This does not change ``(S u) / (z^T u)`` and is therefore used online.
    Offline key features must use the unshifted definition.
    """
    if x.shape[-1] != omega.shape[-1]:
        raise ValueError(f"Feature dimension mismatch: {x.shape[-1]} vs {omega.shape[-1]}")
    original_shape = x.shape[:-1]
    flat_x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float32)
    log_features = flat_x @ omega.T - 0.5 * flat_x.square().sum(dim=-1, keepdim=True)
    if stabilize:
        log_features = log_features - log_features.max(dim=-1, keepdim=True).values
    features = torch.exp(log_features) / (omega.shape[0] ** 0.5)
    return features.reshape(*original_shape, omega.shape[0])


def build_kernel_state(
    output_weight: torch.Tensor,
    input_weight: torch.Tensor,
    output_bias: Optional[torch.Tensor],
    *,
    feature_count: int,
    temperature: float,
    seed: int,
    chunk_size: int,
) -> AlignmentState:
    """Pre-aggregate S and z in sections 5--8 of algo_detail.md."""
    if output_weight.ndim != 2 or input_weight.ndim != 2:
        raise ValueError("Embedding weights must be rank-2 tensors")
    vocab, d_a = output_weight.shape
    if input_weight.shape[0] != vocab:
        raise ValueError("Kernel alignment requires equal vocabulary sizes")
    if temperature <= 0:
        raise ValueError("kernel temperature must be positive")
    device = output_weight.device
    omega = build_orf(feature_count, d_a, seed=seed, device=device)
    d_b = input_weight.shape[1]
    s = torch.zeros((d_b, feature_count), device=device, dtype=torch.float32)
    z = torch.zeros(feature_count, device=device, dtype=torch.float32)
    bias = torch.zeros(vocab, device=device, dtype=torch.float32) if output_bias is None else output_bias.detach().to(device=device, dtype=torch.float32)
    # A common bias shift cancels from S u / z^T u and avoids needless overflow.
    bias_shift = bias.max()
    for start in range(0, vocab, max(1, chunk_size)):
        stop = min(start + max(1, chunk_size), vocab)
        keys = positive_features(output_weight[start:stop], omega)
        alpha = torch.exp(bias[start:stop] - bias_shift).unsqueeze(1)
        weighted_keys = alpha * keys
        values = input_weight[start:stop].detach().to(device=device, dtype=torch.float32)
        s += values.T @ weighted_keys
        z += weighted_keys.sum(dim=0)
    return AlignmentState(
        method="kernel",
        target_norm=input_weight.detach().float().norm(dim=1).mean(),
        omega=omega,
        numerator=s,
        denominator=z,
        temperature=temperature,
    )


def build_linear_state(output_weight: torch.Tensor, input_weight: torch.Tensor, *, ridge: float) -> AlignmentState:
    """Build the row-vector equivalent of the documented least-squares map."""
    if output_weight.shape[0] != input_weight.shape[0]:
        raise ValueError("Linear alignment requires equal vocabulary sizes")
    output = output_weight.detach().float()
    target = input_weight.detach().to(device=output.device, dtype=torch.float32)
    gram = output.T @ output
    matrix = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], device=output.device), output.T @ target)
    return AlignmentState("linear", target.norm(dim=1).mean(), matrix=matrix)


def apply_alignment(hidden: torch.Tensor, state: AlignmentState) -> torch.Tensor:
    """Apply one state to row-vector hidden states of shape ``[..., d_a]``."""
    original_dtype = hidden.dtype
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).float()
    if state.method == "identical":
        aligned = flat_hidden
        # Existing behaviour: identity mapping followed by embedding-norm scaling.
        aligned = aligned * (state.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    elif state.method == "linear":
        assert state.matrix is not None
        aligned = flat_hidden @ state.matrix
        aligned = aligned * (state.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    elif state.method == "kernel":
        assert state.omega is not None and state.numerator is not None and state.denominator is not None
        u = positive_features(flat_hidden / state.temperature, state.omega, stabilize=True)
        denom = u @ state.denominator
        if not torch.isfinite(denom).all() or (denom <= torch.finfo(denom.dtype).eps).any():
            raise FloatingPointError("Kernel alignment denominator is non-positive or non-finite")
        # Paper: h_B = S u / (z^T u).  Here u is a row vector: u @ S.T.
        aligned = (u @ state.numerator.T) / denom.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported alignment method {state.method}")
    return aligned.reshape(*hidden.shape[:-1], aligned.shape[-1]).to(original_dtype)
