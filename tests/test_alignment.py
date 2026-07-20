"""Formula-level tests for the row-vector implementation of docs/algo_detail.md."""

import unittest

import torch

from alignment import AlignmentState, apply_alignment, build_kernel_state, build_linear_state, build_orf, positive_features


class KernelAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.w_out = torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.3, 0.5]], dtype=torch.float32)
        self.w_in = torch.tensor([[1.0, 0.0, -0.5], [0.2, 0.8, 0.4], [-0.4, 0.3, 0.9]], dtype=torch.float32)
        self.bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)

    def test_positive_feature_matches_document_formula(self) -> None:
        omega = torch.tensor([[1.0, -2.0], [-0.5, 0.25]], dtype=torch.float32)
        x = torch.tensor([[0.4, -0.3]], dtype=torch.float32)
        actual = positive_features(x, omega)
        expected = torch.exp(x @ omega.T - 0.5 * x.square().sum(dim=-1, keepdim=True)) / (2 ** 0.5)
        torch.testing.assert_close(actual, expected)

    def test_orf_has_orthogonal_directions_per_block(self) -> None:
        omega = build_orf(5, 3, seed=9, device=torch.device("cpu"))
        for block in (omega[:3], omega[3:]):
            directions = block / block.norm(dim=1, keepdim=True)
            torch.testing.assert_close(directions @ directions.T, torch.eye(len(block)), atol=2e-6, rtol=0)

    def test_preaggregation_and_online_formula_match_direct_sum(self) -> None:
        state = build_kernel_state(
            self.w_out, self.w_in, self.bias,
            feature_count=7, temperature=0.8, seed=13, chunk_size=2,
        )
        assert state.omega is not None and state.numerator is not None and state.denominator is not None
        keys = positive_features(self.w_out, state.omega)
        alpha = torch.exp(self.bias - self.bias.max()).unsqueeze(1)
        # Column-vector document formula: S=sum_i c_i alpha_i k_i^T.
        # Row-vector storage: S=W_in.T @ (alpha * K).
        expected_s = self.w_in.T @ (alpha * keys)
        expected_z = (alpha * keys).sum(dim=0)
        torch.testing.assert_close(state.numerator, expected_s)
        torch.testing.assert_close(state.denominator, expected_z)
        # docs use exp(b_i); the common -max(b) shift is applied to both
        # statistics and cancels from the final normalized output.
        raw_alpha = torch.exp(self.bias).unsqueeze(1)
        scale = torch.exp(-self.bias.max())
        torch.testing.assert_close(state.numerator, self.w_in.T @ (raw_alpha * keys) * scale)
        torch.testing.assert_close(state.denominator, (raw_alpha * keys).sum(dim=0) * scale)

        hidden = torch.tensor([[0.15, -0.4], [-0.2, 0.1]], dtype=torch.float32)
        actual = apply_alignment(hidden, state)
        u = positive_features(hidden / state.temperature, state.omega)
        # The common online stabilization factor cancels in this ratio.
        expected = (u @ expected_s.T) / (u @ expected_z).unsqueeze(-1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_kernel_preserves_batch_and_sequence_dimensions(self) -> None:
        state = build_kernel_state(
            self.w_out, self.w_in, self.bias,
            feature_count=7, temperature=1.0, seed=21, chunk_size=2,
        )
        hidden = torch.tensor(
            [[[0.1, 0.2], [-0.3, 0.4], [0.2, -0.1]], [[-0.1, 0.5], [0.3, 0.0], [0.4, -0.2]]],
            dtype=torch.float32,
        )  # [B=2, L=3, d_A=2]
        aligned = apply_alignment(hidden, state)
        self.assertEqual(aligned.shape, (2, 3, 3))  # [B, L, d_B]
        torch.testing.assert_close(aligned.reshape(-1, 3), apply_alignment(hidden.reshape(-1, 2), state))
    def test_linear_uses_row_vector_equivalent_of_document_map(self) -> None:
        state = build_linear_state(self.w_out, self.w_in, ridge=1e-5)
        hidden = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
        raw = hidden @ state.matrix
        expected = raw * (state.target_norm / raw.norm(dim=-1, keepdim=True))
        torch.testing.assert_close(apply_alignment(hidden, state), expected)

    def test_identical_scales_but_does_not_rotate(self) -> None:
        hidden = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        state = AlignmentState("identical", torch.tensor(2.0))
        torch.testing.assert_close(apply_alignment(hidden, state), torch.tensor([[1.2, 1.6]]))

