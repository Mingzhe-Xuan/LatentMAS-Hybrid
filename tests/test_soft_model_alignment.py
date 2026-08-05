"""ModelWrapper integration checks for exact soft alignment."""

from types import SimpleNamespace
import unittest

import torch

from models import ModelWrapper


class _Tokenizer:
    def __init__(self, vocabulary):
        self._vocabulary = vocabulary

    def get_vocab(self):
        return self._vocabulary


class _Model:
    def __init__(self, output_weight, input_weight, output_bias=None):
        self._output = torch.nn.Linear(
            output_weight.shape[1], output_weight.shape[0],
            bias=output_bias is not None,
        )
        self._input = torch.nn.Embedding(*input_weight.shape)
        with torch.no_grad():
            self._output.weight.copy_(output_weight)
            self._input.weight.copy_(input_weight)
            if output_bias is not None:
                self._output.bias.copy_(output_bias)

    def get_output_embeddings(self):
        return self._output

    def get_input_embeddings(self):
        return self._input


def _wrapper(model, vocabulary, *, temperature=1.0, chunk_size=1):
    wrapper = ModelWrapper.__new__(ModelWrapper)
    wrapper.model = model
    wrapper.tokenizer = _Tokenizer(vocabulary)
    wrapper.align_method = "soft"
    wrapper.args = SimpleNamespace(
        soft_temperature=temperature,
        soft_chunk_size=chunk_size,
    )
    wrapper._alignment_states = {}
    return wrapper


class SoftModelAlignmentTests(unittest.TestCase):
    def test_cross_model_uses_source_probabilities_and_target_embeddings(self):
        source_output = torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.3, 0.5]], dtype=torch.float32
        )
        source_bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
        target_input = torch.tensor(
            [[1.0, 0.0, -0.5], [0.2, 0.8, 0.4], [-0.4, 0.3, 0.9]],
            dtype=torch.float32,
        )
        source = _wrapper(
            _Model(source_output, torch.zeros(3, 2), source_bias),
            {"a": 0, "b": 1, "c": 2},
            temperature=0.8,
        )
        target = _wrapper(
            _Model(torch.zeros(3, 3), target_input),
            {"a": 0, "b": 1, "c": 2},
            temperature=0.8,
        )
        hidden = torch.tensor([[0.15, -0.4], [-0.2, 0.1]])
        expected = torch.softmax(
            (hidden @ source_output.T + source_bias) / 0.8, dim=-1
        ) @ target_input
        torch.testing.assert_close(source.align_hidden_to(hidden, target), expected)

    def test_cross_model_rejects_different_token_to_id_vocabulary(self):
        weight = torch.eye(2)
        source = _wrapper(_Model(weight, weight), {"a": 0, "b": 1})
        target = _wrapper(_Model(weight, weight), {"a": 1, "b": 0})
        with self.assertRaisesRegex(ValueError, "identical token-to-ID"):
            source.align_hidden_to(torch.ones(1, 2), target)


if __name__ == "__main__":
    unittest.main()