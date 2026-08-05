"""Regression tests for model-neutral prompts and modern HF caches."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
    build_agent_messages_hierarchical_text_mas,
    build_agent_messages_sequential_text_mas,
    build_agent_messages_single_agent,
)
from reasoning_models import model_adds_think_tag, resolve_manual_think

try:
    import torch
    import models
except ModuleNotFoundError:
    torch = None
    models = None


def _args(model_name: str, method: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        method=method,
        task="arc_easy",
        text_mas_context_length=-1,
    )


class PromptModelFamilyTests(unittest.TestCase):
    def test_qwen_system_identity_is_preserved(self):
        messages = build_agent_messages_single_agent(
            "Question", args=_args("Qwen/Qwen3-8B", "baseline")
        )
        self.assertEqual(
            messages[0]["content"],
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        )

    def test_deepseek_is_accepted_by_all_prompt_builders(self):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        cases = (
            build_agent_message_sequential_latent_mas(
                "planner", "Question", method="latent_mas", args=_args(model_name, "latent_mas")
            ),
            build_agent_message_hierarchical_latent_mas(
                "planner", "Question", method="latent_mas", args=_args(model_name, "latent_mas")
            ),
            build_agent_messages_sequential_text_mas(
                "planner", "Question", method="text_mas", args=_args(model_name, "text_mas")
            ),
            build_agent_messages_hierarchical_text_mas(
                "planner", "Question", method="text_mas", args=_args(model_name, "text_mas")
            ),
            build_agent_messages_single_agent(
                "Question", args=_args(model_name, "baseline")
            ),
        )
        for messages in cases:
            self.assertEqual(messages[0], {"role": "system", "content": "You are a helpful assistant."})


class ReasoningModelTests(unittest.TestCase):
    def test_deepseek_r1_disables_manual_think_by_default(self):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.assertTrue(model_adds_think_tag(model_name))
        self.assertFalse(resolve_manual_think(model_name, None))

    def test_local_deepseek_checkpoint_is_recognized_by_basename(self):
        model_name = "/models/DeepSeek-R1-Distill-Llama-8B"
        self.assertTrue(model_adds_think_tag(model_name))

    def test_other_models_enable_manual_think_by_default(self):
        self.assertTrue(resolve_manual_think("Qwen/Qwen3-8B", None))

    def test_explicit_cli_choice_overrides_registry_default(self):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.assertTrue(resolve_manual_think(model_name, True))
        self.assertFalse(resolve_manual_think("Qwen/Qwen3-8B", False))


@unittest.skipIf(torch is None or models is None, "torch/transformers are not installed")
class PastLengthTests(unittest.TestCase):
    def test_legacy_tuple_cache(self):
        key = torch.zeros(1, 2, 7, 4)
        value = torch.zeros_like(key)
        self.assertEqual(models._past_length(((key, value),)), 7)

    def test_cache_object_uses_public_sequence_length_api(self):
        class FakeCache:
            def get_seq_length(self):
                return 11

        with patch.object(models, "Cache", FakeCache):
            self.assertEqual(models._past_length(FakeCache()), 11)


if __name__ == "__main__":
    unittest.main()
