"""Regression tests for model-neutral prompts and modern HF caches."""

import ast
import unittest
from pathlib import Path
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


class HybridContextFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[1]
        cls.hybrid_source = root.joinpath("methods", "latent_mas_hybrid.py").read_text(encoding="utf-8")
        cls.hybrid_tree = ast.parse(cls.hybrid_source)
        cls.model_tree = ast.parse(root.joinpath("models.py").read_text(encoding="utf-8"))

    def _hybrid_method(self, name):
        hybrid_class = next(
            node
            for node in self.hybrid_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LatentMASMethod"
        )
        return next(
            node
            for node in hybrid_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        )

    def test_context_is_aligned_before_native_prompt_embeddings(self):
        method_source = ast.get_source_segment(
            self.hybrid_source, self._hybrid_method("_combine_context_and_prompt")
        )
        self.assertIn("torch.cat([aligned_context, prompt_embeds], dim=1)", method_source)

    def test_next_context_contains_complete_prefill_then_latent_output(self):
        method_source = ast.get_source_segment(
            self.hybrid_source, self._hybrid_method("_prefill_and_latent")
        )
        self.assertIn("torch.cat([prefill_hidden, latent_hidden_states], dim=1)", method_source)
        self.assertLess(
            method_source.index("last_hidden = outputs.hidden_states[-1][:, -1, :]"),
            method_source.index("latent_hidden_list.append(last_hidden.unsqueeze(1))"),
        )

    def test_new_context_chain_is_the_only_non_vllm_run_batch(self):
        self._hybrid_method("run_batch")
        self.assertNotIn("_run_batch_legacy", self.hybrid_source)

    def test_model_wrapper_supports_embedding_prefix_generation(self):
        wrapper = next(
            node
            for node in self.model_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelWrapper"
        )
        method_names = {node.name for node in wrapper.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("generate_text_from_embeds_batch", method_names)


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
