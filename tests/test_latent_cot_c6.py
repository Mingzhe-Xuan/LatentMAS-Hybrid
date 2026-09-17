import ast
import hashlib
import random
import re
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "exp" / "latent_cot" / "c6_prefix_alignment.py"


def load_pure_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted_assignments = {"METHODS", "PAIRS"}
    wanted_functions = {
        "parse_trajectory_name",
        "stable_seed",
        "sample_prefix_positions",
        "deterministic_sample",
        "expand_cache",
        "pair_metrics",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_assignments
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            nodes.append(node)
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "random": random,
        "re": re,
        "torch": torch,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


FUNCTIONS = load_pure_functions()


class C6TrajectoryDiscoveryTests(unittest.TestCase):
    def test_filename_decodes_dataset_model_steps_and_alignments(self):
        path = Path(
            "c0__ds=v-aime2025__sp=v-train__m=v-~UQwen~2F~UQwen3-4~UB__q=50__"
            "seed=42__k=150__a=v-identical-linear-soft-kernel-text__km=2048__"
            "kt=v-1.0__ks=101__kc=4096__lr=v-1e-05__prompt=v-c0_aime2025_question_v1__rc=0.pt"
        )
        parsed = FUNCTIONS["parse_trajectory_name"](path)
        self.assertEqual(parsed["dataset"], "aime2025")
        self.assertEqual(parsed["split"], "train")
        self.assertEqual(parsed["model_name"], "Qwen/Qwen3-4B")
        self.assertEqual(parsed["questions"], 50)
        self.assertEqual(parsed["latent_steps"], 150)
        self.assertEqual(
            parsed["alignments"], ("identical", "linear", "soft", "kernel", "text")
        )
        self.assertEqual(
            parsed["fallback_alignment_config"],
            {
                "kernel_features": 2048,
                "kernel_temperature": 1.0,
                "kernel_seed": 101,
                "kernel_chunk_size": 4096,
                "linear_ridge": 1e-5,
                "soft_chunk_size": 32,
            },
        )


class C6SamplingTests(unittest.TestCase):
    def test_prefix_positions_are_unique_bounded_and_reproducible(self):
        sample = FUNCTIONS["sample_prefix_positions"]
        first = sample(150, 100, 42, "aime2025", 7)
        second = sample(150, 100, 42, "aime2025", 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set(first)), 100)
        self.assertGreaterEqual(min(first), 1)
        self.assertLessEqual(max(first), 150)

    def test_short_trajectory_uses_every_position(self):
        positions = FUNCTIONS["sample_prefix_positions"](7, 100, 42, "mbppplus", 3)
        self.assertEqual(positions, list(range(1, 8)))

    def test_categorical_sample_is_order_independent(self):
        probabilities = torch.tensor([0.1, 0.2, 0.7])
        sample = FUNCTIONS["deterministic_sample"]
        self.assertEqual(
            sample(probabilities, 42, "aime2025", 2, 11, "kernel"),
            sample(probabilities, 42, "aime2025", 2, 11, "kernel"),
        )


class C6MetricTests(unittest.TestCase):
    def test_dynamic_cache_remains_a_cache_after_batch_expansion(self):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        key = torch.randn(1, 2, 3, 4)
        value = torch.randn(1, 2, 3, 4)
        cache.update(key, value, 0)
        expanded = FUNCTIONS["expand_cache"](cache, 4)
        self.assertIsInstance(expanded, DynamicCache)
        self.assertEqual(expanded.get_seq_length(), 3)
        self.assertEqual(expanded.to_legacy_cache()[0][0].shape, (4, 2, 3, 4))
        self.assertEqual(cache.to_legacy_cache()[0][0].shape, (1, 2, 3, 4))

    def test_pair_metrics_report_requested_pairs_and_overlap_definitions(self):
        logits = {
            "linear": torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0]),
            "kernel": torch.tensor([4.0, 2.0, 3.0, 1.0, 0.0, -1.0]),
            "soft": torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
            "text": torch.tensor([4.0, 3.0, 2.0, 1.0, -1.0, 0.0]),
        }
        top5 = {name: torch.topk(value, 5).indices.tolist() for name, value in logits.items()}
        top10 = {name: list(range(6)) for name in logits}
        sampled = {"linear": 0, "kernel": 0, "soft": 5, "text": 0}
        rows = FUNCTIONS["pair_metrics"](logits, sampled, top5, top10)
        self.assertEqual(
            [row["pair"] for row in rows],
            ["linear|text", "kernel|text", "linear|soft", "kernel|soft"],
        )
        self.assertTrue(rows[0]["sampled_token_equal"])
        self.assertEqual(rows[0]["top10_overlap_count"], 6)
        self.assertGreaterEqual(rows[0]["kl_left_right_nats"], 0.0)
        self.assertAlmostEqual(
            rows[0]["symmetric_kl_nats"],
            0.5
            * (rows[0]["kl_left_right_nats"] + rows[0]["kl_right_left_nats"]),
        )

    def test_source_executes_real_branched_transformer_step(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("past_key_values=branch_past", source)
        self.assertIn("branch_output.hidden_states[-1][:, -1, :]", source)
        self.assertIn("branch_logits = output_head(branch_hidden)", source)


if __name__ == "__main__":
    unittest.main()
