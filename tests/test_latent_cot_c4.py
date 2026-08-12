import argparse
import ast
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None

ROOT = Path(__file__).parents[1]
RUN_SOURCE = ROOT / "exp" / "latent_cot" / "run.py"
C4_SOURCE = ROOT / "exp" / "latent_cot" / "c4_noise_ablation.py"
MODEL_SOURCE = ROOT / "models.py"
METHOD_SOURCE = ROOT / "methods" / "latent_mas.py"


def load_parse_args():
    tree = ast.parse(RUN_SOURCE.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = {
        "argparse": argparse,
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "auto_device": lambda value: value,
        "resolve_manual_think": lambda model, requested: True if requested is None else requested,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUN_SOURCE), "exec"), namespace)
    return namespace["parse_args"]


def load_hidden_transform():
    tree = ast.parse(C4_SOURCE.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SAMPLED_STEPS" for target in node.targets):
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "HiddenTransform":
            nodes.append(node)
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(C4_SOURCE), "exec"), namespace)
    return namespace["HiddenTransform"]


def load_identity():
    tree = ast.parse(C4_SOURCE.read_text(encoding="utf-8"))
    wanted = {"_dataset_identity", "_identity"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "SCHEMA_VERSION": 1,
        "ROLES": ("planner", "critic", "refiner"),
        "_implementation_sha": lambda: "test-code",
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(C4_SOURCE), "exec"), namespace)
    return namespace["_identity"]


class C4ContractTests(unittest.TestCase):
    def test_defaults_are_fixed_to_requested_matrix(self):
        args = load_parse_args()(["--study", "c4"])
        self.assertEqual(args.model_name, "Qwen/Qwen3-8B")
        self.assertEqual(args.dataset, "aime2025")
        self.assertEqual(args.split, "train")
        self.assertEqual(args.max_questions, 30)
        self.assertEqual(args.latent_steps, 120)
        self.assertEqual(args.alignments, ["linear", "kernel"])
        self.assertEqual(args.repeat_seeds, [42, 43, 44, 45])
        self.assertEqual(args.generate_bs, 2)
        self.assertEqual(args.max_new_tokens, 20000)

    def test_rollout_matrix_has_480_question_rows(self):
        source = C4_SOURCE.read_text(encoding="utf-8")
        self.assertIn('ALIGNMENTS = ("linear", "kernel")', source)
        self.assertIn('CONDITIONS = ("clean", "random_hidden")', source)
        self.assertEqual(2 * 2 * 4 * 30, 480)
        self.assertNotIn("kernel_early_stopping", source)
        self.assertNotIn('"soft"', source)

    def test_cache_identity_separates_condition_alignment_and_seed(self):
        identity = load_identity()
        args = SimpleNamespace(
            model_name="Qwen/Qwen3-8B", noise_seed_offset=10000,
            temperature=0.6, top_p=0.95, max_new_tokens=20000,
            generate_bs=2, think=True, align_ridge=1e-5,
            kernel_features=1024, kernel_temperature=0.6,
            kernel_seed=101, kernel_chunk_size=4096,
        )
        items = [(0, {"question": "q", "gold": "1"})]
        baseline = identity(args, items, "linear", "clean", 42)
        self.assertNotEqual(baseline, identity(args, items, "linear", "random_hidden", 42))
        self.assertNotEqual(baseline, identity(args, items, "kernel", "clean", 42))
        self.assertNotEqual(baseline, identity(args, items, "linear", "clean", 43))

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_random_hidden_is_deterministic_and_norm_matched(self):
        transform_cls = load_hidden_transform()
        hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, -1.0, 2.0]])
        first = transform_cls("random_hidden", 123, "linear", 42)
        first.begin_batch([0, 1])
        out1 = first("planner", 0, hidden)
        second = transform_cls("random_hidden", 123, "linear", 42)
        second.begin_batch([0, 1])
        out2 = second("planner", 0, hidden)
        self.assertTrue(torch.allclose(out1, out2))
        self.assertTrue(torch.allclose(out1.norm(dim=-1), hidden.norm(dim=-1), rtol=1e-5, atol=1e-6))
        self.assertFalse(torch.allclose(out1, hidden))

    def test_transform_hook_covers_prefill_and_recurrent_outputs(self):
        model_source = MODEL_SOURCE.read_text(encoding="utf-8")
        method_source = METHOD_SOURCE.read_text(encoding="utf-8")
        self.assertIn("last_hidden = hidden_state_transform(0, last_hidden)", model_source)
        self.assertIn("last_hidden = hidden_state_transform(step + 1, last_hidden)", model_source)
        self.assertIn("hidden_state_transform=hidden_transform", method_source)
        self.assertIn("latent_hidden_transform:Optional[", method_source.replace(" ", ""))

    def test_plot_only_rerun_uses_all_cell_caches(self):
        source = C4_SOURCE.read_text(encoding="utf-8")
        self.assertIn("if not cache_hit:", source)
        self.assertIn('logger.info("C4 cache hit:', source)
        self.assertIn("all_cache_hits", source)
        self.assertIn("--reuse_trajectory requested but C4 cache is absent", source)


if __name__ == "__main__":
    unittest.main()
