import argparse
import ast
import hashlib
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).parents[1]
MAS_SOURCE = ROOT / "exp" / "latent_cot" / "mas_analysis.py"
RUN_SOURCE = ROOT / "exp" / "latent_cot" / "run.py"
MODEL_SOURCE = ROOT / "models.py"
METHOD_SOURCE = ROOT / "methods" / "latent_mas.py"


def load_first_items(mbpp_loader, aime_loader=None):
    tree = ast.parse(MAS_SOURCE.read_text(encoding="utf-8"))
    wanted = {"first_mas_items", "first_mbppplus_items"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {
        "load_mbppplus": mbpp_loader,
        "load_aime2025": aime_loader or mbpp_loader,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(MAS_SOURCE), "exec"),
        namespace,
    )
    return namespace


def load_parse_args():
    tree = ast.parse(RUN_SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    ]
    namespace = {
        "argparse": argparse,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        ),
        "auto_device": lambda value: value,
        "resolve_manual_think": lambda model_name, requested: (
            True if requested is None else requested
        ),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(RUN_SOURCE), "exec"),
        namespace,
    )
    return namespace["parse_args"]


def load_cache_helpers():
    tree = ast.parse(MAS_SOURCE.read_text(encoding="utf-8"))
    wanted = {"_safe_name", "_dataset_identity", "_cache_identity", "_cache_paths"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "re": re,
        "Path": Path,
        "CACHE_SCHEMA_VERSION": 3,
        "CACHE_DIR": Path("cache"),
        "_implementation_sha256": lambda: "implementation-test-hash",
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MAS_SOURCE), "exec"), namespace)
    return namespace["_cache_identity"], namespace["_cache_paths"]


def load_c3_summary():
    tree = ast.parse(MAS_SOURCE.read_text(encoding="utf-8"))
    wanted_functions = {"_bootstrap_mean", "_summarize_c3"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions
    ]
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(MAS_SOURCE), "exec"),
        namespace,
    )
    return namespace["_summarize_c3"]


def load_c2_summary():
    tree = ast.parse(MAS_SOURCE.read_text(encoding="utf-8"))
    wanted_assignments = {"MAS_ALIGNMENTS"}
    wanted_functions = {"_bootstrap_mean", "_summarize_c2"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_assignments
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            nodes.append(node)
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MAS_SOURCE), "exec"), namespace)
    return namespace["_summarize_c2"]


class MasDatasetTests(unittest.TestCase):
    def test_c1_c2_use_dataset_order_prefix(self):
        calls = []

        def loader(split):
            calls.append(split)
            for index in range(40):
                yield {"question": f"q-{index}", "gold": f"g-{index}"}

        rows = load_first_items(loader)["first_mbppplus_items"]("test", 30)
        self.assertEqual(calls, ["test"])
        self.assertEqual([item_id for item_id, _ in rows], list(range(30)))
        self.assertEqual(rows[-1][1]["question"], "q-29")

    def test_aime2025_uses_train_prefix(self):
        calls = []

        def loader(split):
            calls.append(split)
            for index in range(30):
                yield {"question": f"aime-{index}", "gold": str(index)}

        helpers = load_first_items(lambda split: (), loader)
        rows = helpers["first_mas_items"]("aime2025", "train", 30)
        self.assertEqual(calls, ["train"])
        self.assertEqual([item_id for item_id, _ in rows], list(range(30)))


class MasArgumentTests(unittest.TestCase):
    def test_c1_defaults_match_experiment_contract(self):
        args = load_parse_args()(["--study", "c1"])
        self.assertEqual(args.model_name, "Qwen/Qwen3-8B")
        self.assertEqual(args.dataset, "all")
        self.assertEqual(args.max_questions, 30)
        self.assertEqual(
            args.latent_step_values,
            [20, 40, 60, 80, 100, 120, 140, 160, 180],
        )
        self.assertEqual(
            args.alignments,
            ["identical", "linear", "soft", "kernel", "text"],
        )
        self.assertEqual(
            args.aime_latent_step_values,
            [20, 60, 100, 140, 180],
        )
        self.assertEqual(args.max_new_tokens, 20000)
        self.assertEqual(args.generation_seed, 42)
        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(args.top_p, 0.95)
        self.assertTrue(args.think)
        self.assertTrue(args.trust_remote_code)

    def test_c3_defaults_match_shared_experiment_contract(self):
        args = load_parse_args()(["--study", "c3"])
        self.assertEqual(args.model_name, "Qwen/Qwen3-8B")
        self.assertEqual(args.dataset, "all")
        self.assertEqual(args.max_questions, 30)

    def test_main_assigns_dataset_specific_k_grids(self):
        source = RUN_SOURCE.read_text(encoding="utf-8")
        self.assertIn('if dataset == "aime2025"', source)
        self.assertIn('list(args.aime_latent_step_values)', source)
        self.assertIn('else list(args.latent_step_values)', source)

    def test_single_aime2025_dataset_is_accepted(self):
        args = load_parse_args()(["--study", "c2", "--dataset", "aime2025"])
        self.assertEqual(args.dataset, "aime2025")

    def test_c0_defaults_remain_backward_compatible(self):
        args = load_parse_args()([])
        self.assertEqual(args.model_name, "Qwen/Qwen3-4B")
        self.assertEqual(args.dataset, "all")
        self.assertEqual(args.max_questions, 512)


class MasCacheTests(unittest.TestCase):
    @staticmethod
    def args(**overrides):
        values = {
            "study": "c1",
            "dataset": "mbppplus",
            "split": "test",
            "model_name": "Qwen/Qwen3-8B",
            "latent_step_values": [20, 40],
            "alignments": ["identical", "linear"],
            "max_questions": 2,
            "generation_seed": 42,
            "temperature": 0.6,
            "top_p": 0.95,
            "think": True,
            "align_ridge": 1e-5,
            "kernel_features": 2048,
            "kernel_temperature": 1.0,
            "kernel_seed": 101,
            "kernel_chunk_size": 4096,
            "soft_chunk_size": 32,
            "trust_remote_code": True,
            "max_new_tokens": 20000,
            "bootstrap_replicates": 1000,
            "probe_seed": 42,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_study_and_plot_settings_share_rollout_cache(self):
        identity, paths = load_cache_helpers()
        items = [(0, {"question": "q0", "gold": "g0"}), (1, {"question": "q1", "gold": "g1"})]
        first_args = self.args(
            study="c1", bootstrap_replicates=100, probe_seed=1
        )
        second_args = self.args(
            study="c2", bootstrap_replicates=5000, probe_seed=999
        )
        third_args = self.args(study="c3")
        first = identity(first_args, items)
        second = identity(second_args, items)
        third = identity(third_args, items)
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(paths(first_args, first), paths(second_args, second))
        self.assertEqual(paths(first_args, first), paths(third_args, third))

    def test_datasets_use_distinct_cache_identities(self):
        identity, paths = load_cache_helpers()
        items = [(0, {"question": "q0", "gold": "g0"})]
        mbpp_args = self.args(dataset="mbppplus", max_questions=1)
        aime_args = self.args(
            dataset="aime2025", split="train", max_questions=1
        )
        mbpp_identity = identity(mbpp_args, items)
        aime_identity = identity(aime_args, items)
        self.assertNotEqual(mbpp_identity, aime_identity)
        self.assertNotEqual(
            paths(mbpp_args, mbpp_identity),
            paths(aime_args, aime_identity),
        )

    def test_rollout_settings_invalidate_cache(self):
        identity, _ = load_cache_helpers()
        items = [(0, {"question": "q0", "gold": "g0"}), (1, {"question": "q1", "gold": "g1"})]
        baseline = identity(self.args(study="c2"), items)
        self.assertNotEqual(
            baseline,
            identity(self.args(study="c2", generation_seed=78), items),
        )
        self.assertNotEqual(
            baseline,
            identity(self.args(study="c2", max_new_tokens=2048), items),
        )

    def test_cache_hit_skips_model_rollout(self):
        source = MAS_SOURCE.read_text(encoding="utf-8")
        self.assertIn("if cache_hit:", source)
        self.assertIn("rollout skipped: reusing shared C1/C2/C3 cache", source)
        self.assertIn(
            "if not cache_hit:\n            cache_manifest = _save_metrics_cache",
            source,
        )
        self.assertIn("latent_roles_only=False", source)


class MasStudyDefinitionTests(unittest.TestCase):
    def test_default_grid_and_alignments_are_declared(self):
        run_source = RUN_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "default=[20, 40, 60, 80, 100, 120, 140, 160, 180]",
            run_source,
        )
        mas_source = MAS_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            'MAS_ALIGNMENTS = ("identical", "linear", "soft", "kernel", "text")',
            mas_source,
        )
        self.assertIn(
            'LATENT_ROLES = ("planner", "critic", "refiner")',
            mas_source,
        )

    def test_cumulative_step_offsets_each_agent_segment(self):
        source = MAS_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            '"cumulative_step": ROLE_INDEX[role] * self.latent_steps',
            source,
        )
        self.assertIn('"total_latent_steps": 3 * self.latent_steps', source)

    def test_active_latent_mas_exposes_observer_without_changing_default(self):
        model_source = MODEL_SOURCE.read_text(encoding="utf-8")
        method_source = METHOD_SOURCE.read_text(encoding="utf-8")
        self.assertIn("step_observer: Optional[Callable", model_source)
        self.assertIn("latent_step_observer", method_source)
        self.assertIn("latent_roles_only: bool = False", method_source)

    def test_observer_loop_defines_the_step_it_reports(self):
        tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "generate_latent_batch"
        )
        observer_loop = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "step_observer"
                for child in ast.walk(node)
            )
        )
        self.assertIsInstance(observer_loop.target, ast.Name)
        self.assertEqual(observer_loop.target.id, "step")

    def test_c1_rejects_an_all_failed_entropy_run(self):
        source = MAS_SOURCE.read_text(encoding="utf-8")
        self.assertIn("if empty_series:", source)
        self.assertIn("C1 produced no finite entropy observations for:", source)

    def test_text_alignment_uses_greedy_hard_token_feedback(self):
        source = MODEL_SOURCE.read_text(encoding="utf-8")
        self.assertIn('if self.align_method == "text":', source)
        self.assertIn('output_head(last_hidden).argmax(dim=-1)', source)
        self.assertIn('model_inputs = {"input_ids": next_token.unsqueeze(1)}', source)
        self.assertIn('model_inputs = {"inputs_embeds": latent_embed}', source)

    def test_sampling_and_greedy_temperature_zero_are_supported(self):
        source = MODEL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("do_sample = temperature > 0", source)
        self.assertIn("if do_sample else {}", source)
        mas_source = MAS_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"decoding": "sampling" if args.temperature > 0 else "greedy"', mas_source)


@unittest.skipIf(np is None, "NumPy is not installed")
class MasTimeSummaryTests(unittest.TestCase):
    def test_c3_reports_mean_time_per_question(self):
        rows = [
            {
                "alignment": "identical",
                "latent_steps_per_agent": 20,
                "wall_seconds": 2.0,
            },
            {
                "alignment": "identical",
                "latent_steps_per_agent": 20,
                "wall_seconds": 4.0,
            },
        ]
        args = SimpleNamespace(
            dataset="mbppplus",
            alignments=["identical"],
            latent_step_values=[20],
            bootstrap_replicates=20,
            probe_seed=42,
        )
        summary = load_c3_summary()(rows, args)
        point = summary["series"]["identical"][0]
        self.assertEqual(summary["study"], "c3")
        self.assertEqual(point["timed_questions"], 2)
        self.assertEqual(point["mean_seconds_per_question"], 3.0)
        self.assertEqual(point["median_seconds_per_question"], 3.0)
        self.assertEqual(point["total_latent_steps"], 60)


@unittest.skipIf(np is None, "NumPy is not installed")
class MasAccuracySummaryTests(unittest.TestCase):
    def test_failures_remain_in_accuracy_denominator(self):
        rows = [
            {
                "alignment": "identical",
                "latent_steps_per_agent": 20,
                "correct": True,
                "parse_success": True,
            },
            {
                "alignment": "identical",
                "latent_steps_per_agent": 20,
                "correct": False,
                "parse_success": False,
            },
        ]
        args = SimpleNamespace(
            dataset="mbppplus",
            alignments=["identical"],
            latent_step_values=[20],
            bootstrap_replicates=20,
            probe_seed=42,
        )
        point = load_c2_summary()(rows, args)["series"]["identical"][0]
        self.assertEqual(point["processed"], 2)
        self.assertEqual(point["correct"], 1)
        self.assertEqual(point["accuracy"], 0.5)
        self.assertEqual(point["parse_success"], 1)


if __name__ == "__main__":
    unittest.main()
