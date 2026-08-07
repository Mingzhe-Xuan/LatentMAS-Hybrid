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


def load_first_items(loader):
    tree = ast.parse(MAS_SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "first_mbppplus_items"
    ]
    namespace = {"load_mbppplus": loader}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MAS_SOURCE), "exec"), namespace)
    return namespace["first_mbppplus_items"]


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
        "CACHE_SCHEMA_VERSION": 2,
        "CACHE_DIR": Path("cache"),
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

        rows = load_first_items(loader)("test", 30)
        self.assertEqual(calls, ["test"])
        self.assertEqual([item_id for item_id, _ in rows], list(range(30)))
        self.assertEqual(rows[-1][1]["question"], "q-29")


class MasArgumentTests(unittest.TestCase):
    def test_c1_defaults_match_experiment_contract(self):
        args = load_parse_args()(["--study", "c1"])
        self.assertEqual(args.model_name, "Qwen/Qwen3-8B")
        self.assertEqual(args.dataset, "mbppplus")
        self.assertEqual(args.max_questions, 30)
        self.assertEqual(
            args.latent_step_values,
            [20, 40, 60, 80, 100, 120, 140, 160, 180],
        )

    def test_c3_defaults_match_shared_experiment_contract(self):
        args = load_parse_args()(["--study", "c3"])
        self.assertEqual(args.model_name, "Qwen/Qwen3-8B")
        self.assertEqual(args.dataset, "mbppplus")
        self.assertEqual(args.max_questions, 30)

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
            "split": "test",
            "model_name": "Qwen/Qwen3-8B",
            "latent_step_values": [20, 40],
            "alignments": ["identical", "linear"],
            "max_questions": 2,
            "generation_seed": 77,
            "align_ridge": 1e-5,
            "kernel_features": 2048,
            "kernel_temperature": 1.0,
            "kernel_seed": 101,
            "kernel_chunk_size": 4096,
            "soft_chunk_size": 32,
            "trust_remote_code": False,
            "max_new_tokens": 4096,
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
            'MAS_ALIGNMENTS = ("identical", "linear", "soft", "kernel")',
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

    def test_greedy_temperature_zero_is_supported(self):
        source = MODEL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("do_sample = temperature > 0", source)
        self.assertIn("if do_sample else {}", source)


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
