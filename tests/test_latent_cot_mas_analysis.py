import argparse
import ast
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

    def test_c0_defaults_remain_backward_compatible(self):
        args = load_parse_args()([])
        self.assertEqual(args.model_name, "Qwen/Qwen3-4B")
        self.assertEqual(args.dataset, "all")
        self.assertEqual(args.max_questions, 512)


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
