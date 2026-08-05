import ast
import hashlib
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).parents[1]
TRAJECTORY_SOURCE = ROOT / "exp" / "latent_cot" / "trajectory.py"
RUN_SOURCE = ROOT / "exp" / "latent_cot" / "run.py"


def load_prompt_functions():
    tree = ast.parse(TRAJECTORY_SOURCE.read_text(encoding="utf-8"))
    selected = []
    wanted_assignments = {"ALIGNMENTS", "SYSTEM_PROMPT", "PROMPT_TEMPLATES"}
    wanted_functions = {
        "prompt_template_version",
        "prompt_messages",
        "prompt_template_sha256",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_assignments
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            selected.append(node)
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "Dict": Dict,
        "List": List,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(TRAJECTORY_SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace


def load_run_functions(gsm8k_loader, mbppplus_loader):
    tree = ast.parse(RUN_SOURCE.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"selected_datasets", "sampled_items"}
    ]
    namespace = {
        "load_gsm8k": gsm8k_loader,
        "load_mbppplus": mbppplus_loader,
        "random": random,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(RUN_SOURCE), "exec"),
        namespace,
    )
    return namespace


def load_cache_difference_functions():
    tree = ast.parse(RUN_SOURCE.read_text(encoding="utf-8"))
    wanted = {"identity_differences", "trajectory_cache_differences"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(RUN_SOURCE), "exec"),
        namespace,
    )
    return namespace


def load_collect_item():
    tree = ast.parse(TRAJECTORY_SOURCE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_past_length",
            "_empty_hidden",
            "collect_item",
        }:
            node.decorator_list = []
            selected.append(node)
    namespace = {
        "torch": torch,
        "Any": object,
        "Dict": Dict,
        "List": List,
        "C0Model": object,
        "AlignmentState": object,
        "prompt_messages": lambda question, dataset: [
            {"role": "user", "content": question}
        ],
        "prompt_template_version": lambda dataset: "test-prompt-v1",
        "apply_alignment": lambda hidden, state: hidden,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(TRAJECTORY_SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace["collect_item"]


class FakeAxis:
    def __init__(self):
        self.title = None
        self.lines = []

    def plot(self, *args, **kwargs):
        self.lines.append(kwargs)

    def fill_between(self, *args, **kwargs):
        pass

    def set_xlabel(self, *args, **kwargs):
        pass

    def set_ylabel(self, *args, **kwargs):
        pass

    def set_title(self, title):
        self.title = title

    def grid(self, *args, **kwargs):
        pass

    def legend(self, *args, **kwargs):
        pass


class FakeFigure:
    def __init__(self):
        self.title = None

    def suptitle(self, *args, **kwargs):
        self.title = args[0]

    def tight_layout(self):
        pass

    def savefig(self, path):
        self.saved_path = path


class FakePlot:
    def __init__(self):
        self.axes = [FakeAxis(), FakeAxis()]
        self.figure = FakeFigure()
        self.subplots_args = None

    def subplots(self, rows, columns, **kwargs):
        self.subplots_args = (rows, columns, kwargs)
        return self.figure, [self.axes]

    def close(self, figure):
        pass


def load_plot_summary(fake_plot):
    tree = ast.parse(RUN_SOURCE.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "plot_summary"
    ]
    namespace = {
        "plt": fake_plot,
        "np": SimpleNamespace(array=lambda values: values, nan=float("nan")),
        "write_json": lambda path, payload: None,
        "ALIGNMENTS": PROMPTS["ALIGNMENTS"],
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(RUN_SOURCE), "exec"),
        namespace,
    )
    return namespace["plot_summary"]


PROMPTS = load_prompt_functions()


class LatentCotPromptTests(unittest.TestCase):
    def test_gsm8k_prompt_remains_unchanged(self):
        messages = PROMPTS["prompt_messages"]("What is 2 + 2?", "gsm8k")
        self.assertEqual(
            messages[1]["content"],
            "Solve the following math problem. Reason step by step.\n\n"
            "Question: What is 2 + 2?\n\n"
            "Work out the solution carefully.",
        )

    def test_mbppplus_prompt_uses_parallel_programming_structure(self):
        messages = PROMPTS["prompt_messages"]("Write add(a, b).", "mbppplus")
        content = messages[1]["content"]
        self.assertIn("Python programming problem", content)
        self.assertIn("Reason step by step", content)
        self.assertIn("Task: Write add(a, b).", content)
        self.assertIn("self-contained solution", content)

    def test_dataset_prompts_have_distinct_versions_and_hashes(self):
        version = PROMPTS["prompt_template_version"]
        digest = PROMPTS["prompt_template_sha256"]
        self.assertEqual(version("gsm8k"), "c0_gsm8k_question_v1")
        self.assertEqual(version("mbppplus"), "c0_mbppplus_question_v1")
        self.assertNotEqual(digest("gsm8k"), digest("mbppplus"))


class LatentCotDatasetDispatchTests(unittest.TestCase):
    def test_default_all_selects_both_datasets_in_plot_order(self):
        functions = load_run_functions(lambda split: (), lambda split: ())
        selected = functions["selected_datasets"](SimpleNamespace(dataset="all"))
        self.assertEqual(selected, ("gsm8k", "mbppplus"))

    def test_single_dataset_selection_remains_available(self):
        functions = load_run_functions(lambda split: (), lambda split: ())
        selected = functions["selected_datasets"](
            SimpleNamespace(dataset="mbppplus")
        )
        self.assertEqual(selected, ("mbppplus",))

    def test_mbppplus_loader_receives_requested_split(self):
        calls = []

        def gsm8k_loader(split):
            raise AssertionError("GSM8K loader must not be called")

        def mbppplus_loader(split):
            calls.append(split)
            return ({"question": f"task-{index}"} for index in range(4))

        sampled_items = load_run_functions(
            gsm8k_loader, mbppplus_loader
        )["sampled_items"]
        args = SimpleNamespace(
            dataset="mbppplus", split="test", probe_seed=42, max_questions=2
        )
        rows = sampled_items(args)
        self.assertEqual(calls, ["test"])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row[1]["question"].startswith("task-") for row in rows))


class LatentCotPlotTests(unittest.TestCase):
    def test_combined_result_uses_two_labeled_subplots(self):
        fake_plot = FakePlot()
        plot_summary = load_plot_summary(fake_plot)
        step = {
            "step": 0,
            "mean": 1.0,
            "median": 0.9,
            "ci95_low": 0.8,
            "ci95_high": 1.2,
        }
        alignments = PROMPTS["ALIGNMENTS"]
        series = {alignment: {"steps": [step]} for alignment in alignments}
        plot_summary(
            {
                "gsm8k": {"alignments": series},
                "mbppplus": {"alignments": series},
            },
            ROOT / "unused.pdf",
            {},
        )
        self.assertEqual(fake_plot.subplots_args[0:2], (1, 2))
        self.assertTrue(fake_plot.subplots_args[2]["sharey"])
        self.assertIn("Solid lines: mean across questions", fake_plot.figure.title)
        self.assertIn("shaded bands: 95% bootstrap CI", fake_plot.figure.title)
        self.assertEqual(
            [axis.title for axis in fake_plot.axes], ["GSM8K", "MBPP+"]
        )
        for axis in fake_plot.axes:
            self.assertEqual(
                [line["label"] for line in axis.lines], list(alignments)
            )
            self.assertEqual(len({line["color"] for line in axis.lines}), 5)


class LatentCotAlignmentTests(unittest.TestCase):
    def test_alignment_order_is_fixed(self):
        self.assertEqual(
            PROMPTS["ALIGNMENTS"],
            ("identical", "linear", "soft", "kernel", "text"),
        )

    def test_recurrence_applies_alignment_before_feedback(self):
        source = TRAJECTORY_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "latent_vec = apply_alignment(last_hidden, alignment_state)", source
        )
        self.assertIn('"alignment": alignment', source)

    def test_text_recurrence_greedily_feeds_back_token_ids(self):
        source = TRAJECTORY_SOURCE.read_text(encoding="utf-8")
        self.assertIn('if alignment == "text":', source)
        self.assertIn("output_head(last_hidden).argmax(dim=-1)", source)
        self.assertIn(
            'model_inputs["input_ids"] = next_token.unsqueeze(1)', source
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_mock_model_uses_expected_feedback_for_all_recurrences(self):
        collect_item = load_collect_item()

        class FakeTokenizer:
            def __call__(self, *args, **kwargs):
                return {
                    "input_ids": torch.tensor([[11, 12]]),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                }

        class FakeOutputHead:
            def __call__(self, hidden):
                return torch.tensor([[0.0, 1.0, 5.0, 2.0]])

        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace(hidden_size=2)
                self.calls = []
                self.output_head = FakeOutputHead()

            def get_output_embeddings(self):
                return self.output_head

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                sequence_length = 1 + len(self.calls)
                past = ((
                    torch.zeros((1, 1, sequence_length, 1)),
                    torch.zeros((1, 1, sequence_length, 1)),
                ),)
                hidden = torch.tensor([[[1.0, 2.0]]])
                return SimpleNamespace(
                    past_key_values=past, hidden_states=(hidden,)
                )

        def make_wrapper():
            model = FakeModel()
            return SimpleNamespace(
                model=model,
                tokenizer=FakeTokenizer(),
                device="cpu",
                model_name="fake-model",
                render_chat=lambda messages, add_generation_prompt: "prompt",
            )

        for alignment in PROMPTS["ALIGNMENTS"]:
            fake_wrapper = make_wrapper()
            state = None if alignment == "text" else object()
            record = collect_item(
                fake_wrapper,
                7,
                {"question": "question"},
                3,
                "gsm8k",
                alignment,
                state,
            )
            self.assertTrue(record["rollout_complete"])
            self.assertEqual(record["hidden_states"].shape[0], 3)
            recurrence_calls = fake_wrapper.model.calls[1:]
            self.assertEqual(len(recurrence_calls), 3)
            if alignment == "text":
                self.assertEqual(record["generated_token_ids"], [2, 2, 2])
                self.assertTrue(
                    all("input_ids" in call for call in recurrence_calls)
                )
                self.assertTrue(
                    all("inputs_embeds" not in call for call in recurrence_calls)
                )
                self.assertTrue(
                    all(call["input_ids"].item() == 2 for call in recurrence_calls)
                )
            else:
                self.assertEqual(record["generated_token_ids"], [])
                self.assertTrue(
                    all("inputs_embeds" in call for call in recurrence_calls)
                )
                self.assertTrue(
                    all("input_ids" not in call for call in recurrence_calls)
                )


class LatentCotTrajectoryCacheTests(unittest.TestCase):
    def test_legacy_implementation_hash_is_ignored(self):
        compare = load_cache_difference_functions()["trajectory_cache_differences"]
        expected = {
            "schema_version": 1,
            "cache_identity": {
                "dataset": "gsm8k",
                "latent_steps": 50,
                "prompt_template_sha256": "prompt-hash",
            },
        }
        actual = {
            "schema_version": 1,
            "cache_identity": {
                **expected["cache_identity"],
                "trajectory_implementation_sha256": "legacy-code-hash",
            },
        }
        self.assertEqual(compare(expected, actual), [])

    def test_changed_semantic_identity_is_rejected(self):
        compare = load_cache_difference_functions()["trajectory_cache_differences"]
        expected = {
            "schema_version": 1,
            "cache_identity": {"dataset": "gsm8k", "latent_steps": 50},
        }
        actual = {
            "schema_version": 1,
            "cache_identity": {"dataset": "gsm8k", "latent_steps": 20},
        }
        differences = compare(expected, actual)
        self.assertEqual(differences[0]["field"], "cache_identity.latent_steps")


if __name__ == "__main__":
    unittest.main()
