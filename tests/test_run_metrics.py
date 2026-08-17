import ast
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_run_function(name):
    source = Path(__file__).parents[1].joinpath("run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "run.py", "exec"), namespace)
    return namespace[name]


count_output_tokens = load_run_function("count_output_tokens")
summarize_role_metrics = load_run_function("summarize_role_metrics")

def load_utils_function(name):
    source = Path(__file__).parents[1].joinpath("utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "utils.py", "exec"), namespace)
    return namespace[name]


def load_models_function(name):
    source = Path(__file__).parents[1].joinpath("models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"KERNEL_ENTROPY_CHECK_INTERVAL": 10}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "models.py", "exec"), namespace)
    return namespace[name]


build_agent_metrics = load_utils_function("build_agent_metrics")
latent_vocab_decode_steps = load_models_function("latent_vocab_decode_steps")


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, *, add_special_tokens):
        self.calls.append((texts, add_special_tokens))
        return {"input_ids": [text.split() for text in texts]}


class OutputTokenCountTests(unittest.TestCase):
    def test_counts_visible_agent_outputs_only(self):
        tokenizer = FakeTokenizer()
        preds = [
            {
                "raw_prediction": "must not be counted twice",
                "agents": [
                    {"output": "", "latent_steps": 45},
                    {"output": "one two three"},
                ],
            },
            {
                "agents": [
                    {"output": "four five"},
                    {"latent_steps": 45},
                ],
            },
        ]

        self.assertEqual(count_output_tokens(preds, tokenizer), 5)
        self.assertEqual(
            tokenizer.calls,
            [(["one two three", "four five"], False)],
        )

    def test_empty_results_have_zero_output_tokens(self):
        tokenizer = FakeTokenizer()

        self.assertEqual(count_output_tokens([], tokenizer), 0)
        self.assertEqual(tokenizer.calls, [])


class AgentMetricBuilderTests(unittest.TestCase):
    def test_amortizes_batch_timings_and_preserves_token_types(self):
        metrics = build_agent_metrics(
            text_input_tokens=12,
            latent_input_tokens=45,
            latent_output_tokens=45,
            phase_metrics={
                "prefill_seconds": 2.0,
                "latent_decode_seconds": 4.0,
                "alignment_seconds": 1.0,
                "timing_source": "test",
            },
            batch_size=2,
        )
        self.assertEqual(
            metrics["tokens"],
            {"text_input": 12, "latent_input": 45, "text_output": 0, "latent_output": 45},
        )
        self.assertEqual(metrics["timing"]["prefill_seconds"], 1.0)
        self.assertEqual(metrics["timing"]["latent_decode_seconds"], 2.0)
        self.assertEqual(metrics["timing"]["alignment_seconds"], 0.5)
        self.assertEqual(metrics["timing"]["source"], "test")


class LatentVocabDecodeCountTests(unittest.TestCase):
    def test_counts_only_full_vocabulary_alignment_steps(self):
        self.assertEqual(latent_vocab_decode_steps("soft", 23), 23)
        self.assertEqual(latent_vocab_decode_steps("kernel_early_stopping", 23), 2)
        self.assertEqual(latent_vocab_decode_steps("kernel_early_stopping", 9), 0)
        self.assertEqual(latent_vocab_decode_steps("kernel", 23), 0)

class RoleMetricTests(unittest.TestCase):
    def test_aggregates_tokens_and_timings_by_role(self):
        preds = [
            {
                "agents": [
                    {
                        "role": "planner",
                        "metrics": {
                            "tokens": {"text_input": 10, "latent_output": 45},
                            "timing": {
                                "prefill_seconds": 0.2,
                                "latent_decode_seconds": 1.0,
                                "alignment_seconds": 0.1,
                                "source": "test",
                            },
                        },
                    },
                    {
                        "role": "judger",
                        "metrics": {
                            "tokens": {
                                "text_input": 20,
                                "latent_input": 45,
                                "text_output": 3,
                            },
                            "timing": {
                                "prefill_seconds": 0.4,
                                "text_decode_seconds": 2.0,
                                "source": "test",
                            },
                        },
                    },
                ]
            },
            {
                "agents": [
                    {
                        "role": "planner",
                        "metrics": {
                            "tokens": {"text_input": 14, "latent_output": 45},
                            "timing": {
                                "prefill_seconds": 0.4,
                                "latent_decode_seconds": 1.2,
                                "alignment_seconds": 0.2,
                                "source": "test",
                            },
                        },
                    },
                    {
                        "role": "judger",
                        "metrics": {
                            "tokens": {
                                "text_input": 24,
                                "latent_input": 45,
                                "text_output": 5,
                            },
                            "timing": {
                                "prefill_seconds": 0.6,
                                "text_decode_seconds": 3.0,
                                "source": "test",
                            },
                        },
                    },
                ]
            },
        ]

        roles, tokens, timing = summarize_role_metrics(preds)

        self.assertEqual(roles["planner"]["output_type"], "latent")
        self.assertEqual(roles["planner"]["tokens"]["text_input"]["average_per_problem"], 12.0)
        self.assertEqual(roles["planner"]["tokens"]["latent_output"]["average_per_problem"], 45.0)
        self.assertEqual(roles["planner"]["timing"]["alignment_seconds"]["average_per_problem"], 0.15)
        self.assertEqual(roles["judger"]["output_type"], "text")
        self.assertEqual(roles["judger"]["tokens"]["text_output"]["average_per_problem"], 4.0)
        self.assertEqual(tokens["latent_input"]["average_per_problem"], 45.0)
        self.assertEqual(timing["text_decode_seconds"]["average_per_problem"], 2.5)

    def test_empty_metrics_are_well_formed(self):
        roles, tokens, timing = summarize_role_metrics([])
        self.assertEqual(roles, {})
        self.assertEqual(tokens["text_input"], {"total": 0, "average_per_problem": 0.0})
        self.assertEqual(timing["prefill_seconds"], {"total": 0.0, "average_per_problem": 0.0})
if __name__ == "__main__":
    unittest.main()
