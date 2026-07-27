import ast
import unittest
from pathlib import Path
from typing import Any, Dict, List


def load_count_output_tokens():
    source = Path(__file__).parents[1].joinpath("run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "count_output_tokens"
    )
    namespace = {"Any": Any, "Dict": Dict, "List": List}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "run.py", "exec"), namespace)
    return namespace["count_output_tokens"]


count_output_tokens = load_count_output_tokens()


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


if __name__ == "__main__":
    unittest.main()
