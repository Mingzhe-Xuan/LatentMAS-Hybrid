import argparse
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "exp" / "latent_comm" / "run.py"
MODEL_SOURCE = ROOT / "models.py"


def load_nodes(*names, namespace=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in names for target in targets):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
    scope = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope


class M0ContractTests(unittest.TestCase):
    def test_exact_model_pair_and_alignment_grid(self):
        scope = load_nodes("MODELS", "MODEL_PAIRS", "ALIGNMENTS")
        self.assertEqual(
            scope["MODEL_PAIRS"],
            (
                ("Qwen/Qwen3-4B", "Qwen/Qwen3-4B"),
                ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"),
                ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B"),
                ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
            ),
        )
        self.assertEqual(scope["ALIGNMENTS"], ("linear", "kernel", "soft", "text"))

    def test_defaults_match_m0_contract(self):
        scope = load_nodes(
            "ALIGNMENTS",
            "parse_args",
            namespace={
                "argparse": argparse,
                "torch": SimpleNamespace(
                    cuda=SimpleNamespace(is_available=lambda: False)
                ),
            },
        )
        args = scope["parse_args"]([])
        self.assertEqual(args.dataset, "arc_easy")
        self.assertEqual(args.max_questions, 100)
        self.assertEqual(args.prompt, "sequential")
        self.assertEqual(args.alignments, ["linear", "kernel", "soft", "text"])

    def test_sampling_is_seeded_and_has_100_questions(self):
        def loader(split):
            self.assertEqual(split, "test")
            for index in range(200):
                yield {"question": f"question-{index}", "gold": "a"}

        scope = load_nodes(
            "sampled_items",
            namespace={"load_arc_easy": loader, "random": __import__("random")},
        )
        args = SimpleNamespace(split="test", sample_seed=42, max_questions=100)
        first = scope["sampled_items"](args)
        second = scope["sampled_items"](args)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(len({item_id for item_id, _ in first}), 100)

    def test_receiver_prompt_is_question_blind(self):
        scope = load_nodes("_receiver_messages")
        serialized = repr(scope["_receiver_messages"]("Qwen/Qwen3-8B"))
        private_question = "Which planet is closest to the Sun? a: Venus b: Mercury"
        self.assertNotIn(private_question, serialized)
        self.assertIn("not shown to you", serialized)
        self.assertIn("aligned hidden-state sequence", serialized)

    def test_source_enforces_question_blind_audit_and_transfer_protocols(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("M0 receiver prompt leaked the original question", source)
        self.assertIn('"receiver_question_visible": False', source)
        self.assertIn('if alignment == "text":', source)
        self.assertIn("source.align_hidden_to(hidden, target)", source)
        self.assertIn('return [{"role": "user", "content": question}]', source)
        self.assertIn('"prefill_hidden_states": prefill_hidden_states', source)
        self.assertNotIn("for _ in range(args.latent_steps)", source)

    def test_embedding_generation_supports_greedy_receiver_decode(self):
        source = MODEL_SOURCE.read_text(encoding="utf-8")
        function = ast.parse(source)
        target = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.FunctionDef)
            and node.name == "generate_text_from_embeds_batch"
        )
        rendered = ast.unparse(target)
        self.assertIn("do_sample = temperature > 0", rendered)
        self.assertIn("**generation_kwargs", rendered)


if __name__ == "__main__":
    unittest.main()
