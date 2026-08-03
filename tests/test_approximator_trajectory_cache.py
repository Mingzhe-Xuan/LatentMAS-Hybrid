import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).parents[1] / "exp" / "approximator" / "run.py"
FUNCTIONS = {
    "generation_config",
    "text_generation_config",
    "alignment_config",
    "normalized_question_records",
    "expected_manifest",
    "manifest_differences",
    "cached_question_records",
    "trajectory_cache_differences",
}


def load_cache_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    namespace = {
        "json": json,
        "ROLES": ("planner", "critic", "refiner", "judger"),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace


CACHE = load_cache_functions()
expected_manifest = CACHE["expected_manifest"]
trajectory_cache_differences = CACHE["trajectory_cache_differences"]


def args():
    return SimpleNamespace(
        agent_models=("model-a", "model-a", "model-a", "model-a"),
        prompt="sequential",
        latent_steps=50,
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=512,
        s4_text_max_new_tokens=256,
        text_mas_context_length=-1,
        kernel_features=2048,
        kernel_temperature=1.0,
        kernel_seed=101,
        kernel_chunk_size=4096,
        probe_seed=42,
    )


class TrajectoryCacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.requested = [(1, {"question": "one"}), (2, {"question": "two"})]
        self.expected = expected_manifest(args(), self.requested)
        self.actual = {
            "schema_version": 2,
            "cache_identity": {
                **self.expected["cache_identity"],
                "questions": {},
                "trajectory_implementation_sha256": "legacy-hash",
                "tokenizer_fingerprint": {"model-a": "legacy-hash"},
            },
        }
        self.trajectory = {
            "questions": [
                {"item_id": 1, "source_record": {"question": "one"}},
                {"item_id": 2, "source_record": {"question": "two"}},
                {"item_id": 3, "source_record": {"question": "extra"}},
            ]
        }

    def test_manifest_contains_only_minimal_identity(self):
        self.assertEqual(set(self.expected), {"cache_identity"})
        self.assertEqual(
            set(self.expected["cache_identity"]),
            {
                "questions",
                "role_mapping",
                "generation_config",
                "alignment_config",
                "generation_and_question_seed",
            },
        )

    def test_legacy_hashes_and_extra_cached_questions_are_ignored(self):
        self.assertEqual(
            trajectory_cache_differences(
                self.expected, self.actual, self.trajectory
            ),
            [],
        )

    def test_changed_question_content_is_rejected(self):
        self.trajectory["questions"][1]["source_record"]["question"] = "changed"
        differences = trajectory_cache_differences(
            self.expected, self.actual, self.trajectory
        )
        self.assertEqual(
            differences[0]["field"], "cache_identity.questions.2"
        )

    def test_changed_generation_config_is_rejected(self):
        self.actual["cache_identity"]["generation_config"] = {
            **self.expected["cache_identity"]["generation_config"],
            "latent_steps": 20,
        }
        differences = trajectory_cache_differences(
            self.expected, self.actual, self.trajectory
        )
        self.assertTrue(
            any(
                row["field"]
                == "cache_identity.generation_config.latent_steps"
                for row in differences
            )
        )

    def test_text_generation_config_retains_full_context_by_default(self):
        config = CACHE["text_generation_config"](args())
        self.assertEqual(config["method"], "text_mas")
        self.assertEqual(config["max_new_tokens_each"], 256)
        self.assertEqual(config["text_mas_context_length"], -1)
        self.assertEqual(config["collected_through_role"], "refiner")


if __name__ == "__main__":
    unittest.main()
