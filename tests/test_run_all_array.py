"""Static checks for the filtered 288-task PBS experiment array."""

from itertools import product
from pathlib import Path
import re
import shlex
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN_ALL = (ROOT / "run_all.sh").read_text(encoding="utf-8")
RUN_ALL_FAST = (ROOT / "run_all_fast.sh").read_text(encoding="utf-8")
RUN = (ROOT / "run.sh").read_text(encoding="utf-8")


def bash_array(name: str) -> list[str]:
    match = re.search(rf"^{name}=\(\s*(.*?)\s*\)$", RUN_ALL, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"missing Bash array: {name}")
    return shlex.split(match.group(1), comments=True)


class RunAllArrayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.datasets = bash_array("DATASETS")
        self.models = bash_array("MODELS")
        self.four_b_datasets = bash_array("FOUR_B_DATASETS")
        methods = bash_array("METHODS")
        prompts = bash_array("PROMPTS")
        alignments = bash_array("ALIGNMENTS")
        self.configs = list(zip(methods, prompts, alignments, strict=True))
        self.model_dataset_pairs = [
            *((model, dataset) for model in self.models[:2] for dataset in self.datasets),
            *((self.models[2], dataset) for dataset in self.four_b_datasets),
        ]

    def test_pbs_array_directive_and_dimensions(self) -> None:
        self.assertRegex(RUN_ALL, r"(?m)^#PBS -J 1-288%3$")
        self.assertEqual(len(self.datasets), 9)
        self.assertEqual(len(self.models), 3)
        self.assertEqual(len(self.four_b_datasets), 6)
        self.assertEqual(len(self.configs), 12)
        self.assertEqual(len(self.model_dataset_pairs) * len(self.configs), 288)

    def test_model_order_and_four_b_exclusions(self) -> None:
        self.assertEqual(
            self.models,
            ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-4B"],
        )
        excluded = {"aime2024", "aime2025", "gpqa"}
        self.assertTrue(excluded.isdisjoint(self.four_b_datasets))
        self.assertEqual(set(self.datasets) - set(self.four_b_datasets), excluded)
        self.assertEqual(self.model_dataset_pairs[0], ("Qwen/Qwen3-8B", "aime2024"))
        self.assertEqual(self.model_dataset_pairs[9], ("Qwen/Qwen3-14B", "aime2024"))
        self.assertEqual(self.model_dataset_pairs[18], ("Qwen/Qwen3-4B", "arc_challenge"))

    def test_fast_array_excludes_slow_datasets_for_every_model(self) -> None:
        self.assertRegex(RUN_ALL_FAST, r"(?m)^#PBS -J 1-216%3$")
        self.assertIn("FAST_ONLY=true", RUN_ALL_FAST)
        self.assertIn('RUN_ALL_SCRIPT="${SUBMIT_DIR}/run_all.sh"', RUN_ALL_FAST)
        self.assertIn('exec bash "${RUN_ALL_SCRIPT}"', RUN_ALL_FAST)
        self.assertIn('if [[ "${FAST_ONLY}" == "true" ]]', RUN_ALL)
        self.assertEqual(
            len(self.four_b_datasets) * len(self.models) * len(self.configs),
            216,
        )
        excluded = {"aime2024", "aime2025", "gpqa"}
        self.assertTrue(excluded.isdisjoint(self.four_b_datasets))

    def test_exact_configuration_matrix(self) -> None:
        expected = [
            ("baseline", "sequential", "identical"),
            ("baseline", "hierarchical", "identical"),
            ("text_mas", "sequential", "identical"),
            ("text_mas", "hierarchical", "identical"),
            ("latent_mas", "sequential", "identical"),
            ("latent_mas", "sequential", "linear"),
            ("latent_mas", "sequential", "kernel"),
            ("latent_mas", "sequential", "soft"),
            ("latent_mas", "hierarchical", "identical"),
            ("latent_mas", "hierarchical", "linear"),
            ("latent_mas", "hierarchical", "kernel"),
            ("latent_mas", "hierarchical", "soft"),
        ]
        self.assertEqual(self.configs, expected)

    def test_all_state_file_names_are_unique(self) -> None:
        names = []
        for (model, dataset), (method, prompt, alignment) in product(
            self.model_dataset_pairs, self.configs
        ):
            effective_method = f"{method}_{alignment}" if method == "latent_mas" else method
            model_slug = re.sub(r"[^A-Za-z0-9._-]", "_", model)
            names.append(f"{dataset}_{effective_method}_{prompt}_{model_slug}_state.txt")
        self.assertEqual(len(names), 288)
        self.assertEqual(len(set(names)), 288)
        self.assertIn(
            "arc_easy_latent_mas_kernel_sequential_Qwen_Qwen3-4B_state.txt",
            names,
        )
        for dataset in ("aime2024", "aime2025", "gpqa"):
            self.assertFalse(any(name.startswith(f"{dataset}_") and "Qwen_Qwen3-4B" in name for name in names))

    def test_alignment_is_in_latent_state_identity(self) -> None:
        self.assertIn('STATE_METHOD="${CONFIG_METHOD}_${CONFIG_ALIGNMENT}"', RUN_ALL)
        latent_names = {
            f"latent_mas_{alignment}_{prompt}"
            for method, prompt, alignment in self.configs
            if method == "latent_mas"
        }
        self.assertEqual(len(latent_names), 8)

    def test_skip_force_and_progress_ledger(self) -> None:
        self.assertIn('FORCE_ALL="${FORCE_ALL:-false}"', RUN_ALL)
        self.assertIn('[[ "${FORCE_ALL}" != "true" && -e "${STATE_PATH}" ]]', RUN_ALL)
        self.assertNotIn('state_${TASK}', RUN_ALL)
        self.assertIn('PROGRESS_FILE="${SUBMIT_DIR}/state.txt"', RUN_ALL)
        self.assertIn("flock -x 9", RUN_ALL)
        for status in ("STARTED", "SKIPPED", "COMPLETED", "FAILED"):
            self.assertRegex(RUN_ALL, rf"append_progress {status}\b")

    def test_single_config_entry_and_validation(self) -> None:
        self.assertIn("SINGLE_CONFIG=true", RUN_ALL)
        self.assertIn('bash "${RUN_SCRIPT}"', RUN_ALL)
        self.assertIn('run_repeated "${CONFIG_METHOD}" "${CONFIG_PROMPT}" "${CONFIG_ALIGNMENT}"', RUN)
        self.assertIn("baseline|text_mas", RUN)
        self.assertIn("only supports identical alignment", RUN)
        self.assertIn("identical|linear|kernel|soft", RUN)
        self.assertIn('--soft_temperature "${SOFT_TEMPERATURE}"', RUN)
        self.assertIn('--soft_chunk_size "${SOFT_CHUNK_SIZE}"', RUN)
        self.assertIn("SOFT_TEMPERATURE=${SOFT_TEMPERATURE}", RUN_ALL)
        self.assertIn("SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE}", RUN_ALL)
        self.assertIn("SOFT_TEMPERATURE=${SOFT_TEMPERATURE}", RUN_ALL_FAST)
        self.assertIn("SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE}", RUN_ALL_FAST)
        self.assertIn('STATE_FILE="${STATE_FILE:-state/run_state.txt}"', RUN)
        self.assertNotIn('STATE_FILE="${STATE_FILE:-state.txt}"', RUN)


if __name__ == "__main__":
    unittest.main()
