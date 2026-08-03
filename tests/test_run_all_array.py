"""Static checks for the 270-task PBS experiment array."""

from itertools import product
from pathlib import Path
import re
import shlex
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN_ALL = (ROOT / "run_all.sh").read_text(encoding="utf-8")
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
        methods = bash_array("METHODS")
        prompts = bash_array("PROMPTS")
        alignments = bash_array("ALIGNMENTS")
        self.configs = list(zip(methods, prompts, alignments, strict=True))

    def test_pbs_array_directive_and_dimensions(self) -> None:
        self.assertRegex(RUN_ALL, r"(?m)^#PBS -J 1-270%3$")
        self.assertEqual(len(self.datasets), 9)
        self.assertEqual(len(self.models), 3)
        self.assertEqual(len(self.configs), 10)
        self.assertEqual(len(self.datasets) * len(self.models) * len(self.configs), 270)

    def test_exact_configuration_matrix(self) -> None:
        expected = [
            ("baseline", "sequential", "identical"),
            ("baseline", "hierarchical", "identical"),
            ("text_mas", "sequential", "identical"),
            ("text_mas", "hierarchical", "identical"),
            ("latent_mas", "sequential", "identical"),
            ("latent_mas", "sequential", "linear"),
            ("latent_mas", "sequential", "kernel"),
            ("latent_mas", "hierarchical", "identical"),
            ("latent_mas", "hierarchical", "linear"),
            ("latent_mas", "hierarchical", "kernel"),
        ]
        self.assertEqual(self.configs, expected)

    def test_all_state_file_names_are_unique(self) -> None:
        names = []
        for dataset, model, (method, prompt, alignment) in product(
            self.datasets, self.models, self.configs
        ):
            effective_method = f"{method}_{alignment}" if method == "latent_mas" else method
            model_slug = re.sub(r"[^A-Za-z0-9._-]", "_", model)
            names.append(f"{dataset}_{effective_method}_{prompt}_{model_slug}_state.txt")
        self.assertEqual(len(names), 270)
        self.assertEqual(len(set(names)), 270)
        self.assertIn(
            "arc_easy_latent_mas_kernel_sequential_Qwen_Qwen3-4B_state.txt",
            names,
        )

    def test_alignment_is_in_latent_state_identity(self) -> None:
        self.assertIn('STATE_METHOD="${CONFIG_METHOD}_${CONFIG_ALIGNMENT}"', RUN_ALL)
        latent_names = {
            f"latent_mas_{alignment}_{prompt}"
            for method, prompt, alignment in self.configs
            if method == "latent_mas"
        }
        self.assertEqual(len(latent_names), 6)

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
        self.assertIn("identical|linear|kernel", RUN)
        self.assertIn('STATE_FILE="${STATE_FILE:-state/run_state.txt}"', RUN)
        self.assertNotIn('STATE_FILE="${STATE_FILE:-state.txt}"', RUN)


if __name__ == "__main__":
    unittest.main()
