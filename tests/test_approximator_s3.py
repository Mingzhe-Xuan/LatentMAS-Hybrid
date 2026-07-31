import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

APPROXIMATOR_ROOT = Path(__file__).resolve().parents[1] / "exp" / "approximator"
sys.path.insert(0, str(APPROXIMATOR_ROOT))

from stages import s3


class S3NumericalStabilityTests(unittest.TestCase):
    def test_log_kernel_estimate_survives_float32_feature_underflow(self):
        key = torch.tensor([20.0], dtype=torch.float32)
        query = torch.tensor([20.0], dtype=torch.float32)
        omega = torch.zeros((4, 1), dtype=torch.float32)

        direct = (
            s3.positive_features(key[None], omega)[0]
            @ s3.positive_features(query[None], omega)[0]
        )
        log_estimate = s3._single_kernel_log_estimate(key, query, omega)

        self.assertEqual(float(direct), 0.0)
        self.assertAlmostEqual(float(log_estimate), -400.0, places=10)
        self.assertGreater(float(log_estimate.exp()), 0.0)

    def test_log_sample_variance_remains_finite_below_float_range(self):
        log_values = [torch.tensor(value) for value in (-400.0, -399.0, -401.0)]

        log_variance = s3._log_sample_variance(log_values)

        self.assertTrue(torch.isfinite(log_variance))
        self.assertLess(float(log_variance / math.log(10)), -300.0)

    def test_tail_rows_report_question_and_seed_exceedance_rates(self):
        error_rows = [
            {"item_id": 1, "errors": torch.tensor([0.1, 0.3])},
            {"item_id": 1, "errors": torch.tensor([0.2, 0.4])},
            {"item_id": 2, "errors": torch.tensor([float("nan"), 0.1])},
        ]

        question_rows, seed_rows = s3._tail_rows_for_configuration(
            error_rows,
            feature_count=512,
            temperature=1.0,
            seeds=[1001, 1002],
            epsilons=[0.25],
        )

        first = next(row for row in question_rows if row["item_id"] == 1)
        self.assertEqual(first["exceedance_count"], 2)
        self.assertEqual(first["valid_observation_count"], 4)
        self.assertAlmostEqual(first["empirical_tail_probability"], 0.5)
        self.assertAlmostEqual(first["mean_squared_l2_error"], 0.075)
        self.assertAlmostEqual(first["markov_mse_upper_bound"], 1.0)

        first_seed = next(row for row in seed_rows if row["seed"] == 1001)
        second_seed = next(row for row in seed_rows if row["seed"] == 1002)
        self.assertEqual(first_seed["question_count"], 1)
        self.assertEqual(first_seed["invalid_state_count"], 1)
        self.assertAlmostEqual(first_seed["exceedance_rate"], 0.0)
        self.assertAlmostEqual(second_seed["exceedance_rate"], 0.5)
        self.assertAlmostEqual(second_seed["pooled_exceedance_rate"], 2 / 3)

    def test_tail_probability_uses_strict_greater_than_epsilon(self):
        question_rows, _ = s3._tail_rows_for_configuration(
            [{"item_id": 7, "errors": torch.tensor([0.2, 0.3])}],
            feature_count=2048,
            temperature=1.0,
            seeds=[1001, 1002],
            epsilons=[0.2],
        )

        self.assertEqual(question_rows[0]["exceedance_count"], 1)
        self.assertAlmostEqual(
            question_rows[0]["empirical_tail_probability"], 0.5
        )
    def test_tail_epsilon_plot_contains_empirical_bound_and_ci(self):
        rows = []
        for item_id, offset in ((1, 0.0), (2, 0.1)):
            for feature_count in (512, 1024, 2048):
                for epsilon in (0.1, 0.2):
                    empirical = max(0.0, 0.8 - epsilon - offset)
                    rows.append(
                        {
                            "item_id": item_id,
                            "m": feature_count,
                            "tau": 1.0,
                            "epsilon": epsilon,
                            "empirical_tail_probability": empirical,
                            "markov_mse_upper_bound": min(1.0, empirical + 0.15),
                        }
                    )
        captured = {}

        def capture(figure, stem):
            captured["stem"] = stem
            captured["line_count"] = len(figure.axes[0].lines)
            captured["collection_count"] = len(figure.axes[0].collections)

        args = SimpleNamespace(bootstrap_replicates=20, probe_seed=42)
        with patch.object(s3, "save_figure", side_effect=capture):
            s3.plot_tail_epsilon(rows, args)

        self.assertEqual(captured["stem"], "s3_tail_probability_epsilon")
        self.assertEqual(captured["line_count"], 6)
        self.assertGreaterEqual(captured["collection_count"], 3)


if __name__ == "__main__":
    unittest.main()
