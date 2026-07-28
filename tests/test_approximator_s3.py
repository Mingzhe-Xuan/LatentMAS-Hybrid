import math
import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()