import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

APPROXIMATOR_ROOT = Path(__file__).resolve().parents[1] / "exp" / "approximator"
sys.path.insert(0, str(APPROXIMATOR_ROOT))

from stages import s4


class S4PlotTests(unittest.TestCase):
    def test_pca_explained_variance_uses_numpy_singular_values(self):
        rows = []
        embeddings = {
            "exact": ([1.0, 0.0], [0.0, 1.0]),
            "linear": ([2.0, 0.0], [0.0, 2.0]),
            "kernel": ([1.0, 1.0], [-1.0, 1.0]),
        }
        for item_id in range(2):
            for method, values in embeddings.items():
                rows.append(
                    {
                        "item_id": item_id,
                        "position": 0,
                        "turn_id": 0,
                        "agent_id": 0,
                        "method": method,
                        "entropy": 0.5,
                        "embedding": values[item_id],
                    }
                )

        args = SimpleNamespace(
            probe_seed=42,
            bootstrap_replicates=0,
            s4_tsne=False,
        )
        with patch.object(s4, "save_figure"), patch.object(s4, "write_rows"):
            summary = s4.plot_s4(rows, args)

        pca = summary["pca"]
        self.assertEqual(summary["mapped_embedding_count"], 6)
        self.assertEqual(pca["fit"], "per_method")
        for method in ("exact", "linear", "kernel"):
            method_pca = pca["by_method"][method]
            self.assertGreaterEqual(
                method_pca["pc1_explained_variance_ratio"], 0.0
            )
            self.assertLessEqual(
                method_pca["pc1_pc2_cumulative_ratio"], 1.0
            )
            self.assertAlmostEqual(
                method_pca["pc1_pc2_cumulative_ratio"], 1.0
            )


if __name__ == "__main__":
    unittest.main()