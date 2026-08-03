import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

APPROXIMATOR_ROOT = Path(__file__).resolve().parents[1] / "exp" / "approximator"
sys.path.insert(0, str(APPROXIMATOR_ROOT))

from stages import s4


class S4PlotTests(unittest.TestCase):
    def _rows(self):
        rows = []
        embeddings = {
            "hidden": ([10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]),
            "exact": ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            "linear": ([1.0, 0.05, 0.0], [0.0, 1.0, 0.05], [0.05, 0.0, 1.0]),
            "kernel": ([0.95, 0.0, 0.05], [0.05, 0.95, 0.0], [0.0, 0.05, 0.95]),
        }
        for item_id in range(3):
            for method, values in embeddings.items():
                rows.append(
                    {
                        "item_id": item_id,
                        "position": 0,
                        "turn_id": 0,
                        "agent_id": 0,
                        "role": "refiner",
                        "state_kind": "latent_reply_hidden",
                        "method": method,
                        "entropy": 0.5,
                        "embedding": values[item_id],
                    }
                )
            rows.append(
                {
                    "item_id": item_id,
                    "position": 0,
                    "turn_id": 0,
                    "agent_id": 2,
                    "role": "refiner",
                    "state_kind": "text_mas_token_embedding",
                    "method": "text",
                    "entropy": None,
                    "embedding": [0.8, 0.1 * item_id, 0.2],
                }
            )
        return rows

    def test_pca_uses_separate_joint_fit_for_alignment_and_reference(self):
        args = SimpleNamespace(
            probe_seed=42,
            bootstrap_replicates=0,
            s4_tsne=False,
            s4_text_mas=True,
        )
        saved_figures = {}
        written = {}

        def capture_figure(figure, stem):
            saved_figures[stem] = {
                "axis_count": len(figure.axes),
                "collection_count": len(figure.axes[0].collections),
            }

        def capture_rows(rows, stem):
            written[stem] = list(rows)

        with patch.object(s4, "save_figure", side_effect=capture_figure), patch.object(
            s4, "write_rows", side_effect=capture_rows
        ):
            summary = s4.plot_s4(self._rows(), args)

        self.assertEqual(summary["selected_state_count"], 3)
        self.assertEqual(summary["joint_point_count_per_reduction"], 9)
        self.assertEqual(
            summary["pca"]["fit"],
            "independent joint fit for every alignment x reference",
        )
        self.assertEqual(
            set(summary["pca"]["by_alignment_and_reference"]),
            {"linear", "kernel"},
        )
        for alignment in ("linear", "kernel"):
            self.assertEqual(
                set(summary["pca"]["by_alignment_and_reference"][alignment]),
                {"exact", "text"},
            )
        self.assertEqual(
            set(saved_figures),
            {
                "s4_linear_exact_joint_reduction",
                "s4_linear_text_joint_reduction",
                "s4_kernel_exact_joint_reduction",
                "s4_kernel_text_joint_reduction",
            },
        )
        for details in saved_figures.values():
            self.assertEqual(details["axis_count"], 1)
            self.assertEqual(details["collection_count"], 3)

        coordinates = written["s4_joint_pca_coordinates"]
        self.assertEqual(len(coordinates), 36)
        for alignment in ("linear", "kernel"):
            for reference in ("exact", "text"):
                subset = [
                    row
                    for row in coordinates
                    if row["alignment"] == alignment
                    and row["reference"] == reference
                ]
                self.assertEqual(len(subset), 9)
                self.assertEqual(
                    {row["space"] for row in subset},
                    {"hidden", "reference", "aligned"},
                )
                self.assertTrue(all(row["reducer"] == "pca" for row in subset))

    def test_incomplete_state_is_excluded_from_joint_fit(self):
        rows = self._rows()
        rows = [
            row
            for row in rows
            if not (row["item_id"] == 2 and row["method"] == "kernel")
        ]
        args = SimpleNamespace(
            probe_seed=42,
            bootstrap_replicates=0,
            s4_tsne=False,
            s4_text_mas=True,
        )
        with patch.object(s4, "save_figure"), patch.object(s4, "write_rows"):
            summary = s4.plot_s4(rows, args)

        self.assertEqual(summary["input_state_count"], 3)
        self.assertEqual(summary["valid_state_count"], 2)
        self.assertEqual(summary["invalid_state_count"], 1)
        self.assertEqual(summary["selected_state_count"], 2)

    def test_text_comparison_can_be_disabled(self):
        args = SimpleNamespace(
            probe_seed=42,
            bootstrap_replicates=0,
            s4_tsne=False,
            s4_text_mas=False,
        )
        saved_figures = []
        with patch.object(
            s4, "save_figure", side_effect=lambda figure, stem: saved_figures.append(stem)
        ), patch.object(s4, "write_rows"):
            summary = s4.plot_s4(self._rows(), args)

        self.assertFalse(summary["text_mas_enabled"])
        self.assertEqual(summary["selected_text_token_count"], 0)
        self.assertEqual(
            set(saved_figures),
            {
                "s4_linear_exact_joint_reduction",
                "s4_kernel_exact_joint_reduction",
            },
        )


if __name__ == "__main__":
    unittest.main()
