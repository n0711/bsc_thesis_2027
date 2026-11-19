"""Unit tests for the finite synthetic target-shift theorem and auditor."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment import (EMBED, EPS, HOLDOUT_TARGETS, TRAIN_TARGETS,
                        enumerate_space, score)
from target_transfer_audit import (audit_length, build_report, latex_table,
                                   target_pairing_distance,
                                   uniform_embedding_radius, write_outputs)


class TargetTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_input_validation_and_transport_symmetry(self):
        with self.assertRaises(ValueError):
            target_pairing_distance((), ())
        with self.assertRaises(ValueError):
            target_pairing_distance(((0., 0.),), ())
        with self.assertRaises(ValueError):
            target_pairing_distance(((0., 0.),), ((1.,),))
        self.assertAlmostEqual(
            target_pairing_distance(TRAIN_TARGETS, HOLDOUT_TARGETS),
            target_pairing_distance(HOLDOUT_TARGETS, TRAIN_TARGETS),
        )

    def test_order_invariance_of_pairing(self):
        reversed_holdout = tuple(reversed(HOLDOUT_TARGETS))
        self.assertAlmostEqual(
            target_pairing_distance(TRAIN_TARGETS, reversed_holdout),
            target_pairing_distance(),
        )

    def test_uniform_radius_growth(self):
        max_norm = max(math.hypot(*values) for values in EMBED.values())
        for length in (2, 3, 4):
            self.assertAlmostEqual(
                uniform_embedding_radius(length),
                max_norm * (1 + 0.05 * (length - 1)),
            )
        with self.assertRaises(ValueError):
            uniform_embedding_radius(0)

    def test_all_finite_candidates_have_transfer_regret_bound(self):
        for item in self.report["lengths"]:
            self.assertEqual(item["candidate_count"], 6 ** item["length"])
            self.assertEqual(item["certified_order_errors"], 0)
            self.assertLessEqual(
                item["maximum_absolute_score_gap"],
                item["analytic_uniform_gap_upper_bound"] + 1e-12,
            )
            self.assertLessEqual(
                item["maximum_absolute_regret_difference"],
                item["exact_gap_oscillation"] + 1e-12,
            )
            self.assertLessEqual(
                item["heldout_regret_of_training_oracle"],
                item["exact_gap_oscillation"] + 1e-12,
            )
            self.assertLessEqual(
                item["certified_order_preserving_pairs"] +
                item["observed_reversed_order_pairs"],
                item["comparable_unordered_pairs"],
            )

    def test_oracle_selection_recomputed(self):
        for item in self.report["lengths"]:
            length = item["length"]
            prompts = list(enumerate_space(length))
            train_max = max(score(p, TRAIN_TARGETS) for p in prompts)
            heldout_max = max(score(p, HOLDOUT_TARGETS) for p in prompts)
            self.assertAlmostEqual(item["training_oracle_score"], train_max)
            self.assertAlmostEqual(item["heldout_oracle_score"], heldout_max)
            train_best = tuple(item["training_oracle_prompt"])
            self.assertAlmostEqual(score(train_best, TRAIN_TARGETS), train_max)
            self.assertAlmostEqual(
                heldout_max - score(train_best, HOLDOUT_TARGETS),
                item["heldout_regret_of_training_oracle"],
            )

    def test_output_is_deterministic_and_marked_synthetic(self):
        self.assertEqual(self.report["evidence_type"],
                         "synthetic_surrogate_not_LLM")
        with tempfile.TemporaryDirectory() as d:
            write_outputs(self.report, d)
            report_file = Path(d) / "results/target_transfer_audit.json"
            tex_file = Path(d) / "tex/target_transfer_audit.tex"
            first = (report_file.read_bytes(), tex_file.read_bytes())
            write_outputs(self.report, d)
            self.assertEqual(
                first, (report_file.read_bytes(), tex_file.read_bytes())
            )
            self.assertIn("\\label{tab:target-transfer}",
                          latex_table(self.report))


if __name__ == "__main__":
    unittest.main()
