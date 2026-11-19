"""Adversarial checks for the executable referee over committed synthetic data."""
import copy
import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from independent_referee import audit, exact_sign_test, holm_adjust


class IndependentRefereeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "results/runs.csv").open(newline="", encoding="utf-8") as f:
            cls.rows = list(csv.DictReader(f))
        cls.summary = json.loads(
            (ROOT / "results/summary.json").read_text(encoding="utf-8")
        )
        cls.paired = json.loads(
            (ROOT / "results/paired_analysis.json").read_text(encoding="utf-8")
        )

    def test_archive_recomputes_and_heldout_discordance(self):
        report = audit(self.rows, self.summary, self.paired)
        self.assertEqual(report["record_count_verified"], 1200)
        self.assertEqual(report["configurations_checked"], 12)
        self.assertTrue(report["all_training_mean_differences_positive"])
        self.assertEqual(report["configurations_with_positive_heldout_difference"], 5)
        self.assertEqual(report["holm_family_size"], 12)
        self.assertEqual(report["holm_adjusted_train_sign_p_below_0_05"], 12)

    def test_holm_stepdown_known_values(self):
        self.assertEqual(holm_adjust([.01, .04, .03]), [.03, .06, .06])
        self.assertEqual(holm_adjust([.5, .1]), [.5, .2])
        self.assertEqual(holm_adjust([]), [])
        with self.assertRaises(ValueError):
            holm_adjust([float("nan")])

    def test_sign_test_known_values(self):
        self.assertEqual(exact_sign_test(0, 0), 1.0)
        self.assertAlmostEqual(exact_sign_test(4, 0), 0.125)
        self.assertAlmostEqual(exact_sign_test(2, 2), 1.0)

    def test_rejects_tampered_training_score(self):
        records = [dict(r) for r in self.rows]
        records[0]["train_score"] = "0.0"
        with self.assertRaisesRegex(ValueError, "train_score"):
            audit(records, self.summary, self.paired)

    def test_rejects_tampered_heldout_score(self):
        records = [dict(r) for r in self.rows]
        records[0]["heldout_score"] = "0.0"
        with self.assertRaisesRegex(ValueError, "heldout_score"):
            audit(records, self.summary, self.paired)

    def test_rejects_duplicate_record(self):
        records = [dict(r) for r in self.rows]
        records[-1] = dict(records[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit(records, self.summary, self.paired)

    def test_rejects_unmatched_budget(self):
        records = [dict(r) for r in self.rows]
        records[0]["calls"] = "0"
        with self.assertRaisesRegex(ValueError, "objective-call"):
            audit(records, self.summary, self.paired)

    def test_rejects_altered_pvalue(self):
        paired = copy.deepcopy(self.paired)
        paired["configurations"][0]["two_sided_sign_p_unadjusted"] = 0.5
        with self.assertRaisesRegex(ValueError, "paired sign p"):
            audit(self.rows, self.summary, paired)


if __name__ == "__main__":
    unittest.main()
