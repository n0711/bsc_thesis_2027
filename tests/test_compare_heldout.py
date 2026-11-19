"""Synthetic fixtures only: no live model calls and no actual predictions."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compare_heldout import compare, exact_discordance_p


def record(ex_id, prompt, expected, prediction, valid=True):
    return {
        "id": ex_id, "task": "classification", "model": "fixture-model",
        "prompt_id": prompt, "expected": expected, "prediction": prediction,
        "output_valid": valid, "input_tokens": 10 if prompt == "base" else 12,
        "output_tokens": 1, "latency_ms": 100.0 if prompt == "base" else 110.0,
    }


class HeldoutPairingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            record("a", "base", "yes", "no"),
            record("a", "candidate", "yes", "yes"),
            record("b", "base", "no", "no"),
            record("b", "candidate", "no", "yes"),
            record("c", "base", "yes", "yes"),
            record("c", "candidate", "yes", "yes"),
        ]

    def test_paired_metrics_and_efficiency(self):
        r = compare(self.rows, "base", "candidate")
        self.assertEqual(r["evidence_type"],
                         "externally_supplied_model_predictions_origin_unverified")
        x = r["comparisons"][0]
        self.assertEqual(x["n_paired_examples"], 3)
        self.assertAlmostEqual(x["paired_valid_exact_match_difference"], 0.0)
        self.assertEqual((x["candidate_only_correct"],
                          x["baseline_only_correct"],
                          x["both_same_correctness"]), (1, 1, 1))
        self.assertAlmostEqual(x["input_tokens_mean_difference"], 2)
        self.assertAlmostEqual(x["latency_ms_mean_difference"], 10)

    def test_invalid_output_does_not_count_as_correct(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["output_valid"] = False
        x = compare(rows, "base", "candidate")["comparisons"][0]
        self.assertEqual(x["candidate_only_correct"], 0)
        self.assertAlmostEqual(x["paired_valid_exact_match_difference"], -1/3)
        self.assertAlmostEqual(x["valid_output_rate_difference"], -1/3)

    def test_reject_different_ids(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["id"] = "not-a"
        with self.assertRaisesRegex(ValueError, "unmatched examples"):
            compare(rows, "base", "candidate")

    def test_reject_different_gold_labels(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["expected"] = "different"
        with self.assertRaisesRegex(ValueError, "different reference outputs"):
            compare(rows, "base", "candidate")

    def test_reject_duplicate_and_third_method(self):
        with self.assertRaises(ValueError):
            compare(self.rows + [self.rows[0]], "base", "candidate")
        rows = copy.deepcopy(self.rows)
        rows[0]["prompt_id"] = "third"
        with self.assertRaisesRegex(ValueError, "unexpected prompt ID"):
            compare(rows, "base", "candidate")

    def test_exact_discordance_boundary_and_ids(self):
        self.assertAlmostEqual(exact_discordance_p(4, 0), .125)
        self.assertEqual(exact_discordance_p(0, 0), 1.0)
        with self.assertRaises(ValueError):
            compare(self.rows, "base", "base")


if __name__ == "__main__":
    unittest.main()
