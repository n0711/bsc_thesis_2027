import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gradient_audit


class GradientAuditTests(unittest.TestCase):
    def test_midrank_spearman_handles_ties_and_constant_rankings(self):
        self.assertAlmostEqual(
            gradient_audit._spearman([1, 1, 2], [4, 4, 8]), 1.0
        )
        self.assertAlmostEqual(
            gradient_audit._spearman([1, 2, 3], [3, 2, 1]), -1.0
        )
        self.assertIsNone(gradient_audit._spearman([1, 1], [1, 2]))

    def test_length_two_audit_counts_all_moves_and_reports_state_rankings(self):
        result = gradient_audit.audit(2)
        self.assertEqual(result["prompt_count"], 36)
        self.assertEqual(result["directed_move_count"], 360)
        self.assertEqual(result["sign_counts"], {
            "true_positive": 180,
            "false_positive": 20,
            "false_negative": 0,
            "true_negative": 160,
        })
        self.assertEqual(len(result["states"]), 36)
        self.assertTrue(all(state["neighbor_count"] == 10 for state in result["states"]))
        self.assertTrue(all(
            state["spearman_rank_correlation"] is not None
            for state in result["states"]
        ))
        top1_count = sum(state["top1_agreement"] for state in result["states"])
        self.assertEqual(top1_count, result["top1_agreement_count"])
        self.assertEqual(result["top1_agreement_rate"], top1_count / 36)
        self.assertTrue(any(not state["top1_agreement"] for state in result["states"]))
        self.assertEqual(result["nonimproving_nonzero_gradient_state_count"], 2)
        self.assertTrue(all(
            state["max_discrete_change"] <= gradient_audit.EPS
            and state["gradient_norm"] > gradient_audit.EPS
            for state in result["states"]
            if state["nonzero_gradient_without_improvement"]
        ))

    def test_json_and_latex_outputs_are_deterministic(self):
        report = {
            "schema_version": 1,
            "evidence_type": "synthetic_surrogate_not_LLM",
            "improvement_tolerance": 1e-12,
            "ranking_ties": "test",
            "lengths": [gradient_audit.audit(2)],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gradient_audit.write_outputs(report, root)
            json_before = (root / "results" / "gradient_audit.json").read_bytes()
            tex_before = (root / "tex" / "gradient_moves.tex").read_bytes()
            self.assertNotIn(b"\r", json_before)
            self.assertNotIn(b"\r", tex_before)
            gradient_audit.write_outputs(report, root)
            self.assertEqual(
                json_before, (root / "results" / "gradient_audit.json").read_bytes()
            )
            self.assertEqual(
                tex_before, (root / "tex" / "gradient_moves.tex").read_bytes()
            )
            decoded = json.loads(json_before)
            self.assertEqual(decoded["lengths"][0]["prompt_count"], 36)


if __name__ == "__main__":
    unittest.main()
