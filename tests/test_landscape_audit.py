import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import landscape_audit


class LandscapeAuditTests(unittest.TestCase):
    def test_length_two_stopping_points_include_nonzero_gradients(self):
        result = landscape_audit.landscape(2)
        self.assertEqual(result["prompt_count"], 36)
        self.assertEqual(result["directed_comparison_count"], 360)
        self.assertEqual(result["terminal_count"], 2)
        self.assertEqual(result["suboptimal_terminal_count"], 1)
        self.assertEqual(result["nonimproving_nonzero_gradient_count"], 2)
        self.assertTrue(all(
            state["max_neighbor_change"] <= landscape_audit.EPS
            and state["nonzero_gradient"]
            for state in result["terminal_states"]
        ))

    def test_json_and_latex_outputs_are_deterministic(self):
        report = {
            "schema_version": 1,
            "evidence_type": "synthetic_surrogate_not_LLM",
            "improvement_tolerance": landscape_audit.EPS,
            "lengths": [landscape_audit.landscape(2)],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            landscape_audit.write_outputs(report, root)
            json_before = (root / "results" / "landscape_audit.json").read_bytes()
            tex_before = (root / "tex" / "coordinate_landscape.tex").read_bytes()
            self.assertNotIn(b"\r", json_before)
            self.assertNotIn(b"\r", tex_before)
            landscape_audit.write_outputs(report, root)
            self.assertEqual(
                json_before, (root / "results" / "landscape_audit.json").read_bytes()
            )
            self.assertEqual(
                tex_before, (root / "tex" / "coordinate_landscape.tex").read_bytes()
            )
            decoded = json.loads(json_before)
            self.assertEqual(decoded["lengths"][0]["terminal_count"], 2)


if __name__ == "__main__":
    unittest.main()
