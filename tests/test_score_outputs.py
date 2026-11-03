import unittest
from pathlib import Path
from sys import path
path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from score_outputs import summarize

RECORD = {"id":"a", "task":"classification", "model":"offline-fixture", "prompt_id":"p1",
          "expected":"yes", "prediction":"yes", "output_valid":True,
          "input_tokens":20, "output_tokens":1, "latency_ms":50}


class ScoringTests(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError): summarize([])

    def test_exact_match_and_validity(self):
        b = {**RECORD, "id":"b", "prediction":"no", "output_valid":False, "latency_ms":60}
        g = summarize([RECORD, b])["groups"]["offline-fixture/p1/classification"]
        self.assertEqual(g["n"], 2)
        self.assertEqual(g["exact_match"], 0.5)
        self.assertEqual(g["valid_output_rate"], 0.5)
        self.assertEqual(g["mean_latency_ms"], 55)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): summarize([RECORD, RECORD])

    def test_missing_field_rejected(self):
        a = {k:v for k,v in RECORD.items() if k != "prediction"}
        with self.assertRaises(ValueError): summarize([a])

    def test_bad_types_rejected(self):
        with self.assertRaises(ValueError): summarize([{**RECORD,"output_valid":"true"}])
        with self.assertRaises(ValueError): summarize([{**RECORD,"latency_ms":-1}])


if __name__ == '__main__': unittest.main()
