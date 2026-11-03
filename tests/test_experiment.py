import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path
from sys import path
path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment import (VOCAB, TRAIN_TARGETS, HOLDOUT_TARGETS, EPS, BUDGETS, LENGTHS,
                        sigmoid, score, score_gradient, embed_prompt, enumerate_space,
                        random_search, coordinate_search, exact_oracle,
                        exact_random_optimal_probability, exact_random_expected_best,
                        bootstrap_mean_interval, run)


class SurrogateTests(unittest.TestCase):
    def test_sigmoid_extremes(self):
        self.assertAlmostEqual(sigmoid(1000), 1.0)
        self.assertAlmostEqual(sigmoid(-1000), 0.0)
        self.assertAlmostEqual(sigmoid(0), 0.5)

    def test_enumeration_and_boundaries(self):
        self.assertEqual(len(list(enumerate_space(3))), 216)
        self.assertEqual(len(list(enumerate_space(4))), 1296)
        with self.assertRaises(ValueError): list(enumerate_space(0))

    def test_validation(self):
        with self.assertRaises(ValueError): embed_prompt(())
        with self.assertRaises(ValueError): embed_prompt(("unknown",))
        with self.assertRaises(ValueError): random_search(3, 0, 0)
        with self.assertRaises(ValueError): coordinate_search(3, 0, 0)

    def test_gradient_central_difference(self):
        x = (0.23, -0.17)
        grad = score_gradient(x)
        h = 1e-6
        for j in range(2):
            def f(delta):
                y = list(x)
                y[j] += delta
                return sum(sigmoid(3*sum(a*b for a,b in zip(y,u))) for u in TRAIN_TARGETS)/len(TRAIN_TARGETS)
            self.assertAlmostEqual(grad[j], (f(h)-f(-h))/(2*h), places=7)

    def test_oracle_and_nonnegative_regret(self):
        oracle = exact_oracle(3)
        self.assertEqual(oracle["space_size"], 216)
        self.assertEqual(len(oracle["prompts"]), 1)
        self.assertEqual(oracle["prompts"][0], ("formal",) * 3)
        self.assertTrue(all(oracle["score"] >= score(p)-EPS for p in enumerate_space(3)))

    def test_methods_use_exact_budget(self):
        for n in LENGTHS:
            for b in BUDGETS:
                for method in (random_search, coordinate_search):
                    r = method(n, b, 7)
                    self.assertEqual(r.calls, b)
                    self.assertLessEqual(r.unique_candidates, b)
                    self.assertLessEqual(r.train_score, exact_oracle(n)["score"]+EPS)

    def test_determinism(self):
        for meth in (random_search, coordinate_search):
            self.assertEqual(meth(3, 36, 5), meth(3, 36, 5))

    def test_exact_random_probability(self):
        self.assertAlmostEqual(exact_random_optimal_probability(4, 1, 2), 7/16)
        self.assertAlmostEqual(exact_random_optimal_probability(2, 1, 1), 0.5)
        self.assertEqual(exact_random_optimal_probability(6, 6, 4), 1.0)

    def test_expected_best(self):
        self.assertAlmostEqual(exact_random_expected_best([0., 1.], 1), .5)
        self.assertAlmostEqual(exact_random_expected_best([0., 1.], 2), .75)

    def test_separate_targets(self):
        self.assertNotEqual(TRAIN_TARGETS, HOLDOUT_TARGETS)
        self.assertNotEqual(score(("formal",)*3, TRAIN_TARGETS), score(("formal",)*3, HOLDOUT_TARGETS))

    def test_bootstrap_reproducible(self):
        self.assertEqual(bootstrap_mean_interval([0.2,0.3,0.4], nboot=100),
                         bootstrap_mean_interval([0.2,0.3,0.4], nboot=100))

    def test_full_run_schema(self):
        with tempfile.TemporaryDirectory() as d:
            summary = run(d)
            self.assertEqual(len(summary["experiments"]), 12)
            self.assertTrue(all(e["methods"]["random"]["mean_regret"] >= 0 for e in summary["experiments"]))
            with (Path(d)/"runs.csv").open() as f:
                self.assertEqual(sum(1 for _ in f)-1, 1200)
            self.assertEqual(json.loads((Path(d)/"summary.json").read_text())["evidence_type"],
                             "synthetic_surrogate_not_LLM")


if __name__ == "__main__": unittest.main()
