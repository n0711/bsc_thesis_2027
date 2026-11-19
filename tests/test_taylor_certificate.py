"""Regression checks for mathematically certified synthetic token moves."""
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment import TRAIN_TARGETS, score_gradient, sigmoid
from taylor_certificate import (audit_length, build_report,
                                gradient_lipschitz_bound, move_certificate)


def continuous_score(point):
    return sum(
        sigmoid(3 * sum(a * b for a, b in zip(point, u)))
        for u in TRAIN_TARGETS
    ) / len(TRAIN_TARGETS)


class TaylorCertificateTests(unittest.TestCase):
    def test_global_hessian_bound(self):
        # Sum of squared fixed target norms = 1.80; the mean is 0.45.
        self.assertAlmostEqual(gradient_lipschitz_bound(), .45 * math.sqrt(3) / 2)
        p = (3 - math.sqrt(3)) / 6
        self.assertAlmostEqual(abs(p * (1 - p) * (1 - 2 * p)),
                               math.sqrt(3) / 18)

    def test_off_grid_continuous_taylor_remainders(self):
        rng = random.Random(701)
        bound = gradient_lipschitz_bound()
        for _ in range(100):
            x = (rng.uniform(-1.3, 1.3), rng.uniform(-1.3, 1.3))
            y = (rng.uniform(-1.3, 1.3), rng.uniform(-1.3, 1.3))
            cert = move_certificate(x, y, score_gradient(x), bound)
            change = continuous_score(y) - continuous_score(x)
            self.assertLessEqual(cert["lower"] - 1e-12, change)
            self.assertLessEqual(change, cert["upper"] + 1e-12)

    def test_certificate_validation(self):
        with self.assertRaises(ValueError):
            gradient_lipschitz_bound(())
        with self.assertRaises(ValueError):
            move_certificate((0, 0), (1,), (0, 0), 1)
        with self.assertRaises(ValueError):
            move_certificate((0,), (1,), (0,), -1)

    def test_exhaustive_covered_moves_and_uncertainty(self):
        report = build_report()
        by_length = {r["length"]: r for r in report["lengths"]}
        for length, moves, improvements, decreases, false_positives in (
            (2, 360, 101, 79, 20),
            (3, 3240, 1081, 906, 133),
            (4, 25920, 9297, 8118, 851),
        ):
            result = by_length[length]
            self.assertEqual(result["moves"], moves)
            self.assertEqual(result["certified_improvements"], improvements)
            self.assertEqual(result["certified_decreases"], decreases)
            self.assertEqual(result["first_order_false_positives"], false_positives)
            self.assertEqual(result["certificate_sign_errors"], 0)
            self.assertEqual(result["taylor_bound_violations"], 0)
            self.assertEqual(
                result["uncertified_moves"],
                moves - improvements - decreases,
            )
        total = report["total"]
        self.assertEqual(total["moves"], 29520)
        self.assertEqual(total["first_order_false_positives"], 1004)
        self.assertEqual(total["certified_improvements"], 10479)
        self.assertEqual(total["certified_decreases"], 9103)
        self.assertAlmostEqual(total["certified_coverage"], 19582 / 29520)


if __name__ == "__main__":
    unittest.main()
