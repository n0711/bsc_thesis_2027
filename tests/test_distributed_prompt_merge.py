"""Algebraic checks only, no network, robotics or actual language models."""
import itertools
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distributed_prompt_merge import merge_all, merge_best


class DistributedMergeTests(unittest.TestCase):
    def setUp(self):
        self.a = (0.42, ("short", "clear"))
        self.b = (0.81, ("verify", "formal"))
        self.c = (0.81, ("verify", "json"))

    def test_identity(self):
        self.assertEqual(merge_best(None, self.a), self.a)
        self.assertEqual(merge_best(self.a, None), self.a)
        self.assertIsNone(merge_best(None, None))

    def test_idempotence(self):
        self.assertEqual(merge_best(self.a, self.a), self.a)

    def test_commutativity(self):
        for x, y in itertools.product((None, self.a, self.b, self.c), repeat=2):
            self.assertEqual(merge_best(x, y), merge_best(y, x))

    def test_associativity(self):
        for x, y, z in itertools.product((None, self.a, self.b, self.c), repeat=3):
            self.assertEqual(merge_best(merge_best(x, y), z),
                             merge_best(x, merge_best(y, z)))

    def test_permutations_and_duplicates(self):
        for order in itertools.permutations((self.a, self.b, self.c)):
            self.assertEqual(merge_all(list(order) + [self.a, self.c]), self.c)

    def test_missing_message_then_recovery(self):
        partial = merge_all([self.a, self.b])
        self.assertEqual(partial, self.b)
        self.assertEqual(merge_best(partial, self.c), self.c)
        self.assertLessEqual(partial[0], self.c[0])

    def test_reject_nonfinite_or_invalid(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                merge_best(self.a, (value, ("x",)))
        with self.assertRaises(ValueError):
            merge_best((0.4, ()), self.b)


if __name__ == "__main__":
    unittest.main()
