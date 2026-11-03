import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paired_analysis import analyze, exact_two_sided_sign_p, paired_bootstrap

class PairedAnalysisTests(unittest.TestCase):
    def test_sign_ties(self):
        self.assertEqual(exact_two_sided_sign_p(0,0),1.0)
        self.assertEqual(exact_two_sided_sign_p(4,0),0.125)
        self.assertEqual(exact_two_sided_sign_p(2,2),1.0)
    def test_bootstrap_determinism(self):
        self.assertEqual(paired_bootstrap([1.,2.,3.]),paired_bootstrap([1.,2.,3.]))
        self.assertEqual(paired_bootstrap([2.,2.]),[2.,2.])
    def test_reject_incomplete_pairs(self):
        with self.assertRaises(ValueError):
            analyze([{'length':2,'budget':1,'seed':0,'method':'random','calls':1,'train_score':.2}])
    def test_reject_budget_mismatch(self):
        rows=[{'length':2,'budget':2,'seed':0,'method':m,'calls':1,'train_score':.2} for m in ('random','coordinate')]
        with self.assertRaises(ValueError): analyze(rows)
    def test_paired_mean(self):
        rows=[{'length':2,'budget':1,'seed':0,'method':m,'calls':1,'train_score':s} for m,s in [('random',.2),('coordinate',.4)]]
        result=analyze(rows)['configurations'][0]
        self.assertAlmostEqual(result['mean_coordinate_minus_random'],.2)
        self.assertEqual(result['wins'],1)
