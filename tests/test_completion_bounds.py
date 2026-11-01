import importlib
from itertools import product
import os
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from completion_monitoring.paths import activate_sources

class CompletionBoundsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        activate_sources()
        cls.previous = Path.cwd()
        cls.temp = tempfile.TemporaryDirectory()
        os.chdir(cls.temp.name)
        cls.engine = importlib.import_module('chronological_protocol')

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.previous)
        cls.temp.cleanup()

    def test_certificates_match_every_admissible_completion(self):
        for n in range(1, 7):
            for observed in product((None, 0, 1), repeat=n):
                feedback = {i: value for i, value in enumerate(observed) if value is not None}
                unresolved = [i for i, value in enumerate(observed) if value is None]
                block = {'sample_ids': list(range(n)), 'reference_loss': 0.2}
                decision = self.engine.evaluate_bounds({'feedback': feedback}, block)
                completed_statuses = set()
                for remaining in product((0, 1), repeat=len(unresolved)):
                    errors = sum(feedback.values()) + sum(remaining)
                    status = 'DEGRADATION' if errors / n - 0.2 > self.engine.EPSILON + self.engine.FLOAT_TOL else 'TOLERABLE'
                    completed_statuses.add(status)
                if decision is None:
                    self.assertEqual(len(completed_statuses), 2)
                else:
                    self.assertEqual(completed_statuses, {decision['decision']})
if __name__ == '__main__':
    unittest.main()
