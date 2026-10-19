import unittest
from experiments.strict_engine.verify import verify_metric_record


class StrictAuditTests(unittest.TestCase):
    def test_tampered_metric_is_rejected(self):
        good = dict(ranking=[2,1], truth=1, top1=0, top3=1, rr=.5)
        verify_metric_record(good)
        with self.assertRaisesRegex(AssertionError, 'metric'):
            verify_metric_record(dict(good, top1=1))


class StrictGridAuditTests(unittest.TestCase):
    def test_missing_method_records_are_rejected(self):
        from experiments.strict_engine.verify import verify_prediction_grid
        counts = {'fusion_strict': 10, 'projection': 10, 'cosine': 10, 'fusion_open_topology': 1}
        verify_prediction_grid(counts, 1, 1)
        with self.assertRaisesRegex(AssertionError, 'grid'):
            verify_prediction_grid(dict(counts, cosine=9), 1, 1)


if __name__ == '__main__': unittest.main()
