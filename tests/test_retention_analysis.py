import unittest

import pandas as pd

from src.retention_analysis import priority_table, rank_customers


class RetentionTests(unittest.TestCase):
    def test_capture_precision_and_lift(self):
        labels = pd.Series([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0]
        result = priority_table(rank_customers(labels, scores))
        self.assertEqual(result["Customers reviewed"].tolist(), [1, 2, 3])
        self.assertEqual(result["Churners captured"].tolist(), [1, 2, 2])
        self.assertEqual(result["Capture rate"].tolist(), [0.5, 1.0, 1.0])
        self.assertAlmostEqual(result.iloc[0]["Lift"], 5.0)
        self.assertAlmostEqual(result.iloc[2]["Precision"], 2 / 3)

    def test_ties_preserve_input_order(self):
        ranked = rank_customers(pd.Series([0, 1]), [0.5, 0.5])
        self.assertEqual(ranked["churn"].tolist(), [0, 1])

    def test_invalid_or_undefined_inputs(self):
        for labels, scores in [([0, 0], [0.1, 0.2]), ([1], [float("nan")]), ([1], [1.1]), ([2], [0.5])]:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(ValueError):
                    rank_customers(pd.Series(labels), scores)
