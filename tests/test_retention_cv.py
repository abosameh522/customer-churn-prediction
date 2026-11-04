"""Unit tests for the retention-cost helpers in src/retention_cv.py.

The cross-validation and holdout orchestration runs are exercised with a small
synthetic frame and a reduced fold count so tests stay fast and offline.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")

from src.data_preparation import create_preprocessor
from src.retention_cv import (
    RETENTION_COST,
    final_holdout_check,
    fit_pipeline,
    metrics_at,
    retention_cost,
    run_cross_validation,
    save_feature_importance,
    select_threshold,
    summarize,
    unit_cost_of_miss,
)


def synthetic_frame(rows=60):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "tenure": rng.integers(0, 72, rows),
        "MonthlyCharges": rng.uniform(20, 120, rows).round(2),
        "Contract": rng.choice(["Month-to-month", "Two year"], rows),
        "InternetService": rng.choice(["DSL", "Fiber optic"], rows),
    })
    churn = np.where((frame["tenure"] < 20) & (rng.random(rows) < 0.8), 1, 0)
    if churn.sum() < 8:  # keep both classes well represented for stratification
        churn[:8] = 1
    return frame, churn


class RetentionCostTests(unittest.TestCase):
    def test_counts_offers_and_missed_revenue(self):
        labels = np.array([1, 0, 1, 0])
        probabilities = np.array([0.9, 0.4, 0.1, 0.2])
        miss_cost = np.array([840.0, 0.0, 120.0, 0.0])
        result = retention_cost(labels, probabilities, 0.5, miss_cost)

        self.assertEqual(result["customers_offered"], 1)
        self.assertEqual(result["churners_missed"], 1)
        self.assertAlmostEqual(result["outreach_spend"], RETENTION_COST)
        self.assertAlmostEqual(result["lost_revenue"], 120.0)
        self.assertAlmostEqual(result["total_cost"], 120.0 + RETENTION_COST)

    def test_threshold_boundary_counts_as_positive(self):
        result = retention_cost(
            np.array([1]), np.array([0.5]), 0.5, np.array([12.0])
        )
        self.assertEqual(result["customers_offered"], 1)
        self.assertEqual(result["churners_missed"], 0)


class UnitCostTests(unittest.TestCase):
    def test_miss_cost_is_monthly_charges_times_horizon(self):
        charges = pd.DataFrame({"MonthlyCharges": [50.0, 70.0]})
        costs = unit_cost_of_miss(pd.Series([1, 0]), charges)
        np.testing.assert_allclose(costs, [50.0 * 12, 70.0 * 12])


class ThresholdTests(unittest.TestCase):
    def test_select_threshold_returns_curve_minimum(self):
        labels = pd.Series(np.concatenate([np.ones(10), np.zeros(10)]).astype(int))
        probabilities = np.concatenate([
            np.full(9, 0.9), np.full(1, 0.06), np.full(10, 0.4)
        ])
        miss_cost = np.full(20, 100.0)
        best, curve = select_threshold(labels, probabilities, miss_cost)

        self.assertEqual(curve.loc[curve["total_cost"].idxmin(), "threshold"], best)
        self.assertGreaterEqual(curve["total_cost"].min(), 0)

    def test_metrics_at_boundary(self):
        # A probability of exactly 0.5 counts as positive at threshold 0.5.
        measured = metrics_at(np.array([1, 0]), np.array([0.5, 0.55]), 0.5)
        self.assertEqual(measured["recall"], 1.0)
        self.assertEqual(measured["precision"], 0.5)


class SummarizeTests(unittest.TestCase):
    def test_reports_mean_and_std_per_model(self):
        folds = pd.DataFrame({
            "Model": ["A", "A"], "Fold": [1, 2], "ROC_AUC": [0.8, 0.9],
        })
        summary = summarize(folds)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary.loc[0, "ROC_AUC_mean"], 0.85)
        # pandas .std() uses sample std (ddof=1): sqrt(0.005) for two folds.
        self.assertAlmostEqual(summary.loc[0, "ROC_AUC_std"], 0.05 * np.sqrt(2))


class CrossValidationTests(unittest.TestCase):
    def test_run_cross_validation_covers_every_model_and_fold(self):
        features, labels = synthetic_frame()
        with mock.patch("src.retention_cv.N_FOLDS", 2):
            folds = run_cross_validation(features, pd.Series(labels))
        self.assertEqual(len(folds), len(folds["Model"].unique()) * 2)
        self.assertTrue(np.isfinite(folds["TotalCost@tuned"]).all())
        self.assertTrue(folds["Threshold"].between(0.05, 0.95).all())
        self.assertTrue(folds["Recall@tuned"].between(0, 1).all())

    def test_final_holdout_check_runs_once_on_test_split(self):
        features, labels = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("src.retention_cv.REPORTS_DIR", Path(directory)):
                with mock.patch("src.retention_cv.N_FOLDS", 2):
                    result, pipeline = final_holdout_check(
                        features, pd.Series(labels), "Logistic Regression"
                    )
            self.assertEqual(pipeline.predict_proba(features).shape[0], len(features))
            self.assertGreaterEqual(result["ROC_AUC"], 0.0)
            self.assertLessEqual(result["ROC_AUC"], 1.0)
            self.assertTrue((Path(directory) / "threshold_cost_curve.csv").is_file())


class FeatureImportanceTests(unittest.TestCase):
    def test_importance_matches_raw_feature_count(self):
        features, labels = synthetic_frame()
        pipeline = fit_pipeline(
            LogisticRegression(max_iter=1000),
            create_preprocessor(features),
        )
        pipeline.fit(features, labels)
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            with mock.patch("src.retention_cv.REPORTS_DIR", reports):
                with mock.patch("src.retention_cv.FIGURES_DIR", reports / "figures"):
                    importance = save_feature_importance(
                        pipeline, features, pd.Series(labels), "Logistic Regression"
                    )
            self.assertEqual(len(importance), features.shape[1])
            self.assertEqual(
                importance.iloc[0]["Feature"] in features.columns, True
            )
            self.assertTrue((reports / "permutation_importance.csv").is_file())
            self.assertTrue(
                (reports / "figures" / "permutation_importance.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
