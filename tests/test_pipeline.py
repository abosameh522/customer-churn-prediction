"""Offline tests for cleaning, preprocessing, training, evaluation, and reporting.

These tests use small synthetic frames so they run in under a few seconds and
need neither the real dataset nor a fitted artifact.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preparation import clean_data, create_preprocessor
from src.evaluate import (
    compare_models,
    save_confusion_matrix,
    save_feature_importance,
    save_roc_curves,
)
from src.train import train_models


def synthetic_frame(rows=120):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "customerID": [f"id-{i:04d}" for i in range(rows)],
            "tenure": rng.integers(0, 72, rows),
            "MonthlyCharges": rng.uniform(20, 120, rows).round(2),
            "TotalCharges": rng.uniform(0, 8000, rows).round(2).astype(str),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], rows),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], rows),
            "PaymentMethod": rng.choice(["Electronic check", "Credit card"], rows),
        }
    )
    # Roughly 30% churn, correlated with short tenure so models can learn.
    frame["Churn"] = np.where(
        (frame["tenure"] < 20) & (rng.random(rows) < 0.7), "Yes", "No"
    )
    return frame


def fitted_bundle():
    frame = clean_data(synthetic_frame())
    X = frame.drop(columns="Churn")
    y = frame["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipelines = train_models(X_train, y_train, create_preprocessor(X_train))
    return pipelines, X_test, y_test


class CleanDataTests(unittest.TestCase):
    def test_maps_target_and_drops_customer_id(self):
        cleaned = clean_data(synthetic_frame())

        self.assertNotIn("customerID", cleaned.columns)
        self.assertEqual(set(cleaned["Churn"].unique()), {0, 1})
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned["Churn"]))

    def test_blank_total_charges_becomes_nan(self):
        frame = synthetic_frame()
        frame.loc[0, "TotalCharges"] = " "
        cleaned = clean_data(frame)

        self.assertTrue(np.isnan(cleaned.loc[0, "TotalCharges"]))

    def test_rejects_unexpected_target_values(self):
        frame = synthetic_frame()
        frame.loc[0, "Churn"] = "Maybe"

        with self.assertRaises(ValueError):
            clean_data(frame)


class PreprocessorTests(unittest.TestCase):
    def test_fit_transform_shapes_and_types(self):
        frame = clean_data(synthetic_frame())
        X = frame.drop(columns="Churn")
        preprocessor = create_preprocessor(X)
        matrix = preprocessor.fit_transform(X)

        self.assertEqual(matrix.shape[0], len(X))
        self.assertGreater(matrix.shape[1], X.shape[1])
        self.assertTrue(np.isfinite(matrix).all())

    def test_unseen_category_is_ignored_not_raised(self):
        frame = clean_data(synthetic_frame())
        X = frame.drop(columns="Churn")
        preprocessor = create_preprocessor(X).fit(X)

        unseen = X.iloc[:3].copy()
        unseen["Contract"] = "Lifetime deal"
        matrix = preprocessor.transform(unseen)

        self.assertEqual(matrix.shape[0], 3)
        self.assertTrue(np.isfinite(matrix).all())


class TrainingTests(unittest.TestCase):
    def test_returns_three_fitted_pipelines(self):
        pipelines, _, _ = fitted_bundle()

        self.assertEqual(
            set(pipelines),
            {"Logistic Regression", "Decision Tree", "Random Forest"},
        )
        for pipeline in pipelines.values():
            self.assertTrue(hasattr(pipeline, "predict_proba"))
            self.assertTrue(hasattr(pipeline, "predict"))

    def test_all_models_produce_probabilities_in_unit_interval(self):
        pipelines, X_test, _ = fitted_bundle()

        for name, pipeline in pipelines.items():
            with self.subTest(model=name):
                probabilities = pipeline.predict_proba(X_test)[:, 1]
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertGreaterEqual(probabilities.min(), 0)
                self.assertLessEqual(probabilities.max(), 1)


class EvaluationTests(unittest.TestCase):
    def test_comparison_is_sorted_by_auc_and_has_all_metrics(self):
        pipelines, X_test, y_test = fitted_bundle()
        comparison = compare_models(pipelines, X_test, y_test)

        self.assertEqual(
            list(comparison.columns),
            ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
        )
        self.assertTrue(comparison["ROC_AUC"].is_monotonic_decreasing)
        self.assertTrue(comparison["ROC_AUC"].between(0, 1).all())

    def test_reporting_writes_expected_files(self):
        pipelines, X_test, y_test = fitted_bundle()
        best_name = "Random Forest"
        with tempfile.TemporaryDirectory() as directory:
            figures = Path(directory)
            report = save_confusion_matrix(
                pipelines[best_name], best_name, X_test, y_test, figures
            )
            save_roc_curves(pipelines, X_test, y_test, figures)
            importance = save_feature_importance(pipelines[best_name], figures)

            self.assertIn("Churned", report)
            self.assertTrue((figures / "confusion_matrix.png").is_file())
            self.assertTrue((figures / "roc_curve.png").is_file())
            self.assertTrue((figures / "feature_importance.png").is_file())
            self.assertEqual(list(importance.columns), ["Feature", "Importance"])
            self.assertTrue(importance["Importance"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
