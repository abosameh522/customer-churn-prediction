from pathlib import Path

import joblib
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


def main():
    project_dir = Path(__file__).resolve().parent
    data = pd.read_csv(project_dir / "data" / "telco_churn.csv")
    data = clean_data(data)
    X = data.drop(columns="Churn")
    y = data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocessor = create_preprocessor(X_train)
    pipelines = train_models(X_train, y_train, preprocessor)
    results = compare_models(pipelines, X_test, y_test)
    print(f"Customers after cleaning: {len(data):,}")
    print(
        f"Training customers: {len(X_train):,}; "
        f"test customers: {len(X_test):,}"
    )
    print("\nTest-set model comparison (sorted by ROC-AUC):")
    print(results.to_string(index=False, float_format="{:.4f}".format))

    best_name = results.iloc[0]["Model"]
    best_pipeline = pipelines[best_name]
    feature_count = len(
        best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    print(f"\nBest model: {best_name}")
    print(
        f"Features: {X.shape[1]} original, "
        f"{feature_count} after preprocessing"
    )

    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, models_dir / "churn_model.joblib")

    reports_dir = project_dir / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    report = save_confusion_matrix(
        best_pipeline, best_name, X_test, y_test, figures_dir
    )
    print(f"\nClassification report — {best_name}\n{report}")
    save_roc_curves(pipelines, X_test, y_test, figures_dir)
    importance = save_feature_importance(
        pipelines["Random Forest"], figures_dir
    )
    results.to_csv(reports_dir / "model_comparison.csv", index=False)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    (reports_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    print("\nTop 10 Random Forest features:")
    print(
        importance.head(10).to_string(
            index=False, float_format="{:.4f}".format
        )
    )

    print("\nSaved models/churn_model.joblib and results in reports/.")


if __name__ == "__main__":
    main()
