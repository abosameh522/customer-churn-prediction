"""Retention-cost threshold selection and stratified k-fold cross-validation.

Churn is about retention: this module treats a missed churner as losing that
customer's expected remaining revenue, and treating a loyal customer as paying
for an unnecessary retention offer. The decision threshold is chosen to
minimise total simulated cost on out-of-fold predictions, never on the test set.

Cross-validation uses stratified folds so every fold keeps the 26.5% churn
share, and reports mean and standard deviation for each metric.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data_preparation import clean_data, create_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
N_FOLDS = 5

# Retention economics. A retention offer costs RETENTION_COST per contacted
# customer. Failing to keep a churner loses their expected remaining revenue,
# approximated here by average monthly charges times remaining contract horizon.
RETENTION_COST = 20.0  # currency units per outreach attempt
DEFAULT_HORIZON_MONTHS = 12.0


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data = clean_data(pd.read_csv(DATA_PATH))
    X = data.drop(columns="Churn")
    y = data["Churn"]
    return X, y


def build_models() -> dict[str, Pipeline]:
    """Pipelines that carry imbalance handling inside them, so nothing leaks."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.08,
            max_iter=250,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def fit_pipeline(estimator: Pipeline, preprocessor: ColumnTransformer) -> Pipeline:
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    return pipeline


def unit_cost_of_miss(y_true: pd.Series, X_frame: pd.DataFrame) -> np.ndarray:
    """Expected revenue lost per churned customer: monthly charge x horizon."""
    monthly = pd.to_numeric(X_frame["MonthlyCharges"], errors="coerce").fillna(
        pd.to_numeric(X_frame["MonthlyCharges"], errors="coerce").median()
    )
    return monthly.to_numpy() * DEFAULT_HORIZON_MONTHS


def retention_cost(
    y_true, probabilities, threshold, miss_unit_cost, offer_cost=RETENTION_COST
) -> dict[str, float]:
    """Simulated spend: offers sent plus revenue lost through missed churners."""
    y_true = np.asarray(y_true)
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    missed = (y_true == 1) & (predicted == 0)
    offered = predicted == 1
    lost_revenue = float(np.sum(np.asarray(miss_unit_cost)[missed]))
    outreach_spend = float(offered.sum()) * offer_cost
    return {
        "threshold": float(threshold),
        "customers_offered": int(offered.sum()),
        "churners_missed": int(missed.sum()),
        "lost_revenue": lost_revenue,
        "outreach_spend": outreach_spend,
        "total_cost": lost_revenue + outreach_spend,
    }


def select_threshold(y_true, probabilities, miss_unit_cost, offer_cost=RETENTION_COST):
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = [
        retention_cost(y_true, probabilities, t, miss_unit_cost, offer_cost)
        for t in thresholds
    ]
    table = pd.DataFrame(rows)
    best = table.loc[table["total_cost"].idxmin()]
    return float(best["threshold"]), table


def metrics_at(y_true, probabilities, threshold) -> dict[str, float]:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
    }


def run_cross_validation(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Stratified k-fold CV with in-fold preprocessing and cost-based thresholds."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for name, estimator in build_models().items():
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            preprocessor = create_preprocessor(X_train)
            pipeline = fit_pipeline(estimator, preprocessor)
            pipeline.fit(X_train, y_train)

            probabilities = pipeline.predict_proba(X_valid)[:, 1]
            miss_cost = unit_cost_of_miss(y_valid, X_valid)

            # The threshold must come from training-side signals only, so it is
            # chosen on out-of-fold probabilities within the training fold.
            inner_cv = StratifiedKFold(
                n_splits=min(3, int(y_train.value_counts().min())),
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            inner_probabilities = np.zeros(len(X_train))
            for inner_train, inner_valid in inner_cv.split(X_train, y_train):
                inner_prep = create_preprocessor(X_train.iloc[inner_train])
                inner_pipeline = fit_pipeline(estimator, inner_prep)
                inner_pipeline.fit(X_train.iloc[inner_train], y_train.iloc[inner_train])
                inner_probabilities[inner_valid] = inner_pipeline.predict_proba(
                    X_train.iloc[inner_valid]
                )[:, 1]
            train_miss_cost = unit_cost_of_miss(y_train, X_train)
            threshold, _ = select_threshold(y_train, inner_probabilities, train_miss_cost)

            default_metrics = metrics_at(y_valid, probabilities, 0.5)
            tuned_metrics = metrics_at(y_valid, probabilities, threshold)
            default_cost = retention_cost(y_valid, probabilities, 0.5, miss_cost)
            tuned_cost = retention_cost(y_valid, probabilities, threshold, miss_cost)

            records.append(
                {
                    "Model": name,
                    "Fold": fold,
                    "ROC_AUC": roc_auc_score(y_valid, probabilities),
                    "Recall@0.5": default_metrics["recall"],
                    "Precision@0.5": default_metrics["precision"],
                    "F1@0.5": default_metrics["f1"],
                    "Threshold": threshold,
                    "Recall@tuned": tuned_metrics["recall"],
                    "Precision@tuned": tuned_metrics["precision"],
                    "F1@tuned": tuned_metrics["f1"],
                    "OffersSent@tuned": tuned_cost["customers_offered"],
                    "ChurnersMissed@tuned": tuned_cost["churners_missed"],
                    "LostRevenue@tuned": tuned_cost["lost_revenue"],
                    "OutreachSpend@tuned": tuned_cost["outreach_spend"],
                    "TotalCost@0.5": default_cost["total_cost"],
                    "TotalCost@tuned": tuned_cost["total_cost"],
                    "ValidRows": len(y_valid),
                }
            )
            print(
                f"{name:22s} fold {fold}: auc={roc_auc_score(y_valid, probabilities):.3f} "
                f"thr={threshold:.2f} recall@0.5={default_metrics['recall']:.3f} "
                f"recall@thr={tuned_metrics['recall']:.3f} "
                f"cost={tuned_cost['total_cost']:,.0f}"
            )

    return pd.DataFrame(records)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in folds.columns if c not in ("Model", "Fold")]
    summary = folds.groupby("Model")[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["Model"] + [f"{m}_{s}" for m, s in summary.columns[1:]]
    return summary


def final_holdout_check(X: pd.DataFrame, y: pd.Series, best_model_name: str):
    """Fit once on the 80% training split, tune threshold there, evaluate test once."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    estimator = build_models()[best_model_name]
    preprocessor = create_preprocessor(X_train)
    pipeline = fit_pipeline(estimator, preprocessor)
    pipeline.fit(X_train, y_train)

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    inner_probabilities = np.zeros(len(X_train))
    for inner_train, inner_valid in inner_cv.split(X_train, y_train):
        inner_prep = create_preprocessor(X_train.iloc[inner_train])
        inner_pipeline = fit_pipeline(estimator, inner_prep)
        inner_pipeline.fit(X_train.iloc[inner_train], y_train.iloc[inner_train])
        inner_probabilities[inner_valid] = inner_pipeline.predict_proba(
            X_train.iloc[inner_valid]
        )[:, 1]
    train_miss_cost = unit_cost_of_miss(y_train, X_train)
    threshold, curve = select_threshold(y_train, inner_probabilities, train_miss_cost)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    miss_cost = unit_cost_of_miss(y_test, X_test)
    tuned = retention_cost(y_test, probabilities, threshold, miss_cost)
    default = retention_cost(y_test, probabilities, 0.5, miss_cost)
    tuned_metrics = metrics_at(y_test, probabilities, threshold)
    default_metrics = metrics_at(y_test, probabilities, 0.5)

    result = {
        "Model": best_model_name,
        "Threshold": threshold,
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "Recall@0.5": default_metrics["recall"],
        "Precision@0.5": default_metrics["precision"],
        "F1@0.5": default_metrics["f1"],
        "Recall@tuned": tuned_metrics["recall"],
        "Precision@tuned": tuned_metrics["precision"],
        "F1@tuned": tuned_metrics["f1"],
        "OffersSent@tuned": tuned["customers_offered"],
        "ChurnersMissed@tuned": tuned["churners_missed"],
        "LostRevenue@tuned": tuned["lost_revenue"],
        "OutreachSpend@tuned": tuned["outreach_spend"],
        "TotalCost@0.5": default["total_cost"],
        "TotalCost@tuned": tuned["total_cost"],
        "TestRows": len(y_test),
    }
    curve_path = REPORTS_DIR / "threshold_cost_curve.csv"
    curve.to_csv(curve_path, index=False)
    return result, pipeline


def save_feature_importance(pipeline: Pipeline, X_frame, y_train, model_name: str):
    """Permutation importance on the raw input features of the selected model.

    permutation_importance shuffles the raw input columns, so its importances
    must be named with the raw feature names, not the transformed ones.
    """
    import matplotlib.pyplot as plt

    perm = permutation_importance(
        pipeline,
        X_frame,
        y_train,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    names = np.asarray(X_frame.columns)
    importance = (
        pd.DataFrame({"Feature": names, "Importance": perm.importances_mean})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    top = importance.head(15).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["Feature"], top["Importance"], color="#1f6feb")
    ax.set_xlabel("Mean decrease in ROC-AUC when shuffled")
    ax.set_title(f"Permutation importance: {model_name}")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "permutation_importance.png", dpi=150)
    plt.close(fig)
    return importance


def main() -> None:
    X, y = load_data()
    print(
        f"Customers: {len(X):,}; churn rate: {y.mean():.2%}; "
        f"folds: {N_FOLDS} (stratified, random_state={RANDOM_STATE})"
    )
    print(
        f"Retention offer cost: {RETENTION_COST:.0f}/contact; "
        f"miss cost: MonthlyCharges x {DEFAULT_HORIZON_MONTHS:.0f} months\n"
    )

    folds = run_cross_validation(X, y)
    summary = summarize(folds)

    REPORTS_DIR.mkdir(exist_ok=True)
    folds.to_csv(REPORTS_DIR / "cv_folds.csv", index=False)
    summary.to_csv(REPORTS_DIR / "cv_summary.csv", index=False)

    display_cols = [
        "Model",
        "ROC_AUC_mean",
        "ROC_AUC_std",
        "Recall@0.5_mean",
        "Recall@0.5_std",
        "Recall@tuned_mean",
        "Recall@tuned_std",
        "TotalCost@0.5_mean",
        "TotalCost@tuned_mean",
        "TotalCost@tuned_std",
        "Threshold_mean",
        "Threshold_std",
    ]
    print("Stratified 5-fold cross-validation (mean +/- std across folds)")
    print(summary[display_cols].to_string(index=False, float_format="{:.3f}".format))
    print()

    best_row = summary.loc[summary["TotalCost@tuned_mean"].idxmin()]
    best_model = best_row["Model"]
    print(f"Selected by lowest mean simulated cost: {best_model}\n")

    holdout, pipeline = final_holdout_check(X, y, best_model)
    holdout_df = pd.DataFrame([holdout])
    holdout_df.to_csv(REPORTS_DIR / "final_holdout.csv", index=False)
    print("Final check on the untouched 20% test split")
    print(holdout_df.to_string(index=False, float_format="{:.3f}".format))
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    importance = save_feature_importance(pipeline, X_test, y_test, best_model)
    print("Top 10 permutation importances (test features, selected model)")
    print(importance.head(10).to_string(index=False, float_format="{:.4f}".format))

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, models_dir / "churn_model.joblib")
    print("\nWrote reports/cv_folds.csv, cv_summary.csv, final_holdout.csv,")
    print("threshold_cost_curve.csv, permutation_importance.csv")
    print("Wrote reports/figures/permutation_importance.png and models/churn_model.joblib")


if __name__ == "__main__":
    main()
