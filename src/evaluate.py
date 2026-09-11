import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

plt.switch_backend("Agg")


def compare_models(pipelines, X_test, y_test):
    results = []
    for name, pipeline in pipelines.items():
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
        })
    comparison = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    return comparison.reset_index(drop=True)


def save_confusion_matrix(pipeline, name, X_test, y_test, figures_dir):
    predictions = pipeline.predict(X_test)
    report = classification_report(
        y_test, predictions, target_names=["Stayed", "Churned"],
        zero_division=0,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Stayed", "Churned"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return report


def save_roc_curves(pipelines, X_test, y_test, figures_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipeline in pipelines.items():
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(
            y_test, probabilities, name=name, ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def save_feature_importance(pipeline, figures_dir):
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("Importance", ascending=False)

    top_features = importance.head(15).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_features["Feature"], top_features["Importance"])
    ax.set_xlabel("Impurity-based importance")
    ax.set_title("Random Forest: top 15 features")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    return importance
