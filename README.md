# Customer Churn Prediction

Churn analysis and classification on IBM Telco customer data, with training-only preprocessing pipelines and a comparison of three models.

## Project Overview

Customer churn is a common business problem: identifying customers likely to leave can support retention work. This project analyzes the IBM Telco dataset, looks at which customer groups churn most, and compares three classifiers for predicting churn.

The workflow keeps all preprocessing inside each model's pipeline and fits it only on training data, so the test set stays untouched until evaluation.

## Dataset Snapshot

Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv), included as `data/telco_churn.csv`.

| Item | Value |
| --- | --- |
| Customers | 7,043 |
| Input features | 19 |
| Target | `Churn` (Yes/No mapped to 1/0) |
| Churn rate | 26.54% (1,869 churned) |
| Training / test split | 5,634 / 1,409 |
| Test-set churners | 374 |
| Transformed features | 45 after encoding and scaling |

## Workflow

```mermaid
flowchart LR
    A[Telco CSV] --> B[Cleaning]
    B --> C[Preprocessing]
    C --> D[EDA]
    D --> E[Model training]
    E --> F[Evaluation]
    F --> G[Saved pipeline]
```

## Data Preparation

- `TotalCharges` is text in the raw CSV. `pd.to_numeric(errors="coerce")` exposes **11 blank values**, all belonging to customers with zero tenure. All 7,043 customers are kept.
- `customerID` is dropped; `Churn` is mapped Yes = 1, No = 0.
- Numerical features get median imputation and `StandardScaler`.
- Categorical features get one-hot encoding with `handle_unknown="ignore"`.
- An 80/20 stratified split (`random_state=42`) preserves the churn share in both sets.
- Imputation, encoding, and scaling are fitted inside each model's pipeline, only on the training data.

## Exploratory Analysis

The notebook (`notebooks/01_eda.ipynb`) checks the data and uses nine charts to compare churn across customer groups. A few stand out:

| Group | Churn rate |
| --- | ---: |
| Month-to-month contract | 42.71% |
| Two-year contract | 2.83% |
| Tenure 0–12 months | 47.44% |
| Tenure 49–72 months | 9.51% |
| Electronic check payment | 45.29% |
| Fiber optic internet | 41.89% |
| No technical support | 41.64% |
| No online security | 41.77% |

Churned customers also pay more on average (74.44 vs 61.27 in monthly charges), though the distributions overlap, so charges alone do not separate the groups. These are patterns in this dataset, not proof of cause.

## Model Comparison

Held-out test set: 1,409 customers, including 374 churners. Values are from `reports/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| Decision Tree | 0.7984 | 0.6347 | 0.5668 | 0.5989 | 0.8303 |
| Random Forest | 0.8027 | 0.6633 | 0.5214 | 0.5838 | 0.8410 |

![ROC curves](reports/figures/roc_curve.png)

![Confusion matrix — Logistic Regression](reports/figures/confusion_matrix.png)

## Why Logistic Regression?

Logistic Regression has the highest ROC-AUC (0.8419) on this split, narrowly ahead of Random Forest (0.8410), and keeps the best precision/recall balance among the three. Its recall is still only 55.88%, so it misses a meaningful share of churners. The margin over Random Forest is small, so the choice is specific to this split, not a general conclusion that logistic regression is better for churn.

## Feature Importance

The Random Forest feature importances describe that model, not the selected Logistic Regression, and do not show causation.

![Random Forest feature importance](reports/figures/feature_importance.png)

`tenure` and `TotalCharges` rank highest, followed by the month-to-month contract, `MonthlyCharges`, and the "no" categories for online security and technical support.

## Key Findings

- Contract type is the largest gap: month-to-month customers churn far more than one- or two-year customers.
- Churn is highest in the first year and falls steadily with tenure.
- Electronic-check payers churn more than customers on automatic payments.
- Customers without technical support or online security churn more.
- Accuracy is a weak guide here because only 26.54% of customers churned; recall, F1, and ROC-AUC carry more signal.

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory analysis
├── src/
│   ├── data_preparation.py     # Cleaning and preprocessing pipeline
│   ├── train.py                # Model definitions and training
│   └── evaluate.py             # Metrics and figures
├── models/
│   └── churn_model.joblib      # Saved best pipeline
├── reports/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── classification_report.txt
│   └── figures/
├── main.py
├── requirements.txt
└── README.md
```

## How to Run

Tested with Python 3.14.7.

```bash
git clone https://github.com/abosameh522/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

`main.py` trains the three models, prints the comparison, and rewrites the reports, figures, and `models/churn_model.joblib`. To open the analysis notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Saved Pipeline

`models/churn_model.joblib` stores the selected model together with its fitted preprocessing, so the same pipeline can be loaded and applied to new customer rows without repeating the transformations:

```python
import joblib
import pandas as pd

pipeline = joblib.load("models/churn_model.joblib")
customers = pd.read_csv("data/telco_churn.csv")
customers["TotalCharges"] = pd.to_numeric(customers["TotalCharges"], errors="coerce")
customers["Churn"] = customers["Churn"].map({"Yes": 1, "No": 0})
features = customers.drop(columns=["customerID", "Churn"])
probabilities = pipeline.predict_proba(features)[:, 1]
```

The artifact is tied to the library versions in `requirements.txt`; loading it with very different versions is not guaranteed.

## Limitations

- The winner is chosen from a single test split; there is no cross-validation.
- Recall for churners is 55.88%, so many churners are still missed at the 0.5 threshold.
- The EDA describes the full dataset and churn patterns are associations, not causes.
- Feature importance is reported for Random Forest, not the selected Logistic Regression.
- Saved-pipeline compatibility depends on the installed scikit-learn version.

## Future Improvements

- Use cross-validation instead of a single split for model selection.
- Tune the classification threshold on validation data to improve churn recall.
- Add a boosting model as an additional comparison.
- Package the saved pipeline with its exact dependency versions.

## Author

[Ahmed Sameh](https://github.com/abosameh522)
