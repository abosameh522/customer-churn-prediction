# Customer Churn Prediction

Predicting customer churn on the IBM Telco dataset using three scikit-learn classifiers. All preprocessing is handled inside pipelines to prevent data leakage, and the trained model can rank customers by churn probability for retention targeting.

Built with Python, pandas, scikit-learn, Matplotlib, joblib, and Jupyter.

## Overview

Retaining an existing customer is usually cheaper than acquiring a new one, so a churn model is most useful when it can flag at-risk customers early and show which groups they belong to. This project analyzes the IBM Telco dataset, identifies which customer segments churn most, and compares a linear model against two tree-based models for churn classification.

All preprocessing is fitted on training rows. Evaluation and the ranking analysis below use the same test split. The final pipeline and its evaluation artifacts are saved to disk.

## Business Context

Suppose a retention team can review only part of its customer base. Predicted churn probabilities can provide a priority order for outreach: score customers, rank them from highest to lowest risk, and inspect how many observed churners fall within that review budget.

This project simulates that workflow on historical test rows. No customers were contacted, and no campaign outcomes or financial savings are measured. High churn risk does not establish that a customer will respond to outreach; that would require intervention data.

## Dataset

Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv), committed as `data/telco_churn.csv`.

| Property | Value |
| --- | --- |
| Customers | 7,043 |
| Input features | 19 |
| Target | `Churn` (Yes/No mapped to 1/0) |
| Churn rate | 26.54% (1,869 churned) |
| Training rows | 5,634 |
| Test rows | 1,409 |
| Test churners | 374 |
| Transformed features | 45 after encoding and scaling |

## Workflow

```mermaid
flowchart LR
    A[Telco CSV] --> B[Cleaning]
    B --> C[Stratified split]
    C --> D[Preprocessing pipeline]
    D --> E[Model training]
    E --> F[Evaluation]
    F --> G[Saved pipeline]
```

## Data Preparation

- `TotalCharges` is stored as text. Converting with `pd.to_numeric(errors="coerce")` exposes 11 blank values, all for customers with zero tenure. Duplicate customer records are removed. Missing TotalCharges values are imputed from the training median within the pipeline.
- Identity and target handling: `customerID` is dropped before modeling, and `Churn` is mapped to 1/0.
- Numerical features receive median imputation and `StandardScaler`; categorical features receive one-hot encoding with `handle_unknown="ignore"`.
- An 80/20 split is stratified on the target with `random_state=42`, keeping the churn share equal in both sets.
- Imputation, encoding, and scaling live inside the pipeline object, so they are fit on training data only and applied unchanged to the test set.

## Exploratory Analysis

The notebook `notebooks/01_eda.ipynb` inspects the data and charts churn across customer groups. Churn rates by segment:

| Segment | Churn rate |
| --- | ---: |
| Month-to-month contract | 42.71% |
| Two-year contract | 2.83% |
| Tenure 0–12 months | 47.44% |
| Tenure 49–72 months | 9.51% |
| Electronic check payment | 45.29% |
| Fiber optic internet | 41.89% |
| No technical support | 41.64% |
| No online security | 41.77% |

Churned customers pay more per month on average (74.44 vs 61.27), but the distributions overlap, so charges alone do not separate churners from non-churners. The relationships above are observed associations in this dataset.

## Model Comparison

Evaluated on the 1,409-row test split (374 churners) at a 0.5 threshold. Values come from `reports/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| Decision Tree | 0.7984 | 0.6347 | 0.5668 | 0.5989 | 0.8303 |
| Random Forest | 0.8027 | 0.6633 | 0.5214 | 0.5838 | 0.8410 |

![ROC curves](reports/figures/roc_curve.png)

![Confusion matrix — Logistic Regression](reports/figures/confusion_matrix.png)

## Selected Model

Logistic Regression recorded the highest ROC-AUC on this split (0.8419) and the best precision/recall balance of the three, so it is the saved model. The margin over Random Forest (0.8410) is small and specific to this split; the result is not a claim that logistic regression is generally the stronger churn model. Recall is 55.88%, so roughly 44% of churners are still missed at the 0.5 threshold.

## Retention Prioritization

The selected Logistic Regression pipeline ranks the 1,409 test customers by `predict_proba`, without changing the model or its 0.5 classification threshold. The test set contains 374 observed churners.

| Highest-risk share | Customers prioritized | Churners captured | Capture of all churners | Churn rate in group | Lift vs random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top 10% | 141 | 105 | 28.07% | 74.47% | 2.81× |
| Top 20% | 282 | 186 | 49.73% | 65.96% | 2.48× |
| Top 30% | 423 | 243 | 64.97% | 57.45% | 2.16× |

![Simulated retention ranking: cumulative gains and capture by review budget](reports/figures/cumulative_gains.png)

At a simulated 20% review budget, the ranked group contains 186 of the 374 churners. Random selection would capture about 20% in expectation; this ranking captures 49.73%. That is ranking lift, not an estimate of customers retained.

Cutoffs use the nearest whole number of rows (`round(share * 1409)`). Equal scores retain their input order. Capture is `churners in group / 374`; group precision is `churners in group / group size`; lift divides group precision by the test churn rate (`374 / 1409`). Exact values and actual population fractions are saved in `reports/retention_priority.csv`.

These cutoffs describe the existing test split, not a tuned outreach policy. Model selection already used that split; confirmation on separate data is needed before choosing a review budget.

## Key Findings

- Contract type is the sharpest split: month-to-month churn is 42.71% versus 2.83% for two-year contracts.
- Churn falls with tenure, from 47.44% in the first year to 9.51% in months 49–72.
- Electronic-check payers churn at 45.29%, higher than customers on automatic payments.
- Customers without technical support or online security churn at 41.64% and 41.77%, well above those with the services.
- With a 26.54% churn rate, accuracy is a weak signal; recall, F1, and ROC-AUC drive the evaluation.

### Feature Importance

The Random Forest importances describe that model, not the selected Logistic Regression, and do not imply causation.

![Random Forest feature importance](reports/figures/feature_importance.png)

`tenure` (0.142) and `TotalCharges` (0.127) rank highest, followed by the month-to-month contract (0.091), `MonthlyCharges` (0.089), and the "no" flags for online security, internet service, and technical support.

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
│   ├── evaluate.py             # Metrics, confusion matrix, ROC, importance
│   └── retention_analysis.py   # Test-set ranking, capture, and gains
├── tests/
│   └── test_retention_analysis.py  # Unit tests for retention metrics
├── models/
│   └── churn_model.joblib      # Saved selected pipeline
├── reports/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── classification_report.txt
│   ├── retention_priority.csv
│   └── figures/
├── main.py                     # Entry point
├── requirements.txt
├── pyproject.toml             # Ruff and pytest configuration
├── .pre-commit-config.yaml    # Pre-commit hooks
├── LICENSE
└── README.md
```

## Running the Project

Verified with Python 3.14.7.

```bash
git clone https://github.com/abosameh522/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

`main.py` loads the CSV, splits and trains the three models, prints the comparison, and rewrites every file under `models/` and `reports/`.

To open the analysis notebook instead:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Outputs

| Output | Location |
| --- | --- |
| Selected pipeline (model + preprocessing) | `models/churn_model.joblib` |
| Model metrics table | `reports/model_comparison.csv` |
| Per-class metrics | `reports/classification_report.txt` |
| Random Forest importances | `reports/feature_importance.csv` |
| Confusion matrix | `reports/figures/confusion_matrix.png` |
| ROC curves | `reports/figures/roc_curve.png` |
| Feature importance plot | `reports/figures/feature_importance.png` |
| Retention capture and lift table | `reports/retention_priority.csv` |
| Cumulative gains chart | `reports/figures/cumulative_gains.png` |

`python main.py` also regenerates the ranking outputs. Check the capture/lift calculations with:

```bash
python -m unittest discover -s tests -v
```

The saved pipeline can be applied to new rows without repeating the transformations:

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

The artifact is tied to the library versions in `requirements.txt`; loading it under very different versions is not guaranteed.

## Technical Decisions

- **Preprocessing inside `Pipeline`.** Imputation, encoding, and scaling are fitted per model on training data, which removes a common source of leakage.
- **Stratified split.** Keeps the churn proportion equal in train and test, which matters with a 26.54% positive class.
- **Median imputation for `TotalCharges`.** The blank values belong to zero-tenure customers, and the median is fit on training data only.
- **`handle_unknown="ignore"`.** Unseen categories at inference become all-zero instead of raising an error.
- **Fixed `random_state=42`.** Makes the split and the tree models reproducible.
- **Selection by ROC-AUC.** Chosen because the classes are imbalanced and accuracy alone is misleading.

## Limitations

- Model selection uses a single train/test split without cross-validation.
- The threshold is fixed at 0.5, resulting in 55.88% recall (many churners are missed).
- The EDA describes the full dataset including test rows, so the charts are descriptive rather than held-out validation.
- Feature importance is reported for Random Forest, not for the selected Logistic Regression.
- The saved model is version-sensitive and may not load under mismatched scikit-learn versions.

## Future Improvements

- Use k-fold cross-validation to produce more stable model selection and performance estimates.
- Tune the decision threshold on a separate validation set to optimize the recall/precision tradeoff for the business case.
- Add probability calibration (Platt scaling or isotonic regression) and generate precision/recall curves across thresholds.
- Benchmark against gradient-boosting models (XGBoost, LightGBM, CatBoost) to see if additional complexity improves ROC-AUC.
- Add integration tests around the saved pipeline artifact and build a minimal inference API or batch scoring script.

## Author

[Ahmed Sameh](https://github.com/abosameh522)
