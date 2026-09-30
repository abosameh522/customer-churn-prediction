# Customer Churn Prediction

Churn analysis and a comparison of three classifiers on IBM Telco customer data, with preprocessing kept inside reproducible scikit-learn pipelines.

`Python` · `pandas` · `scikit-learn` · `Matplotlib` · `joblib` · `Jupyter`

---

## Overview

Retaining an existing customer is usually cheaper than acquiring a new one, so a churn model is most useful when it can flag at-risk customers early and show which groups they belong to. This project analyzes the IBM Telco dataset, identifies which customer segments churn most, and compares a linear model against two tree-based models for churn classification.

All preprocessing is fitted inside each model's pipeline on the training split only, so the test split is used once, at evaluation. The final pipeline and its evaluation artifacts are saved to disk.

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

- `TotalCharges` is stored as text. Converting with `pd.to_numeric(errors="coerce")` exposes 11 blank values, all for customers with zero tenure. No customers are dropped; the pipeline imputes them from the training median.
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

Churned customers pay more per month on average (74.44 vs 61.27), but the distributions overlap, so charges alone do not separate churners from non-churners. These are associations in this dataset, not evidence of cause.

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
│   └── evaluate.py             # Metrics, confusion matrix, ROC, importance
├── models/
│   └── churn_model.joblib      # Saved selected pipeline
├── reports/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── classification_report.txt
│   └── figures/
├── main.py                     # Entry point
├── requirements.txt
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

- Model selection uses a single train/test split; there is no cross-validation.
- The threshold is fixed at 0.5, and churn recall is 55.88%, so many churners are missed.
- The EDA describes the full dataset, including test rows, so the charts are descriptive rather than held-out.
- Feature importance is reported for Random Forest, not for the selected Logistic Regression.
- The saved model is version-sensitive and may not load under mismatched scikit-learn versions.

## Future Improvements

- Add cross-validation to stabilize model selection.
- Tune the decision threshold on validation data to raise churn recall.
- Add probability calibration and report a precision/recall curve.
- Compare against a gradient-boosting model.
- Add inference tests around the saved pipeline and a small serving wrapper.

## Author

[Ahmed Sameh](https://github.com/abosameh522)
