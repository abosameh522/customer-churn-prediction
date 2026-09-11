# Customer Churn Prediction

## About

This project uses customer and account information to predict telecom customer
churn. It includes data cleaning, exploratory analysis, and a comparison of
three classification models using Python and Scikit-learn.

## Dataset

- [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv), included in `data/telco_churn.csv`.
- **7,043 customers** and **19 input features** after removing `customerID`.
- Target: `Churn`, with `Yes` mapped to 1 and `No` to 0.
- **26.54%** of customers churned.

`TotalCharges` contains 11 blank values, all for customers with zero tenure.
I converted the column to numeric and filled missing values using a median
imputer fitted on training data. All customers were retained; there were no
exact duplicate rows.

## What I Did

- Created nine EDA charts in `notebooks/01_eda.ipynb`.
- Used a stratified 80/20 split: **5,634 training** and **1,409 test** customers.
- Scaled 4 numerical features and one-hot encoded 15 categorical features,
  producing **45 transformed features**.
- Kept imputation, scaling, encoding, and each model in a Scikit-learn pipeline.
- Compared accuracy, precision, recall, F1-score, and ROC-AUC.
- Saved the best pipeline, a confusion matrix, ROC curves, and Random Forest
  feature importances.

## Models

- Logistic Regression: `max_iter=1000`
- Decision Tree: `max_depth=5`
- Random Forest: `n_estimators=200`, `max_depth=10`

All models and the train/test split use `random_state=42`.

## Results

**Logistic Regression — Accuracy: 80.55%; ROC-AUC: 0.8419.**
It had the highest test ROC-AUC, narrowly ahead of Random Forest, and was saved
as `models/churn_model.joblib` with its complete preprocessing pipeline.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| Decision Tree | 0.7984 | 0.6347 | 0.5668 | 0.5989 | 0.8303 |
| Random Forest | 0.8027 | 0.6633 | 0.5214 | 0.5838 | 0.8410 |

Values are rounded to four decimal places from `reports/model_comparison.csv`.
Precision, recall, and F1 refer to churners at the default 0.5 threshold.
The selected model's recall is **55.88%**, so accuracy alone tells only part of
the story. These results use one split, with the test set also used to choose
the winner; cross-validation has not been performed.

![Model ROC curves](reports/figures/roc_curve.png)

## Main Findings

- Month-to-month customers have 42.71% churn, versus 2.83% for two-year contracts.
- Churn is higher at 0–12 months of tenure (47.44%) than at 49–72 months (9.51%).
- Average monthly charges are higher among churners: 74.44 versus 61.27.
- Customers without technical support or online security have higher churn.

Random Forest ranks tenure and TotalCharges highest in feature importance.
These importances describe that model and do not prove causation.

## Project Structure

```text
customer-churn-prediction/
├── data/telco_churn.csv
├── notebooks/01_eda.ipynb
├── src/
│   ├── data_preparation.py
│   ├── train.py
│   └── evaluate.py
├── models/churn_model.joblib
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── classification_report.txt
│   └── feature_importance.csv
├── requirements.txt
├── .gitignore
├── README.md
└── main.py
```

## Run the Project

Tested with Python 3.14.7. Dependencies are listed in `requirements.txt`.

```bash
git clone https://github.com/abosameh522/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows (Command Prompt):

```bat
.venv\Scripts\activate
```

Then install dependencies and run:

```bash
pip install -r requirements.txt
python main.py
```

The script trains all three models and regenerates the model, reports, and
three evaluation figures. To open the notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## What I Learned

- How to clean real customer data.
- How to use pipelines for preprocessing without learning from the test set.
- Why accuracy is not enough for churn prediction.
- How to compare classification models and interpret their results.

## Next Steps

- Cross-validation
- Threshold tuning
- Additional models
