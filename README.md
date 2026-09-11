# Customer Churn Prediction

## About the Project

I built this project to practice data cleaning, EDA, preprocessing,
classification, and model evaluation. I used the IBM Telco Customer Churn
dataset to predict whether a customer would leave the company. I compared
three models and used charts to look for differences between customers who
stayed and those who churned.

## Dataset

- [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv), included in `data/telco_churn.csv`.
- **7,043 customers** and **19 model input features**, mixing numerical and categorical data.
- Target: `Churn`, mapped from `Yes`/`No` to 1/0; `customerID` is excluded.
- `TotalCharges` contained **11 blank values**. I kept all customers.

## Problems I Ran Into

### TotalCharges

`TotalCharges` looks numerical, but Pandas loaded it as text because of the
blank strings. I used `pd.to_numeric(errors="coerce")` and filled the missing
values with a training-set median inside the pipeline. All 11 blanks belonged
to customers with zero tenure.

### Class imbalance

Only **26.54%** of customers churned. Predicting that everyone stays would give
73.46% accuracy on the full dataset, so I also looked at recall, F1, and ROC-AUC.

### Model comparison

Logistic Regression and Random Forest were very close in ROC-AUC: **0.8419**
and **0.8410**. Logistic Regression was slightly better on this split, so I kept it.

## Some Choices I Made

- I used an 80/20 split: **5,634 training rows** and **1,409 test rows**.
- I used `stratify=y` because churners are the smaller class and I wanted a
  similar churn percentage in both sets.
- I used median imputation and `StandardScaler` for the 4 numerical features.
- I one-hot encoded the 15 categorical features because they have no natural
  numeric order. `handle_unknown="ignore"` handles categories not seen in training.
- I kept preprocessing inside each model's pipeline and fitted it only on the
  training data. This produced **45 transformed features**.
- I compared a linear model with two tree-based models using fixed settings.

## Models

- Logistic Regression: `max_iter=1000`
- Decision Tree: `max_depth=5`
- Random Forest: `n_estimators=200`, `max_depth=10`

All models and the train/test split use `random_state=42`.

## Results

Logistic Regression gave me the highest ROC-AUC at **0.8419**. Its accuracy was
**80.55%**, but recall was only **55.88%**, so it still misses quite a few churners.
I saved the complete pipeline as `models/churn_model.joblib`.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| Decision Tree | 0.7984 | 0.6347 | 0.5668 | 0.5989 | 0.8303 |
| Random Forest | 0.8027 | 0.6633 | 0.5214 | 0.5838 | 0.8410 |

These values are rounded to four decimal places from `reports/model_comparison.csv`.
Accuracy counts all correct predictions. Precision tells me how many predicted
churners actually churned; recall tells me how many actual churners I caught.
F1 balances precision and recall, while ROC-AUC measures how well the model ranks
churners above non-churners across thresholds.

![Model ROC curves](reports/figures/roc_curve.png)

## What I Noticed in the Data

The notebook contains nine EDA charts. A few patterns stood out:

- Month-to-month churn was 42.71%, compared with 2.83% for two-year contracts.
- Churn was higher at 0–12 months of tenure (47.44%) than at 49–72 months (9.51%).
- Churned customers had higher average monthly charges: 74.44 versus 61.27.
- Customers without technical support or online security had higher churn.

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
├── reports/                  # Metrics, classification report, and figures
├── requirements.txt
├── .gitignore
├── README.md
└── main.py
```

## Run the Project

Tested with Python 3.14.7.

```bash
git clone https://github.com/abosameh522/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
```

Activate on Linux/macOS with `source .venv/bin/activate`, or on Windows
(Command Prompt) with `.venv\Scripts\activate`. Then run:

```bash
pip install -r requirements.txt
python main.py
```

This trains the three models and saves the pipeline, reports, confusion matrix,
ROC curves, and Random Forest feature-importance plot. Open the notebook with:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Project Notes

- Using `Pipeline` let me keep preprocessing and the model together instead of
  manually transforming the training and test sets.
- Logistic Regression's result was a reminder that a more complicated model
  is not automatically better on a given split.
- The Random Forest ranked tenure and TotalCharges highest. Its importances
  describe that model, not the selected Logistic Regression, and do not show causation.

## Notes

- I used the test split to choose the winner; I have not run cross-validation.
- The classification threshold is still 0.5, with room to improve churn recall.
- The EDA describes the full dataset.

## Next Things I Want to Try

- Use cross-validation instead of relying on one split.
- Try different classification thresholds on validation data to improve recall.
- Compare with a boosting model later.
