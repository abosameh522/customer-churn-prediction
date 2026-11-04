# Customer Churn Prediction

This project studies the IBM Telco customer churn dataset and uses churn probabilities to simulate a retention campaign. The focus is business cost: the expense of offering a retention discount to a customer versus the expected revenue lost if a churner is missed.

Built with Python, pandas, scikit-learn, Matplotlib, joblib, and Jupyter. No external credit scoring or risk framework; this is pure retention economics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or `\.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Running the analysis

The main script trains three models, evaluates them with stratified cross-validation, picks the best model by simulated cost, and checks it once on a held-out test split.

```bash
python -m src.retention_cv
```

This writes `reports/cv_folds.csv`, `reports/cv_summary.csv`, `reports/final_holdout.csv`, `reports/threshold_cost_curve.csv`, `reports/permutation_importance.csv`, `reports/figures/permutation_importance.png`, and `models/churn_model.joblib`.

The notebook `notebooks/01_eda.ipynb` covers the exploratory charts (contract type, internet service, payment method, tenure, senior status).

## Dataset and preprocessing

Data source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv), committed as `data/telco_churn.csv`.

| Property | Value |
|---|---|
| Customers | 7,043 |
| Input features | 19 |
| Churn rate | 26.54% (1,869 churned) |
| Blank `TotalCharges` | 11, all zero-tenure customers |
| Duplicate rows | 0 |
| Training / test rows | 5,634 / 1,409 |
| Test churners | 374 |
| Features after encoding | 45 |

Preprocessing facts:
- `customerID` is dropped before modeling.
- `TotalCharges` is read as text in the source CSV; 11 blank values become NaN and are imputed with the training median inside the pipeline.
- Features are split into numeric and categorical, median-imputed, scaled, and one-hot encoded. The preprocessor is fitted only on the training fold, so nothing leaks into evaluation.
- Class imbalance is handled inside each model pipeline (`class_weight=balanced` or `balanced_subsample`), never by resampling the data outside the pipeline.

All preprocessing, training, threshold tuning, and scoring are inside scikit-learn pipelines.

## Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for tabular ML |
| Dataframes | pandas, numpy | Loading, cleaning, cost calculations |
| Modeling | scikit-learn | Pipelines keep preprocessing leakage-free |
| Plots | matplotlib (Agg) | Reproducible figures in CI and headless runs |
| Notebook | Jupyter | Exploratory analysis kept separate from training code |
| Testing | pytest + pytest-cov | Offline tests with no dataset or artifact dependency |
| Lint | Ruff | Fast, replaces flake8 + isort |

## Results from a real run

Stratified 5-fold cross-validation (5 folds, random_state=42), class weights inside the pipeline, threshold chosen per fold from out-of-fold probabilities on the training side only:

| Model | ROC-AUC (mean±std) | Recall@0.5 (mean±std) | Recall@tuned (mean±std) | Cost@0.5 | Cost@tuned (mean±std) | Best threshold |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.845±0.015 | 0.801±0.017 | 0.987±0.008 | 66,650 | 25,947±1,188 | 0.074 |
| Random Forest | 0.845±0.014 | 0.738±0.022 | 0.989±0.006 | 89,367 | 25,102±1,310 | 0.070 |
| HistGradientBoosting | 0.827±0.013 | 0.698±0.017 | 0.955±0.007 | 103,550 | 31,754±1,561 | 0.050 |

Models are ranked by mean simulated retention cost, not by ROC-AUC.

Retention economics (per outreach attempt):
- Offer cost: 20 per contacted customer.
- Miss cost: `MonthlyCharges × 12` months, the expected revenue at risk if a churner is not kept.

Selected model: **Random Forest** (lowest mean simulated cost). The cost-based tuned threshold (0.07) cuts expected cost from roughly 89,367 down to roughly 25,102, far more than accuracy or ROC-AUC alone would suggest.

Final check on the untouched 20% test split (1,409 customers, Random Forest, threshold 0.08):

| Metric | Value |
|---|---|
| ROC-AUC | 0.839 |
| Recall@0.5 | 0.738 |
| Recall@0.08 (tuned) | 0.987 |
| Precision@0.08 | 0.328 |
| Churners missed | 5 |
| Offers sent | 1,126 |
| Lost revenue | 4,175 |
| Outreach spend | 22,520 |
| Total cost @0.5 | 89,536 |
| Total cost @0.08 | 26,695 |

## Model comparison

- **Logistic Regression** catches the most churners at the default 0.5 threshold (recall 0.801) and keeps costs stable, but its tuned cost is slightly above Random Forest.
- **Random Forest** wins on cost. Its tuned threshold is 0.07, meaning it recommends offers to a much larger share of the base to avoid missing churners.
- **HistGradientBoosting** performed the worst on this dataset (ROC-AUC 0.827, tuned cost the highest). It did not beat the simpler models here; on imbalanced churn data with this column set, it needs more tuning than I ran. I leave it in for completeness rather than claim it is better.

Honest note: tuned thresholds around 0.05–0.10 send offers to a large share of customers, so precision drops into the 0.33 range. The model is good at *finding* churners, but the campaign is still expensive per retention. That is the trade-off the cost model makes explicit.

## Feature importance (Random Forest)

Permutation importance on the held-out test split, measured as drop in ROC-AUC when each feature is shuffled. These are real outputs from the run above; higher is more important:

| Feature | Importance |
|---|---|
| Contract | 0.0573 |
| tenure | 0.0202 |
| InternetService | 0.0146 |
| TotalCharges | 0.0118 |
| OnlineSecurity | 0.0079 |
| TechSupport | 0.0023 |
| MonthlyCharges | 0.0013 |
| PaymentMethod | 0.0010 |
| OnlineBackup | 0.0010 |
| PaperlessBilling | 0.0010 |
| Dependents | 0.0006 |
| MultipleLines | 0.0005 |
| StreamingMovies | 0.0005 |
| SeniorCitizen | 0.0003 |
| PhoneService | 0.0001 |
| StreamingTV | -0.0002 |
| gender | -0.0003 |
| Partner | -0.0004 |
| DeviceProtection | -0.0006 |

Top drivers: `Contract` (month-to-month customers churn far more), `tenure` (short-tenure risk), `InternetService`, and `TotalCharges`. This matches the EDA: month-to-month contracts and fiber-optic service had the highest churn rates, while 90-day and two-year contracts had low churn. Several features (StreamingTV, Partner, DeviceProtection, Gender) show near-zero or slightly negative importance, meaning they do not help here.

The full list, plus per-fold cost curves and thresholds, is in `reports/`.

## Limitations

- The final holdout check is a single 20% split, not new data.
- Miss cost is an approximation: every missed churner is charged `MonthlyCharges × 12`, regardless of their actual contract horizon.
- Tuned thresholds (0.05–0.10) send many offers; precision near 0.33 means roughly two of three offers go to customers who would have stayed anyway.
- The EDA notebook uses the full dataset, so it is descriptive, not held-out validation.

## Roadmap

- [x] Stratified k-fold cross-validation reporting mean and std per metric (`src/retention_cv.py`)
- [x] Threshold tuning on out-of-fold training probabilities, not the test split
- [x] Keep the trained model artifact out of git; rebuild it by running the script
- [ ] Probability calibration and precision/recall curves across thresholds
- [ ] Contract-aware miss cost instead of a flat 12-month horizon
- [ ] Batch scoring from the saved pipeline

## License

MIT — see [LICENSE](LICENSE). The Telco dataset keeps its original IBM terms; check them before redistributing.

## Where to look

- `src/retention_cv.py` — CV, cost-based threshold selection, final holdout check, importance.
- `src/train.py`, `src/evaluate.py`, `src/data_preparation.py` — training, metrics, preprocessing.
- `src/retention_analysis.py` — ranking and priority tables from `main.py`.
- `notebooks/01_eda.ipynb` — exploratory analysis.
- `reports/` — all CSV outputs and figures from this run.
- `models/churn_model.joblib` — trained Random Forest pipeline, written by the run (kept out of git; rebuild with `python -m src.retention_cv` or `python main.py`).
