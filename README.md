# BCG X Data Science Job Simulation
## Customer Churn Prediction — PowerCo Energy Provider

Virtual internship project completed on the [Forage](https://www.theforage.com/simulations/bcg/data-science-ccdz) platform as part of the **BCG X Data Science Program**.  
The goal: investigate and predict customer churn for **PowerCo**, a major energy utility client.

## Business Context

PowerCo is a major gas and electricity utility that supplies to small and medium-sized enterprises. The energy market has become increasingly competitive following liberalisation, and PowerCo has been experiencing significant customer churn.

**The client's hypothesis:** Customers are churning because of price sensitivity — they are switching to cheaper competitors.

**Our task:** Validate or disprove this hypothesis using data, and build a predictive model to identify customers at risk of churning.

## Project Structure

```
bcg-x-churn-prediction/
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_and_evaluation.ipynb
│
├── data/
│   ├── clean_data_after_eda.csv
│   ├── price_data (1).csv
|   ├── client_data (1).csv
│   └── data_for_predictions.csv
│
└── README.md

```

## Workflow Overview

The project followed BCG X's 5-step data science framework across 6 tasks:

1. Task 1 → Business Framing     : Understand PowerCo's churn problem
2. Task 2 → Data Request         : Draft email outlining data & technique needs
3. Task 3 → EDA & Cleaning       : Explore datasets, handle missing values, visualise distributions
4. Task 4 → Feature Engineering  : Create 19 new predictive features from raw data
5. Task 5 → Modeling & Evaluation: Train Random Forest classifier, evaluate performance
6. Task 6 → Insights             : Translate findings into client-ready recommendations

## Key Findings

### On the Price Sensitivity Hypothesis

**The hypothesis is only partially supported.**

Feature importance analysis revealed that **price-related features are NOT the primary driver of churn**. The top predictors are:

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | `margin_net_pow_ele` | 8.37% | Margin |
| 2 | `margin_gross_pow_ele` | 8.04% | Margin |
| 3 | `cons_12m` | 5.16% | Consumption |
| 4 | `forecast_meter_rent_12m` | 4.47% | Contract |
| 5 | `months_activ` | 4.32% | Tenure |

Price features (`off_peak_peak_var_mean_diff`, `var_year_price_off_peak`, etc.) **do appear in the top 15**, but are not dominant. This suggests **profitability and customer engagement are stronger churn signals** than price alone.

### Feature Engineering Highlights

From the raw dataset (44 columns), we engineered **19 new features** across 5 categories:

| Category | Features Created |
|----------|----------------|
| Date-derived | `tenure_days`, `days_to_end`, `months_to_renewal`, `months_since_modif`, `activ_month`, `activ_year` |
| Price sensitivity | `price_off_vs_peak_diff`, `price_sensitivity_ratio`, `price_change_6m_vs_1y`, `avg_price_change_yearly`, `avg_price_change_6m` |
| Consumption | `cons_ratio_last_to_12m`, `forecast_vs_actual`, `has_gas_usage` |
| Margin | `margin_ratio`, `net_margin_per_product` |
| Encoded | `has_gas_bin`, `channel_sales_encoded`, `origin_up_encoded` |


## Model Results

- **Algorithm:** Random Forest Classifier  
- **Dataset:** 14,606 customers | **Churn rate:** 9.7% (severely imbalanced)  
- **Split:** 75% train / 25% test (stratified)

### Handling Class Imbalance

The dataset has a **9.3:1 class imbalance** (non-churners vs churners). Without addressing this, the model achieves 90% accuracy but **0% recall** — it simply predicts everyone stays. We applied:
- `class_weight='balanced'` — penalises missed churner predictions more heavily
- `stratify=y` in train-test split — preserves churn ratio in both sets

### Performance Metrics

| Metric | Score | Notes |
|--------|-------|-------|
| Accuracy | 78.0% | Misleading on imbalanced data |
| Precision | 20.5% | 1 in 5 flagged customers truly churns |
| Recall | 43.7% | Primary focus — catches 44% of churners |
| F1 Score | 27.9% | Balanced precision/recall trade-off |
| ROC-AUC | 0.698 | Meaningful discriminative power above random (0.5 |

### Confusion Matrix


| | Predicted: Stay | Predicted: Churn |
|--------|--------|--------|
|Actual: Stay | 2,694 | 603 |
|Actual: Churn | 200 | 155 |


**155 churners** correctly identified out of 355 in the test set — a significant improvement over BCG's baseline model (18/366 without class balancing).

### Why These Metrics?

- Accuracy is reported but deprioritised — a model predicting "everyone stays" scores 90% on an imbalanced dataset, yet is useless.
- Recall is critical in a churn context. Missing a churning customer means permanently losing that revenue. We want to maximise the customers we catch.
- ROC-AUC is threshold-independent, making it robust for comparing models regardless of the operating point chosen.
- F1 balances recall and precision — avoiding both "flag everyone" and "flag no one" extremes.


## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- RandomForestClassifier

## Certificate

This project was completed as part of the **BCG X Data Science Job Simulation** on Forage.
[BCG X Data Science Program on Forage](https://drive.google.com/file/d/1-f0Az1ROm13EEJr0eD861MKuf7WePnEj/view)
