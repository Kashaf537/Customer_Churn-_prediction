# Customer Churn Prediction

A Jupyter notebook that builds, compares, and explains models for predicting which telecom customers are likely to churn, so retention efforts can be targeted before they leave.

## Dataset

`telecom_churn.csv` — 3,333 customers, 11 columns, no missing values or duplicates.

| Column | Description |
|---|---|
| `Churn` | Target: 1 if the customer churned, 0 otherwise |
| `AccountWeeks` | Length of account tenure, in weeks |
| `ContractRenewal` | Whether the contract was recently renewed |
| `DataPlan` | Whether the customer has a data plan |
| `DataUsage` | Data usage (GB) |
| `CustServCalls` | Number of customer service calls |
| `DayMins` | Daytime minutes used |
| `DayCalls` | Number of daytime calls |
| `MonthlyCharge` | Monthly bill amount |
| `OverageFee` | Largest recent overage fee |
| `RoamMins` | Roaming minutes used |

The churn rate is roughly **14.5%** — a meaningfully imbalanced target, which the notebook addresses directly.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap jupyter
```

## What the notebook covers

1. **Setup** — imports and config.
2. **Load and explore the data** — shape, dtypes, missing/duplicate checks, class imbalance, feature distributions by churn status, and correlation with churn.
3. **Train/test split** — stratified, so the train and test sets preserve the churn rate.
4. **Handling class imbalance** — two strategies are used and compared:
   - `class_weight='balanced'` (Logistic Regression, Random Forest)
   - SMOTE oversampling, applied only to the training data (XGBoost)
   - `scale_pos_weight` (XGBoost, weighting without resampling)
5. **Model training & comparison** — four model configurations trained and evaluated side by side:

   | Model | Imbalance strategy |
   |---|---|
   | Logistic Regression | `class_weight='balanced'` |
   | Random Forest | `class_weight='balanced'` |
   | XGBoost | SMOTE-resampled training data |
   | XGBoost | `scale_pos_weight` |

   Compared visually (bar charts, ROC curves) and numerically (precision, recall, F1, ROC-AUC, PR-AUC).
6. **Selecting the best model** — confusion matrix, full classification report, and precision-recall curve for the chosen model.
7. **Explainability with SHAP** — global feature importance, mean absolute SHAP ranking, dependence plots for the top drivers, and an individual prediction explanation.
8. **Business summary** — key drivers of churn and concrete recommended actions.

## Results

**Final model: Random Forest (`class_weight='balanced'`)**

| Model | Precision (churn) | Recall (churn) | F1 (churn) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest (class_weight)** | 0.649 | 0.702 | **0.675** | 0.869 | 0.724 |
| XGBoost (scale_pos_weight) | 0.646 | 0.694 | 0.669 | 0.867 | 0.736 |
| XGBoost (SMOTE) | 0.576 | 0.719 | 0.640 | 0.860 | 0.711 |
| Logistic Regression (class_weight) | 0.341 | 0.727 | 0.464 | 0.814 | 0.426 |

The Random Forest model catches roughly **7 in 10 customers** who actually churn, and is correct about **65% of the time** when it flags someone as at-risk — well above the ~14.5% base rate, making retention outreach far more efficient than contacting customers at random.

### Top churn drivers (from SHAP)

1. **Monthly charge** — higher bills correlate with higher churn risk; price sensitivity is the single biggest lever.
2. **Customer service calls** — 4+ calls is a sharp, actionable early-warning signal.
3. **Contract renewal** — customers who haven't recently renewed churn far more often.
4. **Day minutes** — heavy daytime usage is associated with higher churn.
5. **Data plan / data usage** — having a data plan is mildly *protective* against churn, likely due to switching costs or bundled value.

### Recommended actions

- Flag accounts with 4+ customer service calls for proactive retention outreach.
- Review pricing/discount strategy for high-monthly-charge customers who also show other risk factors.
- Target contract renewal campaigns at customers approaching renewal, especially heavy users.
- Rank customers by the model's predicted probability (not just the 0/1 label) to prioritize retention budget on the highest-risk, highest-value segment first.

## Running it

```bash
jupyter notebook churn_prediction.ipynb
```

Place `telecom_churn.csv` in the same directory as the notebook (or update the `pd.read_csv(...)` path in section 2) before running all cells.

## Output

Running the full notebook saves 11 charts to an `outputs/` folder, ready to drop into a slide deck or report:

| File | Contents |
|---|---|
| `01_class_imbalance.png` | Churn class distribution |
| `02_feature_distributions.png` | Feature distributions by churn status |
| `03_correlation_with_churn.png` | Correlation heatmap |
| `04_model_comparison.png` | Bar chart comparing all four models |
| `05_roc_curves.png` | ROC curves overlay |
| `06_confusion_matrix.png` | Confusion matrix for the final model |
| `07_precision_recall_curve.png` | Precision-recall curve |
| `08_shap_summary.png` | SHAP summary plot |
| `09_shap_feature_importance.png` | Mean absolute SHAP importance ranking |
| `10_shap_dependence.png` | SHAP dependence plots for top drivers |
| `11_shap_individual_explanation.png` | SHAP explanation for one individual prediction |

## Notes

- SMOTE is applied **only to the training split**, never to test data, to avoid leaking synthetic signal into evaluation.
- Logistic Regression uses scaled features (`StandardScaler`); the tree-based models (Random Forest, XGBoost) use raw features.
- The notebook compares weighting vs. resampling head-to-head — in this dataset, `class_weight`/`scale_pos_weight` slightly outperformed SMOTE, but both are viable and worth testing on your own data.
