# Feature Engineering & Tuned Ensembles for Regression

## Dataset and validation strategy

- Input: `ames_cleaned.csv` (1,460 rows).
- Target: `SalePrice`; all models use `log1p(SalePrice)` and return predictions to the original dollar scale for evaluation.
- Task 3 hold-out preserved exactly: 80/20, `random_state=42`.
- Development data: **1,168** rows; untouched hold-out: **292** rows.
- Cross-validation: **5-fold shuffled KFold**, `random_state=42`.
- Feature engineering, imputation, scaling, and one-hot encoding are inside the Pipeline and therefore fitted independently inside each CV fold.

## Engineered features and justification

- **TotalSF:** Total basement + first-floor + second-floor square footage; summarizes overall usable size.
- **HouseAge:** Year sold minus year built; captures property age at the time of sale.
- **YearsSinceRemodel:** Year sold minus most recent remodel year; captures renovation recency.
- **TotalBathrooms:** Full baths plus half baths weighted at 0.5, including basement bathrooms; captures bathroom capacity.
- **HasRemodeled:** Binary indicator for whether YearRemodAdd is later than YearBuilt; captures whether the home has been remodeled.

All engineered features are computed only from property characteristics and sale-year information available without `SalePrice`; no target-derived feature is used.

## Required CV comparison

Mean CV RMSE is shown as **mean ± standard deviation** across five folds.

| Model | Mean CV RMSE ± SD | Mean CV R² | Training time (s) | Note |
|---|---:|---:|---:|---|
| Task 3-style Ridge (alpha=0.1) | $48,100.76 ± $31,204.02 | 0.4873 | 0.68 | Task 3-style regularized linear baseline with Task 5 features; log target. |
| Random Forest | $29,293.70 ± $4,820.62 | 0.8539 | 634.93 | RandomizedSearchCV over 5 hyperparameters; best params: {'regressor__model__n_estimators': 300, 'regressor__model__min_samples_split': 2, 'regressor__model__min_samples_leaf': 2, 'regressor__model__max_features': 0.5, 'regressor__model__max_depth': None}. |
| XGBoost | $25,701.02 ± $3,991.40 | 0.8878 | 58.35 | RandomizedSearchCV over 8 hyperparameters; best params: {'regressor__model__subsample': 1.0, 'regressor__model__reg_lambda': 5.0, 'regressor__model__reg_alpha': 0.01, 'regressor__model__n_estimators': 1200, 'regressor__model__min_child_weight': 3, 'regressor__model__max_depth': 3, 'regressor__model__learning_rate': 0.1, 'regressor__model__colsample_bytree': 0.7}. |

## Final untouched hold-out check

The hold-out was not used for hyperparameter tuning and is evaluated only after model selection.

| Model | Hold-out RMSE | Hold-out R² | Training time (s) |
|---|---:|---:|---:|
| Task 3-style Ridge (alpha=0.1) | $23,616.12 | 0.9273 | 0.11 |
| Random Forest | $29,265.74 | 0.8883 | 634.93 |
| XGBoost | $29,669.32 | 0.8852 | 58.35 |

## Verdict

XGBoost achieved the lowest mean 5-fold CV RMSE at **$25,701.02 ± $3,991.40**, compared with **$29,293.70 ± $4,820.62** for Random Forest. This indicates that XGBoost performed better than the tuned Random Forest during cross-validation.

However, the final untouched hold-out produced a different result. The Task 3-style Ridge model achieved a hold-out RMSE of **$23,616.12** and R² of **0.9273**, compared with **$29,265.74** RMSE and **0.8883** R² for Random Forest, and **$29,669.32** RMSE and **0.8852** R² for XGBoost.

Therefore, the CV improvement from XGBoost does not translate into better performance on this particular untouched hold-out. Based on the available hold-out evidence, the Ridge baseline provides the strongest final generalization result while also requiring substantially less training time. The tuned ensemble models demonstrate the value of feature engineering and nonlinear models, but their additional complexity and training cost are not justified by a better hold-out result in this experiment.

The difference between the CV and hold-out results should be treated as an important limitation of the experiment rather than as evidence that one model will always outperform another dataset-wide.

## Leakage and quality checks

- `SalePrice` is never used to create engineered features.
- `Id` is removed as an identifier.
- Preprocessing is inside the Pipeline and fitted only on each CV training fold.
- RandomizedSearchCV tunes on the development partition only.
- The 20% Task 3 hold-out remains untouched until final evaluation.

## Reproduce

```bash
pip install pandas numpy scikit-learn xgboost

python tuned_models.py ames_cleaned.csv --n-iter 5
```
